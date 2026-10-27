"""Session output capture shared by Gradio and its spawned workers.

Ported from Applio (IAHispano/Applio) `rvc/lib/terminal.py`, adapted for the
`arvc` package layout. The capture replaces stdout/stderr with a tee that
keeps the original streams going (so `python -m arvc` still prints to a real
console) while mirroring everything to a temp log file. The Gradio "Terminal"
tab polls that file and shows the live session output.

Usage
-----
    from arvc.utils.terminal import start_capture, get_capture

    start_capture()        # call once at process start
    capture = get_capture()  # may return None if start_capture() was never called
    text = capture.read()    # read the latest ~128 KB of output

The capture is process-local: each `python -m arvc` process has its own
session log, so multiple workers don't clobber each other.
"""

from __future__ import annotations

import atexit
import codecs
import os
from pathlib import Path
import re
import sys
import tempfile
import threading

# Env var that holds the session log path so spawned workers can find it.
_ENV = "ARVC_TERMINAL_LOG"
# Hard cap on how much we send back to the UI per poll.
_MAX_VIEW = 128 * 1024
# Strip ANSI escape sequences (colours, tqdm progress bars, etc.) for display.
_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")

_capture: "_TerminalCapture | None" = None


# ---------------------------------------------------------------------------
# Windows console writer (no-op on POSIX; mirrors Applio upstream behaviour)
# ---------------------------------------------------------------------------
class _TerminalWriter:
    """Write Unicode to Windows consoles, raw bytes to pipes/files."""

    def __init__(self, descriptor: int):
        self.descriptor = descriptor
        self.console = None
        if sys.platform == "win32":
            try:
                import ctypes
                import msvcrt

                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.GetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                handle = msvcrt.get_osfhandle(descriptor)
                mode = ctypes.c_ulong()
                if kernel.GetConsoleMode(handle, ctypes.byref(mode)):
                    kernel.WriteConsoleW.argtypes = [
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_ulong,
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                    ]
                    self.console = (kernel, handle)
                    self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            except OSError:
                self.console = None

    def write(self, data: bytes, final: bool = False) -> None:
        if self.console is None:
            # POSIX path — just write the raw bytes.
            view = memoryview(data)
            while view:
                view = view[os.write(self.descriptor, view) :]
            return
        import ctypes

        # A multi-byte glyph may straddle two reads from the session log.
        text = self.decoder.decode(data, final=final)
        encoded = text.encode("utf-16-le")
        buffer = ctypes.create_string_buffer(encoded)
        position = 0
        kernel, handle = self.console
        while position < len(encoded):
            written = ctypes.c_ulong()
            count = min((len(encoded) - position) // 2, 16000)
            if not kernel.WriteConsoleW(
                handle,
                ctypes.byref(buffer, position),
                count,
                ctypes.byref(written),
                None,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if not written.value:
                raise OSError("Console write made no progress")
            position += written.value * 2


def _open_append(path: str) -> int:
    """Open *path* for append-only writing, returning a raw file descriptor."""
    if sys.platform != "win32":
        return os.open(path, os.O_WRONLY | os.O_APPEND)
    import ctypes
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    GENERIC_WRITE = 0x40000000
    FILE_SHARE_READ = 1
    FILE_SHARE_WRITE = 2
    OPEN_ALWAYS = 4
    FILE_APPEND_DATA = 0x0004
    handle = kernel.CreateFileW(
        path,
        FILE_APPEND_DATA,
        FILE_SHARE_READ | FILE_SHARE_WRITE | 0x4,  # FILE_SHARE_DELETE
        None,
        OPEN_ALWAYS,
        0,
        None,
    )
    if handle == -1 or handle == 0xFFFFFFFFFFFFFFFF:
        raise ctypes.WinError(ctypes.get_last_error())
    return msvcrt.open_osfhandle(handle, 0)


def _redirect(path: str) -> None:
    """Replace stdout/stderr with file objects that append to *path*."""
    fd = _open_append(path)
    # Make sure the original file descriptors 1 and 2 point at the log file.
    for target in (1, 2):
        try:
            os.dup2(fd, target)
        except OSError:
            # If we can't dup2 (e.g. closed fd), fall back to os.fdopen on the new fd.
            pass
    os.close(fd)
    sys.stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8", errors="replace")
    sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8", errors="replace")


def capture_worker_output() -> None:
    """Worker entry point: redirect this process's stdout/stderr to the parent's log.

    Called from subprocess workers via ``ARVC_TERMINAL_LOG=<path> python -m arvc ...``.
    Safe to call from the main process too (it is idempotent if the env is unset).
    """
    log_path = os.environ.get(_ENV)
    if not log_path:
        return
    try:
        _redirect(log_path)
    except OSError:
        # If the parent's temp file is gone, leave stdout alone — the worker
        # just won't be mirrored into the UI, which is non-fatal.
        pass


# ---------------------------------------------------------------------------
# Capture singleton
# ---------------------------------------------------------------------------
class _TerminalCapture:
    """Tee stdout/stderr into a temp file the Gradio UI can read."""

    def __init__(self) -> None:
        self.directory = tempfile.TemporaryDirectory(
            prefix="arvc-terminal-", ignore_cleanup_errors=True
        )
        self.path = Path(self.directory.name) / "session.log"
        self.path.touch()

        self.original_streams = (sys.stdout, sys.stderr)
        self.original_fds: tuple[int, int] = (os.dup(1), os.dup(2))
        self.previous_env = {
            key: os.environ.get(key)
            for key in (_ENV, "PYTHONUNBUFFERED", "PYTHONIOENCODING")
        }
        self.stop = threading.Event()
        self.closed = False

        os.environ[_ENV] = str(self.path)
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["PYTHONIOENCODING"] = "utf-8"
        _redirect(str(self.path))

        # Background thread that mirrors the log file back onto the original
        # stdout/stderr descriptors so a real terminal still sees the output.
        self.thread = threading.Thread(
            target=self._echo, name="terminal-mirror", daemon=True
        )
        self.thread.start()
        atexit.register(self.close)

    def _echo(self) -> None:
        writer = _TerminalWriter(self.original_fds[0])
        with self.path.open("rb", buffering=0) as stream:
            while True:
                chunk = stream.read(64 * 1024)
                if chunk:
                    try:
                        writer.write(chunk)
                    except OSError:
                        # A closed terminal must not break the UI log.
                        pass
                elif self.stop.is_set():
                    try:
                        writer.write(b"", final=True)
                    except OSError:
                        pass
                    return
                else:
                    self.stop.wait(0.05)

    def read(self, start: int = 0) -> str:
        """Return at most ``_MAX_VIEW`` bytes of the log file, ending at EOF."""
        with self.path.open("rb") as stream:
            end = stream.seek(0, os.SEEK_END)
            offset = max(int(start), end - _MAX_VIEW)
            stream.seek(offset)
            data = stream.read(_MAX_VIEW)
        text = _ANSI.sub("", data.decode("utf-8", errors="replace"))
        # tqdm uses CR to refresh the current line; keep only the last update.
        text = "\n".join(
            line.split("\r")[-1]
            for line in text.replace("\r\n", "\n").split("\n")
        )
        prefix = (
            "[Earlier output omitted from this view]\n" if offset > int(start) else ""
        )
        return prefix + text

    def position(self) -> int:
        """Return the current byte position of the log (i.e. its size)."""
        return self.path.stat().st_size

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for stream in (sys.stdout, sys.stderr):
            stream.flush()
        for target, original in zip((1, 2), self.original_fds):
            os.dup2(original, target)
            if sys.platform == "win32":
                import ctypes
                import msvcrt

                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.SetStdHandle.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
                kernel.SetStdHandle(
                    (-11 if target == 1 else -12) & 0xFFFFFFFF,
                    msvcrt.get_osfhandle(target),
                )
        redirected = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = self.original_streams
        for stream in redirected:
            stream.close()
        for key, value in self.previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.stop.set()
        self.thread.join(timeout=2)
        if not self.thread.is_alive():
            for descriptor in self.original_fds:
                os.close(descriptor)
            try:
                self.directory.cleanup()
            except OSError:
                pass


def start_capture() -> "_TerminalCapture":
    """Idempotently start the terminal capture for this process."""
    global _capture
    if _capture is None or _capture.closed:
        _capture = _TerminalCapture()
    return _capture


def get_capture() -> "_TerminalCapture | None":
    """Return the active capture (or None if start_capture was never called)."""
    return _capture
