"""Live session terminal tab — ported from Applio's ``tabs/terminal/terminal.py``.

Polls the ``arvc.utils.terminal`` capture every 500 ms while the tab is
visible, so the user can follow training, downloads and inference output
without leaving the web UI.
"""

from __future__ import annotations

import gradio as gr

from arvc.utils.terminal import get_capture
from arvc.utils.variables import translations, configs, logger

# Inline CSS scoped to the terminal textarea — dark monospace panel that
# reads like a real terminal, regardless of the active theme.
TERMINAL_CSS = """
#arvc-terminal textarea {
    font-family: ui-monospace, Consolas, "Cascadia Mono", monospace;
    background: #101014;
    color: #e4e4e7;
    font-size: 13px;
    line-height: 1.5;
}
"""


def terminal_tab():
    """Render the Terminal tab. Returns the timer so callers can pause/resume it."""
    capture = get_capture()
    start = gr.State(0)
    last_position = gr.State(-1)

    gr.Markdown(
        translations.get(
            "terminal_intro",
            "Live output from this ARVC session — training, downloads and inference workers all stream here.",
        )
    )
    with gr.Row():
        live = gr.Checkbox(
            value=True,
            label=translations.get("terminal_live", "Live updates"),
        )
        clear = gr.Button(
            translations.get("terminal_clear", "Clear view"),
            size="sm",
        )

    output = gr.Textbox(
        label=translations.get("terminal_label", "Terminal"),
        value=(capture.read() if capture else ""),
        lines=24,
        max_lines=24,
        interactive=False,
        autoscroll=True,
        elem_id="arvc-terminal",
    )

    if capture is None:
        gr.Markdown(
            f"> ⚠️ {translations.get('terminal_unavailable', 'Terminal capture not started — `start_capture()` was never called for this process.')}"
        )
        return None, live

    # Poll only while the Terminal tab is visible. We rely on Gradio 5+'s
    # `gr.Timer` (older versions fall back to a no-op timer).
    try:
        timer = gr.Timer(0.5, active=False)
    except (AttributeError, TypeError):
        timer = None

    def refresh(cursor, previous):
        if capture is None:
            return gr.skip(), gr.skip()
        position = capture.position()
        if position == previous:
            return gr.skip(), gr.skip()
        return capture.read(cursor), position

    def clear_view():
        if capture is None:
            return gr.skip(), gr.skip(), gr.skip()
        position = capture.position()
        return position, position, ""

    if timer is not None:
        timer.tick(
            refresh,
            inputs=[start, last_position],
            outputs=[output, last_position],
            queue=False,
            show_progress="hidden",
        )
        live.change(
            lambda active: gr.Timer(active=active),
            inputs=live,
            outputs=timer,
            queue=False,
        )
        clear.click(clear_view, outputs=[start, last_position, output], queue=False)
    else:
        logger.warning(
            translations.get(
                "terminal_timer_unsupported",
                "Installed Gradio version does not support gr.Timer; terminal tab will not auto-refresh.",
            )
        )

    return timer, live
