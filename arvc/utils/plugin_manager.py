"""
Plugin manager for Advanced RVC Inference.

Ported from Applio's plugin system and adapted to the ``arvc`` package
layout. A plugin is a folder containing a ``plugin.py`` module that
exposes an ``applio_plugin()`` function which builds arbitrary Gradio
components (typically inside the plugin's own TabItem, rendered by the
Plugins tab). A plugin folder may also ship an optional
``requirements.txt`` which is installed with pip on first install.

Layout::

    <project>/arvc/assets/plugins/
        <plugin_name>/
            plugin.py          # must define applio_plugin()
            requirements.txt   # optional
"""

import os
import shutil
import subprocess
import sys
import zipfile

from arvc.utils.variables import ASSETS_PATH

PLUGINS_PATH = ASSETS_PATH / "plugins"


def _ensure_plugins_path():
    """Create the plugins directory if it does not exist yet."""
    PLUGINS_PATH.mkdir(parents=True, exist_ok=True)


def list_installed_plugins():
    """Return the sorted names of installed plugin folders.

    A folder counts as a plugin only when it contains a ``plugin.py``.
    """
    _ensure_plugins_path()
    return sorted(
        entry
        for entry in os.listdir(PLUGINS_PATH)
        if (PLUGINS_PATH / entry / "plugin.py").is_file()
    )


def load_plugin(name):
    """Import and return the ``plugin`` module of an installed plugin."""
    import importlib.util

    plugin_file = PLUGINS_PATH / name / "plugin.py"
    if not plugin_file.is_file():
        raise FileNotFoundError(f"Plugin '{name}' has no plugin.py")

    spec = importlib.util.spec_from_file_location(f"arvc_plugin_{name}", plugin_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for plugin '{name}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_extract(zip_path, target_dir):
    """Extract a zip archive while rejecting entries that escape ``target_dir``.

    Improves on the Applio original by blocking zip-slip archives (entries
    with ``..`` or absolute paths) instead of blindly extracting them.
    """
    target = os.path.realpath(target_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            member_path = os.path.realpath(os.path.join(target_dir, member))
            if member_path != target and not member_path.startswith(target + os.sep):
                raise zipfile.BadZipFile(f"Unsafe path in plugin zip: {member}")
        zf.extractall(target_dir)


def install_plugin(zip_path):
    """Install a plugin from a ``.zip`` file.

    Steps: validate the archive, extract it into ``PLUGINS_PATH``, install
    the plugin's ``requirements.txt`` (if any) with pip, and return a
    human-readable status message. A restart of the app is required for
    the new plugin's tab to appear.
    """
    _ensure_plugins_path()

    if not zip_path or not str(zip_path).lower().endswith(".zip"):
        return "Invalid file: plugins must be installed from a .zip archive."

    file_name = os.path.basename(zip_path)
    folder_name = os.path.splitext(file_name)[0]
    folder_path = os.path.join(PLUGINS_PATH, folder_name)

    # Remove any previous installation of the same plugin
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path, ignore_errors=True)

    zip_file_path = os.path.join(PLUGINS_PATH, file_name)
    shutil.move(zip_path, zip_file_path)

    # Snapshot the directory so a failed validation can undo everything
    # the archive created (prevents stray or misplaced folders).
    before = set(os.listdir(PLUGINS_PATH))
    try:
        _safe_extract(zip_file_path, PLUGINS_PATH)
    finally:
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)

    # The archive must contain a top-level folder named after the zip that
    # holds plugin.py (standard Applio plugin layout: <name>/<name>/plugin.py
    # is NOT valid — plugin.py must live directly inside <name>/).
    if not os.path.isfile(os.path.join(folder_path, "plugin.py")):
        for entry in set(os.listdir(PLUGINS_PATH)) - before:
            full = os.path.join(PLUGINS_PATH, entry)
            if os.path.isdir(full):
                shutil.rmtree(full, ignore_errors=True)
            else:
                os.remove(full)
        return (
            f"Invalid plugin archive: '{file_name}' must contain "
            f"'{folder_name}/plugin.py' with an applio_plugin() function."
        )

    # Install optional plugin dependencies
    requirements = os.path.join(folder_path, "requirements.txt")
    if os.path.isfile(requirements):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements],
            shell=False,
            check=False,
        )
    else:
        print(f"No requirements.txt found for plugin '{folder_name}'.")

    return (
        f"Plugin '{folder_name}' installed successfully in {PLUGINS_PATH}. "
        "Restart the app to load its tab."
    )
