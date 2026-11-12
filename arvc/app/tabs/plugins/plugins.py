"""
Plugins tab for the Advanced RVC Inference GUI.

Ported from Applio's plugin system (tabs/plugins) and adapted to the
``arvc`` package: install plugin.zip archives, then every installed
plugin's ``applio_plugin()`` function is rendered inside its own tab.
"""

import gradio as gr

from arvc.utils.variables import translations
from arvc.utils.feedback import gr_info
from arvc.utils.restart import restart_app
from arvc.utils.plugin_manager import install_plugin, list_installed_plugins, load_plugin


def plugins_tab(app):
    """Create the Plugins tab (installer + one sub-tab per installed plugin)."""

    def _install_with_toast(dropbox):
        gr_info(translations.get("plugin_installing", "Installing plugin..."))
        try:
            return install_plugin(dropbox)
        except Exception as e:  # noqa: BLE001
            return translations.get("plugin_install_failed", "Plugin installation failed") + f": {e}"

    with gr.TabItem(translations.get("plugins", "Plugins")):
        with gr.Tabs():
            # ── Installer ──
            with gr.TabItem(translations.get("plugin_installer", "Plugin Installer")):
                gr.Markdown(
                    translations.get(
                        "plugin_markdown",
                        "Drop a plugin.zip below. Each plugin adds its own tab after a restart.",
                    )
                )
                dropbox = gr.File(
                    label=translations.get(
                        "plugin_drop_info",
                        "Drop a plugin.zip here or use the browse button to install it",
                    ),
                    file_types=[".zip"],
                    type="filepath",
                )
                install_status = gr.Textbox(
                    label=translations.get("status", "Status"),
                    interactive=False,
                )

                with gr.Row():
                    restart_button = gr.Button(
                        translations.get("restart_app", "Restart App"),
                        variant="primary",
                    )

                dropbox.upload(
                    fn=_install_with_toast,
                    inputs=[dropbox],
                    outputs=[install_status],
                )

                restart_button.click(
                    fn=lambda: restart_app(app),
                    inputs=None,
                    outputs=None,
                )

            # ── Installed plugins ──
            for plugin_name in list_installed_plugins():
                try:
                    plugin_module = load_plugin(plugin_name)
                    with gr.TabItem(plugin_name):
                        plugin_module.applio_plugin()
                except Exception as e:  # noqa: BLE001
                    import traceback

                    traceback.print_exc()
                    with gr.TabItem(plugin_name):
                        gr.Markdown(
                            f"**{translations.get('plugin_load_failed', 'Failed to load plugin')} `{plugin_name}`**\n\n"
                            f"```\n{e}\n```"
                        )
