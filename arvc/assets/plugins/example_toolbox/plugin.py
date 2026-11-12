"""
Example plugin for Advanced RVC Inference.

Demonstrates the plugin contract (same as Applio):

- Ship a folder containing a ``plugin.py`` zipped as ``<name>.zip``
- ``plugin.py`` must define ``applio_plugin()`` which builds Gradio
  components directly (they are rendered inside the plugin's own tab)
- An optional ``requirements.txt`` is installed with pip on install

Install it live from the Plugins tab, or keep it in this folder to ship
it with the app.
"""

import gradio as gr


def applio_plugin():
    """Build the example plugin UI (rendered inside its own tab)."""
    gr.Markdown("### Example plugin\nThis UI comes from a plugin in `arvc/assets/plugins/example_toolbox/`.")

    with gr.Row():
        name_box = gr.Textbox(label="Your name", placeholder="Type something...", interactive=True)
        greeting_box = gr.Textbox(label="Greeting", interactive=False)

    hello_button = gr.Button("Say hello", variant="primary")
    hello_button.click(
        fn=lambda name: f"Hello, {name or 'world'}! This response was produced by a plugin.",
        inputs=[name_box],
        outputs=[greeting_box],
    )
