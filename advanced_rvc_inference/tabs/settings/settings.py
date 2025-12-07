import os
import sys
import gradio as gr

# Add project root to path
now_dir = os.getcwd()
sys.path.append(now_dir)

from advanced_rvc_inference.assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

from advanced_rvc_inference.tabs.settings.sections.lang import lang_tab
from advanced_rvc_inference.tabs.settings.sections.restart import restart_tab
from advanced_rvc_inference.tabs.settings.sections.model_author import model_author_tab
from advanced_rvc_inference.tabs.settings.sections.precision import precision_tab
from advanced_rvc_inference.tabs.settings.sections.filter import filter_tab, get_filter_trigger


def settings_tab(filter_state_trigger=None):
    if filter_state_trigger is None:
        filter_state_trigger = get_filter_trigger()

    with gr.TabItem(label=i18n("General")):
        filter_component = filter_tab()

        filter_component.change(
            fn=lambda checked: gr.Checkbox.update(value=str(checked)),
            inputs=[filter_component],
            outputs=[filter_state_trigger],
            show_progress=False,
        )
        lang_tab()
        restart_tab()
    with gr.TabItem(label=i18n("Training")):
        model_author_tab()
        precision_tab()
