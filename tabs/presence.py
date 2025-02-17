import os
import sys
import gradio as gr
import json
from assets.discord_presence import RPCManager

now_dir = os.getcwd()
sys.path.append(now_dir)


config_file = os.path.join(now_dir, "assets", "config.json")


def load_config_presence():
    with open(config_file, "r", encoding="utf8") as file:
        config = json.load(file)
        return config["discord_presence"]


def save_config(value):
    with open(config_file, "r", encoding="utf8") as file:
        config = json.load(file)
        config["discord_presence"] = value
    with open(config_file, "w", encoding="utf8") as file:
        json.dump(config, file, indent=2)


def presence_tab():
    with gr.Row():
        with gr.Column():
            presence = gr.Checkbox(
                label="Enable Advanced-RVC integration with Discord presence",
                info="It will activate the possibility of displaying the current Advanced-RVC activity in Discord.",
                interactive=True,
                value=load_config_presence(),
            )
            presence.change(
                fn=toggle,
                inputs=[presence],
                outputs=[],
            )


def toggle(checkbox):
    save_config(bool(checkbox))
    if load_config_presence() == True:
        try:
            RPCManager.start_presence()
        except KeyboardInterrupt:
            RPCManager.stop_presence()
    else:
        RPCManager.stop_presence()
