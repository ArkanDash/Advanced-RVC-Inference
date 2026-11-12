"""
Report tab for the Advanced RVC Inference GUI.

Ported from Applio (tabs/report, MIT licensed): record the screen from
the browser, save the clip as a temporary MP4, and attach it to a GitHub
issue for easier bug reports.
"""

import base64
import pathlib
import tempfile

import gradio as gr

from arvc.utils.variables import translations

REPORT_JS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ISSUES_URL = "https://github.com/ArkanDash/Advanced-RVC-Inference/issues"


def save_base64_video(base64_string):
    """Decode a base64 (webm/mp4) recording into a temporary .mp4 file."""
    video_data = base64.b64decode(base64_string)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(video_data)
    return temp_filename


def report_tab():
    """Create the Report tab with the screen-recording helper."""
    recorder_js = (REPORT_JS_DIR / "recorder.js").read_text(encoding="utf-8")
    main_js = (REPORT_JS_DIR / "main.js").read_text(encoding="utf-8")
    record_button_js = (
        (REPORT_JS_DIR / "record_button.js")
        .read_text(encoding="utf-8")
        .replace("let recorder_js = null;", recorder_js)
        .replace("let main_js = null;", main_js)
    )

    instructions = [
        "# " + translations.get("report_title", "How to Report an Issue on GitHub"),
        "1. " + translations.get(
            "report_step_1",
            "Click on the 'Record Screen' button below to start recording the issue you are experiencing.",
        ),
        "2. " + translations.get(
            "report_step_2",
            "Once you have finished recording the issue, click on the 'Stop Recording' button (the same button, but the label changes depending on whether you are actively recording or not).",
        ),
        "3. " + translations.get(
            "report_step_3",
            "Go to GitHub Issues and click on the 'New Issue' button.",
        ).replace("GitHub Issues", f"[GitHub Issues]({REPO_ISSUES_URL})"),
        "4. " + translations.get(
            "report_step_4",
            "Complete the provided issue template, ensuring to include details as needed, and utilize the assets section to upload the recorded file from the previous step.",
        ),
    ]
    for instruction in instructions:
        gr.Markdown(value=instruction)

    start_button = gr.Button(translations.get("record_screen", "Record Screen"))
    video_component = gr.Video(interactive=False)

    def toggle_button_label(returned_string):
        if returned_string.startswith("Record"):
            # "Record" is the value the record_button.js callback returns to
            # signal that recording started; it is a protocol token, not a
            # translatable string.
            return gr.Button(value=translations.get("stop_recording", "Stop Recording")), None
        try:
            temp_filename = save_base64_video(returned_string)
        except Exception as error:  # noqa: BLE001
            print(f"An error occurred converting video to mp4: {error}")
            return gr.Button(value=translations.get("record_screen", "Record Screen")), gr.Warning(
                translations.get("report_convert_failed", "Failed to convert video to mp4") + f": {error}"
            )
        return gr.Button(value=translations.get("record_screen", "Record Screen")), gr.Video(
            value=temp_filename, interactive=False
        )

    start_button.click(
        fn=toggle_button_label,
        inputs=[start_button],
        outputs=[start_button, video_component],
        js=record_button_js,
    )
