import os 
import gradio as gr
import assets.themes.loadThemes as loadThemes

def theme_loader():
  gr.Markdown("Theme Loader for UI")
  themes_select = gr.Dropdown(
    label = "Theme",
    info = "Select the theme you want to use. (Requires restarting the App)",
    choices = loadThemes.get_list(),
    value = loadThemes.read_json(),
    visible = True
  )
  goofy_output = gr.Textbox(visible = False)
  themes_select.change(
    fn = loadThemes.select_theme,
    inputs = themes_select,
    outputs = [goofy_output]
  )
