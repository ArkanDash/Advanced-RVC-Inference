import os, io
#import ssl
import sys
import time
import codecs
import logging
import warnings

import gradio as gr

sys.path.append(os.getcwd())
start_time = time.time()

from advanced_rvc_inference.tabs.extra.extra import extra_tab
from advanced_rvc_inference.tabs.training.training import training_tab
from advanced_rvc_inference.tabs.downloads.downloads import download_tab
from advanced_rvc_inference.tabs.inference.inference import inference_tab
from advanced_rvc_inference.variables import logger, config, translations, theme, font, configs, language, allow_disk
from advanced_rvc_inference.mainjs import js_code
#ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings("ignore")
for l in ["httpx", "uvicorn", "httpcore", "urllib3"]:
    logging.getLogger(l).setLevel(logging.ERROR)


client_mode = "--client" in sys.argv

with gr.Blocks(
    title="📱 Advanced RVC Inference",
    js=js_code if client_mode else None, 
    theme=theme,
) as app:
    gr.HTML("<h1 style='text-align: center;'>Advanced RVC Inference</h1>")
    

    with gr.Tabs():      
        inference_tab()


        if client_mode:
            from advanced_rvc_inference.tabs.realtime.realtime_client import realtime_client_tab
            realtime_client_tab()
        else:
            from advanced_rvc_inference.tabs.realtime.realtime import realtime_tab
            realtime_tab()

        training_tab()
        download_tab()
        extra_tab(app)

    
    with gr.Row(): 
        gr.Markdown(translations["terms_of_use"])
    with gr.Row():
        gr.Markdown(translations["exemption"])
    
    # This is the corrected line. It now correctly checks if the script is being run directly.
    if __name__ == "__main__":
        logger.info(config.device.replace("privateuseone", "dml"))
        logger.info(translations["start_app"])
        logger.info(translations["set_lang"].format(lang=language))
        port = configs.get("app_port", 7860)
        server_name = configs.get("server_name", "0.0.0.0")
        share = "--share" in sys.argv
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()

        for i in range(configs.get("num_of_restart", 5)):
            try:
                gradio_app, _, share_url = app.launch(
                    server_name=server_name, 
                    server_port=port, 
                    show_error=configs.get("app_show_error", False), 
                    inbrowser="--open" in sys.argv, 
                    share=share,
                    ssr_mode=True,
                    prevent_thread_lock=True,
                    allowed_paths=allow_disk,
                )
                break
            except OSError:
                logger.debug(translations["port"].format(port=port))
                port -= 1
            except Exception as e:
                logger.error(translations["error_occurred"].format(e=e))
                sys.exit(1)

        if client_mode:
            from advanced_rvc_inference.core.realtime_client import app as fastapi_app
            gradio_app.mount("/api", fastapi_app)
        
        sys.stdout = original_stdout
      
        
        logger.info(f"{translations['gradio_start']}: {(time.time() - start_time):.2f}s")

        print(f"{server_name}:{port}")
        if share: print(f"{share_url}")
        while 1:
            time.sleep(5)


#endcode
