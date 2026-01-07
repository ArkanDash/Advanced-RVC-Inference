import os, io, ssl, sys
import time, codecs, logging, warnings
import gradio as gr
from pathlib import Path

sys.path.append(os.getcwd())
start_time = time.time()

from advanced_rvc_inference.app.tabs.extra.extra import extra_tab
from advanced_rvc_inference.app.tabs.training.training import training_tab
from advanced_rvc_inference.app.tabs.downloads.downloads import download_tab
from advanced_rvc_inference.app.tabs.inference.inference import inference_tab
from advanced_rvc_inference.utils.variables import logger, config, translations, theme, font, configs, language, allow_disk
from advanced_rvc_inference.mainjs import js_code
ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings("ignore")
for l in ["httpx", "gradio", "uvicorn", "httpcore", "urllib3"]:
    logging.getLogger(l).setLevel(logging.ERROR)


def get_package_assets_path():
    """Get assets directory path, the package handling both source and installed cases."""
    # Try to get path from the package's assets module
    try:
        from advanced_rvc_inference.assets import ASSETS_PATH
        return str(ASSETS_PATH)
    except ImportError:
        pass
    
    # Fallback: try to get path from this file's location
    try:
        package_root = Path(__file__).parent.parent
        assets_path = package_root / "assets"
        if assets_path.exists():
            return str(assets_path)
    except Exception:
        pass
    
    # Last resort: try using importlib to find the package location
    try:
        import importlib.util
        spec = importlib.util.find_spec("advanced_rvc_inference")
        if spec and spec.origin:
            package_dir = Path(spec.origin).parent.parent
            assets_path = package_dir / "assets"
            if assets_path.exists():
                return str(assets_path)
    except Exception:
        pass
    
    return None


# Build the allowed paths list - include package assets directory
allowed_paths_list = list(allow_disk) if allow_disk else []
assets_path = get_package_assets_path()
if assets_path and assets_path not in allowed_paths_list:
    allowed_paths_list.append(assets_path)
    logger.debug(f"Added package assets path to allowed_paths: {assets_path}")


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
            from advanced_rvc_inference.app.tabs.realtime.realtime_client import realtime_client_tab
            realtime_client_tab()
        else:
            from advanced_rvc_inference.app.tabs.realtime.realtime import realtime_tab
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
                    allowed_paths=allowed_paths_list,
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

        print(f"* Run On: {server_name}:{port}")
        if share: print(f"* Run On Public URL: {share_url}")
        while 1:
            time.sleep(5)


#endcode
