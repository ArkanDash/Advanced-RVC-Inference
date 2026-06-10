"""
GUI Module for Advanced RVC Inference.

Launches the Gradio web interface for voice conversion,
training, and real-time processing.
"""

import os
import sys
import time
import logging
import signal
import threading
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup the environment for RVC operations."""
    # Add current directory to path
    cwd = Path.cwd()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))


def _setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    if sys.platform != 'win32':
        original_sigint = signal.getsignal(signal.SIGINT)
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            sys.exit(0)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)


def _keep_alive():
    """Keep the process alive for tunnel maintenance."""
    try:
        while True:
            time.sleep(60)
            logger.debug("Tunnel keepalive check...")
    except KeyboardInterrupt:
        logger.info("Keepalive loop interrupted")


def launch(
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    inbrowser: bool = False,
    show_error: bool = False,
    prevent_thread_lock: bool = True,
    keep_alive: bool = True,
):
    """
    Launch the Gradio web interface.

    Args:
        share: Whether to create a public URL using Gradio sharing
        server_name: Host to bind to
        server_port: Port to bind to
        inbrowser: Whether to open the URL in a browser
        show_error: Whether to show errors in the UI
        prevent_thread_lock: Whether to prevent thread locking
        keep_alive: Whether to keep the tunnel alive (useful for Colab)
    """
    setup_environment()
    _setup_signal_handlers()

    try:
        import gradio as gr

        # Import required modules
        from arvc.utils.variables import (
            logger,          # Note: this overrides the top-level logger; consider using a different name
            config,
            translations,
            theme,
            configs,
            language,
            allow_disk,
        )
        from arvc.app.mainjs import js_code

        # Check for client mode
        client_mode = "--client" in sys.argv

        # Start time tracking
        start_time = time.time()

        # Build the UI
        # Gradio 6.0+ moved theme/js to launch(); we pass them to
        # Blocks() only if the installed version still accepts them.
        import packaging.version as pv
        _gradio_ver = pv.parse(gr.__version__)
        _use_blocks_theme = _gradio_ver < pv.parse("6.0")

        blocks_kwargs = dict(title="Advanced RVC Inference")
        if _use_blocks_theme:
            blocks_kwargs["theme"] = theme
            blocks_kwargs["js"] = js_code if client_mode else None

        with gr.Blocks(**blocks_kwargs) as app:
            gr.HTML(
                f"<h1 style='text-align: center;'>Advanced RVC Inference</h1>"
            )

            from arvc.app.tabs.inference.inference import inference_tab
            from arvc.app.tabs.realtime.realtime import realtime_tab
            from arvc.app.tabs.training.training import training_tab
            from arvc.app.tabs.downloads.downloads import download_tab
            from arvc.app.tabs.extra.extra import extra_tab

            with gr.Tabs():
                with gr.TabItem("Inference"):
                    inference_tab()
                    if client_mode:
                        from arvc.app.tabs.realtime.realtime_client import (
                            realtime_client_tab,
                        )
                        realtime_client_tab()
                    else:
                        realtime_tab()
                
                with gr.TabItem("Models"):
                    download_tab()
                    training_tab()
                
                extra_tab(app)

            with gr.Row():
                gr.Markdown(translations["terms_of_use"])
            with gr.Row():
                gr.Markdown(translations["exemption"])

        # Log startup
        logger.info(f"Device: {config.device.replace('privateuseone', 'dml')}")
        logger.info(translations["start_app"])
        logger.info(translations["set_lang"].format(lang=language))

        # Determine port and share settings
        port = configs.get("app_port", server_port)
        server_name = configs.get("server_name", server_name)
        share = share or "--share" in sys.argv

        # Build allowed paths list - include package assets directory
        def get_package_assets_path():
            """Get assets directory path, handling both source and installed cases."""
            # Try to get path from the package's assets module
            try:
                from arvc.assets import ASSETS_PATH
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
                spec = importlib.util.find_spec("arvc")
                if spec and spec.origin:
                    package_dir = Path(spec.origin).parent.parent
                    assets_path = package_dir / "assets"
                    if assets_path.exists():
                        return str(assets_path)
            except Exception:
                pass
            
            return None

        allowed_paths_list = list(allow_disk) if allow_disk else []
        assets_path = get_package_assets_path()
        if assets_path and assets_path not in allowed_paths_list:
            allowed_paths_list.append(assets_path)
            logger.debug(f"Added package assets path to allowed_paths: {assets_path}")
        
        logger.debug(f"Allowed paths: {allowed_paths_list}")

        # Launch the app
        logger.info("Starting Gradio server...")

        # Gradio 6.0+ accepts theme/js in launch()
        launch_kwargs = dict(
            server_name=server_name,
            server_port=port,
            show_error=show_error,
            inbrowser=inbrowser or "--open" in sys.argv,
            share=share,
            allowed_paths=allowed_paths_list,
        )
        if not _use_blocks_theme:
            launch_kwargs["theme"] = theme
            if client_mode:
                launch_kwargs["js"] = js_code

        try:
            app.launch(**launch_kwargs)

            # Log successful startup
            startup_time = time.time() - start_time
            logger.info(f"Server started in {startup_time:.2f}s")
            logger.info(f"Access the UI at: http://{server_name}:{port}")

            if share and app.share_url:
                logger.info(f"Public URL: {app.share_url}")

        except Exception as tunnel_error:
            error_msg = str(tunnel_error).lower()

            # Check if it's a tunnel-related error
            if "tunnel" in error_msg or "share" in error_msg or "connection" in error_msg:
                logger.warning("Gradio share link creation failed")
                logger.info("Retrying with share=False (local access only)...")

                # Try again without share
                retry_kwargs = dict(
                    server_name=server_name,
                    server_port=port,
                    show_error=show_error,
                    inbrowser=False,
                    share=False,
                    allowed_paths=allowed_paths_list,
                )
                if not _use_blocks_theme:
                    retry_kwargs["theme"] = theme
                    if client_mode:
                        retry_kwargs["js"] = js_code

                app.launch(**retry_kwargs)

                logger.warning("Share link disabled - using local access only")
                logger.info(f"Access locally at: http://{server_name}:{port}")
            else:
                # Re-raise if it's not a share/tunnel issue
                raise tunnel_error

        # Keep tunnel alive in notebook environments (Colab, etc.)
        if keep_alive and share:
            logger.info("Keeping tunnel alive... (Press Ctrl+C to stop)")
            try:
                _keep_alive()
            except KeyboardInterrupt:
                logger.info("Shutting down...")

        return 0

    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please ensure all dependencies are installed")
        return 1
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {port} is already in use")
            logger.info("Try using a different port with: rvc-cli serve --port 7861")
        else:
            logger.error(f"Failed to start server: {e}")
        return 1
    except Exception as e:
        import traceback
        logger.error(f"Failed to launch web interface: {e}")
        logger.debug(traceback.format_exc())
        logger.info("Try running with: rvc-gui --share=False")
        return 1


def create_app():
    """
    Create the Gradio app instance for embedding.

    Returns:
        gr.Blocks: The Gradio app instance
    """
    setup_environment()

    try:
        import gradio as gr
        import packaging.version as pv

        from arvc.utils.variables import (
            config,
            translations,
            theme,
            configs,
            language,
            allow_disk,
        )
        from arvc.app.mainjs import js_code

        client_mode = "--client" in sys.argv
        _gradio_ver = pv.parse(gr.__version__)
        _use_blocks_theme = _gradio_ver < pv.parse("6.0")

        blocks_kwargs = dict(title="Advanced RVC Inference")
        if _use_blocks_theme:
            blocks_kwargs["theme"] = theme
            blocks_kwargs["js"] = js_code if client_mode else None

        with gr.Blocks(**blocks_kwargs) as app:
            gr.HTML("<h1 style='text-align: center;'>Advanced RVC Inference</h1>")

            from arvc.app.tabs.inference.inference import inference_tab
            from arvc.app.tabs.realtime.realtime import realtime_tab
            from arvc.app.tabs.training.training import training_tab
            from arvc.app.tabs.downloads.downloads import download_tab
            from arvc.app.tabs.extra.extra import extra_tab

            with gr.Tabs():
                with gr.TabItem("Inference"):
                    inference_tab()
                    if client_mode:
                        from arvc.app.tabs.realtime.realtime_client import (
                            realtime_client_tab,
                        )
                        realtime_client_tab()
                    else:
                        realtime_tab()

                with gr.TabItem("Models"):
                    download_tab()
                    training_tab()
                
                with gr.TabItem("Extra"):
                    extra_tab(app)

            with gr.Row():
                gr.Markdown(translations["terms_of_use"])
            with gr.Row():
                gr.Markdown(translations["exemption"])

        return app

    except ImportError as e:
        logger.error(f"Failed to create app: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch Advanced RVC Inference GUI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public URL (may fail in some environments)")
    parser.add_argument("--no-share", action="store_true", help="Disable public URL, use local access only")
    parser.add_argument("--open", action="store_true", help="Open in browser")
    parser.add_argument("--keep-alive", action="store_true", default=True, help="Keep tunnel alive (default: True)")

    args, _unknown = parser.parse_known_args()

    sys.exit(
        launch(
            share=args.share and not args.no_share,
            server_name=args.host,
            server_port=args.port,
            inbrowser=args.open,
            keep_alive=args.keep_alive,
        )
    )
