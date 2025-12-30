"""
GUI Module for Advanced RVC Inference.

Launches the Gradio web interface for voice conversion,
training, and real-time processing.
"""

import os
import sys
import time
import logging
import subprocess
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


def get_version():
    """Get the package version."""
    try:
        from advanced_rvc_inference._version import __version__

        return __version__
    except ImportError:
        return "2.0.0"


def try_install_localtunnel():
    """Try to install localtunnel for alternative tunneling."""
    try:
        import importlib.util
        if importlib.util.find_spec("localtunnel") is None:
            logger.info("Installing localtunnel for alternative tunneling...")
            subprocess.run([sys.executable, "-m", "pip", "install", "localtunnel", "-q"],
                          check=True, capture_output=True)
            logger.info("localtunnel installed successfully")
            return True
    except Exception as e:
        logger.debug(f"Failed to install localtunnel: {e}")
    return False


def launch_with_localtunnel(port: int = 7860, app=None):
    """Launch with localtunnel as fallback."""
    try:
        import localtunnel
        
        # Start localtunnel
        tunnel = localtunnel.Server(port=port)
        
        # Get the public URL
        public_url = tunnel.url
        
        logger.info(f"LocalTunnel public URL: {public_url}")
        
        # Launch the app without share (localtunnel handles it)
        if app:
            app.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=False,
                ssr_mode=True,
                prevent_thread_lock=True,
            )
        
        return public_url
    except Exception as e:
        logger.error(f"LocalTunnel failed: {e}")
        return None


def launch(
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    inbrowser: bool = False,
    show_error: bool = False,
    prevent_thread_lock: bool = True,
    enable_localtunnel: bool = False,
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
        enable_localtunnel: Whether to use localtunnel as fallback if gradio share fails
    """
    setup_environment()

    try:
        import gradio as gr

        # Import required modules
        from advanced_rvc_inference.variables import (
            logger,
            config,
            translations,
            theme,
            font,
            configs,
            language,
            allow_disk,
        )
        from advanced_rvc_inference.mainjs import js_code
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context

        # Check for client mode
        client_mode = "--client" in sys.argv

        # Start time tracking
        start_time = time.time()

        # Build the UI
        with gr.Blocks(
            title=f"Advanced RVC Inference v{get_version()}",
            js=js_code if client_mode else None,
            theme=theme,
        ) as app:
            gr.HTML(
                f"<h1 style='text-align: center;'>Advanced RVC Inference v{get_version()}</h1>"
            )

            from advanced_rvc_inference.tabs.inference.inference import inference_tab
            from advanced_rvc_inference.tabs.realtime.realtime import realtime_tab
            from advanced_rvc_inference.tabs.training.training import training_tab
            from advanced_rvc_inference.tabs.downloads.downloads import download_tab
            from advanced_rvc_inference.tabs.extra.extra import extra_tab

            with gr.Tabs():
                inference_tab()

                if client_mode:
                    from advanced_rvc_inference.tabs.realtime.realtime_client import (
                        realtime_client_tab,
                    )

                    realtime_client_tab()
                else:
                    realtime_tab()

                training_tab()
                download_tab()
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

        # Log share warning for Colab users
        if share:
            logger.info("Attempting to create public share link...")
            logger.info("Note: If share link fails, use local URL http://0.0.0.0:7860")
            if enable_localtunnel:
                logger.info("LocalTunnel fallback is enabled")

        # Launch the app with retry logic for tunnel issues
        gradio_url = None
        share_failed = False

        try:
            logger.info("Starting Gradio server...")

            app.launch(
                server_name=server_name,
                server_port=port,
                show_error=show_error,
                inbrowser=inbrowser or "--open" in sys.argv,
                share=share,
                ssr_mode=True,
                prevent_thread_lock=prevent_thread_lock,
                allowed_paths=allow_disk,
            )

            # Log successful startup
            startup_time = time.time() - start_time
            logger.info(f"Server started in {startup_time:.2f}s")

            if share:
                logger.info(f"Access the UI locally: http://{server_name}:{port}")

        except Exception as tunnel_error:
            share_failed = True
            error_msg = str(tunnel_error).lower()

            # Check if it's a tunnel-related error
            if "tunnel" in error_msg or "share" in error_msg or "connection" in error_msg:
                logger.warning("Gradio share link creation failed")
                logger.info("Retrying with share=False (local access only)...")

                # Try again without share
                app.launch(
                    server_name=server_name,
                    server_port=port,
                    show_error=show_error,
                    inbrowser=False,
                    share=False,
                    ssr_mode=True,
                    prevent_thread_lock=prevent_thread_lock,
                    allowed_paths=allow_disk,
                )

                logger.warning("Share link disabled - using local access only")
                logger.info(f"Access locally at: http://{server_name}:{port}")

                if enable_localtunnel:
                    logger.info("Attempting LocalTunnel as alternative...")
                    try_install_localtunnel()
                    launch_with_localtunnel(port, None)
            else:
                raise tunnel_error

        # Final instructions
        logger.info("=" * 60)
        logger.info("Web UI is ready!")
        logger.info(f"Local URL:  http://{server_name}:{port}")
        if share and not share_failed:
            logger.info("Public URL: (check above for share link)")
        logger.info("=" * 60)

        return 0

    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please ensure all dependencies are installed")
        return 1
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {server_port} is already in use")
            logger.info("Try using a different port with: rvc-cli serve --port 7861")
        else:
            logger.error(f"Failed to start server: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to launch web interface: {e}")
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

        from advanced_rvc_inference.variables import (
            config,
            translations,
            theme,
            configs,
            language,
            allow_disk,
        )
        from advanced_rvc_inference.mainjs import js_code
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context

        client_mode = "--client" in sys.argv

        with gr.Blocks(
            title="Advanced RVC Inference",
            js=js_code if client_mode else None,
            theme=theme,
        ) as app:
            gr.HTML("<h1 style='text-align: center;'>Advanced RVC Inference</h1>")

            from advanced_rvc_inference.tabs.inference.inference import inference_tab
            from advanced_rvc_inference.tabs.realtime.realtime import realtime_tab
            from advanced_rvc_inference.tabs.training.training import training_tab
            from advanced_rvc_inference.tabs.downloads.downloads import download_tab
            from advanced_rvc_inference.tabs.extra.extra import extra_tab

            with gr.Tabs():
                inference_tab()

                if client_mode:
                    from advanced_rvc_inference.tabs.realtime.realtime_client import (
                        realtime_client_tab,
                    )

                    realtime_client_tab()
                else:
                    realtime_tab()

                training_tab()
                download_tab()
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
    parser.add_argument("--localtunnel", action="store_true", help="Use localtunnel as fallback if share fails")
    parser.add_argument("--open", action="store_true", help="Open in browser")

    args = parser.parse_args()

    sys.exit(
        launch(
            share=args.share and not args.no_share,
            server_name=args.host,
            server_port=args.port,
            inbrowser=args.open,
            enable_localtunnel=args.localtunnel,
        )
    )
