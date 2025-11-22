"""
Professional Application Launcher
Handles the main Gradio application creation and management
"""

import os
import sys
import logging
import ssl
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gradio as gr
    from advanced_rvc_inference.assets.i18n.i18n import I18nAuto
    from advanced_rvc_inference.config import Config
    from advanced_rvc_inference.core.memory_manager import memory_manager, should_cleanup
    KADVC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    KADVC_AVAILABLE = False


class AdvancedRVCApp:
    """
    Professional Application Launcher for Advanced RVC Inference
    Manages the complete Gradio application lifecycle
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the application launcher.
        
        Args:
            config: Configuration instance (uses global config if None)
        """
        self.config = config
        if config is None:
            from advanced_rvc_inference.config import config as global_config
            self.config = global_config
        
        self._logger = logging.getLogger(__name__)
        self._app = None
        self._i18n = None
        self._theme = "gradio/default"
        
        # Setup SSL and warnings
        self._setup_environment()
        
        # Initialize i18n
        self._initialize_i18n()
        
        self._logger.info("Advanced RVC Application Launcher initialized")
    
    def _setup_environment(self) -> None:
        """Setup environment settings."""
        # SSL handling
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Suppress warnings
        warnings.filterwarnings("ignore")
        for logger_name in ["httpx", "gradio", "uvicorn", "httpcore", "urllib3", "faiss"]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        
        self._logger.debug("Environment setup completed")
    
    def _initialize_i18n(self) -> None:
        """Initialize internationalization."""
        try:
            self._i18n = I18nAuto()
            self._logger.debug("Internationalization initialized")
        except Exception as e:
            self._logger.warning(f"Could not initialize i18n: {e}")
            self._i18n = None
    
    def _get_system_status_html(self) -> str:
        """Generate system status HTML."""
        try:
            memory_info = memory_manager.get_memory_info()
            perf_report = self.config.get_performance_report()
            
            # Format memory usage
            if memory_info.get('gpu'):
                gpu_used = memory_info['gpu']['allocated_gb']
                gpu_total = memory_info['gpu']['total_gb']
                gpu_percent = (gpu_used / gpu_total) * 100 if gpu_total > 0 else 0
                gpu_status = f"🖥️ GPU: {gpu_used:.1f}GB / {gpu_total:.1f}GB ({gpu_percent:.1f}%)"
            else:
                gpu_status = "🖥️ GPU: Not Available"
            
            system_used = memory_info['system']['used_gb']
            system_total = memory_info['system']['total_gb']
            system_percent = memory_info['system']['percent']
            
            status_html = f"""
            <div class="config-panel">
                <h3>🔧 System Status</h3>
                <div class="status-indicator status-success">
                    ✅ System Ready - Advanced RVC Inference v{self.config.app_config['version']}
                </div>
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                    <div><strong>🖥️ Platform:</strong> {os.name}</div>
                    <div><strong>🐍 Python:</strong> {sys.version.split()[0]}</div>
                    <div><strong>📁 Working Dir:</strong> {os.getcwd()}</div>
                    <div><strong>⚙️ Config:</strong> Loaded</div>
                </div>
                <div style="margin-top: 15px;">
                    <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                        <div><strong>💾 System RAM:</strong> {system_used:.1f}GB / {system_total:.1f}GB ({system_percent:.1f}%)</div>
                        <div>{gpu_status}</div>
                        <div><strong>🚀 Device:</strong> {perf_report['device']}</div>
                        <div><strong>📦 Batch Size:</strong> {perf_report['batch_size']}</div>
                    </div>
                </div>
                <div style="margin-top: 15px;">
                    <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                        <div><strong>🎵 Sample Rate:</strong> {perf_report['sample_rate']}Hz</div>
                        <div><strong>⚡ Mixed Precision:</strong> {'Enabled' if perf_report['mixed_precision'] else 'Disabled'}</div>
                        <div><strong>🔧 Compile:</strong> {'Enabled' if perf_report['compile_enabled'] else 'Disabled'}</div>
                        <div><strong>🚀 KADVC:</strong> {'Available' if KADVC_AVAILABLE else 'Not Available'}</div>
                    </div>
                </div>
            </div>
            """
            
            return status_html
            
        except Exception as e:
            self._logger.warning(f"Could not generate system status: {e}")
            return """
            <div class="config-panel">
                <h3>🔧 System Status</h3>
                <div class="status-indicator status-info">
                    ℹ️ System status could not be displayed
                </div>
            </div>
            """
    
    def _get_enhanced_css(self) -> str:
        """Generate enhanced CSS for the application."""
        font_url = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
        
        css = f"""
        @import url('{font_url}');
        
        * {{
            font-family: 'Inter', sans-serif !important;
        }}
        
        body, html {{
            font-family: 'Inter', sans-serif !important;
            line-height: 1.6;
        }}
        
        .enhanced-header {{
            text-align: center; 
            padding: 30px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            border-radius: 15px; 
            margin-bottom: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        
        .enhanced-tab {{ 
            padding: 20px; 
            border-radius: 12px; 
            margin: 10px;
            border: 1px solid #e1e5e9;
            background: #ffffff;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .status-indicator {{
            padding: 15px; 
            margin: 15px 0; 
            border-radius: 8px; 
            text-align: center;
            font-weight: 600;
            border: 1px solid transparent;
        }}
        
        .status-success {{ 
            background-color: #d1edda; 
            color: #155724; 
            border-color: #c3e6cb;
        }}
        
        .status-error {{ 
            background-color: #f8d7da; 
            color: #721c24; 
            border-color: #f5c6cb;
        }}
        
        .status-info {{ 
            background-color: #d1ecf1; 
            color: #0c5460; 
            border-color: #bee5eb;
        }}
        
        .config-panel {{
            background: linear-gradient(145deg, #f8f9fa, #e9ecef);
            padding: 25px;
            border-radius: 15px;
            margin: 15px 0;
            border: 1px solid #dee2e6;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }}
        
        .gradio-container {{
            max-width: 1200px !important;
            margin: auto !important;
        }}
        
        .tab-nav {{
            background: linear-gradient(145deg, #f8f9fa, #e9ecef) !important;
            border-radius: 12px !important;
            padding: 8px !important;
        }}
        
        .tab-nav button {{
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }}
        
        .tab-nav button:hover {{
            background: rgba(102, 126, 234, 0.1) !important;
            transform: translateY(-1px) !important;
        }}
        
        .primary-button {{
            background: linear-gradient(145deg, #667eea, #764ba2) !important;
            border: none !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }}
        
        .primary-button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4) !important;
        }}
        
        /* Loading spinner */
        .loading-spinner {{
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        """
        
        return css
    
    def _load_tab_with_error_handling(self, tab_func, tab_name: str):
        """Load a tab with error handling."""
        try:
            tab_func()
            self._logger.debug(f"Successfully loaded {tab_name} tab")
        except Exception as e:
            self._logger.error(f"Error loading {tab_name} tab: {e}")
            gr.HTML(f"""
            <div style="color: red; padding: 20px; text-align: center; border: 1px solid #f5c6cb; border-radius: 8px; background-color: #f8d7da;">
                <h4>❌ Error Loading {tab_name}</h4>
                <p>{str(e)}</p>
                <p>Please check the logs for more details.</p>
            </div>
            """)
    
    def create_app(self):
        """Create the complete Gradio application."""
        try:
            with gr.Blocks(
                theme=self._theme,
                title=self.config.app_config['name'],
                css=self._get_enhanced_css(),
                head="""
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <link rel="icon" type="image/png" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==">
                """
            ) as app:
                
                # Enhanced header
                gr.HTML(f"""
                <div class="enhanced-header">
                    <h1>🎤 {self.config.app_config['name']}</h1>
                    <p>Professional Voice Conversion with Advanced Features</p>
                    <p><small>Version {self.config.app_config['version']} - Enhanced with KADVC</small></p>
                </div>
                """)
                
                # System status
                status_html = self._get_system_status_html()
                status_display = gr.HTML(status_html)
                
                # Memory management indicator
                if should_cleanup():
                    gr.HTML("""
                    <div class="status-indicator status-warning" style="background-color: #fff3cd; color: #856404; border-color: #ffeaa7;">
                        <strong>⚠️ Memory Usage High</strong> - Automatic cleanup will be performed
                    </div>
                    """)
                
                # Main application tabs
                with gr.Tab("🎤 Full Inference"):
                    self._load_tab_with_error_handling(
                        self._load_inference_tab, "Full Inference"
                    )
                
                with gr.Tab("📥 Download Model"):
                    self._load_tab_with_error_handling(
                        self._load_download_tab, "Download Model"
                    )
                
                with gr.Tab("🎵 Datasets Maker"):
                    self._load_tab_with_error_handling(
                        self._load_datasets_tab, "Datasets Maker"
                    )
                
                with gr.Tab("🗣️ TTS"):
                    self._load_tab_with_error_handling(
                        self._load_tts_tab, "TTS"
                    )
                
                with gr.Tab("🛠️ Extra Tools"):
                    self._load_tab_with_error_handling(
                        self._load_extra_tab, "Extra Tools"
                    )
                
                with gr.Tab("🎓 Training"):
                    self._load_tab_with_error_handling(
                        self._load_training_tab, "Training"
                    )
                
                with gr.Tab("📜 Credits"):
                    self._load_tab_with_error_handling(
                        self._load_credits_tab, "Credits"
                    )
                
                with gr.Tab("⚙️ Settings"):
                    self._load_settings_tab()
            
            self._app = app
            self._logger.info("Gradio application created successfully")
            return app
            
        except Exception as e:
            self._logger.error(f"Failed to create application: {e}")
            raise
    
    def _load_inference_tab(self):
        """Load the inference tab."""
        try:
            from advanced_rvc_inference.tabs.inference.full_inference import full_inference_tab
            full_inference_tab()
        except Exception as e:
            self._logger.error(f"Could not load inference tab: {e}")
            raise
    
    def _load_download_tab(self):
        """Load the download tab."""
        try:
            from advanced_rvc_inference.tabs.utilities.download_model import download_model_tab
            download_model_tab()
        except Exception as e:
            self._logger.error(f"Could not load download tab: {e}")
            raise
    
    def _load_datasets_tab(self):
        """Load the datasets tab."""
        try:
            from advanced_rvc_inference.tabs.datasets.datasets_tab import datasets_tab
            datasets_tab()
        except Exception as e:
            self._logger.error(f"Could not load datasets tab: {e}")
            raise
    
    def _load_tts_tab(self):
        """Load the TTS tab."""
        try:
            from advanced_rvc_inference.tabs.inference.tts import tts_tab
            tts_tab()
        except Exception as e:
            self._logger.error(f"Could not load TTS tab: {e}")
            raise
    
    def _load_extra_tab(self):
        """Load the extra tools tab."""
        try:
            from advanced_rvc_inference.tabs.extra.extra_tab import extra_tools_tab
            extra_tools_tab()
        except Exception as e:
            self._logger.error(f"Could not load extra tab: {e}")
            raise
    
    def _load_training_tab(self):
        """Load the training tab."""
        try:
            from advanced_rvc_inference.tabs.training.training_tab import training_tab
            training_tab.create_training_interface()
        except Exception as e:
            self._logger.error(f"Could not load training tab: {e}")
            raise
    
    def _load_credits_tab(self):
        """Load the credits tab."""
        try:
            from advanced_rvc_inference.tabs.credits.credits_tab import credits_tab
            credits_tab()
        except Exception as e:
            self._logger.error(f"Could not load credits tab: {e}")
            raise
    
    def _load_settings_tab(self):
        """Load the settings tab."""
        try:
            from advanced_rvc_inference.tabs.settings.settings import (
                lang_tab, audio_tab, performance_tab, notifications_tab,
                file_management_tab, debug_tab, backup_restore_tab,
                misc_tab, restart_tab
            )
            
            with gr.Tab("🌍 Language"):
                lang_tab()
            
            with gr.Tab("🎵 Audio"):
                audio_tab()
                
            with gr.Tab("⚡ Performance"):
                performance_tab()
                
            with gr.Tab("🔔 Notifications"):
                notifications_tab()
                
            with gr.Tab("💾 File Management"):
                file_management_tab()
                
            with gr.Tab("🐛 Debug"):
                debug_tab()
                
            with gr.Tab("🔄 Backup & Restore"):
                backup_restore_tab()
                
            with gr.Tab("🛠️ Miscellaneous"):
                misc_tab()
                
            restart_tab()
            
        except Exception as e:
            self._logger.error(f"Could not load settings tab: {e}")
            gr.HTML(f"""
            <div style="color: red; padding: 20px; text-align: center;">
                ❌ Error loading Settings: {str(e)}
            </div>
            """)
    
    def get_app(self):
        """Get the created application."""
        return self._app
    
    def reload_app(self):
        """Reload the application (useful for config changes)."""
        self._logger.info("Reloading application...")
        self._app = None
        return self.create_app()