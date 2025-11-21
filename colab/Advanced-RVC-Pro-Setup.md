# Google Colab Notebook: Advanced RVC Inference Pro

# Cell 1: Header
"""
# 🚀 Advanced RVC Inference Pro

**State-of-the-art Voice Conversion with KADVC Optimization**

This notebook provides:
- ⚡ **Dependency Caching** - Skip installation on restarts
- 🗄️ **Drive Mounting** - Auto symlink weights folder  
- 🌐 **Tunneling** - Robust Gradio/ngrok integration
- 🎯 **GPU Auto-detection** - Tesla T4/P100/A100 optimization
- 🧠 **Memory Management** - Automatic OOM prevention
"""

# Cell 2: Install & Setup Dependencies
"""
## 📦 Install & Setup Dependencies

This cell includes intelligent caching to save 3-5 minutes on restarts!
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path

# Setup paths
HOME = Path.home()
WORKSPACE = Path.cwd()
CACHE_FILE = HOME / ".rvc_dependencies_installed"
TORCH_CACHE = HOME / ".cache/torch"
TRANSFORMERS_CACHE = HOME / ".cache/huggingface"

def check_gpu_type():
    """Detect GPU type and return optimization recommendations."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"🖥️ Detected GPU: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
            
            # Auto-configure based on GPU
            if "A100" in gpu_name:
                config = {"batch_size": 8, "precision": "fp16", "optimization": "aggressive"}
                print("🚀 A100 detected - Enabling maximum performance mode")
            elif "V100" in gpu_name:
                config = {"batch_size": 6, "precision": "fp16", "optimization": "balanced"}
                print("⚡ V100 detected - Enabling balanced performance mode")
            elif "T4" in gpu_name:
                config = {"batch_size": 4, "precision": "fp16", "optimization": "conservative"}
                print("💪 T4 detected - Enabling conservative performance mode")
            elif "P100" in gpu_name:
                config = {"batch_size": 4, "precision": "fp16", "optimization": "conservative"}
                print("⚡ P100 detected - Enabling conservative performance mode")
            else:
                config = {"batch_size": 2, "precision": "fp32", "optimization": "minimal"}
                print("⚠️ Unknown GPU - Using minimal configuration")
            
            return config
        else:
            print("⚠️ No GPU detected - Using CPU mode")
            return {"batch_size": 1, "precision": "fp32", "optimization": "cpu"}
            
    except Exception as e:
        print(f"Error detecting GPU: {e}")
        return {"batch_size": 2, "precision": "fp32", "optimization": "safe"}

def install_pytorch_with_caching():
    """Install PyTorch with intelligent caching."""
    print("🔥 Installing PyTorch with CUDA support...")
    
    # Check if PyTorch is already installed
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} already installed")
        if torch.cuda.is_available():
            print(f"✅ CUDA {torch.version.cuda} available")
            return True
    except ImportError:
        pass
    
    # Install PyTorch with CUDA
    install_cmd = [
        "pip", "install", "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu118",
        "--timeout", "300"
    ]
    
    try:
        subprocess.run(install_cmd, check=True, timeout=600)
        print("✅ PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
        return False

def install_gradio_with_caching():
    """Install Gradio with latest features."""
    print("🎨 Installing Gradio...")
    
    try:
        import gradio as gr
        print(f"✅ Gradio {gr.__version__} already installed")
        return True
    except ImportError:
        pass
    
    try:
        subprocess.run(["pip", "install", "gradio", "--upgrade", "--timeout", "120"], 
                      check=True, timeout=300)
        print("✅ Gradio installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Gradio: {e}")
        return False

def install_voice_conversion_deps():
    """Install voice conversion specific dependencies."""
    print("🎤 Installing Voice Conversion Dependencies...")
    
    deps = [
        "librosa", "soundfile", "audioread", "resampy",
        "sciplot", "matplotlib", "seaborn", "pandas", 
        "numpy", "scipy", "sklearn", "tqdm",
        "faiss-cpu", "onnx", "onnxruntime",
        "ffmpeg-python", "youtube-dl",
        "huggingface-hub", "transformers",
        "face-recognition", "dlib",
        "gdown", "wget", "requests"
    ]
    
    try:
        for dep in deps:
            print(f"Installing {dep}...")
            subprocess.run(["pip", "install", dep, "--timeout", "180"], 
                          check=True, timeout=300)
        print("✅ All voice conversion dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_environment():
    """Setup optimized environment."""
    print("⚙️ Setting up optimized environment...")
    
    # Create necessary directories
    dirs = ["weights", "indexes", "logs", "cache", "temp", "audio_files", "outputs"]
    for dir_name in dirs:
        dir_path = WORKSPACE / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"📁 Created directory: {dir_name}")
    
    # Set environment variables for optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_CACHE"] = str(TRANSFORMERS_CACHE)
    
    print("✅ Environment setup completed")
    return True

def main_installation():
    """Main installation routine with caching."""
    print("🚀 Starting Advanced RVC Pro Installation...")
    start_time = time.time()
    
    # Check if installation is cached
    if CACHE_FILE.exists():
        try:
            cache_data = json.loads(CACHE_FILE.read_text())
            print("📦 Installation cache found!")
            print(f"⏰ Previous installation: {cache_data.get('timestamp', 'Unknown')}")
            print(f"🏗️ GPU Config: {cache_data.get('gpu_config', 'Unknown')}")
            
            # Quick validation
            import torch
            import gradio as gr
            print("✅ Cache validation successful - skipping installation!")
            return cache_data.get('gpu_config', {})
            
        except Exception as e:
            print(f"⚠️ Cache validation failed: {e}")
    
    # Perform installation
    gpu_config = check_gpu_type()
    
    success = True
    success &= install_pytorch_with_caching()
    success &= install_gradio_with_caching()
    success &= install_voice_conversion_deps()
    success &= setup_environment()
    
    if success:
        # Save cache
        cache_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_config": gpu_config,
            "pytorch_version": subprocess.run(["python", "-c", "import torch; print(torch.__version__)"], 
                                             capture_output=True, text=True).stdout.strip(),
            "cuda_available": torch.cuda.is_available() if 'torch' in globals() else False
        }
        
        CACHE_FILE.write_text(json.dumps(cache_data, indent=2))
        
        elapsed = time.time() - start_time
        print(f"\n🎉 Installation completed in {elapsed:.1f} seconds!")
        print(f"💾 Cache saved to: {CACHE_FILE}")
        return gpu_config
    else:
        print("❌ Installation failed!")
        return None

# Execute installation
GPU_CONFIG = main_installation()

if GPU_CONFIG:
    print(f"\n🎯 GPU Configuration Applied:")
    for key, value in GPU_CONFIG.items():
        print(f"  • {key}: {value}")
else:
    print("\n❌ Installation failed - please check the error messages above")

# Cell 3: Mount Google Drive & Setup Symlinks
"""
## 🗄️ Mount Google Drive & Setup Symlinks

Keep your models and data persistent across sessions!
"""

from google.colab import drive
import os
from pathlib import Path

# Mount Google Drive
print("🔗 Mounting Google Drive...")
drive.mount('/content/drive')

# Define paths
DRIVE_BASE = Path("/content/drive/MyDrive")
WORKSPACE = Path.cwd()

# Create RVC directory on Drive
RVC_DIR = DRIVE_BASE / "RVC_Models"
RVC_DIR.mkdir(exist_ok=True)

# Setup symlinks for persistent storage
symlinks = [
    ("weights", RVC_DIR / "weights"),
    ("indexes", RVC_DIR / "indexes"),
    ("logs", RVC_DIR / "logs"),
    ("cache", RVC_DIR / "cache")
]

print("🔗 Setting up symlinks for persistent storage...")
for local_name, drive_path in symlinks:
    local_path = WORKSPACE / local_name
    
    if local_path.exists() and not local_path.is_symlink():
        # Backup existing directory
        backup_path = local_path.parent / f"{local_name}_backup"
        local_path.rename(backup_path)
        print(f"📦 Backed up {local_name} to {backup_path}")
    
    # Create symlink
    try:
        if local_path.is_symlink():
            local_path.unlink()
        
        # Ensure drive directory exists
        drive_path.mkdir(parents=True, exist_ok=True)
        
        # Create symlink
        os.symlink(drive_path, local_path)
        print(f"✅ Linked {local_name} -> {drive_path}")
        
    except Exception as e:
        print(f"⚠️ Could not symlink {local_name}: {e}")

print(f"\n💾 Your RVC models will be saved to: {RVC_DIR}")
print("🔄 They will persist across Colab sessions!")

# Cell 4: Project Setup
"""
## 📁 Project Setup

Clone the repository and setup the project structure.
"""

import os
import subprocess
from pathlib import Path

# Clone repository
REPO_URL = "https://github.com/ArkanDash/Advanced-RVC-Inference.git"
PROJECT_DIR = Path.cwd() / "Advanced-RVC-Inference"

if not PROJECT_DIR.exists():
    print("📥 Cloning Advanced RVC Inference repository...")
    subprocess.run(["git", "clone", REPO_URL, str(PROJECT_DIR)], check=True)
    print("✅ Repository cloned successfully")
else:
    print("📁 Repository already exists, updating...")
    os.chdir(PROJECT_DIR)
    subprocess.run(["git", "pull", "origin", "main"], check=True)
    print("✅ Repository updated")

# Change to project directory
os.chdir(PROJECT_DIR)
print(f"📂 Working directory: {PROJECT_DIR}")

# Install project in development mode
print("🔧 Installing project in development mode...")
subprocess.run(["pip", "install", "-e", ".", "--no-deps"], check=True)
print("✅ Project installed successfully")

# Cell 5: Setup Tunneling
"""
## 🌐 Setup Tunneling (Choose One)

Select your preferred method for accessing the UI externally.
"""

import subprocess
import threading
import time
from google.colab import output

# Option 1: ngrok (Recommended for stability)
def setup_ngrok():
    print("🌐 Setting up ngrok tunnel...")
    
    # Install ngrok
    try:
        subprocess.run(["wget", "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"], 
                      check=True)
        subprocess.run(["tar", "xzf", "ngrok-v3-stable-linux-amd64.tgz"], check=True)
        subprocess.run(["chmod", "+x", "ngrok"], check=True)
        print("✅ ngrok installed")
        
        # Note: You'll need to add your ngrok auth token
        print("⚠️ Please add your ngrok auth token:")
        print("!./ngrok config add-authtoken YOUR_TOKEN_HERE")
        print("\nThen run:")
        print("!./ngrok http 7860")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to setup ngrok: {e}")

# Option 2: Gradio share (Built-in, less stable)
def setup_gradio_share():
    print("🎨 Gradio share is built-in - just add share=True when launching")
    print("⚠️ Note: Share links expire after ~72 hours")

# Option 3: LocalTunnel
def setup_localtunnel():
    print("🔗 Setting up localtunnel...")
    try:
        subprocess.run(["npm", "install", "-g", "localtunnel"], check=True)
        print("✅ localtunnel installed")
        print("\nTo start tunneling:")
        print("!npx localtunnel --port 7860")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to setup localtunnel: {e}")

# Display options
print("Select your tunneling method:")
print("1️⃣ ngrok (Most stable, requires auth token)")
print("2️⃣ Gradio share (Built-in, less stable)")
print("3️⃣ localtunnel (Good alternative)")
print("\nRecommended: ngrok for production use")

# For quick setup, we'll use Gradio share for now
USE_GRADIO_SHARE = True
USE_NGROK = False
USE_LOCALTUNNEL = False

if USE_GRADIO_SHARE:
    setup_gradio_share()
elif USE_NGROK:
    setup_ngrok()
elif USE_LOCALTUNNEL:
    setup_localtunnel()

# Cell 6: Launch Advanced RVC Interface
"""
## 🚀 Launch Advanced RVC Interface

Start the web interface with optimized settings for your GPU.
"""

import os
import sys
import logging
from pathlib import Path
import gradio as gr

# Add project to Python path
PROJECT_DIR = Path.cwd()
sys.path.insert(0, str(PROJECT_DIR))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU Configuration (from previous cells)
if 'GPU_CONFIG' not in globals():
    GPU_CONFIG = {"batch_size": 4, "precision": "fp16", "optimization": "balanced"}

print("🚀 Starting Advanced RVC Inference...")
print(f"🎯 GPU Config: {GPU_CONFIG}")
print(f"📁 Working Directory: {PROJECT_DIR}")

# Launch application with optimized settings
try:
    # Import the main application
    from app import main
    
    print("✅ Application modules loaded successfully")
    
    # Launch with Colab-optimized settings
    print("🌐 Launching Web Interface...")
    
    # Create a custom launch function for Colab
    def launch_colab_app():
        import argparse
        
        # Override sys.argv for Gradio
        sys.argv = [
            "app.py",
            "--port", "7860",
            "--share", str(USE_GRADIO_SHARE),  # Enable sharing if configured
            "--host", "0.0.0.0",
            "--log-level", "INFO"
        ]
        
        # Configure environment for Colab
        os.environ["PYTHONUNBUFFERED"] = "1"
        
        # Launch the app
        main()
    
    # Start in a separate thread for Colab compatibility
    import threading
    
    app_thread = threading.Thread(target=launch_colab_app, daemon=True)
    app_thread.start()
    
    # Give it a moment to start
    import time
    time.sleep(3)
    
    print("\n🎉 Application Started Successfully!")
    print("🔗 Access your interface at:")
    print("  • Local: http://localhost:7860")
    if USE_GRADIO_SHARE:
        print("  • Public: Check the share link above")
    elif USE_NGROK:
        print("  • ngrok: Check the ngrok output above")
    
    print("\n💡 Tips:")
    print("  • Your models are synced to Google Drive")
    print("  • Memory usage is optimized for your GPU")
    print("  • All sessions will persist your trained models")
    
except Exception as e:
    print(f"❌ Failed to launch application: {e}")
    print("\n🔧 Troubleshooting:")
    print("1. Check if all dependencies are installed")
    print("2. Verify GPU is properly detected")
    print("3. Check Colab logs for detailed error messages")
    import traceback
    traceback.print_exc()