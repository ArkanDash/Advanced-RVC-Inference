#!/usr/bin/env python3
"""
Advanced RVC Inference - Simple API Example

This script demonstrates how to use the Advanced RVC Inference API
for voice conversion with minimal setup.

Author: ArkanDash & BF667
Version: 4.0.0
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main example function"""
    
    print("🎤 Advanced RVC Inference - Simple API Example")
    print("=" * 50)
    
    try:
        # Import the API
        from advanced_rvc_inference import full_inference_program
        print("✅ Successfully imported full_inference_program!")
        
        # Import additional functions
        from advanced_rvc_inference import (
            get_config,
            check_fp16_support,
            models_vocals
        )
        
        # Show system configuration
        config = get_config()
        print(f"\n📊 System Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Check FP16 support
        fp16_support = check_fp16_support()
        print(f"\n🚀 FP16 Support: {'✅ Available' if fp16_support else '❌ Not Available'}")
        
        # Example usage (requires actual files)
        print(f"\n💡 Example Usage:")
        print("""
# Simple voice conversion
result = full_inference_program(
    model_path="path/to/your/model.pth",
    input_audio_path="path/to/input/audio.wav",
    output_path="path/to/output/converted.wav",
    pitch=2,  # Pitch shift in semitones
    f0_method="rmvpe",  # F0 extraction method
    index_rate=0.5,  # Index rate for voice similarity
    clean_audio=True  # Enable noise reduction
)

print(f"Conversion completed: {result}")
        """)
        
        # Show available models
        try:
            vocals = models_vocals()
            print(f"\n🎤 Available Vocal Models:")
            for name in vocals.keys():
                print(f"  - {name}")
        except Exception as e:
            print(f"Could not load model list: {e}")
        
        print(f"\n✅ API is working correctly!")
        print(f"📝 To perform actual conversion, provide valid model and audio file paths.")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"1. Make sure you're in the correct directory")
        print(f"2. Install dependencies: pip install -r requirements.txt")
        print(f"3. Install the package: pip install -e .")
        return 1
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def example_with_files(model_path: str, input_path: str, output_path: str):
    """
    Example function that performs actual voice conversion
    
    Args:
        model_path: Path to RVC model file (.pth)
        input_path: Path to input audio file
        output_path: Path for output audio file
    """
    
    try:
        from advanced_rvc_inference import full_inference_program
        
        print(f"🎵 Starting voice conversion...")
        print(f"📁 Model: {model_path}")
        print(f"📁 Input: {input_path}")
        print(f"📁 Output: {output_path}")
        
        # Perform conversion with default settings
        result = full_inference_program(
            model_path=model_path,
            input_audio_path=input_path,
            output_path=output_path,
            pitch=0,  # No pitch change
            f0_method="rmvpe",  # Best F0 extraction method
            index_rate=0.5,  # Balanced similarity
            rms_mix_rate=1.0,  # Full volume mixing
            protect=0.33,  # Standard voice protection
            clean_audio=True,  # Enable noise reduction
            export_format="wav"  # Output as WAV
        )
        
        if result and os.path.exists(result):
            print(f"✅ Conversion completed successfully!")
            print(f"📁 Output file: {result}")
            return result
        else:
            print(f"❌ Conversion failed - no output file generated")
            return None
            
    except Exception as e:
        print(f"❌ Conversion error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def advanced_example_with_files(model_path: str, input_path: str, output_path: str):
    """
    Advanced example with custom parameters
    
    Args:
        model_path: Path to RVC model file (.pth)
        input_path: Path to input audio file
        output_path: Path for output audio file
    """
    
    try:
        from advanced_rvc_inference import full_inference_program
        
        print(f"🔬 Starting advanced voice conversion...")
        
        # Advanced conversion with custom parameters
        result = full_inference_program(
            model_path=model_path,
            input_audio_path=input_path,
            output_path=output_path,
            
            # Voice conversion parameters
            pitch=2,  # Raise pitch by 2 semitones
            f0_method="rmvpe",  # Best F0 extraction method
            index_rate=0.7,  # Higher similarity to training voice
            rms_mix_rate=0.8,  # Volume mixing
            protect=0.33,  # Voice protection
            
            # Audio processing
            hop_length=64,  # F0 extraction hop length
            filter_radius=3,  # F0 smoothing
            split_audio=True,  # Split long audio for processing
            clean_audio=True,  # Enable noise reduction
            clean_strength=0.7,  # Noise reduction strength
            
            # Output settings
            export_format="wav",  # Output format
            resample_sr=44100,  # Resample to 44.1kHz
            
            # Model settings
            embedder_model="contentvec",  # Embedder model
            
            # Advanced features
            f0_autotune=False,  # Disable autotune
            f0_autotune_strength=1.0  # Autotune strength if enabled
        )
        
        if result and os.path.exists(result):
            print(f"✅ Advanced conversion completed successfully!")
            print(f"📁 Output file: {result}")
            return result
        else:
            print(f"❌ Conversion failed - no output file generated")
            return None
            
    except Exception as e:
        print(f"❌ Conversion error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    exit_code = main()
    
    # Uncomment and modify these lines to test with actual files:
    # model_path = "path/to/your/model.pth"
    # input_path = "path/to/input/audio.wav"
    # output_path = "path/to/output/converted.wav"
    # 
    # if os.path.exists(model_path) and os.path.exists(input_path):
    #     result = example_with_files(model_path, input_path, output_path)
    #     if result:
    #         print(f"🎉 Success! Converted audio saved to: {result}")
    
    sys.exit(exit_code)