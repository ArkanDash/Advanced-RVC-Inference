"""
Credits and Acknowledgments Tab
Enhanced Advanced RVC Inference Application V3.3

This tab displays all credits, acknowledgments, and references for the
Advanced RVC Inference application and its various integrated features.
"""

import gradio as gr
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

def create_credits_tab():
    """Create the credits and acknowledgments tab"""
    
    with gr.Row():
        with gr.Column():
            gr.HTML("""
            <div style="text-align: center; padding: 20px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #f9f9f9;">
                <h1 style="color: #2E7D32; margin-bottom: 10px;">Enhanced Advanced RVC Inference V3.3</h1>
                <h2 style="color: #388E3C; margin-bottom: 20px;">🎤 Voice Conversion with KADVC Integration 🚀</h2>
                <p style="font-size: 16px; color: #1976D2; font-weight: bold;">
                    2x Faster Training & Inference | Optimized for Google Colab | Enhanced Features
                </p>
            </div>
            """)

    with gr.Tabs():
        with gr.Tab(i18n("About Application")):
            with gr.Column():
                gr.HTML("""
                <div style="background-color: #E3F2FD; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h3 style="color: #1565C0;">🚀 Enhanced Advanced RVC Inference V3.3</h3>
                    <p style="font-size: 14px; line-height: 1.6;">
                        This is an enhanced version of the Advanced RVC Inference application, integrated with 
                        <strong>KADVC (Kernel Advanced Voice Conversion)</strong> for superior performance and 
                        additional features inspired by the Vietnamese-RVC project.
                    </p>
                    <p style="font-size: 14px; line-height: 1.6;">
                        <strong>Key Features:</strong><br>
                        • 2x faster training and inference with custom CUDA kernels<br>
                        • Enhanced memory management for Google Colab<br>
                        • Multiple F0 extraction methods (RMVPE, Librosa, CREPE)<br>
                        • Advanced voice conversion with formant shifting<br>
                        • Real-time inference capabilities<br>
                        • Comprehensive training features
                    </p>
                </div>
                """)

        with gr.Tab(i18n("Original Development")):
            with gr.Column():
                gr.HTML("""
                <div style="background-color: #FFF3E0; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h3 style="color: #E65100;">👨‍💻 Original Advanced RVC Inference</h3>
                    <p style="font-size: 14px; line-height: 1.6;">
                        The original Advanced RVC Inference application was developed to provide an enhanced 
                        user interface and additional features for RVC (Retrieval-based Voice Conversion) inference.
                    </p>
                    <p style="font-size: 14px; line-height: 1.6;">
                        <strong>Core Features from Original:</strong><br>
                        • Enhanced Gradio user interface<br>
                        • Model management and loading<br>
                        • Audio file selection and processing<br>
                        • Batch processing capabilities<br>
                        • Multiple export formats<br>
                        • Comprehensive settings panel
                    </p>
                </div>
                """)

        with gr.Tab(i18n("Vietnamese-RVC Integration")):
            with gr.Column():
                gr.HTML("""
                <div style="background-color: #E8F5E8; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h3 style="color: #2E7D32;">🇻🇳 Vietnamese-RVC Project Inspiration</h3>
                    <p style="font-size: 14px; line-height: 1.6;">
                        This application incorporates architectural improvements and optimizations inspired by the 
                        Vietnamese-RVC project, specifically designed for Vietnamese voice processing and enhanced performance.
                    </p>
                    <p style="font-size: 14px; line-height: 1.6;">
                        <strong>Integrated Vietnamese-RVC Features:</strong><br>
                        • Enhanced SSL handling and security<br>
                        • Optimized inference pipeline<br>
                        • Improved memory management<br>
                        • Enhanced audio processing algorithms<br>
                        • Better error handling and logging<br>
                        • Advanced configuration system
                    </p>
                    <p style="font-size: 14px; line-height: 1.6;">
                        <strong>Vietnamese-RVC Reference:</strong><br>
                        <a href="https://github.com/PhamHuynhAnh16/Vietnamese-RVC" target="_blank" style="color: #1976D2;">
                            https://github.com/PhamHuynhAnh16/Vietnamese-RVC
                        </a>
                    </p>
                </div>
                """)

        with gr.Tab(i18n("KADVC Technology")):
            with gr.Column():
                gr.HTML("""
                <div style="background-color: #F3E5F5; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h3 style="color: #7B1FA2;">⚡ KADVC - Kernel Advanced Voice Conversion</h3>
                    <p style="font-size: 14px; line-height: 1.6;">
                        KADVC is a custom CUDA kernel optimization system designed to provide 
                        <strong>2x faster training and inference</strong> for voice conversion tasks, 
                        specifically optimized for Google Colab environments.
                    </p>
                    <p style="font-size: 14px; line-height: 1.6;">
                        <strong>KADVC Optimizations:</strong><br>
                        • Custom CUDA kernels for F0 extraction<br>
                        • GPU memory optimization<br>
                        • Tensor core utilization<br>
                        • Mixed precision training support<br>
                        • Chunk-based audio processing<br>
                        • Enhanced error handling
                    </p>
                    <p style="font-size: 14px; line-height: 1.6;">
                        <strong>Performance Improvements:</strong><br>
                        • 2x faster F0 extraction<br>
                        • Optimized memory usage<br>
                        • Reduced inference time<br>
                        • Better GPU utilization<br>
                        • Enhanced Colab compatibility
                    </p>
                </div>
                """)

        with gr.Tab(i18n("Technology Stack")):
            with gr.Column():
                gr.HTML("""
                <div style="background-color: #F1F8E9; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h3 style="color: #558B2F;">🔧 Technology Stack & Libraries</h3>
                    <p style="font-size: 14px; line-height: 1.6;">
                        This application uses a comprehensive technology stack for voice processing, 
                        machine learning, and web interface development.
                    </p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.HTML("""
                        <div style="background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #ddd;">
                            <h4 style="color: #1976D2; margin-bottom: 10px;">🧠 Machine Learning & AI</h4>
                            <ul style="font-size: 12px; line-height: 1.4;">
                                <li><strong>PyTorch</strong> - Deep learning framework</li>
                                <li><strong>Torchaudio</strong> - Audio processing</li>
                                <li><strong>Transformers</strong> - Hugging Face models</li>
                                <li><strong>ONNX Runtime</strong> - Model optimization</li>
                                <li><strong>Torchcrepe</strong> - F0 extraction</li>
                            </ul>
                        </div>
                        """)
                    
                    with gr.Column():
                        gr.HTML("""
                        <div style="background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #ddd;">
                            <h4 style="color: #388E3C; margin-bottom: 10px;">🎵 Audio Processing</h4>
                            <ul style="font-size: 12px; line-height: 1.4;">
                                <li><strong>Librosa</strong> - Audio analysis</li>
                                <li><strong>SoundFile</strong> - Audio I/O</li>
                                <li><strong>PyDub</strong> - Audio manipulation</li>
                                <li><strong>Noisereduce</strong> - Noise reduction</li>
                                <li><strong>Pedalboard</strong> - Audio effects</li>
                                <li><strong>Audio-Separator</strong> - Vocal isolation</li>
                            </ul>
                        </div>
                        """)
                
                with gr.Row():
                    with gr.Column():
                        gr.HTML("""
                        <div style="background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #ddd;">
                            <h4 style="color: #F57C00; margin-bottom: 10px;">🌐 Web Interface</h4>
                            <ul style="font-size: 12px; line-height: 1.4;">
                                <li><strong>Gradio</strong> - Web UI framework</li>
                                <li><strong>Rich</strong> - Terminal formatting</li>
                                <li><strong>FastAPI</strong> - Web API (Vietnamese-RVC)</li>
                            </ul>
                        </div>
                        """)
                    
                    with gr.Column():
                        gr.HTML("""
                        <div style="background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #ddd;">
                            <h4 style="color: #7B1FA2; margin-bottom: 10px;">🛠️ Utilities & Tools</h4>
                            <ul style="font-size: 12px; line-height: 1.4;">
                                <li><strong>YT-DLP</strong> - Audio downloading</li>
                                <li><strong>FAISS</strong> - Vector search (optional)</li>
                                <li><strong>OmegaConf</strong> - Configuration</li>
                                <li><strong>Discord.py</strong> - Discord integration</li>
                            </ul>
                        </div>
                        """)

        with gr.Tab(i18n("Performance Optimizations")):
            with gr.Column():
                gr.HTML("""
                <div style="background-color: #FFF8E1; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h3 style="color: #F57F17;">⚡ Performance Optimizations</h3>
                    <p style="font-size: 14px; line-height: 1.6;">
                        This application includes several performance optimizations specifically designed for 
                        Google Colab and local GPU environments.
                    </p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.HTML("""
                        <div style="background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #ddd;">
                            <h4 style="color: #1976D2; margin-bottom: 10px;">🏃 Google Colab Optimizations</h4>
                            <ul style="font-size: 12px; line-height: 1.4;">
                                <li>Memory-efficient algorithms</li>
                                <li>Chunk-based processing</li>
                                <li>Optimized batch sizes</li>
                                <li>Reduced timeout risks</li>
                                <li>Smart cache management</li>
                            </ul>
                        </div>
                        """)
                    
                    with gr.Column():
                        gr.HTML("""
                        <div style="background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #ddd;">
                            <h4 style="color: #388E3C; margin-bottom: 10px;">🚀 CUDA Optimizations</h4>
                            <ul style="font-size: 12px; line-height: 1.4;">
                                <li>Custom kernel implementations</li>
                                <li>Tensor core utilization</li>
                                <li>Memory pool management</li>
                                <li>Parallel processing</li>
                                <li>Mixed precision training</li>
                            </ul>
                        </div>
                        """)

        with gr.Tab(i18n("Special Thanks")):
            with gr.Column():
                gr.HTML("""
                <div style="background-color: #FFF3E0; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h3 style="color: #E65100;">🙏 Special Thanks & Acknowledgments</h3>
                    <p style="font-size: 14px; line-height: 1.6;">
                        This project builds upon the excellent work of many developers and researchers in the voice conversion community.
                    </p>
                </div>
                """)
                
                gr.HTML("""
                <div style="background-color: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
                    <h4 style="color: #1976D2;">🎤 RVC (Retrieval-based Voice Conversion)</h4>
                    <p style="font-size: 13px; line-height: 1.5;">
                        Special thanks to the RVC community for developing the retrieval-based voice conversion 
                        technology that forms the foundation of this application.
                    </p>
                </div>
                """)
                
                gr.HTML("""
                <div style="background-color: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
                    <h4 style="color: #388E3C;">🇻🇳 Vietnamese-RVC Project</h4>
                    <p style="font-size: 13px; line-height: 1.5;">
                        Thanks to <strong>PhamHuynhAnh16</strong> and the Vietnamese-RVC team for their excellent 
                        architectural improvements and optimizations that inspired many features in this enhanced version.
                    </p>
                    <p style="font-size: 13px; line-height: 1.5;">
                        Repository: <a href="https://github.com/PhamHuynhAnh16/Vietnamese-RVC" target="_blank" style="color: #1976D2;">
                            https://github.com/PhamHuynhAnh16/Vietnamese-RVC
                        </a>
                    </p>
                </div>
                """)
                
                gr.HTML("""
                <div style="background-color: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
                    <h4 style="color: #F57C00;">🔧 Library Maintainers</h4>
                    <p style="font-size: 13px; line-height: 1.5;">
                        Thanks to all the library maintainers who make this project possible:
                    </p>
                    <ul style="font-size: 12px; line-height: 1.4;">
                        <li><strong>PyTorch Team</strong> - Deep learning framework</li>
                        <li><strong>Gradio Team</strong> - Web interface framework</li>
                        <li><strong>Librosa Team</strong> - Audio analysis library</li>
                        <li><strong>Hugging Face</strong> - Machine learning models and tools</li>
                        <li><strong>Community Contributors</strong> - Bug reports and improvements</li>
                    </ul>
                </div>
                """)

        with gr.Tab(i18n("Version History")):
            with gr.Column():
                gr.HTML("""
                <div style="background-color: #E1F5FE; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h3 style="color: #0277BD;">📅 Version History</h3>
                </div>
                """)
                
                gr.HTML("""
                <div style="background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
                    <h4 style="color: #2E7D32;">Version 3.3.0 - Enhanced with KADVC (Current)</h4>
                    <p style="font-size: 12px; line-height: 1.4; margin-bottom: 10px;">
                        <strong>Release Date:</strong> November 2025<br>
                        <strong>Major Updates:</strong>
                    </p>
                    <ul style="font-size: 12px; line-height: 1.4;">
                        <li>✅ Integrated KADVC (Kernel Advanced Voice Conversion) for 2x performance improvement</li>
                        <li>✅ Fixed audio selection refresh issues</li>
                        <li>✅ Enhanced CUDA kernel optimizations for Google Colab</li>
                        <li>✅ Added Vietnamese-RVC inspired features and architecture</li>
                        <li>✅ Improved memory management and error handling</li>
                        <li>✅ Added comprehensive credits and documentation tab</li>
                        <li>✅ Fixed KADVC initialization errors</li>
                        <li>✅ Enhanced training and inference features</li>
                    </ul>
                </div>
                """)
                
                gr.HTML("""
                <div style="background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
                    <h4 style="color: #1976D2;">Version 3.2.0 - Enhanced Features</h4>
                    <p style="font-size: 12px; line-height: 1.4;">
                        Enhanced user interface and additional inference features
                    </p>
                </div>
                """)
                
                gr.HTML("""
                <div style="background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
                    <h4 style="color: #7B1FA2;">Version 3.1.0 - Vietnamese-RVC Integration</h4>
                    <p style="font-size: 12px; line-height: 1.4;">
                        Initial integration of Vietnamese-RVC architecture and optimizations
                    </p>
                </div>
                """)

        with gr.Tab(i18n("License & Usage")):
            with gr.Column():
                gr.HTML("""
                <div style="background-color: #F3E5F5; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h3 style="color: #7B1FA2;">📜 License & Usage Information</h3>
                </div>
                """)
                
                gr.HTML("""
                <div style="background-color: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
                    <h4 style="color: #1976D2;">Usage Guidelines</h4>
                    <p style="font-size: 13px; line-height: 1.5;">
                        This application is designed for educational and research purposes. Please ensure you have 
                        proper rights and permissions for any audio content you process with this tool.
                    </p>
                    <p style="font-size: 13px; line-height: 1.5;">
                        <strong>Recommended Use Cases:</strong><br>
                        • Voice conversion research and experimentation<br>
                        • Educational demonstrations of AI voice technology<br>
                        • Creative audio projects with proper licensing<br>
                        • Academic research in speech synthesis and conversion
                    </p>
                </div>
                """)
                
                gr.HTML("""
                <div style="background-color: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
                    <h4 style="color: #388E3C;">Disclaimer</h4>
                    <p style="font-size: 13px; line-height: 1.5;">
                        This application is provided as-is without warranty. Users are responsible for ensuring 
                        compliance with local laws and regulations regarding voice processing and data privacy.
                    </p>
                    <p style="font-size: 13px; line-height: 1.5;">
                        The developers are not responsible for any misuse of this technology.
                    </p>
                </div>
                """)

        with gr.Tab(i18n("Contact & Support")):
            with gr.Column():
                gr.HTML("""
                <div style="background-color: #FFF8E1; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h3 style="color: #F57F17;">💬 Contact & Support</h3>
                </div>
                """)
                
                gr.HTML("""
                <div style="background-color: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
                    <h4 style="color: #1976D2;">📚 Documentation & Resources</h4>
                    <p style="font-size: 13px; line-height: 1.5;">
                        For detailed documentation, tutorials, and additional resources:
                    </p>
                    <ul style="font-size: 12px; line-height: 1.4;">
                        <li>📖 Project README and documentation files</li>
                        <li>🔧 Configuration guides and examples</li>
                        <li>🎥 Video tutorials (coming soon)</li>
                        <li>💡 Community discussions and examples</li>
                    </ul>
                </div>
                """)
                
                gr.HTML("""
                <div style="background-color: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
                    <h4 style="color: #388E3C;">🐛 Bug Reports & Feature Requests</h4>
                    <p style="font-size: 13px; line-height: 1.5;">
                        If you encounter any issues or have suggestions for improvements, please:
                    </p>
                    <ul style="font-size: 12px; line-height: 1.4;">
                        <li>📋 Check the existing issues and documentation</li>
                        <li>🛠️ Provide detailed error messages and steps to reproduce</li>
                        <li>💭 Suggest new features or enhancements</li>
                        <li>🤝 Contribute to the project development</li>
                    </ul>
                </div>
                """)
                
                gr.HTML("""
                <div style="background-color: #E8F5E8; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="color: #2E7D32; text-align: center;">🎉 Thank you for using Enhanced Advanced RVC Inference V3.3!</h4>
                    <p style="text-align: center; font-size: 14px; color: #388E3C;">
                        We hope this tool helps you create amazing voice conversion projects!
                    </p>
                </div>
                """)


def credits_tab():
    """Main function to create and return the credits tab"""
    return create_credits_tab()
