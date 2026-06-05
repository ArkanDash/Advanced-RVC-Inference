import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from arvc.services.process import zip_file

from arvc.services.training import (
    extract, 
    training, 
    preprocess, 
    create_index 
)

from arvc.utils.variables import (
    config, 
    method_f0, 
    file_types, 
    model_name, 
    index_path, 
    pretrainedG, 
    pretrainedD, 
    translations, 
    reference_list, 
    embedders_mode, 
    embedders_model, 
    hybrid_f0_method 
)

from arvc.ui.feedback import (
    visible, 
    unlock_f0, 
    gr_warning, 
    shutil_move, 
    get_gpu_info, 
    vocoders_lock, 
    unlock_ver, 
    unlock_vocoder, 
    hoplength_show, 
    gpu_number_str, 
    pitch_guidance_lock, 
    change_models_choices, 
    change_embedders_mode, 
    change_reference_choices,
    change_pretrained_choices 
)

def training_model_tab():
    with gr.Row():
        gr.Markdown(translations.get("training_markdown", "## Training"))
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    modelname = gr.Textbox(
                        label=translations["modelname"], 
                        info=translations["training_model_name"], 
                        value="", 
                        placeholder=translations["modelname"], 
                        interactive=True
                    )
                    sample_rate = gr.Radio(
                        label=translations["sample_rate"], 
                        info=translations["sample_rate_info"], 
                        choices=[
                            "24k",
                            "32k", 
                            "40k", 
                            "44.1k",
                            "48k"
                        ], 
                        value="48k", 
                        interactive=True
                    ) 
                    rvc_version = gr.Radio(
                        label=translations["training_version"], 
                        info=translations["training_version_info"], 
                        choices=[
                            "v1", 
                            "v2"
                        ], 
                        value="v2", 
                        interactive=True
                    ) 
                    with gr.Group():
                        with gr.Row():
                            clean_dataset = gr.Checkbox(
                                label=translations["clear_dataset"], 
                                value=False, 
                                interactive=True
                            )
                            process_effects = gr.Checkbox(
                                label=translations["preprocess_effect"], 
                                value=False, 
                                interactive=True
                            )
                            pitch_guidance = gr.Checkbox(
                                label=translations["training_pitch"], 
                                value=True, 
                                interactive=True
                            )
                        with gr.Row():
                            custom_reference = gr.Checkbox(
                                label=translations["custom_reference"], 
                                value=False, 
                                interactive=True
                            )
                            checkpointing = gr.Checkbox(
                                label=translations["memory_efficient_training"], 
                                value=False, 
                                interactive=True
                            )
                            dataset_upload = gr.Checkbox(
                                label=translations["upload_dataset"], 
                                value=False, 
                                interactive=True
                            )
                    with gr.Row():
                        split_audio_mode = gr.Radio(
                            label=translations["split_audio_mode"], 
                            info=translations["split_audio_mode_info"], 
                            value="Automatic", 
                            choices=[
                                "Automatic", 
                                "Simple", 
                                "Skip"
                            ], 
                            interactive=True
                        )
                        normalization_mode = gr.Radio(
                            label=translations["normalization_mode"], 
                            info=translations["normalization_mode_info"], 
                            value="post", 
                            choices=[
                                "none", 
                                "pre", 
                                "post"
                            ], 
                            interactive=True
                        )
                    with gr.Row(visible=False) as custom_reference_row:
                        with gr.Accordion(
                            translations["custom_reference"], 
                            open=True
                        ):
                            reference_name = gr.Dropdown(
                                label=translations["reference_name"], 
                                info=translations["reference_name_info"], 
                                choices=reference_list, 
                                value=reference_list[0] if len(reference_list) >= 1 else "", 
                                allow_custom_value=True, 
                                interactive=True
                            )
                            reference_refresh = gr.Button(
                                translations["refresh"], 
                                scale=2
                            )
                    with gr.Row(visible=False) as clean_dataset_row:
                        clean_dataset_strength = gr.Slider(
                            label=translations["clean_strength"], 
                            info=translations["clean_strength_info"], 
                            minimum=0, 
                            maximum=1, 
                            value=0.7, 
                            step=0.1, 
                            interactive=True
                        )
                with gr.Column():
                    preprocess_button = gr.Button(
                        translations["preprocess_button"], 
                        scale=2
                    )
                    upload_dataset = gr.Files(
                        label=translations["drop_audio"], 
                        file_types=file_types, 
                        visible=False
                    )
                    preprocess_info = gr.Textbox(
                        label=translations["preprocess_info"], 
                        value="", 
                        interactive=False, 
                        container=True, 
                        lines=2
                    )
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    with gr.Accordion(
                        label=translations["f0_method"], 
                        open=False
                    ):
                        with gr.Group():
                            with gr.Row():
                                predictor_onnx = gr.Checkbox(
                                    label=translations.get("predictor_onnx", translations.get("f0_onnx_mode", "F0 ONNX Mode")), 
                                    value=False, 
                                    interactive=True
                                )
                                unlock_full_method = gr.Checkbox(
                                    label=translations["f0_unlock"], 
                                    value=False, 
                                    interactive=True
                                )
                                autotune = gr.Checkbox(
                                    label=translations["autotune"], 
                                    value=False, 
                                    interactive=True
                                )
                            f0_method = gr.Radio(
                                label=translations["f0_method"], 
                                info=translations["f0_method_info"], 
                                choices=method_f0, 
                                value="rmvpe", 
                                interactive=True
                            )
                            hybrid_f0method = gr.Dropdown(
                                label=translations["f0_method_hybrid"], 
                                info=translations["f0_method_hybrid_info"], 
                                choices=hybrid_f0_method, 
                                value=hybrid_f0_method[0], 
                                interactive=True, 
                                allow_custom_value=True, 
                                visible=False
                            )
                        hop_length = gr.Slider(
                            label=translations['hop_length'], 
                            info=translations["hop_length_info"], 
                            minimum=64, 
                            maximum=512, 
                            value=160, 
                            step=1, 
                            interactive=True, 
                            visible=False
                        )
                        f0_autotune_strength = gr.Slider(
                            label=translations["autotune_rate"], 
                            info=translations["autotune_rate_info"], 
                            minimum=0, 
                            maximum=1, 
                            value=1, 
                            step=0.1, 
                            interactive=True, 
                            visible=False
                        )
                        alpha = gr.Slider(
                            label=translations["alpha_label"], 
                            info=translations["alpha_info"], 
                            minimum=0.1, 
                            maximum=1, 
                            value=0.5, 
                            step=0.1, 
                            interactive=True, 
                            visible=False
                        )
                    with gr.Accordion(
                        label=translations["hubert_model"], 
                        open=False
                    ):
                        embedders_mix = gr.Checkbox(
                            label=translations.get("embedders_mix", "Embedder Mix"),
                            info=translations.get("embedders_mix_info", "Blend two transformer layer features"),
                            value=False,
                            interactive=True
                        )
                        with gr.Group():
                            embedder_mode = gr.Radio(
                                label=translations["embed_mode"], 
                                info=translations["embed_mode_info"], 
                                value="fairseq", 
                                choices=embedders_mode, 
                                interactive=True, 
                                visible=True
                            )
                            embedders = gr.Radio(
                                label=translations["hubert_model"], 
                                info=translations["hubert_info"], 
                                choices=embedders_model, 
                                value="hubert_base", 
                                interactive=True
                            )
                        with gr.Row():
                            embedders_custom = gr.Textbox(
                                label=translations["modelname"], 
                                info=translations["modelname_info"], 
                                value="", 
                                placeholder="hubert_base", 
                                interactive=True, 
                                visible=False
                            )
                        with gr.Column(visible=False) as embedders_mix_column:
                            embedders_mix_layers = gr.Slider(
                                label=translations.get("embedders_mix_layers", "Mix Layers"), 
                                info=translations.get("embedders_mix_layers_info", "Number of transformer layers to blend"),
                                minimum=1, 
                                maximum=12, 
                                value=9, 
                                step=1, 
                                interactive=True
                            )
                            embedders_mix_ratio = gr.Slider(
                                label=translations.get("embedders_mix_ratio", "Mix Ratio"), 
                                info=translations.get("embedders_mix_ratio_info", "Blending ratio"), 
                                minimum=0.1, 
                                maximum=1, 
                                value=0.5, 
                                step=0.1, 
                                interactive=True
                            )
                with gr.Column():
                    extract_button = gr.Button(
                        translations["extract_button"], 
                        scale=2
                    )
                    extract_info = gr.Textbox(
                        label=translations["extract_info"], 
                        value="", 
                        interactive=False, 
                        lines=2
                    )
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    total_epochs = gr.Slider(
                        label=translations["total_epoch"], 
                        info=translations["total_epoch_info"], 
                        minimum=1, 
                        maximum=10000, 
                        value=300, 
                        step=1, 
                        interactive=True
                    )
                    save_epochs = gr.Slider(
                        label=translations["save_epoch"], 
                        info=translations["save_epoch_info"], 
                        minimum=1, 
                        maximum=10000, 
                        value=50, 
                        step=1, 
                        interactive=True
                    )
                with gr.Column():
                    create_index_button = gr.Button(
                        f"3. {translations['create_index']}", 
                        variant="primary", 
                        scale=2
                    )
                    training_button = gr.Button(
                        f"4. {translations['training_model']}", 
                        variant="primary", 
                        scale=2
                    )
            with gr.Row():
                with gr.Accordion(
                    label=translations.get("setting", "Settings"), 
                    open=False
                ):
                    with gr.Row():
                        index_algorithm = gr.Radio(
                            label=translations["index_algorithm"], 
                            info=translations["index_algorithm_info"], 
                            choices=[
                                "Auto", 
                                "Faiss", 
                                "KMeans"
                            ], 
                            value="Auto", 
                            interactive=True
                        )
                    with gr.Row():
                        nprobe = gr.Slider(
                            label=translations.get("nprobe", "Nprobe"), 
                            info=translations.get("nprobe_info", "Number of probes for FAISS index search"), 
                            minimum=1, 
                            maximum=64, 
                            value=9, 
                            step=1, 
                            interactive=True
                        )
                    with gr.Row():
                        cache_in_gpu = gr.Checkbox(
                            label=translations["cache_in_gpu"], 
                            info=translations["cache_in_gpu_info"], 
                            value=True, 
                            interactive=True
                        )
                        energy = gr.Checkbox(
                            label=translations["train&energy"], 
                            info=translations["train&energy_info"], 
                            value=False, 
                            interactive=True
                        )
                        overtraining_detector = gr.Checkbox(
                            label=translations["overtraining_detector"], 
                            info=translations["overtraining_detector_info"], 
                            value=False, 
                            interactive=True
                        )
                    with gr.Row():
                        custom_dataset = gr.Checkbox(
                            label=translations["custom_dataset"], 
                            info=translations["custom_dataset_info"], 
                            value=False, 
                            interactive=True
                        )
                        save_only_latest = gr.Checkbox(
                            label=translations["save_only_latest"], 
                            info=translations["save_only_latest_info"], 
                            value=True, 
                            interactive=True
                        )
                        save_every_weights = gr.Checkbox(
                            label=translations["save_every_weights"], 
                            info=translations["save_every_weights_info"], 
                            value=True, 
                            interactive=True
                        )
                    with gr.Row():
                        cleanup_training = gr.Checkbox(
                            label=translations["cleanup_training"], 
                            info=translations["cleanup_training_info"], 
                            value=False, 
                            interactive=True
                        )
                        not_use_pretrain = gr.Checkbox(
                            label=translations["not_use_pretrain_2"], 
                            info=translations["not_use_pretrain_info"], 
                            value=False, 
                            interactive=True
                        )
                        custom_pretrain = gr.Checkbox(
                            label=translations["custom_pretrain"], 
                            info=translations["custom_pretrain_info"], 
                            value=False, 
                            interactive=True
                        )
                    with gr.Column():
                        dataset_path = gr.Textbox(
                            label=translations["dataset_folder"], 
                            value="arvc/assets/dataset", 
                            interactive=True, 
                            visible=False
                        )
                    with gr.Column():
                        include_mutes = gr.Slider(
                            minimum=0, 
                            maximum=10, 
                            value=2, 
                            step=1, 
                            label=translations.get("include_mutes", "Include Mutes"), 
                            info=translations.get("include_mutes_info", "Number of mute entries per speaker"), 
                            interactive=True
                        )
                        with gr.Row(visible=False) as simple_option:
                            chunk_len = gr.Slider(
                                minimum=0.5, 
                                maximum=5.0, 
                                value=3.0, 
                                step=0.1, 
                                label=translations["chunk_length"], 
                                info=translations["chunk_length_info"], 
                                interactive=True
                            )
                            overlap_len = gr.Slider(
                                minimum=0.0, 
                                maximum=0.4, 
                                value=0.3, 
                                step=0.1, 
                                label=translations["overlap_length"], 
                                info=translations["overlap_length_info"], 
                                interactive=True
                            )
                        threshold = gr.Slider(
                            minimum=1, 
                            maximum=100, 
                            value=50, 
                            step=1, 
                            label=translations["threshold"], 
                            info=translations.get("overtraining_threshold", translations["threshold"]),
                            interactive=True, 
                            visible=False
                        )
                        with gr.Accordion(
                            translations["setting_cpu_gpu"], 
                            open=False
                        ):
                            with gr.Row():
                                architecture = gr.Radio(
                                    label=translations.get("architecture", "Architecture"), 
                                    info=translations.get("architecture_info", "Model architecture: RVC or SVC"), 
                                    choices=[
                                        "RVC", 
                                        "SVC"
                                    ], 
                                    value="RVC", 
                                    interactive=True
                                )
                            with gr.Column():
                                gpu_number = gr.Textbox(
                                    label=translations["gpu_number"], 
                                    value=gpu_number_str(), 
                                    info=translations["gpu_number_info"], 
                                    interactive=True
                                )
                                gpu_str = get_gpu_info()
                                gpu_len = max(1, len(gpu_str.split('\n'))) if isinstance(gpu_str, str) else 1
                                gr.Textbox(
                                    label=translations["gpu_info"], 
                                    value=gpu_str, 
                                    info=translations["gpu_info_2"], 
                                    interactive=False,
                                    lines=gpu_len
                                )
                                cpu_core = gr.Slider(
                                    label=translations["cpu_core"], 
                                    info=translations["cpu_core_info"], 
                                    minimum=1, 
                                    maximum=os.cpu_count(), 
                                    value=min(os.cpu_count(), 4), 
                                    step=1, 
                                    interactive=True
                                )          
                                batch_size = gr.Slider(
                                    label=translations["batch_size"], 
                                    info=translations["batch_size_info"], 
                                    minimum=1, 
                                    maximum=64, 
                                    value=8, 
                                    step=1, 
                                    interactive=True
                                )
                    with gr.Group():
                        multiscale_mel_loss = gr.Checkbox(
                            label=translations["multiscale_mel_loss"], 
                            info=translations["multiscale_mel_loss_info"], 
                            value=False, 
                            interactive=True
                        )
                        cosine_annealing_lr = gr.Checkbox(
                            label=translations.get("cosine_annealing_lr", "Cosine Annealing LR"), 
                            info=translations.get("cosine_annealing_lr_info", "Use CosineAnnealingLR scheduler"), 
                            value=False, 
                            interactive=True
                        )
                        vocoders = gr.Radio(
                            label=translations["vocoder"], 
                            info=translations["vocoder_info"], 
                            choices=[
                                "Default", 
                                "MRF-HiFi-GAN", 
                                "RefineGAN",
                                "BigVGAN"
                            ], 
                            value="Default", 
                            interactive=True
                        ) 
                    with gr.Row():
                        deterministic = gr.Checkbox(
                            label=translations["deterministic"], 
                            info=translations["deterministic_info"], 
                            value=False, 
                            interactive=config.device.startswith("cuda") and not config.is_zluda
                        )
                        benchmark = gr.Checkbox(
                            label=translations["benchmark"], 
                            info=translations["benchmark_info"], 
                            value=False, 
                            interactive=config.device.startswith("cuda") and not config.is_zluda
                        )
                    with gr.Row():
                        compile_model = gr.Checkbox(
                            label=translations.get("compile_model", "torch.compile()"),
                            info=translations.get("compile_model_info", "Use torch.compile() on generator for PyTorch 2.x speedup"),
                            value=False,
                            interactive=config.device.startswith("cuda")
                        )
                        use_8bit_adam = gr.Checkbox(
                            label=translations.get("use_8bit_adam", "8-bit Adam"),
                            info=translations.get("use_8bit_adam_info", "Use 8-bit Adam optimizer for lower VRAM (requires bitsandbytes)"),
                            value=False,
                            interactive=True
                        )
                        newpytorch = gr.Checkbox(
                            label=translations.get("newpytorch", "PyTorch 2.0+ Format"),
                            info=translations.get("newpytorch_info", "Use PyTorch 2.0+ parametrization format (default). Disable for legacy weight_norm."),
                            value=True,
                            interactive=True
                        )
                    with gr.Row():
                        grad_accum_steps = gr.Slider(
                            label=translations.get("grad_accum_steps", "Gradient Accumulation"),
                            info=translations.get("grad_accum_steps_info", "Gradient accumulation steps (reduces VRAM with larger effective batch)"),
                            minimum=1,
                            maximum=16,
                            value=1,
                            step=1,
                            interactive=True
                        )
                    with gr.Row():
                        optimizer = gr.Radio(
                            label=translations["optimizer"], 
                            info=translations.get("optimizer_info", "Optimizer in training"), 
                            value="AdamW", 
                            choices=[
                                "AdamW", 
                                "RAdam", 
                                "AnyPrecisionAdamW",
                                "AdaBelief",
                                "AdaBeliefV2"
                            ], 
                            interactive=True
                        )
                    with gr.Row():
                        model_author = gr.Textbox(
                            label=translations["training_author"], 
                            info=translations["training_author_info"], 
                            value="", 
                            placeholder=translations["training_author"], 
                            interactive=True
                        )
                    with gr.Row():
                        with gr.Column():
                            with gr.Accordion(
                                translations.get("custom_pretrain_info", "Custom Pretrained"), 
                                open=False, 
                                visible=False
                            ) as pretrain_setting:
                                pretrained_D = gr.Dropdown(
                                    label=translations["pretrain_file"].format(dg="D"), 
                                    choices=pretrainedD, 
                                    value=pretrainedD[0] if len(pretrainedD) > 0 else '', 
                                    interactive=True, 
                                    allow_custom_value=True
                                )
                                pretrained_G = gr.Dropdown(
                                    label=translations["pretrain_file"].format(dg="G"), 
                                    choices=pretrainedG, 
                                    value=pretrainedG[0] if len(pretrainedG) > 0 else '', 
                                    interactive=True, 
                                    allow_custom_value=True
                                )
                                pretrained_refresh = gr.Button(
                                    translations["refresh"], 
                                    scale=2
                                )
            with gr.Row():
                training_info = gr.Textbox(
                    label=translations["train_info"], 
                    value="", 
                    interactive=False, 
                    lines=3
                )
            with gr.Row():
                with gr.Column():
                    with gr.Accordion(
                        translations.get("export_model", "Export Model"), 
                        open=False
                    ):
                        with gr.Row():
                            model_file = gr.Dropdown(
                                label=translations.get("model_name", "Model"), 
                                choices=model_name, 
                                value=model_name[0] if len(model_name) >= 1 else "", 
                                interactive=True, 
                                allow_custom_value=True
                            )
                            index_file = gr.Dropdown(
                                label=translations.get("index_path", "Index"), 
                                choices=index_path, 
                                value=index_path[0] if len(index_path) >= 1 else "", 
                                interactive=True, 
                                allow_custom_value=True
                            )
                        with gr.Row():
                            refresh_file = gr.Button(
                                f"1. {translations['refresh']}", 
                                scale=2
                            )
                            zip_model = gr.Button(
                                translations.get("zip_model", "Zip Model"), 
                                variant="primary", 
                                scale=2
                            )
                        with gr.Row():
                            zip_output = gr.File(
                                label=translations.get("output_zip", "Output ZIP"), 
                                file_types=[".zip"], 
                                interactive=False, 
                                visible=False
                            )
    with gr.Row():
        vocoders.change(
            fn=pitch_guidance_lock, 
            inputs=[
                vocoders
            ], 
            outputs=[
                pitch_guidance
            ]
        )
        pitch_guidance.change(
            fn=vocoders_lock, 
            inputs=[
                pitch_guidance
            ], 
            outputs=[
                vocoders
            ]
        )
        unlock_full_method.change(
            fn=unlock_f0, 
            inputs=[
                unlock_full_method
            ], 
            outputs=[
                f0_method
            ]
        )
    with gr.Row():
        refresh_file.click(
            fn=change_models_choices, 
            inputs=[], 
            outputs=[
                model_file, 
                index_file
            ]
        ) 
        zip_model.click(
            fn=zip_file, 
            inputs=[
                modelname, 
                model_file, 
                index_file
            ], 
            outputs=[
                zip_output
            ]
        )                
        dataset_path.change(
            fn=lambda folder: os.makedirs(folder, exist_ok=True) if folder else None, 
            inputs=[
                dataset_path
            ], 
            outputs=[]
        )
    with gr.Row():
        dataset_upload.change(
            fn=visible, 
            inputs=[
                dataset_upload
            ], 
            outputs=[
                upload_dataset
            ]
        ) 
        overtraining_detector.change(
            fn=visible, 
            inputs=[
                overtraining_detector
            ], 
            outputs=[
                threshold
            ]
        )
        clean_dataset.change(
            fn=visible, 
            inputs=[
                clean_dataset
            ], 
            outputs=[
                clean_dataset_row
            ]
        )
    with gr.Row():
        custom_dataset.change(
            fn=lambda custom_dataset: [
                visible(custom_dataset), 
                "dataset"
            ], 
            inputs=[
                custom_dataset
            ], 
            outputs=[
                dataset_path, 
                dataset_path
            ]
        )
        rvc_version.change(
            fn=unlock_vocoder, 
            inputs=[
                rvc_version, 
                vocoders
            ], 
            outputs=[
                vocoders
            ]
        )
        vocoders.change(
            inputs=[
                rvc_version, 
                vocoders
            ], 
            fn=unlock_ver, 
            outputs=[
                rvc_version
            ]
        )
    with gr.Row():
        custom_reference.change(
            fn=visible, 
            inputs=[
                custom_reference
            ], 
            outputs=[
                custom_reference_row
            ]
        )
        f0_method.change(
            fn=lambda method, hybrid: [
                visible(method == "hybrid"), 
                visible(method == "hybrid"), 
                hoplength_show(method, hybrid)
            ],
            inputs=[
                f0_method, 
                hybrid_f0method
            ], 
            outputs=[
                hybrid_f0method, 
                alpha, 
                hop_length
            ]
        )
        hybrid_f0method.change(
            fn=hoplength_show, 
            inputs=[
                f0_method, 
                hybrid_f0method
            ], 
            outputs=[
                hop_length
            ]
        )
    with gr.Row():
        autotune.change(
            fn=visible, 
            inputs=[
                autotune
            ], 
            outputs=[
                f0_autotune_strength
            ]
        )
        split_audio_mode.change(
            fn=lambda a: visible(a == "Simple"), 
            inputs=[
                split_audio_mode
            ], 
            outputs=[
                simple_option
            ]
        )
        upload_dataset.upload(
            fn=lambda files, folder: [
                shutil_move(f.name, os.path.join(folder, os.path.split(f.name)[1])) 
                for f in files
            ] if folder != "" else gr_warning(translations["dataset_folder1"]),
            inputs=[
                upload_dataset, 
                dataset_path
            ], 
            outputs=[], 
            api_name="upload_dataset"
        )           
    with gr.Row():
        not_use_pretrain.change(
            fn=lambda a, b: visible(a and not b), 
            inputs=[
                custom_pretrain, 
                not_use_pretrain
            ], 
            outputs=[
                pretrain_setting
            ]
        )
        custom_pretrain.change(
            fn=lambda a, b: visible(a and not b), 
            inputs=[
                custom_pretrain, 
                not_use_pretrain
            ], 
            outputs=[
                pretrain_setting
            ]
        )
        pretrained_refresh.click(
            fn=change_pretrained_choices, 
            inputs=[], 
            outputs=[
                pretrained_D, 
                pretrained_G
            ]
        )
    with gr.Row():
        embedder_mode.change(
            fn=change_embedders_mode, 
            inputs=[
                embedder_mode
            ], 
            outputs=[
                embedders
            ]
        )
        embedders.change(
            fn=lambda embedders: visible(embedders == "custom"), 
            inputs=[
                embedders
            ], 
            outputs=[
                embedders_custom
            ]
        )
        reference_refresh.click(
            fn=change_reference_choices, 
            inputs=[], 
            outputs=[
                reference_name
            ]
        )
    with gr.Row():
        embedders_mix.change(
            fn=visible,
            inputs=[
                embedders_mix
            ],
            outputs=[
                embedders_mix_column
            ]
        )
    with gr.Row():
        preprocess_button.click(
            fn=preprocess,
            inputs=[
                modelname, 
                sample_rate, 
                cpu_core,
                split_audio_mode, 
                process_effects,
                dataset_path,
                clean_dataset,
                clean_dataset_strength,
                chunk_len, 
                overlap_len,
                normalization_mode,
                architecture
            ],
            outputs=[
                preprocess_info
            ],
            api_name="preprocess"
        )
    with gr.Row():
        extract_button.click(
            fn=extract,
            inputs=[
                modelname, 
                rvc_version, 
                f0_method, 
                pitch_guidance, 
                hop_length, 
                cpu_core,
                gpu_number,
                sample_rate, 
                embedders, 
                embedders_custom,
                predictor_onnx,
                embedder_mode,
                autotune,
                f0_autotune_strength,
                hybrid_f0method,
                energy,
                alpha,
                include_mutes,
                embedders_mix,
                embedders_mix_layers,
                embedders_mix_ratio,
                architecture
            ],
            outputs=[
                extract_info
            ],
            api_name="extract"
        )
    with gr.Row():
        create_index_button.click(
            fn=create_index,
            inputs=[
                modelname, 
                rvc_version, 
                index_algorithm,
                nprobe
            ],
            outputs=[
                training_info
            ],
            api_name="create_index"
        )
    with gr.Row():
        training_button.click(
            fn=training,
            inputs=[
                modelname, 
                rvc_version, 
                save_epochs, 
                save_only_latest, 
                save_every_weights, 
                total_epochs, 
                sample_rate,
                batch_size, 
                gpu_number,
                pitch_guidance,
                not_use_pretrain,
                custom_pretrain,
                pretrained_G,
                pretrained_D,
                overtraining_detector,
                threshold,
                cleanup_training,
                cache_in_gpu,
                model_author,
                vocoders,
                checkpointing,
                deterministic, 
                benchmark,
                optimizer,
                energy,
                custom_reference,
                reference_name,
                multiscale_mel_loss,
                embedders, 
                embedders_custom,
                cosine_annealing_lr,
                architecture,
                compile_model,
                use_8bit_adam,
                grad_accum_steps,
                newpytorch
            ],
            outputs=[
                training_info
            ],
            api_name="training_model"
        )
