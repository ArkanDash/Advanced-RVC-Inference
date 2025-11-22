import os
import sys

import gradio as gr

sys.path.append(os.getcwd())

from main.app.core.process import zip_file
from main.app.core.training import preprocess, extract, create_index, training
from main.app.variables import translations, model_name, index_path, method_f0, embedders_mode, embedders_model, pretrainedD, pretrainedG, config, file_types, hybrid_f0_method, reference_list
from main.app.core.ui import gr_warning, visible, unlock_f0, hoplength_show, change_models_choices, get_gpu_info, change_embedders_mode, pitch_guidance_lock, vocoders_lock, unlock_ver, unlock_vocoder, change_pretrained_choices, gpu_number_str, shutil_move, change_reference_choices

def training_model_tab():
    with gr.Row():
        gr.Markdown(translations["training_markdown"])
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    training_name = gr.Textbox(label=translations["modelname"], info=translations["training_model_name"], value="", placeholder=translations["modelname"], interactive=True)
                    training_sr = gr.Radio(label=translations["sample_rate"], info=translations["sample_rate_info"], choices=["32k", "40k", "48k"], value="48k", interactive=True) 
                    training_ver = gr.Radio(label=translations["training_version"], info=translations["training_version_info"], choices=["v1", "v2"], value="v2", interactive=True) 
                    with gr.Row():
                        clean_dataset = gr.Checkbox(label=translations["clear_dataset"], value=False, interactive=True)
                        process_effects = gr.Checkbox(label=translations["preprocess_effect"], value=False, interactive=True)
                        training_f0 = gr.Checkbox(label=translations["training_pitch"], value=True, interactive=True)
                        custom_reference = gr.Checkbox(label=translations["custom_reference"], value=False, interactive=True)
                        checkpointing1 = gr.Checkbox(label=translations["memory_efficient_training"], value=False, interactive=True)
                        upload = gr.Checkbox(label=translations["upload_dataset"], value=False, interactive=True)
                    with gr.Row():
                        preprocess_split_audio_mode = gr.Radio(label=translations["split_audio_mode"], info=translations["split_audio_mode_info"], value="Automatic", choices=["Automatic", "Simple", "Skip"], interactive=True)
                        preprocess_normalization_mode = gr.Radio(label=translations["normalization_mode"], info=translations["normalization_mode_info"], value="none", choices=["none", "pre", "post"], interactive=True)
                    with gr.Row(visible=custom_reference.value) as custom_reference_row:
                        with gr.Accordion(translations["custom_reference"], open=True):
                            reference_name = gr.Dropdown(label=translations["reference_name"], info=translations["reference_name_info"], choices=reference_list, value=reference_list[0] if len(reference_list) >= 1 else "", allow_custom_value=True, interactive=True)
                            reference_refresh = gr.Button(translations["refresh"], scale=2)
                    with gr.Row(visible=clean_dataset.value) as clean_dataset_row:
                        clean_dataset_strength = gr.Slider(label=translations["clean_strength"], info=translations["clean_strength_info"], minimum=0, maximum=1, value=0.7, step=0.1, interactive=True)
                with gr.Column():
                    preprocess_button = gr.Button(translations["preprocess_button"], scale=2)
                    upload_dataset = gr.Files(label=translations["drop_audio"], file_types=file_types, visible=upload.value)
                    preprocess_info = gr.Textbox(label=translations["preprocess_info"], value="", interactive=False, container=True, lines=2)
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    with gr.Accordion(label=translations["f0_method"], open=False):
                        with gr.Group():
                            with gr.Row():
                                onnx_f0_mode2 = gr.Checkbox(label=translations["f0_onnx_mode"], value=False, interactive=True)
                                unlock_full_method4 = gr.Checkbox(label=translations["f0_unlock"], value=False, interactive=True)
                                autotune = gr.Checkbox(label=translations["autotune"], value=False, interactive=True)
                            extract_method = gr.Radio(label=translations["f0_method"], info=translations["f0_method_info"], choices=method_f0, value="rmvpe", interactive=True)
                            extract_hybrid_method = gr.Dropdown(label=translations["f0_method_hybrid"], info=translations["f0_method_hybrid_info"], choices=hybrid_f0_method, value=hybrid_f0_method[0], interactive=True, allow_custom_value=True, visible=extract_method.value == "hybrid")
                        extract_hop_length = gr.Slider(label=translations['hop_length'], info=translations["hop_length_info"], minimum=64, maximum=512, value=160, step=1, interactive=True, visible=False)
                        f0_autotune_strength = gr.Slider(minimum=0, maximum=1, label=translations["autotune_rate"], info=translations["autotune_rate_info"], value=1, step=0.1, interactive=True, visible=autotune.value)
                        alpha = gr.Slider(label=translations["alpha_label"], info=translations["alpha_info"], minimum=0.1, maximum=1, value=0.5, step=0.1, interactive=True, visible=False)
                    with gr.Accordion(label=translations["hubert_model"], open=False):
                        with gr.Group():
                            embed_mode2 = gr.Radio(label=translations["embed_mode"], info=translations["embed_mode_info"], value="fairseq", choices=embedders_mode, interactive=True, visible=True)
                            extract_embedders = gr.Radio(label=translations["hubert_model"], info=translations["hubert_info"], choices=embedders_model, value="hubert_base", interactive=True)
                        with gr.Row():
                            extract_embedders_custom = gr.Textbox(label=translations["modelname"], info=translations["modelname_info"], value="", placeholder="hubert_base", interactive=True, visible=extract_embedders.value == "custom")
                with gr.Column():
                    extract_button = gr.Button(translations["extract_button"], scale=2)
                    extract_info = gr.Textbox(label=translations["extract_info"], value="", interactive=False, lines=2)
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    total_epochs = gr.Slider(label=translations["total_epoch"], info=translations["total_epoch_info"], minimum=1, maximum=10000, value=300, step=1, interactive=True)
                    save_epochs = gr.Slider(label=translations["save_epoch"], info=translations["save_epoch_info"], minimum=1, maximum=10000, value=50, step=1, interactive=True)
                with gr.Column():
                    index_button = gr.Button(f"3. {translations['create_index']}", variant="primary", scale=2)
                    training_button = gr.Button(f"4. {translations['training_model']}", variant="primary", scale=2)
            with gr.Row():
                with gr.Accordion(label=translations["setting"], open=False):
                    with gr.Row():
                        index_algorithm = gr.Radio(label=translations["index_algorithm"], info=translations["index_algorithm_info"], choices=["Auto", "Faiss", "KMeans"], value="Auto", interactive=True)
                    with gr.Row():
                        cache_in_gpu = gr.Checkbox(label=translations["cache_in_gpu"], info=translations["cache_in_gpu_info"], value=True, interactive=True)
                        rms_extract = gr.Checkbox(label=translations["train&energy"], info=translations["train&energy_info"], value=False, interactive=True)
                        overtraining_detector = gr.Checkbox(label=translations["overtraining_detector"], info=translations["overtraining_detector_info"], value=False, interactive=True)
                    with gr.Row():
                        custom_dataset = gr.Checkbox(label=translations["custom_dataset"], info=translations["custom_dataset_info"], value=False, interactive=True)
                        save_only_latest = gr.Checkbox(label=translations["save_only_latest"], info=translations["save_only_latest_info"], value=True, interactive=True)
                        save_every_weights = gr.Checkbox(label=translations["save_every_weights"], info=translations["save_every_weights_info"], value=True, interactive=True)
                    with gr.Row():
                        clean_up = gr.Checkbox(label=translations["cleanup_training"], info=translations["cleanup_training_info"], value=False, interactive=True)
                        not_use_pretrain = gr.Checkbox(label=translations["not_use_pretrain_2"], info=translations["not_use_pretrain_info"], value=False, interactive=True)
                        custom_pretrain = gr.Checkbox(label=translations["custom_pretrain"], info=translations["custom_pretrain_info"], value=False, interactive=True)
                    with gr.Column():
                        dataset_path = gr.Textbox(label=translations["dataset_folder"], value="dataset", interactive=True, visible=custom_dataset.value)
                    with gr.Column():
                        with gr.Row(visible=False) as simple_option:
                            chunk_len = gr.Slider(minimum=0.5, maximum=5.0, value=3.0, step=0.1, label=translations["chunk_length"], info=translations["chunk_length_info"], interactive=True)
                            overlap_len = gr.Slider(minimum=0.0, maximum=0.4, value=0.3, step=0.1, label=translations["overlap_length"], info=translations["overlap_length_info"], interactive=True)
                        threshold = gr.Slider(minimum=1, maximum=100, value=50, step=1, label=translations["threshold"], interactive=True, visible=overtraining_detector.value)
                        with gr.Accordion(translations["setting_cpu_gpu"], open=False):
                            with gr.Column():
                                gpu_number = gr.Textbox(label=translations["gpu_number"], value=gpu_number_str(), info=translations["gpu_number_info"], interactive=True)
                                gpu_info = gr.Textbox(label=translations["gpu_info"], value=get_gpu_info(), info=translations["gpu_info_2"], interactive=False)
                                cpu_core = gr.Slider(label=translations["cpu_core"], info=translations["cpu_core_info"], minimum=1, maximum=os.cpu_count(), value=os.cpu_count(), step=1, interactive=True)          
                                train_batch_size = gr.Slider(label=translations["batch_size"], info=translations["batch_size_info"], minimum=1, maximum=64, value=8, step=1, interactive=True)
                    with gr.Group():
                        multiscale_mel_loss = gr.Checkbox(label=translations["multiscale_mel_loss"], info=translations["multiscale_mel_loss_info"], value=False, interactive=True)
                        vocoders = gr.Radio(label=translations["vocoder"], info=translations["vocoder_info"], choices=["Default", "MRF-HiFi-GAN", "RefineGAN"], value="Default", interactive=True) 
                    with gr.Row():
                        deterministic = gr.Checkbox(label=translations["deterministic"], info=translations["deterministic_info"], value=False, interactive=config.device.startswith("cuda"))
                        benchmark = gr.Checkbox(label=translations["benchmark"], info=translations["benchmark_info"], value=False, interactive=config.device.startswith("cuda"))
                    with gr.Row():
                        optimizer = gr.Radio(label=translations["optimizer"], info=translations["optimizer_info"], value="AdamW", choices=["AdamW", "RAdam", "AnyPrecisionAdamW"], interactive=True)
                    with gr.Row():
                        model_author = gr.Textbox(label=translations["training_author"], info=translations["training_author_info"], value="", placeholder=translations["training_author"], interactive=True)
                    with gr.Row():
                        with gr.Column():
                            with gr.Accordion(translations["custom_pretrain_info"], open=False, visible=custom_pretrain.value and not not_use_pretrain.value) as pretrain_setting:
                                pretrained_D = gr.Dropdown(label=translations["pretrain_file"].format(dg="D"), choices=pretrainedD, value=pretrainedD[0] if len(pretrainedD) > 0 else '', interactive=True, allow_custom_value=True)
                                pretrained_G = gr.Dropdown(label=translations["pretrain_file"].format(dg="G"), choices=pretrainedG, value=pretrainedG[0] if len(pretrainedG) > 0 else '', interactive=True, allow_custom_value=True)
                                refresh_pretrain = gr.Button(translations["refresh"], scale=2)
            with gr.Row():
                training_info = gr.Textbox(label=translations["train_info"], value="", interactive=False, lines=3)
            with gr.Row():
                with gr.Column():
                    with gr.Accordion(translations["export_model"], open=False):
                        with gr.Row():
                            model_file = gr.Dropdown(label=translations["model_name"], choices=model_name, value=model_name[0] if len(model_name) >= 1 else "", interactive=True, allow_custom_value=True)
                            index_file = gr.Dropdown(label=translations["index_path"], choices=index_path, value=index_path[0] if len(index_path) >= 1 else "", interactive=True, allow_custom_value=True)
                        with gr.Row():
                            refresh_file = gr.Button(f"1. {translations['refresh']}", scale=2)
                            zip_model = gr.Button(translations["zip_model"], variant="primary", scale=2)
                        with gr.Row():
                            zip_output = gr.File(label=translations["output_zip"], file_types=[".zip"], interactive=False, visible=False)
    with gr.Row():
        vocoders.change(fn=pitch_guidance_lock, inputs=[vocoders], outputs=[training_f0])
        training_f0.change(fn=vocoders_lock, inputs=[training_f0, vocoders], outputs=[vocoders])
        unlock_full_method4.change(fn=unlock_f0, inputs=[unlock_full_method4], outputs=[extract_method])
    with gr.Row():
        refresh_file.click(fn=change_models_choices, inputs=[], outputs=[model_file, index_file]) 
        zip_model.click(fn=zip_file, inputs=[training_name, model_file, index_file], outputs=[zip_output])                
        dataset_path.change(fn=lambda folder: os.makedirs(folder, exist_ok=True), inputs=[dataset_path], outputs=[])
    with gr.Row():
        upload.change(fn=visible, inputs=[upload], outputs=[upload_dataset]) 
        overtraining_detector.change(fn=visible, inputs=[overtraining_detector], outputs=[threshold]) 
        clean_dataset.change(fn=visible, inputs=[clean_dataset], outputs=[clean_dataset_row])
    with gr.Row():
        custom_dataset.change(fn=lambda custom_dataset: [visible(custom_dataset), "dataset"],inputs=[custom_dataset], outputs=[dataset_path, dataset_path])
        training_ver.change(fn=unlock_vocoder, inputs=[training_ver, vocoders], outputs=[vocoders])
        vocoders.change(fn=unlock_ver, inputs=[training_ver, vocoders], outputs=[training_ver])
    with gr.Row():
        custom_reference.change(fn=visible, inputs=[custom_reference], outputs=[custom_reference_row])
        extract_method.change(fn=lambda method, hybrid: [visible(method == "hybrid"), visible(method == "hybrid"), hoplength_show(method, hybrid)], inputs=[extract_method, extract_hybrid_method], outputs=[extract_hybrid_method, alpha, extract_hop_length])
        extract_hybrid_method.change(fn=hoplength_show, inputs=[extract_method, extract_hybrid_method], outputs=[extract_hop_length])
    with gr.Row():
        autotune.change(fn=visible, inputs=[autotune], outputs=[f0_autotune_strength])
        preprocess_split_audio_mode.change(fn=lambda a: visible(a == "Simple"), inputs=[preprocess_split_audio_mode], outputs=[simple_option])
        upload_dataset.upload(
            fn=lambda files, folder: [shutil_move(f.name, os.path.join(folder, os.path.split(f.name)[1])) for f in files] if folder != "" else gr_warning(translations["dataset_folder1"]),
            inputs=[upload_dataset, dataset_path], 
            outputs=[], 
            api_name="upload_dataset"
        )           
    with gr.Row():
        not_use_pretrain.change(fn=lambda a, b: visible(a and not b), inputs=[custom_pretrain, not_use_pretrain], outputs=[pretrain_setting])
        custom_pretrain.change(fn=lambda a, b: visible(a and not b), inputs=[custom_pretrain, not_use_pretrain], outputs=[pretrain_setting])
        refresh_pretrain.click(fn=change_pretrained_choices, inputs=[], outputs=[pretrained_D, pretrained_G])
    with gr.Row():
        preprocess_button.click(
            fn=preprocess,
            inputs=[
                training_name, 
                training_sr, 
                cpu_core,
                preprocess_split_audio_mode, 
                process_effects,
                dataset_path,
                clean_dataset,
                clean_dataset_strength,
                chunk_len, 
                overlap_len,
                preprocess_normalization_mode
            ],
            outputs=[preprocess_info],
            api_name="preprocess"
        )
    with gr.Row():
        embed_mode2.change(fn=change_embedders_mode, inputs=[embed_mode2], outputs=[extract_embedders])
        extract_embedders.change(fn=lambda extract_embedders: visible(extract_embedders == "custom"), inputs=[extract_embedders], outputs=[extract_embedders_custom])
        reference_refresh.click(fn=change_reference_choices, inputs=[], outputs=[reference_name])
    with gr.Row():
        extract_button.click(
            fn=extract,
            inputs=[
                training_name, 
                training_ver, 
                extract_method, 
                training_f0, 
                extract_hop_length, 
                cpu_core,
                gpu_number,
                training_sr, 
                extract_embedders, 
                extract_embedders_custom,
                onnx_f0_mode2,
                embed_mode2,
                autotune,
                f0_autotune_strength,
                extract_hybrid_method,
                rms_extract,
                alpha
            ],
            outputs=[extract_info],
            api_name="extract"
        )
    with gr.Row():
        index_button.click(
            fn=create_index,
            inputs=[
                training_name, 
                training_ver, 
                index_algorithm
            ],
            outputs=[training_info],
            api_name="create_index"
        )
    with gr.Row():
        training_button.click(
            fn=training,
            inputs=[
                training_name, 
                training_ver, 
                save_epochs, 
                save_only_latest, 
                save_every_weights, 
                total_epochs, 
                training_sr,
                train_batch_size, 
                gpu_number,
                training_f0,
                not_use_pretrain,
                custom_pretrain,
                pretrained_G,
                pretrained_D,
                overtraining_detector,
                threshold,
                clean_up,
                cache_in_gpu,
                model_author,
                vocoders,
                checkpointing1,
                deterministic, 
                benchmark,
                optimizer,
                rms_extract,
                custom_reference,
                reference_name,
                multiscale_mel_loss
            ],
            outputs=[training_info],
            api_name="training_model"
        )