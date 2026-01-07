import os
import sys

sys.path.append(os.getcwd())

try:
    argv = sys.argv[1]
except IndexError:
    argv = None

argv_is_allows = ["--audio_effects", "--convert", "--create_dataset", "--create_index", "--extract", "--preprocess", "--separator_music", "--train", "--help_audio_effects", "--help_convert", "--help_create_dataset", "--help_create_index", "--help_extract", "--help_preprocess", "--help_separate_music",  "--help_train", "--help", "--create_reference", "help_create_reference"]

if argv not in argv_is_allows:
    print("Invalid syntax! Use --help for more information")
    quit()


if argv_is_allows[1] in argv: from advanced_rvc_inference.conversion.convert import main
elif argv_is_allows[2] in argv: from advanced_rvc_inference.create_dataset import main
elif argv_is_allows[3] in argv: from advanced_rvc_inference.create_index import main
elif argv_is_allows[4] in argv: from advanced_rvc_inference.extracting.extract import main
elif argv_is_allows[5] in argv: from advanced_rvc_inference.preprocess.preprocess import main
elif argv_is_allows[6] in argv: from advanced_rvc_inference.separate_music import main
elif argv_is_allows[7] in argv: from advanced_rvc_inference.training.train import main
elif argv_is_allows[17] in argv: from advanced_rvc_inference.create_reference import main

elif argv_is_allows[9] in argv:
    print("""Parameters for --convert:
        1. Voice processing configuration:
            - `--pitch` (default: `0`): Adjust pitch.
            - `--filter_radius` (default: `3`): Smoothness of the F0 contour.
            - `--index_rate` (default: `0.5`): Ratio for using voice index.
            - `--rms_mix_rate` (default: `1`): Coefficient for adjusting amplitude/volume.
            - `--protect` (default: `0.33`): Protect consonants.
            - `--hop_length` (default: `64`): Hop length for audio processing.

        2. F0 configuration:
            - `--f0_method` (default: `rmvpe`): F0 prediction method (`pm`, `dio`, `mangio-crepe-tiny`, `mangio-crepe-small`, `mangio-crepe-medium`, `mangio-crepe-large`, `mangio-crepe-full`, `crepe-tiny`, `crepe-small`, `crepe-medium`, `crepe-large`, `crepe-full`, `fcpe`, `fcpe-legacy`, `rmvpe`, `rmvpe-legacy`, `harvest`, `yin`, `pyin`, `swipe`).
            - `--f0_autotune` (default: `False`): Enable F0 autotune.
            - `--f0_autotune_strength` (default: `1`): Strength of F0 autotune correction.
            - `--f0_file` (default: ``): Path to pre-existing F0 file.
            - `--f0_onnx` (default: `False`): Use ONNX version of F0.
            - `--proposal_pitch` (default: `False`): Propose pitch instead of manual adjustment.
            - `--proposal_pitch_threshold` (default: `0.0`): Threshold for pitch frequency estimation.
            - `--alpha` (default: `0.5`): Pitch blending threshold for hybrid pitch estimation.

        3. Embedding model:
            - `--embedder_model` (default: `hubert_base`): Embedding model to use.
            - `--embedders_mode` (default: `fairseq`): Embedding mode (`fairseq`, `transformers`, `onnx`, `whisper`).

        4. File paths:
            - `--input_path` (required): Path to input audio file.
            - `--output_path` (default: `./audios/output.wav`): Path to save output file.
            - `--export_format` (default: `wav`): Export file format.
            - `--pth_path` (required): Path to `.pth` model file.
            - `--index_path` (default: `None`): Path to index file (if any).

        5. Audio cleaning:
            - `--clean_audio` (default: `False`): Apply audio cleaning.
            - `--clean_strength` (default: `0.7`): Cleaning strength.

        6. Resampling & audio splitting:
            - `--resample_sr` (default: `0`): New sampling frequency (0 means keep original).
            - `--split_audio` (default: `False`): Split audio before processing.

        7. Testing & optimization:
            - `--checkpointing` (default: `False`): Enable/disable checkpointing to save RAM.

        8. Formant shifting:
            - `--formant_shifting` (default: `False`): Enable formant shifting effect.
            - `--formant_qfrency` (default: `0.8`): Formant shift frequency coefficient.
            - `--formant_timbre` (default: `0.8`): Voice timbre change coefficient.
    """)
    quit()
elif argv_is_allows[10] in argv:
    print("""Parameters for --create_dataset:
        1. Dataset path & configuration:
            - `--input_data` (required): Audio source link (YouTube link, use `,` to separate multiple links).
            - `--output_dirs` (default: `./dataset`): Output data folder.
            - `--sample_rate` (default: `48000`): Audio sample rate.

        2. Data cleaning:
            - `--clean_dataset` (default: `False`): Apply data cleaning.
            - `--clean_strength` (default: `0.7`): Data cleaning strength.

        3. Vocal separation & effects:
            - `--separate` (default: `True`): Separate music.
            - `--separator_reverb` (default: `False`): Separate vocal reverb.
            - `--model_name` (default: `MDXNET_Main`): Separation model ('Main_340', 'Main_390', 'Main_406', 'Main_427', 'Main_438', 'Inst_full_292', 'Inst_HQ_1', 'Inst_HQ_2', 'Inst_HQ_3', 'Inst_HQ_4', 'Inst_HQ_5', 'Kim_Vocal_1', 'Kim_Vocal_2', 'Kim_Inst', 'Inst_187_beta', 'Inst_82_beta', 'Inst_90_beta', 'Voc_FT', 'Crowd_HQ', 'MDXNET_9482', 'Inst_1', 'Inst_2', 'Inst_3', 'MDXNET_1_9703', 'MDXNET_2_9682', 'MDXNET_3_9662', 'Inst_Main', 'MDXNET_Main', 'HT-Tuned', 'HT-Normal', 'HD_MMI', 'HT_6S', 'HP-1', 'HP-2', 'HP-Vocal-1', 'HP-Vocal-2', 'HP2-1', 'HP2-2', 'HP2-3', 'SP-2B-1', 'SP-2B-2', 'SP-3B-1', 'SP-4B-1', 'SP-4B-2', 'SP-MID-1', 'SP-MID-2').
            - `--reverb_model` (default: `MDX-Reverb`): Separation model ("MDX-Reverb", 'VR-Reverb', 'Echo-Aggressive', 'Echo-Normal').
            - `--denoise_model` (default: `Normal`): Separation model ('Lite', 'Normal').

        4. Audio processing configuration:
            - `--shifts` (default: `2`): Number of predictions.
            - `--batch_size` (default: `1`): Batch size.
            - `--overlap` (default: `0.25`): Overlap between segments.
            - `--aggression` (default: `5`): Intensity of main vocal extraction.
            - `--hop_length` (default: `1024`): MDX hop length during processing.
            - `--window_size` (default: `512`): Window size.
            - `--segments_size` (default: `256`): Audio segment size.
            - `--post_process_threshold` (default: `0.2`): Post-processing level after separation.

        5. Other audio processing configuration:
            - `--enable_tta` (default: `False`): Enhanced inference.
            - `--enable_denoise` (default: `False`): Noise reduction.
            - `--high_end_process` (default: `False`): High-frequency processing.
            - `--enable_post_process` (default: `False`): Post-processing.

        6. Skip audio sections:
            - `--skip_seconds` (default: `False`): Skip audio seconds.
            - `--skip_start_audios` (default: `0`): Time (seconds) to skip at audio start.
            - `--skip_end_audios` (default: `0`): Time (seconds) to skip at audio end.
    """)
    quit()
elif argv_is_allows[11] in argv:
    print("""Parameters for --create_index:
        1. Model information:
            - `--model_name` (required): Model name.
            - `--rvc_version` (default: `v2`): Version (`v1`, `v2`).
            - `--index_algorithm` (default: `Auto`): Index algorithm to use (`Auto`, `Faiss`, `KMeans`).
    """)
    quit()
elif argv_is_allows[12] in argv:
    print("""Parameters for --extract:
        1. Model information:
            - `--model_name` (required): Model name.
            - `--rvc_version` (default: `v2`): RVC version (`v1`, `v2`).

        2. F0 configuration:
            - `--f0_method` (default: `rmvpe`): F0 prediction method (`pm`, `dio`, `mangio-crepe-tiny`, `mangio-crepe-small`, `mangio-crepe-medium`, `mangio-crepe-large`, `mangio-crepe-full`, `crepe-tiny`, `crepe-small`, `crepe-medium`, `crepe-large`, `crepe-full`, `fcpe`, `fcpe-legacy`, `rmvpe`, `rmvpe-legacy`, `harvest`, `yin`, `pyin`, `swipe`).
            - `--f0_onnx` (default: `False`): Use ONNX version of F0.
            - `--pitch_guidance` (default: `True`): Use pitch guidance.
            - `--f0_autotune` (default: `False`): Enable F0 autotune.
            - `--f0_autotune_strength` (default: `1`): Strength of F0 autotune correction.
            - `--alpha` (default: `0.5`): Pitch blending threshold for hybrid pitch estimation.

        3. Processing configuration:
            - `--hop_length` (default: `128`): Hop length during processing.
            - `--cpu_cores` (default: `2`): Number of CPU threads to use.
            - `--gpu` (default: `-`): Specify GPU to use (e.g., `0` for first GPU, `-` to disable GPU).
            - `--sample_rate` (required): Input audio sample rate.

        4. Embedding configuration:
            - `--embedder_model` (default: `hubert_base`): Embedding model name.
            - `--embedders_mode` (default: `fairseq`): Embedding mode (`fairseq`, `transformers`, `onnx`, `whisper`).

        5. RMS:
            - `--rms_extract` (default: False): Also extract RMS energy.
    """)
    quit()
elif argv_is_allows[13] in argv:
    print("""Parameters for --preprocess:
        1. Model information:
            - `--model_name` (required): Model name.

        2. Data configuration:
            - `--dataset_path` (default: `./dataset`): Path to folder containing data files.
            - `--sample_rate` (required): Audio data sample rate.

        3. Processing configuration:
            - `--cpu_cores` (default: `2`): Number of CPU threads to use.
            - `--cut_preprocess` (default: `Automatic`): Preprocessing cut method (`Automatic`, `Simple`, `Skip`).
            - `--process_effects` (default: `False`): Apply preprocessing.
            - `--clean_dataset` (default: `False`): Clean data files.
            - `--clean_strength` (default: `0.7`): Data cleaning strength.

        4. Other configuration:
            - `--chunk_len` (default: `3.0`): Audio chunk length for 'Simple' method.
            - `--overlap_len` (default: `0.3`): Overlap length between slices for 'Simple' method.
            - `--normalization_mode` (default: `none`): Audio normalization processing (`none`, `pre`, `post`)
    """)
    quit()
elif argv_is_allows[14] in argv:
    print("""Parameters for --separate_music:
        1. Input/Output configuration:
            - `--input_path` (required): Path to input audio file.
            - `--output_dirs` (default: `./audios`): Output file save folder.
            - `--export_format` (default: `wav`): Export file format (`wav`, `mp3`,...).
            - `--sample_rate` (default: `44100`): Output audio sample rate.

        2. Model configuration:
            - `--model_name` (default: `MDXNET_Main`): Separation model ('Main_340', 'Main_390', 'Main_406', 'Main_427', 'Main_438', 'Inst_full_292', 'Inst_HQ_1', 'Inst_HQ_2', 'Inst_HQ_3', 'Inst_HQ_4', 'Inst_HQ_5', 'Kim_Vocal_1', 'Kim_Vocal_2', 'Kim_Inst', 'Inst_187_beta', 'Inst_82_beta', 'Inst_90_beta', 'Voc_FT', 'Crowd_HQ', 'MDXNET_9482', 'Inst_1', 'Inst_2', 'Inst_3', 'MDXNET_1_9703', 'MDXNET_2_9682', 'MDXNET_3_9662', 'Inst_Main', 'MDXNET_Main', 'HT-Tuned', 'HT-Normal', 'HD_MMI', 'HT_6S', 'HP-1', 'HP-2', 'HP-Vocal-1', 'HP-Vocal-2', 'HP2-1', 'HP2-2', 'HP2-3', 'SP-2B-1', 'SP-2B-2', 'SP-3B-1', 'SP-4B-1', 'SP-4B-2', 'SP-MID-1', 'SP-MID-2').
            - `--karaoke_model` (default: `MDX-Version-1`): Separation model ('MDX-Version-1', 'MDX-Version-2', 'VR-Version-1', 'VR-Version-2').
            - `--reverb_model` (default: `MDX-Reverb`): Separation model ("MDX-Reverb", 'VR-Reverb', 'Echo-Aggressive', 'Echo-Normal').
            - `--denoise_model` (default: `Normal`): Separation model ('Lite', 'Normal').

        3. Audio processing configuration:
            - `--shifts` (default: `2`): Number of predictions.
            - `--batch_size` (default: `1`): Batch size.
            - `--overlap` (default: `0.25`): Overlap between segments.
            - `--aggression` (default: `5`): Intensity of main vocal extraction.
            - `--hop_length` (default: `1024`): MDX hop length during processing.
            - `--window_size` (default: `512`): Window size.
            - `--segments_size` (default: `256`): Audio segment size.
            - `--post_process_threshold` (default: `0.2`): Post-processing level after separation.

        4. Other audio processing configuration:
            - `--enable_tta` (default: `False`): Enhanced inference.
            - `--enable_denoise` (default: `False`): Noise reduction.
            - `--high_end_process` (default: `False`): High-frequency processing.
            - `--enable_post_process` (default: `False`): Post-processing.
            - `--separate_backing` (default: `False`): Separate backing vocals.
            - `--separate_reverb` (default: `False`): Separate vocal reverb.
    """)
    quit()
elif argv_is_allows[15] in argv:
    print("""Parameters for --train:
        1. Model configuration:
            - `--model_name` (required): Model name.
            - `--rvc_version` (default: `v2`): RVC version (`v1`, `v2`).
            - `--model_author` (optional): Model author.

        2. Save configuration:
            - `--save_every_epoch` (required): Number of epochs between each save.
            - `--save_only_latest` (default: `True`): Save only the latest checkpoint.
            - `--save_every_weights` (default: `True`): Save all model weights.

        3. Training configuration:
            - `--total_epoch` (default: `300`): Total training epochs.
            - `--batch_size` (default: `8`): Batch size during training.

        4. Device configuration:
            - `--gpu` (default: `0`): Specify GPU to use (number or `-` if not using GPU).
            - `--cache_data_in_gpu` (default: `False`): Cache data in GPU for speedup.

        5. Advanced training configuration:
            - `--pitch_guidance` (default: `True`): Use pitch guidance.
            - `--g_pretrained_path` (default: ``): Path to pre-trained G weights.
            - `--d_pretrained_path` (default: ``): Path to pre-trained D weights.
            - `--vocoder` (default: `Default`): Vocoder to use (`Default`, `MRF-HiFi-GAN`, `RefineGAN`).
            - `--energy_use` (default: `False`): Use RMS energy.

        6. Over-training detection:
            - `--overtraining_detector` (default: `False`): Enable/disable over-training detection.
            - `--overtraining_threshold` (default: `50`): Threshold for determining over-training.

        7. Data processing:
            - `--cleanup` (default: `False`): Clean up old training files to retrain from scratch.

        8. Optimization:
            - `--checkpointing` (default: `False`): Enable/disable checkpointing to save RAM.
            - `--deterministic` (default: `False`): When enabled, uses highly deterministic algorithms, ensuring the same input yields the same result each run.
            - `--benchmark` (default: `False`): When enabled, tests and selects the most optimal algorithm for specific hardware and size.
            - `--optimizer` (default: `AdamW`): Optimizer to use (`AdamW`, `RAdam`, `AnyPrecisionAdamW`).
            - `--multiscale_mel_loss` (default: `False`): Compare Mel spectrograms of real and fake audio at multiple scales. Helps model learn better timbre details, brightness, and frequency structure, improving output voice quality and naturalness.

        9. Reference set:
            - `--use_custom_reference` (default: `False`): Use custom reference set.
            - `--reference_path` (default: `False`): Path to reference set.
    """)
    quit()
elif argv_is_allows[18] in argv:
    print("""Parameters for --create_reference:
        1. File paths:
            - `--audio_path` (required): Path to input audio file.
            - `--reference_name` (default: `reference`): Output reference set save path.

        2. Reference set configuration:
            - `--pitch_guidance` (default: `True`): Use pitch guidance.
            - `--energy_use` (default: `False`): Use RMS energy.
            - `--version` (default: `v2`): RVC version (`v1`, `v2`).

        3. Embedding configuration:
            - `--embedder_model` (default: `hubert_base`): Embedding model name.
            - `--embedders_mode` (default: `fairseq`): Embedding mode (`fairseq`, `transformers`, `onnx`, `whisper`).

        4. F0 configuration:
            - `--f0_method` (default: `rmvpe`): F0 prediction method (`pm`, `dio`, `mangio-crepe-tiny`, `mangio-crepe-small`, `mangio-crepe-medium`, `mangio-crepe-large`, `mangio-crepe-full`, `crepe-tiny`, `crepe-small`, `crepe-medium`, `crepe-large`, `crepe-full`, `fcpe`, `fcpe-legacy`, `rmvpe`, `rmvpe-legacy`, `harvest`, `yin`, `pyin`, `swipe`).
            - `--f0_onnx` (default: `False`): Use ONNX version of F0.
            - `--f0_up_key` (default: `0`): Adjust pitch.
            - `--filter_radius` (default: `3`): Smoothness of the F0 contour.
            - `--f0_autotune` (default: `False`): Enable F0 autotune.
            - `--f0_autotune_strength` (default: `1`): Strength of F0 autotune correction.
            - `--f0_file` (default: ``): Path to pre-existing F0 file.
            - `--proposal_pitch` (default: `False`): Propose pitch instead of manual adjustment.
            - `--proposal_pitch_threshold` (default: `0.0`): Threshold for pitch frequency estimation.
            - `--alpha` (default: `0.5`): Pitch blending threshold for hybrid pitch estimation.
    """)
    quit()
elif argv_is_allows[16] in argv:
    print("""Usage:
        1. `--help_audio_effects`: Help for adding audio effects.
        2. `--help_convert`: Help for audio conversion.
        3. `--help_create_dataset`: Help for creating training data.
        4. `--help_create_index`: Help for creating index.
        5. `--help_extract`: Help for extracting training data.
        6. `--help_preprocess`: Help for preprocessing data.
        7. `--help_separate_music`: Help for music separation.
        8. `--help_train`: Help for model training.
        9. `--help_create_reference`: Help for creating reference set.
    """)
    quit()

if __name__ == "__main__":
    import torch.multiprocessing as mp

    if "--train" in argv: mp.set_start_method("spawn")
    if "--preprocess" in argv or "--extract" in argv: mp.set_start_method("spawn", force=True)

    main()
