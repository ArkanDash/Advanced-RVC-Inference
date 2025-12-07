import os
import sys

sys.path.append(os.getcwd())

try:
    argv = sys.argv[1]
except IndexError:
    argv = None

argv_is_allows = ["--audio_effects", "--convert", "--create_dataset", "--create_index", "--extract", "--preprocess", "--separator_music", "--train", "--help_audio_effects", "--help_convert", "--help_create_dataset", "--help_create_index", "--help_extract", "--help_preprocess", "--help_separate_music",  "--help_train", "--help", "--create_reference", "help_create_reference"]

if argv not in argv_is_allows:
    print("Cú pháp không hợp lệ! Sử dụng --help để biết thêm")
    quit()


if argv_is_allows[1] in argv: from advanced_rvc_inference.infer.conversion.convert import main
elif argv_is_allows[2] in argv: from advanced_rvc_inference.infer.create_dataset import main
elif argv_is_allows[3] in argv: from advanced_rvc_inference.infer.create_index import main
elif argv_is_allows[4] in argv: from advanced_rvc_inference.infer.extracting.extract import main
elif argv_is_allows[5] in argv: from advanced_rvc_inference.infer.preprocess.preprocess import main
elif argv_is_allows[6] in argv: from advanced_rvc_inference.infer.separate_music import main
elif argv_is_allows[7] in argv: from advanced_rvc_inference.infer.training.train import main
elif argv_is_allows[17] in argv: from advanced_rvc_inference.infer.create_reference import main

elif argv_is_allows[9] in argv:
    print("""Các tham số của --convert:
        1. Cấu hình xử lí giọng nói:
            - `--pitch` (mặc định: `0`): Điều chỉnh cao độ.
            - `--filter_radius` (mặc định: `3`): Độ mượt của đường F0.
            - `--index_rate` (mặc định: `0.5`): Tỷ lệ sử dụng chỉ mục giọng nói.
            - `--rms_mix_rate` (mặc định: `1`): Hệ số điều chỉnh biên độ âm lượng.
            - `--protect` (mặc định: `0.33`): Bảo vệ phụ âm.
            - `--hop_length` (mặc định: `64`): Bước nhảy khi xử lí âm thanh.

        2. Cấu hình F0:
            - `--f0_method` (mặc định: `rmvpe`): Phương pháp dự đoán F0 (`pm`, `dio`, `mangio-crepe-tiny`, `mangio-crepe-small`, `mangio-crepe-medium`, `mangio-crepe-large`, `mangio-crepe-full`, `crepe-tiny`, `crepe-small`, `crepe-medium`, `crepe-large`, `crepe-full`, `fcpe`, `fcpe-legacy`, `rmvpe`, `rmvpe-legacy`, `harvest`, `yin`, `pyin`, `swipe`).
            - `--f0_autotune` (mặc định: `False`): Có tự động điều chỉnh F0 hay không.
            - `--f0_autotune_strength` (mặc định: `1`): Cường độ hiệu chỉnh tự động F0.
            - `--f0_file` (mặc định: ``): Đường dẫn tệp F0 có sẵn.
            - `--f0_onnx` (mặc định: `False`): Có sử dụng phiên bản ONNX của F0 hay không.
            - `--proposal_pitch` (mặc định: `False`): Đề xuất cao độ thay vì điều chỉnh thủ công.
            - `--proposal_pitch_threshold` (mặc định: `0.0`): Ngưỡng tần số ước tính cao độ.
            - `--alpha` (mặc định: `0.5`): Ngưỡng trộn cao độ khi ước tính cao độ hybrid.

        3. Mô hình nhúng:
            - `--embedder_model` (mặc định: `hubert_base`): Mô hình nhúng sử dụng.
            - `--embedders_mode` (mặc định: `fairseq`): Chế độ nhúng (`fairseq`, `transformers`, `onnx`, `whisper`).

        4. Đường dẫn tệp:
            - `--input_path` (bắt buộc): Đường dẫn tệp âm thanh đầu vào.
            - `--output_path` (mặc định: `./audios/output.wav`): Đường dẫn lưu tệp đầu ra.
            - `--export_format` (mặc định: `wav`): Định dạng xuất tệp.
            - `--pth_path` (bắt buộc): Đường dẫn đến tệp mô hình `.pth`.
            - `--index_path` (mặc định: `None`): Đường dẫn tệp chỉ mục (nếu có).

        5. Làm sạch âm thanh:
            - `--clean_audio` (mặc định: `False`): Có áp dụng làm sạch âm thanh không.
            - `--clean_strength` (mặc định: `0.7`): Mức độ làm sạch.

        6. Resampling & chia nhỏ âm thanh:
            - `--resample_sr` (mặc định: `0`): Tần số lấy mẫu mới (0 nghĩa là giữ nguyên).
            - `--split_audio` (mặc định: `False`): Có chia nhỏ audio trước khi xử lí không.

        7. Kiểm tra & tối ưu hóa:
            - `--checkpointing` (mặc định: `False`): Bật/tắt checkpointing để tiết kiệm RAM.

        8. Dịch formant:
            - `--formant_shifting` (mặc định: `False`): Có bật hiệu ứng dịch formant không.
            - `--formant_qfrency` (mặc định: `0.8`): Hệ số dịch formant theo tần số.
            - `--formant_timbre` (mặc định: `0.8`): Hệ số thay đổi màu sắc giọng.
    """)
    quit()
elif argv_is_allows[10] in argv:
    print("""Các tham số của --create_dataset:
        1. Đường dẫn & cấu hình dataset:
            - `--input_data` (bắt buộc): Đường dẫn liên kết đến âm thanh (Liên kết Youtube, có thể dùng dấu `,` để dùng nhiều liên kết).
            - `--output_dirs` (mặc định: `./dataset`): Thư mục xuất dữ liệu đầu ra.
            - `--sample_rate` (mặc định: `48000`): Tần số lấy mẫu cho âm thanh.

        2. Làm sạch dữ liệu:
            - `--clean_dataset` (mặc định: `False`): Có áp dụng làm sạch dữ liệu hay không.
            - `--clean_strength` (mặc định: `0.7`): Mức độ làm sạch dữ liệu.

        3. Tách giọng & hiệu ứng:
            - `--separate` (mặc định: `True`): có tách nhạc hay không.
            - `--separator_reverb` (mặc định: `False`): Có tách vang giọng không.
            - `--model_name` (mặc định: `MDXNET_Main`): Mô hình tách nhạc ('Main_340', 'Main_390', 'Main_406', 'Main_427', 'Main_438', 'Inst_full_292', 'Inst_HQ_1', 'Inst_HQ_2', 'Inst_HQ_3', 'Inst_HQ_4', 'Inst_HQ_5', 'Kim_Vocal_1', 'Kim_Vocal_2', 'Kim_Inst', 'Inst_187_beta', 'Inst_82_beta', 'Inst_90_beta', 'Voc_FT', 'Crowd_HQ', 'MDXNET_9482', 'Inst_1', 'Inst_2', 'Inst_3', 'MDXNET_1_9703', 'MDXNET_2_9682', 'MDXNET_3_9662', 'Inst_Main', 'MDXNET_Main', 'HT-Tuned', 'HT-Normal', 'HD_MMI', 'HT_6S', 'HP-1', 'HP-2', 'HP-Vocal-1', 'HP-Vocal-2', 'HP2-1', 'HP2-2', 'HP2-3', 'SP-2B-1', 'SP-2B-2', 'SP-3B-1', 'SP-4B-1', 'SP-4B-2', 'SP-MID-1', 'SP-MID-2').
            - `--reverb_model` (mặc định: `MDX-Reverb`): Mô hình tách nhạc ("MDX-Reverb", 'VR-Reverb', 'Echo-Aggressive', 'Echo-Normal').
            - `--denoise_model` (mặc định: `Normal`): Mô hình tách nhạc ('Lite', 'Normal').
 
        4. Cấu hình xử lí âm thanh:
            - `--shifts` (mặc định: `2`): Số lượng dự đoán.
            - `--batch_size` (mặc định: `1`): Kích thước lô.
            - `--overlap` (mặc định: `0.25`): Mức độ chồng lấn giữa các đoạn.
            - `--aggression` (mặc định: `5`): Cường độ chiết xuất thân chính.
            - `--hop_length` (mặc định: `1024`): Bước nhảy MDX khi xử lí.
            - `--window_size` (mặc định: `512`): Kích thước cửa sổ.
            - `--segments_size` (mặc định: `256`): Kích thước phân đoạn âm thanh.
            - `--post_process_threshold` (mặc định: `0.2`): Mức độ xử lí hậu kỳ sau khi tách nhạc.

        5. Cấu hình xử lí âm thanh khác:
            - `--enable_tta` (mặc định: `False`): Tăng cường suy luận.
            - `--enable_denoise` (mặc định: `False`): Khữ tách nhạc.
            - `--high_end_process` (mặc định: `False`): Xử lí dải cao.
            - `--enable_post_process` (mặc định: `False`): Hậu xử lí.

        6. Bỏ qua phần âm thanh:
            - `--skip_seconds` (mặc định: `False`): Có bỏ qua giây âm thanh nào không.
            - `--skip_start_audios` (mặc định: `0`): Thời gian (giây) cần bỏ qua ở đầu audio.
            - `--skip_end_audios` (mặc định: `0`): Thời gian (giây) cần bỏ qua ở cuối audio.
    """)
    quit()
elif argv_is_allows[11] in argv:
    print("""Các tham số của --create_index:
        1. Thông tin mô hình:
            - `--model_name` (bắt buộc): Tên mô hình.
            - `--rvc_version` (mặc định: `v2`): Phiên bản (`v1`, `v2`).
            - `--index_algorithm` (mặc định: `Auto`): Thuật toán index sử dụng (`Auto`, `Faiss`, `KMeans`).
    """)
    quit()
elif argv_is_allows[12] in argv:
    print("""Các tham số của --extract:
        1. Thông tin mô hình:
            - `--model_name` (bắt buộc): Tên mô hình.
            - `--rvc_version` (mặc định: `v2`): Phiên bản RVC (`v1`, `v2`).

        2. Cấu hình F0:
            - `--f0_method` (mặc định: `rmvpe`): Phương pháp dự đoán F0 (`pm`, `dio`, `mangio-crepe-tiny`, `mangio-crepe-small`, `mangio-crepe-medium`, `mangio-crepe-large`, `mangio-crepe-full`, `crepe-tiny`, `crepe-small`, `crepe-medium`, `crepe-large`, `crepe-full`, `fcpe`, `fcpe-legacy`, `rmvpe`, `rmvpe-legacy`, `harvest`, `yin`, `pyin`, `swipe`).
            - `--f0_onnx` (mặc định: `False`): Có sử dụng phiên bản ONNX của F0 hay không.
            - `--pitch_guidance` (mặc định: `True`): Có sử dụng hướng dẫn cao độ hay không.
            - `--f0_autotune` (mặc định: `False`): Có tự động điều chỉnh F0 hay không.
            - `--f0_autotune_strength` (mặc định: `1`): Cường độ hiệu chỉnh tự động F0.
            - `--alpha` (mặc định: `0.5`): Ngưỡng trộn cao độ khi ước tính cao độ hybrid.

        3. Cấu hình xử lí:
            - `--hop_length` (mặc định: `128`): Độ dài bước nhảy trong quá trình xử lí.
            - `--cpu_cores` (mặc định: `2`): Số lượng luồng CPU sử dụng.
            - `--gpu` (mặc định: `-`): Chỉ định GPU sử dụng (ví dụ: `0` cho GPU đầu tiên, `-` để tắt GPU).
            - `--sample_rate` (bắt buộc): Tần số lấy mẫu của âm thanh đầu vào.

        4. Cấu hình nhúng:
            - `--embedder_model` (mặc định: `hubert_base`): Tên mô hình nhúng.
            - `--embedders_mode` (mặc định: `fairseq`): Chế độ nhúng (`fairseq`, `transformers`, `onnx`, `whisper`).
          
        4. RMS:
            - `--rms_extract` (mặc định: False): Trích xuất thêm năng lượng rms.
    """)
    quit()
elif argv_is_allows[13] in argv:
    print("""Các tham số của --preprocess:
        1. Thông tin mô hình:
            - `--model_name` (bắt buộc): Tên mô hình.

        2. Cấu hình dữ liệu:
            - `--dataset_path` (mặc định: `./dataset`): Đường dẫn thư mục chứa tệp dữ liệu.
            - `--sample_rate` (bắt buộc): Tần số lấy mẫu của dữ liệu âm thanh.

        3. Cấu hình xử lí:
            - `--cpu_cores` (mặc định: `2`): Số lượng luồng CPU sử dụng.
            - `--cut_preprocess` (mặc định: `Automatic`): Cách cắt dữ liệu tiền xử lí (`Automatic`, `Simple`, `Skip`).
            - `--process_effects` (mặc định: `False`): Có áp dụng tiền xử lí hay không.
            - `--clean_dataset` (mặc định: `False`): Có làm sạch tệp dữ liệu hay không.
            - `--clean_strength` (mặc định: `0.7`): Độ mạnh của quá trình làm sạch dữ liệu.
        
        4. Cấu hình khác:
            - `--chunk_len` (mặc định: `3.0`): Độ dài của đoạn âm thanh cho phương pháp 'Simple'.
            - `--overlap_len` (mặc định: `0.3`): Độ dài của phần chồng chéo giữa các lát cắt đối với phương pháp 'Simple'.
            - `--normalization_mode` (mặc định: `none`): Có xử lí chuẩn hóa âm thanh không (`none`, `pre`, `post`)
    """)
    quit()
elif argv_is_allows[14] in argv:
    print("""Các tham số của --separate_music:
        1. Cấu hình đầu vào, đầu ra:
            - `--input_path` (bắt buộc): Đường dẫn tệp âm thanh đầu vào.
            - `--output_dirs` (mặc định: `./audios`): Thư mục lưu tệp đầu ra.
            - `--export_format` (mặc định: `wav`): Định dạng xuất tệp (`wav`, `mp3`,...).
            - `--sample_rate` (mặc định: `44100`): Tần số lấy mẫu của âm thanh đầu ra.

        2. Cấu hình mô hình:
            - `--model_name` (mặc định: `MDXNET_Main`): Mô hình tách nhạc ('Main_340', 'Main_390', 'Main_406', 'Main_427', 'Main_438', 'Inst_full_292', 'Inst_HQ_1', 'Inst_HQ_2', 'Inst_HQ_3', 'Inst_HQ_4', 'Inst_HQ_5', 'Kim_Vocal_1', 'Kim_Vocal_2', 'Kim_Inst', 'Inst_187_beta', 'Inst_82_beta', 'Inst_90_beta', 'Voc_FT', 'Crowd_HQ', 'MDXNET_9482', 'Inst_1', 'Inst_2', 'Inst_3', 'MDXNET_1_9703', 'MDXNET_2_9682', 'MDXNET_3_9662', 'Inst_Main', 'MDXNET_Main', 'HT-Tuned', 'HT-Normal', 'HD_MMI', 'HT_6S', 'HP-1', 'HP-2', 'HP-Vocal-1', 'HP-Vocal-2', 'HP2-1', 'HP2-2', 'HP2-3', 'SP-2B-1', 'SP-2B-2', 'SP-3B-1', 'SP-4B-1', 'SP-4B-2', 'SP-MID-1', 'SP-MID-2').
            - `--karaoke_model` (mặc định: `MDX-Version-1`): Mô hình tách nhạc ('MDX-Version-1', 'MDX-Version-2', 'VR-Version-1', 'VR-Version-2').
            - `--reverb_model` (mặc định: `MDX-Reverb`): Mô hình tách nhạc ("MDX-Reverb", 'VR-Reverb', 'Echo-Aggressive', 'Echo-Normal').
            - `--denoise_model` (mặc định: `Normal`): Mô hình tách nhạc ('Lite', 'Normal').

        3. Cấu hình xử lí âm thanh:
            - `--shifts` (mặc định: `2`): Số lượng dự đoán.
            - `--batch_size` (mặc định: `1`): Kích thước lô.
            - `--overlap` (mặc định: `0.25`): Mức độ chồng lấn giữa các đoạn.
            - `--aggression` (mặc định: `5`): Cường độ chiết xuất thân chính.
            - `--hop_length` (mặc định: `1024`): Bước nhảy MDX khi xử lí.
            - `--window_size` (mặc định: `512`): Kích thước cửa sổ.
            - `--segments_size` (mặc định: `256`): Kích thước phân đoạn âm thanh.
            - `--post_process_threshold` (mặc định: `0.2`): Mức độ xử lí hậu kỳ sau khi tách nhạc.

        4. Cấu hình xử lí âm thanh khác:
            - `--enable_tta` (mặc định: `False`): Tăng cường suy luận.
            - `--enable_denoise` (mặc định: `False`): Khữ tách nhạc.
            - `--high_end_process` (mặc định: `False`): Xử lí dải cao.
            - `--enable_post_process` (mặc định: `False`): Hậu xử lí.
            - `--separate_backing` (mặc định: `False`): Tách bè giọng.
            - `--separate_reverb` (mặc định: `False`): Tách vang giọng.
    """)
    quit()
elif argv_is_allows[15] in argv:
    print("""Các tham số của --train:
        1. Cấu hình mô hình:
            - `--model_name` (bắt buộc): Tên mô hình.
            - `--rvc_version` (mặc định: `v2`): Phiên bản RVC (`v1`, `v2`).
            - `--model_author` (tùy chọn): Tác giả của mô hình.

        2. Cấu hình lưu:
            - `--save_every_epoch` (bắt buộc): Số kỷ nguyên giữa mỗi lần lưu.
            - `--save_only_latest` (mặc định: `True`): Chỉ lưu điểm mới nhất.
            - `--save_every_weights` (mặc định: `True`): Lưu tất cả trọng số của mô hình.

        3. Cấu hình huấn luyện:
            - `--total_epoch` (mặc định: `300`): Tổng số kỷ nguyên huấn luyện.
            - `--batch_size` (mặc định: `8`): Kích thước lô trong quá trình huấn luyện.

        4. Cấu hình thiết bị:
            - `--gpu` (mặc định: `0`): Chỉ định GPU để sử dụng (số hoặc `-` nếu không dùng GPU).
            - `--cache_data_in_gpu` (mặc định: `False`): Lưu dữ liệu vào GPU để tăng tốc.

        5. Cấu hình huấn luyện nâng cao:
            - `--pitch_guidance` (mặc định: `True`): Sử dụng hướng dẫn cao độ.
            - `--g_pretrained_path` (mặc định: ``): Đường dẫn đến trọng số G đã huấn luyện trước.
            - `--d_pretrained_path` (mặc định: ``): Đường dẫn đến trọng số D đã huấn luyện trước.
            - `--vocoder` (mặc định: `Default`): Bộ mã hóa được sử dụng (`Default`, `MRF-HiFi-GAN`, `RefineGAN`).
            - `--energy_use` (mặc định: `False`): Sử dụng năng lượng rms.

        6. Phát hiện huấn luyện quá mức:
            - `--overtraining_detector` (mặc định: `False`): Bật/tắt chế độ phát hiện huấn luyện quá mức.
            - `--overtraining_threshold` (mặc định: `50`): Ngưỡng để xác định huấn luyện quá mức.

        7. Xử lí dữ liệu:
            - `--cleanup` (mặc định: `False`): Dọn dẹp tệp huấn luyện cũ để tiến hành huấn luyện lại từ đầu.

        8. Tối ưu:
            - `--checkpointing` (mặc định: `False`): Bật/tắt checkpointing để tiết kiệm RAM.
            - `--deterministic` (mặc định: `False`): Khi bật sẽ sử dụng các thuật toán có tính xác định cao, đảm bảo rằng mỗi lần chạy cùng một dữ liệu đầu vào sẽ cho kết quả giống nhau.
            - `--benchmark` (mặc định: `False`): Khi bật sẽ thử nghiệm và chọn thuật toán tối ưu nhất cho phần cứng và kích thước cụ thể.
            - `--optimizer` (mặc định: `AdamW`): Trình tối ưu hóa được sử dụng (`AdamW`, `RAdam`, `AnyPrecisionAdamW`).
            - `--multiscale_mel_loss` (mặc định: `False`): So sánh phổ Mel của âm thanh thật và âm thanh giả ở nhiều thang độ khác nhau. Giúp mô hình học được chi tiết âm sắc, độ sáng và cấu trúc tần số tốt hơn, từ đó cải thiện chất lượng và độ tự nhiên của giọng nói đầu ra.
          
        9. Bộ tham chiếu:
            - `--use_custom_reference` (mặc định: `False`): Có tùy chỉnh bộ tham chiếu hay không.
            - `--reference_path` (mặc định: `False`): Đường dẫn đến bộ tham chiếu.
    """)
    quit()
elif argv_is_allows[18] in argv:
    print("""Các tham số của --create_reference:
        1. Đường dẫn tệp:
            - `--audio_path` (bắt buộc): Đường dẫn tệp âm thanh đầu vào.
            - `--reference_name` (mặc định: `reference`): Đường dẫn lưu bộ tham chiếu đầu ra.
          
        2. Cấu hình bộ tham chiếu:
            - `--pitch_guidance` (mặc định: `True`): Sử dụng hướng dẫn cao độ.
            - `--energy_use` (mặc định: `False`): Sử dụng năng lượng rms.
            - `--version` (mặc định: `v2`): Phiên bản RVC (`v1`, `v2`).

        3. Cấu hình nhúng:
            - `--embedder_model` (mặc định: `hubert_base`): Tên mô hình nhúng.
            - `--embedders_mode` (mặc định: `fairseq`): Chế độ nhúng (`fairseq`, `transformers`, `onnx`, `whisper`).
        
        4. Cấu hình F0:
            - `--f0_method` (mặc định: `rmvpe`): Phương pháp dự đoán F0 (`pm`, `dio`, `mangio-crepe-tiny`, `mangio-crepe-small`, `mangio-crepe-medium`, `mangio-crepe-large`, `mangio-crepe-full`, `crepe-tiny`, `crepe-small`, `crepe-medium`, `crepe-large`, `crepe-full`, `fcpe`, `fcpe-legacy`, `rmvpe`, `rmvpe-legacy`, `harvest`, `yin`, `pyin`, `swipe`).
            - `--f0_onnx` (mặc định: `False`): Có sử dụng phiên bản ONNX của F0 hay không.
            - `--f0_up_key` (mặc định: `0`): Điều chỉnh cao độ.
            - `--filter_radius` (mặc định: `3`): Độ mượt của đường F0.
            - `--f0_autotune` (mặc định: `False`): Có tự động điều chỉnh F0 hay không.
            - `--f0_autotune_strength` (mặc định: `1`): Cường độ hiệu chỉnh tự động F0.
            - `--f0_file` (mặc định: ``): Đường dẫn tệp F0 có sẵn.
            - `--proposal_pitch` (mặc định: `False`): Đề xuất cao độ thay vì điều chỉnh thủ công.
            - `--proposal_pitch_threshold` (mặc định: `0.0`): Ngưỡng tần số ước tính cao độ.
            - `--alpha` (mặc định: `0.5`): Ngưỡng trộn cao độ khi ước tính cao độ hybrid.
    """)
    quit()
elif argv_is_allows[16] in argv:
    print("""Sử dụng:
        1. `--help_audio_effects`: Trợ giúp về phần thêm hiệu ứng âm thanh.
        2. `--help_convert`: Trợ giúp về chuyển đổi âm thanh.
        3. `--help_create_dataset`: Trợ giúp về tạo dữ liệu huấn luyện.
        4. `--help_create_index`: Trợ giúp về tạo chỉ mục.
        5. `--help_extract`: Trợ giúp về trích xuất dữ liệu huấn luyện.
        6. `--help_preprocess`: Trợ giúp về xử lí trước dữ liệu.
        7. `--help_separate_music`: Trợ giúp về tách nhạc.
        8. `--help_train`: Trợ giúp về huấn luyện mô hình.
        9. `--help_create_reference`: Trợ giúp về tạo bộ tham chiếu.
    """)
    quit()

if __name__ == "__main__":
    import torch.multiprocessing as mp

    if "--train" in argv: mp.set_start_method("spawn")
    if "--preprocess" in argv or "--extract" in argv: mp.set_start_method("spawn", force=True)

    main()