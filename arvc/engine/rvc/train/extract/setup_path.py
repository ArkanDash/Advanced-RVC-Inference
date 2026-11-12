import os

def setup_paths(exp_dir, version = None, rms_extract = False):
    wav_path = os.path.join(exp_dir, "sliced_audios_16k")

    if rms_extract:
        out_path = os.path.join(exp_dir, "energy")
        os.makedirs(out_path, exist_ok=True)

        return wav_path, out_path

    if version:
        out_path = os.path.join(exp_dir, f"{version}_extracted")
        os.makedirs(out_path, exist_ok=True)

        return wav_path, out_path
    else:
        output_root1, output_root2 = os.path.join(exp_dir, "f0"), os.path.join(exp_dir, "f0_voiced")
        os.makedirs(output_root1, exist_ok=True); os.makedirs(output_root2, exist_ok=True)

        return wav_path, output_root1, output_root2