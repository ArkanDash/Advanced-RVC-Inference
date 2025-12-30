import os
import shutil
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from torch.utils.tensorboard import SummaryWriter

def clean_tfevents():
    input_path = input("Enter the full path or filename of the tfevents file: ").strip()
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return

    try:
        epochs_to_keep = int(input("How many full epochs to keep? (e.g., 2): "))
        steps_per_epoch = int(input("How many steps are in 1 epoch? (e.g., 2687): "))
    except ValueError:
        print("Error: Please enter valid integers.")
        return

    epoch_end_step = epochs_to_keep * steps_per_epoch
    standard_metric_cutoff = (epoch_end_step // 50) * 50

    special_tag = "Metric/Mel_Spectrogram_Similarity"

    print(f"\n--- Alignment Plan ---")
    print(f"Epoch End Step: {epoch_end_step} (Keeping '{special_tag}' up to here)")
    print(f"Standard Metric Cutoff: {standard_metric_cutoff} (Rounding down to nearest 50)")
    print(f"----------------------")

    base_dir = os.path.dirname(os.path.abspath(input_path))
    original_filename = os.path.basename(input_path)
    backup_folder = os.path.join(base_dir, "old_tfevents_backup")
    temp_output_dir = os.path.join(base_dir, "temp_trim_process")

    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)

    writer = SummaryWriter(log_dir=temp_output_dir)
    loader = EventFileLoader(input_path)

    count = 0
    removed = 0

    for event in loader.Load():
        keep_event = False

        is_special = False
        if event.summary:
            for value in event.summary.value:
                if value.tag == special_tag:
                    is_special = True
                    break

        if is_special:
            if event.step <= epoch_end_step:
                keep_event = True
        else:
            if event.step <= standard_metric_cutoff:
                keep_event = True

        if event.HasField('file_version'):
            continue

        if keep_event:
            writer.file_writer.add_event(event)
            count += 1
        else:
            removed += 1

    writer.close()

    new_file_name = os.listdir(temp_output_dir)[0]
    new_file_path = os.path.join(temp_output_dir, new_file_name)

    shutil.move(input_path, os.path.join(backup_folder, original_filename))
    shutil.move(new_file_path, os.path.join(base_dir, original_filename))
    os.rmdir(temp_output_dir)

    print(f"Success! Kept {count} events, removed {removed} extra steps/metrics.")

if __name__ == "__main__":
    clean_tfevents()