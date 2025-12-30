import torch
import argparse
import os

def strip_for_finetuning(input_path, output_path=None, new_lr=1e-4):
    assert os.path.isfile(input_path), f"Checkpoint file not found: {input_path}"
    
    checkpoint = torch.load(input_path, map_location="cpu")

    # Remove optimizer
    if "optimizer" in checkpoint:
        print("🧹 Removing optimizer state...")
        del checkpoint["optimizer"]
    else:
        print("ℹ️ No optimizer state found.")

    # Set learning rate to 1e-4 (float)
    checkpoint["learning_rate"] = float(new_lr)
    print(f"🎯 Set learning_rate → {checkpoint['learning_rate']}")

    # Save to new file (or overwrite)
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_finetune{ext}"

    torch.save(checkpoint, output_path)
    print(f"✅ Saved finetune-ready checkpoint to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip optimizer and set learning rate for fine-tuning.")
    parser.add_argument("input_path", help="Path to input checkpoint (.pth)")
    parser.add_argument("--output", "-o", help="Optional output path (default: <input>_finetune.pth)")
    parser.add_argument("--lr", type=float, default=1e-4, help="New learning rate to set (default: 1e-4)")

    args = parser.parse_args()
    strip_for_finetuning(args.input_path, args.output, args.lr)
