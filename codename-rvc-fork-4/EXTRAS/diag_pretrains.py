import torch

def prompt_path(name):
    path = input(f"Enter path for {name} checkpoint: ").strip()
    if not path:
        raise ValueError(f"Path for {name} cannot be empty.")
    return path

def check_special_keys(ckpt, ckpt_name):
    print(f"\n🔎 Checking special keys in {ckpt_name}:")
    # Check for actual discriminator keys seen in your checkpoints
    special_keys = ["mpd", "cqt"]
    optim_keys_candidates = ["optim_d", "optimizer", "optim", "optimizer_state_dict"]

    found_special = [k for k in special_keys if k in ckpt]
    print(f" - Found discriminator nets: {found_special if found_special else 'None'}")

    found_optim = [k for k in optim_keys_candidates if k in ckpt]
    print(f" - Found optimizer keys: {found_optim if found_optim else 'None'}")

def compare_keys(name1, ckpt1, name2, ckpt2):
    print(f"\n🔍 Top-level key comparison: {name1} vs {name2}")
    keys1 = set(ckpt1.keys())
    keys2 = set(ckpt2.keys())

    print(f"✅ Keys in {name1}: {keys1}")
    print(f"✅ Keys in {name2}: {keys2}")

    missing = keys1 - keys2
    extra = keys2 - keys1

    if missing:
        print(f"❌ Missing in {name2}: {missing}")
    if extra:
        print(f"❗ Extra in {name2}: {extra}")
    if not missing and not extra:
        print("✅ Keys match perfectly.")

def deep_compare(name1, d1, name2, d2, component_name):
    print(f"\n🔬 Comparing model component: {component_name} ({name1} vs {name2})")
    keys1 = set(d1.keys())
    keys2 = set(d2.keys())

    missing = keys1 - keys2
    extra = keys2 - keys1

    print(f"🔑 Param keys in {name1}: {len(keys1)}, in {name2}: {len(keys2)}")
    if missing:
        print(f"❌ Missing in {name2}: {missing}")
    if extra:
        print(f"❗ Extra in {name2}: {extra}")

    shape_mismatch = []
    for k in keys1 & keys2:
        if d1[k].shape != d2[k].shape:
            shape_mismatch.append((k, d1[k].shape, d2[k].shape))

    if shape_mismatch:
        print("⚠️ Shape mismatches:")
        for k, s1, s2 in shape_mismatch[:10]:
            print(f" - {k}: {name1}={s1}, {name2}={s2}")
    else:
        print("✅ All matching keys have same shape.")

def main():
    full_path = prompt_path("full")
    clean_path = prompt_path("no-optim")
    finetune_path = prompt_path("finetuned")

    ckpt_full = torch.load(full_path, map_location="cpu")
    ckpt_clean = torch.load(clean_path, map_location="cpu")
    ckpt_finetune = torch.load(finetune_path, map_location="cpu")

    # Check special keys in all checkpoints
    check_special_keys(ckpt_full, "full")
    check_special_keys(ckpt_clean, "clean")
    check_special_keys(ckpt_finetune, "finetuned")

    # Run top-level comparisons
    compare_keys("full", ckpt_full, "clean", ckpt_clean)
    compare_keys("full", ckpt_full, "finetuned", ckpt_finetune)

    # Deep comparison for "mpd" and "cqt"
    for module in ["mpd", "cqt"]:
        deep_compare("full", ckpt_full.get(module, {}), "clean", ckpt_clean.get(module, {}), module)
        deep_compare("full", ckpt_full.get(module, {}), "finetune", ckpt_finetune.get(module, {}), module)

    # === New section: compare clean vs finetuned ===
    print("\n" + "="*30)
    print("🔄 Comparing clean vs finetuned")
    print("="*30)

    compare_keys("clean", ckpt_clean, "finetuned", ckpt_finetune)
    for module in ["mpd", "cqt"]:
        deep_compare("clean", ckpt_clean.get(module, {}), "finetuned", ckpt_finetune.get(module, {}), module)

    print("\n✅ Finished all comparisons.")

if __name__ == "__main__":
    main()
