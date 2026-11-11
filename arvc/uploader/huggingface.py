"""
HuggingFace Hub integration for uploading and loading training results.

Upload structure on the Hub (model repo):
    <repo-id>/
        README.md
        <ModelName>/          <- one folder per model
            G_latest.pth
            D_latest.pth
            added_<name>.index
            ... (other selected training files)

Loading mirrors the same structure: files are downloaded into
``configs["logs_path"]/<ModelName>/`` so training can resume or the
model can be used for inference right away.
"""

import os
from pathlib import Path

from arvc.utils.feedback import gr_info, gr_warning, gr_error
from arvc.utils.variables import configs, logger

# File extensions that are meaningful training artifacts. Intermediate
# dataset folders (sliced audios, extracted features, f0 caches...) are
# excluded from the default selection because they can be huge and are
# regenerated locally during preprocessing.
DEFAULT_UPLOAD_EXTENSIONS = (".pth", ".index", ".json", ".onnx", ".txt", ".ckpt")
# Folder names inside a model directory that hold intermediate data only.
INTERMEDIATE_DIRS = (
    "sliced_audios", "sliced_audios_16k", "f0", "f0_voiced", "energy",
    "v1_extracted", "v2_extracted", "eval", "logs",
)


def _logs_root():
    """Return the logs directory as a Path."""
    return Path(configs["logs_path"])


def list_local_training_models():
    """List local trained model names (subdirectories of the logs path).

    Returns:
        Sorted list of model directory names.
    """
    root = _logs_root()
    if not root.exists():
        return []
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and not entry.name.startswith((".", "_")) and entry.name != "mute"
    )


def list_model_files(model_name, essentials_only=True):
    """List training files that belong to a local model.

    Args:
        model_name: Name of the model folder under the logs path.
        essentials_only: If True, exclude intermediate dataset folders
            (sliced audios, extracted features, f0 caches) from the list.

    Returns:
        Sorted list of file paths relative to the model folder.
    """
    model_dir = _logs_root() / model_name
    if not model_dir.exists() or not model_name:
        return []

    files = []
    for root, dirs, names in os.walk(model_dir):
        rel_root = Path(root).relative_to(model_dir)

        # Skip intermediate folders when only essentials are requested
        if essentials_only:
            parts = set(rel_root.parts)
            if parts & set(INTERMEDIATE_DIRS):
                continue

        for name in sorted(names):
            rel_path = str(rel_root / name) if str(rel_root) != "." else name
            if essentials_only and not name.lower().endswith(DEFAULT_UPLOAD_EXTENSIONS):
                continue
            files.append(rel_path.replace(os.sep, "/"))

    return sorted(files)


def _default_repo_id(hf_repo, hf_token, model_name):
    """Resolve the target repo id, defaulting to <username>/<model_name>."""
    if hf_repo:
        return hf_repo.strip().strip("/")
    try:
        from huggingface_hub import HfApi
        user = HfApi(token=hf_token or None).whoami().get("name")
    except Exception:
        user = None
    if not user:
        return None
    return f"{user}/{model_name}"


def hf_upload_training(model_name, selected_files, hf_token=None, hf_repo=None, private=False):
    """Upload a model's training files to the HuggingFace Hub.

    Files are uploaded under ``<ModelName>/`` inside the target model
    repository, producing the structure::

        <repo-id>/[ModelName]/<training files>

    Args:
        model_name: Local model (folder under the logs path) to upload.
        selected_files: Relative file paths (from list_model_files) to upload.
        hf_token: HuggingFace token with write access. If None, the locally
            cached login token is used.
        hf_repo: Target repo id (e.g. "username/my-model"). Defaults to
            "<your-username>/<model_name>".
        private: Whether to create the repository as private.

    Returns:
        Status message string.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        return gr_error("huggingface_hub is not installed. Run: pip install huggingface_hub")

    if not model_name:
        return gr_warning("Please select a model to upload.")

    model_dir = _logs_root() / model_name
    if not model_dir.exists():
        return gr_warning(f"Model folder not found: {model_dir}")

    if not selected_files:
        return gr_warning("Please select at least one training file to upload.")

    # Drop files that do not exist locally anymore
    valid_files = [f for f in selected_files if (model_dir / f).is_file()]
    if not valid_files:
        return gr_warning("None of the selected files exist on disk anymore. Rescan the file list.")
    skipped = len(selected_files) - len(valid_files)

    repo_id = _default_repo_id(hf_repo, hf_token, model_name)
    if not repo_id:
        return gr_warning(
            "Could not determine the target repository. Provide a repo id like 'username/model-name' "
            "or log in with your HuggingFace token."
        )

    gr_info(f"Uploading training files to HuggingFace Hub: {repo_id} ({model_name}/)")

    try:
        api = HfApi(token=hf_token or None)
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

        for rel_path in valid_files:
            api.upload_file(
                path_or_fileobj=str(model_dir / rel_path),
                path_in_repo=f"{model_name}/{rel_path}",
                repo_id=repo_id,
                repo_type="model",
            )
            gr_info(f"Uploaded: {model_name}/{rel_path}")

        # Auto-generate a README so the repo page stays informative
        readme = (
            "---\n"
            "license: mit\n"
            "tags:\n- rvc\n- voice-cloning\n- advanced-rvc\n---\n\n"
            f"# {model_name}\n\n"
            "Training results uploaded with "
            "[Advanced-RVC-Inference](https://github.com/ArkanDash/Advanced-RVC-Inference).\n\n"
            "## Files\n\n" + "\n".join(f"- `{model_name}/{f}`" for f in valid_files) + "\n"
        )
        api.upload_file(
            path_or_fileobj=readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )

        url = f"https://huggingface.co/{repo_id}"
        gr_info(f"Successfully uploaded training files to: {url}")
        status = f"Uploaded {len(valid_files)} file(s) to {url} under {model_name}/"
        if skipped:
            status += f" ({skipped} missing file(s) skipped)"
        return status

    except Exception as e:
        logger.error(f"Failed to upload training files to HuggingFace Hub: {e}")
        return gr_error(f"Failed to upload to HuggingFace Hub: {e}")


def hf_list_repo_files(hf_repo, hf_token=None):
    """List all files inside a HuggingFace model repository.

    Args:
        hf_repo: Repository id (e.g. "username/my-model").
        hf_token: Optional token (needed for private repositories).

    Returns:
        (file_list, error_message) tuple. Exactly one of the two is None.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None, gr_error("huggingface_hub is not installed. Run: pip install huggingface_hub")

    if not hf_repo:
        return None, gr_warning("Please provide a HuggingFace repository id (e.g. 'username/model-name').")

    try:
        files = HfApi(token=hf_token or None).list_repo_files(repo_id=hf_repo.strip().strip("/"), repo_type="model")
        files = [f for f in files if not f.startswith(".")]  # hide .gitattributes etc.
        return sorted(files), None
    except Exception as e:
        logger.error(f"Failed to list files of {hf_repo}: {e}")
        return None, gr_error(f"Could not list repository files: {e}")


def hf_load_training(hf_repo, hf_token=None, model_name=None, selected_files=None):
    """Load training files from a HuggingFace model repository into the app.

    Files are placed under ``configs["logs_path"]/<ModelName>/`` mirroring
    the upload structure, so the model can be used or training resumed.

    Args:
        hf_repo: Repository id (e.g. "username/my-model").
        hf_token: Optional token (needed for private repositories).
        model_name: Destination model name. Defaults to the repo name part
            of hf_repo.
        selected_files: Optional list of repo file paths to download.
            Defaults to every file (except metadata) in the repository.

    Returns:
        Status message string.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return gr_error("huggingface_hub is not installed. Run: pip install huggingface_hub")

    if not hf_repo:
        return gr_warning("Please provide a HuggingFace repository id (e.g. 'username/model-name').")

    repo_id = hf_repo.strip().strip("/")
    dest_name = (model_name or "").strip() or repo_id.split("/")[-1]
    dest_root = _logs_root() / dest_name
    dest_root.mkdir(parents=True, exist_ok=True)

    files, err = hf_list_repo_files(repo_id, hf_token)
    if err is not None:
        return err

    if selected_files:
        wanted = [f for f in selected_files if f in files]
        if not wanted:
            return gr_warning("None of the selected files exist in the repository. Refresh the file list.")
    else:
        # Skip repo metadata files by default
        wanted = [f for f in files if os.path.basename(f) != "README.md" and not f.startswith(".")]

    if not wanted:
        return gr_warning("The repository contains no downloadable files.")

    gr_info(f"Loading {len(wanted)} file(s) from {repo_id} into {dest_root}")

    downloaded = []
    try:
        for rel_path in wanted:
            hf_hub_download(
                repo_id=repo_id,
                filename=rel_path,
                token=hf_token or None,
                repo_type="model",
                local_dir=str(dest_root),
            )
            downloaded.append(rel_path)
            gr_info(f"Downloaded: {rel_path}")

        gr_info(f"Successfully loaded {len(downloaded)} file(s) from {repo_id}")
        return f"Loaded {len(downloaded)} file(s) into '{dest_name}'. You can now use or resume this model locally."

    except Exception as e:
        logger.error(f"Failed to load training files from HuggingFace Hub: {e}")
        return gr_error(f"Failed to load from HuggingFace Hub: {e}")
