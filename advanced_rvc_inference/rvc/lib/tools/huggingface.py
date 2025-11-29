import os
import requests
from tqdm import tqdm
from pathlib import Path


def HF_download_file(url: str, local_path: str):
    """
    Download a file from Hugging Face or other URL to a local path.
    
    Args:
        url (str): The URL to download from
        local_path (str): The local path to save the file to
    """
    # Ensure the directory exists
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Decode any encoded parts of the URL if needed (the ROT13 encoding in the original code)
    if "uggcf://" in url:
        # This is ROT13 encoded, decode it
        decoded_url = url.replace("uggcf://", "https://").replace("uhttv://", "http://")
        # ROT13 decode the remaining content
        import codecs
        url = codecs.decode(decoded_url, "rot13")
    
    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as f, tqdm(
        desc=local_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)

    print(f"Downloaded: {local_path}")