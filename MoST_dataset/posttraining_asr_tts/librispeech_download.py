import os
import urllib.request
import tarfile
from tqdm import tqdm
from typing import Optional, List, Dict

_DL_URL = "http://www.openslr.org/resources/12/"

# Match the exact URL structure from the reference implementation
_DL_URLS = {
    "clean": {
        "dev": _DL_URL + "dev-clean.tar.gz",
        "test": _DL_URL + "test-clean.tar.gz",
        "train.100": _DL_URL + "train-clean-100.tar.gz",
        "train.360": _DL_URL + "train-clean-360.tar.gz",
    },
    "other": {
        "test": _DL_URL + "test-other.tar.gz",
        "dev": _DL_URL + "dev-other.tar.gz",
        "train.500": _DL_URL + "train-other-500.tar.gz",
    },
    "all": {
        "dev.clean": _DL_URL + "dev-clean.tar.gz",
        "dev.other": _DL_URL + "dev-other.tar.gz",
        "test.clean": _DL_URL + "test-clean.tar.gz",
        "test.other": _DL_URL + "test-other.tar.gz",
        "train.clean.100": _DL_URL + "train-clean-100.tar.gz",
        "train.clean.360": _DL_URL + "train-clean-360.tar.gz",
        "train.other.500": _DL_URL + "train-other-500.tar.gz",
    },
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_librispeech(download_dir: str, subset: str = "all", download_urls: Optional[List[str]] = None):
    """
    Download LibriSpeech datasets to specified directory.
    
    Args:
        download_dir (str): Directory to download the dataset to
        subset (str): Which subset to download - 'clean', 'other', or 'all'. Defaults to 'all'
        download_urls (list, optional): List of specific URLs to download. 
            If provided, overrides the subset parameter. Defaults to None.
    """
    if subset not in _DL_URLS:
        raise ValueError(f"Invalid subset '{subset}'. Must be one of: {list(_DL_URLS.keys())}")
    
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Determine which URLs to download
    urls_to_download: Dict[str, str] = {}
    if download_urls:
        # Filter and validate provided URLs
        for url in download_urls:
            found = False
            for subset_urls in _DL_URLS.values():
                for name, dl_url in subset_urls.items():
                    if url == dl_url:
                        urls_to_download[name] = url
                        found = True
                        break
                if found:
                    break
            if not found:
                print(f"Warning: Invalid URL '{url}' - skipping")
    else:
        urls_to_download = _DL_URLS[subset]
    
    # Download and extract each dataset
    for name, url in urls_to_download.items():
        filename = url.split("/")[-1]
        filepath = os.path.join(download_dir, filename)
        
        # Download if file doesn't exist
        if not os.path.exists(filepath):
            print(f"Downloading {name} from {url}...")
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                urllib.request.urlretrieve(url, filepath, reporthook=t.update_to)
        
        # Extract if not already extracted
        extract_dir = os.path.join(download_dir, filename[:-7])  # Remove .tar.gz
        if not os.path.exists(extract_dir):
            print(f"Extracting {filename}...")
            with tarfile.open(filepath) as tar:
                tar.extractall(path=download_dir)
            print(f"Extracted to {extract_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download LibriSpeech dataset")
    parser.add_argument("--download_dir", type=str, required=True,
                      help="Directory to download the dataset to")
    parser.add_argument("--subset", type=str, default="all", choices=["clean", "other", "all"],
                      help="Which subset to download")
    parser.add_argument("--urls", nargs="+", default=None,
                      help="Specific URLs to download (overrides subset)")
    
    args = parser.parse_args()
    download_librispeech(args.download_dir, args.subset, args.urls)