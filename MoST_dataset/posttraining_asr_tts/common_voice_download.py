import os
import json
import hashlib
import requests
import tarfile
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, List, Tuple

class CommonVoiceDownloader:
    BASE_URL = "https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0/resolve/main/"
    
    def __init__(self, output_dir: str, language: str = None, token: str = None):
        self.output_dir = Path(output_dir)
        self.language = language
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.progress_file = self.output_dir / "download_progress.json"
        self.chunk_size = 1024 * 1024  # 1MB chunks for downloads
        self.progress = {}  # Initialize empty progress
        
    def _load_progress(self) -> Dict:
        """Load progress from file if it exists"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Warning: Progress file corrupted, starting fresh")
        return {}

    def _save_progress(self, new_entries: Dict):
        """Save new progress entries"""
        self.progress.update(new_entries)
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def _get_file_size(self, url: str) -> Optional[int]:
        """Get file size from server"""
        try:
            response = requests.head(url, headers=self.headers)
            return int(response.headers.get('content-length', 0))
        except:
            return None

    def _verify_content(self, directory: Path, pattern: str = "*.mp3") -> bool:
        """Verify that a directory contains expected content recursively"""
        try:
            files = list(directory.glob(pattern))
            return len(files) > 0
        except Exception:
            return False

    def _is_extracted(self, tar_path: Path, extract_path: Path) -> bool:
        """Check if a tar file has been successfully extracted"""
        # If the directory exists and contains mp3 files, consider it extracted
        if extract_path.exists() and self._verify_content(extract_path):
            extraction_key = f"extract:{tar_path}:{extract_path}"
            # Record this as extracted if not already in progress
            if extraction_key not in self.progress:
                self._save_progress({
                    extraction_key: {
                        'completed': True,
                        'path': str(extract_path)
                    }
                })
            return True
        return False

    def _download_file(self, url: str, output_path: Path, desc: str = None) -> bool:
        """Download a file with progress bar and resume capability"""
        # Skip if file exists and is not empty
        if output_path.exists() and output_path.stat().st_size > 0:
            if not output_path.name.endswith('.tar'):  # For non-tar files
                print(f"{desc}: Using existing file")
                return True
            elif self._is_extracted(output_path, output_path.parent):
                print(f"{desc}: Already extracted")
                return True

        total_size = self._get_file_size(url)
        if total_size is None:
            print(f"Warning: Could not get file size for {url}")
            return False

        mode = 'ab' if output_path.exists() else 'wb'
        existing_size = output_path.stat().st_size if output_path.exists() else 0

        if existing_size > total_size:
            output_path.unlink()
            existing_size = 0
            mode = 'wb'

        headers = self.headers.copy()
        if existing_size > 0:
            headers['Range'] = f'bytes={existing_size}-'

        try:
            response = requests.get(url, stream=True, headers=headers)
            
            if response.status_code == 200 and existing_size > 0:
                output_path.unlink()
                existing_size = 0
                response = requests.get(url, stream=True, headers=self.headers)

            response.raise_for_status()

            with open(output_path, mode) as file, \
                 tqdm(
                    desc=desc,
                    initial=existing_size,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    size = file.write(chunk)
                    pbar.update(size)

            file_key = f"{url}:{output_path}"
            self._save_progress({
                file_key: {
                    'size': total_size,
                    'path': str(output_path)
                }
            })
            return True

        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return False

    def _extract_tar(self, tar_path: Path, extract_path: Path) -> bool:
        """Extract tar file with progress tracking"""
        # Create shard-specific directory
        shard_dir = extract_path / tar_path.stem
        if self._is_extracted(tar_path, shard_dir):
            return True

        try:
            with tarfile.open(tar_path, 'r') as tar:
                members = tar.getmembers()
                with tqdm(members, desc=f"Extracting {tar_path.name}") as pbar:
                    for member in pbar:
                        tar.extract(member, path=shard_dir)

            # Record successful extraction
            extraction_key = f"extract:{tar_path}:{shard_dir}"
            self._save_progress({
                extraction_key: {
                    'completed': True,
                    'path': str(shard_dir)
                }
            })
            
            # Remove tar file after successful extraction
            tar_path.unlink()
            return True

        except Exception as e:
            print(f"Error extracting {tar_path}: {str(e)}")
            return False

    def _get_download_tasks(self, language: str, n_shards: dict) -> List[Tuple[str, Path, str]]:
        """Generate download tasks for a language, skipping completed ones"""
        tasks = []
        audio_dir = self.output_dir / 'audio' / language
        transcript_dir = self.output_dir / 'transcript' / language

        splits = ["train", "test", "dev", "other", "validated", "invalidated"]

        # Add transcript tasks
        for split in splits:
            transcript_path = transcript_dir / f"{split}.tsv"
            if transcript_path.exists() and transcript_path.stat().st_size > 0:
                print(f"Skipping existing transcript: {language}_{split}")
                continue

            transcript_url = f"{self.BASE_URL}transcript/{language}/{split}.tsv"
            tasks.append((
                transcript_url,
                transcript_path,
                f"Downloading {language} {split} transcript"
            ))

        # Add audio tasks
        for split in splits:
            split_dir = audio_dir / split
            n_shards_split = n_shards.get(language, {}).get(split, 0)

            for shard_idx in range(n_shards_split):
                tar_path = split_dir / f"{language}_{split}_{shard_idx}.tar"
                shard_dir = split_dir / f"{language}_{split}_{shard_idx}"  # Specific directory for this shard
                
                # Skip if already extracted
                if self._is_extracted(tar_path, shard_dir):
                    print(f"Skipping already extracted shard: {language}_{split}_{shard_idx}")
                    continue

                audio_url = f"{self.BASE_URL}audio/{language}/{split}/{language}_{split}_{shard_idx}.tar"
                tasks.append((
                    audio_url,
                    tar_path,
                    f"Downloading {language} {split} shard {shard_idx+1}/{n_shards_split}"
                ))

        return tasks

    def _create_directories(self, language):
        """Create necessary directories for a language"""
        audio_dir = self.output_dir / 'audio' / language
        transcript_dir = self.output_dir / 'transcript' / language
        
        # Create main directories
        audio_dir.mkdir(parents=True, exist_ok=True)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Create split directories
        splits = ["train", "test", "dev", "other", "validated", "invalidated"]
        for split in splits:
            (audio_dir / split).mkdir(exist_ok=True)

        return audio_dir, transcript_dir

    def download_dataset(self):
        """Download and extract the complete dataset"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing progress
        self.progress = self._load_progress()
        
        # Download n_shards.json
        n_shards_url = f"{self.BASE_URL}n_shards.json"
        n_shards_path = self.output_dir / "n_shards.json"
        
        if not n_shards_path.exists():
            self._download_file(n_shards_url, n_shards_path, "Downloading n_shards.json")
        
        with open(n_shards_path, 'r') as f:
            n_shards = json.load(f)

        # Determine languages to download
        if self.language:
            languages = [self.language]
        else:
            languages = list(n_shards.keys())
            print(f"Found {len(languages)} languages: {', '.join(languages)}")

        # Process each language
        for lang in languages:
            print(f"\nProcessing language: {lang}")
            self._create_directories(lang)

            # Get tasks for this language
            tasks = self._get_download_tasks(lang, n_shards)
            
            if not tasks:
                print(f"All files for language {lang} are already downloaded and extracted")
                continue

            # Process downloads sequentially
            for url, output_path, desc in tasks:
                if self._download_file(url, output_path, desc):
                    if str(output_path).endswith('.tar'):
                        self._extract_tar(output_path, output_path.parent)

def main():
    parser = argparse.ArgumentParser(description='Download Common Voice dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save the dataset')
    parser.add_argument('--language', type=str, default=None,
                      help='Language code to download (e.g., "en"). If not specified, downloads all languages')
    parser.add_argument('--token', type=str, required=True,
                      help='Hugging Face API token')

    args = parser.parse_args()

    downloader = CommonVoiceDownloader(
        args.output_dir,
        args.language,
        args.token
    )
    downloader.download_dataset()

if __name__ == "__main__":
    main()