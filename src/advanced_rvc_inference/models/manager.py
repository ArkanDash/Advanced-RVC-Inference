"""
Enhanced Model Management Module

This module provides comprehensive model management capabilities including
download, caching, validation, and organization of voice models for RVC
(Retrieval-based Voice Conversion).

Features:
- Multi-source model downloads (HuggingFace, GitHub, local repositories)
- Model validation and metadata extraction
- Automatic organization and categorization
- Cache management and optimization
- Model fusion and conversion capabilities
- Vietnamese-RVC specific optimizations
- ContentVec and HubERT model management
"""

import os
import json
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import requests
import time
import shutil

# HuggingFace support
try:
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.utils import RepositoryNotFoundError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Model validation
try:
    import torch
    import onnx
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Git support
try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for voice conversion models."""
    name: str
    model_path: str
    index_path: Optional[str]
    model_type: str = "rvc_v1"  # rvc_v1, rvc_v2, onnx
    sample_rate: int = 44100
    embedder_type: str = "contentvec"  # contentvec, hubert, whisper, spin
    language: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    version: Optional[str] = None
    size_mb: float = 0.0
    download_count: int = 0
    rating: float = 0.0
    tags: List[str] = None
    source: str = "local"  # local, huggingface, github, custom
    last_used: Optional[float] = None
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        return cls(**data)


class EnhancedModelManager:
    """
    Enhanced Model Manager for RVC voice models.
    
    This class provides comprehensive model management including downloads,
    validation, organization, and optimization of voice conversion models.
    """
    
    def __init__(self, 
                 models_dir: Union[str, Path] = "models",
                 cache_dir: Optional[Union[str, Path]] = None,
                 max_cache_size_gb: float = 10.0):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory for storing models
            cache_dir: Directory for caching downloads (defaults to models_dir/cache)
            max_cache_size_gb: Maximum cache size in GB
        """
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.models_dir / "cache"
        self.max_cache_size_gb = max_cache_size_gb
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model storage
        self.models_metadata: Dict[str, ModelMetadata] = {}
        self.model_categories = {
            'vocal': self.models_dir / 'vocals',
            'instrumental': self.models_dir / 'instrumentals',
            'gender_converted': self.models_dir / 'gender_converted',
            'pitch_shifted': self.models_dir / 'pitch_shifted',
            'style_transfer': self.models_dir / 'style_transfer',
            'language_specific': self.models_dir / 'language_specific',
            'custom': self.models_dir / 'custom'
        }
        
        # Create category directories
        for category_dir in self.model_categories.values():
            category_dir.mkdir(exist_ok=True)
        
        # Load existing metadata
        self._load_metadata()
        
        # Download statistics
        self.download_stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_size_mb': 0.0
        }
        
        logger.info(f"Enhanced Model Manager initialized with {len(self.models_metadata)} models")
    
    def _load_metadata(self):
        """Load model metadata from files."""
        metadata_file = self.models_dir / "metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    for model_name, metadata_dict in data.items():
                        try:
                            self.models_metadata[model_name] = ModelMetadata.from_dict(metadata_dict)
                        except Exception as e:
                            logger.warning(f"Failed to load metadata for {model_name}: {e}")
                
                logger.info(f"Loaded metadata for {len(self.models_metadata)} models")
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
    
    def _save_metadata(self):
        """Save model metadata to files."""
        metadata_file = self.models_dir / "metadata.json"
        
        try:
            data = {}
            for model_name, metadata in self.models_metadata.items():
                data[model_name] = metadata.to_dict()
            
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def register_model(self, 
                      model_path: Union[str, Path],
                      name: Optional[str] = None,
                      category: str = 'custom',
                      **kwargs) -> bool:
        """
        Register a new model with metadata.
        
        Args:
            model_path: Path to the model file
            name: Model name (defaults to filename)
            category: Model category
            **kwargs: Additional metadata
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            if name is None:
                name = model_path.stem
            
            # Calculate file size
            size_mb = model_path.stat().st_size / (1024 * 1024)
            
            # Calculate checksum
            checksum = self._calculate_checksum(model_path)
            
            # Extract metadata
            metadata = self._extract_model_metadata(model_path)
            
            # Create metadata object
            model_metadata = ModelMetadata(
                name=name,
                model_path=str(model_path),
                index_path=kwargs.get('index_path'),
                model_type=kwargs.get('model_type', metadata.get('type', 'rvc_v1')),
                sample_rate=kwargs.get('sample_rate', metadata.get('sample_rate', 44100)),
                embedder_type=kwargs.get('embedder_type', metadata.get('embedder', 'contentvec')),
                language=kwargs.get('language', metadata.get('language')),
                description=kwargs.get('description'),
                author=kwargs.get('author'),
                version=kwargs.get('version'),
                size_mb=size_mb,
                tags=kwargs.get('tags', []),
                source=kwargs.get('source', 'local'),
                checksum=checksum
            )
            
            # Update metadata with extracted info
            model_metadata.description = model_metadata.description or metadata.get('description')
            model_metadata.author = model_metadata.author or metadata.get('author')
            
            # Add to registry
            self.models_metadata[name] = model_metadata
            
            # Move to appropriate category if needed
            if category in self.model_categories:
                target_category_dir = self.model_categories[category]
                if model_path.parent != target_category_dir:
                    target_path = target_category_dir / model_path.name
                    if not target_path.exists():
                        shutil.move(str(model_path), str(target_path))
                        model_metadata.model_path = str(target_path)
                
                # Move index file if exists
                if model_metadata.index_path:
                    index_path = Path(model_metadata.index_path)
                    if index_path.exists() and index_path.parent != target_category_dir:
                        target_index_path = target_category_dir / index_path.name
                        if not target_index_path.exists():
                            shutil.move(str(index_path), str(target_index_path))
                            model_metadata.index_path = str(target_index_path)
            
            self._save_metadata()
            logger.info(f"Registered model: {name} ({size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {model_path}: {e}")
            return False
    
    def download_model(self, 
                      source: str,
                      model_name: str,
                      category: str = 'custom',
                      progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        """
        Download a model from various sources.
        
        Args:
            source: Model source URL or identifier
            model_name: Name for the downloaded model
            category: Category to organize the model
            progress_callback: Callback for download progress (0.0 to 1.0)
            
        Returns:
            True if download successful, False otherwise
        """
        self.download_stats['total_downloads'] += 1
        
        try:
            if source.startswith('https://huggingface.co/'):
                return self._download_from_huggingface(source, model_name, category, progress_callback)
            elif source.startswith('https://github.com/'):
                return self._download_from_github(source, model_name, category, progress_callback)
            else:
                return self._download_custom(source, model_name, category, progress_callback)
                
        except Exception as e:
            self.download_stats['failed_downloads'] += 1
            logger.error(f"Failed to download model {model_name}: {e}")
            return False
    
    def _download_from_huggingface(self, 
                                  repo_id: str,
                                  model_name: str,
                                  category: str,
                                  progress_callback: Optional[Callable[[float], None]]) -> bool:
        """Download model from HuggingFace Hub."""
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace Hub not available. Install with: pip install huggingface_hub")
        
        try:
            target_category_dir = self.model_categories.get(category, self.model_categories['custom'])
            
            # List files in repository
            files = list_repo_files(repo_id)
            
            # Find model and index files
            model_files = [f for f in files if f.endswith(('.pth', '.onnx')) and 'G' not in f]
            index_files = [f for f in files if f.endswith('.index')]
            
            if not model_files:
                raise ValueError("No model files found in repository")
            
            # Download model file
            model_file = model_files[0]
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_file,
                cache_dir=str(self.cache_dir),
                resume_download=True
            )
            
            # Download index file if exists
            index_path = None
            if index_files:
                index_file = index_files[0]
                index_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=index_file,
                    cache_dir=str(self.cache_dir),
                    resume_download=True
                )
            
            # Move to category directory
            target_model_path = target_category_dir / Path(model_path).name
            if target_model_path.exists():
                target_model_path.unlink()
            shutil.move(model_path, target_model_path)
            
            if index_path:
                target_index_path = target_category_dir / Path(index_path).name
                if target_index_path.exists():
                    target_index_path.unlink()
                shutil.move(index_path, target_index_path)
            
            # Register model
            success = self.register_model(
                model_path=target_model_path,
                name=model_name,
                category=category,
                source='huggingface'
            )
            
            if index_path:
                self.register_model(
                    model_path=target_index_path,
                    name=f"{model_name}_index",
                    category=category,
                    source='huggingface'
                )
            
            if progress_callback:
                progress_callback(1.0)
            
            self.download_stats['successful_downloads'] += 1
            self.download_stats['total_size_mb'] += Path(target_model_path).stat().st_size / (1024 * 1024)
            
            logger.info(f"Downloaded model from HuggingFace: {model_name}")
            return success
            
        except Exception as e:
            logger.error(f"HuggingFace download failed: {e}")
            return False
    
    def _download_from_github(self, 
                             repo_url: str,
                             model_name: str,
                             category: str,
                             progress_callback: Optional[Callable[[float], None]]) -> bool:
        """Download model from GitHub repository."""
        if not GIT_AVAILABLE:
            raise ImportError("GitPython not available. Install with: pip install GitPython")
        
        try:
            target_category_dir = self.model_categories.get(category, self.model_categories['custom'])
            
            # Clone repository
            repo_dir = self.cache_dir / "github" / model_name
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            
            repo = git.Repo.clone_from(repo_url, repo_dir)
            
            # Find model files in repository
            model_files = []
            for ext in ['.pth', '.onnx', '.pt']:
                model_files.extend(repo_dir.glob(f"**/*{ext}"))
            
            if not model_files:
                logger.warning(f"No model files found in {repo_url}")
                return False
            
            # Download first model file found
            model_file = model_files[0]
            target_model_path = target_category_dir / model_file.name
            shutil.copy2(model_file, target_model_path)
            
            # Register model
            success = self.register_model(
                model_path=target_model_path,
                name=model_name,
                category=category,
                source='github'
            )
            
            if progress_callback:
                progress_callback(1.0)
            
            self.download_stats['successful_downloads'] += 1
            self.download_stats['total_size_mb'] += target_model_path.stat().st_size / (1024 * 1024)
            
            logger.info(f"Downloaded model from GitHub: {model_name}")
            return success
            
        except Exception as e:
            logger.error(f"GitHub download failed: {e}")
            return False
    
    def _download_custom(self, 
                        source_url: str,
                        model_name: str,
                        category: str,
                        progress_callback: Optional[Callable[[float], None]]) -> bool:
        """Download model from custom source."""
        try:
            target_category_dir = self.model_categories.get(category, self.model_categories['custom'])
            
            # Download file
            response = requests.get(source_url, stream=True)
            response.raise_for_status()
            
            target_path = target_category_dir / f"{model_name}.pth"
            
            with open(target_path, 'wb') as f:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress = downloaded / total_size
                            progress_callback(progress)
            
            # Register model
            success = self.register_model(
                model_path=target_path,
                name=model_name,
                category=category,
                source='custom'
            )
            
            self.download_stats['successful_downloads'] += 1
            self.download_stats['total_size_mb'] += target_path.stat().st_size / (1024 * 1024)
            
            logger.info(f"Downloaded model from custom source: {model_name}")
            return success
            
        except Exception as e:
            logger.error(f"Custom download failed: {e}")
            return False
    
    def _extract_model_metadata(self, model_path: Path) -> Dict[str, Any]:
        """Extract metadata from model file."""
        metadata = {}
        
        try:
            if model_path.suffix in ['.pth', '.pt']:
                if TORCH_AVAILABLE:
                    state_dict = torch.load(model_path, map_location='cpu')
                    
                    # Extract common metadata
                    if 'model' in state_dict:
                        model_info = state_dict['model']
                        if isinstance(model_info, dict):
                            metadata['type'] = model_info.get('type', 'rvc_v1')
                            metadata['sample_rate'] = model_info.get('sample_rate', 44100)
                            metadata['embedder'] = model_info.get('embedder', 'contentvec')
            
            elif model_path.suffix == '.onnx':
                if TORCH_AVAILABLE:
                    onnx_model = onnx.load(str(model_path))
                    metadata['type'] = 'onnx'
                    metadata['sample_rate'] = 44100  # Default for ONNX models
                    metadata['embedder'] = 'contentvec'  # Default for ONNX models
            
            # Extract filename-based metadata
            filename_lower = model_path.stem.lower()
            
            if 'v2' in filename_lower:
                metadata['type'] = 'rvc_v2'
            elif 'v1' in filename_lower:
                metadata['type'] = 'rvc_v1'
            
            if 'contentvec' in filename_lower:
                metadata['embedder'] = 'contentvec'
            elif 'hubert' in filename_lower:
                metadata['embedder'] = 'hubert'
            elif 'whisper' in filename_lower:
                metadata['embedder'] = 'whisper'
            elif 'spin' in filename_lower:
                metadata['embedder'] = 'spin'
            
            # Extract language from filename
            for lang in ['vietnamese', 'chinese', 'japanese', 'korean', 'english']:
                if lang in filename_lower:
                    metadata['language'] = lang
                    break
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Failed to extract metadata from {model_path}: {e}")
            return {}
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate checksum for {file_path}: {e}")
            return ""
    
    def validate_model(self, model_name: str) -> Tuple[bool, str]:
        """
        Validate a registered model.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if model_name not in self.models_metadata:
            return False, f"Model {model_name} not found"
        
        metadata = self.models_metadata[model_name]
        model_path = Path(metadata.model_path)
        
        try:
            # Check if model file exists
            if not model_path.exists():
                return False, "Model file not found"
            
            # Check file size
            if metadata.size_mb == 0:
                metadata.size_mb = model_path.stat().st_size / (1024 * 1024)
            
            # Validate model format
            if model_path.suffix in ['.pth', '.pt']:
                if TORCH_AVAILABLE:
                    try:
                        state_dict = torch.load(model_path, map_location='cpu')
                        if not isinstance(state_dict, dict):
                            return False, "Invalid PyTorch model format"
                    except Exception as e:
                        return False, f"Invalid PyTorch model: {e}"
            
            elif model_path.suffix == '.onnx':
                if TORCH_AVAILABLE:
                    try:
                        onnx.load(str(model_path))
                    except Exception as e:
                        return False, f"Invalid ONNX model: {e}"
            
            else:
                return False, f"Unsupported model format: {model_path.suffix}"
            
            # Validate index file if present
            if metadata.index_path:
                index_path = Path(metadata.index_path)
                if not index_path.exists():
                    return False, "Index file not found"
            
            return True, "Model is valid"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def get_models_by_category(self, category: str) -> List[ModelMetadata]:
        """Get all models in a specific category."""
        if category not in self.model_categories:
            return []
        
        category_dir = self.model_categories[category]
        models = []
        
        for metadata in self.models_metadata.values():
            if Path(metadata.model_path).parent == category_dir:
                models.append(metadata)
        
        return sorted(models, key=lambda x: x.name)
    
    def search_models(self, 
                     query: str = "",
                     category: Optional[str] = None,
                     embedder_type: Optional[str] = None,
                     language: Optional[str] = None) -> List[ModelMetadata]:
        """Search models by various criteria."""
        results = []
        
        for metadata in self.models_metadata.values():
            # Category filter
            if category and Path(metadata.model_path).parent != self.model_categories.get(category, category_dir):
                continue
            
            # Embedder type filter
            if embedder_type and metadata.embedder_type != embedder_type:
                continue
            
            # Language filter
            if language and metadata.language != language:
                continue
            
            # Text search
            if query:
                query_lower = query.lower()
                search_fields = [
                    metadata.name,
                    metadata.description or "",
                    metadata.author or "",
                    " ".join(metadata.tags)
                ]
                
                if not any(query_lower in field.lower() for field in search_fields):
                    continue
            
            results.append(metadata)
        
        return sorted(results, key=lambda x: (x.download_count, x.rating), reverse=True)
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model from the registry."""
        if model_name not in self.models_metadata:
            logger.warning(f"Model {model_name} not found in registry")
            return False
        
        try:
            metadata = self.models_metadata[model_name]
            
            # Remove files
            model_path = Path(metadata.model_path)
            if model_path.exists():
                model_path.unlink()
            
            if metadata.index_path:
                index_path = Path(metadata.index_path)
                if index_path.exists():
                    index_path.unlink()
            
            # Remove from registry
            del self.models_metadata[model_name]
            self._save_metadata()
            
            logger.info(f"Removed model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove model {model_name}: {e}")
            return False
    
    def get_download_statistics(self) -> Dict[str, Any]:
        """Get download statistics."""
        return self.download_stats.copy()
    
    def cleanup_cache(self) -> int:
        """Clean up cache directory by removing old files."""
        try:
            cache_files = list(self.cache_dir.rglob("*"))
            cache_files = [f for f in cache_files if f.is_file()]
            
            # Sort by modification time
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove files until cache size is under limit
            total_size = sum(f.stat().st_size for f in cache_files)
            max_size_bytes = self.max_cache_size_gb * 1024 * 1024 * 1024
            
            removed_count = 0
            
            for cache_file in cache_files:
                if total_size <= max_size_bytes:
                    break
                
                file_size = cache_file.stat().st_size
                cache_file.unlink()
                total_size -= file_size
                removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} cache files")
            return removed_count
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            return 0
    
    def export_model_list(self, filepath: Union[str, Path], format: str = "json"):
        """Export model list to file."""
        try:
            filepath = Path(filepath)
            
            if format.lower() == "json":
                data = []
                for metadata in self.models_metadata.values():
                    data.append(metadata.to_dict())
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            
            elif format.lower() == "csv":
                import csv
                with open(filepath, 'w', newline='') as f:
                    if self.models_metadata:
                        writer = csv.DictWriter(f, fieldnames=self.models_metadata[list(self.models_metadata.keys())[0]].to_dict().keys())
                        writer.writeheader()
                        for metadata in self.models_metadata.values():
                            writer.writerow(metadata.to_dict())
            
            logger.info(f"Exported model list to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export model list: {e}")
            return False


# Global model manager instance
_model_manager = None

def get_model_manager(models_dir: Union[str, Path] = "models",
                     cache_dir: Optional[Union[str, Path]] = None) -> EnhancedModelManager:
    """Get or create global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = EnhancedModelManager(
            models_dir=models_dir,
            cache_dir=cache_dir
        )
    return _model_manager