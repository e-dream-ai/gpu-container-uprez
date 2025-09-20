import logging
from pathlib import Path
from typing import Dict, Any
import torch

logger = logging.getLogger(__name__)


class ModelLoader:
    
    def __init__(self):
        self.device = self._get_device()
        self.loaded_models: Dict[str, Any] = {}
        self.model_paths = {
            'realesrgan': Path('/opt/models/realesrgan/RealESRGAN_x2plus.pth'),
            'rife': Path('/opt/models/rife/rife4.26.pth')
        }
        logger.info(f"ModelLoader initialized with device: {self.device}")
    
    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            return device
        else:
            device = torch.device('cpu')
            logger.warning("CUDA not available, using CPU")
            return device
    
    def load_realesrgan_model(
        self, 
        upscale_factor: int = 2, 
        tile_size: int = 512, 
        tile_padding: int = 10
    ) -> Any:
        if upscale_factor != 2:
            logger.warning(f"Unsupported upscale_factor={upscale_factor}. Falling back to 2x.")
            upscale_factor = 2

        model_key = f'realesrgan_x{upscale_factor}'
        
        if model_key in self.loaded_models:
            logger.debug(f"Real-ESRGAN model already loaded: {model_key}")
            return self.loaded_models[model_key]
        
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            logger.info(f"Loading Real-ESRGAN model: {model_key}")

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2
            )
            model_path = self.model_paths['realesrgan']
            
            upsampler = RealESRGANer(
                scale=2,
                model_path=str(model_path),
                model=model,
                tile=tile_size,
                tile_pad=tile_padding,
                pre_pad=0,
                half=True if self.device.type == 'cuda' else False,
                gpu_id=0 if self.device.type == 'cuda' else None
            )
            
            self.loaded_models[model_key] = upsampler
            logger.info(f"Real-ESRGAN model loaded successfully: {model_key}")
            
            return upsampler
            
        except Exception as e:
            logger.error(f"Failed to load Real-ESRGAN model: {e}")
            raise RuntimeError(f"Real-ESRGAN model loading failed: {e}")
    
    def load_rife_model(self) -> Any:
        model_key = 'rife'
        
        if model_key in self.loaded_models:
            logger.debug("RIFE model already loaded")
            return self.loaded_models[model_key]
        
        try:
            import sys
            sys.path.append('/opt/rife-repo')
            
            from model.RIFE_HDv3 import Model
            
            logger.info("Loading RIFE model")
            
            model = Model()

            weights_root = Path('/opt/models/rife')
            train_log_runtime = weights_root / 'train_log'
            repo_train_log = Path('/opt/rife-repo/train_log')

            weight_path: Path | None = None

            try:
                import zipfile
                zip_candidates = list(weights_root.glob('*.zip'))
                for zip_file in zip_candidates:
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        # Extract into /opt/models/rife/train_log
                        logger.info(f"Extracting RIFE weights from zip: {zip_file}")
                        train_log_runtime.mkdir(parents=True, exist_ok=True)
                        zf.extractall(train_log_runtime)
                        # If zip contains nested 'train_log', flatten once
                        nested = train_log_runtime / 'train_log'
                        if nested.exists() and nested.is_dir():
                            for item in nested.iterdir():
                                target = train_log_runtime / item.name
                                if not target.exists():
                                    item.rename(target)
                            nested.rmdir()
                    # keep first zip only
                    break
            except Exception as zip_err:
                logger.warning(f"Failed to extract RIFE zip: {zip_err}")

            if train_log_runtime.is_dir():
                weight_path = train_log_runtime
                logger.info(f"Using RIFE train_log from runtime: {weight_path}")
            else:
                # Look for a direct weight file in /opt/models/rife
                candidate_files = list(weights_root.glob('*.pkl')) + list(weights_root.glob('*.pth'))
                if candidate_files:
                    # Prefer .pkl (ECCV2022) over .pth (Practical-RIFE) if both present
                    pkl_files = [p for p in candidate_files if p.suffix == '.pkl']
                    weight_path = (pkl_files[0] if pkl_files else candidate_files[0])
                    logger.info(f"Using RIFE weight file: {weight_path}")
                elif repo_train_log.is_dir():
                    weight_path = repo_train_log
                    logger.info(f"Using RIFE train_log from repo: {weight_path}")

            if weight_path is None:
                raise FileNotFoundError("No RIFE weights found. Provide train_log/ or a .pkl/.pth file under /opt/models/rife")

            # Model.load_model supports both a directory (train_log) and a direct file in some forks.
            model.load_model(str(weight_path), -1)
            model.eval()
            model.device()
            
            self.loaded_models[model_key] = model
            logger.info("RIFE model loaded successfully")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load RIFE model: {e}")
            raise RuntimeError(f"RIFE model loading failed: {e}")
    
    def unload_model(self, model_key: str) -> None:
        if model_key in self.loaded_models:
            logger.info(f"Unloading model: {model_key}")
            del self.loaded_models[model_key]
            self._cleanup_gpu_memory()
    
    def unload_all_models(self) -> None:
        logger.info("Unloading all models")
        model_keys = list(self.loaded_models.keys())
        
        for key in model_keys:
            self.unload_model(key)
        
        self._cleanup_gpu_memory()
    
    def _cleanup_gpu_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    
    def is_model_loaded(self, model_key: str) -> bool:
        """Check if a specific model is currently loaded."""
        return model_key in self.loaded_models
    
    def get_loaded_models(self) -> list:
        return list(self.loaded_models.keys())