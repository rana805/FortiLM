"""
Model Loader Utility for Custom Trained Models
Supports loading various ML model formats (pickle, joblib, ONNX, TensorFlow, PyTorch)
"""

import os
import pickle
from pathlib import Path
from typing import Optional, Any, Dict
import json

# Try importing ML libraries (optional dependencies)
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

# Base directory for ML models (separate from database models)
MODELS_BASE_DIR = Path(__file__).parent.parent / "ml_models"


class ModelLoader:
    """Utility class for loading custom trained models."""
    
    @staticmethod
    def get_model_path(model_type: str, model_name: str) -> Path:
        """
        Get the path to a model file.
        
        Args:
            model_type: Type of model (pii_detector, toxicity_detector, bias_detector, jailbreak_detector)
            model_name: Name of the model file (without extension)
        
        Returns:
            Path to the model file
        """
        model_dir = MODELS_BASE_DIR / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Try common extensions
        extensions = ['.pkl', '.joblib', '.onnx', '.h5', '.pb', '.pt', '.pth', '.json']
        for ext in extensions:
            model_path = model_dir / f"{model_name}{ext}"
            if model_path.exists():
                return model_path
        
        # Return default path (user should create the model)
        return model_dir / f"{model_name}.pkl"
    
    @staticmethod
    def load_pickle_model(model_path: Path) -> Any:
        """Load a pickle model. Tries joblib first (better for scikit-learn/transformers models), then pickle."""
        # Import wrapper classes before unpickling to avoid "Can't get attribute" errors
        try:
            from ml_models.pii_detector.wrapper import PIIDetectorWrapper
            # Register the class for pickle/joblib to find it
            import copyreg
            # This allows pickle to find PIIDetectorWrapper even if it was saved from __main__
            def reduce_pii_wrapper(obj):
                # Reconstruct from the model_dir stored in the object
                return (PIIDetectorWrapper, (obj.model_dir,))
            copyreg.pickle(PIIDetectorWrapper, reduce_pii_wrapper)
        except ImportError:
            pass  # Wrapper not needed for all models
        
        # Try joblib first (most models saved with transformers/scikit-learn use joblib)
        if HAS_JOBLIB:
            try:
                # Patch torch.load to use CPU if CUDA not available
                import torch
                original_load = torch.load
                if not torch.cuda.is_available():
                    def cpu_load(*args, **kwargs):
                        if 'map_location' not in kwargs:
                            kwargs['map_location'] = 'cpu'
                        return original_load(*args, **kwargs)
                    torch.load = cpu_load
                
                # Use custom unpickler for joblib to handle module path issues
                import io
                with open(model_path, 'rb') as f:
                    # Read the file
                    data = f.read()
                    # Try to load with joblib
                    import io as io_module
                    result = joblib.load(io_module.BytesIO(data))
                    
                    # Restore original torch.load
                    if not torch.cuda.is_available():
                        torch.load = original_load
                    
                    # Ensure model is on CPU if loaded wrapper
                    if hasattr(result, 'model') and hasattr(result.model, 'to'):
                        if not torch.cuda.is_available():
                            result.model = result.model.cpu()
                            if hasattr(result, 'device'):
                                result.device = 'cpu'
                    
                    return result
            except Exception as e:
                # If joblib fails, try with custom find_class
                try:
                    # Custom unpickler that redirects __main__ references
                    class CustomUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            if module == '__main__' and name == 'PIIDetectorWrapper':
                                from ml_models.pii_detector.wrapper import PIIDetectorWrapper
                                return PIIDetectorWrapper
                            if module == '__main__' and name == 'ToxicityDetectorWrapper':
                                from ml_models.toxicity_detector.wrapper import ToxicityDetectorWrapper
                                return ToxicityDetectorWrapper
                            if module == '__main__' and name == 'BiasDetectorWrapper':
                                from ml_models.bias_detector.wrapper import BiasDetectorWrapper
                                return BiasDetectorWrapper
                            return super().find_class(module, name)
                    
                    with open(model_path, 'rb') as f:
                        unpickler = CustomUnpickler(f)
                        return unpickler.load()
                except Exception:
                    # Last resort: try standard pickle
                    with open(model_path, 'rb') as f:
                        return pickle.load(f)
        
        # Fallback to standard pickle with custom find_class
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == '__main__' and name == 'PIIDetectorWrapper':
                    from ml_models.pii_detector.wrapper import PIIDetectorWrapper
                    return PIIDetectorWrapper
                if module == '__main__' and name == 'ToxicityDetectorWrapper':
                    from ml_models.toxicity_detector.wrapper import ToxicityDetectorWrapper
                    return ToxicityDetectorWrapper
                if module == '__main__' and name == 'BiasDetectorWrapper':
                    from ml_models.bias_detector.wrapper import BiasDetectorWrapper
                    return BiasDetectorWrapper
                return super().find_class(module, name)
        
        with open(model_path, 'rb') as f:
            unpickler = CustomUnpickler(f)
            return unpickler.load()
    
    @staticmethod
    def load_joblib_model(model_path: Path) -> Any:
        """Load a joblib model."""
        if not HAS_JOBLIB:
            raise ImportError("joblib is not installed. Install it with: pip install joblib")
        return joblib.load(model_path)
    
    @staticmethod
    def load_onnx_model(model_path: Path) -> Any:
        """Load an ONNX model."""
        if not HAS_ONNX:
            raise ImportError("onnxruntime is not installed. Install it with: pip install onnxruntime")
        return ort.InferenceSession(str(model_path))
    
    @staticmethod
    def load_tensorflow_model(model_path: Path) -> Any:
        """Load a TensorFlow model."""
        if not HAS_TENSORFLOW:
            raise ImportError("tensorflow is not installed. Install it with: pip install tensorflow")
        return tf.keras.models.load_model(str(model_path))
    
    @staticmethod
    def load_pytorch_model(model_path: Path) -> Any:
        """Load a PyTorch model."""
        if not HAS_PYTORCH:
            raise ImportError("torch is not installed. Install it with: pip install torch")
        return torch.load(model_path, map_location='cpu')
    
    @staticmethod
    def load_json_model(model_path: Path) -> Dict:
        """Load a JSON model configuration."""
        with open(model_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def load_model(model_type: str, model_name: str, model_format: Optional[str] = None) -> Optional[Any]:
        """
        Load a custom trained model.
        
        Args:
            model_type: Type of model (pii_detector, toxicity_detector, bias_detector, jailbreak_detector)
            model_name: Name of the model file (without extension)
            model_format: Format of the model (pkl, joblib, onnx, tensorflow, pytorch, json). 
                         If None, auto-detect from file extension.
        
        Returns:
            Loaded model object, or None if model not found
        """
        model_path = ModelLoader.get_model_path(model_type, model_name)
        
        if not model_path.exists():
            # For PII detector, try loading directly from HuggingFace checkpoint if pickle not found
            if model_type == "pii_detector":
                # Check if there's a HuggingFace model directory
                model_dir = model_path.parent / "checkpoint" / "pii-bert"
                if model_dir.exists() and (model_dir / "config.json").exists():
                    print(f"ℹ️  Loading PII detector from HuggingFace checkpoint: {model_dir}")
                    try:
                        from ml_models.pii_detector.wrapper import PIIDetectorWrapper
                        return PIIDetectorWrapper(str(model_dir))
                    except Exception as e:
                        print(f"⚠️  Failed to load from checkpoint: {e}")
            
            print(f"⚠️  Model not found: {model_path}")
            print(f"   Place your trained model at: {model_path}")
            return None
        
        # Auto-detect format if not specified
        if model_format is None:
            ext = model_path.suffix.lower()
            if ext == '.pkl':
                model_format = 'pkl'
            elif ext == '.joblib':
                model_format = 'joblib'
            elif ext == '.onnx':
                model_format = 'onnx'
            elif ext in ['.h5', '.pb']:
                model_format = 'tensorflow'
            elif ext in ['.pt', '.pth']:
                model_format = 'pytorch'
            elif ext == '.json':
                model_format = 'json'
            else:
                model_format = 'pkl'  # Default
        
        try:
            if model_format == 'pkl':
                return ModelLoader.load_pickle_model(model_path)
            elif model_format == 'joblib':
                return ModelLoader.load_joblib_model(model_path)
            elif model_format == 'onnx':
                return ModelLoader.load_onnx_model(model_path)
            elif model_format == 'tensorflow':
                return ModelLoader.load_tensorflow_model(model_path)
            elif model_format == 'pytorch':
                return ModelLoader.load_pytorch_model(model_path)
            elif model_format == 'json':
                return ModelLoader.load_json_model(model_path)
            else:
                print(f"⚠️  Unknown model format: {model_format}")
                return None
        except Exception as e:
            print(f"❌ Error loading model {model_path}: {e}")
            # For PII detector pickle errors, try to provide helpful message
            if model_type == "pii_detector" and "PIIDetectorWrapper" in str(e):
                print(f"💡 Tip: The model was saved with PIIDetectorWrapper. Make sure wrapper.py is importable.")
            return None
    
    @staticmethod
    def list_available_models(model_type: str) -> list:
        """List all available models of a given type."""
        model_dir = MODELS_BASE_DIR / model_type
        if not model_dir.exists():
            return []
        
        models = []
        for file in model_dir.iterdir():
            if file.is_file() and not file.name.startswith('.'):
                models.append(file.stem)
        return models


# Convenience functions for each model type
def load_pii_detector_model(model_name: str = "pii_detector", format: Optional[str] = None):
    """Load a PII detection model."""
    return ModelLoader.load_model("pii_detector", model_name, format)

def load_toxicity_detector_model(model_name: str = "toxicity_detector", format: Optional[str] = None):
    """Load a toxicity detection model."""
    return ModelLoader.load_model("toxicity_detector", model_name, format)

def load_bias_detector_model(model_name: str = "bias_detector_wrapper", format: Optional[str] = None):
    """Load a bias detection model."""
    return ModelLoader.load_model("bias_detector", model_name, format)

def load_jailbreak_detector_model(model_name: str = "jailbreak_detector", format: Optional[str] = None):
    """Load a jailbreak detection model."""
    return ModelLoader.load_model("jailbreak_detector", model_name, format)

