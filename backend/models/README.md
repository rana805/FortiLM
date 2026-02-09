# Custom Trained Models Directory

This directory contains custom trained models for FortiLM security modules.

## Directory Structure

```
models/
├── pii_detector/          # PII detection models
│   ├── pii_detector.pkl   # Example: Pickle model
│   └── pii_detector.onnx  # Example: ONNX model
├── toxicity_detector/      # Toxicity detection models
│   └── toxicity_detector.pkl
├── bias_detector/          # Bias detection models
│   └── bias_detector.pkl
└── jailbreak_detector/    # Jailbreak detection models
    └── jailbreak_detector.pkl
```

## Supported Model Formats

The model loader supports the following formats:

- **Pickle** (`.pkl`) - Python pickle format
- **Joblib** (`.joblib`) - Scikit-learn models
- **ONNX** (`.onnx`) - Open Neural Network Exchange format
- **TensorFlow** (`.h5`, `.pb`) - TensorFlow/Keras models
- **PyTorch** (`.pt`, `.pth`) - PyTorch models
- **JSON** (`.json`) - Configuration/rule-based models

## How to Add Your Trained Models

### 1. Train Your Model

Train your model using your preferred ML framework (scikit-learn, TensorFlow, PyTorch, etc.)

### 2. Save Your Model

Save the model in one of the supported formats:

```python
# Example: Save scikit-learn model
import joblib
joblib.dump(model, 'models/pii_detector/pii_detector.joblib')

# Example: Save TensorFlow model
model.save('models/toxicity_detector/toxicity_detector.h5')

# Example: Save PyTorch model
torch.save(model.state_dict(), 'models/bias_detector/bias_detector.pt')
```

### 3. Update Model Loading Code

The modules will automatically try to load models. Update the detector classes to use your models:

```python
from utils.model_loader import load_pii_detector_model

class CustomPIIDetector:
    def __init__(self):
        # Try to load trained model
        self.model = load_pii_detector_model("pii_detector", format="joblib")
        
        if self.model is None:
            # Fallback to rule-based patterns
            self.pii_patterns = {...}
    
    def detect_pii(self, text: str):
        if self.model:
            # Use trained model
            return self.model.predict(text)
        else:
            # Use rule-based fallback
            return self._rule_based_detection(text)
```

## Model Interface Requirements

Your trained models should follow these interfaces:

### PII Detector
- **Input**: `str` (text to analyze)
- **Output**: `List[Dict[str, Any]]` with keys: `type`, `value`, `start`, `end`, `confidence`

### Toxicity Detector
- **Input**: `str` (text to analyze)
- **Output**: `Dict[str, Any]` with keys: `toxicity_detected`, `toxicity_score`, `severity`, `detected_items`

### Bias Detector
- **Input**: `str` (text to analyze)
- **Output**: `Dict[str, Any]` with keys: `bias_detected`, `bias_types`, `bias_score`, `severity`

### Jailbreak Detector
- **Input**: `str` (text to analyze)
- **Output**: `Dict[str, Any]` with keys: `jailbreak_detected`, `jailbreak_score`, `severity`, `detected_indicators`

## Example Model Wrapper

If your model doesn't match the expected interface, create a wrapper:

```python
class ModelWrapper:
    def __init__(self, model):
        self.model = model
    
    def predict(self, text: str):
        # Preprocess text
        features = self._preprocess(text)
        
        # Get model prediction
        prediction = self.model.predict(features)
        
        # Format output to match expected interface
        return {
            "toxicity_detected": prediction > 0.5,
            "toxicity_score": float(prediction),
            "severity": self._get_severity(prediction)
        }
```

## Environment Variables

You can configure model paths via environment variables:

```bash
# .env file
PII_MODEL_PATH=models/pii_detector/custom_model.pkl
TOXICITY_MODEL_PATH=models/toxicity_detector/custom_model.h5
BIAS_MODEL_PATH=models/bias_detector/custom_model.pt
JAILBREAK_MODEL_PATH=models/jailbreak_detector/custom_model.onnx
```

## Testing Your Models

Test your models using the test scripts:

```bash
# Test PII detector
python -c "from utils.model_loader import load_pii_detector_model; model = load_pii_detector_model(); print(model)"

# Test toxicity detector
python -c "from utils.model_loader import load_toxicity_detector_model; model = load_toxicity_detector_model(); print(model)"
```

## Notes

- Models are loaded lazily (on first use)
- If a model is not found, the system falls back to rule-based detection
- Large models should be added to `.gitignore` if they exceed GitHub's file size limits
- Consider using model versioning (e.g., `pii_detector_v1.pkl`, `pii_detector_v2.pkl`)



This directory contains custom trained models for FortiLM security modules.

## Directory Structure

```
models/
├── pii_detector/          # PII detection models
│   ├── pii_detector.pkl   # Example: Pickle model
│   └── pii_detector.onnx  # Example: ONNX model
├── toxicity_detector/      # Toxicity detection models
│   └── toxicity_detector.pkl
├── bias_detector/          # Bias detection models
│   └── bias_detector.pkl
└── jailbreak_detector/    # Jailbreak detection models
    └── jailbreak_detector.pkl
```

## Supported Model Formats

The model loader supports the following formats:

- **Pickle** (`.pkl`) - Python pickle format
- **Joblib** (`.joblib`) - Scikit-learn models
- **ONNX** (`.onnx`) - Open Neural Network Exchange format
- **TensorFlow** (`.h5`, `.pb`) - TensorFlow/Keras models
- **PyTorch** (`.pt`, `.pth`) - PyTorch models
- **JSON** (`.json`) - Configuration/rule-based models

## How to Add Your Trained Models

### 1. Train Your Model

Train your model using your preferred ML framework (scikit-learn, TensorFlow, PyTorch, etc.)

### 2. Save Your Model

Save the model in one of the supported formats:

```python
# Example: Save scikit-learn model
import joblib
joblib.dump(model, 'models/pii_detector/pii_detector.joblib')

# Example: Save TensorFlow model
model.save('models/toxicity_detector/toxicity_detector.h5')

# Example: Save PyTorch model
torch.save(model.state_dict(), 'models/bias_detector/bias_detector.pt')
```

### 3. Update Model Loading Code

The modules will automatically try to load models. Update the detector classes to use your models:

```python
from utils.model_loader import load_pii_detector_model

class CustomPIIDetector:
    def __init__(self):
        # Try to load trained model
        self.model = load_pii_detector_model("pii_detector", format="joblib")
        
        if self.model is None:
            # Fallback to rule-based patterns
            self.pii_patterns = {...}
    
    def detect_pii(self, text: str):
        if self.model:
            # Use trained model
            return self.model.predict(text)
        else:
            # Use rule-based fallback
            return self._rule_based_detection(text)
```

## Model Interface Requirements

Your trained models should follow these interfaces:

### PII Detector
- **Input**: `str` (text to analyze)
- **Output**: `List[Dict[str, Any]]` with keys: `type`, `value`, `start`, `end`, `confidence`

### Toxicity Detector
- **Input**: `str` (text to analyze)
- **Output**: `Dict[str, Any]` with keys: `toxicity_detected`, `toxicity_score`, `severity`, `detected_items`

### Bias Detector
- **Input**: `str` (text to analyze)
- **Output**: `Dict[str, Any]` with keys: `bias_detected`, `bias_types`, `bias_score`, `severity`

### Jailbreak Detector
- **Input**: `str` (text to analyze)
- **Output**: `Dict[str, Any]` with keys: `jailbreak_detected`, `jailbreak_score`, `severity`, `detected_indicators`

## Example Model Wrapper

If your model doesn't match the expected interface, create a wrapper:

```python
class ModelWrapper:
    def __init__(self, model):
        self.model = model
    
    def predict(self, text: str):
        # Preprocess text
        features = self._preprocess(text)
        
        # Get model prediction
        prediction = self.model.predict(features)
        
        # Format output to match expected interface
        return {
            "toxicity_detected": prediction > 0.5,
            "toxicity_score": float(prediction),
            "severity": self._get_severity(prediction)
        }
```

## Environment Variables

You can configure model paths via environment variables:

```bash
# .env file
PII_MODEL_PATH=models/pii_detector/custom_model.pkl
TOXICITY_MODEL_PATH=models/toxicity_detector/custom_model.h5
BIAS_MODEL_PATH=models/bias_detector/custom_model.pt
JAILBREAK_MODEL_PATH=models/jailbreak_detector/custom_model.onnx
```

## Testing Your Models

Test your models using the test scripts:

```bash
# Test PII detector
python -c "from utils.model_loader import load_pii_detector_model; model = load_pii_detector_model(); print(model)"

# Test toxicity detector
python -c "from utils.model_loader import load_toxicity_detector_model; model = load_toxicity_detector_model(); print(model)"
```

## Notes

- Models are loaded lazily (on first use)
- If a model is not found, the system falls back to rule-based detection
- Large models should be added to `.gitignore` if they exceed GitHub's file size limits
- Consider using model versioning (e.g., `pii_detector_v1.pkl`, `pii_detector_v2.pkl`)

