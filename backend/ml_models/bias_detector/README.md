# Bias Detector Model

This directory contains the trained bias detection model for detecting bias in LLM outputs.

## Model Details

- **Architecture**: DistilBERT (distilbert-base-uncased)
- **Task**: Multi-label classification
- **Labels**: 4 bias types (gender, racial, religious, age)
- **Expected Performance**: 85-92% accuracy, 0.80-0.90 F1-score per label

## Directory Structure

```
bias_detector/
├── model/              # Trained model files (upload after training)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer files
│   └── training_info.json
├── wrapper.py          # Model wrapper for inference
└── README.md           # This file
```

## Usage

The model is automatically loaded by `CustomBiasDetector` in `backend/modules/output_filter.py`.

### Manual Usage

```python
from ml_models.bias_detector.wrapper import BiasDetectorWrapper

# Load model
detector = BiasDetectorWrapper("backend/ml_models/bias_detector/model")

# Detect bias
result = detector.predict("Women are not good at programming")
# Returns:
# {
#     "bias_detected": True,
#     "bias_types": ["gender"],
#     "bias_score": 0.85,
#     "bias_scores": {"gender": 0.85, "racial": 0.1, "religious": 0.05, "age": 0.02},
#     "severity": "high"
# }
```

## Training

See `backend/ml_models/training/QUICK_START_BIAS.md` for training instructions.

## Integration

The model integrates automatically with the Output Filter module. If the model is not available, the system falls back to rule-based bias detection.


