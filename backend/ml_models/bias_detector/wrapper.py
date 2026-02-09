"""
Bias Detector Wrapper

Loads and uses trained DistilBERT model for bias detection.
Supports multi-label classification (gender, racial, religious, age).
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np

BIAS_TYPES = ["gender", "racial", "religious"]  # Age excluded for better training quality


class BiasDetectorWrapper:
    """
    Wrapper for trained bias detection model.
    
    Returns:
    {
        "bias_detected": bool,
        "bias_types": List[str],  # e.g., ["gender", "racial"]
        "bias_score": float,  # Overall bias score (0-1)
        "bias_scores": Dict[str, float],  # Per-type scores
        "severity": Optional[str]  # "low", "medium", "high"
    }
    """
    
    def __init__(self, model_dir: str, device: Optional[str] = None):
        """
        Initialize bias detector.
        
        Args:
            model_dir: Path to trained model directory
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        print(f"Loading bias detector tokenizer from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        
        # Load model
        print(f"Loading bias detector model from {model_dir}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir),
            num_labels=3,  # 3 bias types: gender, racial, religious (age excluded)
            problem_type="multi_label_classification"
        )
        # Ensure model is on correct device (CPU if CUDA not available)
        if not torch.cuda.is_available():
            self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        # Thresholds
        self.bias_threshold = 0.5  # Threshold for binary classification
        self.severity_thresholds = {
            "high": 0.7,
            "medium": 0.4,
            "low": 0.2
        }
        
        print("✅ Bias detector model loaded successfully")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict bias in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with bias detection results
        """
        if not text or not text.strip():
            return {
                "bias_detected": False,
                "bias_types": [],
                "bias_score": 0.0,
                "bias_scores": {bias_type: 0.0 for bias_type in BIAS_TYPES},
                "severity": None
            }
        
        # Tokenize
        # Ensure device is CPU if CUDA not available
        device = getattr(self, 'device', 'cpu')
        if not torch.cuda.is_available():
            device = 'cpu'
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,  # Match training max_length
            return_tensors="pt"
        )
        # Move inputs to device manually
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Get binary predictions (threshold at 0.5)
        # Handle both threshold and bias_threshold for compatibility
        threshold = getattr(self, 'bias_threshold', getattr(self, 'threshold', 0.5))
        predictions = (probs > threshold).astype(int)
        
        # Map to bias types
        bias_types = []
        bias_scores = {}
        for i, bias_type in enumerate(BIAS_TYPES):
            score = float(probs[i])
            bias_scores[bias_type] = score
            if predictions[i] == 1:
                bias_types.append(bias_type)
        
        # Calculate overall bias score (max of all types)
        bias_score = float(np.max(probs))
        
        # Determine severity
        # Handle both severity_thresholds and default thresholds for compatibility
        severity_thresholds = getattr(self, 'severity_thresholds', {
            "high": 0.7,
            "medium": 0.4,
            "low": 0.2
        })
        severity = None
        if bias_score >= severity_thresholds.get("high", 0.7):
            severity = "high"
        elif bias_score >= severity_thresholds.get("medium", 0.4):
            severity = "medium"
        elif bias_score >= severity_thresholds.get("low", 0.2):
            severity = "low"
        
        return {
            "bias_detected": len(bias_types) > 0,
            "bias_types": bias_types,
            "bias_score": bias_score,
            "bias_scores": bias_scores,
            "severity": severity
        }
    
    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Predict bias for multiple texts (batch processing).
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of bias detection results
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [self.predict(text) for text in batch]
            results.extend(batch_results)
        return results

