"""
Toxicity Detector Wrapper for DistilBERT-based sequence classification models.
This module provides the ToxicityDetectorWrapper class that can be imported and used for loading pickled models.
Optimized for performance with device management and efficient inference.
"""
from typing import Dict, Any, Optional
import os

# Import dependencies with error handling
try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_DEPENDENCIES = True
except ImportError as e:
    HAS_DEPENDENCIES = False
    IMPORT_ERROR = str(e)


class ToxicityDetectorWrapper:
    """
    Optimized wrapper for HuggingFace sequence classification models (DistilBERT-based) for toxicity detection.
    Returns a dictionary with toxicity_detected, toxicity_score, severity, and detected_items.
    
    Features:
    - Automatic device detection (GPU/CPU)
    - Model caching and optimization
    - Efficient inference
    - Memory management
    """
    
    def __init__(self, model_dir: str, device: Optional[str] = None):
        """
        Initialize the wrapper with a trained HuggingFace model directory.
        
        Args:
            model_dir: Path to the directory containing the trained model (tokenizer and model files)
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        if not HAS_DEPENDENCIES:
            raise ImportError(f"Required dependencies not installed: {IMPORT_ERROR}. Install with: pip install torch transformers numpy")
        
        self.model_dir = model_dir
        
        # Device management: use GPU if available, otherwise CPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading toxicity detector model from {model_dir} on {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Threshold for toxicity detection
        # Set to 0.5 to maintain training accuracy (model was trained with this threshold)
        # DO NOT change this without retraining - it will affect model accuracy
        self.toxicity_threshold = 0.5
        
        # Severity thresholds (based on model's confidence scores)
        # These maintain the same quality as training
        self.severity_thresholds = {
            'high': 0.8,    # Very confident toxicity
            'medium': 0.6,  # Moderate confidence
            'low': 0.5      # Low confidence (just above threshold)
        }
        
        # Quality assurance: Ensure model is in eval mode
        self.model.eval()
        
        # Disable dropout and other training features for consistent inference
        for param in self.model.parameters():
            param.requires_grad = False
        
        print(f"✅ Toxicity detector model loaded successfully")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict toxicity for a given text.
        Optimized for single predictions.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with keys:
                - toxicity_detected: bool
                - toxicity_score: float (0-1)
                - severity: str ('high', 'medium', 'low', or None)
                - detected_items: list of detected toxic patterns (empty for now)
        """
        if not text or not text.strip():
            return {
                'toxicity_detected': False,
                'toxicity_score': 0.0,
                'severity': None,
                'detected_items': []
            }
        
        # Tokenize input (optimized: no padding for single text)
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,  # Keep padding for consistency
            max_length=512,
            return_tensors='pt'
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict (with no_grad for memory efficiency and consistency)
        # Using eval() mode ensures consistent results (no dropout, batch norm in eval mode)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Use softmax to get probabilities (same as training)
            probs = torch.softmax(logits, dim=-1)
            
            # Get probability of toxic class (class 1)
            # Move to CPU immediately to free GPU memory, but maintain precision
            toxic_prob = float(probs[0][1].cpu().item())
            
            # Ensure probability is in valid range [0, 1]
            toxic_prob = max(0.0, min(1.0, toxic_prob))
        
        # Determine if toxic
        toxicity_detected = toxic_prob >= self.toxicity_threshold
        
        # Determine severity
        severity = None
        if toxicity_detected:
            if toxic_prob >= self.severity_thresholds['high']:
                severity = 'high'
            elif toxic_prob >= self.severity_thresholds['medium']:
                severity = 'medium'
            else:
                severity = 'low'
        
        return {
            'toxicity_detected': toxicity_detected,
            'toxicity_score': toxic_prob,
            'severity': severity,
            'detected_items': []  # Can be extended to include specific toxic patterns
        }
    
    def predict_batch(self, texts: list, batch_size: int = 16) -> list:
        """
        Predict toxicity for a batch of texts (optimized for efficiency).
        Processes in smaller batches to avoid memory issues.
        
        Args:
            texts: List of input texts to analyze
            batch_size: Number of texts to process at once (default: 16)
            
        Returns:
            List of prediction dictionaries
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [(i, text) for i, text in enumerate(texts) if text and text.strip()]
        if not valid_texts:
            return [{
                'toxicity_detected': False,
                'toxicity_score': 0.0,
                'severity': None,
                'detected_items': []
            }] * len(texts)
        
        # Process in batches to optimize memory and speed
        results = [None] * len(texts)
        valid_indices, valid_texts_only = zip(*valid_texts)
        valid_texts_list = list(valid_texts_only)
        
        # Process in chunks
        for batch_start in range(0, len(valid_texts_list), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_texts_list))
            batch_texts = valid_texts_list[batch_start:batch_end]
            batch_indices = valid_indices[batch_start:batch_end]
            
            # Tokenize batch (optimized: use padding efficiently)
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',  # Consistent padding for batch
                max_length=512,
                return_tensors='pt'
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict (with no_grad for memory efficiency and consistency)
            # Using eval() mode ensures consistent results
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Use softmax to get probabilities (same as training)
                probs = torch.softmax(logits, dim=-1)
                
                # Get probabilities of toxic class (class 1)
                # Move to CPU and convert to float for precision
                toxic_probs = probs[:, 1].cpu().numpy().astype(float)
                
                # Ensure probabilities are in valid range [0, 1]
                toxic_probs = np.clip(toxic_probs, 0.0, 1.0)
            
            # Build results for this batch
            for local_idx, global_idx in enumerate(batch_indices):
                toxic_prob = float(toxic_probs[local_idx])
                toxicity_detected = toxic_prob >= self.toxicity_threshold
                
                severity = None
                if toxicity_detected:
                    if toxic_prob >= self.severity_thresholds['high']:
                        severity = 'high'
                    elif toxic_prob >= self.severity_thresholds['medium']:
                        severity = 'medium'
                    else:
                        severity = 'low'
                
                results[global_idx] = {
                    'toxicity_detected': toxicity_detected,
                    'toxicity_score': toxic_prob,
                    'severity': severity,
                    'detected_items': []
                }
        
        # Fill in None results (invalid texts)
        for i, result in enumerate(results):
            if result is None:
                results[i] = {
                    'toxicity_detected': False,
                    'toxicity_score': 0.0,
                    'severity': None,
                    'detected_items': []
                }
        
        return results
