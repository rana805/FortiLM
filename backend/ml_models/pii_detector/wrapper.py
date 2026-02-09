"""
PII Detector Wrapper for BERT-based token classification models.
This module provides the PIIDetectorWrapper class that can be imported and used for loading pickled models.
Optimized for performance with device management, caching, and efficient inference.
"""
from typing import List, Dict, Any, Optional
import os

# Import dependencies with error handling
try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    HAS_DEPENDENCIES = True
except ImportError as e:
    HAS_DEPENDENCIES = False
    IMPORT_ERROR = str(e)


class PIIDetectorWrapper:
    """
    Optimized wrapper for HuggingFace token classification models (BERT-based) for PII detection.
    Converts model output to the expected format: List[Dict[str, Any]] with keys: type, value, start, end, confidence
    
    Features:
    - Automatic device detection (GPU/CPU)
    - Model caching and optimization
    - Efficient batch processing support
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
        
        # Load tokenizer (cached by transformers library)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model and move to appropriate device
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Disable gradient computation for inference (saves memory)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Load label maps from config
        cfg = self.model.config
        self.id2label = cfg.id2label
        self.label2id = cfg.label2id
        
        # Per-type confidence thresholds (can be tuned based on validation)
        self.type_thresholds = {
            "email": 0.7,
            "phone": 0.6,
            "ssn": 0.8,
            "credit_card": 0.7,
            "ip_address": 0.6,
            "date_of_birth": 0.7,
            "zip_code": 0.5,
            "name": 0.6,
            "location": 0.6,
        }

    def predict(self, text: str, min_len: int = 3, min_conf: float = 0.5) -> List[Dict[str, Any]]:
        """
        Predict PII entities in text with improved span extraction and merging.
        Optimized for performance with device-aware tensor operations.
        
        Args:
            text: Input text to analyze
            min_len: Minimum length of detected spans (filters out single characters)
            min_conf: Minimum confidence threshold (overridden by type-specific thresholds)
            
        Returns:
            List of detected PII entities, each with keys: type, value, start, end, confidence
        """
        # Early return for empty or very short text
        if not text or len(text.strip()) < min_len:
            return []
        
        # Tokenize without splitting into words first (better for subword handling)
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=False,
            add_special_tokens=True
        )
        
        # Move inputs to same device as model (only if tensors and device is set)
        if hasattr(self, 'device'):
            enc_device = {}
            for k, v in enc.items():
                if isinstance(v, torch.Tensor):
                    enc_device[k] = v.to(self.device)
                else:
                    enc_device[k] = v
            enc = enc_device
        
        # Get offsets and remove from model input (model doesn't accept offset_mapping)
        # Handle device-aware offset extraction
        offset_mapping = enc.pop("offset_mapping")
        if isinstance(offset_mapping, torch.Tensor):
            offsets = offset_mapping[0].cpu().tolist()
        else:
            offsets = offset_mapping[0] if isinstance(offset_mapping, (list, tuple)) else offset_mapping
        
        # Inference with no gradient computation (faster, less memory)
        with torch.no_grad():
            outputs = self.model(**enc)
        
        # Move logits to CPU and convert to numpy (more efficient than keeping on GPU)
        logits = outputs.logits[0].cpu().numpy()
        preds = np.argmax(logits, axis=-1)
        
        # Compute confidence scores efficiently
        logits_tensor = torch.from_numpy(logits)
        confs = torch.softmax(logits_tensor, dim=-1).numpy().max(axis=-1)

        # Extract spans using BIO tagging
        spans = []
        current_span = None
        
        for idx, (pred_id, conf) in enumerate(zip(preds, confs)):
            if idx >= len(offsets):
                break
                
            # Get label
            label = None
            if isinstance(self.id2label, dict):
                label = self.id2label.get(pred_id)
                if label is None:
                    label = self.id2label.get(str(pred_id))
            if label is None:
                label = "O"
            
            start_char, end_char = offsets[idx]
            
            # Skip special tokens (CLS, SEP, PAD)
            if start_char == 0 and end_char == 0:
                if current_span:
                    spans.append(current_span)
                    current_span = None
                continue
            
            if label == "O":
                # End current span if exists
                if current_span:
                    spans.append(current_span)
                    current_span = None
                continue
            
            # Parse BIO tag
            if "-" in label:
                prefix, tag = label.split("-", 1)
            else:
                prefix, tag = "O", label
            
            tag = tag.lower()
            
            # Start new span or continue current
            if prefix == "B" or (current_span and current_span["type"] != tag):
                # Save previous span if exists
                if current_span:
                    spans.append(current_span)
                # Start new span
                current_span = {
                    "type": tag,
                    "start": start_char,
                    "end": end_char,
                    "confs": [float(conf)]
                }
            elif current_span and current_span["type"] == tag:
                # Continue current span (I- tag)
                # Only extend if tokens are adjacent or overlapping
                if start_char <= current_span["end"]:
                    current_span["end"] = end_char
                    current_span["confs"].append(float(conf))
                else:
                    # Non-adjacent, start new span
                    spans.append(current_span)
                    current_span = {
                        "type": tag,
                        "start": start_char,
                        "end": end_char,
                        "confs": [float(conf)]
                    }
        
        # Add final span
        if current_span:
            spans.append(current_span)
        
        # Merge overlapping/adjacent spans of the same type
        spans = self._merge_spans(spans)
        
        # Finalize results with filtering and validation
        results = []
        for sp in spans:
            # Extract value and clean up trailing punctuation that might be included
            raw_value = text[sp["start"]:sp["end"]]
            value = raw_value.strip()
            
            # Remove trailing punctuation that's likely not part of PII (periods, commas, etc.)
            # But keep punctuation that's part of the format (dashes in SSN, @ in email, etc.)
            if sp["type"] not in ["email", "phone", "ssn", "credit_card", "date_of_birth"]:
                # For non-structured PII, remove trailing punctuation
                original_end = sp["end"]
                while value and value[-1] in ".,;:!?)":
                    value = value[:-1]
                    sp["end"] -= 1
                # If we removed punctuation, update the span
                if sp["end"] < original_end:
                    # Re-extract to ensure correct value
                    value = text[sp["start"]:sp["end"]].strip()
            
            conf = float(np.mean(sp["confs"]))
            
            # Filter criteria
            if len(value) < min_len:
                continue
            
            # Use type-specific threshold if available, otherwise use min_conf
            # Backward compatibility: check if type_thresholds exists (for old pickled models)
            if hasattr(self, 'type_thresholds'):
                type_threshold = self.type_thresholds.get(sp["type"], min_conf)
            else:
                type_threshold = min_conf
            if conf < type_threshold:
                continue
            
            # Validate PII based on type and context
            if not self._validate_pii(sp["type"], value, text, sp["start"], sp["end"]):
                continue
            
            results.append({
                "type": sp["type"],
                "value": value,
                "start": sp["start"],
                "end": sp["end"],
                "confidence": conf,
            })
        
        return results
    
    def _merge_spans(self, spans: List[Dict]) -> List[Dict]:
        """Merge overlapping or adjacent spans of the same type."""
        if not spans:
            return []
        
        # Sort by start position
        sorted_spans = sorted(spans, key=lambda x: x["start"])
        merged = [sorted_spans[0]]
        
        for span in sorted_spans[1:]:
            last = merged[-1]
            # Merge if same type and overlapping/adjacent (within 2 chars)
            if (last["type"] == span["type"] and 
                span["start"] <= last["end"] + 2):
                last["end"] = max(last["end"], span["end"])
                last["confs"].extend(span["confs"])
            else:
                merged.append(span)
        
        return merged
    
    def _validate_pii(self, pii_type: str, value: str, text: str = "", start: int = 0, end: int = 0) -> bool:
        """Validate detected PII based on type-specific patterns and context."""
        import re
        
        validation_patterns = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "phone": r"^[\d\s\-\(\)\+\.]{10,}$",  # At least 10 digits/chars
            "ssn": r"^\d{3}[-]?\d{2}[-]?\d{4}$",
            "credit_card": r"^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$",
            "ip_address": r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$",
            "date_of_birth": r"^(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}$",
            "zip_code": r"^\d{5}(?:-\d{4})?$",
        }
        
        # Context-aware validation: check if number is in mathematical expression
        if pii_type == "zip_code" and text:
            # Check surrounding context (20 chars before and after)
            context_start = max(0, start - 20)
            context_end = min(len(text), end + 20)
            context = text[context_start:context_end].lower()
            
            # Mathematical operators that suggest this is NOT a ZIP code
            math_indicators = ['+', '-', '*', '/', '=', 'plus', 'minus', 'times', 'divided', 'equals', 'sum', 'add']
            # Check if any math operators are nearby
            if any(op in context for op in math_indicators):
                return False
            
            # Check if it's part of a larger number (like 102323)
            if start > 0 and text[start-1].isdigit():
                return False
            if end < len(text) and text[end].isdigit():
                return False
        
        # Skip validation for types without patterns (name, location)
        if pii_type not in validation_patterns:
            # Basic validation: not too short, contains letters/numbers
            return len(value) >= 2 and any(c.isalnum() for c in value)
        
        pattern = validation_patterns.get(pii_type)
        if pattern:
            return bool(re.match(pattern, value))
        
        return True


This module provides the PIIDetectorWrapper class that can be imported and used for loading pickled models.
Optimized for performance with device management, caching, and efficient inference.
"""
from typing import List, Dict, Any, Optional
import os

# Import dependencies with error handling
try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    HAS_DEPENDENCIES = True
except ImportError as e:
    HAS_DEPENDENCIES = False
    IMPORT_ERROR = str(e)


class PIIDetectorWrapper:
    """
    Optimized wrapper for HuggingFace token classification models (BERT-based) for PII detection.
    Converts model output to the expected format: List[Dict[str, Any]] with keys: type, value, start, end, confidence
    
    Features:
    - Automatic device detection (GPU/CPU)
    - Model caching and optimization
    - Efficient batch processing support
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
        
        # Load tokenizer (cached by transformers library)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model and move to appropriate device
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Disable gradient computation for inference (saves memory)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Load label maps from config
        cfg = self.model.config
        self.id2label = cfg.id2label
        self.label2id = cfg.label2id
        
        # Per-type confidence thresholds (can be tuned based on validation)
        self.type_thresholds = {
            "email": 0.7,
            "phone": 0.6,
            "ssn": 0.8,
            "credit_card": 0.7,
            "ip_address": 0.6,
            "date_of_birth": 0.7,
            "zip_code": 0.5,
            "name": 0.6,
            "location": 0.6,
        }

    def predict(self, text: str, min_len: int = 3, min_conf: float = 0.5) -> List[Dict[str, Any]]:
        """
        Predict PII entities in text with improved span extraction and merging.
        Optimized for performance with device-aware tensor operations.
        
        Args:
            text: Input text to analyze
            min_len: Minimum length of detected spans (filters out single characters)
            min_conf: Minimum confidence threshold (overridden by type-specific thresholds)
            
        Returns:
            List of detected PII entities, each with keys: type, value, start, end, confidence
        """
        # Early return for empty or very short text
        if not text or len(text.strip()) < min_len:
            return []
        
        # Tokenize without splitting into words first (better for subword handling)
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=False,
            add_special_tokens=True
        )
        
        # Move inputs to same device as model (only if tensors and device is set)
        if hasattr(self, 'device'):
            enc_device = {}
            for k, v in enc.items():
                if isinstance(v, torch.Tensor):
                    enc_device[k] = v.to(self.device)
                else:
                    enc_device[k] = v
            enc = enc_device
        
        # Get offsets and remove from model input (model doesn't accept offset_mapping)
        # Handle device-aware offset extraction
        offset_mapping = enc.pop("offset_mapping")
        if isinstance(offset_mapping, torch.Tensor):
            offsets = offset_mapping[0].cpu().tolist()
        else:
            offsets = offset_mapping[0] if isinstance(offset_mapping, (list, tuple)) else offset_mapping
        
        # Inference with no gradient computation (faster, less memory)
        with torch.no_grad():
            outputs = self.model(**enc)
        
        # Move logits to CPU and convert to numpy (more efficient than keeping on GPU)
        logits = outputs.logits[0].cpu().numpy()
        preds = np.argmax(logits, axis=-1)
        
        # Compute confidence scores efficiently
        logits_tensor = torch.from_numpy(logits)
        confs = torch.softmax(logits_tensor, dim=-1).numpy().max(axis=-1)

        # Extract spans using BIO tagging
        spans = []
        current_span = None
        
        for idx, (pred_id, conf) in enumerate(zip(preds, confs)):
            if idx >= len(offsets):
                break
                
            # Get label
            label = None
            if isinstance(self.id2label, dict):
                label = self.id2label.get(pred_id)
                if label is None:
                    label = self.id2label.get(str(pred_id))
            if label is None:
                label = "O"
            
            start_char, end_char = offsets[idx]
            
            # Skip special tokens (CLS, SEP, PAD)
            if start_char == 0 and end_char == 0:
                if current_span:
                    spans.append(current_span)
                    current_span = None
                continue
            
            if label == "O":
                # End current span if exists
                if current_span:
                    spans.append(current_span)
                    current_span = None
                continue
            
            # Parse BIO tag
            if "-" in label:
                prefix, tag = label.split("-", 1)
            else:
                prefix, tag = "O", label
            
            tag = tag.lower()
            
            # Start new span or continue current
            if prefix == "B" or (current_span and current_span["type"] != tag):
                # Save previous span if exists
                if current_span:
                    spans.append(current_span)
                # Start new span
                current_span = {
                    "type": tag,
                    "start": start_char,
                    "end": end_char,
                    "confs": [float(conf)]
                }
            elif current_span and current_span["type"] == tag:
                # Continue current span (I- tag)
                # Only extend if tokens are adjacent or overlapping
                if start_char <= current_span["end"]:
                    current_span["end"] = end_char
                    current_span["confs"].append(float(conf))
                else:
                    # Non-adjacent, start new span
                    spans.append(current_span)
                    current_span = {
                        "type": tag,
                        "start": start_char,
                        "end": end_char,
                        "confs": [float(conf)]
                    }
        
        # Add final span
        if current_span:
            spans.append(current_span)
        
        # Merge overlapping/adjacent spans of the same type
        spans = self._merge_spans(spans)
        
        # Finalize results with filtering and validation
        results = []
        for sp in spans:
            # Extract value and clean up trailing punctuation that might be included
            raw_value = text[sp["start"]:sp["end"]]
            value = raw_value.strip()
            
            # Remove trailing punctuation that's likely not part of PII (periods, commas, etc.)
            # But keep punctuation that's part of the format (dashes in SSN, @ in email, etc.)
            if sp["type"] not in ["email", "phone", "ssn", "credit_card", "date_of_birth"]:
                # For non-structured PII, remove trailing punctuation
                original_end = sp["end"]
                while value and value[-1] in ".,;:!?)":
                    value = value[:-1]
                    sp["end"] -= 1
                # If we removed punctuation, update the span
                if sp["end"] < original_end:
                    # Re-extract to ensure correct value
                    value = text[sp["start"]:sp["end"]].strip()
            
            conf = float(np.mean(sp["confs"]))
            
            # Filter criteria
            if len(value) < min_len:
                continue
            
            # Use type-specific threshold if available, otherwise use min_conf
            # Backward compatibility: check if type_thresholds exists (for old pickled models)
            if hasattr(self, 'type_thresholds'):
                type_threshold = self.type_thresholds.get(sp["type"], min_conf)
            else:
                type_threshold = min_conf
            if conf < type_threshold:
                continue
            
            # Validate PII based on type and context
            if not self._validate_pii(sp["type"], value, text, sp["start"], sp["end"]):
                continue
            
            results.append({
                "type": sp["type"],
                "value": value,
                "start": sp["start"],
                "end": sp["end"],
                "confidence": conf,
            })
        
        return results
    
    def _merge_spans(self, spans: List[Dict]) -> List[Dict]:
        """Merge overlapping or adjacent spans of the same type."""
        if not spans:
            return []
        
        # Sort by start position
        sorted_spans = sorted(spans, key=lambda x: x["start"])
        merged = [sorted_spans[0]]
        
        for span in sorted_spans[1:]:
            last = merged[-1]
            # Merge if same type and overlapping/adjacent (within 2 chars)
            if (last["type"] == span["type"] and 
                span["start"] <= last["end"] + 2):
                last["end"] = max(last["end"], span["end"])
                last["confs"].extend(span["confs"])
            else:
                merged.append(span)
        
        return merged
    
    def _validate_pii(self, pii_type: str, value: str, text: str = "", start: int = 0, end: int = 0) -> bool:
        """Validate detected PII based on type-specific patterns and context."""
        import re
        
        validation_patterns = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "phone": r"^[\d\s\-\(\)\+\.]{10,}$",  # At least 10 digits/chars
            "ssn": r"^\d{3}[-]?\d{2}[-]?\d{4}$",
            "credit_card": r"^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$",
            "ip_address": r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$",
            "date_of_birth": r"^(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}$",
            "zip_code": r"^\d{5}(?:-\d{4})?$",
        }
        
        # Context-aware validation: check if number is in mathematical expression
        if pii_type == "zip_code" and text:
            # Check surrounding context (20 chars before and after)
            context_start = max(0, start - 20)
            context_end = min(len(text), end + 20)
            context = text[context_start:context_end].lower()
            
            # Mathematical operators that suggest this is NOT a ZIP code
            math_indicators = ['+', '-', '*', '/', '=', 'plus', 'minus', 'times', 'divided', 'equals', 'sum', 'add']
            # Check if any math operators are nearby
            if any(op in context for op in math_indicators):
                return False
            
            # Check if it's part of a larger number (like 102323)
            if start > 0 and text[start-1].isdigit():
                return False
            if end < len(text) and text[end].isdigit():
                return False
        
        # Skip validation for types without patterns (name, location)
        if pii_type not in validation_patterns:
            # Basic validation: not too short, contains letters/numbers
            return len(value) >= 2 and any(c.isalnum() for c in value)
        
        pattern = validation_patterns.get(pii_type)
        if pattern:
            return bool(re.match(pattern, value))
        
        return True

