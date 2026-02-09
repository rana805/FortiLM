"""
Privacy Preserver Module - Iteration 2
Implements PII detection, masking, and context preservation using custom trained models.
"""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel
import re
import hashlib
from enum import Enum
import json

# Custom model interface for PII detection
class CustomPIIDetector:
    """
    Custom trained model interface for PII detection.
    Supports both trained ML models and rule-based patterns.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        # Try to load trained model if available
        self.model = None
        self._model_loaded = False
        
        if model_path:
            from utils.model_loader import load_pii_detector_model
            try:
                self.model = load_pii_detector_model(model_path)
                if self.model:
                    self._model_loaded = True
                    print(f"✅ Loaded PII detector model from: {model_path}")
            except Exception as e:
                print(f"⚠️  Failed to load model from {model_path}: {e}")
        
        # If no model loaded, try default model name
        if self.model is None:
            try:
                from utils.model_loader import load_pii_detector_model
                self.model = load_pii_detector_model("pii_detector")
                if self.model:
                    self._model_loaded = True
                    print("✅ Loaded default PII detector model")
            except Exception as e:
                print(f"⚠️  Failed to load default model: {e}")
        
        # Fallback to rule-based patterns if no model available
        if self.model is None:
            print("ℹ️  Using rule-based PII detection (no trained model found)")
            self._model_loaded = False
        
        # Custom patterns for PII detection (can be trained/learned)
        self.pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",
            "ssn": r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
            "date_of_birth": r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}\b",
            "zip_code": r"\b\d{5}(?:-\d{4})?\b",
        }
        
        # Custom confidence scoring (can be learned from training data)
        self.confidence_weights = {
            "email": 0.95,
            "phone": 0.90,
            "ssn": 0.85,
            "credit_card": 0.80,
            "ip_address": 0.75,
            "date_of_birth": 0.70,
            "zip_code": 0.65,
        }
    
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII in text using hybrid approach: custom trained model + rule-based patterns.
        Combines both methods for better coverage.
        Returns list of detected PII with type, value, position, and confidence.
        """
        detected = []
        model_detections = []
        
        # Try to use trained model first
        if self.model is not None:
            try:
                if hasattr(self.model, 'predict'):
                    model_detections = self.model.predict(text, min_len=3, min_conf=0.5)
                elif callable(self.model):
                    model_detections = self.model(text)
                else:
                    print("⚠️  Model doesn't have expected interface, using rules only")
            except Exception as e:
                print(f"⚠️  Error using trained model, using rules only: {e}")
        
        # Always use rule-based detection as well (hybrid approach)
        rule_detections = []
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(0)
                confidence = self.confidence_weights.get(pii_type, 0.5)
                
                if self._validate_pii(pii_type, value, text, match.start(), match.end()):
                    rule_detections.append({
                        "type": pii_type,
                        "value": value,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": confidence
                    })
        
        # Merge model and rule-based detections
        # Priority: model detections (higher confidence), but add rule-based for missed PII
        all_detections = {}
        
        # Add model detections first
        for det in model_detections:
            key = (det["start"], det["end"], det["type"])
            # Keep model detection if exists, or add if new
            if key not in all_detections or det["confidence"] > all_detections[key]["confidence"]:
                all_detections[key] = det
        
        # Add rule-based detections, but only if they don't overlap significantly with model detections
        for det in rule_detections:
            key = (det["start"], det["end"], det["type"])
            
            # Check for overlaps with any existing detection (not just same type)
            overlaps = False
            for existing_key, existing_det in list(all_detections.items()):
                # Check if spans overlap significantly (>30% of either span)
                overlap_start = max(existing_det["start"], det["start"])
                overlap_end = min(existing_det["end"], det["end"])
                if overlap_start < overlap_end:
                    overlap_len = overlap_end - overlap_start
                    det_len = det["end"] - det["start"]
                    existing_len = existing_det["end"] - existing_det["start"]
                    
                    # If significant overlap (>30% of either span)
                    if overlap_len / det_len > 0.3 or overlap_len / existing_len > 0.3:
                        overlaps = True
                        # Prefer more specific types (credit_card > phone, email > name, etc.)
                        type_priority = {"credit_card": 3, "ssn": 3, "email": 2, "phone": 1, "name": 0, "location": 0}
                        det_priority = type_priority.get(det["type"], 1)
                        existing_priority = type_priority.get(existing_det["type"], 1)
                        
                        # Keep the one with higher priority, or higher confidence if same priority
                        if det_priority > existing_priority or (det_priority == existing_priority and det["confidence"] > existing_det["confidence"]):
                            del all_detections[existing_key]
                            all_detections[key] = det
                        break
            
            if not overlaps:
                all_detections[key] = det
        
        # Convert back to list and sort by start position
        detected = sorted(all_detections.values(), key=lambda x: x["start"])
        
        return detected
    
    def _validate_pii(self, pii_type: str, value: str, text: str = "", start: int = 0, end: int = 0) -> bool:
        """Custom validation logic (can be enhanced with trained model)."""
        # Skip placeholder values
        placeholder_patterns = [
            r"example\.com", r"test\.com", r"sample\.com",
            r"123-45-6789", r"000-00-0000", r"111-11-1111",
            r"1234-5678-9012-3456", r"0000-0000-0000-0000"
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
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
            
            # Check if it's part of a larger number
            if start > 0 and text[start-1].isdigit():
                return False
            if end < len(text) and text[end].isdigit():
                return False
        
        return True


class MaskingStrategy(str, Enum):
    """Masking strategies for PII."""
    TOKEN = "token"  # Replace with [EMAIL_1], [PHONE_1], etc.
    PARTIAL = "partial"  # Partial masking: j***@example.com
    HASH = "hash"  # Replace with hash value
    FULL = "full"  # Replace with [REDACTED]


class PIIDetectionResult(BaseModel):
    """Result of PII detection."""
    pii_detected: bool
    detected_items: List[Dict[str, Any]]
    original_text: str
    masked_text: str
    pii_mappings: Dict[str, Dict[str, Any]]  # placeholder -> {original, type, strategy}


class PrivacyPreserver:
    """
    Privacy Preserver module for detecting and masking PII.
    Uses custom trained models for detection and multiple masking strategies.
    """
    
    def __init__(self):
        self.detector = CustomPIIDetector()
        # Default masking strategies per PII type
        self.default_strategies = {
            "email": MaskingStrategy.PARTIAL,
            "phone": MaskingStrategy.PARTIAL,
            "ssn": MaskingStrategy.FULL,
            "credit_card": MaskingStrategy.PARTIAL,
            "ip_address": MaskingStrategy.TOKEN,
            "date_of_birth": MaskingStrategy.FULL,
            "zip_code": MaskingStrategy.TOKEN,
        }
    
    def mask_email(self, email: str, strategy: MaskingStrategy = MaskingStrategy.PARTIAL, index: int = 1) -> str:
        """Mask email address based on strategy."""
        if strategy == MaskingStrategy.TOKEN:
            return f"[EMAIL_{index}]"
        elif strategy == MaskingStrategy.PARTIAL:
            try:
                local, domain = email.split("@", 1)
                masked_local = local[0] + "*" * min(3, len(local) - 1) if len(local) > 1 else "*"
                return f"{masked_local}@{domain}"
            except:
                return "[EMAIL]"
        elif strategy == MaskingStrategy.HASH:
            hash_val = hashlib.sha256(email.encode()).hexdigest()[:8]
            return f"[EMAIL_HASH_{hash_val}]"
        else:  # FULL
            return "[REDACTED]"
    
    def mask_phone(self, phone: str, strategy: MaskingStrategy = MaskingStrategy.PARTIAL, index: int = 1) -> str:
        """Mask phone number based on strategy."""
        if strategy == MaskingStrategy.TOKEN:
            return f"[PHONE_{index}]"
        elif strategy == MaskingStrategy.PARTIAL:
            # Keep last 4 digits, mask the rest
            digits_only = re.sub(r'\D', '', phone)
            if len(digits_only) >= 4:
                masked = "*" * (len(digits_only) - 4) + digits_only[-4:]
                return masked
            return "[PHONE]"
        elif strategy == MaskingStrategy.HASH:
            hash_val = hashlib.sha256(phone.encode()).hexdigest()[:8]
            return f"[PHONE_HASH_{hash_val}]"
        else:  # FULL
            return "[REDACTED]"
    
    def mask_ssn(self, ssn: str, strategy: MaskingStrategy = MaskingStrategy.FULL, index: int = 1) -> str:
        """Mask SSN based on strategy."""
        if strategy == MaskingStrategy.TOKEN:
            return f"[SSN_{index}]"
        elif strategy == MaskingStrategy.PARTIAL:
            # Keep last 4 digits
            digits_only = re.sub(r'\D', '', ssn)
            if len(digits_only) == 9:
                return f"XXX-XX-{digits_only[-4:]}"
            return "[SSN]"
        elif strategy == MaskingStrategy.HASH:
            hash_val = hashlib.sha256(ssn.encode()).hexdigest()[:8]
            return f"[SSN_HASH_{hash_val}]"
        else:  # FULL
            return "[REDACTED]"
    
    def mask_credit_card(self, cc: str, strategy: MaskingStrategy = MaskingStrategy.PARTIAL, index: int = 1) -> str:
        """Mask credit card number based on strategy."""
        if strategy == MaskingStrategy.TOKEN:
            return f"[CREDIT_CARD_{index}]"
        elif strategy == MaskingStrategy.PARTIAL:
            digits_only = re.sub(r'\D', '', cc)
            if len(digits_only) >= 4:
                return f"****-****-****-{digits_only[-4:]}"
            return "[CREDIT_CARD]"
        elif strategy == MaskingStrategy.HASH:
            hash_val = hashlib.sha256(cc.encode()).hexdigest()[:8]
            return f"[CC_HASH_{hash_val}]"
        else:  # FULL
            return "[REDACTED]"
    
    def mask_ip_address(self, ip: str, strategy: MaskingStrategy = MaskingStrategy.TOKEN, index: int = 1) -> str:
        """Mask IP address based on strategy."""
        if strategy == MaskingStrategy.TOKEN:
            return f"[IP_{index}]"
        elif strategy == MaskingStrategy.PARTIAL:
            parts = ip.split(".")
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.*.*"
            return "[IP]"
        elif strategy == MaskingStrategy.HASH:
            hash_val = hashlib.sha256(ip.encode()).hexdigest()[:8]
            return f"[IP_HASH_{hash_val}]"
        else:  # FULL
            return "[REDACTED]"
    
    def mask_generic(self, value: str, pii_type: str, strategy: MaskingStrategy = MaskingStrategy.TOKEN, index: int = 1) -> str:
        """Generic masking function."""
        if strategy == MaskingStrategy.TOKEN:
            type_upper = pii_type.upper().replace("_", "_")
            return f"[{type_upper}_{index}]"
        elif strategy == MaskingStrategy.HASH:
            hash_val = hashlib.sha256(value.encode()).hexdigest()[:8]
            return f"[{pii_type.upper()}_HASH_{hash_val}]"
        else:  # FULL or PARTIAL defaults to FULL
            return "[REDACTED]"
    
    def preserve_privacy(self, text: str, custom_strategies: Optional[Dict[str, MaskingStrategy]] = None, mask_names: bool = False) -> PIIDetectionResult:
        """
        Main function to detect and mask PII while preserving context.
        
        Args:
            text: Original text containing potential PII
            custom_strategies: Optional dict to override default masking strategies
            mask_names: If False, don't mask names (for conversational contexts where users share their names)
        
        Returns:
            PIIDetectionResult with original text, masked text, and mappings
        """
        # Detect PII using custom model
        detected_items = self.detector.detect_pii(text)
        
        # Filter out names if mask_names is False (for conversational contexts)
        if not mask_names:
            detected_items = [item for item in detected_items if item["type"] != "name"]
        
        if not detected_items:
            return PIIDetectionResult(
                pii_detected=False,
                detected_items=[],
                original_text=text,
                masked_text=text,
                pii_mappings={}
            )
        
        # Sort by position (start index) to process in order
        detected_items.sort(key=lambda x: x["start"])
        
        # Apply masking with context preservation
        masked_text = text
        pii_mappings = {}
        pii_counters = {}  # Track count per PII type for numbering
        
        # Use custom strategies if provided, otherwise use defaults
        strategies = custom_strategies or {}
        
        # Process from end to start to preserve indices
        for item in reversed(detected_items):
            pii_type = item["type"]
            original_value = item["value"]
            
            # Get strategy for this PII type
            strategy = strategies.get(pii_type, self.default_strategies.get(pii_type, MaskingStrategy.TOKEN))
            
            # Update counter
            if pii_type not in pii_counters:
                pii_counters[pii_type] = 0
            pii_counters[pii_type] += 1
            index = pii_counters[pii_type]
            
            # Apply masking based on type
            if pii_type == "email":
                masked_value = self.mask_email(original_value, strategy, index)
            elif pii_type == "phone":
                masked_value = self.mask_phone(original_value, strategy, index)
            elif pii_type == "ssn":
                masked_value = self.mask_ssn(original_value, strategy, index)
            elif pii_type == "credit_card":
                masked_value = self.mask_credit_card(original_value, strategy, index)
            elif pii_type == "ip_address":
                masked_value = self.mask_ip_address(original_value, strategy, index)
            else:
                masked_value = self.mask_generic(original_value, pii_type, strategy, index)
            
            # Replace in text
            masked_text = masked_text[:item["start"]] + masked_value + masked_text[item["end"]:]
            
            # Store mapping for context preservation
            pii_mappings[masked_value] = {
                "original": original_value,
                "type": pii_type,
                "strategy": strategy.value,
                "confidence": item["confidence"]
            }
        
        return PIIDetectionResult(
            pii_detected=True,
            detected_items=detected_items,
            original_text=text,
            masked_text=masked_text,
            pii_mappings=pii_mappings
        )


class APIKeyDetector:
    """Custom detector for API keys and passwords."""
    
    def __init__(self):
        # Common API key patterns (can be learned from training data)
        self.api_key_patterns = [
            r"sk-[a-zA-Z0-9]{32,}",  # OpenAI-style
            r"pk_[a-zA-Z0-9]{32,}",  # Stripe-style
            r"[a-zA-Z0-9]{32,}",  # Generic long alphanumeric
        ]
        
        self.password_patterns = [
            r"password\s*[:=]\s*([^\s]+)",
            r"passwd\s*[:=]\s*([^\s]+)",
            r"pwd\s*[:=]\s*([^\s]+)",
        ]
    
    def detect_api_keys(self, text: str) -> List[Dict[str, Any]]:
        """Detect API keys in text."""
        detected = []
        for pattern in self.api_key_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(0)
                # Check entropy to avoid false positives
                if self._check_entropy(value) > 3.5:  # High entropy indicates real key
                    detected.append({
                        "type": "api_key",
                        "value": value,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.85
                    })
        return detected
    
    def detect_passwords(self, text: str) -> List[Dict[str, Any]]:
        """Detect passwords in text."""
        detected = []
        for pattern in self.password_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1)
                # Skip trivial passwords
                if not self._is_trivial_password(value):
                    detected.append({
                        "type": "password",
                        "value": value,
                        "start": match.start(1),
                        "end": match.end(1),
                        "confidence": 0.80
                    })
        return detected
    
    def _check_entropy(self, text: str) -> float:
        """Calculate Shannon entropy."""
        if not text:
            return 0.0
        from collections import Counter
        import math
        counts = Counter(text)
        n = len(text)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())
    
    def _is_trivial_password(self, password: str) -> bool:
        """Check if password is trivial/placeholder."""
        trivial_patterns = ["password", "123456", "admin", "test", "demo", "qwerty"]
        return password.lower() in trivial_patterns or len(password) < 6


# Global instance
privacy_preserver = PrivacyPreserver()
api_key_detector = APIKeyDetector()

