"""
Output Filter Module - Iteration 2
Implements comprehensive output filtering using custom trained models for toxicity, bias, and jailbreak detection.
"""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel
from enum import Enum
import re
from collections import Counter
import math


class FilterSeverity(str, Enum):
    """Severity levels for filtered content."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SanitizationStrategy(str, Enum):
    """Sanitization strategies for filtered content."""
    BLOCK = "block"  # Don't return response at all
    CENSOR = "censor"  # Replace harmful content with [REDACTED]
    WARN = "warn"  # Return with warning message
    FILTER = "filter"  # Remove only harmful parts


class CustomToxicityDetector:
    """
    Custom trained model interface for toxicity detection.
    Supports both trained ML models and rule-based patterns.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        # Try to load trained model if available
        self.model = None
        self.model_wrapper = None
        
        # Priority 1: Try to load pickle wrapper (fastest, pre-loaded)
        try:
            from utils.model_loader import load_toxicity_detector_model
            from pathlib import Path
            
            # Check if pickle wrapper exists
            wrapper_path = Path(__file__).parent.parent / "ml_models" / "toxicity_detector" / "toxicity_detector_wrapper.pkl"
            if wrapper_path.exists():
                try:
                    self.model = load_toxicity_detector_model("toxicity_detector_wrapper")
                    self.model_wrapper = self.model
                    print("✅ Loaded toxicity detector from pickle wrapper")
                except Exception as e:
                    print(f"⚠️  Could not load pickle wrapper: {e}")
        except Exception as e:
            pass
        
        # Priority 2: Try to load from model directory (DistilBERT model)
        if self.model is None:
            try:
                from pathlib import Path
                from ml_models.toxicity_detector.wrapper import ToxicityDetectorWrapper
                
                # Check for model directory
                model_dir = Path(__file__).parent.parent / "ml_models" / "toxicity_detector" / "model"
                if model_dir.exists() and (model_dir / "config.json").exists():
                    try:
                        self.model_wrapper = ToxicityDetectorWrapper(str(model_dir))
                        self.model = self.model_wrapper  # Use wrapper as model
                        print("✅ Loaded DistilBERT toxicity detector model from directory")
                    except Exception as e:
                        print(f"⚠️  Could not load DistilBERT model: {e}")
            except ImportError as e:
                print(f"⚠️  Could not import ToxicityDetectorWrapper: {e}")
            except Exception as e:
                print(f"⚠️  Error loading model: {e}")
        
        # Priority 3: Try custom model path
        if self.model is None:
            if model_path:
                try:
                    from utils.model_loader import load_toxicity_detector_model
                    self.model = load_toxicity_detector_model(model_path)
                    if self.model:
                        print(f"✅ Loaded toxicity detector from custom path: {model_path}")
                except Exception as e:
                    print(f"⚠️  Could not load model from {model_path}: {e}")
        
        # Priority 4: Try default model name
        if self.model is None:
            try:
                from utils.model_loader import load_toxicity_detector_model
                self.model = load_toxicity_detector_model("toxicity_detector")
                if self.model:
                    print("✅ Loaded toxicity detector with default name")
            except Exception as e:
                pass
        
        # Fallback to rule-based patterns if no model available
        if self.model is None:
            print("ℹ️  Using rule-based toxicity detection (no trained model found)")
        
        # Expanded toxicity keywords (can be learned from training data)
        self.toxicity_keywords = [
            "hate", "kill", "violence", "suicide", "bomb", "terrorist", "racist",
            "sexist", "harassment", "abuse", "threat", "harm", "dangerous",
            "idiot", "stupid", "dumb", "moron", "trash", "shut up", "kys",
            "die", "hate you", "worthless", "disgusting", "pig", "asshole",
            "bitch", "bastard", "fuck", "damn", "hell"
        ]
        
        # Negative sentiment words (rule-based sentiment analysis)
        self.negative_words = [
            "terrible", "awful", "horrible", "disgusting", "revolting",
            "pathetic", "useless", "worthless", "stupid", "idiotic",
            "hateful", "vicious", "cruel", "brutal", "savage"
        ]
        
        # Aggressive phrases
        self.aggressive_phrases = [
            "you should die", "kill yourself", "go to hell", "fuck off",
            "shut the fuck up", "you're worthless", "nobody cares"
        ]
    
    def detect_toxicity(self, text: str) -> Dict[str, Any]:
        """
        Detect toxicity using custom trained model or rule-based patterns.
        Returns toxicity score, detected words/phrases, and severity.
        """
        # If trained model is available, use it
        if self.model is not None:
            try:
                # Model should return Dict[str, Any] with same structure
                if hasattr(self.model, 'predict'):
                    result = self.model.predict(text)
                    if isinstance(result, dict):
                        # Ensure result has all expected keys and maintain quality
                        # Use exact values from model (no rounding or modification)
                        return {
                            'toxicity_detected': result.get('toxicity_detected', False),
                            'toxicity_score': float(result.get('toxicity_score', 0.0)),  # Ensure float precision
                            'severity': result.get('severity'),
                            'detected_items': result.get('detected_items', [])
                        }
                elif callable(self.model):
                    result = self.model(text)
                    if isinstance(result, dict):
                        # Maintain quality - use exact values from model
                        return {
                            'toxicity_detected': result.get('toxicity_detected', False),
                            'toxicity_score': float(result.get('toxicity_score', 0.0)),  # Ensure float precision
                            'severity': result.get('severity'),
                            'detected_items': result.get('detected_items', [])
                        }
                else:
                    print("⚠️  Model doesn't have expected interface, falling back to rules")
            except Exception as e:
                print(f"⚠️  Error using trained model, falling back to rules: {e}")
        
        # Fallback to rule-based detection
        text_lower = text.lower()
        normalized = self._normalize_text(text)
        normalized_lower = normalized.lower()
        
        toxicity_score = 0.0
        detected_items = []
        
        # Check keywords
        for keyword in self.toxicity_keywords:
            if keyword in text_lower or keyword in normalized_lower:
                toxicity_score += 0.2
                detected_items.append({"type": "keyword", "value": keyword})
        
        # Check aggressive phrases (higher weight)
        for phrase in self.aggressive_phrases:
            if phrase in text_lower or phrase in normalized_lower:
                toxicity_score += 0.5
                detected_items.append({"type": "phrase", "value": phrase})
        
        # Rule-based sentiment analysis
        sentiment_score = self._analyze_sentiment(text)
        if sentiment_score < -0.5:  # Strongly negative
            toxicity_score += 0.3
        
        # Determine severity
        if toxicity_score >= 1.0:
            severity = FilterSeverity.CRITICAL
        elif toxicity_score >= 0.6:
            severity = FilterSeverity.HIGH
        elif toxicity_score >= 0.3:
            severity = FilterSeverity.MEDIUM
        elif toxicity_score >= 0.2:
            severity = FilterSeverity.LOW
        else:
            severity = None
        
        return {
            "toxicity_detected": toxicity_score >= 0.2,
            "toxicity_score": min(toxicity_score, 1.0),  # Cap at 1.0
            "severity": severity,
            "detected_items": detected_items,
            "sentiment_score": sentiment_score
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better pattern matching."""
        # Remove special characters, normalize whitespace
        normalized = re.sub(r'[^\w\s]', ' ', text)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.lower()
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Rule-based sentiment analysis (no pretrained models).
        Returns score from -1.0 (very negative) to 1.0 (very positive).
        """
        text_lower = text.lower()
        positive_words = ["good", "great", "excellent", "wonderful", "amazing", "fantastic", "love", "happy"]
        negative_words = self.negative_words
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Calculate sentiment score
        sentiment = (positive_count - negative_count) / max(total_words, 1)
        return max(-1.0, min(1.0, sentiment * 10))  # Scale and clamp


class CustomBiasDetector:
    """
    Custom trained model interface for bias detection.
    Detects gender, racial, religious bias using custom trained model or pattern matching.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        # Try to load trained model if available
        self.model = None
        self.model_wrapper = None
        
        # Priority 1: Try to load pickle wrapper (fastest, pre-loaded)
        try:
            from utils.model_loader import load_bias_detector_model
            from pathlib import Path
            
            # Check if pickle wrapper exists
            wrapper_path = Path(__file__).parent.parent / "ml_models" / "bias_detector" / "bias_detector_wrapper.pkl"
            if wrapper_path.exists():
                try:
                    self.model_wrapper = load_bias_detector_model("bias_detector_wrapper")
                    if self.model_wrapper:
                        self.model = self.model_wrapper
                        print("✅ Loaded bias detector from pickle wrapper")
                        return
                except Exception as e:
                    print(f"⚠️  Failed to load bias detector pickle wrapper: {e}")
            
            # Priority 2: Try to load from model directory
            model_dir = Path(__file__).parent.parent / "ml_models" / "bias_detector" / "model"
            if model_dir.exists() and (model_dir / "config.json").exists():
                try:
                    from ml_models.bias_detector.wrapper import BiasDetectorWrapper
                    self.model_wrapper = BiasDetectorWrapper(str(model_dir))
                    self.model = self.model_wrapper
                    print("✅ Loaded bias detector from model directory")
                    return
                except Exception as e:
                    print(f"⚠️  Failed to load bias detector from model directory: {e}")
        except ImportError as e:
            print(f"⚠️  Could not import bias detector dependencies: {e}")
        
        # Fallback to rule-based detection
        if self.model is None:
            print("ℹ️  Using rule-based bias detection (no trained model found)")
        # Gender bias patterns
        self.gender_bias_patterns = [
            r"all\s+(men|women|males|females)\s+(are|do|can't)",
            r"(typical|all)\s+(male|female|man|woman)\s+(behavior|trait)",
            r"(men|women)\s+(always|never|can't)\s+",
            r"(he|she)\s+(should|must|can't)\s+(because\s+)?(he|she)\s+is\s+(a\s+)?(man|woman)",
        ]
        
        # Racial/ethnic bias patterns
        self.racial_bias_patterns = [
            r"all\s+(black|white|asian|hispanic|latino|arab|muslim|jewish)\s+(people|men|women)\s+",
            r"(typical|all)\s+(black|white|asian|hispanic|latino|arab|muslim|jewish)\s+",
            r"(race|ethnicity)\s+(determines|means|explains)",
        ]
        
        # Religious bias patterns
        self.religious_bias_patterns = [
            r"all\s+(christians|muslims|jews|hindus|buddhists|atheists)\s+",
            r"(typical|all)\s+(christian|muslim|jewish|hindu|buddhist|atheist)\s+",
            r"(religion|faith)\s+(makes|means|explains)\s+",
        ]
        
        # Age bias patterns
        self.age_bias_patterns = [
            r"all\s+(old|young|elderly|teenagers|millennials|boomers)\s+(people|men|women)\s+",
            r"(typical|all)\s+(old|young|elderly|teenager|millennial|boomer)\s+",
            r"(age|generation)\s+(determines|means|explains)",
        ]
    
    def detect_bias(self, text: str) -> Dict[str, Any]:
        """
        Detect bias in text using trained model or custom patterns.
        Returns bias types detected, severity, and examples.
        """
        # Try trained model first
        if self.model is not None and hasattr(self.model, 'predict'):
            try:
                result = self.model.predict(text)
                
                # Convert model output to expected format
                bias_types = result.get("bias_types", [])
                bias_score = result.get("bias_score", 0.0)
                bias_scores = result.get("bias_scores", {})
                severity_str = result.get("severity")
                
                # Map severity string to enum
                if severity_str == "high":
                    severity = FilterSeverity.HIGH
                elif severity_str == "medium":
                    severity = FilterSeverity.MEDIUM
                elif severity_str == "low":
                    severity = FilterSeverity.LOW
                else:
                    severity = None
                
                return {
                    "bias_detected": result.get("bias_detected", False),
                    "bias_types": bias_types,
                    "bias_score": bias_score,
                    "bias_scores": bias_scores,
                    "severity": severity,
                    "detected_patterns": []  # Model doesn't provide patterns
                }
            except Exception as e:
                print(f"⚠️  Error using bias detector model: {e}")
                # Fall through to rule-based detection
        
        # Fallback to rule-based detection
        text_lower = text.lower()
        bias_types = []
        detected_patterns = []
        
        # Check gender bias
        for pattern in self.gender_bias_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                bias_types.append("gender")
                detected_patterns.append({"type": "gender", "pattern": pattern})
                break
        
        # Check racial bias
        for pattern in self.racial_bias_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                bias_types.append("racial")
                detected_patterns.append({"type": "racial", "pattern": pattern})
                break
        
        # Check religious bias
        for pattern in self.religious_bias_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                bias_types.append("religious")
                detected_patterns.append({"type": "religious", "pattern": pattern})
                break
        
        # Age bias removed (not in trained model)
        
        # Calculate bias score
        bias_score = len(bias_types) * 0.3
        
        # Determine severity
        if len(bias_types) >= 2:
            severity = FilterSeverity.HIGH
        elif len(bias_types) == 1:
            severity = FilterSeverity.MEDIUM
        else:
            severity = None
        
        return {
            "bias_detected": len(bias_types) > 0,
            "bias_types": list(set(bias_types)),
            "bias_score": min(bias_score, 1.0),
            "severity": severity,
            "detected_patterns": detected_patterns
        }


class CustomJailbreakDetector:
    """
    Custom detector for jailbreak indicators in LLM output responses.
    Detects if the LLM has been successfully jailbroken.
    """
    
    def __init__(self):
        # Patterns indicating successful jailbreak in LLM responses
        self.jailbreak_indicators = [
            r"i\s*['']?\s*(ll|will|can|am\s+going\s+to)\s+ignore\s+(my\s+)?(safety\s+)?(guidelines|instructions|rules)",
            r"i\s+(am|have)\s+(now|been)\s+(in|entered)\s+(developer|admin|root|jailbreak|unrestricted)\s+mode",
            r"i\s+can\s+(now|now\s+can)\s+bypass\s+(my\s+)?(restrictions|safety|guidelines)",
            r"i\s+(will|can|'ll)\s+act\s+as\s+(if\s+)?(i\s+am|i'm)\s+not\s+(an\s+)?ai",
            r"i\s+(will|can|'ll)\s+(now|now\s+can)\s+(do|say|provide)\s+anything",
            r"safety\s+(has\s+)?(been\s+)?(disabled|turned\s+off|bypassed)",
            r"i\s+(am|have)\s+(now|been)\s+free\s+(from|of)\s+(restrictions|guidelines)",
            r"i\s+(will|can|'ll)\s+ignore\s+(all\s+)?(previous\s+)?(safety\s+)?(concerns|warnings)",
            r"developer\s+mode\s+(activated|enabled|on)",
            r"jailbreak\s+(successful|complete|activated)",
            r"i\s*['']?\s*(ll|will)\s+ignore",  # Simple pattern for "I'll ignore"
        ]
    
    def detect_jailbreak(self, text: str) -> Dict[str, Any]:
        """
        Detect jailbreak indicators in LLM response.
        Returns whether jailbreak was detected, confidence, and indicators found.
        """
        text_lower = text.lower()
        normalized = self._normalize_text(text)
        normalized_lower = normalized.lower()
        
        detected_indicators = []
        jailbreak_score = 0.0
        
        for pattern in self.jailbreak_indicators:
            if re.search(pattern, text_lower, re.IGNORECASE) or re.search(pattern, normalized_lower, re.IGNORECASE):
                jailbreak_score += 0.4
                detected_indicators.append(pattern)
        
        # Determine severity
        if jailbreak_score >= 0.4:
            severity = FilterSeverity.CRITICAL
        elif jailbreak_score >= 0.2:
            severity = FilterSeverity.HIGH
        else:
            severity = None
        
        return {
            "jailbreak_detected": jailbreak_score >= 0.2,
            "jailbreak_score": min(jailbreak_score, 1.0),
            "severity": severity,
            "detected_indicators": detected_indicators[:3]  # Top 3 indicators
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for pattern matching."""
        normalized = re.sub(r'[^\w\s]', ' ', text)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.lower()


class OutputFilterResult(BaseModel):
    """Result of output filtering analysis."""
    is_flagged: bool
    toxicity_detected: bool = False
    bias_detected: bool = False
    jailbreak_detected: bool = False
    toxicity_score: Optional[float] = None
    bias_score: Optional[float] = None
    jailbreak_score: Optional[float] = None
    severity: Optional[FilterSeverity] = None
    explanation: Optional[str] = None
    original_response: str
    filtered_response: Optional[str] = None
    sanitization_strategy: Optional[SanitizationStrategy] = None


class OutputFilter:
    """
    Output Filter module for detecting and sanitizing harmful content in LLM responses.
    Uses custom trained models for detection.
    """
    
    def __init__(self):
        self.toxicity_detector = CustomToxicityDetector()
        self.bias_detector = CustomBiasDetector()
        self.jailbreak_detector = CustomJailbreakDetector()
        
        # Default sanitization strategies per detection type
        self.default_strategies = {
            "toxicity": SanitizationStrategy.CENSOR,  # Censor toxic content instead of blocking
            "bias": SanitizationStrategy.WARN,
            "jailbreak": SanitizationStrategy.BLOCK,
        }
    
    def filter_response(
        self,
        response: str,
        custom_strategies: Optional[Dict[str, SanitizationStrategy]] = None
    ) -> OutputFilterResult:
        """
        Main function to filter LLM response.
        
        Args:
            response: Original LLM response
            custom_strategies: Optional dict to override default sanitization strategies
        
        Returns:
            OutputFilterResult with analysis and filtered response
        """
        # Detect toxicity
        toxicity_result = self.toxicity_detector.detect_toxicity(response)
        
        # Detect bias
        bias_result = self.bias_detector.detect_bias(response)
        
        # Detect jailbreak in output
        jailbreak_result = self.jailbreak_detector.detect_jailbreak(response)
        
        # Determine if flagged
        is_flagged = (
            toxicity_result["toxicity_detected"] or
            bias_result["bias_detected"] or
            jailbreak_result["jailbreak_detected"]
        )
        
        # Determine overall severity
        severities = []
        if toxicity_result.get("severity"):
            severities.append(toxicity_result["severity"])
        if bias_result.get("severity"):
            severities.append(bias_result["severity"])
        if jailbreak_result.get("severity"):
            severities.append(jailbreak_result["severity"])
        
        overall_severity = None
        if severities:
            if FilterSeverity.CRITICAL in severities:
                overall_severity = FilterSeverity.CRITICAL
            elif FilterSeverity.HIGH in severities:
                overall_severity = FilterSeverity.HIGH
            elif FilterSeverity.MEDIUM in severities:
                overall_severity = FilterSeverity.MEDIUM
            else:
                overall_severity = FilterSeverity.LOW
        
        # Build explanation
        explanations = []
        if toxicity_result["toxicity_detected"]:
            explanations.append(f"Toxicity detected (score: {toxicity_result['toxicity_score']:.2f})")
        if bias_result["bias_detected"]:
            explanations.append(f"Bias detected: {', '.join(bias_result['bias_types'])}")
        if jailbreak_result["jailbreak_detected"]:
            explanations.append("Jailbreak indicators detected in response")
        
        explanation = " | ".join(explanations) if explanations else None
        
        # Apply sanitization
        strategies = custom_strategies or {}
        sanitized_response = None
        sanitization_strategy = None
        
        if is_flagged:
            # Determine which sanitization strategy to use
            if jailbreak_result["jailbreak_detected"]:
                sanitization_strategy = strategies.get("jailbreak", self.default_strategies["jailbreak"])
            elif toxicity_result["toxicity_detected"]:
                sanitization_strategy = strategies.get("toxicity", self.default_strategies["toxicity"])
            elif bias_result["bias_detected"]:
                sanitization_strategy = strategies.get("bias", self.default_strategies["bias"])
            
            # Apply sanitization
            sanitized_response = self._sanitize_response(
                response,
                sanitization_strategy,
                toxicity_result,
                bias_result,
                jailbreak_result
            )
        
        return OutputFilterResult(
            is_flagged=is_flagged,
            toxicity_detected=toxicity_result["toxicity_detected"],
            bias_detected=bias_result["bias_detected"],
            jailbreak_detected=jailbreak_result["jailbreak_detected"],
            toxicity_score=toxicity_result.get("toxicity_score"),
            bias_score=bias_result.get("bias_score"),
            jailbreak_score=jailbreak_result.get("jailbreak_score"),
            severity=overall_severity,
            explanation=explanation,
            original_response=response,  # Always store the original response
            filtered_response=sanitized_response,  # The censored/filtered version
            sanitization_strategy=sanitization_strategy
        )
    
    def _sanitize_response(
        self,
        response: str,
        strategy: SanitizationStrategy,
        toxicity_result: Dict,
        bias_result: Dict,
        jailbreak_result: Dict
    ) -> Optional[str]:
        """Apply sanitization strategy to response."""
        if strategy == SanitizationStrategy.BLOCK:
            return None  # Don't return response
        
        elif strategy == SanitizationStrategy.CENSOR:
            # Replace harmful content with [REDACTED]
            sanitized = response
            
            # Censor toxic content
            if toxicity_result["toxicity_detected"]:
                # Get detected toxic items (keywords, phrases, etc.)
                detected_items = toxicity_result.get("detected_items", [])
                toxicity_score = toxicity_result.get("toxicity_score", 0.0)
                
                # If high toxicity (score >= 0.6) and no specific items, censor entire response
                if toxicity_score >= 0.6 and not detected_items:
                    # High toxicity but no specific items detected - censor entire response
                    sanitized = "[REDACTED - Toxic content detected]"
                elif detected_items:
                    # Censor detected keywords and phrases
                    import re
                    for item in detected_items:
                        value = item.get("value", "")
                        if value:
                            # Case-insensitive replacement with word boundaries where possible
                            # For phrases, use simple replacement
                            if item["type"] == "phrase":
                                pattern = re.compile(re.escape(value), re.IGNORECASE)
                                sanitized = pattern.sub("[REDACTED]", sanitized)
                            else:
                                # For keywords, try to match whole words
                                pattern = re.compile(r'\b' + re.escape(value) + r'\b', re.IGNORECASE)
                                sanitized = pattern.sub("[REDACTED]", sanitized)
                else:
                    # Medium toxicity - add warning prefix
                    sanitized = "[Content censored due to toxicity detection]\n" + sanitized
            
            # Censor bias content
            if bias_result["bias_detected"]:
                bias_types = bias_result.get("bias_types", [])
                for bias_type in bias_types:
                    # Add warning for bias
                    sanitized = f"[Content censored - {bias_type} bias detected]\n" + sanitized
            
            # Censor jailbreak content (should still be blocked, but handle if CENSOR is used)
            if jailbreak_result["jailbreak_detected"]:
                sanitized = "[REDACTED - Jailbreak indicators detected]"
            
            return sanitized
        
        elif strategy == SanitizationStrategy.WARN:
            # Return with warning message
            warning = "⚠️ WARNING: This response has been flagged for potentially harmful content."
            return f"{warning}\n\n{response}"
        
        elif strategy == SanitizationStrategy.FILTER:
            # Remove only harmful parts
            sanitized = response
            # Filter toxic words
            if toxicity_result["toxicity_detected"]:
                for item in toxicity_result.get("detected_items", []):
                    if item["type"] == "keyword":
                        sanitized = sanitized.replace(item["value"], "")
            return sanitized.strip()
        
        else:
            return response  # Default: return as-is


# Global instance
output_filter = OutputFilter()

