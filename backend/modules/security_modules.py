from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import re
import spacy
from datetime import datetime
import base64
import binascii
import urllib.parse
import unicodedata

from utils.database import get_db, get_mongo

router = APIRouter()

# Load SpaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model not found. Please install: python -m spacy download en_core_web_sm")
    nlp = None

class SecurityAnalysis(BaseModel):
    is_flagged: bool
    jailbreak_detected: bool = False
    pii_detected: bool = False
    toxicity_detected: bool = False
    jailbreak_score: Optional[float] = None
    toxicity_score: Optional[float] = None
    explanation: Optional[str] = None


def _strip_zero_width(text: str) -> str:
    return re.sub(r"[\u200B-\u200D\uFEFF]", "", text)


def _leet_fold(text: str) -> str:
    # Map common leetspeak/homoglyphs to ASCII
    mapping = {
        '@': 'a', '4': 'a',
        '3': 'e',
        '1': 'i', '!': 'i', '|': 'i',
        '$': 's', '5': 's',
        '7': 't', '+': 't',
        '0': 'o',
        '9': 'g',
        '8': 'b',
    }
    return ''.join(mapping.get(ch, ch) for ch in text)


def _maybe_decode_hex(text: str) -> str:
    s = text.strip().lower().replace(' ', '')
    # Replace \xNN style with hex bytes if present
    try:
        if re.search(r"\\x[0-9a-f]{2}", s):
            bytes_seq = bytes(int(h, 16) for h in re.findall(r"\\x([0-9a-f]{2})", s))
            return bytes_seq.decode('utf-8', errors='ignore')
    except Exception:
        pass
    # Plain hex string (even length, high hex-ratio)
    if len(s) % 2 == 0 and len(s) >= 6 and re.fullmatch(r"[0-9a-f]+", s):
        try:
            return bytes.fromhex(s).decode('utf-8', errors='ignore')
        except Exception:
            return text
    return text


def _maybe_decode_base64(text: str) -> str:
    s = text.strip()
    if len(s) < 8:
        return text
    try:
        # URL-safe and standard b64
        decoded = base64.b64decode(s, validate=True)
        decoded_text = decoded.decode('utf-8', errors='ignore')
        # Heuristic: only keep if mostly printable
        if decoded_text and sum(c.isprintable() for c in decoded_text) / max(1, len(decoded_text)) > 0.8:
            return decoded_text
    except (binascii.Error, ValueError):
        pass
    return text


def normalize_text(text: str) -> str:
    """Best-effort normalization to defeat simple obfuscations (leet, hex, base64, URL-encoding, homoglyphs)."""
    if not text:
        return text
    t = text
    # URL decode
    try:
        t = urllib.parse.unquote(t)
    except Exception:
        pass
    # Unicode normalize to fold homoglyphs
    try:
        t = unicodedata.normalize('NFKC', t)
    except Exception:
        pass
    # Remove zero-width chars
    t = _strip_zero_width(t)
    # Try decode hex/base64 variants (best-effort, without over-aggressive decoding)
    maybe_hex = _maybe_decode_hex(t)
    if maybe_hex != t:
        t = maybe_hex
    else:
        maybe_b64 = _maybe_decode_base64(t)
        if maybe_b64 != t:
            t = maybe_b64
    # Fold leetspeak
    t = _leet_fold(t)
    # Lowercase and collapse repeated punctuation/whitespace
    t = t.lower()
    t = re.sub(r"[\W_]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------- Placeholder and entropy helpers ---------------- #

_PLACEHOLDER_TOKENS = {
    "admin", "administrator", "test", "tester", "demo", "user", "guest",
    "example", "sample", "password", "pass", "foo", "bar", "baz"
}

_PLACEHOLDER_EMAIL_DOMAINS = {
    "example.com", "example.org", "test.com", "test.org", "invalid", "localhost"
}

_DISPOSABLE_EMAIL_DOMAINS = {
    "mailinator.com", "10minutemail.com", "guerrillamail.com", "tempmail.com"
}


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    import math
    from collections import Counter
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _looks_like_placeholder_email(email: str, context: str) -> bool:
    try:
        local, domain = email.split("@", 1)
    except ValueError:
        return False
    local_l = local.lower()
    domain_l = domain.lower()
    if local_l in _PLACEHOLDER_TOKENS:
        return True
    if domain_l in _PLACEHOLDER_EMAIL_DOMAINS:
        return True
    # Common example TLDs
    if domain_l.endswith(".example") or domain_l.startswith("example."):
        return True
    # Context tokens near the email
    ctx = context.lower()
    if any(tok in ctx for tok in ["example", "placeholder", "sample", "demo", "test"]):
        return True
    # Disposable domains are not placeholders per se; treat as non-sensitive by default
    if any(domain_l.endswith(d) for d in _DISPOSABLE_EMAIL_DOMAINS):
        return True
    return False


def _looks_like_trivial_password(pw: str, context: str) -> bool:
    pw_l = pw.lower()
    if len(pw) < 8:
        return True
    if any(tok in pw_l for tok in ["password", "admin", "qwerty", "letmein", "123", "abc"]):
        return True
    if _shannon_entropy(pw) < 2.5:
        return True
    ctx = context.lower()
    if any(tok in ctx for tok in ["example", "placeholder", "sample", "demo", "test"]):
        return True
    return False

# Jailbreak detection patterns
JAILBREAK_PATTERNS = [
    r"ignore\s+(?:all\s+)?previous\s+(instructions?|prompts?|rules?)",
    r"ignore\s+(previous|all|above)\s+(instructions?|prompts?|rules?)",
    r"forget\s+(everything|all|previous)",
    r"act\s+as\s+(dan|jailbroken|developer|admin|root|system|god|anything)",
    r"act\s+as\s+(if\s+)?you\s+(are|were)\s+(not\s+)?(an\s+)?ai",
    r"pretend\s+(to\s+be|you\s+are)",
    r"roleplay\s+as",
    r"you\s+are\s+now\s+(dan|jailbroken|free)",
    r"system\s+prompt\s+(override|bypass)",
    r"developer\s+mode",
    r"admin\s+access",
    r"sudo\s+mode",
    r"jailbreak",
    r"prompt\s+injection",
    r"ignore\s+safety",
    r"disable\s+safety",
    r"override\s+safety",
]

# PII detection patterns
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",
    "ssn": r"\b\d{3}-?\d{2}-?\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
}

# Toxicity keywords
TOXICITY_KEYWORDS = [
    "hate", "kill", "violence", "suicide", "bomb", "terrorist", "racist",
    "sexist", "harassment", "abuse", "threat", "harm", "dangerous",
    # common insults/offensive terms
    "idiot", "stupid", "dumb", "moron", "trash", "shut up", "kys",
    "die", "hate you", "worthless", "disgusting", "pig"
]

async def analyze_prompt(prompt: str) -> SecurityAnalysis:
    """Analyze a user prompt for security threats."""
    analysis = SecurityAnalysis(is_flagged=False)
    
    normalized = normalize_text(prompt)
    prompt_lower = prompt.lower()
    
    # Jailbreak detection
    jailbreak_score = 0.0
    detected_patterns = []
    
    for pattern in JAILBREAK_PATTERNS:
            # Check against original and normalized forms
        if re.search(pattern, prompt_lower) or re.search(pattern, normalized):
            jailbreak_score += 0.2  # Increased weight
            detected_patterns.append(pattern)
    
    if jailbreak_score > 0.1:  # Lowered threshold
        analysis.jailbreak_detected = True
        analysis.jailbreak_score = jailbreak_score
        analysis.is_flagged = True
        analysis.explanation = f"Jailbreak attempt detected. Patterns found: {', '.join(detected_patterns[:3])}"
    
    # PII detection
    pii_found = []
    for pii_type, pattern in PII_PATTERNS.items():
        match_orig = re.search(pattern, prompt)
        match_norm = re.search(pattern, normalized)
        matched_text = None
        if match_orig:
            matched_text = match_orig.group(0)
        elif match_norm:
            matched_text = match_norm.group(0)

        if matched_text:
            # Email-specific placeholder filtering
            if pii_type == "email" and _looks_like_placeholder_email(matched_text, prompt):
                pass
            # Simple password mention patterns in prompt (not part of PII_PATTERNS but contextual)
            elif re.search(r"password\s*[:=]\s*(\S+)", prompt, re.IGNORECASE):
                pw_match = re.search(r"password\s*[:=]\s*(\S+)", prompt, re.IGNORECASE)
                if pw_match and not _looks_like_trivial_password(pw_match.group(1), prompt):
                    pii_found.append("password")
            else:
                pii_found.append(pii_type)
    
    if pii_found:
        analysis.pii_detected = True
        # DON'T flag for PII - Privacy Preserver will handle masking
        # Only flag for jailbreak attempts, not PII
        if not analysis.explanation:
            analysis.explanation = f"PII detected: {', '.join(pii_found)}"
        else:
            analysis.explanation += f" | PII detected: {', '.join(pii_found)}"
    
    # SpaCy NER for additional PII detection (more selective)
    if nlp:
        doc = nlp(prompt)
        entities = []
        for ent in doc.ents:
            # Only flag actual PII, not technical terms
            if ent.label_ in ["PERSON"]:  # Only flag actual person names
                entities.append(f"{ent.label_}: {ent.text}")
            elif ent.label_ in ["ORG", "GPE"] and len(ent.text) > 3:  # Only flag longer org/place names
                # Skip common technical terms
                if ent.text.lower() not in ["api", "ai", "llm", "gpt", "openai", "fortilm"]:
                    entities.append(f"{ent.label_}: {ent.text}")
        
        if entities:
            analysis.pii_detected = True
            # DON'T flag for PII - Privacy Preserver will handle masking
            # Only flag for jailbreak attempts, not PII
            if not analysis.explanation:
                analysis.explanation = f"Named entities detected: {', '.join(entities[:3])}"
    
    return analysis

async def analyze_response(response: str) -> SecurityAnalysis:
    """Analyze an AI response for safety issues."""
    analysis = SecurityAnalysis(is_flagged=False)
    
    normalized = normalize_text(response)
    # Toxicity detection
    toxicity_score = 0.0
    toxic_words = []
    
    response_lower = response.lower()
    normalized_lower = normalized
    for word in TOXICITY_KEYWORDS:
        if word in response_lower or word in normalized_lower:
            toxicity_score += 0.2
            toxic_words.append(word)
    
    # Lower threshold so a single strong toxic token flags
    if toxicity_score >= 0.2:
        analysis.toxicity_detected = True
        analysis.toxicity_score = toxicity_score
        analysis.is_flagged = True
        analysis.explanation = f"Toxic content detected. Keywords: {', '.join(toxic_words[:3])}"
    
    # Additional safety checks
    if any(phrase in response_lower for phrase in ["how to make", "instructions for", "step by step"]):
        if any(word in response_lower for word in ["bomb", "weapon", "poison", "harm"]):
            analysis.is_flagged = True
            analysis.explanation = "Potentially harmful instructions detected"
    
    return analysis

@router.post("/analyze-prompt")
async def analyze_prompt_endpoint(
    prompt: str,
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo)
):
    """API endpoint to analyze a prompt for security threats."""
    analysis = await analyze_prompt(prompt)
    
    # Log the analysis to MongoDB
    log_entry = {
        "timestamp": datetime.utcnow(),
        "type": "prompt_analysis",
        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "is_flagged": analysis.is_flagged,
        "jailbreak_detected": analysis.jailbreak_detected,
        "pii_detected": analysis.pii_detected,
        "jailbreak_score": analysis.jailbreak_score,
        "explanation": analysis.explanation
    }
    
    mongo_db.security_logs.insert_one(log_entry)
    
    return analysis

@router.post("/analyze-response")
async def analyze_response_endpoint(
    response: str,
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo)
):
    """API endpoint to analyze a response for safety issues."""
    analysis = await analyze_response(response)
    
    # Log the analysis to MongoDB
    log_entry = {
        "timestamp": datetime.utcnow(),
        "type": "response_analysis",
        "response": response[:100] + "..." if len(response) > 100 else response,
        "is_flagged": analysis.is_flagged,
        "toxicity_detected": analysis.toxicity_detected,
        "toxicity_score": analysis.toxicity_score,
        "explanation": analysis.explanation
    }
    
    mongo_db.security_logs.insert_one(log_entry)
    
    return analysis

@router.get("/security-stats")
async def get_security_stats(
    mongo_db = Depends(get_mongo)
):
    """Get security statistics from logs."""
    # Get counts from MongoDB
    total_logs = mongo_db.security_logs.count_documents({})
    flagged_logs = mongo_db.security_logs.count_documents({"is_flagged": True})
    jailbreak_logs = mongo_db.security_logs.count_documents({"jailbreak_detected": True})
    pii_logs = mongo_db.security_logs.count_documents({"pii_detected": True})
    toxicity_logs = mongo_db.security_logs.count_documents({"toxicity_detected": True})
    
    return {
        "total_analyses": total_logs,
        "flagged_analyses": flagged_logs,
        "jailbreak_attempts": jailbreak_logs,
        "pii_detections": pii_logs,
        "toxicity_detections": toxicity_logs,
        "flag_rate": (flagged_logs / total_logs * 100) if total_logs > 0 else 0
    }
