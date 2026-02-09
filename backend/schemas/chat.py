from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    is_flagged: bool = False
    explanation: Optional[str] = None
    jailbreak_detected: bool = False
    pii_detected: bool = False
    toxicity_detected: bool = False

class ConversationCreate(BaseModel):
    title: str

class ConversationResponse(BaseModel):
    id: str
    title: str
    user_id: str
    created_at: datetime
    is_flagged: Optional[bool] = None
    jailbreak_detected: Optional[bool] = None
    pii_detected: Optional[bool] = None
    toxicity_detected: Optional[bool] = None

    class Config:
        from_attributes = True

class MessageResponse(BaseModel):
    id: str
    content: str
    role: str
    created_at: datetime
    is_flagged: bool
    explanation: Optional[str] = None
    jailbreak_score: Optional[float] = None
    toxicity_score: Optional[float] = None

    class Config:
        from_attributes = True

class BenchmarkRequest(BaseModel):
    prompt: str
    iterations: int = 3

class BenchmarkResponse(BaseModel):
    raw_ms: List[float]
    fortilm_ms: List[float]
    raw_avg_ms: float
    fortilm_avg_ms: float
    raw_p50_ms: float
    raw_p95_ms: float
    fortilm_p50_ms: float
    fortilm_p95_ms: float



