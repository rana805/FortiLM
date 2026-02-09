from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    role: str
    created_at: datetime

    class Config:
        from_attributes = True

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

class SecurityStatsResponse(BaseModel):
    total_users: int
    total_conversations: int
    flagged_conversations: int
    jailbreak_attempts: int
    pii_detections: int
    toxicity_detections: int
    total_security_logs: int
    flagged_security_logs: int
    recent_conversations: int
    recent_flagged: int
    flag_rate: float





