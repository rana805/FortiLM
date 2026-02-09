from sqlalchemy import Column, String, DateTime, Boolean, Text, Float
from sqlalchemy.sql import func
from utils.database import Base

class SecurityEvent(Base):
    __tablename__ = "security_events"

    id = Column(String, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    type = Column(String, nullable=False, index=True)  # jailbreak, pii, toxicity, normal
    severity = Column(String, nullable=False)  # low, medium, high
    message = Column(Text, nullable=False)
    conversation_id = Column(String, nullable=True, index=True)
    user_id = Column(String, nullable=True, index=True)
    is_flagged = Column(Boolean, default=False)
    explanation = Column(Text, nullable=True)
    
    # Optional: store scores for analysis
    jailbreak_score = Column(Float, nullable=True)
    toxicity_score = Column(Float, nullable=True)



