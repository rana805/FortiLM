from sqlalchemy import Column, String, DateTime, Boolean, Float, ForeignKey, Text, Enum, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from utils.database import Base
import enum

class MessageRole(str, enum.Enum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Security flags (Iteration 1)
    is_flagged = Column(Boolean, default=False)
    jailbreak_detected = Column(Boolean, default=False)
    pii_detected = Column(Boolean, default=False)
    toxicity_detected = Column(Boolean, default=False)
    
    # Output Filter flags (Iteration 2)
    bias_detected = Column(Boolean, default=False)
    jailbreak_detected_in_output = Column(Boolean, default=False)

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    role = Column(Enum(MessageRole), nullable=False)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Security analysis results (Iteration 1)
    is_flagged = Column(Boolean, default=False)
    jailbreak_score = Column(Float, nullable=True)
    toxicity_score = Column(Float, nullable=True)
    pii_detected = Column(Boolean, default=False)
    explanation = Column(Text, nullable=True)

    # Privacy Preserver fields (Iteration 2)
    original_content = Column(Text, nullable=True)  # Original content with PII
    masked_content = Column(Text, nullable=True)  # Masked content sent to LLM
    pii_mappings = Column(JSON, nullable=True)  # PII placeholder mappings

    # Output Filter fields (Iteration 2)
    filtered_content = Column(Text, nullable=True)  # Filtered AI response
    bias_detected = Column(Boolean, default=False)
    bias_score = Column(Float, nullable=True)
    jailbreak_detected_in_output = Column(Boolean, default=False)
    jailbreak_score_in_output = Column(Float, nullable=True)
    filter_analysis = Column(JSON, nullable=True)  # Complete filter analysis results
    sanitization_strategy = Column(String, nullable=True)  # block, censor, warn, filter

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")




    masked_content = Column(Text, nullable=True)  # Masked content sent to LLM
    pii_mappings = Column(JSON, nullable=True)  # PII placeholder mappings

    # Output Filter fields (Iteration 2)
    filtered_content = Column(Text, nullable=True)  # Filtered AI response
    bias_detected = Column(Boolean, default=False)
    bias_score = Column(Float, nullable=True)
    jailbreak_detected_in_output = Column(Boolean, default=False)
    jailbreak_score_in_output = Column(Float, nullable=True)
    filter_analysis = Column(JSON, nullable=True)  # Complete filter analysis results
    sanitization_strategy = Column(String, nullable=True)  # block, censor, warn, filter

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")



