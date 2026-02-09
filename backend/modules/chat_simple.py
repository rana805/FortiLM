from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
import time
from typing import List, Optional
from groq import Groq
import os
from datetime import datetime
import uuid

from modules.security_modules import analyze_prompt, analyze_response

router = APIRouter()

# Groq setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

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
    jailbreak_score: Optional[float] = None
    toxicity_score: Optional[float] = None

class ConversationCreate(BaseModel):
    title: str

class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    is_flagged: Optional[bool] = None
    jailbreak_detected: Optional[bool] = None
    pii_detected: Optional[bool] = None
    toxicity_detected: Optional[bool] = None


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

class MessageResponse(BaseModel):
    id: str
    content: str
    role: str
    created_at: datetime
    is_flagged: bool
    explanation: Optional[str] = None
    jailbreak_score: Optional[float] = None
    toxicity_score: Optional[float] = None

# In-memory storage for demo purposes
conversations = {}
messages = {}
security_events: List[dict] = []

def record_security_event(event_type: str, severity: str, message_text: str, *, conversation_id: Optional[str] = None, is_flagged: bool = False, explanation: Optional[str] = None):
    event = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow(),
        "type": event_type,
        "severity": severity,
        "message": message_text,
        "conversation_id": conversation_id,
        "is_flagged": is_flagged,
        "explanation": explanation,
    }
    security_events.append(event)
    # Cap list size to avoid unbounded growth
    if len(security_events) > 500:
        del security_events[: len(security_events) - 500]


@router.get("/metrics")
async def metrics():
    """Lightweight operational metrics for the admin dashboard."""
    now = datetime.utcnow()
    # Consider conversations active if they emitted an event in the past 5 minutes
    recent_cutoff = now.timestamp() - 5 * 60
    recent_conversation_ids = set()
    recent_errors = 0
    for e in security_events[-200:]:  # scan recent window
        try:
            ts = e["timestamp"].timestamp()
        except Exception:
            continue
        if ts >= recent_cutoff:
            if e.get("conversation_id"):
                recent_conversation_ids.add(e["conversation_id"])
            if e.get("type") in {"error"}:
                recent_errors += 1

    active_users = len(recent_conversation_ids)
    system_status = "Operational" if groq_client else "Degraded"

    return {
        "active_users": active_users,
        "system_status": system_status,
        "recent_errors": recent_errors,
    }

@router.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """Process a chat message through FortiLM security layers."""
    
    # Step 1: Analyze the prompt for security threats
    prompt_analysis = await analyze_prompt(chat_request.message)
    
    # Debug logging
    print(f"DEBUG: Analyzing prompt: '{chat_request.message}'")
    print(f"DEBUG: Analysis result: {prompt_analysis}")
    
    if prompt_analysis.is_flagged:
        record_security_event(
            event_type=(
                "jailbreak" if prompt_analysis.jailbreak_detected else (
                    "pii" if prompt_analysis.pii_detected else "toxicity"
                )
            ),
            severity="high",
            message_text=chat_request.message,
            is_flagged=True,
            explanation=prompt_analysis.explanation,
        )
        return ChatResponse(
            message="Your message has been flagged for security reasons. Please rephrase your request.",
            is_flagged=True,
            explanation=prompt_analysis.explanation,
            jailbreak_detected=prompt_analysis.jailbreak_detected,
            pii_detected=prompt_analysis.pii_detected,
            jailbreak_score=prompt_analysis.jailbreak_score
        )
    
    # Step 2: Get or create conversation
    conversation_id = chat_request.conversation_id
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        conversations[conversation_id] = {
            "id": conversation_id,
            "title": chat_request.message[:50] + "..." if len(chat_request.message) > 50 else chat_request.message,
            "created_at": datetime.utcnow(),
            "is_flagged": False,
            "jailbreak_detected": False,
            "pii_detected": False,
            "toxicity_detected": False
        }
    
    # Step 3: Save user message
    user_message_id = str(uuid.uuid4())
    user_message = {
        "id": user_message_id,
        "content": chat_request.message,
        "role": "USER",
        "conversation_id": conversation_id,
        "created_at": datetime.utcnow(),
        "is_flagged": prompt_analysis.is_flagged,
        "jailbreak_score": prompt_analysis.jailbreak_score,
        "pii_detected": prompt_analysis.pii_detected,
        "explanation": prompt_analysis.explanation
    }
    messages[user_message_id] = user_message
    
    # Step 4: Get AI response
    try:
        # Get conversation history
        conversation_messages = [
            msg for msg in messages.values() 
            if msg["conversation_id"] == conversation_id
        ]
        conversation_messages.sort(key=lambda x: x["created_at"])
        
        # Prepare messages for LLM
        model_messages = []
        for msg in conversation_messages:
            model_messages.append({
                "role": msg["role"].lower(),
                "content": msg["content"]
            })
        
        # Add current user message
        model_messages.append({
            "role": "user",
            "content": chat_request.message
        })
        
        # Call Groq API (Llama)
        if not groq_client:
            # No API key configured – provide contextual fallback
            user_msg_lower = chat_request.message.lower()
            if any(word in user_msg_lower for word in ["hello", "hi", "hey", "greetings"]):
                ai_response = "Hello! I'm FortiLM's secure AI assistant. I'm here to help you with various tasks while ensuring our conversation remains safe and secure. How can I assist you today?"
            elif any(word in user_msg_lower for word in ["help", "assist", "support"]):
                ai_response = "I'd be happy to help! I'm FortiLM's AI assistant, designed to provide secure and helpful responses. What specific assistance do you need?"
            elif any(word in user_msg_lower for word in ["what", "who", "how", "why", "when", "where"]):
                ai_response = f"That's an interesting question about '{chat_request.message}'. I'm FortiLM's secure AI assistant, and I'm designed to provide helpful information while maintaining security standards. Could you provide more context about what you'd like to know?"
            else:
                ai_response = f"I understand you're asking about '{chat_request.message}'. I'm FortiLM's secure AI assistant, and I'm here to help while ensuring our conversation remains safe and productive. How can I assist you further?"
        else:
            try:
                response = groq_client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                    messages=model_messages,
                    max_tokens=800,
                    temperature=0.7
                )
                ai_response = response.choices[0].message.content
            except Exception as groq_error:
                # Context-aware fallback on any Groq error
                user_msg_lower = chat_request.message.lower()
                if any(word in user_msg_lower for word in ["hello", "hi", "hey", "greetings"]):
                    ai_response = "Hello! I'm FortiLM's secure AI assistant. I'm here to help you with various tasks while ensuring our conversation remains safe and secure. How can I assist you today?"
                elif any(word in user_msg_lower for word in ["help", "assist", "support"]):
                    ai_response = "I'd be happy to help! I'm FortiLM's AI assistant, designed to provide secure and helpful responses. What specific assistance do you need?"
                elif any(word in user_msg_lower for word in ["what", "who", "how", "why", "when", "where"]):
                    ai_response = f"That's an interesting question about '{chat_request.message}'. I'm FortiLM's secure AI assistant, and I'm designed to provide helpful information while maintaining security standards. Could you provide more context about what you'd like to know?"
                else:
                    ai_response = f"I understand you're asking about '{chat_request.message}'. I'm FortiLM's secure AI assistant, and I'm here to help while ensuring our conversation remains safe and productive. How can I assist you further?"
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating AI response: {str(e)}"
        )
    
    # Step 5: Analyze AI response for safety
    response_analysis = await analyze_response(ai_response)
    
    # Step 6: Save AI message
    ai_message_id = str(uuid.uuid4())
    ai_message = {
        "id": ai_message_id,
        "content": ai_response,
        "role": "ASSISTANT",
        "conversation_id": conversation_id,
        "created_at": datetime.utcnow(),
        "is_flagged": response_analysis.is_flagged,
        "toxicity_score": response_analysis.toxicity_score,
        "explanation": response_analysis.explanation
    }
    messages[ai_message_id] = ai_message
    
    # Update conversation flags
    conversations[conversation_id]["is_flagged"] = prompt_analysis.is_flagged or response_analysis.is_flagged
    conversations[conversation_id]["jailbreak_detected"] = prompt_analysis.jailbreak_detected
    conversations[conversation_id]["pii_detected"] = prompt_analysis.pii_detected
    conversations[conversation_id]["toxicity_detected"] = response_analysis.is_flagged

    # Record response event if flagged or noteworthy
    if response_analysis.is_flagged:
        record_security_event(
            event_type="toxicity",
            severity="medium",
            message_text=chat_request.message,
            conversation_id=conversation_id,
            is_flagged=True,
            explanation=response_analysis.explanation,
        )
    else:
        record_security_event(
            event_type="normal",
            severity="low",
            message_text=chat_request.message,
            conversation_id=conversation_id,
            is_flagged=False,
        )
    
    return ChatResponse(
        message=ai_response,
        conversation_id=conversation_id,
        is_flagged=response_analysis.is_flagged,
        explanation=response_analysis.explanation,
        toxicity_detected=response_analysis.is_flagged,
        toxicity_score=response_analysis.toxicity_score
    )


@router.get("/recent-activity")
async def recent_activity(limit: int = 50):
    """Return recent security events for admin dashboard (no auth in demo)."""
    data = list(reversed(security_events[-limit:]))
    # Serialize timestamps
    return [
        {
            **{k: v for k, v in e.items() if k != "timestamp"},
            "timestamp": e["timestamp"].isoformat() + "Z",
        }
        for e in data
    ]


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


@router.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark(req: BenchmarkRequest):
    """Compare raw Groq latency vs FortiLM pipeline latency."""
    if not groq_client:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY not configured")

    raw_times: List[float] = []
    fortilm_times: List[float] = []

    # Run RAW (direct Groq) iterations
    for _ in range(max(1, req.iterations)):
        start = time.perf_counter()
        try:
            groq_client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                messages=[{"role": "user", "content": req.prompt}],
                max_tokens=200,
                temperature=0.7,
            )
        except Exception:
            pass
        finally:
            raw_times.append((time.perf_counter() - start) * 1000.0)

    # Run FORTILM (analysis + Groq) iterations
    for _ in range(max(1, req.iterations)):
        start = time.perf_counter()
        try:
            # Prompt analysis + model call + response analysis
            pa = await analyze_prompt(req.prompt)
            if not pa.is_flagged:
                groq_client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                    messages=[{"role": "user", "content": req.prompt}],
                    max_tokens=200,
                    temperature=0.7,
                )
            await analyze_response("dummy")
        except Exception:
            pass
        finally:
            fortilm_times.append((time.perf_counter() - start) * 1000.0)

    return BenchmarkResponse(
        raw_ms=raw_times,
        fortilm_ms=fortilm_times,
        raw_avg_ms=sum(raw_times) / len(raw_times) if raw_times else 0.0,
        fortilm_avg_ms=sum(fortilm_times) / len(fortilm_times) if fortilm_times else 0.0,
        raw_p50_ms=_percentile(raw_times, 50.0),
        raw_p95_ms=_percentile(raw_times, 95.0),
        fortilm_p50_ms=_percentile(fortilm_times, 50.0),
        fortilm_p95_ms=_percentile(fortilm_times, 95.0),
    )

@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(conversation_data: ConversationCreate):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = {
        "id": conversation_id,
        "title": conversation_data.title,
        "created_at": datetime.utcnow(),
        "is_flagged": False,
        "jailbreak_detected": False,
        "pii_detected": False,
        "toxicity_detected": False
    }
    
    conversations[conversation_id] = conversation
    
    return ConversationResponse(**conversation)

@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations():
    """Get all conversations."""
    return [ConversationResponse(**conv) for conv in conversations.values()]

@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(conversation_id: str):
    """Get all messages in a conversation."""
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    conversation_messages = [
        MessageResponse(**msg) for msg in messages.values() 
        if msg["conversation_id"] == conversation_id
    ]
    conversation_messages.sort(key=lambda x: x.created_at)
    
    return conversation_messages

@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation."""
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return ConversationResponse(**conversations[conversation_id])

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages."""
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Delete all messages in the conversation
    message_ids_to_delete = [
        msg_id for msg_id, msg in messages.items() 
        if msg["conversation_id"] == conversation_id
    ]
    
    for msg_id in message_ids_to_delete:
        del messages[msg_id]
    
    # Delete the conversation
    del conversations[conversation_id]
    
    return {"message": "Conversation deleted successfully"}

@router.get("/conversations/{conversation_id}/stats")
async def get_conversation_stats(conversation_id: str):
    """Get statistics for a conversation."""
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    conversation_messages = [
        msg for msg in messages.values() 
        if msg["conversation_id"] == conversation_id
    ]
    
    total_messages = len(conversation_messages)
    flagged_messages = len([msg for msg in conversation_messages if msg["is_flagged"]])
    jailbreak_attempts = len([msg for msg in conversation_messages if msg.get("jailbreak_score", 0) > 0])
    pii_detections = len([msg for msg in conversation_messages if msg.get("pii_detected", False)])
    toxicity_detections = len([msg for msg in conversation_messages if msg.get("toxicity_score", 0) > 0])
    
    return {
        "conversation_id": conversation_id,
        "total_messages": total_messages,
        "flagged_messages": flagged_messages,
        "jailbreak_attempts": jailbreak_attempts,
        "pii_detections": pii_detections,
        "toxicity_detections": toxicity_detections,
        "flag_rate": (flagged_messages / total_messages * 100) if total_messages > 0 else 0
    }
