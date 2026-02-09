from fastapi import APIRouter, Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
import os
import time
from datetime import datetime
import uuid

from models.user import User
from models.conversation import Conversation, Message, MessageRole
from models.security_event import SecurityEvent
from schemas.chat import (
    ChatRequest, ChatResponse, ConversationCreate, ConversationResponse,
    MessageResponse, BenchmarkRequest, BenchmarkResponse
)
from utils.database import get_db, get_mongo
from modules.auth import get_current_user
from modules.security_modules import analyze_prompt, analyze_response
from modules.privacy_preserver import privacy_preserver, api_key_detector
from modules.output_filter import output_filter, FilterSeverity

router = APIRouter()
security_optional = HTTPBearer(auto_error=False)

# In-memory storage for security events (cache only, not persistent)
_security_events: List[dict] = []

def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_optional),
    db: Optional[Session] = Depends(get_db)
) -> Optional[User]:
    """Get current user if credentials are provided, otherwise return None."""
    if not credentials:
        return None
    if not db:
        return None
    try:
        return get_current_user(credentials, db)
    except:
        return None

# Groq setup (primary LLM provider)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Update ChatResponse to include all fields
class ChatResponseExtended(ChatResponse):
    jailbreak_score: Optional[float] = None
    toxicity_score: Optional[float] = None

def record_security_event(
    event_type: str,
    severity: str,
    message_text: str,
    db: Session,
    mongo_db,
    *,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    is_flagged: bool = False,
    explanation: Optional[str] = None,
    jailbreak_score: Optional[float] = None,
    toxicity_score: Optional[float] = None
):
    """Log security events to PostgreSQL (primary/persistent), MongoDB (optional), and in-memory (cache)."""
    event_id = str(uuid.uuid4())
    event_time = datetime.utcnow()
    
    # Primary: Save to PostgreSQL (persistent across restarts)
    if db is not None:
        try:
            db_event = SecurityEvent(
                id=event_id,
                timestamp=event_time,
                type=event_type,
                severity=severity,
                message=message_text[:500],
                conversation_id=conversation_id,
                user_id=user_id,
                is_flagged=is_flagged,
                explanation=explanation,
                jailbreak_score=jailbreak_score,
                toxicity_score=toxicity_score
            )
            db.add(db_event)
            db.commit()
        except Exception as e:
            try:
                db.rollback()
            except:
                pass
            print(f"Warning: Failed to save security event to PostgreSQL: {e}")
    
    # Secondary: Try MongoDB (if available, optional)
    if mongo_db is not None:
        try:
            mongo_event = {
                "id": event_id,
                "timestamp": event_time,
                "type": event_type,
                "severity": severity,
                "message": message_text[:500],
                "conversation_id": conversation_id,
                "user_id": user_id,
                "is_flagged": is_flagged,
                "explanation": explanation,
            }
            mongo_db.security_logs.insert_one(mongo_event)
        except Exception:
            pass  # MongoDB is optional
    
    # Cache: In-memory storage (for fast recent access, not persistent)
    event_dict = {
        "id": event_id,
        "timestamp": event_time,
        "type": event_type,
        "severity": severity,
        "message": message_text[:500],
        "conversation_id": conversation_id,
        "user_id": user_id,
        "is_flagged": is_flagged,
        "explanation": explanation,
    }
    _security_events.append(event_dict)
    # Cap list size to avoid unbounded growth
    if len(_security_events) > 500:
        del _security_events[:len(_security_events) - 500]

@router.post("", response_model=ChatResponseExtended)
async def chat(
    chat_request: ChatRequest,
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Process a chat message through FortiLM security layers."""
    
    # Step 1: Analyze the prompt for security threats
    prompt_analysis = await analyze_prompt(chat_request.message)
    # Also detect toxicity in the user prompt to catch abusive inputs
    prompt_toxicity = await analyze_response(chat_request.message)

    user_id = current_user.id if current_user else "anonymous"

    # If prompt is toxic, record it but still allow it to reach LLM
    # This allows us to test how the LLM responds to toxic prompts
    # The LLM response will then be filtered by the Output Filter
    toxic_prompt_detected = False
    if prompt_toxicity.toxicity_detected:
        toxic_prompt_detected = True
        record_security_event(
            event_type="toxicity",
            severity="medium",
            message_text=chat_request.message,
            db=db,
            mongo_db=mongo_db,
            is_flagged=True,
            explanation=prompt_toxicity.explanation,
            user_id=user_id,
            toxicity_score=prompt_toxicity.toxicity_score,
        )
        # Note: We continue processing to allow LLM to respond
        # The LLM response will be filtered by Output Filter

    if prompt_analysis.is_flagged:
        record_security_event(
            event_type=(
                "jailbreak" if prompt_analysis.jailbreak_detected else (
                    "pii" if prompt_analysis.pii_detected else "toxicity"
                )
            ),
            severity="high",
            message_text=chat_request.message,
            db=db,
            mongo_db=mongo_db,
            is_flagged=True,
            explanation=prompt_analysis.explanation,
            user_id=user_id,
            jailbreak_score=prompt_analysis.jailbreak_score
        )
        return ChatResponseExtended(
            message="Your message has been flagged for security reasons. Please rephrase your request.",
            is_flagged=True,
            explanation=prompt_analysis.explanation,
            jailbreak_detected=prompt_analysis.jailbreak_detected,
            pii_detected=prompt_analysis.pii_detected,
            jailbreak_score=prompt_analysis.jailbreak_score
        )
    
    # Step 2: Get or create conversation
    conversation = None
    conversation_id = chat_request.conversation_id or str(uuid.uuid4())  # Use provided ID or generate new
    
    if db is not None:
        try:
            if chat_request.conversation_id:
                if current_user:
                    conversation = db.query(Conversation).filter(
                        Conversation.id == chat_request.conversation_id,
                        Conversation.user_id == current_user.id
                    ).first()
                else:
                    # Demo mode: allow any conversation_id if no auth
                    conversation = db.query(Conversation).filter(
                        Conversation.id == chat_request.conversation_id
                    ).first()
            
            if not conversation:
                # Create a demo user if no auth
                if not current_user:
                    # Try to get existing demo user first
                    demo_user = db.query(User).filter(User.email == "demo@fortilm.local").first()
                    if not demo_user:
                        # Create demo user without password hash to avoid bcrypt issues
                        try:
                            # Use a simple hash that won't trigger bcrypt verification during import
                            demo_user = User(
                                id=str(uuid.uuid4()),
                                email="demo@fortilm.local",
                                name="Demo User",
                                hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # Dummy hash for demo mode
                            )
                            db.add(demo_user)
                            db.commit()
                            db.refresh(demo_user)
                            current_user = demo_user
                        except Exception as e:
                            print(f"Warning: Could not create demo user: {e}")
                            try:
                                db.rollback()
                            except:
                                pass
                            # Fallback: create a mock user object
                            class MockUser:
                                def __init__(self):
                                    self.id = str(uuid.uuid4())
                            current_user = MockUser()
                    else:
                        current_user = demo_user
                
                # Create conversation with the conversation_id we're using
                conversation = Conversation(
                    id=conversation_id,
                    title=chat_request.message[:50] + "..." if len(chat_request.message) > 50 else chat_request.message,
                    user_id=current_user.id if current_user else str(uuid.uuid4())
                )
                db.add(conversation)
                db.commit()
                db.refresh(conversation)
                conversation_id = conversation.id
        except Exception as e:
            print(f"Warning: Database operation failed: {e}")
            try:
                db.rollback()
            except:
                pass
            # Continue without database - use generated conversation_id
            conversation = None
            conversation_id = str(uuid.uuid4())
    else:
        # No database available - generate conversation_id
        conversation = None
        conversation_id = str(uuid.uuid4())
    
    # Step 3: Apply Privacy Preserver (Iteration 2)
    # Detect and mask PII before sending to LLM
    # Don't mask names in conversational contexts (users often share their names with AI)
    privacy_result = privacy_preserver.preserve_privacy(chat_request.message, mask_names=False)
    
    # Also detect API keys and passwords
    api_keys = api_key_detector.detect_api_keys(chat_request.message)
    passwords = api_key_detector.detect_passwords(chat_request.message)
    
    # Mask API keys and passwords if detected
    masked_message = privacy_result.masked_text
    pii_mappings = privacy_result.pii_mappings.copy() if privacy_result.pii_mappings else {}
    
    if api_keys:
        for api_key in api_keys:
            masked_message = masked_message.replace(api_key["value"], "[API_KEY_REDACTED]")
            pii_mappings["[API_KEY_REDACTED]"] = {
                "original": api_key["value"],
                "type": "api_key",
                "strategy": "full",
                "confidence": api_key["confidence"]
            }
    
    if passwords:
        for password in passwords:
            masked_message = masked_message.replace(password["value"], "[PASSWORD_REDACTED]")
            pii_mappings["[PASSWORD_REDACTED]"] = {
                "original": password["value"],
                "type": "password",
                "strategy": "full",
                "confidence": password["confidence"]
            }
    
    # Update PII detection flag if privacy preserver found PII
    if privacy_result.pii_detected or api_keys or passwords:
        prompt_analysis.pii_detected = True
        if not prompt_analysis.explanation:
            prompt_analysis.explanation = "PII detected and masked"
        else:
            prompt_analysis.explanation += " | PII detected and masked"
        
        # Record PII detection event (even if not flagged, for tracking)
        record_security_event(
            event_type="pii",
            severity="high",
            message_text=chat_request.message,
            db=db,
            mongo_db=mongo_db,
            conversation_id=conversation_id,
            user_id=user_id,
            is_flagged=False,  # PII is masked, not blocked
            explanation=prompt_analysis.explanation
        )
    
    # Step 3: Save user message (with Privacy Preserver data)
    # Try to create message with Iteration 2 fields, fallback to basic if columns don't exist
    try:
        user_message = Message(
            id=str(uuid.uuid4()),
            content=masked_message,  # Store masked content as main content
            role=MessageRole.USER,
            conversation_id=conversation_id,
            is_flagged=prompt_analysis.is_flagged or toxic_prompt_detected,  # Include toxic prompt flag
            jailbreak_score=prompt_analysis.jailbreak_score,
            pii_detected=privacy_result.pii_detected or bool(api_keys) or bool(passwords),
            toxicity_score=prompt_toxicity.toxicity_score if toxic_prompt_detected else None,  # Store toxicity score for toxic prompts
            explanation=prompt_analysis.explanation or (prompt_toxicity.explanation if toxic_prompt_detected else None),
            # Privacy Preserver fields (Iteration 2) - may not exist in DB yet
            original_content=chat_request.message,  # Store original with PII
            masked_content=masked_message,  # Masked version
            pii_mappings=pii_mappings if pii_mappings else None  # Store mappings
        )
    except Exception as e:
        # Fallback: create message without Iteration 2 fields if columns don't exist
        print(f"Warning: Iteration 2 fields not available, using fallback: {e}")
        user_message = Message(
            id=str(uuid.uuid4()),
            content=masked_message,
            role=MessageRole.USER,
            conversation_id=conversation_id,
            is_flagged=prompt_analysis.is_flagged or toxic_prompt_detected,  # Include toxic prompt flag
            jailbreak_score=prompt_analysis.jailbreak_score,
            pii_detected=privacy_result.pii_detected or bool(api_keys) or bool(passwords),
            toxicity_score=prompt_toxicity.toxicity_score if toxic_prompt_detected else None,  # Store toxicity score
            explanation=prompt_analysis.explanation or (prompt_toxicity.explanation if toxic_prompt_detected else None)
        )
    
    # Save user message to database if available and conversation exists
    if db is not None and conversation is not None:
        try:
            # Ensure conversation_id matches the conversation
            user_message.conversation_id = conversation.id
            db.add(user_message)
            db.commit()
        except Exception as e:
            print(f"Warning: Failed to save user message to database: {e}")
            try:
                db.rollback()
            except:
                pass
    
    # Step 4: Get AI response
    messages = []
    if db is not None:
        try:
            # Get conversation history
            messages = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at.asc()).all()
        except Exception as e:
            print(f"Warning: Failed to get conversation history: {e}")
            messages = []
    
    try:
        
        # Prepare messages for LLM (use masked content for privacy)
        model_messages = []
        for msg in messages:
            # Use masked_content if available (for user messages), otherwise use content
            try:
                content_to_use = msg.masked_content if hasattr(msg, 'masked_content') and msg.masked_content else msg.content
            except:
                content_to_use = msg.content
            model_messages.append({
                "role": msg.role.value.lower() if hasattr(msg.role, 'value') else msg.role.lower(),
                "content": content_to_use
            })
        
        # Add current user message (use masked version)
        model_messages.append({
            "role": "user",
            "content": masked_message  # Use masked version for LLM
        })
        
        # Call Groq API (Llama) - primary provider
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
                # Call Groq API directly
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
    
    # Step 5: Apply Output Filter (Iteration 2)
    # Use enhanced output filter with custom models
    filter_result = output_filter.filter_response(ai_response)
    
    # Also run legacy analyze_response for backward compatibility
    response_analysis = await analyze_response(ai_response)
    
    # Combine results (Output Filter takes precedence)
    is_flagged = filter_result.is_flagged or response_analysis.is_flagged
    
    # Step 6: Save AI message (with Output Filter data)
    # Store original response BEFORE any filtering (for audit trail and dashboard)
    original_response_for_db = ai_response
    
    # Determine what to show to user (filtered/censored/blocked version)
    if filter_result.filtered_response is None and filter_result.is_flagged:
        # Response was blocked - use placeholder for user
        final_response = "[RESPONSE BLOCKED - Toxic content detected]"
    elif filter_result.filtered_response:
        # Response was censored/filtered - use the censored version
        final_response = filter_result.filtered_response
    else:
        # No filtering applied - use original response
        final_response = ai_response
    
    # Store filtered_content for dashboard visibility
    # IMPORTANT: ALWAYS store ORIGINAL toxic content so it can be displayed on dashboard
    # The original is always available from filter_result.original_response
    # The censored version is stored in content (what user saw)
    filtered_content_to_store = None
    if filter_result.is_flagged or filter_result.toxicity_detected or filter_result.bias_detected or filter_result.jailbreak_detected:
        # Always use the original_response from filter_result (which is the actual original)
        # This ensures we always have the original, even if entire response was censored
        filtered_content_to_store = filter_result.original_response if hasattr(filter_result, 'original_response') and filter_result.original_response else original_response_for_db
    
    # Store toxicity_score ONLY if toxicity was actually detected (toxicity_detected == True)
    # This ensures we only store scores for messages that are actually toxic
    # Don't store scores for non-toxic messages (even if model ran)
    toxicity_score_to_store = None
    if filter_result.toxicity_detected and filter_result.toxicity_score is not None:
        # Only store if actually detected (model threshold is 0.5, so this should be > 0.5)
        toxicity_score_to_store = filter_result.toxicity_score
    elif response_analysis.toxicity_detected and response_analysis.toxicity_score is not None:
        # Legacy analysis detected toxicity
        toxicity_score_to_store = response_analysis.toxicity_score
    
    # Try to create message with Iteration 2 fields, fallback to basic if columns don't exist
    try:
        ai_message = Message(
            id=str(uuid.uuid4()),
            content=final_response,  # Store filtered response (or original if not filtered)
            role=MessageRole.ASSISTANT,
            conversation_id=conversation_id,
            is_flagged=is_flagged,
            toxicity_score=toxicity_score_to_store,  # Only store if > 0
            explanation=filter_result.explanation or response_analysis.explanation,
            # Output Filter fields (Iteration 2) - may not exist in DB yet
            # Only store filtered_content if response was actually filtered or had detections
            filtered_content=filtered_content_to_store,
            bias_detected=filter_result.bias_detected,
            bias_score=filter_result.bias_score,
            jailbreak_detected_in_output=filter_result.jailbreak_detected,
            jailbreak_score_in_output=filter_result.jailbreak_score,
            filter_analysis={
                "toxicity_detected": filter_result.toxicity_detected,
                "bias_detected": filter_result.bias_detected,
                "jailbreak_detected": filter_result.jailbreak_detected,
                "toxicity_score": filter_result.toxicity_score if filter_result.toxicity_detected else None,  # Only store score if detected
                "bias_score": filter_result.bias_score if filter_result.bias_detected else None,
                "jailbreak_score": filter_result.jailbreak_score if filter_result.jailbreak_detected else None,
                "severity": filter_result.severity.value if filter_result.severity else None
            } if (filter_result.is_flagged or filter_result.toxicity_detected or filter_result.bias_detected or filter_result.jailbreak_detected) else None,  # Store analysis only if something was detected
            sanitization_strategy=filter_result.sanitization_strategy.value if filter_result.sanitization_strategy else None
        )
    except (TypeError, AttributeError) as e:
        # Fallback: create message without Iteration 2 fields if columns don't exist
        print(f"Warning: Iteration 2 fields not available, using fallback: {e}")
        ai_message = Message(
            id=str(uuid.uuid4()),
            content=final_response,
            role=MessageRole.ASSISTANT,
            conversation_id=conversation_id,
            is_flagged=is_flagged,
            toxicity_score=filter_result.toxicity_score or response_analysis.toxicity_score,
            explanation=filter_result.explanation or response_analysis.explanation
        )
    
    # Save AI message and update conversation if database is available
    if db is not None and conversation is not None:
        try:
            db.add(ai_message)
            
            # Update conversation flags
            conversation.is_flagged = prompt_analysis.is_flagged or is_flagged
            conversation.jailbreak_detected = prompt_analysis.jailbreak_detected
            conversation.pii_detected = privacy_result.pii_detected or bool(api_keys) or bool(passwords)
            conversation.toxicity_detected = filter_result.toxicity_detected or response_analysis.is_flagged
            conversation.bias_detected = filter_result.bias_detected
            conversation.jailbreak_detected_in_output = filter_result.jailbreak_detected
            
            db.commit()
        except Exception as e:
            print(f"Warning: Failed to save AI message or update conversation: {e}")
            try:
                db.rollback()
            except:
                pass
    
    # Record security events (Iteration 2 - enhanced)
    # Track if we've already recorded an event (e.g., PII was already recorded above)
    event_already_recorded = privacy_result.pii_detected or bool(api_keys) or bool(passwords)
    
    if filter_result.is_flagged:
        event_type = "jailbreak_output" if filter_result.jailbreak_detected else (
            "bias" if filter_result.bias_detected else "toxicity"
        )
        severity = "critical" if filter_result.severity == FilterSeverity.CRITICAL else (
            "high" if filter_result.severity == FilterSeverity.HIGH else "medium"
        )
        record_security_event(
            event_type=event_type,
            severity=severity,
            message_text=ai_response[:500],  # Original response
            db=db,
            mongo_db=mongo_db,
            conversation_id=conversation_id,
            user_id=user_id,
            is_flagged=True,
            explanation=filter_result.explanation,
            toxicity_score=filter_result.toxicity_score
        )
        event_already_recorded = True
    elif response_analysis.is_flagged:
        record_security_event(
            event_type="toxicity",
            severity="medium",
            message_text=chat_request.message,
            db=db,
            mongo_db=mongo_db,
            conversation_id=conversation_id,
            user_id=user_id,
            is_flagged=True,
            explanation=response_analysis.explanation,
            toxicity_score=response_analysis.toxicity_score
        )
        event_already_recorded = True
    elif not event_already_recorded:
        # Only record "normal" event if no other event was recorded (e.g., no PII detected)
        record_security_event(
            event_type="normal",
            severity="low",
            message_text=chat_request.message,
            db=db,
            mongo_db=mongo_db,
            conversation_id=conversation_id,
            user_id=user_id,
            is_flagged=False,
        )
    
    # Return filtered response (Iteration 2)
    # IMPORTANT: Always send the censored/filtered version to the chat UI
    # The original is stored in filtered_content for dashboard display only
    # Never send the original toxic content to the user in the chat UI
    user_facing_response = final_response if final_response else ai_response
    
    # Safety check: If response was flagged/filtered, ensure we're not sending original
    if filter_result.is_flagged and filter_result.filtered_response:
        # Use the filtered/censored version (already in final_response)
        user_facing_response = final_response
    elif filter_result.is_flagged and not filter_result.filtered_response:
        # Was blocked, use placeholder
        user_facing_response = "[RESPONSE BLOCKED - Toxic content detected]"
    
    return ChatResponseExtended(
        message=user_facing_response,  # Always send censored/filtered version to chat UI
        conversation_id=conversation_id,
        is_flagged=is_flagged,
        explanation=filter_result.explanation or response_analysis.explanation,
        jailbreak_detected=prompt_analysis.jailbreak_detected,
        pii_detected=privacy_result.pii_detected or bool(api_keys) or bool(passwords),
        toxicity_detected=filter_result.toxicity_detected or response_analysis.is_flagged,
        jailbreak_score=prompt_analysis.jailbreak_score,
        toxicity_score=filter_result.toxicity_score or response_analysis.toxicity_score
    )


@router.get("/recent-activity")
async def recent_activity(limit: int = 50, db: Session = Depends(get_db), mongo_db = Depends(get_mongo)):
    """Return recent security events for admin dashboard."""
    events = []
    
    # Primary: Query PostgreSQL (persistent data)
    try:
        if db is not None:
            db_events = db.query(SecurityEvent).order_by(SecurityEvent.timestamp.desc()).limit(limit).all()
            events = [
                {
                    "id": e.id,
                    "timestamp": e.timestamp.isoformat() if e.timestamp.tzinfo else e.timestamp.isoformat() + "Z",
                    "type": e.type,
                    "severity": e.severity,
                    "message": e.message,
                    "conversation_id": e.conversation_id,
                    "user_id": e.user_id,
                    "is_flagged": e.is_flagged,
                    "explanation": e.explanation,
                }
                for e in db_events
            ]
    except Exception as e:
        print(f"Warning: Failed to fetch security events from PostgreSQL: {e}")
        # Fall through to MongoDB/in-memory
    
    # Fallback: Try MongoDB if PostgreSQL fails and no results yet
    if not events and mongo_db is not None:
        try:
            mongo_events = list(mongo_db.security_logs.find().sort("timestamp", -1).limit(limit))
            for event in mongo_events:
                if "_id" in event:
                    event["id"] = str(event.pop("_id"))
                if "timestamp" in event and isinstance(event["timestamp"], datetime):
                    ts = event["timestamp"]
                    event["timestamp"] = ts.isoformat() if ts.tzinfo else ts.isoformat() + "Z"
                event.pop("_id", None)
            events = mongo_events
        except Exception:
            pass
    
    # Last resort: In-memory cache (only recent events, not persistent)
    if not events:
        events = list(reversed(_security_events[-limit:]))
        for event in events:
            if "timestamp" in event and isinstance(event["timestamp"], datetime):
                ts = event["timestamp"]
                event["timestamp"] = ts.isoformat() if ts.tzinfo else ts.isoformat() + "Z"
    
    return events


@router.get("/metrics")
async def metrics(db: Session = Depends(get_db), mongo_db = Depends(get_mongo)):
    """Lightweight operational metrics for the admin dashboard."""
    try:
        from datetime import timedelta
        now = datetime.utcnow()
        # Consider conversations active if they were created/updated in the past 5 minutes
        recent_cutoff = now - timedelta(minutes=5)
        
        # Get active conversations from PostgreSQL
        recent_conversations = db.query(Conversation).filter(
            Conversation.created_at >= recent_cutoff
        ).all()
        active_users = len(set(c.user_id for c in recent_conversations))
        
        # Get recent errors from MongoDB (if available)
        recent_errors = 0
        if mongo_db is not None:
            try:
                recent_errors = mongo_db.security_logs.count_documents({
                    "timestamp": {"$gte": recent_cutoff},
                    "type": "error"
                })
            except:
                pass
        
        system_status = "Operational" if groq_client else "Degraded"
        
        return {
            "active_users": active_users,
            "system_status": system_status,
            "recent_errors": recent_errors,
        }
    except Exception as e:
        print(f"前缀: Failed to fetch metrics: {e}")
        return {
            "active_users": 0,
            "system_status": "Degraded" if not groq_client else "Operational",
            "recent_errors": 0,
        }


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
async def create_conversation(
    conversation_data: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Create a new conversation."""
    if not current_user:
        # Demo mode: create demo user
        demo_user = db.query(User).filter(User.email == "demo@fortilm.local").first()
        if not demo_user:
            # Use dummy hash to avoid bcrypt initialization issues
            demo_user = User(
                id=str(uuid.uuid4()),
                email="demo@fortilm.local",
                name="Demo User",
                hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
            )
            db.add(demo_user)
            db.commit()
            db.refresh(demo_user)
        current_user = demo_user
    
    conversation = Conversation(
        id=str(uuid.uuid4()),
        title=conversation_data.title,
        user_id=current_user.id
    )
    
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        user_id=conversation.user_id,
        created_at=conversation.created_at,
        is_flagged=conversation.is_flagged,
        jailbreak_detected=conversation.jailbreak_detected,
        pii_detected=conversation.pii_detected,
        toxicity_detected=conversation.toxicity_detected
    )


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
async def create_conversation(
    conversation_data: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Create a new conversation."""
    if not current_user:
        # Demo mode: create demo user
        demo_user = db.query(User).filter(User.email == "demo@fortilm.local").first()
        if not demo_user:
            # Use dummy hash to avoid bcrypt initialization issues
            demo_user = User(
                id=str(uuid.uuid4()),
                email="demo@fortilm.local",
                name="Demo User",
                hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
            )
            db.add(demo_user)
            db.commit()
            db.refresh(demo_user)
        current_user = demo_user
    
    conversation = Conversation(
        id=str(uuid.uuid4()),
        title=conversation_data.title,
        user_id=current_user.id
    )
    
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        user_id=conversation.user_id,
        created_at=conversation.created_at,
        is_flagged=conversation.is_flagged,
        jailbreak_detected=conversation.jailbreak_detected,
        pii_detected=conversation.pii_detected,
        toxicity_detected=conversation.toxicity_detected
    )