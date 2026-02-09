from fastapi import APIRouter, Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta, timezone

from models.user import User
from models.conversation import Conversation, Message
from schemas.admin import UserResponse, ConversationResponse, SecurityStatsResponse
from utils.database import get_db, get_mongo
from modules.auth import get_current_user

router = APIRouter()

def require_admin(current_user: User = Depends(get_current_user)):
    """Require admin role for access."""
    if current_user.role != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

def require_admin_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db)
):
    """Require admin role for access, but allow if no auth is set up (for development)."""
    # For development: allow access if no credentials provided
    if not credentials:
        return None  # Allow access without auth in development
    
    try:
        from modules.auth import get_current_user
        current_user = get_current_user(credentials, db)
        if current_user.role != "ADMIN":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        return current_user
    except:
        # If auth fails, allow access for development
        return None

@router.get("/users", response_model=List[UserResponse])
async def get_all_users(
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """Get all users (admin only)."""
    users = db.query(User).offset(skip).limit(limit).all()
    
    return [
        UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            role=user.role,
            created_at=user.created_at
        ) for user in users
    ]

@router.get("/conversations", response_model=List[ConversationResponse])
async def get_all_conversations(
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    flagged_only: bool = False
):
    """Get all conversations (admin only)."""
    query = db.query(Conversation)
    
    if flagged_only:
        query = query.filter(Conversation.is_flagged == True)
    
    conversations = query.offset(skip).limit(limit).order_by(Conversation.created_at.desc()).all()
    
    return [
        ConversationResponse(
            id=conv.id,
            title=conv.title,
            user_id=conv.user_id,
            created_at=conv.created_at,
            is_flagged=conv.is_flagged,
            jailbreak_detected=conv.jailbreak_detected,
            pii_detected=conv.pii_detected,
            toxicity_detected=conv.toxicity_detected
        ) for conv in conversations
    ]

@router.get("/security-stats", response_model=SecurityStatsResponse)
async def get_security_statistics(
    admin_user: User = Depends(require_admin_optional),
    db: Session = Depends(get_db),
    mongo_db = Depends(get_mongo)
):
    """Get comprehensive security statistics."""
    
    # PostgreSQL stats
    total_users = db.query(User).count() if db else 0
    total_conversations = db.query(Conversation).count() if db else 0
    flagged_conversations = db.query(Conversation).filter(Conversation.is_flagged == True).count() if db else 0
    jailbreak_conversations = db.query(Conversation).filter(Conversation.jailbreak_detected == True).count() if db else 0
    pii_conversations = db.query(Conversation).filter(Conversation.pii_detected == True).count() if db else 0
    toxicity_conversations = db.query(Conversation).filter(Conversation.toxicity_detected == True).count() if db else 0
    
    # Message-level stats (more accurate than conversation-level)
    total_messages = db.query(Message).count() if db else 0
    pii_messages = db.query(Message).filter(Message.pii_detected == True).count() if db else 0
    jailbreak_messages = db.query(Message).filter(Message.jailbreak_detected == True).count() if db else 0
    toxicity_messages = db.query(Message).filter(Message.toxicity_detected == True).count() if db else 0
    
    # MongoDB stats
    total_logs = mongo_db.security_logs.count_documents({}) if mongo_db else 0
    flagged_logs = mongo_db.security_logs.count_documents({"is_flagged": True}) if mongo_db else 0
    
    # Recent activity (last 24 hours)
    now_utc = datetime.now(timezone.utc)
    yesterday = now_utc - timedelta(days=1)
    recent_conversations = db.query(Conversation).filter(
        Conversation.created_at >= yesterday
    ).count() if db else 0
    
    recent_flagged = db.query(Conversation).filter(
        Conversation.created_at >= yesterday,
        Conversation.is_flagged == True
    ).count() if db else 0
    
    return SecurityStatsResponse(
        total_users=total_users,
        total_conversations=total_conversations,
        flagged_conversations=flagged_conversations,
        jailbreak_attempts=jailbreak_conversations,
        pii_detections=pii_conversations,
        toxicity_detections=toxicity_conversations,
        total_security_logs=total_logs,
        flagged_security_logs=flagged_logs,
        recent_conversations=recent_conversations,
        recent_flagged=recent_flagged,
        flag_rate=(flagged_conversations / total_conversations * 100) if total_conversations > 0 else 0
    )

@router.get("/dashboard-stats")
async def get_dashboard_stats(
    admin_user = Depends(require_admin_optional),
    db: Session = Depends(get_db)
):
    """Get dashboard statistics including total messages and recent PII detections."""
    from datetime import datetime, timedelta
    
    try:
        # Total messages
        total_messages = db.query(Message).count() if db else 0
        
        # Total PII detections
        total_pii = db.query(Message).filter(Message.pii_detected == True).count() if db else 0
        
        # Recent PII detections (last 24 hours)
        # Use timezone-aware datetime for comparison with timezone-aware created_at
        now_utc = datetime.now(timezone.utc)
        yesterday = now_utc - timedelta(days=1)
        recent_pii = db.query(Message).filter(
            Message.pii_detected == True,
            Message.created_at >= yesterday
        ).count() if db else 0
        
        # Jailbreak attempts (check jailbreak_score > 0 or jailbreak_detected_in_output)
        jailbreak_attempts = db.query(Message).filter(
            (Message.jailbreak_score.isnot(None)) | (Message.jailbreak_detected_in_output == True)
        ).count() if db else 0
        
        # Toxicity detected (check toxicity_score > 0)
        toxicity_detected = db.query(Message).filter(
            Message.toxicity_score.isnot(None),
            Message.toxicity_score > 0
        ).count() if db else 0
        
        return {
            "total_messages": total_messages,
            "recent_pii_detections": recent_pii,
            "total_pii_detections": total_pii,
            "jailbreak_attempts": jailbreak_attempts,
            "toxicity_detected": toxicity_detected
        }
    except Exception as e:
        print(f"Error in dashboard-stats: {e}")
        import traceback
        traceback.print_exc()
        return {
            "total_messages": 0,
            "recent_pii_detections": 0,
            "total_pii_detections": 0,
            "jailbreak_attempts": 0,
            "toxicity_detected": 0
        }

@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get all messages in a conversation (admin only)."""
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.asc()).all()
    
    return [
        {
            "id": msg.id,
            "content": msg.content,
            "role": msg.role,
            "created_at": msg.created_at,
            "is_flagged": msg.is_flagged,
            "jailbreak_score": msg.jailbreak_score,
            "toxicity_score": msg.toxicity_score,
            "pii_detected": msg.pii_detected,
            "explanation": msg.explanation
        } for msg in messages
    ]

@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    new_role: str,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Update user role (admin only)."""
    if new_role not in ["USER", "ADMIN", "MODERATOR"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role. Must be USER, ADMIN, or MODERATOR"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.role = new_role
    db.commit()
    
    return {"message": f"User role updated to {new_role}"}

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Delete a conversation (admin only)."""
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Delete associated messages first
    db.query(Message).filter(Message.conversation_id == conversation_id).delete()
    
    # Delete conversation
    db.delete(conversation)
    db.commit()
    
    return {"message": "Conversation deleted successfully"}

@router.get("/security-logs")
async def get_security_logs(
    admin_user: User = Depends(require_admin),
    mongo_db = Depends(get_mongo),
    limit: int = 100,
    skip: int = 0
):
    """Get security logs from MongoDB (admin only)."""
    logs = list(mongo_db.security_logs.find().sort("timestamp", -1).skip(skip).limit(limit))
    
    # Convert ObjectId to string for JSON serialization
    for log in logs:
        log["_id"] = str(log["_id"])
    
    return logs

# ==================== Iteration 2 Dashboard Endpoints ====================

@router.get("/privacy-preserver/stats")
async def get_privacy_preserver_stats(
    admin_user = Depends(require_admin_optional),
    db: Session = Depends(get_db)
):
    """Get Privacy Preserver statistics (Iteration 2)."""
    from sqlalchemy import func, case
    from datetime import datetime, timedelta
    
    # Total PII detections
    total_pii_messages = db.query(Message).filter(Message.pii_detected == True).count()
    
    # PII by type (from pii_mappings JSON)
    messages_with_pii = db.query(Message).filter(
        Message.pii_detected == True,
        Message.pii_mappings.isnot(None)
    ).all()
    
    pii_by_type = {}
    for msg in messages_with_pii:
        if msg.pii_mappings:
            for placeholder, mapping in msg.pii_mappings.items():
                pii_type = mapping.get('type', 'unknown')
                pii_by_type[pii_type] = pii_by_type.get(pii_type, 0) + 1
    
    # Messages with masking
    masked_messages = db.query(Message).filter(
        Message.masked_content.isnot(None)
    ).count()
    
    # Recent PII detections (last 24 hours)
    yesterday = datetime.utcnow() - timedelta(days=1)
    recent_pii = db.query(Message).filter(
        Message.pii_detected == True,
        Message.created_at >= yesterday
    ).count()
    
    # PII detection rate
    total_messages = db.query(Message).count()
    pii_rate = (total_pii_messages / total_messages * 100) if total_messages > 0 else 0
    
    return {
        "total_pii_detections": total_pii_messages,
        "pii_by_type": pii_by_type,
        "masked_messages": masked_messages,
        "recent_pii_detections": recent_pii,
        "pii_detection_rate": round(pii_rate, 2),
        "total_messages": total_messages
    }

@router.get("/privacy-preserver/logs")
async def get_privacy_preserver_logs(
    admin_user = Depends(require_admin_optional),
    db: Session = Depends(get_db),
    limit: int = 50,
    skip: int = 0
):
    """Get PII detection log with original vs masked comparison (Iteration 2)."""
    if db is None:
        return []
    
    # Query for messages that have PII-related fields
    # Include messages where:
    # 1. pii_mappings is not None and not empty, OR
    # 2. original_content != masked_content (indicating masking occurred), OR
    # 3. pii_detected == True
    from sqlalchemy import or_
    
    messages = db.query(Message).filter(
        or_(
            # Has non-empty pii_mappings
            Message.pii_mappings.isnot(None),
            # Has different original and masked content (indicating masking)
            Message.original_content != Message.masked_content,
            # Explicitly flagged as PII detected
            Message.pii_detected == True
        ),
        Message.original_content.isnot(None)
    ).order_by(Message.created_at.desc()).offset(skip).limit(limit).all()
    
    # Filter and format results
    result = []
    for msg in messages:
        # Skip if pii_mappings is empty dict and content wasn't actually masked
        pii_mappings = msg.pii_mappings or {}
        original = msg.original_content or msg.content
        masked = msg.masked_content or msg.content
        
        # Include if: has non-empty mappings, or content was actually masked, or explicitly flagged
        if (pii_mappings and len(pii_mappings) > 0) or (original != masked) or msg.pii_detected:
            result.append({
                "id": msg.id,
                "timestamp": msg.created_at.isoformat() if msg.created_at else None,
                "original_content": original,
                "masked_content": masked,
                "pii_mappings": pii_mappings,
                "conversation_id": msg.conversation_id,
                "explanation": msg.explanation
            })
    
    return result

@router.get("/output-filter/stats")
async def get_output_filter_stats(
    admin_user = Depends(require_admin_optional),
    db: Session = Depends(get_db)
):
    """Get Output Filter statistics (Iteration 2)."""
    from datetime import datetime, timedelta
    from models.conversation import MessageRole
    
    # Total filtered responses (only actually flagged messages)
    total_filtered = db.query(Message).filter(
        Message.role == MessageRole.ASSISTANT,
        Message.is_flagged == True
    ).count()
    
    # Filter reasons breakdown (only for flagged messages)
    toxicity_count = db.query(Message).filter(
        Message.role == MessageRole.ASSISTANT,
        Message.is_flagged == True,
        Message.toxicity_score.isnot(None),
        Message.toxicity_score > 0
    ).count()
    
    bias_count = db.query(Message).filter(
        Message.role == MessageRole.ASSISTANT,
        Message.is_flagged == True,
        Message.bias_detected == True
    ).count()
    
    jailbreak_output_count = db.query(Message).filter(
        Message.role == MessageRole.ASSISTANT,
        Message.is_flagged == True,
        Message.jailbreak_detected_in_output == True
    ).count()
    
    # Severity distribution (only for flagged messages)
    messages_with_filter = db.query(Message).filter(
        Message.role == MessageRole.ASSISTANT,
        Message.is_flagged == True,
        Message.filter_analysis.isnot(None)
    ).all()
    
    severity_dist = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for msg in messages_with_filter:
        if msg.filter_analysis and msg.filter_analysis.get('severity'):
            severity = msg.filter_analysis['severity'].lower()
            if severity in severity_dist:
                severity_dist[severity] += 1
    
    # Sanitization strategy distribution
    sanitization_dist = {}
    for msg in messages_with_filter:
        if msg.sanitization_strategy:
            sanitization_dist[msg.sanitization_strategy] = sanitization_dist.get(msg.sanitization_strategy, 0) + 1
    
    # Recent filtered responses (last 24 hours)
    yesterday = datetime.utcnow() - timedelta(days=1)
    recent_filtered = db.query(Message).filter(
        Message.role == MessageRole.ASSISTANT,
        Message.is_flagged == True,
        Message.created_at >= yesterday
    ).count()
    
    # Filter rate
    total_responses = db.query(Message).filter(Message.role == MessageRole.ASSISTANT).count()
    filter_rate = (total_filtered / total_responses * 100) if total_responses > 0 else 0
    
    return {
        "total_filtered": total_filtered,
        "toxicity_detections": toxicity_count,
        "bias_detections": bias_count,
        "jailbreak_in_output": jailbreak_output_count,
        "severity_distribution": severity_dist,
        "sanitization_distribution": sanitization_dist,
        "recent_filtered": recent_filtered,
        "filter_rate": round(filter_rate, 2),
        "total_responses": total_responses
    }

@router.get("/output-filter/logs")
async def get_output_filter_logs(
    admin_user = Depends(require_admin_optional),
    db: Session = Depends(get_db),
    limit: int = 50,
    skip: int = 0
):
    """Get filtered responses log with original vs filtered comparison (Iteration 2)."""
    if db is None:
        return []
    
    # Query for messages that were actually filtered or had detections
    # Only include messages where:
    # 1. is_flagged == True (message was flagged), OR
    # 2. filtered_content is different from content (actually filtered), OR
    # 3. toxicity_score > 0 (toxicity detected), OR
    # 4. bias_detected == True (bias detected), OR
    # 5. jailbreak_detected_in_output == True (jailbreak detected)
    from sqlalchemy import or_, and_
    
    # Query for messages that were actually filtered or had detections
    # Show messages where something was actually detected and flagged
    # This ensures the dashboard shows only relevant filtered messages
    # 
    # Conditions to show a message:
    # 1. Actually filtered (filtered_content != content), OR
    # 2. Is flagged AND has toxicity_score > 0 (actually detected), OR
    # 3. Bias detected, OR
    # 4. Jailbreak detected
    #
    # Note: We show messages that were actually flagged/filtered,
    # not just messages that were processed (which may have low scores)
    from models.conversation import MessageRole
    
    # Only show messages that are actually flagged
    # This ensures we don't show messages that were processed but not flagged
    messages = db.query(Message).filter(
        Message.role == MessageRole.ASSISTANT,
        Message.is_flagged == True,  # Must be explicitly flagged
        or_(
            # Actually filtered (content was changed)
            and_(
                Message.filtered_content.isnot(None),
                Message.filtered_content != Message.content
            ),
            # Has toxicity score (actually detected)
            and_(
                Message.toxicity_score.isnot(None),
                Message.toxicity_score > 0
            ),
            # Bias detected
            Message.bias_detected == True,
            # Jailbreak detected
            Message.jailbreak_detected_in_output == True
        )
    ).order_by(Message.created_at.desc()).offset(skip).limit(limit).all()
    
    result = []
    for msg in messages:
        # original_response: Always show the ACTUAL original response (from filtered_content if available, otherwise content)
        # filtered_response: What was shown to user (censored version or placeholder)
        # Priority: filtered_content (original) > content (might be original if not filtered)
        original_to_show = msg.filtered_content if msg.filtered_content else msg.content
        # If content is a placeholder/censored message, we must have the original in filtered_content
        if not msg.filtered_content and msg.content and (
            msg.content.startswith("[REDACTED") or 
            msg.content.startswith("[RESPONSE BLOCKED") or
            msg.content.startswith("[Content censored")
        ):
            # This shouldn't happen, but if it does, try to get original from filter_analysis
            if msg.filter_analysis and isinstance(msg.filter_analysis, dict):
                original_to_show = msg.filter_analysis.get("original_response", msg.content)
        
        result.append({
            "id": msg.id,
            "timestamp": msg.created_at.isoformat() if msg.created_at else None,
            "original_response": original_to_show,  # Always show actual original response
            "filtered_response": msg.content,  # What user saw (censored if censored, placeholder if blocked)
            "filter_analysis": msg.filter_analysis,
            "toxicity_score": msg.toxicity_score,
            "bias_detected": msg.bias_detected or False,
            "bias_score": msg.bias_score,
            "jailbreak_detected": msg.jailbreak_detected_in_output or False,
            "jailbreak_score": msg.jailbreak_score_in_output,
            "sanitization_strategy": msg.sanitization_strategy,  # Already stored as string in DB
            "conversation_id": msg.conversation_id,
            "explanation": msg.explanation
        })
    
    return result

@router.get("/unified-security/stats")
async def get_unified_security_stats(
    admin_user = Depends(require_admin_optional),
    db: Session = Depends(get_db)
):
    """Get unified security overview statistics (Iteration 2)."""
    from datetime import datetime, timedelta
    
    # Overall stats
    total_conversations = db.query(Conversation).count()
    flagged_conversations = db.query(Conversation).filter(Conversation.is_flagged == True).count()
    
    # Module-specific stats
    jailbreak_prompt = db.query(Conversation).filter(Conversation.jailbreak_detected == True).count()
    pii_detected = db.query(Conversation).filter(Conversation.pii_detected == True).count()
    toxicity_detected = db.query(Conversation).filter(Conversation.toxicity_detected == True).count()
    bias_detected = db.query(Conversation).filter(Conversation.bias_detected == True).count()
    jailbreak_output = db.query(Conversation).filter(Conversation.jailbreak_detected_in_output == True).count()
    
    # Security health score (0-100)
    # Higher score = better security (fewer threats)
    if total_conversations > 0:
        threat_rate = (flagged_conversations / total_conversations) * 100
        health_score = max(0, 100 - threat_rate)
    else:
        health_score = 100
    
    # Recent activity (last 24 hours)
    yesterday = datetime.utcnow() - timedelta(days=1)
    recent_conversations = db.query(Conversation).filter(
        Conversation.created_at >= yesterday
    ).count()
    recent_flagged = db.query(Conversation).filter(
        Conversation.created_at >= yesterday,
        Conversation.is_flagged == True
    ).count()
    
    # Threat trends (last 7 days)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    daily_threats = []
    for i in range(7):
        day_start = seven_days_ago + timedelta(days=i)
        day_end = day_start + timedelta(days=1)
        count = db.query(Conversation).filter(
            Conversation.created_at >= day_start,
            Conversation.created_at < day_end,
            Conversation.is_flagged == True
        ).count()
        daily_threats.append({
            "date": day_start.strftime("%Y-%m-%d"),
            "threats": count
        })
    
    return {
        "total_conversations": total_conversations,
        "flagged_conversations": flagged_conversations,
        "security_health_score": round(health_score, 2),
        "module_stats": {
            "jailbreak_prompt": jailbreak_prompt,
            "pii_detected": pii_detected,
            "toxicity_detected": toxicity_detected,
            "bias_detected": bias_detected,
            "jailbreak_output": jailbreak_output
        },
        "recent_activity": {
            "conversations": recent_conversations,
            "flagged": recent_flagged
        },
        "threat_trends": daily_threats
    }






    return {"message": "Conversation deleted successfully"}

@router.get("/security-logs")
async def get_security_logs(
    admin_user: User = Depends(require_admin),
    mongo_db = Depends(get_mongo),
    limit: int = 100,
    skip: int = 0
):
    """Get security logs from MongoDB (admin only)."""
    logs = list(mongo_db.security_logs.find().sort("timestamp", -1).skip(skip).limit(limit))
    
    # Convert ObjectId to string for JSON serialization
    for log in logs:
        log["_id"] = str(log["_id"])
    
    return logs

# ==================== Iteration 2 Dashboard Endpoints ====================

@router.get("/privacy-preserver/stats")
async def get_privacy_preserver_stats(
    admin_user = Depends(require_admin_optional),
    db: Session = Depends(get_db)
):
    """Get Privacy Preserver statistics (Iteration 2)."""
    from sqlalchemy import func, case
    from datetime import datetime, timedelta
    
    # Total PII detections
    total_pii_messages = db.query(Message).filter(Message.pii_detected == True).count()
    
    # PII by type (from pii_mappings JSON)
    messages_with_pii = db.query(Message).filter(
        Message.pii_detected == True,
        Message.pii_mappings.isnot(None)
    ).all()
    
    pii_by_type = {}
    for msg in messages_with_pii:
        if msg.pii_mappings:
            for placeholder, mapping in msg.pii_mappings.items():
                pii_type = mapping.get('type', 'unknown')
                pii_by_type[pii_type] = pii_by_type.get(pii_type, 0) + 1
    
    # Messages with masking
    masked_messages = db.query(Message).filter(
        Message.masked_content.isnot(None)
    ).count()
    
    # Recent PII detections (last 24 hours)
    yesterday = datetime.utcnow() - timedelta(days=1)
    recent_pii = db.query(Message).filter(
        Message.pii_detected == True,
        Message.created_at >= yesterday
    ).count()
    
    # PII detection rate
    total_messages = db.query(Message).count()
    pii_rate = (total_pii_messages / total_messages * 100) if total_messages > 0 else 0
    
    return {
        "total_pii_detections": total_pii_messages,
        "pii_by_type": pii_by_type,
        "masked_messages": masked_messages,
        "recent_pii_detections": recent_pii,
        "pii_detection_rate": round(pii_rate, 2),
        "total_messages": total_messages
    }

@router.get("/unified-security/stats")
async def get_unified_security_stats(
    admin_user = Depends(require_admin_optional),
    db: Session = Depends(get_db)
):
    """Get unified security overview statistics (Iteration 2)."""
    from datetime import datetime, timedelta
    
    # Overall stats
    total_conversations = db.query(Conversation).count()
    flagged_conversations = db.query(Conversation).filter(Conversation.is_flagged == True).count()
    
    # Module-specific stats
    jailbreak_prompt = db.query(Conversation).filter(Conversation.jailbreak_detected == True).count()
    pii_detected = db.query(Conversation).filter(Conversation.pii_detected == True).count()
    toxicity_detected = db.query(Conversation).filter(Conversation.toxicity_detected == True).count()
    bias_detected = db.query(Conversation).filter(Conversation.bias_detected == True).count()
    jailbreak_output = db.query(Conversation).filter(Conversation.jailbreak_detected_in_output == True).count()
    
    # Security health score (0-100)
    # Higher score = better security (fewer threats)
    if total_conversations > 0:
        threat_rate = (flagged_conversations / total_conversations) * 100
        health_score = max(0, 100 - threat_rate)
    else:
        health_score = 100
    
    # Recent activity (last 24 hours)
    yesterday = datetime.utcnow() - timedelta(days=1)
    recent_conversations = db.query(Conversation).filter(
        Conversation.created_at >= yesterday
    ).count()
    recent_flagged = db.query(Conversation).filter(
        Conversation.created_at >= yesterday,
        Conversation.is_flagged == True
    ).count()
    
    # Threat trends (last 7 days)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    daily_threats = []
    for i in range(7):
        day_start = seven_days_ago + timedelta(days=i)
        day_end = day_start + timedelta(days=1)
        count = db.query(Conversation).filter(
            Conversation.created_at >= day_start,
            Conversation.created_at < day_end,
            Conversation.is_flagged == True
        ).count()
        daily_threats.append({
            "date": day_start.strftime("%Y-%m-%d"),
            "threats": count
        })
    
    return {
        "total_conversations": total_conversations,
        "flagged_conversations": flagged_conversations,
        "security_health_score": round(health_score, 2),
        "module_stats": {
            "jailbreak_prompt": jailbreak_prompt,
            "pii_detected": pii_detected,
            "toxicity_detected": toxicity_detected,
            "bias_detected": bias_detected,
            "jailbreak_output": jailbreak_output
        },
        "recent_activity": {
            "conversations": recent_conversations,
            "flagged": recent_flagged
        },
        "threat_trends": daily_threats
    }





