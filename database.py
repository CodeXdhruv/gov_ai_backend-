"""
Database utilities for Supabase integration.
"""
from supabase import create_client, Client
from typing import Optional, Dict, Any, List
from config import SUPABASE_URL, SUPABASE_KEY

# Initialize Supabase client
_supabase_client: Optional[Client] = None


def get_supabase() -> Client:
    """Get or create Supabase client instance."""
    global _supabase_client
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(
            "Supabase credentials not configured. "
            "Please set SUPABASE_URL and SUPABASE_KEY in your .env file."
        )
    
    if _supabase_client is None:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    return _supabase_client


# ============ USER OPERATIONS ============

def create_user(email: str, password_hash: str, name: str) -> Optional[Dict[str, Any]]:
    """Create a new user in the database."""
    supabase = get_supabase()
    
    result = supabase.table("users").insert({
        "email": email,
        "password_hash": password_hash,
        "name": name
    }).execute()
    
    if result.data:
        return result.data[0]
    return None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email address."""
    supabase = get_supabase()
    
    result = supabase.table("users").select("*").eq("email", email).execute()
    
    if result.data and len(result.data) > 0:
        return result.data[0]
    return None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user by ID."""
    supabase = get_supabase()
    
    result = supabase.table("users").select("*").eq("id", user_id).execute()
    
    if result.data and len(result.data) > 0:
        return result.data[0]
    return None


# ============ ANALYSIS OPERATIONS ============

def save_analysis(
    user_id: str,
    analysis_id: str,
    filename: str,
    total_consumers: int,
    anomalies_detected: int,
    contamination: float,
    used_autoencoder: bool,
    summary: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Save analysis record to database."""
    supabase = get_supabase()
    
    result = supabase.table("analyses").insert({
        "user_id": user_id,
        "analysis_id": analysis_id,
        "filename": filename,
        "total_consumers": total_consumers,
        "anomalies_detected": anomalies_detected,
        "contamination": contamination,
        "used_autoencoder": used_autoencoder,
        "summary": summary
    }).execute()
    
    if result.data:
        return result.data[0]
    return None


def get_user_analyses(user_id: str) -> List[Dict[str, Any]]:
    """Get all analyses for a user."""
    supabase = get_supabase()
    
    result = supabase.table("analyses") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .execute()
    
    return result.data if result.data else []


def get_analysis_by_id(analysis_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific analysis by ID for a user."""
    supabase = get_supabase()
    
    result = supabase.table("analyses") \
        .select("*") \
        .eq("analysis_id", analysis_id) \
        .eq("user_id", user_id) \
        .execute()
    
    if result.data and len(result.data) > 0:
        return result.data[0]
    return None


# ============ CONSUMER OPERATIONS ============

def save_consumers(analysis_ref: str, consumers: List[Dict[str, Any]]) -> bool:
    """Save consumer records for an analysis."""
    if not consumers:
        return True
        
    supabase = get_supabase()
    
    # Prepare records with analysis reference
    records = []
    for consumer in consumers:
        records.append({
            "analysis_ref": analysis_ref,
            "consumer_id": consumer.get("consumer_id"),
            "region": consumer.get("region"),
            "avg_consumption": consumer.get("avg_consumption"),
            "anomaly_score": consumer.get("anomaly_score"),
            "status": consumer.get("status"),
            "is_anomaly": consumer.get("is_anomaly"),
            "features": consumer.get("features", {})
        })
    
    # Insert in batches of 100
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table("consumers").insert(batch).execute()
    
    return True


def get_analysis_consumers(
    analysis_ref: str,
    page: int = 1,
    limit: int = 50,
    status_filter: Optional[str] = None
) -> tuple[List[Dict[str, Any]], int]:
    """Get consumers for an analysis with pagination."""
    supabase = get_supabase()
    
    query = supabase.table("consumers").select("*", count="exact").eq("analysis_ref", analysis_ref)
    
    if status_filter:
        query = query.eq("status", status_filter)
    
    # Apply pagination
    offset = (page - 1) * limit
    query = query.order("anomaly_score", desc=True).range(offset, offset + limit - 1)
    
    result = query.execute()
    
    total = result.count if result.count else 0
    consumers = result.data if result.data else []
    
    return consumers, total
