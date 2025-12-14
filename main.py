"""
GovAI: Unsupervised Electricity Theft Detection System
Main FastAPI Application

This module provides the REST API for the electricity theft detection system.
"""

import os
import io
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Local imports
from data_preprocessing import clean_data, aggregate_by_consumer, get_consumer_timeseries
from feature_engineering import create_feature_matrix, scale_features, identify_feature_profiles
from model_isolationforest import run_isolation_forest_analysis, get_analysis_summary
from model_autoencoder import run_autoencoder_analysis, ensemble_scores
from report_generator import ReportGenerator
from utils.data_utils import generate_ai_explanation, get_anomaly_status
from auth import (
    UserRegister, 
    UserLogin, 
    register_user, 
    login_user, 
    get_current_user,
    get_current_user_optional
)


# Data storage (in-memory for demo, replace with DB for production)
class DataStore:
    def __init__(self):
        self.raw_data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.aggregated_data: Optional[pd.DataFrame] = None
        self.results: Optional[pd.DataFrame] = None
        self.summary: Optional[Dict[str, Any]] = None
        self.analysis_history: List[Dict[str, Any]] = []
        self.current_analysis_id: Optional[str] = None
        self.upload_filename: Optional[str] = None
        self.current_user_id: Optional[str] = None


data_store = DataStore()


# Create required directories
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), 'uploads')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


# Pydantic models for API
class AnalyzeRequest(BaseModel):
    contamination: float = 0.05
    use_autoencoder: bool = False


class ConsumerResponse(BaseModel):
    consumer_id: str
    region: str
    avg_consumption: float
    anomaly_score: float
    status: str
    is_anomaly: bool


class SummaryResponse(BaseModel):
    total_consumers: int
    anomalies_detected: int
    high_risk_count: int
    suspicious_count: int
    review_needed_count: int
    avg_anomaly_score: float
    high_risk_zones: List[str]
    zone_anomaly_counts: Dict[str, int]


class AnalysisResultResponse(BaseModel):
    analysis_id: str
    summary: SummaryResponse
    consumers: List[Dict[str, Any]]
    timestamp: str


# FastAPI app
app = FastAPI(
    title="GovAI: Electricity Theft Detection API",
    description="Unsupervised machine learning system for detecting electricity theft patterns",
    version="1.0.0"
)

# CORS middleware
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001,http://localhost:3002").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],  # Allow configured + all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ PUBLIC ENDPOINTS ============

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "GovAI Electricity Theft Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "auth": {
                "register": "/api/auth/register",
                "login": "/api/auth/login",
                "me": "/api/auth/me"
            },
            "upload": "/api/upload",
            "analyze": "/api/analyze",
            "results": "/api/results",
            "consumer": "/api/consumer/{consumer_id}",
            "report": "/api/report",
            "history": "/api/history"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": data_store.raw_data is not None,
        "analysis_complete": data_store.results is not None
    }


# ============ AUTH ENDPOINTS ============

@app.post("/api/auth/register")
async def api_register(request: UserRegister):
    """
    Register a new user account.
    
    Returns JWT token on successful registration.
    """
    try:
        result = register_user(request.email, request.password, request.name)
        return result
    except HTTPException:
        raise
    except Exception as e:
        # If Supabase is not configured, return helpful error
        if "Supabase credentials not configured" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Database not configured. Please set SUPABASE_URL and SUPABASE_KEY in .env file."
            )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/login")
async def api_login(request: UserLogin):
    """
    Login with email and password.
    
    Returns JWT token on successful login.
    """
    try:
        result = login_user(request.email, request.password)
        return result
    except HTTPException:
        raise
    except Exception as e:
        if "Supabase credentials not configured" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Database not configured. Please set SUPABASE_URL and SUPABASE_KEY in .env file."
            )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auth/me")
async def api_get_me(current_user: dict = Depends(get_current_user)):
    """
    Get current authenticated user information.
    
    Requires valid JWT token in Authorization header.
    """
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "name": current_user["name"],
        "created_at": current_user["created_at"]
    }


# ============ PROTECTED ENDPOINTS ============

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload CSV file with smart meter data.
    
    Expected columns: LCLid/consumer_id, DateTime/timestamp, KWH/hh/consumption
    Requires authentication.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read file content
        contents = await file.read()
        
        # Parse CSV
        df = pd.read_csv(io.BytesIO(contents))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Store raw data with user context
        data_store.raw_data = df
        data_store.upload_filename = file.filename
        data_store.current_user_id = current_user["id"]
        
        # Generate preview (first 5 rows)
        preview = df.head(5).to_dict('records')
        
        return {
            "status": "success",
            "message": "File uploaded successfully",
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "preview": preview,
            "user_id": current_user["id"]
        }
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or malformed")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/api/analyze")
async def analyze_data(
    request: AnalyzeRequest = AnalyzeRequest(),
    current_user: dict = Depends(get_current_user)
):
    """
    Run anomaly detection analysis on uploaded data.
    
    Parameters:
    - contamination: Expected proportion of anomalies (0.01-0.5)
    - use_autoencoder: Whether to use autoencoder (slower but may be more accurate)
    
    Requires authentication.
    """
    try:
        if data_store.raw_data is None:
            raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())[:8]
        data_store.current_analysis_id = analysis_id
        
        # Step 1: Clean data
        cleaned_df, cleaning_report = clean_data(data_store.raw_data)
        data_store.cleaned_data = cleaned_df
        
        if len(cleaned_df) == 0:
            raise HTTPException(status_code=400, detail="No valid data after cleaning")
        
        # Step 2: Aggregate by consumer
        aggregated_df = aggregate_by_consumer(cleaned_df)
        data_store.aggregated_data = aggregated_df
        
        if len(aggregated_df) == 0:
            raise HTTPException(status_code=400, detail="No consumers found after aggregation")
        
        # Step 3: Create features
        feature_matrix, feature_names, features_df = create_feature_matrix(aggregated_df)
        feature_matrix_scaled, scaler = scale_features(feature_matrix)
        
        # Step 4: Run Isolation Forest
        results_df = run_isolation_forest_analysis(
            features_df, 
            feature_matrix_scaled, 
            feature_names,
            contamination=request.contamination
        )
        
        # Step 5: Optionally run Autoencoder
        if request.use_autoencoder:
            try:
                ae_results = run_autoencoder_analysis(
                    results_df,
                    feature_matrix_scaled,
                    feature_names
                )
                
                # Ensemble scores
                if 'ae_anomaly_score' in ae_results.columns:
                    results_df['ae_anomaly_score'] = ae_results['ae_anomaly_score']
                    results_df['ensemble_score'] = ensemble_scores(
                        results_df['anomaly_score'].values,
                        results_df['ae_anomaly_score'].values
                    )
                    results_df['anomaly_score'] = results_df['ensemble_score']
                    results_df['status'] = results_df['anomaly_score'].apply(
                        lambda x: get_anomaly_status(x)
                    )
            except Exception as e:
                print(f"Autoencoder failed, using Isolation Forest only: {e}")
        
        # Step 6: Add AI explanations
        results_df['ai_explanation'] = results_df.apply(
            lambda row: generate_ai_explanation(row.to_dict()), 
            axis=1
        )
        
        # Identify consumption profiles
        results_df = identify_feature_profiles(results_df)
        
        # Store results
        data_store.results = results_df
        
        # Generate summary
        summary = get_analysis_summary(results_df)
        data_store.summary = summary
        
        # Add to history with user ID
        history_entry = {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "filename": data_store.upload_filename,
            "total_consumers": summary['total_consumers'],
            "anomalies_detected": summary['anomalies_detected'],
            "contamination": request.contamination,
            "used_autoencoder": request.use_autoencoder,
            "user_id": current_user["id"]
        }
        data_store.analysis_history.append(history_entry)
        
        # TODO: Save to Supabase database
        # database.save_analysis(...)
        
        return {
            "status": "success",
            "analysis_id": analysis_id,
            "message": "Analysis completed successfully",
            "summary": summary,
            "cleaning_report": cleaning_report,
            "consumers_analyzed": len(results_df),
            "timestamp": datetime.now().isoformat(),
            "user_id": current_user["id"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.get("/api/results")
async def get_results(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=500),
    status_filter: Optional[str] = Query(None, description="Filter by status: 'High Risk', 'Suspicious', 'Review Needed', 'Normal'"),
    region_filter: Optional[str] = Query(None, description="Filter by region/zone"),
    sort_by: str = Query("anomaly_score", description="Sort field"),
    sort_desc: bool = Query(True, description="Sort descending"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get analysis results with pagination and filtering.
    Requires authentication.
    """
    if data_store.results is None or data_store.summary is None:
        raise HTTPException(status_code=400, detail="No analysis results available. Please run analysis first.")
    
    df = data_store.results.copy()
    
    # Apply filters
    if status_filter:
        df = df[df['status'] == status_filter]
    
    if region_filter:
        df = df[df['region'] == region_filter]
    
    # Sort
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not sort_desc)
    
    # Paginate
    total = len(df)
    start = (page - 1) * limit
    end = start + limit
    page_data = df.iloc[start:end]
    
    # Prepare consumer list
    consumers = []
    for _, row in page_data.iterrows():
        consumers.append({
            "consumer_id": str(row.get('consumer_id', 'N/A')),
            "region": str(row.get('region', 'N/A')),
            "avg_consumption": round(float(row.get('avg_consumption', 0)), 3),
            "anomaly_score": round(float(row.get('anomaly_score', 0)), 4),
            "status": str(row.get('status', 'N/A')),
            "is_anomaly": bool(row.get('is_anomaly', False)),
            "consumption_profile": str(row.get('consumption_profile', 'Normal'))
        })
    
    return {
        "analysis_id": data_store.current_analysis_id,
        "summary": data_store.summary,
        "consumers": consumers,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": (total + limit - 1) // limit
        },
        "filters": {
            "status": status_filter,
            "region": region_filter
        }
    }


@app.get("/api/consumer/{consumer_id}")
async def get_consumer_detail(
    consumer_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get detailed information for a specific consumer.
    Includes identification of anomalous individual readings.
    Requires authentication.
    """
    if data_store.results is None:
        raise HTTPException(status_code=400, detail="No analysis results available")
    
    # Find consumer in results
    consumer_data = data_store.results[
        data_store.results['consumer_id'].astype(str) == consumer_id
    ]
    
    if consumer_data.empty:
        raise HTTPException(status_code=404, detail=f"Consumer {consumer_id} not found")
    
    consumer_row = consumer_data.iloc[0]
    
    # Get time series data if available
    timeseries = []
    anomalous_records = []
    
    if data_store.cleaned_data is not None:
        ts_data = get_consumer_timeseries(data_store.cleaned_data, consumer_id)
        if not ts_data.empty:
            # Calculate thresholds for anomaly detection at reading level
            avg_consumption = float(consumer_row.get('avg_consumption', 0))
            std_consumption = float(consumer_row.get('std_consumption', 0))
            
            # Thresholds for flagging individual readings
            low_threshold = max(0.01, avg_consumption * 0.1)
            high_threshold = avg_consumption + (3 * std_consumption) if std_consumption > 0 else avg_consumption * 3
            zero_threshold = 0.005
            
            for idx, row in ts_data.head(500).iterrows():
                consumption = float(row.get('consumption', 0))
                timestamp = row['timestamp'].isoformat() if pd.notna(row.get('timestamp')) else None
                
                # Determine if this reading is anomalous
                is_anomalous = False
                anomaly_reason = None
                
                if consumption <= zero_threshold:
                    is_anomalous = True
                    anomaly_reason = "Zero/Near-zero reading - possible meter bypass"
                elif consumption < low_threshold:
                    is_anomalous = True
                    anomaly_reason = "Unusually low reading"
                elif consumption > high_threshold:
                    is_anomalous = True
                    anomaly_reason = "Unusually high spike"
                
                # Check for night-time anomalies
                if timestamp and not is_anomalous:
                    try:
                        hour = pd.to_datetime(timestamp).hour
                        if 1 <= hour <= 5 and consumption > avg_consumption * 1.5:
                            is_anomalous = True
                            anomaly_reason = "High consumption during night hours (1-5 AM)"
                    except:
                        pass
                
                record = {
                    "timestamp": timestamp,
                    "consumption": round(consumption, 4),
                    "is_anomalous": is_anomalous,
                    "anomaly_reason": anomaly_reason
                }
                
                timeseries.append(record)
                
                if is_anomalous:
                    anomalous_records.append(record)
    
    # Build response
    response = {
        "consumer_id": consumer_id,
        "region": str(consumer_row.get('region', 'N/A')),
        "avg_consumption": round(float(consumer_row.get('avg_consumption', 0)), 3),
        "std_consumption": round(float(consumer_row.get('std_consumption', 0)), 3),
        "min_consumption": round(float(consumer_row.get('min_consumption', 0)), 3),
        "max_consumption": round(float(consumer_row.get('max_consumption', 0)), 3),
        "anomaly_score": round(float(consumer_row.get('anomaly_score', 0)), 4),
        "status": str(consumer_row.get('status', 'N/A')),
        "is_anomaly": bool(consumer_row.get('is_anomaly', False)),
        "consumption_profile": str(consumer_row.get('consumption_profile', 'Normal')),
        "ai_explanation": str(consumer_row.get('ai_explanation', 'No explanation available')),
        "night_day_ratio": round(float(consumer_row.get('night_day_ratio', 0)), 3),
        "weekend_weekday_ratio": round(float(consumer_row.get('weekend_weekday_ratio', 0)), 3),
        "timeseries": timeseries,
        "anomalous_records": anomalous_records,
        "anomalous_record_count": len(anomalous_records),
        "total_records": len(timeseries)
    }
    
    return response


@app.get("/api/report")
async def download_report(
    format: str = Query("pdf", description="Report format: 'pdf' or 'excel'"),
    current_user: dict = Depends(get_current_user)
):
    """
    Generate and download analysis report.
    Requires authentication.
    """
    if data_store.results is None or data_store.summary is None:
        raise HTTPException(status_code=400, detail="No analysis results available")
    
    try:
        generator = ReportGenerator(REPORTS_DIR)
        
        # Prepare consumers list
        consumers = []
        for _, row in data_store.results.iterrows():
            consumers.append({
                "consumer_id": str(row.get('consumer_id', 'N/A')),
                "region": str(row.get('region', 'N/A')),
                "avg_consumption": float(row.get('avg_consumption', 0)),
                "anomaly_score": float(row.get('anomaly_score', 0)),
                "status": str(row.get('status', 'N/A'))
            })
        
        # Generate report
        if format.lower() == 'pdf':
            filepath = generator.generate_pdf_report(data_store.summary, consumers)
            media_type = "application/pdf"
        else:
            filepath = generator.generate_excel_report(data_store.summary, consumers)
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
        filename = os.path.basename(filepath)
        
        return FileResponse(
            filepath,
            media_type=media_type,
            filename=filename
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


@app.get("/api/history")
async def get_analysis_history(
    current_user: dict = Depends(get_current_user)
):
    """
    Get history of all analyses performed by the current user.
    Requires authentication.
    """
    # Filter history by user
    user_history = [
        h for h in data_store.analysis_history 
        if h.get("user_id") == current_user["id"]
    ]
    
    return {
        "history": user_history,
        "total": len(user_history)
    }


@app.get("/api/zones")
async def get_zones(
    current_user: dict = Depends(get_current_user)
):
    """
    Get list of zones and their anomaly statistics.
    Requires authentication.
    """
    if data_store.results is None:
        raise HTTPException(status_code=400, detail="No analysis results available")
    
    zone_stats = data_store.results.groupby('region').agg({
        'consumer_id': 'count',
        'anomaly_score': 'mean',
        'is_anomaly': 'sum'
    }).reset_index()
    
    zone_stats.columns = ['zone', 'consumer_count', 'avg_anomaly_score', 'anomaly_count']
    
    zones = []
    for _, row in zone_stats.iterrows():
        zones.append({
            "zone": row['zone'],
            "consumer_count": int(row['consumer_count']),
            "avg_anomaly_score": round(float(row['avg_anomaly_score']), 4),
            "anomaly_count": int(row['anomaly_count']),
            "anomaly_rate": round(float(row['anomaly_count'] / row['consumer_count']), 4)
        })
    
    return {"zones": sorted(zones, key=lambda x: x['anomaly_rate'], reverse=True)}


# For running with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
