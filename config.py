"""
Configuration settings for the GovAI backend.
Environment variables are loaded from .env file.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://hujdkqvkgkdislmdnhqt.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh1amRrcXZrZ2tkaXNsbWRuaHF0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU3MzI1NzcsImV4cCI6MjA4MTMwODU3N30.z8Vf21-r9JKNtRLFc0FbbB0D33zdCvJ9ogdJLpQpZAo")

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Application SettingseyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh1amRrcXZrZ2tkaXNsbWRuaHF0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU3MzI1NzcsImV4cCI6MjA4MTMwODU3N30.z8Vf21-r9JKNtRLFc0FbbB0D33zdCvJ9ogdJLpQpZAo
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001,http://localhost:3002").split(",")
