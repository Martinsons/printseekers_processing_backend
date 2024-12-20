from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Set
import os
from functools import lru_cache

class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "Batch Processing App"
    DEBUG: bool = False
    
    # API Settings
    API_V1_STR: str = "/api"
    
    # Supabase Settings
    SUPABASE_URL: str
    SUPABASE_KEY: str
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",  # Vue dev server
        "http://localhost:5174",  # Vue dev server alternate port
        "http://localhost:5175",  # Vue dev server alternate port
        "http://localhost:8000",  # Local backend
        "https://printseekers-processing-backend.onrender.com",  # Your Render app
        "https://*.netlify.app",  # Netlify deployed frontend
        "https://printseekerstest1.netlify.app"  # Your Netlify app
    ]
    
    # File Settings
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: Set[str] = {'.xlsx', '.csv', '.pdf'}
    
    # Storage Settings
    STORAGE_TYPE: str = "local"
    PROCESSED_DIR: str = "processed_files/default"
    TEMP_DIR: str = "temp"
    UPLOAD_DIR: str = "uploads"
    
    # Cleanup Settings
    FILE_CLEANUP_HOURS: int = 24
    
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding='utf-8'
    )

@lru_cache()
def get_settings() -> Settings:
    try:
        settings = Settings()
        # Validate required settings
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError("Missing required Supabase configuration")
        return settings
    except Exception as e:
        raise Exception(f"Failed to load settings: {str(e)}")

# Constants