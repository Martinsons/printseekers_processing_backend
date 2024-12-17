from pathlib import Path
import logging
from typing import Optional
import os
from fastapi import UploadFile
import aiofiles
import asyncio
from datetime import datetime

class StorageManager:
    def __init__(self):
        self.temp_dir = Path("temp")
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure necessary directories exist"""
        self.temp_dir.mkdir(exist_ok=True)

    async def save_temp_file(self, file: UploadFile) -> Path:
        """Save an uploaded file temporarily"""
        self.temp_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = self.temp_dir / safe_filename

        try:
            async with aiofiles.open(file_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)
            logging.info(f"Temporary file saved: {file_path}")
            return file_path
        except Exception as e:
            logging.error(f"Error saving temporary file {file_path}: {str(e)}")
            raise

    async def cleanup_temp_files(self, max_age_hours: int = 1):
        """Clean up temporary files older than specified hours"""
        current_time = datetime.now().timestamp()
        
        if not self.temp_dir.exists():
            return

        for file_path in self.temp_dir.glob("*"):
            if file_path.is_file():
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > (max_age_hours * 3600):
                    try:
                        file_path.unlink()
                        logging.info(f"Cleaned up old temporary file: {file_path}")
                    except Exception as e:
                        logging.error(f"Error deleting temporary file {file_path}: {str(e)}")

    def remove_temp_file(self, file_path: Path):
        """Remove a specific temporary file"""
        try:
            if file_path.exists() and file_path.is_relative_to(self.temp_dir):
                file_path.unlink()
                logging.info(f"Removed temporary file: {file_path}")
        except Exception as e:
            logging.error(f"Error removing temporary file {file_path}: {str(e)}")

storage = StorageManager()
