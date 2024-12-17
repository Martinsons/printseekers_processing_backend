from pathlib import Path
import logging
from typing import Optional
import secrets
import string

def generate_secure_filename(original_filename: str) -> str:
    """Generate a secure filename with random component"""
    # Get the file extension
    suffix = Path(original_filename).suffix
    
    # Generate random string
    alphabet = string.ascii_letters + string.digits
    random_part = ''.join(secrets.choice(alphabet) for _ in range(16))
    
    return f"{random_part}{suffix}"

def validate_file_path(file_path: Path, base_dir: Path) -> bool:
    """Validate that a file path is within allowed directory"""
    try:
        return file_path.resolve().is_relative_to(base_dir.resolve())
    except Exception as e:
        logging.error(f"Path validation error: {str(e)}")
        return False

def validate_file_size(file_size: int, max_size_mb: int = 10) -> bool:
    """Validate file size"""
    return file_size <= (max_size_mb * 1024 * 1024)  # Convert MB to bytes

def validate_file_type(filename: str, allowed_extensions: set) -> bool:
    """Validate file extension"""
    return Path(filename).suffix.lower() in allowed_extensions
