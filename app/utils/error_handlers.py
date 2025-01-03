from fastapi import HTTPException, status
from typing import Optional
import asyncio

class FileProcessingError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail
        )

class FileNotFoundError(HTTPException):
    def __init__(self, detail: str = "File not found"):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail
        )

class InvalidFileError(HTTPException):
    def __init__(self, detail: str = "Invalid file"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )

class StorageError(HTTPException):
    def __init__(self, detail: str = "Storage operation failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )

async def handle_file_operation(operation: callable, error_msg: str, *args, **kwargs):
    """Generic error handler for file operations"""
    try:
        if asyncio.iscoroutinefunction(operation):
            return await operation(*args, **kwargs)
        return operation(*args, **kwargs)
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e))
    except PermissionError:
        raise StorageError("Permission denied while accessing file")
    except Exception as e:
        raise StorageError(f"{error_msg}: {str(e)}")
