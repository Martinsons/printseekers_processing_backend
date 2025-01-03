import os
import sys
import logging
import gc
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import asyncio
from contextlib import asynccontextmanager
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from urllib.parse import unquote
from .config import get_settings, Settings
from .supabase import supabaseDb
from .processors.fedex_bill.processor import FedexBillProcessor
from .utils.storage import storage
from .utils.security import (
    validate_file_path,
    validate_file_type,
    validate_file_size
)
from .utils.error_handlers import (
    FileProcessingError,
    FileNotFoundError,
    InvalidFileError,
    StorageError,
    handle_file_operation
)

# Constants for file handling and memory management
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB limit
CHUNK_SIZE = 64 * 1024  # Reduced to 64KB for better memory management
MAX_CONCURRENT_REQUESTS = 4
REQUEST_TIMEOUT = 300  # 5 minutes
BATCH_SIZE = 50  # Batch size for database operations
GC_THRESHOLD = 100 * 1024 * 1024  # 100MB threshold for garbage collection
PROCESSING_DELAY = 0.5  # Delay between batch processing

app = FastAPI(
    title="Batch Processing API",
    docs_url="/docs",
    redoc_url="/redoc"
)
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configure memory optimization settings
app.conf = {
    "MAX_CONTENT_LENGTH": MAX_UPLOAD_SIZE,
    "CHUNK_SIZE": CHUNK_SIZE,
    "MAX_CONCURRENT_REQUESTS": MAX_CONCURRENT_REQUESTS,
    "REQUEST_TIMEOUT": REQUEST_TIMEOUT,
    "BATCH_SIZE": BATCH_SIZE
}

@asynccontextmanager
async def managed_file_operation():
    """Context manager for file operations with memory management"""
    try:
        yield
    finally:
        gc.collect()

# Add middleware for request limiting
@app.middleware("http")
async def limit_concurrent_requests(request: Request, call_next):
    app.state.active_requests = getattr(app.state, "active_requests", 0)
    
    if app.state.active_requests >= MAX_CONCURRENT_REQUESTS:
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "Server is busy. Please try again later."}
        )
    
    app.state.active_requests += 1
    try:
        response = await call_next(request)
        return response
    finally:
        app.state.active_requests -= 1

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"success": False, "error": "Request timeout"}
        )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "localhost",
        "127.0.0.1",
        "printseekers-processing-backend.onrender.com"
    ]
)

# Ensure temp directory exists
os.makedirs(settings.TEMP_DIR, exist_ok=True)

startup_completed = False

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("=== Application Startup ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"PYTHONPATH: {os.getenv('PYTHONPATH')}")
        logger.info(f"PORT: {os.getenv('PORT')}")
        
        # Initialize directories
        for dir_path in [settings.TEMP_DIR, settings.UPLOAD_DIR, settings.PROCESSED_DIR]:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created/verified directory: {dir_path}")
        
        logger.info("Application initialized")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        raise

async def validate_upload_file(file: UploadFile) -> UploadFile:
    """Validate uploaded file with memory-efficient chunk reading"""
    if not validate_file_type(file.filename, settings.ALLOWED_EXTENSIONS):
        raise InvalidFileError(f"Invalid file type. Allowed types: {settings.ALLOWED_EXTENSIONS}")
    
    # Read file in chunks to validate size
    total_size = 0
    while chunk := await file.read(CHUNK_SIZE):
        total_size += len(chunk)
        if total_size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise InvalidFileError(f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB")
        await asyncio.sleep(0)
    
    await file.seek(0)
    return file

@app.post("/api/process/fedex-bill")
async def process_fedex_bill(file: UploadFile = File(...)):
    temp_file_path = None
    try:
        await validate_upload_file(file)
        temp_file_path = Path(settings.UPLOAD_DIR) / f"temp_{datetime.now().timestamp()}.pdf"
        
        async with managed_file_operation():
            with open(temp_file_path, "wb") as temp_file:
                while chunk := await file.read(CHUNK_SIZE):
                    temp_file.write(chunk)
                    await asyncio.sleep(0)
            
            processor = FedexBillProcessor(str(temp_file_path))
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, processor.process)
            
            return JSONResponse(
                content={"message": "File processed successfully", "result": result},
                status_code=status.HTTP_200_OK
            )
            
    except Exception as e:
        logger.error(f"Error processing FedEx bill: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        if temp_file_path and Path(temp_file_path).exists():
            Path(temp_file_path).unlink()
            gc.collect()

@app.get("/api/download/{file_path:path}")
async def download_file(file_path: str):
    try:
        file_path = unquote(file_path)
        base_dir = Path(__file__).parent.parent.parent
        processed_dir = base_dir / settings.PROCESSED_DIR
        processed_dir.mkdir(exist_ok=True, parents=True)
        
        file_path = file_path.replace('default/', '')
        full_path = (processed_dir / file_path).resolve()
        
        if not validate_file_path(full_path, processed_dir):
            raise HTTPException(status_code=400, detail="Invalid file path")
            
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
            
        media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        if full_path.suffix.lower() == '.csv':
            media_type = 'text/csv'
            
        return FileResponse(
            path=str(full_path),
            filename=full_path.name,
            media_type=media_type
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

@app.post("/api/compare-shipping-costs")
async def compare_shipping_costs(
    file: UploadFile = File(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500)
):
    temp_file_path = None
    try:
        await validate_upload_file(file)
        temp_file_path = Path(settings.TEMP_DIR) / f"compare_costs_{datetime.now().timestamp()}.xlsx"
        
        async with managed_file_operation():
            # Write uploaded file in chunks
            with open(temp_file_path, "wb") as temp_file:
                while chunk := await file.read(CHUNK_SIZE):
                    temp_file.write(chunk)
                    await asyncio.sleep(0)

            # Process Excel file in chunks
            excel_data = []
            chunk_iterator = pd.read_excel(
                temp_file_path,
                usecols=['Sutijuma Nr', 'Summa', 'Dimensijas'],
                dtype={'Sutijuma Nr': str, 'Dimensijas': str},
                engine='openpyxl',
                chunksize=BATCH_SIZE
            )
            for chunk_df in chunk_iterator:
                excel_data.extend(chunk_df.to_dict('records'))
                gc.collect()

            # Process tracking numbers
            tracking_numbers = [str(row['Sutijuma Nr']).strip() for row in excel_data]
            excel_costs = {str(row['Sutijuma Nr']).strip(): row['Summa'] for row in excel_data}

            # Process in batches
            fedex_data = {}
            printseekers_data = {}
            
            for i in range(0, len(tracking_numbers), BATCH_SIZE):
                batch = tracking_numbers[i:i + BATCH_SIZE]
                
                # FedEx data fetch
                try:
                    result = supabaseDb.from_('fedex_orderi') \
                        .select('masterTrackingNumber,estimatedShippingCosts,serviceType,length,width,height,senderContactName,recipientCountry') \
                        .in_('masterTrackingNumber', batch) \
                        .execute()

                    if hasattr(result, 'data'):
                        for item in result.data:
                            if item['masterTrackingNumber']:
                                fedex_data[item['masterTrackingNumber']] = {
                                    'estimatedShippingCosts': float(item['estimatedShippingCosts']) if item.get('estimatedShippingCosts') else None,
                                    'serviceType': item.get('serviceType', 'N/A'),
                                    'dimensions': f"{item.get('length', 'N/A')} x {item.get('width', 'N/A')} x {item.get('height', 'N/A')}",
                                    'senderContactName': item.get('senderContactName', 'N/A'),
                                    'recipientCountry': item.get('recipientCountry', 'N/A')
                                }
                except Exception as e:
                    logger.error(f"Error fetching fedex_orderi batch: {str(e)}")

                # PrintSeekers data fetch
                try:
                    result = supabaseDb.from_('printseekers_orderi') \
                        .select('TrackingNumber,ProductType,ProductCategory') \
                        .in_('TrackingNumber', batch) \
                        .execute()

                    if hasattr(result, 'data'):
                        for item in result.data:
                            if item['TrackingNumber']:
                                printseekers_data[item['TrackingNumber']] = {
                                    'productType': item.get('ProductType', 'N/A'),
                                    'productCategory': item.get('ProductCategory', 'N/A')
                                }
                except Exception as e:
                    logger.error(f"Error fetching printseekers_orderi batch: {str(e)}")
                
                await asyncio.sleep(PROCESSING_DELAY)
                gc.collect()

            def calculate_difference(excel_cost, db_cost):
                try:
                    if excel_cost is None or db_cost is None:
                        return 'N/A'
                    excel_value = float(str(excel_cost).replace(',', '.').strip())
                    db_value = float(db_cost)
                    difference = round(excel_value - db_value, 2)
                    return difference if abs(difference) >= 0.01 else 'OK'
                except (ValueError, TypeError):
                    return 'N/A'

            # Process comparisons in batches
            comparisons = []
            for i in range(0, len(tracking_numbers), BATCH_SIZE):
                batch = tracking_numbers[i:i + BATCH_SIZE]
                batch_comparisons = []
                
                for tracking_number in batch:
                    fedex_info = fedex_data.get(tracking_number, {})
                    printseekers_info = printseekers_data.get(tracking_number, {})
                    
                    comparison = {
                        'trackingNumber': tracking_number,
                        'excelCost': str(excel_costs.get(tracking_number, 'Invalid value')),
                        'databaseCost': str(fedex_info.get('estimatedShippingCosts', 'Not found')),
                        'costDifference': calculate_difference(
                            excel_costs.get(tracking_number),
                            fedex_info.get('estimatedShippingCosts')
                        ),
                        'serviceType': str(fedex_info.get('serviceType', 'N/A')),
                        'dimensions': str(fedex_info.get('dimensions', 'N/A')),
                        'recipientCountry': str(fedex_info.get('recipientCountry', 'N/A')),
                        'productType': str(printseekers_info.get('productType', 'N/A')),
                        'productCategory': str(printseekers_info.get('productCategory', 'N/A'))
                    }
                    
                    # Only add if there's a difference or data not found
                    if comparison['costDifference'] not in ['OK', 'N/A'] or not fedex_info:
                        batch_comparisons.append(comparison)
                
                comparisons.extend(batch_comparisons)
                gc.collect()

            # Calculate summary statistics
            total_records = len(tracking_numbers)
            matches_found = sum(1 for tn in tracking_numbers if tn in fedex_data)
            cost_differences = sum(1 for c in comparisons if c['costDifference'] not in ['OK', 'N/A'])

            # Sort comparisons
            def sort_key(x):
                diff = x['costDifference']
                if diff == 'N/A':
                    return (1, 0)
                if diff == 'OK':
                    return (0, 0)
                try:
                    return (0, abs(float(diff)))
                except:
                    return (1, 0)

            comparisons.sort(key=sort_key, reverse=True)
            
            # Paginate results
            total_items = len(comparisons)
            total_pages = (total_items + page_size - 1) // page_size
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, total_items)
            
            paginated_data = comparisons[start_idx:end_idx]
            
            response_data = {
                "success": True,
                "data": paginated_data,
                "pagination": {
                    "page": page,
                    "pageSize": page_size,
                    "totalItems": total_items,
                    "totalPages": total_pages
                },
                "summary": {
                    "totalProcessed": total_records,
                    "matchesFound": matches_found,
                    "costDifferences": cost_differences,
                    "notFound": total_records - matches_found
                }
            }
            
            return JSONResponse(
                content=response_data,
                status_code=status.HTTP_200_OK
            )

    except Exception as e:
        logger.error(f"Error in compare_shipping_costs: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()
            gc.collect()

@app.post("/api/files/cleanup")
async def cleanup_files():
    try:
        await storage.cleanup_temp_files(max_age_hours=1)
        return {"status": "success", "message": "Temporary files cleaned up successfully"}
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise StorageError(str(e))

@app.get("/health")
async def health_check():
    """Detailed health check endpoint with memory metrics"""
    try:
        memory_info = {
            "gc_count": gc.get_count(),
            "gc_threshold": gc.get_threshold(),
            "active_requests": getattr(app.state, "active_requests", 0)
        }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "memory_info": memory_info,
            "environment": {
                "python_version": sys.version,
                "cwd": os.getcwd(),
                "path": os.getenv("PATH"),
                "pythonpath": os.getenv("PYTHONPATH"),
                "port": os.getenv("PORT")
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/")
@app.head("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "name": "PrintSeekers Processing API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "process_fedex_bill": "/api/process/fedex-bill",
            "download_file": "/api/download/{file_path}",
            "cleanup_files": "/api/files/cleanup",
            "compare_shipping_costs": "/api/compare-shipping-costs",
            "health": "/health"
        }
    }

# Setup periodic cleanup task with memory management
@app.on_event("startup")
async def setup_periodic_cleanup():
    async def cleanup_task():
        while True:
            try:
                async with managed_file_operation():
                    await storage.cleanup_temp_files(max_age_hours=1)
                await asyncio.sleep(1800)  # Run every 30 minutes
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
                await asyncio.sleep(60)

    asyncio.create_task(cleanup_task())