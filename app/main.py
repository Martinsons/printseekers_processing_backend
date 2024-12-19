import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import asyncio
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
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for file handling
MAX_CONCURRENT_REQUESTS = 4
REQUEST_TIMEOUT = 300  # 5 minutes

app = FastAPI(
    title="Batch Processing API",
    docs_url="/docs",
    redoc_url="/redoc"
)
settings = get_settings()

# Configure logging at the very start
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
    "REQUEST_TIMEOUT": REQUEST_TIMEOUT
}

# Add middleware for request limiting
@app.middleware("http")
async def limit_concurrent_requests(request: Request, call_next):
    app.state.active_requests = getattr(app.state, "active_requests", 0)
    
    if app.state.active_requests >= MAX_CONCURRENT_REQUESTS:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "Server is busy. Please try again later."
            }
        )
    
    app.state.active_requests += 1
    try:
        response = await call_next(request)
        return response
    finally:
        app.state.active_requests -= 1

# Add middleware for request timeout
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error": "Request timeout"
            }
        )

# Add CORS middleware with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Replace with your specific hosts in production
)

# Ensure temp directory exists
os.makedirs(settings.TEMP_DIR, exist_ok=True)

# Add these at the top with your other imports
startup_completed = False

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("=== Application Startup ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"PYTHONPATH: {os.getenv('PYTHONPATH')}")
        logger.info(f"PORT: {os.getenv('PORT')}")
        
        # Initialize necessary directories
        for dir_path in [settings.TEMP_DIR, settings.UPLOAD_DIR, settings.PROCESSED_DIR]:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created/verified directory: {dir_path}")
        
        logger.info("Application initialized")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        raise

async def validate_upload_file(file: UploadFile) -> UploadFile:
    """Validate uploaded file"""
    if not validate_file_type(file.filename, settings.ALLOWED_EXTENSIONS):
        raise InvalidFileError(f"Invalid file type. Allowed types: {settings.ALLOWED_EXTENSIONS}")
    
    content = await file.read()
    if not validate_file_size(len(content), settings.MAX_UPLOAD_SIZE_MB):
        raise InvalidFileError(f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB")
    
    # Reset file pointer
    await file.seek(0)
    return file

@app.post("/api/process/fedex-bill")
async def process_fedex_bill(file: UploadFile = File(...)):
    try:
        # Validate file before processing
        await validate_upload_file(file)
        
        # Create a temporary file to store chunks
        temp_file_path = Path(settings.UPLOAD_DIR) / f"temp_{datetime.now().timestamp()}.pdf"
        
        try:
            # Process file in chunks to avoid memory issues
            with open(temp_file_path, "wb") as temp_file:
                while chunk := await file.read(CHUNK_SIZE):
                    temp_file.write(chunk)
                    await asyncio.sleep(0)  # Allow other tasks to run
            
            # Initialize processor with the temporary file
            processor = FedexBillProcessor(str(temp_file_path))
            
            # Process the file
            result = await handle_file_operation(
                lambda: processor.process(),
                "Error processing FedEx bill"
            )
            
            # Clean up the temporary file
            if temp_file_path.exists():
                temp_file_path.unlink()
            
            return JSONResponse(
                content={
                    "message": "File processed successfully",
                    "result": result
                },
                status_code=status.HTTP_200_OK
            )
            
        except Exception as e:
            # Ensure cleanup in case of error
            if temp_file_path.exists():
                temp_file_path.unlink()
            raise
            
    except Exception as e:
        logger.error(f"Error processing FedEx bill: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/download/{file_path:path}")
async def download_file(file_path: str):
    try:
        # URL decode the file path
        file_path = unquote(file_path)
        
        # Get base directory and construct processed directory path
        base_dir = Path(__file__).parent.parent.parent  # Gets to the root of the project
        processed_dir = base_dir / settings.PROCESSED_DIR
        
        # Create processed directory if it doesn't exist
        processed_dir.mkdir(exist_ok=True, parents=True)
        
        # Clean up the file path and construct full path
        file_path = file_path.replace('default/', '')  # Remove only default/ since it's part of PROCESSED_DIR
        full_path = (processed_dir / file_path).resolve()
        
        logging.info(f"Attempting to download file. Base dir: {base_dir}")
        logging.info(f"Processed dir: {processed_dir}")
        logging.info(f"File path: {file_path}")
        logging.info(f"Full path: {full_path}")

        # Security validation
        if not validate_file_path(full_path, processed_dir):
            logging.error(f"Invalid file path attempted: {file_path}")
            raise HTTPException(status_code=400, detail="Invalid file path")
            
        if not full_path.exists():
            logging.error(f"File not found at path: {full_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
            
        logging.info(f"File found, preparing download: {full_path}")
        
        # Determine media type
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
        logging.error(f"Error downloading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

@app.post("/api/files/cleanup")
async def cleanup_files():
    try:
        await storage.cleanup_temp_files(max_age_hours=1)
        return {"status": "success", "message": "Temporary files cleaned up successfully"}
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")
        raise StorageError(str(e))

@app.post("/api/compare-shipping-costs")
async def compare_shipping_costs(file: UploadFile = File(...)):
    try:
        # Validate file
        await validate_upload_file(file)
        
        # Create a temporary file to store chunks
        temp_file_path = Path(settings.TEMP_DIR) / f"compare_costs_{datetime.now().timestamp()}.xlsx"
        
        try:
            logger.info("Starting file processing")
            # Process file in chunks to avoid memory issues
            with open(temp_file_path, "wb") as temp_file:
                while chunk := await file.read(CHUNK_SIZE):
                    temp_file.write(chunk)
                    await asyncio.sleep(0)  # Allow other tasks to run
            
            logger.info("Reading Excel file")
            # Read Excel file with optimized settings
            df = pd.read_excel(
                temp_file_path,
                usecols=['Sutijuma Nr', 'Summa', 'Dimensijas'],
                dtype={'Sutijuma Nr': str, 'Dimensijas': str},
                engine='openpyxl'
            )
            
            total_rows = len(df)
            logger.info(f"Processing {total_rows} rows from Excel")
            
            # Validate required columns
            required_columns = ['Sutijuma Nr', 'Summa']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Clean tracking numbers
            df['Sutijuma Nr'] = df['Sutijuma Nr'].astype(str).str.strip()
            
            def clean_cost(value):
                try:
                    if isinstance(value, str):
                        value = value.replace('€', '').replace(',', '.').strip()
                    cost = pd.to_numeric(value, errors='coerce')
                    return float(cost) if not pd.isna(cost) else None
                except Exception as e:
                    logger.error(f"Error cleaning cost value '{value}': {str(e)}")
                    return None

            df['Summa'] = df['Summa'].apply(clean_cost)
            tracking_numbers = df['Sutijuma Nr'].tolist()

            # Process in batches
            BATCH_SIZE = 50
            DELAY_BETWEEN_BATCHES = 0.5
            fedex_data = {}
            printseekers_data = {}
            
            total_batches = (len(tracking_numbers) + BATCH_SIZE - 1) // BATCH_SIZE
            processed_batches = 0
            
            logger.info("Fetching FedEx data")
            # Fetch data from fedex_orderi
            for i in range(0, len(tracking_numbers), BATCH_SIZE):
                batch = tracking_numbers[i:i + BATCH_SIZE]
                processed_batches += 1
                logger.info(f"Processing FedEx batch {processed_batches}/{total_batches}")
                
                try:
                    result = supabaseDb.from_('fedex_orderi') \
                        .select('''
                            masterTrackingNumber,
                            estimatedShippingCosts,
                            serviceType,
                            length,
                            width,
                            height,
                            senderContactName,
                            recipientCountry    
                        ''') \
                        .in_('masterTrackingNumber', batch) \
                        .execute()

                    if hasattr(result, 'data') and result.data:
                        for item in result.data:
                            if item['masterTrackingNumber']:
                                dimensions = f"{item.get('length', 'N/A')} x {item.get('width', 'N/A')} x {item.get('height', 'N/A')}"
                                fedex_data[item['masterTrackingNumber']] = {
                                    'estimatedShippingCosts': float(item['estimatedShippingCosts']) if item.get('estimatedShippingCosts') else None,
                                    'serviceType': item.get('serviceType', 'N/A'),
                                    'dimensions': dimensions,
                                    'senderContactName': item.get('senderContactName', 'N/A'),
                                    'recipientCountry': item.get('recipientCountry', 'N/A')
                                }
                except Exception as e:
                    logger.error(f"Error fetching fedex_orderi batch: {str(e)}")
                    continue
                
                await asyncio.sleep(DELAY_BETWEEN_BATCHES)

            logger.info("Fetching PrintSeekers data")
            processed_batches = 0
            # Fetch data from printseekers_orderi
            for i in range(0, len(tracking_numbers), BATCH_SIZE):
                batch = tracking_numbers[i:i + BATCH_SIZE]
                processed_batches += 1
                logger.info(f"Processing PrintSeekers batch {processed_batches}/{total_batches}")
                
                try:
                    result = supabaseDb.from_('printseekers_orderi') \
                        .select('TrackingNumber, ProductType, ProductCategory') \
                        .in_('TrackingNumber', batch) \
                        .execute()

                    if hasattr(result, 'data') and result.data:
                        for item in result.data:
                            if item['TrackingNumber']:
                                printseekers_data[item['TrackingNumber']] = {
                                    'productType': item.get('ProductType', 'N/A'),
                                    'productCategory': item.get('ProductCategory', 'N/A')
                                }
                except Exception as e:
                    logger.error(f"Error fetching printseekers_orderi batch: {str(e)}")
                    continue
                
                await asyncio.sleep(DELAY_BETWEEN_BATCHES)

            logger.info("Preparing comparison data")
            # Prepare comparison data
            comparisons = []
            matches_found = 0
            cost_differences = 0
            
            for _, row in df.iterrows():
                tracking_number = row['Sutijuma Nr']
                excel_cost = row['Summa']
                fedex_info = fedex_data.get(tracking_number, {})
                printseekers_info = printseekers_data.get(tracking_number, {})
                
                if fedex_info:
                    matches_found += 1
                
                def calculate_difference(excel_cost, db_cost):
                    try:
                        if excel_cost is None or db_cost is None:
                            return 'N/A'
                        excel_value = float(excel_cost)
                        db_value = float(db_cost)
                        difference = excel_value - db_value
                        if abs(difference) >= 0.01:
                            nonlocal cost_differences
                            cost_differences += 1
                        return 'OK' if abs(difference) < 0.01 else difference
                    except (ValueError, TypeError):
                        return 'N/A'

                comparison = {
                    'trackingNumber': tracking_number,
                    'excelCost': excel_cost if excel_cost is not None else 'Invalid value',
                    'databaseCost': fedex_info.get('estimatedShippingCosts', 'Not found'),
                    'costDifference': calculate_difference(
                        excel_cost,
                        fedex_info.get('estimatedShippingCosts')
                    ),
                    'serviceType': fedex_info.get('serviceType', 'N/A'),
                    'excelDimensions': str(row.get('Dimensijas', '')).strip() if not pd.isna(row.get('Dimensijas')) else 'N/A',
                    'dimensions': fedex_info.get('dimensions', 'N/A'),
                    'recipientCountry': fedex_info.get('recipientCountry', 'N/A'),
                    'productType': printseekers_info.get('productType', 'N/A'),
                    'productCategory': printseekers_info.get('productCategory', 'N/A')
                }
                comparisons.append(comparison)

            logger.info("Comparison completed")
            return JSONResponse(
                content={
                    "success": True,
                    "data": comparisons,
                    "summary": {
                        "totalProcessed": total_rows,
                        "matchesFound": matches_found,
                        "costDifferences": cost_differences,
                        "notFound": total_rows - matches_found
                    }
                },
                status_code=status.HTTP_200_OK
            )

        except Exception as e:
            logger.error(f"Error processing comparison: {str(e)}", exc_info=True)
            return JSONResponse(
                content={
                    "success": False,
                    "error": str(e)
                },
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        finally:
            # Clean up temporary file
            if temp_file_path.exists():
                temp_file_path.unlink()
                logger.info("Temporary file cleaned up")
                
    except Exception as e:
        logger.error(f"Error in compare_shipping_costs: {str(e)}", exc_info=True)
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    logger.info("Health check endpoint called")
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
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

# Cleanup scheduler
@app.on_event("startup")
async def setup_periodic_cleanup():
    async def cleanup_task():
        while True:
            try:
                await storage.cleanup_temp_files(max_age_hours=1)
                await asyncio.sleep(1800)  # Run every 30 minutes
            except Exception as e:
                logging.error(f"Error in cleanup task: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    asyncio.create_task(cleanup_task())