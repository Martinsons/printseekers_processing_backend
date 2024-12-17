import os
import sys
import logging
from pathlib import Path


# Configure logging at the very start
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Log environment information immediately
logger.info("=== Application Initialization ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"PYTHONPATH: {os.getenv('PYTHONPATH')}")
logger.info(f"PORT: {os.getenv('PORT')}")
logger.info(f"Directory contents: {os.listdir()}")

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

TEMP_DIR = os.path.join("backend", "temp")

app = FastAPI(title="Batch Processing API")
settings = get_settings()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add these at the top with your other imports
startup_completed = False

@app.on_event("startup")
async def startup_event():
    """Application startup handler with detailed logging"""
    try:
        logger.info("=== Application Startup ===")
        
        # Check environment variables
        required_vars = ['SUPABASE_URL', 'SUPABASE_KEY']
        for var in required_vars:
            value = os.getenv(var)
            logger.info(f"{var} is {'set' if value else 'NOT SET'}")

        # Check directories
        dirs_to_check = [
            Path("processed_files/default"),
            Path("temp")
        ]
        
        for dir_path in dirs_to_check:
            logger.info(f"Checking directory: {dir_path}")
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"  Created/verified directory: {dir_path}")
                logger.info(f"  Exists: {dir_path.exists()}")
                logger.info(f"  Is directory: {dir_path.is_dir()}")
                logger.info(f"  Permissions: {oct(dir_path.stat().st_mode)[-3:]}")
            except Exception as e:
                logger.error(f"  Error creating directory {dir_path}: {str(e)}")

        global startup_completed
        startup_completed = True
        logger.info("Application startup completed successfully")
        
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
        # Validate file
        file = await validate_upload_file(file)
        
        # Save file temporarily
        file_path = await storage.save_temp_file(file)
        
        try:
            # Process the file
            processor = FedexBillProcessor(str(file_path))
            result = processor.process()
            
            if result.get("error"):
                raise FileProcessingError(result["error"])
                
            return result
            
        finally:
            # Always clean up the temporary file
            storage.remove_temp_file(file_path)
            
    except Exception as e:
        logging.error(f"Error processing FedEx bill: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        file = await validate_upload_file(file)
        
        # Save file
        file_path = await storage.save_temp_file(file)
        
        try:
            # Read Excel file
            df = pd.read_excel(file_path)
            
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
                    logging.error(f"Error cleaning cost value '{value}': {str(e)}")
                    return None

            df['Summa'] = df['Summa'].apply(clean_cost)
            tracking_numbers = df['Sutijuma Nr'].tolist()

            try:
                # Process in batches
                BATCH_SIZE = 50
                DELAY_BETWEEN_BATCHES = 0.5
                fedex_data = {}
                printseekers_data = {}
                
                # Fetch data from fedex_orderi
                for i in range(0, len(tracking_numbers), BATCH_SIZE):
                    batch = tracking_numbers[i:i + BATCH_SIZE]
                    
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
                    
                    await asyncio.sleep(DELAY_BETWEEN_BATCHES)

                # Fetch data from printseekers_orderi
                for i in range(0, len(tracking_numbers), BATCH_SIZE):
                    batch = tracking_numbers[i:i + BATCH_SIZE]
                    
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
                    
                    await asyncio.sleep(DELAY_BETWEEN_BATCHES)

                # Prepare comparison data
                comparisons = []
                for _, row in df.iterrows():
                    tracking_number = row['Sutijuma Nr']
                    excel_cost = row['Summa']
                    fedex_info = fedex_data.get(tracking_number, {})
                    printseekers_info = printseekers_data.get(tracking_number, {})
                    
                    def calculate_difference(excel_cost, db_cost):
                        try:
                            excel_value = float(excel_cost)
                            db_value = float(db_cost)
                            difference = excel_value - db_value
                            
                            # Return 'OK' if difference is effectively zero
                            if abs(difference) < 0.01:
                                return 'OK'
                            
                            return difference
                        except (ValueError, TypeError):
                            return 'N/A'

                    comparison = {
                        'trackingNumber': tracking_number,
                        'excelCost': excel_cost if excel_cost is not None else 'Invalid value',
                        'databaseCost': fedex_info.get('estimatedShippingCosts', 'Not found'),
                        'costDifference': calculate_difference(
                            excel_cost if excel_cost is not None else None,
                            fedex_info.get('estimatedShippingCosts')
                        ),
                        'serviceType': fedex_info.get('serviceType', 'N/A'),
                        'excelDimensions': str(row.get('Dimensijas', '')).strip() if not pd.isna(row.get('Dimensijas')) else 'N/A',
                        'dimensions': fedex_info.get('dimensions', 'N/A'),
                        'recipientCountry': fedex_info.get('recipientCountry', 'N/A'),
                        'senderContactName': fedex_info.get('senderContactName', 'N/A'),
                        'productType': printseekers_info.get('productType', 'N/A'),
                        'productCategory': printseekers_info.get('productCategory', 'N/A'),
                        'recipient': str(row.get('Sanemejs', '')) if not pd.isna(row.get('Sanemejs')) else '',
                        'serviceData': str(row.get('Servisa dati', '')) if not pd.isna(row.get('Servisa dati')) else '',
                        'deliveryZone': str(row.get('Piegades zona', '')) if not pd.isna(row.get('Piegades zona')) else '',
                        'invoice': str(row.get('Invoice', '')) if not pd.isna(row.get('Invoice')) else ''
                    }
                    comparisons.append(comparison)

                return {
                    "status": "success",
                    "comparisons": comparisons,
                    "summary": {
                        "total_processed": len(comparisons),
                        "matches_found": len(fedex_data),
                        "not_found": len(comparisons) - len(fedex_data)
                    }
                }
                
            except Exception as e:
                logging.error(f"Database error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
            
        finally:
            if file_path.exists():
                file_path.unlink()
                
    except Exception as e:
        logging.error(f"Error comparing shipping costs: {str(e)}")
        raise FileProcessingError(str(e))

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