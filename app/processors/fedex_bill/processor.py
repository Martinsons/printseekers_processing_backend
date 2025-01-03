from ..base_processor import BaseProcessor
import pdfplumber
import pandas as pd
from datetime import datetime
from pathlib import Path
import os
import logging
import re
from typing import Dict, Any, Optional
import gc
import resource
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set memory limit (1GB)
resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, -1))

# Configure garbage collection for more aggressive collection
gc.set_threshold(50, 3, 3)

# European Union country codes (excluding Norway, Switzerland, and Gibraltar)
EU_COUNTRIES = {
    'AUSTRIA', 'BELGIUM', 'BULGARIA', 'CROATIA', 'CYPRUS', 'CZECH REPUBLIC', 
    'DENMARK', 'ESTONIA', 'FINLAND', 'FRANCE', 'GERMANY', 'GREECE', 'HUNGARY', 
    'IRELAND', 'ITALY', 'LATVIA', 'LITHUANIA', 'LUXEMBOURG', 'MALTA', 'NETHERLANDS', 
    'POLAND', 'PORTUGAL', 'ROMANIA', 'SLOVAKIA', 'SLOVENIA', 'SPAIN', 'SWEDEN'
}

# Other European countries (non-EU but still in Europe)
OTHER_EUROPEAN_COUNTRIES = {
    'ALBANIA', 'ANDORRA', 'BELARUS', 'BOSNIA AND HERZEGOVINA', 'GIBRALTAR', 'ICELAND', 
    'LIECHTENSTEIN', 'MOLDOVA', 'MONACO', 'MONTENEGRO', 'NORTH MACEDONIA', 
    'NORWAY', 'SAN MARINO', 'SERBIA', 'SWITZERLAND', 'UKRAINE', 'UNITED KINGDOM', 
    'VATICAN CITY'
}

# Combined set of all European countries
EUROPEAN_COUNTRIES = EU_COUNTRIES | OTHER_EUROPEAN_COUNTRIES

class FedexBillProcessor(BaseProcessor):
    CHUNK_SIZE = 5  # Number of pages to process at once
    PAGE_TIMEOUT = 30  # Timeout in seconds for processing a single page
    TEST_MODE = True  # Set to True to process only first few pages
    TEST_PAGES = 3  # Number of pages to process in test mode
    
    def __init__(self, input_file: str, user_id: str = "default"):
        super().__init__(input_file, user_id)
        self.today_date = datetime.now().strftime("%d-%m-%Y")
        self.timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        self.processed_files = {}
        self._initialize_data()

    def _initialize_data(self) -> None:
        """Initialize data structure with minimal memory footprint"""
        self.data = {
            "Invoice": [],
            "Sutijuma nr un dimensijas": [],
            "Sanemejs dati": [],
            "Servisa dati": [],
            "Summa": [],
            "Piegādes zona": [],
            "Dimensijas": [],
            "Valsts": []
        }

    def _load_file(self) -> None:
        """Validate file format"""
        if not str(self.input_file).lower().endswith('.pdf'):
            raise ValueError("Invalid file format. Please upload a PDF file.")

    def validate_data(self) -> bool:
        """
        Validate PDF content with minimal memory usage
        Returns:
            bool: True if valid, raises ValueError if invalid
        """
        try:
            with pdfplumber.open(self.input_file) as pdf:
                first_page = pdf.pages[0]
                text = first_page.extract_text()
                if not text or 'FedEx Express Latvia SIA' not in text:
                    raise ValueError("This does not appear to be a valid FedEx bill")
                first_page.flush_cache()
                del text
                gc.collect()
            return True
        except Exception as e:
            logging.error(f"Validation error: {str(e)}")
            raise ValueError(f"Invalid FedEx bill format: {str(e)}")

    def _extract_invoice_number(self, text: str) -> Optional[str]:
        """
        Extract invoice number efficiently
        Args:
            text (str): The text content of the PDF page
        Returns:
            Optional[str]: The invoice number if found, None otherwise
        """
        try:
            match = re.search(r"Rēķina numurs:\s*(\d+)", text)
            return match.group(1) if match else None
        except Exception as e:
            logging.error(f"Error extracting invoice number: {str(e)}")
            return None

    def _clean_amount(self, amount_str: str) -> float:
        """
        Clean and convert amount string to float
        Args:
            amount_str (str): Amount string to clean
        Returns:
            float: Cleaned amount
        """
        try:
            # Remove spaces and convert European format to standard
            cleaned = amount_str.replace(' ', '').replace('.', '').replace(',', '.')
            amount = float(cleaned)
            return amount if not pd.isna(amount) else 0.0
        except (ValueError, AttributeError):
            return 0.0

    def _process_table_row(self, table: list, invoice_number: Optional[str]) -> None:
        """Process a single table row efficiently"""
        if not table:
            return

        try:
            # Log table structure for debugging
            logger.debug(f"Processing table with {len(table)} rows")
            for row_idx, row in enumerate(table):
                logger.debug(f"Row {row_idx}: {row}")

            # Skip header rows
            if len(table) <= 2:  # Skip if table is too small
                return

            # Process each row after the header
            for row in table[2:]:  # Skip first two rows (headers)
                if not row or not any(cell and str(cell).strip() for cell in row):
                    continue

                try:
                    # Extract tracking number and dimensions
                    tracking_cell = str(row[0]).strip() if row[0] else ""
                    if not tracking_cell or tracking_cell.lower() in ['tracking number', 'sutijuma nr']:
                        continue

                    # Extract data with proper error handling
                    tracking = tracking_cell
                    recipient = str(row[1]).strip() if len(row) > 1 and row[1] else ""
                    service = str(row[2]).strip() if len(row) > 2 and row[2] else ""
                    
                    # Handle amount - it might be in different columns
                    amount = 0.0
                    for col_idx in [3, 4, 5]:  # Check multiple possible columns for amount
                        if len(row) > col_idx and row[col_idx]:
                            try:
                                amount = self._clean_amount(str(row[col_idx]))
                                if amount > 0:
                                    break
                            except:
                                continue

                    # Extract remaining data
                    zone = str(row[4]).strip() if len(row) > 4 and row[4] else ""
                    dimensions = str(row[5]).strip() if len(row) > 5 and row[5] else ""
                    country = str(row[6]).strip() if len(row) > 6 and row[6] else ""

                    # Only add if we have essential data
                    if tracking and (recipient or amount > 0):
                        self.data["Invoice"].append(invoice_number)
                        self.data["Sutijuma nr un dimensijas"].append(tracking)
                        self.data["Sanemejs dati"].append(recipient)
                        self.data["Servisa dati"].append(service)
                        self.data["Summa"].append(amount)
                        self.data["Piegādes zona"].append(zone)
                        self.data["Dimensijas"].append(dimensions)
                        self.data["Valsts"].append(country)
                        logger.debug(f"Added row data: {tracking} | {recipient} | {amount}")

                except Exception as e:
                    logger.error(f"Error processing row in table: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error processing table: {str(e)}")

    def _apply_vat(self) -> None:
        """Apply VAT for EU countries efficiently"""
        for i, (amount, country) in enumerate(zip(self.data["Summa"], self.data["Valsts"])):
            try:
                amt = float(amount)
                if country.upper() in EU_COUNTRIES:
                    amt = round(amt * 1.21, 2)
                self.data["Summa"][i] = str(amt)
            except (ValueError, AttributeError):
                self.data["Summa"][i] = "0.0"

    def _save_to_excel(self, df: pd.DataFrame) -> str:
        """
        Save DataFrame to Excel efficiently
        Args:
            df (pd.DataFrame): DataFrame to save
        Returns:
            str: Relative path to saved file
        """
        output_filename = f'fedex_bill_analysis_{self.timestamp}.xlsx'
        output_path = self.output_dir / output_filename
        
        # Use context manager for proper resource handling
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        
        return str(output_path.relative_to(self.processed_dir))

    def _prepare_result_data(self) -> list:
        """
        Prepare result data efficiently
        Returns:
            list: List of result dictionaries
        """
        result_data = []
        for i in range(len(self.data["Invoice"])):
            try:
                row_data = {
                    "invoice": self.data["Invoice"][i],
                    "trackingNumber": self.data["Sutijuma nr un dimensijas"][i],
                    "recipientData": self.data["Sanemejs dati"][i],
                    "serviceData": self.data["Servisa dati"][i],
                    "amount": str(self.data["Summa"][i]),  # Convert to string for consistency
                    "deliveryZone": self.data["Piegādes zona"][i],
                    "dimensions": self.data["Dimensijas"][i],
                    "country": self.data["Valsts"][i]
                }
                result_data.append(row_data)
            except Exception as e:
                logger.error(f"Error preparing row {i} data: {str(e)}")
                continue
        
        logger.info(f"Prepared {len(result_data)} rows of data for return")
        return result_data

    def process(self) -> Dict[str, Any]:
        """
        Process the FedEx bill PDF with optimal memory usage
        Returns:
            Dict[str, Any]: Processing results
        """
        try:
            logger.info("Starting FedEx bill processing")
            start_time = time.time()
            
            logger.info("Validating data...")
            self.validate_data()
            
            with pdfplumber.open(self.input_file) as pdf:
                logger.info(f"Opened PDF with {len(pdf.pages)} pages")
                
                logger.info("Extracting invoice number from first page")
                first_page = pdf.pages[0]
                invoice_number = self._extract_invoice_number(first_page.extract_text())
                logger.info(f"Found invoice number: {invoice_number}")
                first_page.flush_cache()
                del first_page

                total_pages = len(pdf.pages)
                if self.TEST_MODE:
                    total_pages = min(total_pages, self.TEST_PAGES)
                    logger.info(f"TEST MODE: Processing only first {total_pages} pages")
                
                logger.info(f"Processing {total_pages} pages in chunks of {self.CHUNK_SIZE}")
                
                for i in range(0, total_pages, self.CHUNK_SIZE):
                    chunk_start = time.time()
                    logger.info(f"Processing chunk {i//self.CHUNK_SIZE + 1}/{(total_pages + self.CHUNK_SIZE - 1)//self.CHUNK_SIZE}")
                    chunk_pages = pdf.pages[i:i + self.CHUNK_SIZE]
                    
                    for page_num, page in enumerate(chunk_pages, start=i):
                        page_start = time.time()
                        try:
                            if time.time() - page_start > self.PAGE_TIMEOUT:
                                logger.warning(f"Timeout processing page {page_num + 1}")
                                continue
                                
                            logger.info(f"Processing page {page_num + 1}/{total_pages}")
                            tables = page.extract_tables()
                            
                            if not tables:
                                logger.info(f"No tables found on page {page_num + 1}")
                                continue
                                
                            logger.info(f"Found {len(tables)} tables on page {page_num + 1}")
                            for table_num, table in enumerate(tables, 1):
                                logger.debug(f"Table {table_num} structure: {len(table)} rows")
                                if table and len(table) > 2:  
                                    self._process_table_row(table, invoice_number)
                                    logger.debug(f"Processed table {table_num} on page {page_num + 1}")
                                del table
                            del tables
                        except Exception as e:
                            logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                            continue
                        finally:
                            page.flush_cache()
                            del page
                            
                        if time.time() - page_start > self.PAGE_TIMEOUT:
                            logger.warning(f"Page {page_num + 1} processing exceeded timeout")
                    
                    chunk_time = time.time() - chunk_start
                    logger.info(f"Chunk processed in {chunk_time:.2f} seconds")
                    gc.collect()

            row_count = len(self.data["Invoice"])
            if row_count == 0:
                logger.error("No valid data found in PDF")
                raise ValueError("No valid data found in PDF")
            else:
                logger.info(f"Successfully processed {row_count} rows of data")

            logger.info("Applying VAT calculations")
            self._apply_vat()

            # Create DataFrame in chunks to minimize memory usage
            logger.info("Creating DataFrame from processed data")
            chunk_size = 1000
            dfs = []
            data_length = len(self.data["Invoice"])
            logger.info(f"Total rows to process: {data_length}")
            
            for i in range(0, data_length, chunk_size):
                logger.info(f"Processing DataFrame chunk {i//chunk_size + 1}/{(data_length + chunk_size - 1)//chunk_size}")
                chunk_data = {
                    k: v[i:i + chunk_size] 
                    for k, v in self.data.items()
                }
                df_chunk = pd.DataFrame(chunk_data)
                dfs.append(df_chunk)
                del chunk_data
                gc.collect()

            logger.info("Concatenating DataFrame chunks")
            df = pd.concat(dfs, ignore_index=True)
            del dfs
            gc.collect()

            logger.info("Renaming DataFrame columns")
            df = df.rename(columns={
                "Invoice": "Invoice",
                "Sutijuma nr un dimensijas": "Sutijuma Nr",
                "Sanemejs dati": "Sanemejs",
                "Servisa dati": "Servisa dati",
                "Summa": "Summa",
                "Piegādes zona": "Piegades zona",
                "Dimensijas": "Dimensijas",
                "Valsts": "Valsts"
            })

            logger.info("Saving to Excel")
            relative_path = self._save_to_excel(df)
            del df
            gc.collect()
            
            self.processed_files['analysis_file'] = relative_path
            
            # Prepare the return data
            processed_data = self._prepare_result_data()
            if not processed_data:
                logger.error("No data was processed successfully")
                return {
                    "status": "error",
                    "error": "No data could be extracted from the PDF"
                }
            
            total_time = time.time() - start_time
            logger.info(f"Processing completed in {total_time:.2f} seconds")
            logger.info(f"Returning {len(processed_data)} rows of data")

            return {
                "status": "success",
                "files": self.processed_files,
                "data": processed_data,
                "summary": {
                    "total_rows": len(processed_data),
                    "processing_time": f"{total_time:.2f} seconds"
                }
            }

        except Exception as e:
            logger.error(f"Error processing FedEx bill: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
        finally:
            logger.info("Cleaning up resources")
            self._initialize_data()
            gc.collect()