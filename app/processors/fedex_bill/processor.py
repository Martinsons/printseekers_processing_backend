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

# Set memory limit (450MB)
resource.setrlimit(resource.RLIMIT_AS, (450 * 1024 * 1024, -1))

# Configure garbage collection
gc.set_threshold(100, 5, 5)

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
    CHUNK_SIZE = 10  # Number of pages to process at once
    
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
        """
        Process a single table row efficiently
        Args:
            table (list): Table data
            invoice_number (Optional[str]): Invoice number
        """
        try:
            # Basic validation
            if len(table) <= 3 or len(table[0]) <= 8:
                return

            # Extract tracking number and dimensions
            sutijuma_data = table[1][0]
            tracking_number = ''.join(filter(str.isdigit, sutijuma_data.split()[0]))
            
            dimensijas_index = sutijuma_data.find("Dimensijas")
            dimensions = (sutijuma_data[dimensijas_index + len("Dimensijas"):].strip() 
                        if dimensijas_index != -1 else "")

            # Process recipient data
            sanemejs_data = table[2][3].split()
            country = sanemejs_data[-1] if sanemejs_data else ""
            recipient = ' '.join(sanemejs_data[1:4] if len(sanemejs_data) > 3 else sanemejs_data[1:])
            recipient = ''.join(c for c in recipient if not c.isdigit()).strip()

            # Process service data
            service_data = table[1][3].replace('Aprēķinātais svars', '').replace('kg', '').strip()
            service_data = ''.join(c for c in service_data if not c.isdigit()).strip().replace(',', '')

            # Process amount
            amount_str = table[3][8].replace('Kopā EUR', '').strip()
            amount = self._clean_amount(amount_str)

            # Process delivery zone
            delivery_zone = ""
            zone_text = table[2][4]
            if "Attālinātā piegādes zona" in zone_text:
                delivery_zone = zone_text.split("Attālinātā piegādes zona", 1)[1].strip().split('\n')[0].strip()

            # Append all data
            self.data["Invoice"].append(invoice_number or "Not found")
            self.data["Sutijuma nr un dimensijas"].append(tracking_number)
            self.data["Dimensijas"].append(dimensions)
            self.data["Valsts"].append(country)
            self.data["Sanemejs dati"].append(recipient)
            self.data["Servisa dati"].append(service_data)
            self.data["Summa"].append(str(amount))
            self.data["Piegādes zona"].append(delivery_zone)

        except Exception as e:
            logging.error(f"Error processing table row: {str(e)}")
            raise

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
        
        # Use optimal Excel writing settings
        writer = pd.ExcelWriter(output_path, engine='openpyxl')
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.save()
        writer.close()
        
        return str(output_path.relative_to(self.processed_dir))

    def _prepare_result_data(self) -> list:
        """
        Prepare result data efficiently
        Returns:
            list: List of result dictionaries
        """
        return [
            {
                "invoice": inv,
                "trackingNumber": track,
                "recipientData": recip,
                "serviceData": serv,
                "amount": amt,
                "deliveryZone": zone,
                "dimensions": dim,
                "country": country
            }
            for inv, track, recip, serv, amt, zone, dim, country in zip(
                self.data["Invoice"],
                self.data["Sutijuma nr un dimensijas"],
                self.data["Sanemejs dati"],
                self.data["Servisa dati"],
                self.data["Summa"],
                self.data["Piegādes zona"],
                self.data["Dimensijas"],
                self.data["Valsts"]
            )
        ]

    def process(self) -> Dict[str, Any]:
        """
        Process the FedEx bill PDF with optimal memory usage
        Returns:
            Dict[str, Any]: Processing results
        """
        try:
            self.validate_data()
            
            with pdfplumber.open(self.input_file) as pdf:
                # Get invoice number from first page
                first_page = pdf.pages[0]
                invoice_number = self._extract_invoice_number(first_page.extract_text())
                first_page.flush_cache()

                # Process pages in chunks
                total_pages = len(pdf.pages)
                for i in range(0, total_pages, self.CHUNK_SIZE):
                    chunk_pages = pdf.pages[i:i + self.CHUNK_SIZE]
                    
                    for page in chunk_pages:
                        tables = page.extract_tables()
                        for table in tables:
                            if len(table) > 3 and len(table[0]) > 8:
                                self._process_table_row(table, invoice_number)
                        page.flush_cache()
                    
                    # Force garbage collection after each chunk
                    gc.collect()

            if not any(self.data.values()):
                raise ValueError("No valid data found in PDF")

            # Apply VAT and prepare data
            self._apply_vat()

            # Create and process DataFrame
            df = pd.DataFrame(self.data)
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

            # Save to Excel and prepare response
            relative_path = self._save_to_excel(df)
            self.processed_files['analysis_file'] = relative_path

            return {
                "status": "success",
                "files": self.processed_files,
                "data": self._prepare_result_data()
            }

        except Exception as e:
            logging.error(f"Error processing FedEx bill: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
        finally:
            # Clean up
            self._initialize_data()
            gc.collect()