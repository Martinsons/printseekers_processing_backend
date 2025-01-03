from ..base_processor import BaseProcessor
import pdfplumber
import pandas as pd
from datetime import datetime
from pathlib import Path
import os
import logging
import re
from typing import Dict, Any, Optional

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
    def __init__(self, input_file: str, user_id: str = "default"):
        super().__init__(input_file, user_id)
        self.today_date = datetime.now().strftime("%d-%m-%Y")
        self.timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        self.processed_files = {}
        
        # The output_dir is already set by the parent class
        self.temp_dir = self.temp_dir
        self.temp_dir.mkdir(exist_ok=True)

    def _load_file(self) -> None:
        """Override parent's _load_file method since we're dealing with PDFs"""
        if not str(self.input_file).lower().endswith('.pdf'):
            raise ValueError("Invalid file format. Please upload a PDF file.")
        return None

    def validate_data(self) -> bool:
        """
        Override abstract method to validate the PDF file
        Returns:
            bool: True if valid, raises ValueError if invalid
        """
        try:
            with pdfplumber.open(self.input_file) as pdf:
                first_page = pdf.pages[0]
                text = first_page.extract_text()
                if not text or 'FedEx Express Latvia SIA' not in text:
                    raise ValueError("This does not appear to be a valid FedEx bill")
            return True
        except Exception as e:
            logging.error(f"Validation error: {str(e)}")
            raise ValueError("Invalid FedEx bill format")

    def _extract_invoice_number(self, text: str) -> Optional[str]:
        """
        Extract invoice number from the PDF text.
        Args:
            text (str): The text content of the PDF page
        Returns:
            Optional[str]: The invoice number if found, None otherwise
        """
        try:
            # Look for "Rēķina numurs:" followed by numbers
            match = re.search(r"Rēķina numurs:\s*(\d+)", text)
            if match:
                return match.group(1)
            return None
        except Exception as e:
            logging.error(f"Error extracting invoice number: {str(e)}")
            return None

    def process(self) -> Dict[str, Any]:
        """
        Override abstract method to process the FedEx bill PDF
        Returns:
            Dict[str, Any]: Processing results with status and data
        """
        try:
            self.validate_data()

            data = {
                "Invoice": [],
                "Sutijuma nr un dimensijas": [],
                "Sanemejs dati": [],
                "Servisa dati": [],
                "Summa": [],
                "Piegādes zona": [],
                "Dimensijas": [],
                "Valsts": []
            }

            with pdfplumber.open(self.input_file) as pdf:
                # Extract invoice number from first page
                first_page = pdf.pages[0]
                first_page_text = first_page.extract_text()
                invoice_number = self._extract_invoice_number(first_page_text)
                
                if not invoice_number:
                    logging.warning("Could not find invoice number in the PDF")
                
                for page in pdf.pages:
                    tables = page.extract_tables()

                    for table in tables:
                        if len(table) > 3 and len(table[0]) > 8:
                            try:
                                # Add invoice number to each row
                                data["Invoice"].append(invoice_number or "Not found")
                                
                                sutijuma_nr_un_dimensijas_value = table[1][0]
                                
                                # Extract tracking number
                                sutijuma_nr = ''.join(filter(str.isdigit, sutijuma_nr_un_dimensijas_value.split()[0]))
                                data["Sutijuma nr un dimensijas"].append(sutijuma_nr)
                                
                                # Extract dimensions
                                dimensijas_index = sutijuma_nr_un_dimensijas_value.find("Dimensijas")
                                if dimensijas_index != -1:
                                    dimensijas_value = sutijuma_nr_un_dimensijas_value[dimensijas_index + len("Dimensijas"):].strip()
                                    data["Dimensijas"].append(dimensijas_value)
                                else:
                                    data["Dimensijas"].append("")

                                # Extract recipient data and country
                                sanemejs_dati_value = table[2][3]
                                sanemejs_dati_words = sanemejs_dati_value.split()
                                
                                valsts_value = sanemejs_dati_words[-1] if sanemejs_dati_words else ""
                                data["Valsts"].append(valsts_value)
                                
                                if len(sanemejs_dati_words) > 3:
                                    sanemejs_dati_value = ' '.join(sanemejs_dati_words[1:4])
                                else:
                                    sanemejs_dati_value = ' '.join(sanemejs_dati_words[1:])
                                sanemejs_dati_value = ''.join(filter(lambda x: not x.isdigit(), sanemejs_dati_value)).strip()
                                data["Sanemejs dati"].append(sanemejs_dati_value)

                                # Extract service data
                                servisa_dati_value = table[1][3].replace('Aprēķinātais svars', '').replace('kg', '').strip()
                                servisa_dati_value = ''.join(filter(lambda x: not x.isdigit(), servisa_dati_value)).strip()
                                servisa_dati_value = servisa_dati_value.replace(',', '')
                                data["Servisa dati"].append(servisa_dati_value)

                                # Extract amount
                                summa_value = table[3][8].replace('Kopā EUR', '').strip()
                                data["Summa"].append(summa_value)

                                # Extract delivery zone
                                if "Attālinātā piegādes zona" in table[2][4]:
                                    piegades_zona_value = table[2][4].split("Attālinātā piegādes zona", 1)[1].strip().split('\n')[0].strip()
                                    data["Piegādes zona"].append(piegades_zona_value)
                                else:
                                    data["Piegādes zona"].append("")

                            except Exception as e:
                                logging.error(f"Error processing table row: {str(e)}")
                                continue

            if not any(data.values()):
                raise ValueError("No valid data found in PDF")

            # Apply VAT for EU countries before creating DataFrame
            for i in range(len(data["Summa"])):
                amount = data["Summa"][i]
                try:
                    amount = float(amount.replace(',', '.').replace(' ', ''))
                    country = data["Valsts"][i].upper()
                    if country in EU_COUNTRIES:
                        amount = round(amount * 1.21, 2)
                    data["Summa"][i] = str(amount)
                except (ValueError, AttributeError):
                    continue

            # Create DataFrame and save to Excel
            df = pd.DataFrame(data)

            # Define new column names
            column_mapping = {
                "Invoice": "Invoice",
                "Sutijuma nr un dimensijas": "Sutijuma Nr",
                "Sanemejs dati": "Sanemejs",
                "Servisa dati": "Servisa dati",
                "Summa": "Summa",
                "Piegādes zona": "Piegades zona",
                "Dimensijas": "Dimensijas",
                "Valsts": "Valsts"
            }

            # Rename the columns
            df = df.rename(columns=column_mapping)
            
            output_filename = f'fedex_bill_analysis_{self.timestamp}.xlsx'
            output_path = self.output_dir / output_filename
            
            # Save Excel file with index=False
            df.to_excel(output_path, index=False, engine='openpyxl')
            
            # Store relative path for the API response
            relative_path = output_path.relative_to(self.processed_dir)
            self.processed_files['analysis_file'] = str(relative_path)

            # Prepare response data with invoice number
            result_data = []
            for i in range(len(data["Sutijuma nr un dimensijas"])):
                # Get the amount and convert to float
                amount = data["Summa"][i]
                try:
                    amount = float(amount.replace(',', '.').replace(' ', ''))
                except (ValueError, AttributeError):
                    amount = 0.0
                
                result_data.append({
                    "invoice": data["Invoice"][i],
                    "trackingNumber": data["Sutijuma nr un dimensijas"][i],
                    "recipientData": data["Sanemejs dati"][i],
                    "serviceData": data["Servisa dati"][i],
                    "amount": str(amount),  
                    "deliveryZone": data["Piegādes zona"][i],
                    "dimensions": data["Dimensijas"][i],
                    "country": data["Valsts"][i]
                })

            return {
                "status": "success",
                "files": self.processed_files,
                "data": result_data
            }

        except Exception as e:
            logging.error(f"Error processing FedEx bill: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
