from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import os
import shutil

class BaseProcessor:
    def __init__(self, input_file: str, user_id: str):
        """Initialize processor with input file"""
        self.input_file = input_file
        self.user_id = user_id
        self.timestamp = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        
        # Get the absolute path to the backend directory
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
        self.processed_dir = self.base_dir / "processed_files"
        
        # Log paths for debugging
        logging.info(f"Base dir: {self.base_dir}")
        logging.info(f"Processed dir: {self.processed_dir}")
        
        # Create user-specific output directory
        self.output_dir = self.processed_dir / user_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output dir: {self.output_dir}")
        
        # Dictionary to store paths to processed files
        self.processed_files = {}
        
        # Load and clean the input file
        self.df = self._load_file()
        
        # Create subdirectories
        self.invoices_dir = self.output_dir / "Invoices"
        self.invoices_dir.mkdir(exist_ok=True, parents=True)
        
        self.fedex_dir = self.output_dir / "FedEx"
        self.fedex_dir.mkdir(exist_ok=True, parents=True)
        
        # Create temp directory
        self.temp_dir = self.base_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        logging.info(f"Temp directory: {self.temp_dir}")
        
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names by removing whitespace and special characters"""
        # Create a copy of the DataFrame to avoid modifying the original
        df = df.copy()
        
        # Store original column names for mapping
        original_columns = df.columns.tolist()
        
        # Clean column names but preserve case
        cleaned_columns = df.columns.str.strip()  # Remove leading/trailing whitespace
        cleaned_columns = cleaned_columns.str.replace(' ', '')  # Remove spaces
        cleaned_columns = cleaned_columns.str.replace('#', '')  # Remove #
        cleaned_columns = cleaned_columns.str.replace('_', '')  # Remove _
        cleaned_columns = cleaned_columns.str.replace('-', '')  # Remove -
        cleaned_columns = cleaned_columns.str.replace('.', '')  # Remove .
        
        # Create mapping of cleaned to original column names
        self.column_mapping = dict(zip(cleaned_columns, original_columns))
        
        # Rename DataFrame columns
        df.columns = cleaned_columns
        
        # Log column mapping for debugging
        logging.info(f"Column mapping: {self.column_mapping}")
        
        return df
        
    def _load_file(self) -> pd.DataFrame:
        """Load Excel or CSV file into DataFrame"""
        try:
            logging.info(f"Starting file load process for: {self.input_file}")
            
            # Check if file exists
            if not os.path.exists(self.input_file):
                raise FileNotFoundError(f"Input file not found: {self.input_file}")
                
            # Check file size
            file_size = os.path.getsize(self.input_file)
            logging.info(f"File size: {file_size / 1024:.2f} KB")
            
            # Load file based on extension
            file_extension = Path(self.input_file).suffix.lower()
            if file_extension == '.csv':
                logging.info("Loading CSV file...")
                df = pd.read_csv(self.input_file)
            elif file_extension in ['.xlsx', '.xls']:
                logging.info("Loading Excel file...")
                df = pd.read_excel(self.input_file)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Clean column names
            df = self._clean_column_names(df)
            
            # Log DataFrame info
            logging.info(f"File loaded successfully:")
            logging.info(f"- Columns: {list(df.columns)}")
            logging.info(f"- Rows: {len(df)}")
            logging.info(f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            return df
            
        except pd.errors.EmptyDataError:
            logging.error("The file is empty")
            raise
        except pd.errors.ParserError as e:
            logging.error(f"Error parsing file: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error loading file: {str(e)}")
            logging.error(f"File path: {self.input_file}")
            logging.error(f"File exists: {os.path.exists(self.input_file)}")
            raise
            
    def validate_data(self) -> bool:
        """Validate the input data"""
        raise NotImplementedError("Subclasses must implement validate_data()")
        
    def convert_countries_in_df(self, df: pd.DataFrame, country_column: str) -> pd.DataFrame:
        """Convert country names to ISO codes in a DataFrame"""
        df = df.copy()
        df[country_column] = df[country_column].apply(lambda x: country_abbreviations.get(x, x))
        return df
        
    def clean_up(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.input_file) and str(self.input_file).startswith(str(self.temp_dir)):
                os.remove(self.input_file)
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
            
    def save_file(self, df: pd.DataFrame, filename: str, file_type: str = None) -> str:
        """Save DataFrame to file"""
        try:
            # Determine the output directory based on file type
            if file_type == "fedex":
                output_dir = self.fedex_dir
            elif file_type == "dhl":
                output_dir = self.output_dir
            elif file_type == "ups":
                output_dir = self.output_dir
            elif file_type == "invoice":
                output_dir = self.invoices_dir
            else:
                output_dir = self.output_dir

            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save the file
            output_path = output_dir / filename
            df.to_excel(output_path, index=False)
            logging.info(f"Successfully saved file to: {output_path}")

            # Return relative path from processed_files directory
            try:
                rel_path = output_path.relative_to(self.processed_dir)
                return str(rel_path)
            except ValueError:
                # If relative_to fails, return the full path
                return str(output_path)

        except Exception as e:
            logging.error(f"Error saving file: {str(e)}")
            raise
            
    def process(self) -> Dict[str, Any]:
        """Process the file and return results"""
        raise NotImplementedError("Subclasses must implement process()")