import pandas as pd
import numpy as np
from pathlib import Path
import io
import logging
from typing import Union, Optional
import chardet

class DataProcessor:
    """Enterprise-grade data processing with robust error handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_csv(self, file_input: Union[str, io.BytesIO]) -> pd.DataFrame:
        """
        Load CSV with automatic encoding detection and error handling
        Supports both file paths and uploaded file objects
        """
        try:
            if isinstance(file_input, str):
                # File path
                return self._load_from_path(file_input)
            else:
                # Uploaded file object
                return self._load_from_upload(file_input)
                
        except Exception as e:
            self.logger.error(f"Error loading CSV: {str(e)}")
            raise ValueError(f"Failed to load CSV file: {str(e)}")
    
    def _load_from_path(self, file_path: str) -> pd.DataFrame:
        """Load CSV from file path"""
        base_dir = Path(__file__).resolve().parent.parent
        csv_path = base_dir / file_path
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                return self._preprocess_dataframe(df)
            except UnicodeDecodeError:
                continue
        
        raise ValueError("Could not decode CSV file with any supported encoding")
    
    def _load_from_upload(self, uploaded_file) -> pd.DataFrame:
        """Load CSV from uploaded file with encoding detection"""
        # Read file content
        content = uploaded_file.getvalue()
        
        # Detect encoding
        detected = chardet.detect(content)
        encoding = detected.get('encoding', 'utf-8')
        
        # Try detected encoding first, then fallbacks
        encodings = [encoding, 'utf-8', 'latin-1', 'cp1252']
        
        for enc in encodings:
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc)
                return self._preprocess_dataframe(df)
            except (UnicodeDecodeError, pd.errors.EmptyDataError):
                continue
        
        raise ValueError("Could not decode uploaded CSV file")
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess dataframe with automatic column detection"""
        if df.empty:
            raise ValueError("CSV file is empty")
        
        # Auto-detect date column
        date_col = self._detect_date_column(df)
        if not date_col:
            raise ValueError("No date column found. CSV must contain a date-like column (date, order_date, timestamp, etc.)")
        
        # Convert to datetime
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Remove rows with invalid dates
        initial_rows = len(df)
        df = df.dropna(subset=['date'])
        
        if len(df) == 0:
            raise ValueError("No valid dates found in the date column")
        
        if len(df) < initial_rows * 0.8:
            self.logger.warning(f"Removed {initial_rows - len(df)} rows with invalid dates")
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Add time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Validate numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for analysis")
        
        # Handle missing values in numeric columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                # Fill with median for robustness
                df[col] = df[col].fillna(df[col].median())
        
        # Remove duplicate dates (keep last occurrence)
        df = df.drop_duplicates(subset=['date'], keep='last')
        
        return df
    
    def _detect_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect date column from common patterns"""
        date_patterns = [
            'date', 'time', 'timestamp', 'created', 'updated', 
            'order_date', 'transaction_date', 'sale_date',
            'datetime', 'created_at', 'updated_at'
        ]
        
        # First, look for exact matches
        for col in df.columns:
            if col.lower() in date_patterns:
                return col
        
        # Then, look for partial matches
        for col in df.columns:
            col_lower = col.lower()
            for pattern in date_patterns:
                if pattern in col_lower:
                    return col
        
        # Finally, try to detect by data type
        for col in df.columns:
            try:
                pd.to_datetime(df[col].head(10), errors='raise')
                return col
            except:
                continue
        
        return None
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Generate comprehensive data summary"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max(),
                'days': (df['date'].max() - df['date'].min()).days
            },
            'numeric_columns': numeric_cols,
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        return summary

# Legacy function for backward compatibility
def load_and_preprocess(file_path):
    processor = DataProcessor()
    return processor.load_csv(file_path)
