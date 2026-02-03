"""
KPI Calculator for Enterprise Analytics Dashboard
Calculates comprehensive business metrics and performance indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats
import logging

class KPICalculator:
    """Calculate comprehensive KPIs for business analytics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all KPIs for the dashboard"""
        try:
            kpis = {}
            
            # Basic metrics
            kpis.update(self._calculate_basic_metrics(df))
            
            # Anomaly metrics
            kpis.update(self._calculate_anomaly_metrics(df))
            
            # Trend metrics
            kpis.update(self._calculate_trend_metrics(df))
            
            # Performance metrics
            kpis.update(self._calculate_performance_metrics(df))
            
            # Time-based metrics
            kpis.update(self._calculate_time_metrics(df))
            
            return kpis
            
        except Exception as e:
            self.logger.error(f"Error calculating KPIs: {str(e)}")
            return self._fallback_kpis(df)
    
    def _calculate_basic_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistical metrics"""
        numeric_cols = self._get_numeric_columns(df)
        
        if not numeric_cols:
            return {}
        
        main_metric = numeric_cols[0]  # Use first numeric column as primary
        
        return {
            'total_records': len(df),
            'total_value': df[main_metric].sum(),
            'average_value': df[main_metric].mean(),
            'median_value': df[main_metric].median(),
            'max_value': df[main_metric].max(),
            'min_value': df[main_metric].min(),
            'std_deviation': df[main_metric].std(),
            'variance': df[main_metric].var(),
            'range_value': df[main_metric].max() - df[main_metric].min()
        }
    
    def _calculate_anomaly_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate anomaly-related metrics"""
        if 'anomaly' not in df.columns:
            return {
                'anomaly_count': 0,
                'anomaly_percentage': 0.0,
                'normal_count': len(df)
            }
        
        anomaly_count = (df['anomaly'] == -1).sum()
        normal_count = (df['anomaly'] == 1).sum()
        total_count = len(df)
        
        return {
            'anomaly_count': int(anomaly_count),
            'normal_count': int(normal_count),
            'anomaly_percentage': (anomaly_count / total_count) * 100 if total_count > 0 else 0,
            'normal_percentage': (normal_count / total_count) * 100 if total_count > 0 else 0
        }
    
    def _calculate_trend_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend analysis metrics"""
        numeric_cols = self._get_numeric_columns(df)
        
        if not numeric_cols or len(df) < 2:
            return {
                'trend_direction': 'Stable',
                'trend_strength': 50.0,
                'trend_slope': 0.0
            }
        
        main_metric = numeric_cols[0]
        
        # Sort by date for trend analysis
        df_sorted = df.sort_values('date')
        
        # Calculate trend using linear regression
        x = np.arange(len(df_sorted))
        y = df_sorted[main_metric].values
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Determine trend direction
            if abs(slope) < std_err:  # Not statistically significant
                trend_direction = 'Stable'
            elif slope > 0:
                trend_direction = 'Up'
            else:
                trend_direction = 'Down'
            
            # Calculate trend strength (based on R-squared)
            trend_strength = abs(r_value) * 100
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'trend_slope': slope,
                'trend_r_squared': r_value ** 2,
                'trend_p_value': p_value
            }
            
        except Exception:
            # Fallback to simple comparison
            recent_avg = df_sorted[main_metric].tail(7).mean()
            historical_avg = df_sorted[main_metric].head(7).mean()
            
            if recent_avg > historical_avg * 1.05:
                trend_direction = 'Up'
            elif recent_avg < historical_avg * 0.95:
                trend_direction = 'Down'
            else:
                trend_direction = 'Stable'
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': 50.0,
                'trend_slope': 0.0
            }
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance and quality metrics"""
        numeric_cols = self._get_numeric_columns(df)
        
        if not numeric_cols:
            return {}
        
        main_metric = numeric_cols[0]
        
        # Calculate performance indicators
        recent_period = df.tail(30) if len(df) > 30 else df
        historical_period = df.head(30) if len(df) > 60 else df.head(len(df)//2) if len(df) > 2 else df
        
        recent_avg = recent_period[main_metric].mean()
        historical_avg = historical_period[main_metric].mean()
        
        # Performance change
        performance_change = ((recent_avg - historical_avg) / historical_avg) * 100 if historical_avg > 0 else 0
        
        # Volatility (coefficient of variation)
        volatility = (df[main_metric].std() / df[main_metric].mean()) * 100 if df[main_metric].mean() > 0 else 0
        
        # Consistency score (inverse of volatility, normalized)
        consistency_score = max(0, 100 - volatility)
        
        return {
            'recent_average': recent_avg,
            'historical_avg': historical_avg,
            'performance_change': performance_change,
            'volatility': volatility,
            'consistency_score': consistency_score,
            'coefficient_of_variation': volatility / 100
        }
    
    def _calculate_time_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate time-based metrics"""
        if 'date' not in df.columns:
            return {}
        
        date_range = df['date'].max() - df['date'].min()
        
        # Monthly aggregation
        monthly_data = df.groupby(df['date'].dt.to_period('M')).size()
        
        # Weekly aggregation
        weekly_data = df.groupby(df['date'].dt.to_period('W')).size()
        
        return {
            'date_range_days': date_range.days,
            'start_date': df['date'].min(),
            'end_date': df['date'].max(),
            'unique_dates': df['date'].nunique(),
            'avg_records_per_day': len(df) / max(1, date_range.days),
            'monthly_periods': len(monthly_data),
            'weekly_periods': len(weekly_data),
            'data_frequency': self._determine_data_frequency(df)
        }
    
    def _determine_data_frequency(self, df: pd.DataFrame) -> str:
        """Determine the frequency of data collection"""
        if len(df) < 2:
            return 'Unknown'
        
        # Calculate average time between records
        df_sorted = df.sort_values('date')
        time_diffs = df_sorted['date'].diff().dropna()
        avg_diff = time_diffs.mean()
        
        if avg_diff <= pd.Timedelta(hours=1):
            return 'Hourly'
        elif avg_diff <= pd.Timedelta(days=1):
            return 'Daily'
        elif avg_diff <= pd.Timedelta(days=7):
            return 'Weekly'
        elif avg_diff <= pd.Timedelta(days=31):
            return 'Monthly'
        else:
            return 'Irregular'
    
    def calculate_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate data quality metrics"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        
        # Completeness
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        # Uniqueness (for date column)
        date_uniqueness = (df['date'].nunique() / len(df)) * 100 if 'date' in df.columns else 100
        
        # Consistency (no negative values in positive metrics)
        numeric_cols = self._get_numeric_columns(df)
        consistency_issues = 0
        
        for col in numeric_cols:
            if df[col].min() < 0 and col.lower() in ['sales', 'revenue', 'amount', 'quantity', 'price']:
                consistency_issues += (df[col] < 0).sum()
        
        consistency = max(0, 100 - (consistency_issues / len(df)) * 100)
        
        # Validity (reasonable date ranges)
        validity = 100
        if 'date' in df.columns:
            current_year = pd.Timestamp.now().year
            future_dates = (df['date'].dt.year > current_year + 1).sum()
            old_dates = (df['date'].dt.year < current_year - 10).sum()
            validity = max(0, 100 - ((future_dates + old_dates) / len(df)) * 100)
        
        # Overall quality score
        overall_quality = (completeness + date_uniqueness + consistency + validity) / 4
        
        return {
            'completeness': completeness,
            'uniqueness': date_uniqueness,
            'consistency': consistency,
            'validity': validity,
            'overall_quality': overall_quality
        }
    
    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Get numeric columns excluding system columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude system-generated columns
        exclude_cols = [
            'anomaly', 'anomaly_score', 'year', 'month', 'day', 
            'weekday', 'quarter', 'week_of_year'
        ]
        
        return [col for col in numeric_cols if col not in exclude_cols]
    
    def _fallback_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback KPIs when calculation fails"""
        return {
            'total_records': len(df),
            'anomaly_count': 0,
            'anomaly_percentage': 0.0,
            'trend_direction': 'Stable',
            'trend_strength': 50.0,
            'historical_avg': 0.0,
            'performance_change': 0.0,
            'volatility': 0.0
        }