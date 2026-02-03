import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import stats
import logging
from typing import Dict, List, Tuple, Optional

class AnomalyDetector:
    """Enterprise-grade anomaly detection with multiple algorithms"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
    def detect(self, df: pd.DataFrame, method: str = 'isolation_forest', 
               contamination: float = 0.1) -> pd.DataFrame:
        """
        Detect anomalies using specified method
        
        Args:
            df: Input dataframe
            method: 'isolation_forest', 'statistical', 'dbscan', or 'ensemble'
            contamination: Expected proportion of anomalies
        """
        try:
            df_result = df.copy()
            
            # Get numeric columns for analysis
            numeric_cols = self._get_numeric_columns(df)
            
            if not numeric_cols:
                raise ValueError("No numeric columns found for anomaly detection")
            
            # Apply selected method
            if method == 'isolation_forest':
                anomalies = self._isolation_forest_detection(df, numeric_cols, contamination)
            elif method == 'statistical':
                anomalies = self._statistical_detection(df, numeric_cols)
            elif method == 'dbscan':
                anomalies = self._dbscan_detection(df, numeric_cols)
            elif method == 'ensemble':
                anomalies = self._ensemble_detection(df, numeric_cols, contamination)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            df_result['anomaly'] = anomalies
            df_result['anomaly_score'] = self._calculate_anomaly_scores(df, numeric_cols)
            
            # Add anomaly severity
            df_result['anomaly_severity'] = self._calculate_severity(df_result)
            
            return df_result
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            # Fallback: return original data with no anomalies
            df_result = df.copy()
            df_result['anomaly'] = 1
            df_result['anomaly_score'] = 0.0
            df_result['anomaly_severity'] = 'Normal'
            return df_result
    
    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Get numeric columns excluding derived columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove time-based features and existing anomaly columns
        exclude_cols = ['year', 'month', 'day', 'weekday', 'quarter', 'week_of_year', 
                       'anomaly', 'anomaly_score', 'anomaly_severity']
        
        return [col for col in numeric_cols if col not in exclude_cols]
    
    def _isolation_forest_detection(self, df: pd.DataFrame, numeric_cols: List[str], 
                                  contamination: float, fast_mode: bool = False) -> np.ndarray:
        """Isolation Forest anomaly detection - with performance optimization"""
        # Optimize parameters for large datasets
        if fast_mode and len(df) > 10000:
            n_estimators = 50  # Reduced from 100
            max_samples = min(1000, len(df))  # Limit sample size
        else:
            n_estimators = 100
            max_samples = 'auto'
        
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=n_estimators,
            max_samples=max_samples,
            n_jobs=-1  # Use all available cores
        )
        
        X = df[numeric_cols].values
        anomalies = model.fit_predict(X)
        
        return anomalies
    
    def _statistical_detection(self, df: pd.DataFrame, numeric_cols: List[str]) -> np.ndarray:
        """Statistical anomaly detection using Z-score and IQR"""
        anomalies = np.ones(len(df))
        
        for col in numeric_cols:
            # Z-score method
            z_scores = np.abs(stats.zscore(df[col]))
            z_anomalies = z_scores > 3
            
            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_anomalies = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            # Combine methods (anomaly if detected by either method)
            col_anomalies = z_anomalies | iqr_anomalies
            anomalies[col_anomalies] = -1
        
        return anomalies
    
    def _dbscan_detection(self, df: pd.DataFrame, numeric_cols: List[str]) -> np.ndarray:
        """DBSCAN clustering for anomaly detection"""
        X = self.scaler.fit_transform(df[numeric_cols])
        
        # Auto-tune eps parameter
        eps = self._estimate_eps(X)
        
        dbscan = DBSCAN(eps=eps, min_samples=max(2, len(df) // 50))
        clusters = dbscan.fit_predict(X)
        
        # Points in cluster -1 are considered anomalies
        anomalies = np.where(clusters == -1, -1, 1)
        
        return anomalies
    
    def _ensemble_detection(self, df: pd.DataFrame, numeric_cols: List[str], 
                          contamination: float) -> np.ndarray:
        """Ensemble method combining multiple detection algorithms"""
        # Get predictions from different methods
        iso_anomalies = self._isolation_forest_detection(df, numeric_cols, contamination)
        stat_anomalies = self._statistical_detection(df, numeric_cols)
        dbscan_anomalies = self._dbscan_detection(df, numeric_cols)
        
        # Voting mechanism: anomaly if detected by at least 2 methods
        anomaly_votes = (iso_anomalies == -1).astype(int) + \
                       (stat_anomalies == -1).astype(int) + \
                       (dbscan_anomalies == -1).astype(int)
        
        ensemble_anomalies = np.where(anomaly_votes >= 2, -1, 1)
        
        return ensemble_anomalies
    
    def _calculate_anomaly_scores(self, df: pd.DataFrame, numeric_cols: List[str]) -> np.ndarray:
        """Calculate anomaly scores for ranking"""
        scores = np.zeros(len(df))
        
        for col in numeric_cols:
            # Normalized distance from median
            median_val = df[col].median()
            mad = np.median(np.abs(df[col] - median_val))  # Median Absolute Deviation
            
            if mad > 0:
                col_scores = np.abs(df[col] - median_val) / mad
                scores += col_scores
        
        # Normalize to 0-1 range
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def _calculate_severity(self, df: pd.DataFrame) -> List[str]:
        """Calculate anomaly severity levels"""
        severity = []
        
        for _, row in df.iterrows():
            if row['anomaly'] == 1:
                severity.append('Normal')
            else:
                score = row['anomaly_score']
                if score > 0.8:
                    severity.append('Critical')
                elif score > 0.6:
                    severity.append('High')
                elif score > 0.4:
                    severity.append('Medium')
                else:
                    severity.append('Low')
        
        return severity
    
    def _estimate_eps(self, X: np.ndarray) -> float:
        """Estimate optimal eps parameter for DBSCAN"""
        from sklearn.neighbors import NearestNeighbors
        
        # Use k=4 as a rule of thumb
        k = min(4, X.shape[0] - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, _ = nbrs.kneighbors(X)
        
        # Sort distances and find the "elbow"
        distances = np.sort(distances[:, k-1])
        
        # Simple heuristic: use 75th percentile
        eps = np.percentile(distances, 75)
        
        return max(eps, 0.1)  # Ensure minimum eps value
    
    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive anomaly summary"""
        if 'anomaly' not in df.columns:
            return {"error": "No anomaly detection results found"}
        
        anomalies = df[df['anomaly'] == -1]
        total_records = len(df)
        anomaly_count = len(anomalies)
        
        summary = {
            'total_records': total_records,
            'anomaly_count': anomaly_count,
            'anomaly_percentage': (anomaly_count / total_records) * 100 if total_records > 0 else 0,
            'severity_distribution': df['anomaly_severity'].value_counts().to_dict() if 'anomaly_severity' in df.columns else {},
            'date_range': {
                'first_anomaly': anomalies['date'].min() if len(anomalies) > 0 else None,
                'last_anomaly': anomalies['date'].max() if len(anomalies) > 0 else None
            },
            'top_anomalies': anomalies.nlargest(5, 'anomaly_score')[['date', 'anomaly_score', 'anomaly_severity']].to_dict('records') if len(anomalies) > 0 else []
        }
        
        return summary

# Legacy function for backward compatibility
def detect_anomalies(df):
    detector = AnomalyDetector()
    return detector.detect(df)
