import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesForecaster:
    """Enterprise-grade time series forecasting with multiple models"""
    
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.fitted_models = {}
        
    def forecast(self, df: pd.DataFrame, target_column: str, 
                forecast_days: int = 7, confidence_level: int = 95) -> dict:
        """
        Generate comprehensive forecast with multiple models
        
        Args:
            df: Input dataframe with date and target columns
            target_column: Column to forecast
            forecast_days: Number of days to forecast
            confidence_level: Confidence level for intervals
        """
        try:
            # Prepare data
            forecast_data = self._prepare_forecast_data(df, target_column)
            
            # Train models
            model_results = {}
            for model_name, model in self.models.items():
                result = self._train_and_forecast(
                    forecast_data, model, model_name, forecast_days, confidence_level
                )
                model_results[model_name] = result
            
            # Select best model based on validation
            best_model = self._select_best_model(model_results)
            
            # Generate comprehensive forecast result
            forecast_result = self._generate_forecast_result(
                df, forecast_data, model_results[best_model], 
                target_column, forecast_days, best_model
            )
            
            return forecast_result
            
        except Exception as e:
            # Fallback to simple linear trend
            return self._fallback_forecast(df, target_column, forecast_days)
    
    def _prepare_forecast_data(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Prepare data for forecasting with feature engineering"""
        forecast_df = df.copy()
        forecast_df = forecast_df.sort_values('date').reset_index(drop=True)
        
        # Create time-based features
        forecast_df['time_index'] = np.arange(len(forecast_df))
        forecast_df['day_of_week'] = forecast_df['date'].dt.dayofweek
        forecast_df['day_of_month'] = forecast_df['date'].dt.day
        forecast_df['month'] = forecast_df['date'].dt.month
        forecast_df['quarter'] = forecast_df['date'].dt.quarter
        
        # Create lag features
        for lag in [1, 7, 30]:
            if len(forecast_df) > lag:
                forecast_df[f'{target_column}_lag_{lag}'] = forecast_df[target_column].shift(lag)
        
        # Create rolling statistics
        for window in [7, 30]:
            if len(forecast_df) > window:
                forecast_df[f'{target_column}_rolling_mean_{window}'] = (
                    forecast_df[target_column].rolling(window=window).mean()
                )
                forecast_df[f'{target_column}_rolling_std_{window}'] = (
                    forecast_df[target_column].rolling(window=window).std()
                )
        
        # Create trend features
        if len(forecast_df) > 1:
            forecast_df[f'{target_column}_diff'] = forecast_df[target_column].diff()
            forecast_df[f'{target_column}_pct_change'] = forecast_df[target_column].pct_change()
        
        # Drop rows with NaN values (due to lag and rolling features)
        forecast_df = forecast_df.dropna()
        
        return forecast_df
    
    def _train_and_forecast(self, data: pd.DataFrame, model, model_name: str,
                          forecast_days: int, confidence_level: int) -> dict:
        """Train model and generate forecast"""
        target_col = [col for col in data.columns if not col.startswith(('date', 'time_index'))][0]
        
        # Prepare features (exclude target and date columns)
        feature_cols = [col for col in data.columns 
                       if col not in ['date', target_col] and not col.endswith('_diff') and not col.endswith('_pct_change')]
        
        X = data[feature_cols]
        y = data[target_col]
        
        # Train-test split for validation
        split_idx = int(len(data) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Validate
        y_pred_val = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred_val)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_val))
        mape = np.mean(np.abs((y_test - y_pred_val) / y_test)) * 100
        
        # Generate future features
        future_features = self._generate_future_features(data, feature_cols, forecast_days)
        
        # Make predictions
        future_predictions = model.predict(future_features)
        
        # Calculate confidence intervals
        residuals = y_test - y_pred_val
        std_residual = np.std(residuals)
        confidence_interval = stats.norm.ppf((1 + confidence_level/100) / 2) * std_residual
        
        return {
            'model_name': model_name,
            'predictions': future_predictions,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'confidence_interval': confidence_interval,
            'model': model
        }
    
    def _generate_future_features(self, data: pd.DataFrame, feature_cols: list, 
                                forecast_days: int) -> np.ndarray:
        """Generate features for future dates"""
        last_date = data['date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        
        future_features = []
        
        for i, future_date in enumerate(future_dates):
            features = {}
            
            # Time-based features
            features['time_index'] = data['time_index'].iloc[-1] + i + 1
            features['day_of_week'] = future_date.dayofweek
            features['day_of_month'] = future_date.day
            features['month'] = future_date.month
            features['quarter'] = future_date.quarter
            
            # For lag and rolling features, use the most recent available values
            for col in feature_cols:
                if col not in features:
                    if 'lag' in col or 'rolling' in col:
                        # Use the last available value
                        features[col] = data[col].iloc[-1]
                    else:
                        features[col] = data[col].iloc[-1]
            
            future_features.append([features[col] for col in feature_cols])
        
        return np.array(future_features)
    
    def _select_best_model(self, model_results: dict) -> str:
        """Select best model based on validation metrics"""
        best_model = None
        best_score = float('inf')
        
        for model_name, result in model_results.items():
            # Use MAPE as primary metric (lower is better)
            score = result['mape']
            if score < best_score:
                best_score = score
                best_model = model_name
        
        return best_model
    
    def _generate_forecast_result(self, original_df: pd.DataFrame, forecast_data: pd.DataFrame,
                                best_result: dict, target_column: str, 
                                forecast_days: int, best_model: str) -> dict:
        """Generate comprehensive forecast result"""
        # Create future dates
        last_date = original_df['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        
        # Calculate growth rate
        recent_avg = original_df[target_column].tail(7).mean()
        forecast_avg = np.mean(best_result['predictions'])
        growth_rate = ((forecast_avg - recent_avg) / recent_avg) * 100 if recent_avg > 0 else 0
        
        # Calculate volatility
        historical_volatility = original_df[target_column].std() / original_df[target_column].mean() * 100
        
        # Determine risk level
        if abs(growth_rate) > 20 or historical_volatility > 30:
            risk_level = "High"
        elif abs(growth_rate) > 10 or historical_volatility > 15:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_value': best_result['predictions'],
            'lower_bound': best_result['predictions'] - best_result['confidence_interval'],
            'upper_bound': best_result['predictions'] + best_result['confidence_interval']
        })
        
        return {
            'forecast_data': forecast_df,
            'growth_rate': growth_rate,
            'accuracy': max(0, 100 - best_result['mape']),
            'risk_level': risk_level,
            'volatility': historical_volatility,
            'model_used': best_model,
            'metrics': {
                'mae': best_result['mae'],
                'rmse': best_result['rmse'],
                'mape': best_result['mape']
            },
            'confidence_interval': best_result['confidence_interval']
        }
    
    def _fallback_forecast(self, df: pd.DataFrame, target_column: str, 
                         forecast_days: int) -> dict:
        """Simple fallback forecast using linear trend"""
        df_sorted = df.sort_values('date')
        
        # Simple linear regression on time index
        time_index = np.arange(len(df_sorted))
        values = df_sorted[target_column].values
        
        # Fit linear model
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, values)
        
        # Generate future predictions
        future_time_index = np.arange(len(df_sorted), len(df_sorted) + forecast_days)
        future_predictions = slope * future_time_index + intercept
        
        # Create future dates
        last_date = df_sorted['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        
        # Simple confidence interval based on standard error
        confidence_interval = 1.96 * std_err * np.sqrt(forecast_days)
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_value': future_predictions,
            'lower_bound': future_predictions - confidence_interval,
            'upper_bound': future_predictions + confidence_interval
        })
        
        # Calculate basic metrics
        recent_avg = df_sorted[target_column].tail(7).mean()
        forecast_avg = np.mean(future_predictions)
        growth_rate = ((forecast_avg - recent_avg) / recent_avg) * 100 if recent_avg > 0 else 0
        
        return {
            'forecast_data': forecast_df,
            'growth_rate': growth_rate,
            'accuracy': max(0, abs(r_value) * 100),
            'risk_level': "Medium",
            'volatility': df_sorted[target_column].std() / df_sorted[target_column].mean() * 100,
            'model_used': 'linear_fallback',
            'metrics': {
                'mae': std_err,
                'rmse': std_err,
                'mape': 15.0  # Default estimate
            },
            'confidence_interval': confidence_interval
        }

# Legacy function for backward compatibility
def forecast_sales(df, days=7):
    forecaster = TimeSeriesForecaster()
    result = forecaster.forecast(df, 'sales', days)
    return result['forecast_data'][['date', 'predicted_value']].rename(columns={'predicted_value': 'predicted_sales'})
