"""
Test script for Enterprise Analytics Dashboard
Validates core functionality before deployment
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_data_processing():
    """Test data processing functionality"""
    print("🧪 Testing data processing...")
    
    try:
        from preprocessing.preprocess import DataProcessor
        
        processor = DataProcessor()
        
        # Test with sample data
        sample_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'sales': np.random.randint(100, 200, 10),
            'revenue': np.random.randint(1000, 2000, 10)
        })
        
        # Save test data
        sample_data.to_csv('test_data.csv', index=False)
        
        # Test processing
        processed_data = processor.load_csv('test_data.csv')
        
        assert len(processed_data) > 0, "No data processed"
        assert 'date' in processed_data.columns, "Date column missing"
        assert processed_data['date'].dtype == 'datetime64[ns]', "Date not converted to datetime"
        
        print("✅ Data processing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False

def test_anomaly_detection():
    """Test anomaly detection functionality"""
    print("🧪 Testing anomaly detection...")
    
    try:
        from ml_engine.anomaly import AnomalyDetector
        
        detector = AnomalyDetector()
        
        # Create test data with obvious anomalies
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=20),
            'sales': [100] * 18 + [500, 600]  # Last two are anomalies
        })
        
        result = detector.detect(test_data)
        
        assert 'anomaly' in result.columns, "Anomaly column not added"
        assert 'anomaly_score' in result.columns, "Anomaly score not added"
        assert (result['anomaly'] == -1).sum() > 0, "No anomalies detected"
        
        print("✅ Anomaly detection test passed")
        return True
        
    except Exception as e:
        print(f"❌ Anomaly detection test failed: {e}")
        return False

def test_forecasting():
    """Test forecasting functionality"""
    print("🧪 Testing forecasting...")
    
    try:
        from ml_engine.forecasting import TimeSeriesForecaster
        
        forecaster = TimeSeriesForecaster()
        
        # Create test data with trend
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30),
            'sales': np.arange(100, 130)  # Linear trend
        })
        
        result = forecaster.forecast(test_data, 'sales', 7)
        
        assert 'forecast_data' in result, "Forecast data missing"
        assert len(result['forecast_data']) == 7, "Wrong forecast length"
        assert 'growth_rate' in result, "Growth rate missing"
        
        print("✅ Forecasting test passed")
        return True
        
    except Exception as e:
        print(f"❌ Forecasting test failed: {e}")
        return False

def test_kpi_calculation():
    """Test KPI calculation functionality"""
    print("🧪 Testing KPI calculation...")
    
    try:
        from dashboard.kpi_calculator import KPICalculator
        
        calculator = KPICalculator()
        
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'sales': np.random.randint(100, 200, 10),
            'anomaly': [1] * 8 + [-1] * 2  # 2 anomalies
        })
        
        kpis = calculator.calculate_all_kpis(test_data)
        
        assert 'total_records' in kpis, "Total records missing"
        assert 'anomaly_count' in kpis, "Anomaly count missing"
        assert 'trend_direction' in kpis, "Trend direction missing"
        assert kpis['anomaly_count'] == 2, "Wrong anomaly count"
        
        print("✅ KPI calculation test passed")
        return True
        
    except Exception as e:
        print(f"❌ KPI calculation test failed: {e}")
        return False

def test_visualization():
    """Test visualization functionality"""
    print("🧪 Testing visualization...")
    
    try:
        from dashboard.visualization_engine import VisualizationEngine
        
        viz_engine = VisualizationEngine()
        
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'sales': np.random.randint(100, 200, 10),
            'anomaly': [1] * 8 + [-1] * 2
        })
        
        # Test time series chart
        fig = viz_engine.create_time_series_chart(test_data, 'sales')
        assert fig is not None, "Time series chart not created"
        
        # Test pie chart
        fig = viz_engine.create_anomaly_pie_chart(test_data)
        assert fig is not None, "Pie chart not created"
        
        print("✅ Visualization test passed")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def test_sample_data():
    """Test sample data loading"""
    print("🧪 Testing sample data...")
    
    try:
        sample_path = Path('data/sample_sales.csv')
        assert sample_path.exists(), "Sample data file missing"
        
        df = pd.read_csv(sample_path)
        assert len(df) > 0, "Sample data is empty"
        assert 'date' in df.columns, "Date column missing in sample data"
        
        # Test date parsing
        df['date'] = pd.to_datetime(df['date'])
        assert df['date'].dtype == 'datetime64[ns]', "Date parsing failed"
        
        print("✅ Sample data test passed")
        return True
        
    except Exception as e:
        print(f"❌ Sample data test failed: {e}")
        return False

def cleanup():
    """Clean up test files"""
    test_files = ['test_data.csv']
    for file in test_files:
        if Path(file).exists():
            Path(file).unlink()

def main():
    """Run all tests"""
    print("🚀 Enterprise Analytics Dashboard - Core Functionality Tests")
    print("=" * 70)
    
    tests = [
        ("Sample Data", test_sample_data),
        ("Data Processing", test_data_processing),
        ("Anomaly Detection", test_anomaly_detection),
        ("Forecasting", test_forecasting),
        ("KPI Calculation", test_kpi_calculation),
        ("Visualization", test_visualization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}...")
        if test_func():
            passed += 1
    
    # Cleanup
    cleanup()
    
    # Results
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Core functionality is working correctly.")
        print("✅ Ready for Streamlit deployment")
    else:
        print(f"❌ {total - passed} tests failed. Please fix issues before deployment.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)