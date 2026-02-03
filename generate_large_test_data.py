"""
Generate large test datasets for performance testing
Creates CSV files with 100k+ rows for testing dashboard performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_large_sales_data(num_rows=100000, filename="large_test_data.csv"):
    """Generate large sales dataset for testing"""
    print(f"🚀 Generating {num_rows:,} rows of test data...")
    
    # Date range - last 3 years
    start_date = datetime.now() - timedelta(days=3*365)
    dates = pd.date_range(start=start_date, periods=num_rows, freq='H')
    
    # Regions and products for variety
    regions = ['North', 'South', 'East', 'West', 'Central']
    products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
    channels = ['Online', 'Retail', 'Wholesale', 'Direct']
    
    # Generate realistic business data
    np.random.seed(42)  # For reproducible results
    
    data = {
        'date': dates,
        'sales': np.random.normal(1500, 300, num_rows).astype(int),
        'revenue': np.random.normal(15000, 3000, num_rows).astype(int),
        'quantity': np.random.poisson(50, num_rows),
        'profit': np.random.normal(4500, 900, num_rows).astype(int),
        'customer_count': np.random.poisson(30, num_rows),
        'region': np.random.choice(regions, num_rows),
        'product': np.random.choice(products, num_rows),
        'channel': np.random.choice(channels, num_rows),
        'discount_rate': np.random.uniform(0, 0.3, num_rows).round(3),
        'marketing_spend': np.random.normal(500, 100, num_rows).astype(int)
    }
    
    # Add some seasonal patterns
    day_of_year = dates.dayofyear
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
    data['sales'] = (data['sales'] * seasonal_factor).astype(int)
    data['revenue'] = (data['revenue'] * seasonal_factor).astype(int)
    
    # Add some anomalies (5% of data)
    anomaly_indices = np.random.choice(num_rows, size=int(num_rows * 0.05), replace=False)
    for idx in anomaly_indices:
        # Create anomalous values
        data['sales'][idx] = int(data['sales'][idx] * np.random.choice([0.1, 3.0]))  # Very low or very high
        data['revenue'][idx] = int(data['revenue'][idx] * np.random.choice([0.1, 3.0]))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure positive values
    df['sales'] = df['sales'].clip(lower=50)
    df['revenue'] = df['revenue'].clip(lower=500)
    df['profit'] = df['profit'].clip(lower=100)
    df['marketing_spend'] = df['marketing_spend'].clip(lower=50)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    
    print(f"✅ Generated {filename} with {num_rows:,} rows")
    print(f"📊 File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"💰 Sales range: ${df['sales'].min():,} to ${df['sales'].max():,}")
    
    return df

def generate_multiple_test_files():
    """Generate multiple test files of different sizes"""
    sizes = [
        (10000, "test_10k.csv"),
        (50000, "test_50k.csv"), 
        (100000, "test_100k.csv"),
        (250000, "test_250k.csv")
    ]
    
    for size, filename in sizes:
        generate_large_sales_data(size, filename)
        print()

if __name__ == "__main__":
    print("🚀 Large Dataset Generator for Performance Testing")
    print("=" * 60)
    
    choice = input("Generate: (1) 100k rows, (2) Multiple sizes, (3) Custom size: ")
    
    if choice == "1":
        generate_large_sales_data(100000, "test_100k_rows.csv")
    elif choice == "2":
        generate_multiple_test_files()
    elif choice == "3":
        try:
            custom_size = int(input("Enter number of rows: "))
            filename = f"test_{custom_size//1000}k_rows.csv"
            generate_large_sales_data(custom_size, filename)
        except ValueError:
            print("Invalid input. Using default 100k rows.")
            generate_large_sales_data(100000, "test_100k_rows.csv")
    else:
        print("Invalid choice. Generating 100k rows by default.")
        generate_large_sales_data(100000, "test_100k_rows.csv")
    
    print("\n🎯 Test files generated! Upload them to your dashboard to test performance.")