#!/bin/bash

echo "🚀 Starting Enterprise Analytics Dashboard..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+ and try again."
    exit 1
fi

# Install/update requirements
echo "📦 Installing/updating dependencies..."
pip install -r requirements.txt --quiet

# Run the Streamlit dashboard
echo
echo "🌟 Starting Streamlit dashboard at http://localhost:8501"
echo "📊 Enterprise Analytics Dashboard - Power BI Style with ML & AI"
echo
echo "Press Ctrl+C to stop the dashboard"
echo

streamlit run app.py