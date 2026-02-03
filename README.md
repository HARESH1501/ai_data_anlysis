# Enterprise Analytics Dashboard

A production-ready Power BI-style analytics dashboard with Machine Learning and Generative AI integration, built with **Streamlit** for easy deployment and use.

## 🚀 Features

### 📊 Power BI Style Interface
- **Interactive KPI Cards**: Real-time metrics with trend indicators
- **Professional Visualizations**: Time series, bar charts, pie charts, histograms
- **Multi-page Navigation**: Overview, Forecast, Anomalies, Reports
- **Responsive Design**: Optimized for desktop and mobile

### 🤖 Machine Learning Capabilities
- **Anomaly Detection**: Multiple algorithms (Isolation Forest, Statistical, DBSCAN, Ensemble)
- **Time Series Forecasting**: Advanced forecasting with confidence intervals
- **Trend Analysis**: Automated trend detection and change-point analysis
- **Data Quality Assessment**: Comprehensive quality metrics

### 🧠 Generative AI Insights
- **Business-Friendly Explanations**: Natural language insights for executives
- **Anomaly Analysis**: AI-powered anomaly interpretation
- **Forecast Insights**: Strategic recommendations based on predictions
- **Executive Summaries**: Comprehensive business reports

### 📈 Advanced Analytics
- **Multi-Metric Analysis**: Support for multiple business metrics
- **Date Range Filtering**: Interactive time-based analysis
- **Export Capabilities**: Text, HTML, and Excel reports
- **Real-Time Processing**: Instant analysis on data upload

## 🛠️ Installation & Setup

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd enterprise-analytics-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (Optional for AI features)
   Create a `.env` file:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```
   Or use the convenience scripts:
   ```bash
   # Windows
   start.bat
   
   # Linux/Mac
   chmod +x start.sh
   ./start.sh
   ```

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select this repository

3. **Configure secrets** (Optional)
   In Streamlit Cloud settings, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

4. **Deploy**
   - Click "Deploy"
   - Your app will be available at `https://your-app-name.streamlit.app`

## 📊 Data Requirements

### CSV File Format
Your CSV file must contain:
- **Date Column**: Any column with "date", "time", "timestamp" in the name
- **Numeric Columns**: At least one numeric column for analysis

### Supported Date Formats
- `2024-01-01`
- `01/01/2024`
- `2024-01-01 12:00:00`
- `Jan 1, 2024`

### Example Data Structure
```csv
date,sales,revenue,quantity
2024-01-01,120,1200,10
2024-01-02,130,1300,11
2024-01-03,125,1250,9
```

## 🎯 Usage Guide

### 1. Data Upload
- Use the sidebar to upload your CSV file
- Or click "Use Sample Data" to try with demo data
- Data is automatically processed and validated

### 2. Overview Tab
- View key performance indicators
- Analyze trends and patterns
- Get AI-generated business insights
- Interactive visualizations with anomaly highlighting

### 3. Forecast Tab
- Select metrics and forecast periods (7, 14, or 30 days)
- Generate predictions with confidence intervals
- View accuracy metrics and risk assessments
- Get strategic recommendations

### 4. Anomalies Tab
- Review detected anomalies and their severity
- Analyze anomaly patterns over time
- Get AI explanations of unusual patterns
- Export anomaly reports

### 5. Reports Tab
- Generate executive summaries
- Export comprehensive PDF reports
- Download Excel files with analysis
- View data quality metrics

## 🔧 Configuration

### Theme Customization
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
```

### ML Model Parameters
Modify settings in respective modules:
- `ml_engine/anomaly.py`: Anomaly detection parameters
- `ml_engine/forecasting.py`: Forecasting model settings
- `genai_engine/insight_generator.py`: AI prompt configurations

## 📁 Project Structure

```
enterprise-analytics-dashboard/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── .streamlit/
│   └── config.toml               # Streamlit configuration
├── preprocessing/
│   └── preprocess.py             # Data processing and validation
├── ml_engine/
│   ├── anomaly.py               # Anomaly detection algorithms
│   └── forecasting.py           # Time series forecasting
├── genai_engine/
│   └── insight_generator.py     # AI insight generation
├── dashboard/
│   ├── kpi_calculator.py        # KPI calculations
│   └── visualization_engine.py  # Chart generation
├── utils/
│   ├── export_manager.py        # Report export functionality
│   └── theme_manager.py         # UI theming
└── data/
    └── sample_sales.csv         # Sample dataset
```

## 🚀 Production Features

### Performance Optimization
- **Caching**: Streamlit caching for ML models and data processing
- **Lazy Loading**: Components loaded on demand
- **Memory Management**: Efficient data handling for large datasets
- **Error Handling**: Robust error handling with graceful fallbacks

### Security
- **Input Validation**: Comprehensive data validation
- **File Size Limits**: Configurable upload limits
- **XSRF Protection**: Built-in security features
- **Environment Variables**: Secure API key management

### Scalability
- **Modular Architecture**: Easy to extend and maintain
- **Cloud-Ready**: Optimized for Streamlit Cloud deployment
- **Multi-User Support**: Concurrent user handling
- **Resource Monitoring**: Built-in performance tracking

## 🔍 Troubleshooting

### Common Issues

1. **"No date column found"**
   - Ensure your CSV has a column with "date", "time", or "timestamp" in the name
   - Check date format compatibility

2. **"No numeric columns found"**
   - Verify your CSV contains at least one numeric column
   - Check for proper number formatting (no text in numeric columns)

3. **AI insights not generating**
   - Verify GROQ_API_KEY is set correctly
   - Check internet connection for API access
   - Fallback insights will be provided if API fails

4. **Large file upload issues**
   - Files over 200MB may fail to upload
   - Consider data sampling for very large datasets
   - Use data preprocessing to reduce file size

### Performance Tips
- Use date range filtering for large datasets
- Sample data for initial exploration
- Export filtered results for detailed analysis
- Clear browser cache if experiencing issues

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review the configuration documentation

## 🎉 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- ML powered by [scikit-learn](https://scikit-learn.org/)
- AI insights by [Groq](https://groq.com/)
- Visualizations by [Plotly](https://plotly.com/)

---

**Enterprise Analytics Dashboard** - Transforming data into actionable business insights with the power of AI and Machine Learning.