import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import the new components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.preprocess import DataProcessor
from ml_engine.anomaly import AnomalyDetector
from ml_engine.forecasting import TimeSeriesForecaster
from genai_engine.insight_generator import InsightGenerator
from dashboard.kpi_calculator import KPICalculator
from dashboard.visualization_engine import VisualizationEngine

st.set_page_config(
    page_title="Enterprise Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Enterprise Analytics Dashboard")
st.caption("Professional Data Analytics | Power BI–Style View | ML & AI Enhanced")

# Initialize components
@st.cache_resource
def get_components():
    return {
        'processor': DataProcessor(),
        'anomaly_detector': AnomalyDetector(),
        'forecaster': TimeSeriesForecaster(),
        'insight_generator': InsightGenerator(),
        'kpi_calculator': KPICalculator(),
        'viz_engine': VisualizationEngine()
    }

components = get_components()

# Sidebar
with st.sidebar:
    st.header("🎛️ Dashboard Controls")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload CSV File",
        type=['csv'],
        help="Upload your business data for analysis"
    )
    
    # Sample data option
    if st.button("📊 Use Sample Data"):
        try:
            sample_df = pd.read_csv("../data/sample_sales.csv")
            st.session_state['data'] = sample_df
            st.success("✅ Sample data loaded!")
        except Exception as e:
            st.error(f"❌ Error loading sample data: {str(e)}")

# Main content
if uploaded_file is not None:
    try:
        # Process uploaded file
        with st.spinner("🔄 Processing your data..."):
            data = components['processor'].load_csv(uploaded_file)
            st.session_state['data'] = data
        
        st.success(f"✅ Successfully loaded {len(data):,} records")
        
        # Show data preview
        with st.expander("🔍 Data Preview", expanded=False):
            st.dataframe(data.head(10), use_container_width=True)
        
        # Process with ML and AI
        with st.spinner("🤖 Running ML analysis and generating AI insights..."):
            # Detect anomalies
            data_with_anomalies = components['anomaly_detector'].detect(data)
            
            # Calculate KPIs
            kpis = components['kpi_calculator'].calculate_all_kpis(data_with_anomalies)
            
            # Generate insights
            insights = components['insight_generator'].generate_comprehensive_insights(data_with_anomalies, kpis)
        
        # Dashboard tabs
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔮 Forecast", "🚨 Anomalies"])
        
        with tab1:
            st.subheader("📈 Key Performance Indicators")
            
            # KPI Cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total Records", f"{kpis['total_records']:,}")
                st.metric("💰 Total Value", f"{kpis.get('total_value', 0):,.0f}")
            
            with col2:
                st.metric("📈 Average Value", f"{kpis.get('average_value', 0):.2f}")
                st.metric("⬆️ Maximum Value", f"{kpis.get('max_value', 0):,.0f}")
            
            with col3:
                st.metric("🚨 Anomalies", f"{kpis['anomaly_count']:,}")
                st.metric("📊 Anomaly Rate", f"{kpis['anomaly_percentage']:.1f}%")
            
            with col4:
                st.metric("📈 Trend", kpis['trend_direction'])
                st.metric("🎯 Confidence", f"{kpis['trend_strength']:.1f}%")
            
            st.divider()
            
            # Visualizations
            numeric_cols = data_with_anomalies.select_dtypes(include=['number']).columns.tolist()
            if 'anomaly' in numeric_cols:
                numeric_cols.remove('anomaly')
            
            if numeric_cols:
                selected_metric = st.selectbox("📈 Select Metric for Analysis", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Time series chart
                    fig = components['viz_engine'].create_time_series_chart(data_with_anomalies, selected_metric)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Anomaly distribution
                    fig = components['viz_engine'].create_anomaly_pie_chart(data_with_anomalies)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Distribution and monthly charts
                col3, col4 = st.columns(2)
                
                with col3:
                    fig = components['viz_engine'].create_distribution_chart(data_with_anomalies, selected_metric)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col4:
                    fig = components['viz_engine'].create_monthly_bar_chart(data_with_anomalies, selected_metric)
                    st.plotly_chart(fig, use_container_width=True)
            
            # AI Insights
            st.subheader("🧠 AI-Generated Business Insights")
            st.info(insights['overview'])
        
        with tab2:
            st.subheader("🔮 Time Series Forecasting")
            
            if numeric_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    forecast_metric = st.selectbox("📈 Metric to Forecast", numeric_cols, key="forecast_metric")
                
                with col2:
                    forecast_days = st.selectbox("📅 Forecast Period", [7, 14, 30], index=0)
                
                if st.button("🚀 Generate Forecast"):
                    with st.spinner("🔄 Generating forecast..."):
                        forecast_result = components['forecaster'].forecast(
                            data_with_anomalies, forecast_metric, forecast_days
                        )
                        
                        # Display forecast chart
                        fig = components['viz_engine'].create_forecast_chart(
                            data_with_anomalies, forecast_result, forecast_metric
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("📈 Growth Rate", f"{forecast_result['growth_rate']:.1f}%")
                        
                        with col2:
                            st.metric("🎯 Accuracy", f"{forecast_result['accuracy']:.1f}%")
                        
                        with col3:
                            st.metric("⚠️ Risk Level", forecast_result['risk_level'])
                        
                        # AI insights for forecast
                        st.subheader("🧠 Forecast Insights")
                        forecast_insights = components['insight_generator'].generate_forecast_insights(forecast_result)
                        st.info(forecast_insights)
        
        with tab3:
            st.subheader("🚨 Anomaly Detection Results")
            
            anomalies = data_with_anomalies[data_with_anomalies['anomaly'] == -1]
            
            # Anomaly summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🚨 Total Anomalies", len(anomalies))
            
            with col2:
                st.metric("📊 Anomaly Rate", f"{(len(anomalies)/len(data_with_anomalies)*100):.1f}%")
            
            with col3:
                if len(anomalies) > 0:
                    st.metric("📅 Latest Anomaly", anomalies['date'].max().strftime('%Y-%m-%d'))
                else:
                    st.metric("📅 Latest Anomaly", "None")
            
            if len(anomalies) > 0:
                # Anomaly timeline
                fig = components['viz_engine'].create_anomaly_timeline(data_with_anomalies)
                st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly details
                st.subheader("📋 Anomaly Details")
                st.dataframe(anomalies.head(20), use_container_width=True)
                
                # AI insights for anomalies
                st.subheader("🧠 Anomaly Analysis")
                anomaly_insights = components['insight_generator'].generate_anomaly_insights(anomalies, data_with_anomalies)
                st.warning(anomaly_insights)
            else:
                st.success("✅ Excellent! No anomalies detected in your data.")
                st.balloons()
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file contains a date column and at least one numeric column.")

elif 'data' in st.session_state:
    # Use previously loaded data
    st.info("Using previously loaded data. Upload a new file to analyze different data.")

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 3rem 0;'>
        <h2>🚀 Welcome to Enterprise Analytics</h2>
        <p style='font-size: 1.2rem; color: #666; margin: 2rem 0;'>
            Upload your business data to unlock powerful insights with Machine Learning and AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Power BI Style Interface
        - Interactive KPI cards
        - Professional visualizations
        - Real-time data updates
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Machine Learning
        - Automated anomaly detection
        - Time-series forecasting
        - Trend analysis
        """)
    
    with col3:
        st.markdown("""
        ### 🧠 Generative AI
        - Natural language insights
        - Business recommendations
        - Executive summaries
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Enterprise Analytics Dashboard | Powered by Streamlit, ML & AI"
    "</div>", 
    unsafe_allow_html=True
)
