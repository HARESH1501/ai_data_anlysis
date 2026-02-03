import os
from dotenv import load_dotenv
from groq import Groq
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

load_dotenv()

class InsightGenerator:
    """Enterprise-grade AI insight generation with comprehensive analysis"""
    
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.logger = logging.getLogger(__name__)
        
    def generate_comprehensive_insights(self, df: pd.DataFrame, kpis: Dict) -> Dict[str, str]:
        """Generate comprehensive insights for all dashboard sections"""
        try:
            insights = {}
            
            # Overview insights
            insights['overview'] = self._generate_overview_insights(df, kpis)
            
            # Trend insights
            insights['trends'] = self._generate_trend_insights(df, kpis)
            
            # Performance insights
            insights['performance'] = self._generate_performance_insights(df, kpis)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {str(e)}")
            return self._fallback_insights(df, kpis)
    
    def _generate_overview_insights(self, df: pd.DataFrame, kpis: Dict) -> str:
        """Generate overview insights for the main dashboard"""
        try:
            # Prepare context
            context = self._prepare_data_context(df, kpis)
            
            prompt = f"""
            As a senior business analyst, provide executive-level insights based on this data analysis:
            
            DATA SUMMARY:
            - Total Records: {context['total_records']:,}
            - Date Range: {context['date_range']}
            - Anomaly Rate: {context['anomaly_percentage']:.1f}%
            - Trend Direction: {context['trend_direction']}
            - Key Metrics: {context['key_metrics']}
            
            Provide a concise, business-focused summary (2-3 sentences) that:
            1. Highlights the most important findings
            2. Identifies key opportunities or risks
            3. Uses executive-level language (avoid technical jargon)
            
            Focus on actionable business insights, not technical details.
            """
            
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior business analyst who explains data insights in clear, executive-level language for business decision makers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return self._fallback_overview_insight(df, kpis)
    
    def generate_anomaly_insights(self, anomalies_df: pd.DataFrame, full_df: pd.DataFrame) -> str:
        """Generate insights specifically for anomaly analysis"""
        try:
            if len(anomalies_df) == 0:
                return "✅ Excellent data quality! No significant anomalies detected in your dataset. This indicates stable, predictable business performance."
            
            # Analyze anomaly patterns
            anomaly_rate = len(anomalies_df) / len(full_df) * 100
            
            # Get time patterns
            anomalies_by_month = anomalies_df.groupby(anomalies_df['date'].dt.month).size()
            peak_month = anomalies_by_month.idxmax() if len(anomalies_by_month) > 0 else None
            
            # Get severity distribution
            severity_dist = anomalies_df['anomaly_severity'].value_counts() if 'anomaly_severity' in anomalies_df.columns else {}
            
            prompt = f"""
            Analyze these anomaly detection results for business impact:
            
            ANOMALY ANALYSIS:
            - Anomaly Rate: {anomaly_rate:.1f}% of total data
            - Total Anomalies: {len(anomalies_df):,}
            - Peak Month: {peak_month if peak_month else 'N/A'}
            - Severity Distribution: {dict(severity_dist)}
            - Date Range: {anomalies_df['date'].min()} to {anomalies_df['date'].max()}
            
            Provide business-focused insights (2-3 sentences) that:
            1. Explain what these anomalies might indicate for the business
            2. Suggest potential causes or areas to investigate
            3. Recommend immediate actions if the anomaly rate is concerning
            
            Use clear, non-technical language suitable for business stakeholders.
            """
            
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a business intelligence expert who translates anomaly detection results into actionable business insights."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=250
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"⚠️ Detected {len(anomalies_df)} anomalies ({len(anomalies_df)/len(full_df)*100:.1f}% of data). These unusual patterns may indicate data quality issues, seasonal variations, or significant business events that warrant investigation."
    
    def generate_forecast_insights(self, forecast_result: Dict) -> str:
        """Generate insights for forecasting results"""
        try:
            growth_rate = forecast_result.get('growth_rate', 0)
            accuracy = forecast_result.get('accuracy', 0)
            risk_level = forecast_result.get('risk_level', 'Medium')
            volatility = forecast_result.get('volatility', 0)
            
            prompt = f"""
            Analyze this forecasting result for strategic business planning:
            
            FORECAST ANALYSIS:
            - Predicted Growth Rate: {growth_rate:.1f}%
            - Model Accuracy: {accuracy:.1f}%
            - Risk Level: {risk_level}
            - Volatility: {volatility:.1f}%
            - Model Used: {forecast_result.get('model_used', 'N/A')}
            
            Provide strategic insights (2-3 sentences) that:
            1. Interpret what this forecast means for business planning
            2. Highlight key opportunities or risks based on the growth trend
            3. Suggest strategic actions based on the risk level and volatility
            
            Focus on strategic implications, not technical model details.
            """
            
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strategic business advisor who translates forecasting results into actionable strategic recommendations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            growth_direction = "growth" if growth_rate > 0 else "decline" if growth_rate < 0 else "stability"
            return f"📈 The forecast indicates {growth_direction} with {growth_rate:.1f}% projected change. With {accuracy:.1f}% model confidence and {risk_level.lower()} risk level, consider adjusting business strategies accordingly."
    
    def generate_executive_summary(self, df: pd.DataFrame, kpis: Dict) -> str:
        """Generate comprehensive executive summary"""
        try:
            context = self._prepare_data_context(df, kpis)
            
            prompt = f"""
            Create an executive summary for this business analytics report:
            
            BUSINESS METRICS:
            - Dataset: {context['total_records']:,} records from {context['date_range']}
            - Data Quality: {100 - context['anomaly_percentage']:.1f}% normal data
            - Business Trend: {context['trend_direction']}
            - Key Performance Indicators: {context['key_metrics']}
            
            Create a professional executive summary with:
            
            ## 📊 Key Findings
            [2-3 bullet points of most important insights]
            
            ## 🎯 Business Impact
            [What this means for business performance]
            
            ## 🚀 Recommendations
            [2-3 actionable recommendations]
            
            ## ⚠️ Risk Assessment
            [Key risks or areas requiring attention]
            
            Use professional business language suitable for C-level executives.
            """
            
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior business consultant creating executive summaries for C-level decision makers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=400
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return self._fallback_executive_summary(df, kpis)
    
    def _prepare_data_context(self, df: pd.DataFrame, kpis: Dict) -> Dict:
        """Prepare data context for AI analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'anomaly' in numeric_cols:
            numeric_cols.remove('anomaly')
        
        context = {
            'total_records': len(df),
            'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
            'anomaly_percentage': kpis.get('anomaly_percentage', 0),
            'trend_direction': kpis.get('trend_direction', 'Stable'),
            'key_metrics': {col: f"{df[col].sum():,.0f}" for col in numeric_cols[:3]}  # Top 3 metrics
        }
        
        return context
    
    def _generate_trend_insights(self, df: pd.DataFrame, kpis: Dict) -> str:
        """Generate trend-specific insights"""
        trend_direction = kpis.get('trend_direction', 'Stable')
        trend_strength = kpis.get('trend_strength', 50)
        
        if trend_direction == 'Up':
            return f"📈 Strong upward trend detected with {trend_strength:.1f}% confidence. This positive momentum suggests effective business strategies and market conditions."
        elif trend_direction == 'Down':
            return f"📉 Downward trend identified with {trend_strength:.1f}% confidence. Consider investigating underlying causes and implementing corrective measures."
        else:
            return f"📊 Stable performance pattern with {trend_strength:.1f}% confidence. Consistent results indicate predictable business operations."
    
    def _generate_performance_insights(self, df: pd.DataFrame, kpis: Dict) -> str:
        """Generate performance-specific insights"""
        anomaly_rate = kpis.get('anomaly_percentage', 0)
        
        if anomaly_rate < 5:
            return "🟢 Excellent data quality and consistent performance. Your business operations show strong stability and predictability."
        elif anomaly_rate < 15:
            return "🟡 Moderate variability detected. Some fluctuations are normal, but monitor for patterns that might indicate operational issues."
        else:
            return "🔴 High variability in performance data. Investigate potential causes such as market changes, operational issues, or data quality problems."
    
    def _fallback_insights(self, df: pd.DataFrame, kpis: Dict) -> Dict[str, str]:
        """Fallback insights when AI generation fails"""
        return {
            'overview': f"Analysis of {len(df):,} records shows {kpis.get('trend_direction', 'stable')} performance with {kpis.get('anomaly_percentage', 0):.1f}% anomaly rate.",
            'trends': f"Current trend direction is {kpis.get('trend_direction', 'stable')} based on recent data patterns.",
            'performance': f"Data quality is {'excellent' if kpis.get('anomaly_percentage', 0) < 5 else 'good' if kpis.get('anomaly_percentage', 0) < 15 else 'needs attention'} with current anomaly levels."
        }
    
    def _fallback_overview_insight(self, df: pd.DataFrame, kpis: Dict) -> str:
        """Fallback overview insight"""
        anomaly_rate = kpis.get('anomaly_percentage', 0)
        trend = kpis.get('trend_direction', 'Stable')
        
        return f"📊 Analysis of {len(df):,} records reveals {trend.lower()} performance trends with {anomaly_rate:.1f}% anomaly rate. {'Strong data quality supports reliable business insights.' if anomaly_rate < 10 else 'Consider investigating data variations for optimization opportunities.'}"
    
    def _fallback_executive_summary(self, df: pd.DataFrame, kpis: Dict) -> str:
        """Fallback executive summary"""
        return f"""
## 📊 Key Findings
• Analyzed {len(df):,} records with {kpis.get('trend_direction', 'stable')} performance trend
• Data quality shows {100 - kpis.get('anomaly_percentage', 0):.1f}% normal patterns
• Current business metrics indicate {'positive' if kpis.get('trend_direction') == 'Up' else 'stable'} trajectory

## 🎯 Business Impact
Performance data suggests {'growth opportunities' if kpis.get('trend_direction') == 'Up' else 'operational stability'} with manageable risk levels.

## 🚀 Recommendations
• Continue monitoring key performance indicators
• {'Capitalize on positive trends' if kpis.get('trend_direction') == 'Up' else 'Maintain current operational efficiency'}
• Regular data quality assessments recommended

## ⚠️ Risk Assessment
{'Low risk with positive indicators' if kpis.get('anomaly_percentage', 0) < 10 else 'Moderate risk - monitor data variations'}
        """

# Legacy function for backward compatibility
def generate_insight(summary_text: str) -> str:
    generator = InsightGenerator()
    try:
        response = generator.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional data analyst who explains insights clearly to business users."
                },
                {
                    "role": "user",
                    "content": f"Explain the following data insight in simple business language:\n{summary_text}"
                }
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Analysis shows: {summary_text}. This indicates normal business operations with some variations that may warrant further investigation."
