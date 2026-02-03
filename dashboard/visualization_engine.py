"""
Visualization Engine for Enterprise Analytics Dashboard
Creates professional Power BI-style charts and visualizations
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

class VisualizationEngine:
    """Create professional visualizations for the analytics dashboard"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'anomaly': '#d62728',
            'normal': '#2ca02c'
        }
        
    def create_time_series_chart(self, df: pd.DataFrame, metric_column: str, max_points: int = 1000) -> go.Figure:
        """Create interactive time series chart with anomaly highlighting - optimized for large data"""
        try:
            # Optimize data for visualization
            if len(df) > max_points:
                # Intelligent downsampling
                df_viz = self._downsample_timeseries(df, max_points)
                show_downsampling_note = True
            else:
                df_viz = df
                show_downsampling_note = False
            
            fig = go.Figure()
            
            # Separate normal and anomaly data
            if 'anomaly' in df_viz.columns:
                normal_data = df_viz[df_viz['anomaly'] == 1]
                anomaly_data = df_viz[df_viz['anomaly'] == -1]
                
                # Add normal data line
                fig.add_trace(go.Scatter(
                    x=normal_data['date'],
                    y=normal_data[metric_column],
                    mode='lines+markers',
                    name='Normal Data',
                    line=dict(color=self.color_palette['primary'], width=2),
                    marker=dict(size=3),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> %{y:,.0f}<extra></extra>'
                ))
                
                # Add anomaly points
                if len(anomaly_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=anomaly_data['date'],
                        y=anomaly_data[metric_column],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(
                            color=self.color_palette['anomaly'],
                            size=6,
                            symbol='diamond',
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate='<b>⚠️ Anomaly</b><br><b>Date:</b> %{x}<br><b>Value:</b> %{y:,.0f}<extra></extra>'
                    ))
            else:
                # Simple line chart without anomaly detection
                fig.add_trace(go.Scatter(
                    x=df_viz['date'],
                    y=df_viz[metric_column],
                    mode='lines+markers',
                    name=metric_column.title(),
                    line=dict(color=self.color_palette['primary'], width=2),
                    marker=dict(size=3),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> %{y:,.0f}<extra></extra>'
                ))
            
            # Update layout
            title = f'📈 {metric_column.title()} Trend Analysis'
            if show_downsampling_note:
                title += f' (Showing {len(df_viz):,} of {len(df):,} points)'
            
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title=metric_column.title(),
                hovermode='x unified',
                showlegend=True,
                height=400,
                template='plotly_white',
                font=dict(size=12)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating time series chart: {str(e)}")
            return self._create_fallback_chart("Time Series Chart Error")
    
    def _downsample_timeseries(self, df: pd.DataFrame, max_points: int) -> pd.DataFrame:
        """Intelligent downsampling for time series data"""
        if len(df) <= max_points:
            return df
        
        # Sort by date
        df_sorted = df.sort_values('date')
        
        # Calculate step size
        step = len(df_sorted) // max_points
        
        # Take every nth point, but always include anomalies
        downsampled_indices = list(range(0, len(df_sorted), step))
        
        # Add anomaly indices if they exist
        if 'anomaly' in df.columns:
            anomaly_indices = df_sorted[df_sorted['anomaly'] == -1].index.tolist()
            downsampled_indices.extend(anomaly_indices)
        
        # Remove duplicates and sort
        downsampled_indices = sorted(list(set(downsampled_indices)))
        
        # Ensure we don't exceed max_points
        if len(downsampled_indices) > max_points:
            downsampled_indices = downsampled_indices[:max_points]
        
        return df_sorted.iloc[downsampled_indices].reset_index(drop=True)
    
    def create_anomaly_pie_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create pie chart showing normal vs anomaly distribution"""
        try:
            if 'anomaly' not in df.columns:
                return self._create_fallback_chart("No Anomaly Data Available")
            
            # Count normal vs anomaly
            anomaly_counts = df['anomaly'].value_counts()
            normal_count = anomaly_counts.get(1, 0)
            anomaly_count = anomaly_counts.get(-1, 0)
            
            labels = ['Normal Data', 'Anomalies']
            values = [normal_count, anomaly_count]
            colors = [self.color_palette['normal'], self.color_palette['anomaly']]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker_colors=colors,
                textinfo='label+percent+value',
                textfont_size=12,
                hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
            )])
            
            fig.update_layout(
                title='🥧 Data Quality Distribution',
                height=400,
                template='plotly_white',
                font=dict(size=12),
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating pie chart: {str(e)}")
            return self._create_fallback_chart("Pie Chart Error")
    
    def create_monthly_bar_chart(self, df: pd.DataFrame, metric_column: str) -> go.Figure:
        """Create monthly aggregation bar chart"""
        try:
            # Group by month
            monthly_data = df.groupby(df['date'].dt.to_period('M'))[metric_column].agg(['sum', 'mean', 'count']).reset_index()
            monthly_data['date'] = monthly_data['date'].astype(str)
            
            fig = go.Figure()
            
            # Add total bar
            fig.add_trace(go.Bar(
                x=monthly_data['date'],
                y=monthly_data['sum'],
                name='Monthly Total',
                marker_color=self.color_palette['primary'],
                hovertemplate='<b>Month:</b> %{x}<br><b>Total:</b> %{y:,.0f}<br><b>Records:</b> %{customdata}<extra></extra>',
                customdata=monthly_data['count']
            ))
            
            # Add average line
            fig.add_trace(go.Scatter(
                x=monthly_data['date'],
                y=monthly_data['mean'],
                mode='lines+markers',
                name='Monthly Average',
                yaxis='y2',
                line=dict(color=self.color_palette['secondary'], width=3),
                marker=dict(size=6),
                hovertemplate='<b>Month:</b> %{x}<br><b>Average:</b> %{y:,.2f}<extra></extra>'
            ))
            
            # Update layout with secondary y-axis
            fig.update_layout(
                title=f'📊 Monthly {metric_column.title()} Performance',
                xaxis_title='Month',
                yaxis=dict(title='Total Value', side='left'),
                yaxis2=dict(title='Average Value', side='right', overlaying='y'),
                height=400,
                template='plotly_white',
                font=dict(size=12),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating monthly bar chart: {str(e)}")
            return self._create_fallback_chart("Monthly Chart Error")
    
    def create_distribution_chart(self, df: pd.DataFrame, metric_column: str) -> go.Figure:
        """Create histogram showing value distribution"""
        try:
            fig = go.Figure()
            
            # Create histogram
            fig.add_trace(go.Histogram(
                x=df[metric_column],
                nbinsx=30,
                name='Distribution',
                marker_color=self.color_palette['info'],
                opacity=0.7,
                hovertemplate='<b>Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
            ))
            
            # Add mean line
            mean_value = df[metric_column].mean()
            fig.add_vline(
                x=mean_value,
                line_dash="dash",
                line_color=self.color_palette['warning'],
                annotation_text=f"Mean: {mean_value:,.0f}",
                annotation_position="top"
            )
            
            # Add median line
            median_value = df[metric_column].median()
            fig.add_vline(
                x=median_value,
                line_dash="dot",
                line_color=self.color_palette['success'],
                annotation_text=f"Median: {median_value:,.0f}",
                annotation_position="bottom"
            )
            
            fig.update_layout(
                title=f'📊 {metric_column.title()} Distribution',
                xaxis_title=metric_column.title(),
                yaxis_title='Frequency',
                height=400,
                template='plotly_white',
                font=dict(size=12),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating distribution chart: {str(e)}")
            return self._create_fallback_chart("Distribution Chart Error")
    
    def create_forecast_chart(self, historical_df: pd.DataFrame, 
                            forecast_result: Dict, metric_column: str) -> go.Figure:
        """Create forecast visualization with confidence intervals"""
        try:
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_df['date'],
                y=historical_df[metric_column],
                mode='lines+markers',
                name='Historical Data',
                line=dict(color=self.color_palette['primary'], width=2),
                marker=dict(size=4)
            ))
            
            # Forecast data
            forecast_df = forecast_result['forecast_data']
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
                y=pd.concat([forecast_df['upper_bound'], forecast_df['lower_bound'][::-1]]),
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                hoverinfo='skip'
            ))
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['predicted_value'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color=self.color_palette['secondary'], width=3, dash='dash'),
                marker=dict(size=6),
                hovertemplate='<b>Date:</b> %{x}<br><b>Predicted:</b> %{y:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'🔮 {metric_column.title()} Forecast ({forecast_result["model_used"]})',
                xaxis_title='Date',
                yaxis_title=metric_column.title(),
                height=500,
                template='plotly_white',
                font=dict(size=12),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating forecast chart: {str(e)}")
            return self._create_fallback_chart("Forecast Chart Error")
    
    def create_anomaly_timeline(self, df: pd.DataFrame) -> go.Figure:
        """Create timeline showing anomaly occurrences"""
        try:
            if 'anomaly' not in df.columns:
                return self._create_fallback_chart("No Anomaly Data Available")
            
            # Get anomalies over time
            daily_anomalies = df.groupby(df['date'].dt.date).agg({
                'anomaly': lambda x: (x == -1).sum()
            }).reset_index()
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=daily_anomalies['date'],
                y=daily_anomalies['anomaly'],
                name='Daily Anomalies',
                marker_color=self.color_palette['anomaly'],
                hovertemplate='<b>Date:</b> %{x}<br><b>Anomalies:</b> %{y}<extra></extra>'
            ))
            
            fig.update_layout(
                title='🚨 Anomaly Timeline',
                xaxis_title='Date',
                yaxis_title='Number of Anomalies',
                height=400,
                template='plotly_white',
                font=dict(size=12)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating anomaly timeline: {str(e)}")
            return self._create_fallback_chart("Anomaly Timeline Error")
    
    def create_anomaly_severity_chart(self, anomalies_df: pd.DataFrame) -> go.Figure:
        """Create chart showing anomaly severity distribution"""
        try:
            if 'anomaly_severity' not in anomalies_df.columns:
                return self._create_fallback_chart("No Severity Data Available")
            
            severity_counts = anomalies_df['anomaly_severity'].value_counts()
            
            # Define colors for severity levels
            severity_colors = {
                'Critical': '#8B0000',
                'High': '#DC143C',
                'Medium': '#FF8C00',
                'Low': '#FFD700'
            }
            
            colors = [severity_colors.get(level, self.color_palette['primary']) for level in severity_counts.index]
            
            fig = go.Figure(data=[go.Bar(
                x=severity_counts.index,
                y=severity_counts.values,
                marker_color=colors,
                hovertemplate='<b>Severity:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
            )])
            
            fig.update_layout(
                title='⚠️ Anomaly Severity Distribution',
                xaxis_title='Severity Level',
                yaxis_title='Count',
                height=400,
                template='plotly_white',
                font=dict(size=12)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating severity chart: {str(e)}")
            return self._create_fallback_chart("Severity Chart Error")
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap for numeric columns"""
        try:
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove system columns
            exclude_cols = ['anomaly', 'year', 'month', 'day', 'weekday', 'quarter', 'week_of_year']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if len(numeric_cols) < 2:
                return self._create_fallback_chart("Insufficient Numeric Columns for Correlation")
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='🔗 Correlation Matrix',
                height=500,
                template='plotly_white',
                font=dict(size=12)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {str(e)}")
            return self._create_fallback_chart("Correlation Heatmap Error")
    
    def _create_fallback_chart(self, error_message: str) -> go.Figure:
        """Create a fallback chart when visualization fails"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"⚠️ {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            font=dict(size=16, color="red"),
            showarrow=False
        )
        
        fig.update_layout(
            title="Chart Error",
            height=400,
            template='plotly_white',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig