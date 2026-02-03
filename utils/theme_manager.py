"""
Theme Manager for Enterprise Analytics Dashboard
Handles UI theming and styling
"""

import streamlit as st
from typing import Dict, Any

class ThemeManager:
    """Manage dashboard themes and styling"""
    
    def __init__(self):
        self.themes = {
            'dark': {
                'primary_color': '#1f77b4',
                'background_color': '#0e1117',
                'secondary_background_color': '#262730',
                'text_color': '#fafafa',
                'font': 'sans serif'
            },
            'light': {
                'primary_color': '#1f77b4',
                'background_color': '#ffffff',
                'secondary_background_color': '#f0f2f6',
                'text_color': '#262730',
                'font': 'sans serif'
            }
        }
    
    def apply_theme(self, theme_name: str):
        """Apply the selected theme to the dashboard"""
        if theme_name not in self.themes:
            theme_name = 'dark'  # Default fallback
        
        theme = self.themes[theme_name]
        
        # Apply custom CSS
        css = self._generate_css(theme)
        st.markdown(css, unsafe_allow_html=True)
    
    def _generate_css(self, theme: Dict[str, str]) -> str:
        """Generate CSS for the selected theme"""
        return f"""
        <style>
        /* Main container styling */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        
        /* Metric cards styling */
        [data-testid="metric-container"] {{
            background-color: {theme['secondary_background_color']};
            border: 1px solid rgba(49, 51, 63, 0.2);
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        /* Sidebar styling */
        .css-1d391kg {{
            background-color: {theme['secondary_background_color']};
        }}
        
        /* Header styling */
        h1, h2, h3 {{
            color: {theme['text_color']};
            font-family: {theme['font']};
        }}
        
        /* Custom metric styling */
        .metric-card {{
            background: linear-gradient(135deg, {theme['primary_color']}22, {theme['secondary_background_color']});
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid {theme['primary_color']};
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .metric-title {{
            font-size: 0.9rem;
            color: {theme['text_color']}aa;
            margin-bottom: 0.5rem;
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: {theme['primary_color']};
            margin-bottom: 0.25rem;
        }}
        
        .metric-delta {{
            font-size: 0.8rem;
            color: {theme['text_color']}cc;
        }}
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
            background-color: {theme['secondary_background_color']};
            border-radius: 5px 5px 0 0;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {theme['primary_color']};
            color: white;
        }}
        
        /* Button styling */
        .stButton > button {{
            background-color: {theme['primary_color']};
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            background-color: {theme['primary_color']}dd;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        /* Selectbox styling */
        .stSelectbox > div > div {{
            background-color: {theme['secondary_background_color']};
            border-color: {theme['primary_color']}44;
        }}
        
        /* File uploader styling */
        .stFileUploader > div {{
            background-color: {theme['secondary_background_color']};
            border: 2px dashed {theme['primary_color']}44;
            border-radius: 10px;
        }}
        
        /* Info/success/warning boxes */
        .stAlert {{
            border-radius: 5px;
        }}
        
        /* Dataframe styling */
        .stDataFrame {{
            border-radius: 5px;
            overflow: hidden;
        }}
        
        /* Progress bar */
        .stProgress > div > div {{
            background-color: {theme['primary_color']};
        }}
        
        /* Spinner */
        .stSpinner > div {{
            border-top-color: {theme['primary_color']};
        }}
        
        /* Custom animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .fade-in {{
            animation: fadeIn 0.5s ease-out;
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .main .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}
            
            .metric-value {{
                font-size: 1.5rem;
            }}
        }}
        </style>
        """
    
    def get_color_palette(self, theme_name: str = 'dark') -> Dict[str, str]:
        """Get color palette for charts and visualizations"""
        if theme_name == 'dark':
            return {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'success': '#2ca02c',
                'warning': '#d62728',
                'info': '#9467bd',
                'background': '#0e1117',
                'surface': '#262730',
                'text': '#fafafa'
            }
        else:  # light theme
            return {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'success': '#2ca02c',
                'warning': '#d62728',
                'info': '#9467bd',
                'background': '#ffffff',
                'surface': '#f0f2f6',
                'text': '#262730'
            }
    
    def create_metric_card_html(self, title: str, value: str, delta: str = "", 
                               theme_name: str = 'dark') -> str:
        """Create custom HTML metric card"""
        colors = self.get_color_palette(theme_name)
        
        return f"""
        <div class="metric-card fade-in">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-delta">{delta}</div>' if delta else ''}
        </div>
        """
    
    def create_status_indicator(self, status: str, theme_name: str = 'dark') -> str:
        """Create status indicator with appropriate colors"""
        colors = self.get_color_palette(theme_name)
        
        status_colors = {
            'excellent': colors['success'],
            'good': colors['info'],
            'warning': colors['secondary'],
            'critical': colors['warning']
        }
        
        color = status_colors.get(status.lower(), colors['primary'])
        
        return f"""
        <div style="
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background-color: {color}22;
            color: {color};
            border: 1px solid {color}44;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 500;
        ">
            {status.title()}
        </div>
        """