"""
Production startup script for Enterprise Analytics Dashboard
Handles environment setup and application launch
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment variables and configuration"""
    logger.info("Setting up environment...")
    
    # Load environment variables from .env if it exists
    env_file = Path('.env')
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("Loaded environment variables from .env file")
    
    # Check required environment variables
    required_vars = ['GROQ_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.info("AI insights will use fallback mode")
    else:
        logger.info("All environment variables configured")
    
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Run: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies available")
    return True

def run_streamlit_app():
    """Launch the Streamlit application"""
    logger.info("Starting Enterprise Analytics Dashboard...")
    
    # Streamlit configuration
    config_args = [
        '--server.port=8501',
        '--server.address=0.0.0.0',
        '--server.headless=false',
        '--browser.gatherUsageStats=false',
        '--server.maxUploadSize=200'
    ]
    
    # Build command
    cmd = [sys.executable, '-m', 'streamlit', 'run', 'app.py'] + config_args
    
    try:
        # Launch Streamlit
        logger.info("Dashboard starting at http://localhost:8501")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🚀 Enterprise Analytics Dashboard")
    print("=" * 50)
    
    # Setup environment
    if not setup_environment():
        logger.error("Environment setup failed")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed")
        sys.exit(1)
    
    # Run application
    logger.info("All checks passed. Starting dashboard...")
    run_streamlit_app()

if __name__ == "__main__":
    main()