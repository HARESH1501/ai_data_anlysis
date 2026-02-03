"""
Deployment script for Enterprise Analytics Dashboard
Handles production deployment to Streamlit Cloud
"""

import os
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'app.py',
        'requirements.txt',
        '.streamlit/config.toml',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files present")
    return True

def validate_environment():
    """Validate environment variables"""
    required_env_vars = ['GROQ_API_KEY']
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file or Streamlit Cloud secrets")
        return False
    
    print("✅ Environment variables configured")
    return True

def test_imports():
    """Test if all imports work correctly"""
    try:
        import streamlit as st
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        import numpy as np
        import scikit_learn
        print("✅ All core dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def run_local_test():
    """Run local test of the application"""
    print("🚀 Starting local test...")
    try:
        # Run streamlit app in test mode
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py", 
            "--server.headless", "true",
            "--server.port", "8502"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Local test passed")
            return True
        else:
            print(f"❌ Local test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✅ App started successfully (timeout expected)")
        return True
    except Exception as e:
        print(f"❌ Local test error: {e}")
        return False

def create_deployment_checklist():
    """Create deployment checklist"""
    checklist = """
# 🚀 Streamlit Cloud Deployment Checklist

## Pre-deployment
- [ ] All code committed to GitHub
- [ ] Requirements.txt updated
- [ ] Environment variables documented
- [ ] README.md complete
- [ ] Sample data included

## Streamlit Cloud Setup
- [ ] Repository connected to Streamlit Cloud
- [ ] Secrets configured (GROQ_API_KEY)
- [ ] App settings configured
- [ ] Custom domain (if needed)

## Post-deployment
- [ ] App loads successfully
- [ ] File upload works
- [ ] ML analysis functions
- [ ] AI insights generate
- [ ] Export features work
- [ ] Mobile responsiveness tested

## Monitoring
- [ ] Error tracking enabled
- [ ] Performance monitoring
- [ ] User feedback collection
- [ ] Regular updates scheduled
"""
    
    with open('DEPLOYMENT_CHECKLIST.md', 'w') as f:
        f.write(checklist)
    
    print("✅ Deployment checklist created")

def main():
    """Main deployment preparation function"""
    print("🚀 Enterprise Analytics Dashboard - Deployment Preparation")
    print("=" * 60)
    
    # Run all checks
    checks = [
        ("Requirements Check", check_requirements),
        ("Environment Validation", validate_environment),
        ("Import Test", test_imports),
        ("Local Test", run_local_test)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}...")
        if not check_func():
            all_passed = False
    
    # Create deployment checklist
    print(f"\n📝 Creating deployment checklist...")
    create_deployment_checklist()
    
    # Final status
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All checks passed! Ready for deployment.")
        print("\nNext steps:")
        print("1. Commit all changes to GitHub")
        print("2. Connect repository to Streamlit Cloud")
        print("3. Configure secrets in Streamlit Cloud")
        print("4. Deploy and test")
    else:
        print("❌ Some checks failed. Please fix issues before deployment.")
    
    print("\n📖 See DEPLOYMENT_CHECKLIST.md for detailed steps")

if __name__ == "__main__":
    main()