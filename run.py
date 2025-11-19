"""
SmartLabel - Quick Start Script
Run this to launch the complete system
"""
import subprocess
import sys
import os

def main():
    print("="*80)
    print("SMARTLABEL - AUTONOMOUS TRANSACTION CATEGORIZER")
    print("GHCI Round 2 Hackathon Submission")
    print("="*80)
    print()
    
    # Check if model exists
    if not os.path.exists('models/classifier.pkl'):
        print("⚠️  Model not found. Training model first...")
        print("This may take a few minutes...\n")
        
        try:
            subprocess.run([sys.executable, 'main.py'], check=True)
        except subprocess.CalledProcessError:
            print("\n❌ Error during training. Please check the logs above.")
            return
    
    print("\n" + "="*80)
    print("🚀 LAUNCHING WEB INTERFACE")
    print("="*80)
    print()
    
    # Install Flask if needed
    try:
        import flask
    except ImportError:
        print("Installing Flask...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'flask'], check=True)
    
    # Run the app
    subprocess.run([sys.executable, 'app.py'])

if __name__ == "__main__":
    main()
