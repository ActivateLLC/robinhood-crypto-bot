#!/usr/bin/env python3
"""
Robinhood Crypto Bot - GreenGekko Connector
This script connects your Robinhood crypto bot with GreenGekko trading platform.
It sets up the necessary environment variables and launches GreenGekko with your
Robinhood API credentials.
"""

import os
import sys
import subprocess
import json
import dotenv
import time
import re
from pathlib import Path

# Load environment variables from .env file
dotenv.load_dotenv()

# Required environment variables
REQUIRED_ENV_VARS = [
    'ROBINHOOD_API_KEY',
    'ROBINHOOD_PRIVATE_KEY',
]

def check_env_variables():
    """Check if all required environment variables are set."""
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing_vars:
        print(f"Error: The following environment variables are missing: {', '.join(missing_vars)}")
        print("Please make sure they are set in your .env file.")
        sys.exit(1)
    print("✅ All required environment variables are set.")

def check_node_version():
    """Check Node.js version and warn if incompatible."""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True, check=True)
        version = result.stdout.strip()
        print(f"Detected Node.js version: {version}")
        
        # Extract version numbers
        match = re.match(r'v(\d+)\.(\d+)\.(\d+)', version)
        if match:
            major = int(match.group(1))
            if major >= 18:
                print("⚠️  Warning: GreenGekko may have compatibility issues with Node.js v18+")
                print("   For best results, consider using Node.js v16.x (LTS)")
                
                if major >= 20:
                    choice = input("Would you like to continue anyway? This might not work. (y/n): ")
                    if choice.lower() != 'y':
                        print("Installation cancelled. Please install Node.js v16.x and try again.")
                        print("You can use nvm to manage multiple Node.js versions:")
                        print("  nvm install 16")
                        print("  nvm use 16")
                        sys.exit(1)
    except subprocess.CalledProcessError:
        print("Error: Node.js not found. Please install Node.js before continuing.")
        sys.exit(1)

def check_greengekko_path():
    """Check if GreenGekko is installed in the expected location."""
    greengekko_path = Path('../GreenGekko')
    if not greengekko_path.exists():
        print("Error: GreenGekko directory not found at ../GreenGekko")
        print("Please make sure you've cloned GreenGekko in the parent directory.")
        sys.exit(1)
    print(f"✅ GreenGekko found at {greengekko_path.resolve()}")
    return greengekko_path

def setup_greengekko_env(greengekko_path):
    """Set up GreenGekko environment with Robinhood credentials."""
    env_file = greengekko_path / '.env'
    
    # Create or update .env file for GreenGekko
    with open(env_file, 'w') as f:
        f.write(f"ROBINHOOD_API_KEY={os.getenv('ROBINHOOD_API_KEY')}\n")
        f.write(f"ROBINHOOD_PRIVATE_KEY={os.getenv('ROBINHOOD_PRIVATE_KEY')}\n")
    
    print(f"✅ Created .env file for GreenGekko with Robinhood credentials")

def install_dependencies(greengekko_path):
    """Install GreenGekko dependencies."""
    print("Installing GreenGekko dependencies...")
    
    # First, modify package.json to use compatible versions
    try:
        # Read package.json
        package_json_path = greengekko_path / 'package.json'
        with open(package_json_path, 'r') as f:
            package_data = json.load(f)
        
        # Update problematic dependencies
        if 'dependencies' in package_data:
            # Update tulind to a version compatible with newer Node.js
            if 'tulind' in package_data['dependencies']:
                package_data['dependencies']['tulind'] = "^0.8.20"
            
            # Update other potentially problematic packages
            if 'talib' in package_data['dependencies']:
                package_data['dependencies']['talib'] = "^1.1.4"
        
        # Write updated package.json
        with open(package_json_path, 'w') as f:
            json.dump(package_data, f, indent=2)
        
        print("✅ Updated package.json with compatible dependency versions")
    except Exception as e:
        print(f"Warning: Could not update package.json: {e}")
    
    # Install dependencies with --no-optional to skip problematic packages
    try:
        subprocess.run(['npm', 'install', '--no-optional'], cwd=greengekko_path, check=True)
        print("✅ GreenGekko dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing GreenGekko dependencies: {e}")
        
        # Attempt fallback installation without problematic packages
        try:
            print("Attempting fallback installation without technical indicators...")
            subprocess.run(['npm', 'install', '--no-optional', '--no-package-lock'], cwd=greengekko_path, check=True)
            print("✅ Basic GreenGekko dependencies installed (some features may be limited)")
            
            # Create a warning file to notify about limited functionality
            with open(greengekko_path / 'LIMITED_FUNCTIONALITY.txt', 'w') as f:
                f.write("GreenGekko is running with limited functionality.\n")
                f.write("Technical indicators (tulind, talib) could not be installed.\n")
                f.write("Some strategies may not work correctly.\n")
            
        except subprocess.CalledProcessError as e2:
            print(f"Error with fallback installation: {e2}")
            print("Please try installing GreenGekko manually:")
            print(f"  cd {greengekko_path}")
            print("  npm install --no-optional")
            sys.exit(1)

def check_robinhood_adapter(greengekko_path):
    """Check if Robinhood adapter exists in GreenGekko."""
    adapter_path = greengekko_path / 'exchange' / 'wrappers' / 'robinhood.js'
    if not adapter_path.exists():
        print("Error: Robinhood adapter not found in GreenGekko")
        print("Please make sure you've added the robinhood.js file to GreenGekko/exchange/wrappers/")
        sys.exit(1)
    print("✅ Robinhood adapter found in GreenGekko")

def start_greengekko(greengekko_path, mode='paper-trader'):
    """Start GreenGekko with the specified mode."""
    valid_modes = ['backtest', 'paper-trader', 'trader']
    if mode not in valid_modes:
        print(f"Error: Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}")
        sys.exit(1)
    
    print(f"Starting GreenGekko in {mode} mode...")
    
    # Command to run GreenGekko with Robinhood config
    cmd = [
        'node', 
        'gekko.js', 
        '--config', 
        'config/robinhood-config.js',
        '--ui'
    ]
    
    if mode == 'backtest':
        cmd.append('--backtest')
    elif mode == 'paper-trader':
        cmd.append('--paper-trader')
    elif mode == 'trader':
        cmd.append('--trader')
    
    try:
        # Start GreenGekko
        process = subprocess.Popen(cmd, cwd=greengekko_path)
        print(f"✅ GreenGekko started with PID {process.pid}")
        print("GreenGekko UI will be available at http://localhost:3000")
        print("Press Ctrl+C to stop GreenGekko")
        
        # Keep the script running
        process.wait()
    except KeyboardInterrupt:
        print("\nStopping GreenGekko...")
        process.terminate()
        process.wait()
        print("GreenGekko stopped")
    except Exception as e:
        print(f"Error starting GreenGekko: {e}")
        sys.exit(1)

def main():
    """Main function."""
    print("=" * 80)
    print("Robinhood Crypto Bot - GreenGekko Connector")
    print("=" * 80)
    
    # Check Node.js version
    check_node_version()
    
    # Check environment variables
    check_env_variables()
    
    # Check GreenGekko path
    greengekko_path = check_greengekko_path()
    
    # Check Robinhood adapter
    check_robinhood_adapter(greengekko_path)
    
    # Set up GreenGekko environment
    setup_greengekko_env(greengekko_path)
    
    # Install dependencies
    install_dependencies(greengekko_path)
    
    # Ask user for mode
    print("\nChoose a mode to start GreenGekko:")
    print("1. Paper Trader (simulated trading)")
    print("2. Backtest (test strategy on historical data)")
    print("3. Live Trader (real trading with real money)")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        mode = 'paper-trader'
    elif choice == '2':
        mode = 'backtest'
    elif choice == '3':
        confirm = input("WARNING: Live trading will use real money. Are you sure? (y/n): ")
        if confirm.lower() != 'y':
            print("Live trading cancelled")
            sys.exit(0)
        mode = 'trader'
    else:
        print("Invalid choice. Defaulting to paper-trader mode.")
        mode = 'paper-trader'
    
    # Start GreenGekko
    start_greengekko(greengekko_path, mode)

if __name__ == "__main__":
    main()
