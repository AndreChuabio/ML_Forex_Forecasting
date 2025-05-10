#!/usr/bin/env python
"""
Directory Setup Script

This script creates all necessary directories for our forex forecasting project. Please run this before using any other scripts to ensure the required folder exists.
"""

import os
import sys
from datetime import datetime

# Define color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

# def print_status(message,status='info'):
# """print a formatted status message"""
# timestamp = datetime.now().strftime("%H:%M:%S")

# if status == 'success':
# print(f"{GREEN}  {timestamp} - {message}{RESET}")
# elif status == 'warning':
#    print(f"{YELLOW}  {timestamp} - {message}{RESET}")
# elif status == "error":
#   print(f"{RED}  {timestamp} - {message}{RESET}")
# else:
# print(f"[*] {timestamp} - {message}")


def print_status(message, status_type="info"):

    colors = {
        'info': "\033[94m",  # blue
        'success': "\033[92m",  # green
        'warning': "\033[93m",  # yellow
        'error': "\033[91m",  # red
        'reset': "\033[0m"  # reset

    }
    timestamp = datetime.now().strftime("%H:%M:%S")
    status_color = colors.get(status_type.lower(), colors['info'])

    print(
        f"{status_color}[{timestamp}]{message}{colors['reset']}", file=sys.stderr)


def create_directories():
    directories = [
        "output",
        "models",
        "results",
        "model_info",
        "results/forecasts",
        "results/analysis",
        "results/metrics",
        "docs/images",
        "backup"
    ]

    # create directories if they dont exist

    created_count = 0
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print_status(f"Created directory: {directory}", "success")
                created_count += 1
            except Exception as e:
                print_status(
                    f"Created {created_count} directories successfully.", "success")
        else:
            print_status("All directories already exist", "Warning")


def main():
    print(f"\n{BOLD}=== Forex Forecasting- Directory Setup==={RESET}\n")

    try:
        create_directories()
        print_status(
            "Directory setup complete!, project is ready to use", "Success")
        return 0
    except Exception as e:
        print_status(f"Unexpected error occured: {str(e)}", "error")
        return 1

    if __name__ == "__main__":
        sys.exit(main())
