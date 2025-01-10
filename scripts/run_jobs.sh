#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Usage: $0 <script_directory>"
  exit 1
fi

SCRIPT_DIR="$1"

if [ ! -d "$SCRIPT_DIR" ]; then
  echo "Error: Directory $SCRIPT_DIR does not exist."
  exit 1
fi

for script in "$SCRIPT_DIR"/*.sh; do
  if [ -f "$script" ]; then
    echo "Running $script..."
    bash "$script" || { echo "Error running $script. Exiting."; exit 1; }
  fi
done

echo "All scripts in $SCRIPT_DIR executed successfully."
