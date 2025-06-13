#!/bin/bash

echo "🛠 Setting up your Python environment..."

# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install required packages
pip install -r requirements.txt

echo "✅ Setup complete. To activate your environment later, run:"
echo "source venv/bin/activate"
