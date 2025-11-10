#!/bin/bash
# Setup script for Mesh Processing Assignment

echo "=========================================="
echo "Mesh Processing Assignment - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python --version

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To run the project:"
echo "  1. Run main script:    python main.py"
echo "  2. Run notebook:       jupyter notebook mesh_analysis.ipynb"
echo ""
echo "Output will be saved to: outputs/"
echo ""
