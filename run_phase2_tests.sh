#!/bin/bash
# Phase 2 Test Runner Script
# 
# This script runs the test validation for Phase 2: Loss Module Foundation

set -e  # Exit on error

echo "======================================"
echo "Phase 2: Loss Module Foundation Tests"
echo "======================================"

# Set up Python path to include Ludwig
export PYTHONPATH=".:$PYTHONPATH"

echo "Running validation script..."
python test_phase2_validation.py

echo ""
echo "Running unit tests for loss utilities..."
python -m pytest tests/ludwig/utils/test_loss_utils.py -v

echo ""
echo "Running unit tests for loss modules..."
python -m pytest tests/ludwig/modules/test_loss_modules.py::TestFeatureTensorSupport -v
python -m pytest tests/ludwig/modules/test_loss_modules.py::TestLossModuleRegistry -v

echo ""
echo "======================================"
echo "Phase 2 Testing Complete"
echo "======================================"