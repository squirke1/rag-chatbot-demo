#!/bin/bash
# Quick test script for RAG evaluation system

echo "=================================="
echo "RAG Evaluation System Quick Test"
echo "=================================="
echo ""

# Check if scikit-learn is installed
python3 -c "import sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  scikit-learn not found. Installing..."
    pip install scikit-learn
    echo ""
fi

# Step 1: Re-ingest documents with new comprehensive guide
echo "Step 1: Ingesting documents..."
echo "-----------------------------------"
python3 src/ingest.py --data-dir data
echo ""

# Step 2: Test with single question
echo "Step 2: Testing with single question..."
echo "-----------------------------------"
python3 src/eval.py --question "What is RAG?" --method similarity
echo ""

# Step 3: Run full dataset evaluation
echo "Step 3: Running full dataset evaluation..."
echo "-----------------------------------"
python3 src/eval.py --dataset data/test_questions.json --method similarity --output evaluation_results.json
echo ""

echo "✅ Evaluation complete!"
echo ""
echo "Results saved to: evaluation_results.json"
echo "To view results: cat evaluation_results.json | python3 -m json.tool"
