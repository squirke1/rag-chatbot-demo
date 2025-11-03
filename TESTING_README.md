# Testing Your RAG Evaluation System

## Quick Start (3 Steps)

### 1. Install the new dependency:
```bash
pip install scikit-learn
```

### 2. Run the automated test:
```bash
./test_evaluation.sh
```

This will:
- ✅ Ingest the new comprehensive RAG guide document
- ✅ Test with a single question
- ✅ Run full evaluation on 10 test questions
- ✅ Generate a results JSON file

### 3. View results:
```bash
cat evaluation_results.json | python3 -m json.tool
```

---

## What Was Added

### New Documents
- **`data/rag_comprehensive_guide.txt`** - 2000+ word comprehensive guide about RAG systems
  - Covers RAG basics, components, strategies, evaluation, best practices
  - Perfect for testing the evaluation system

### Updated Test Dataset
- **`data/test_questions.json`** - 10 questions with ground truth answers
  - Questions specifically designed to match the comprehensive guide
  - Tests various aspects: definitions, processes, comparisons, recommendations

### Documentation
- **`EVALUATION_GUIDE.md`** - Complete guide covering:
  - How to run evaluations (single question and dataset)
  - How to add your own documents
  - Example datasets for different domains
  - Metric explanations
  - Troubleshooting tips

### Test Script  
- **`test_evaluation.sh`** - Automated test runner
  - One command to run full evaluation
  - Auto-installs dependencies
  - Generates results file

---

## Manual Testing

### Test Single Question:
```bash
python3 src/eval.py \
  --question "What is RAG?" \
  --method similarity
```

### Test with Ground Truth:
```bash
python3 src/eval.py \
  --question "What is MMR?" \
  --ground-truth "MMR balances relevance and diversity in retrieval" \
  --method similarity
```

### Test Full Dataset:
```bash
python3 src/eval.py \
  --dataset data/test_questions.json \
  --method similarity \
  --output results.json
```

### Compare Methods:
```bash
# Similarity search
python3 src/eval.py --dataset data/test_questions.json --method similarity --output results_sim.json

# MMR search  
python3 src/eval.py --dataset data/test_questions.json --method mmr --output results_mmr.json
```

---

## Understanding Your Results

### Good Scores (>0.7):
- ✅ System is working well
- ✅ Retrieving relevant context
- ✅ Generating accurate answers

### Medium Scores (0.5-0.7):
- ⚠️ Acceptable but room for improvement
- Consider tuning chunk size or retrieval parameters

### Low Scores (<0.5):
- ❌ Something needs fixing
- Check if documents contain relevant information
- Verify vector index created correctly
- Review retrieval method and parameters

---

## Next Steps

1. **Run the test**: `./test_evaluation.sh`
2. **Review metrics**: Check evaluation_results.json
3. **Add your own data**: Put documents in `data/` folder
4. **Create questions**: Update `data/test_questions.json`
5. **Iterate**: Tune parameters in `configs/rag.yaml`

---

## Example: Adding Custom Data

```bash
# 1. Add your document
echo "Your content here" > data/my_document.txt

# 2. Re-ingest
python3 src/ingest.py --data-dir data

# 3. Add test questions
# Edit data/test_questions.json

# 4. Run evaluation
python3 src/eval.py --dataset data/test_questions.json
```

---

## Files You Can Explore

- `data/rag_comprehensive_guide.txt` - Sample document
- `data/test_questions.json` - Sample test questions  
- `EVALUATION_GUIDE.md` - Detailed documentation
- `evaluation_results.json` - Results (after running test)

Happy testing! 🎯
