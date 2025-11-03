# RAG Evaluation System - Quick Start Guide

## Prerequisites

1. **Install scikit-learn** (new dependency):
```bash
pip install scikit-learn
```

2. **Ensure you have a vector index** (run ingestion first if needed):
```bash
python src/ingest.py --data-dir data
```

3. **Set your OpenAI API key** in `.env`:
```bash
OPENAI_API_KEY=your-api-key-here
```

---

## How to Test the Evaluation System

### Option 1: Single Question Evaluation

Test with one question to see how the system performs:

```bash
python src/eval.py \
  --question "What is RAG?" \
  --method similarity
```

**With ground truth comparison:**
```bash
python src/eval.py \
  --question "What is RAG?" \
  --ground-truth "RAG stands for Retrieval-Augmented Generation" \
  --method similarity
```

**Expected Output:**
```
Evaluating single question...
============================================================
Question: What is RAG?

Answer: [Generated answer from your RAG system]

Metrics:
------------------------------------------------------------
  Answer Relevancy:  0.856
  Context Relevancy: 0.782
  Groundedness:      0.794
  Faithfulness:      0.823
  Ground Truth Sim:  0.791

Sources: sample.txt
```

---

### Option 2: Dataset Evaluation

Evaluate multiple questions at once:

```bash
python src/eval.py \
  --dataset data/test_questions.json \
  --method similarity
```

**Save results to file:**
```bash
python src/eval.py \
  --dataset data/test_questions.json \
  --method similarity \
  --output evaluation_results.json
```

**Compare similarity vs MMR:**
```bash
# Similarity search
python src/eval.py --dataset data/test_questions.json --method similarity --output results_similarity.json

# MMR search
python src/eval.py --dataset data/test_questions.json --method mmr --output results_mmr.json
```

---

## Adding Your Own Test Data

### Step 1: Add Documents to `data/` folder

**Example: Add a file about machine learning:**

Create `data/machine_learning.txt`:
```
Machine Learning Overview

Machine learning is a subset of artificial intelligence that enables systems to learn 
and improve from experience without being explicitly programmed. It focuses on the 
development of computer programs that can access data and use it to learn for themselves.

Types of Machine Learning:
1. Supervised Learning - Learning from labeled data
2. Unsupervised Learning - Finding patterns in unlabeled data
3. Reinforcement Learning - Learning through trial and error

Common Algorithms:
- Linear Regression
- Decision Trees
- Neural Networks
- Support Vector Machines
```

### Step 2: Re-run Ingestion

Process the new documents:
```bash
python src/ingest.py --data-dir data
```

This will:
- Load all files in `data/` folder
- Chunk them into smaller pieces
- Generate embeddings
- Update the vector index

### Step 3: Create Evaluation Questions

Create or update `data/test_questions.json`:
```json
[
  {
    "question": "What is machine learning?",
    "ground_truth": "Machine learning is a subset of AI that enables systems to learn from experience without explicit programming."
  },
  {
    "question": "What are the main types of machine learning?",
    "ground_truth": "The main types are supervised learning, unsupervised learning, and reinforcement learning."
  },
  {
    "question": "Name some common ML algorithms",
    "ground_truth": "Common algorithms include linear regression, decision trees, neural networks, and support vector machines."
  }
]
```

### Step 4: Run Evaluation

```bash
python src/eval.py --dataset data/test_questions.json
```

---

## Good Example Datasets

### For Technical Documentation:

**Python Programming Guide** (`data/python_guide.txt`):
```
Python is a high-level programming language known for its simplicity and readability.

Key Features:
- Dynamic typing
- Automatic memory management
- Rich standard library
- Cross-platform compatibility

Common Use Cases:
- Web development (Django, Flask)
- Data science (Pandas, NumPy)
- Machine learning (TensorFlow, PyTorch)
- Automation and scripting
```

**Test Questions:**
```json
{
  "question": "What are the key features of Python?",
  "ground_truth": "Python features dynamic typing, automatic memory management, rich standard library, and cross-platform compatibility."
}
```

---

### For Product Documentation:

**Product FAQ** (`data/product_faq.txt`):
```
Product: SmartHome Hub

Q: What devices are compatible?
A: The SmartHome Hub works with all major smart home devices including 
lights, thermostats, cameras, and door locks from brands like Philips, 
Nest, Ring, and August.

Q: How do I set it up?
A: Setup takes 5 minutes:
1. Download the app
2. Plug in the hub
3. Follow in-app instructions
4. Connect your devices

Q: What's the warranty?
A: We offer a 2-year warranty covering hardware defects.
```

**Test Questions:**
```json
{
  "question": "What is the warranty period?",
  "ground_truth": "The SmartHome Hub has a 2-year warranty covering hardware defects."
}
```

---

### For Research Papers:

**Research Summary** (`data/research_summary.txt`):
```
Study: Impact of Sleep on Cognitive Performance

Methodology:
- 200 participants aged 18-65
- 6-month longitudinal study
- Sleep tracked via wearables

Key Findings:
- 7-9 hours of sleep optimal for cognitive performance
- Sleep deprivation (<6 hours) reduced task accuracy by 15%
- Sleep quality more important than quantity
- Consistency in sleep schedule improved memory retention

Recommendations:
- Maintain regular sleep schedule
- Avoid screens 1 hour before bed
- Keep bedroom temperature between 65-68°F
```

**Test Questions:**
```json
{
  "question": "What is the optimal sleep duration?",
  "ground_truth": "7-9 hours of sleep is optimal for cognitive performance."
}
```

---

## Understanding the Metrics

### Answer Relevancy (0-1)
- How semantically similar is the answer to the question?
- **Good**: > 0.7
- **Excellent**: > 0.85

### Context Relevancy (0-1)
- How relevant are the retrieved documents to the question?
- **Good**: > 0.7
- **Excellent**: > 0.85

### Groundedness (0-1)
- Is the answer based on the retrieved context?
- **Good**: > 0.7 (answer uses context)
- **Bad**: < 0.5 (potential hallucination)

### Faithfulness (0-1)
- Does the answer stay true to the source documents?
- **Good**: > 0.7 (accurate to sources)
- **Bad**: < 0.5 (may contain fabrications)

---

## Troubleshooting

### Error: "No module named 'scikit-learn'"
```bash
pip install scikit-learn
```

### Error: "RAG system not initialized"
Make sure you've run ingestion first:
```bash
python src/ingest.py --data-dir data
```

### Error: "OPENAI_API_KEY not set"
Add to your `.env` file:
```bash
OPENAI_API_KEY=sk-...
```

### Low Scores
If all metrics are low (<0.5):
- Check if documents actually contain relevant information
- Verify vector index was created successfully
- Try different retrieval methods (similarity vs mmr)
- Ensure ground truth questions match your document content

---

## Quick Test Workflow

```bash
# 1. Add your documents to data/ folder
cp my_documents.txt data/

# 2. Ingest documents
python src/ingest.py --data-dir data

# 3. Test with a quick question
python src/eval.py --question "Your test question here"

# 4. Create proper test dataset
# Edit data/test_questions.json with relevant questions

# 5. Run full evaluation
python src/eval.py --dataset data/test_questions.json --output results.json

# 6. Review results
cat results.json
```

---

## Next Steps

1. Add domain-specific documents to `data/`
2. Create test questions that match your documents
3. Run evaluation and iterate on:
   - Chunk size/overlap in `configs/rag.yaml`
   - Retrieval parameters (top_k, threshold)
   - Prompt templates in `src/prompt.py`

Happy evaluating! 🎯
