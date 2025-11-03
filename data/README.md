# Evaluation Dataset

This directory contains test datasets for evaluating RAG system performance.

## test_questions.json

Sample evaluation dataset with questions and ground truth answers.

### Format

```json
[
  {
    "question": "Your question here",
    "ground_truth": "Expected answer (optional)"
  }
]
```

- `question`: The question to ask the RAG system
- `ground_truth`: (Optional) Expected answer for comparison

## Usage

### Evaluate entire dataset:
```bash
python src/eval.py --dataset data/test_questions.json
```

### Evaluate with MMR retrieval:
```bash
python src/eval.py --dataset data/test_questions.json --method mmr
```

### Save evaluation report:
```bash
python src/eval.py --dataset data/test_questions.json --output results.json
```

### Evaluate single question:
```bash
python src/eval.py --question "What is RAG?" --ground-truth "RAG stands for..."
```

## Metrics

The evaluation system calculates:

- **Answer Relevancy**: How relevant is the answer to the question?
- **Context Relevancy**: How relevant are the retrieved documents?
- **Groundedness**: Is the answer grounded in the retrieved context?
- **Faithfulness**: Does the answer stay faithful to the sources?
- **Ground Truth Similarity**: (If provided) How similar to expected answer?

All metrics range from 0.0 to 1.0, where higher is better.
