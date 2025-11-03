"""
RAG Evaluation System

Provides metrics for evaluating RAG system performance:
- Answer Relevancy: How relevant is the answer to the question?
- Context Relevancy: How relevant are the retrieved documents?
- Groundedness: Is the answer grounded in the retrieved context?
- Faithfulness: Does the answer stay faithful to the source documents?

Usage:
    python src/eval.py --dataset path/to/test_questions.json
    python src/eval.py --question "What is RAG?" --ground-truth "RAG stands for..."
"""

import os
import json
import yaml
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path
import re
from collections import Counter

from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import numpy as np

from src.rag_chain import RAGChain
from src.configuration import resolve_config_path


class RAGEvaluator:
    """Evaluator for RAG system performance."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path or resolve_config_path()
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize RAG chain
        self.rag_chain = RAGChain(config_path=self.config_path)
        
        # Initialize embedding model for metric calculations
        embedding_config = self.config['embeddings']
        self.embedding_model = SentenceTransformer(embedding_config['model'])
        
        print(f"✓ RAG Evaluator initialized")
        print(f"  Configuration: {self.config_path}")
        print(f"  Embedding model: {embedding_config['model']}")
    
    def answer_relevancy(self, question: str, answer: str) -> float:
        """
        Calculate answer relevancy using semantic similarity.
        
        Measures how well the answer addresses the question by computing
        cosine similarity between question and answer embeddings.
        
        Args:
            question: The input question
            answer: The generated answer
            
        Returns:
            Score between 0 and 1 (higher is better)
        """
        # Embed question and answer
        question_embedding = self.embedding_model.encode([question])
        answer_embedding = self.embedding_model.encode([answer])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(question_embedding, answer_embedding)[0][0]
        
        # Normalize to 0-1 range (cosine similarity is already -1 to 1)
        score = (similarity + 1) / 2
        
        return float(score)
    
    def context_relevancy(self, question: str, contexts: List[str]) -> float:
        """
        Calculate context relevancy.
        
        Measures how relevant the retrieved documents are to the question
        by computing average similarity between question and each context.
        
        Args:
            question: The input question
            contexts: List of retrieved context documents
            
        Returns:
            Score between 0 and 1 (higher is better)
        """
        if not contexts:
            return 0.0
        
        # Embed question
        question_embedding = self.embedding_model.encode([question])
        
        # Embed contexts
        context_embeddings = self.embedding_model.encode(contexts)
        
        # Calculate similarities
        similarities = cosine_similarity(question_embedding, context_embeddings)[0]
        
        # Average similarity, normalized to 0-1
        avg_similarity = np.mean([(sim + 1) / 2 for sim in similarities])
        
        return float(avg_similarity)
    
    def groundedness(self, answer: str, contexts: List[str]) -> float:
        """
        Calculate groundedness score.
        
        Measures how well the answer is grounded in the retrieved contexts
        by checking token overlap and semantic similarity.
        
        Args:
            answer: The generated answer
            contexts: List of retrieved context documents
            
        Returns:
            Score between 0 and 1 (higher is better)
        """
        if not contexts:
            return 0.0
        
        # Combine all contexts
        combined_context = " ".join(contexts)
        
        # Token-based overlap (simplified)
        answer_tokens = set(re.findall(r'\w+', answer.lower()))
        context_tokens = set(re.findall(r'\w+', combined_context.lower()))
        
        # Calculate Jaccard similarity
        if not answer_tokens:
            token_overlap = 0.0
        else:
            intersection = answer_tokens.intersection(context_tokens)
            union = answer_tokens.union(context_tokens)
            token_overlap = len(intersection) / len(union) if union else 0.0
        
        # Semantic similarity
        answer_embedding = self.embedding_model.encode([answer])
        context_embedding = self.embedding_model.encode([combined_context])
        semantic_sim = cosine_similarity(answer_embedding, context_embedding)[0][0]
        semantic_sim = (semantic_sim + 1) / 2  # Normalize to 0-1
        
        # Combine both metrics (weighted average)
        score = 0.3 * token_overlap + 0.7 * semantic_sim
        
        return float(score)
    
    def faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Calculate faithfulness score.
        
        Measures whether the answer stays faithful to the source documents
        without hallucinating information not present in the contexts.
        
        This is a simplified version using entailment-like scoring.
        
        Args:
            answer: The generated answer
            contexts: List of retrieved context documents
            
        Returns:
            Score between 0 and 1 (higher is better)
        """
        if not contexts:
            return 0.0
        
        # Split answer into sentences/claims
        answer_sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
        
        if not answer_sentences:
            return 0.0
        
        # For each claim, check if it's supported by contexts
        supported_count = 0
        
        for sentence in answer_sentences:
            sentence_embedding = self.embedding_model.encode([sentence])
            
            # Check similarity with each context
            max_similarity = 0.0
            for context in contexts:
                context_embedding = self.embedding_model.encode([context])
                similarity = cosine_similarity(sentence_embedding, context_embedding)[0][0]
                max_similarity = max(max_similarity, similarity)
            
            # If sentence is highly similar to any context, consider it supported
            # Threshold of 0.5 (after normalization to 0-1)
            normalized_sim = (max_similarity + 1) / 2
            if normalized_sim > 0.5:
                supported_count += 1
        
        # Faithfulness = proportion of supported claims
        score = supported_count / len(answer_sentences)
        
        return float(score)
    
    def evaluate_single(
        self,
        question: str,
        ground_truth: Optional[str] = None,
        method: str = "similarity"
    ) -> Dict[str, Any]:
        """
        Evaluate a single question through the RAG system.
        
        Args:
            question: The question to evaluate
            ground_truth: Optional ground truth answer for comparison
            method: Retrieval method to use
            
        Returns:
            Dictionary containing all metrics and the generated answer
        """
        # Get RAG response with context
        result = self.rag_chain.query(question=question, method=method, return_context=True)
        answer = result["answer"]
        documents = result.get("context", [])
        
        # Extract text content from documents
        contexts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
        
        # Calculate metrics
        metrics = {
            "question": question,
            "answer": answer,
            "method": method,
            "answer_relevancy": self.answer_relevancy(question, answer),
            "context_relevancy": self.context_relevancy(question, contexts),
            "groundedness": self.groundedness(answer, contexts),
            "faithfulness": self.faithfulness(answer, contexts),
            "num_contexts": len(contexts),
            "sources": result["sources"]
        }
        
        # If ground truth provided, calculate similarity
        if ground_truth:
            gt_embedding = self.embedding_model.encode([ground_truth])
            answer_embedding = self.embedding_model.encode([answer])
            similarity = cosine_similarity(gt_embedding, answer_embedding)[0][0]
            metrics["ground_truth_similarity"] = float((similarity + 1) / 2)
        
        return metrics
    
    def evaluate_dataset(self, dataset_path: str, method: str = "similarity") -> Dict[str, Any]:
        """
        Evaluate the RAG system on a dataset of questions.
        
        Dataset format (JSON):
        [
            {
                "question": "What is RAG?",
                "ground_truth": "RAG stands for..." (optional)
            },
            ...
        ]
        
        Args:
            dataset_path: Path to JSON file with test questions
            method: Retrieval method to use
            
        Returns:
            Dictionary with aggregated metrics
        """
        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        print(f"\nEvaluating {len(dataset)} questions...")
        print("=" * 60)
        
        # Evaluate each question
        results = []
        for i, item in enumerate(dataset, 1):
            question = item["question"]
            ground_truth = item.get("ground_truth")
            
            print(f"\n[{i}/{len(dataset)}] {question}")
            
            metrics = self.evaluate_single(question, ground_truth, method)
            results.append(metrics)
            
            # Print metrics for this question
            print(f"  Answer Relevancy:  {metrics['answer_relevancy']:.3f}")
            print(f"  Context Relevancy: {metrics['context_relevancy']:.3f}")
            print(f"  Groundedness:      {metrics['groundedness']:.3f}")
            print(f"  Faithfulness:      {metrics['faithfulness']:.3f}")
            if ground_truth:
                print(f"  Ground Truth Sim:  {metrics['ground_truth_similarity']:.3f}")
        
        # Calculate aggregate metrics
        aggregated = {
            "total_questions": len(results),
            "method": method,
            "average_metrics": {
                "answer_relevancy": np.mean([r["answer_relevancy"] for r in results]),
                "context_relevancy": np.mean([r["context_relevancy"] for r in results]),
                "groundedness": np.mean([r["groundedness"] for r in results]),
                "faithfulness": np.mean([r["faithfulness"] for r in results]),
            },
            "individual_results": results
        }
        
        if any("ground_truth_similarity" in r for r in results):
            aggregated["average_metrics"]["ground_truth_similarity"] = np.mean(
                [r.get("ground_truth_similarity", 0) for r in results if "ground_truth_similarity" in r]
            )
        
        return aggregated
    
    def print_report(self, evaluation_results: Dict[str, Any]):
        """
        Print a formatted evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_dataset()
        """
        print("\n" + "=" * 60)
        print("RAG SYSTEM EVALUATION REPORT")
        print("=" * 60)
        
        print(f"\nTotal Questions Evaluated: {evaluation_results['total_questions']}")
        print(f"Retrieval Method: {evaluation_results['method']}")
        
        print("\nAverage Metrics:")
        print("-" * 60)
        metrics = evaluation_results['average_metrics']
        print(f"  Answer Relevancy:  {metrics['answer_relevancy']:.3f}")
        print(f"  Context Relevancy: {metrics['context_relevancy']:.3f}")
        print(f"  Groundedness:      {metrics['groundedness']:.3f}")
        print(f"  Faithfulness:      {metrics['faithfulness']:.3f}")
        
        if "ground_truth_similarity" in metrics:
            print(f"  Ground Truth Sim:  {metrics['ground_truth_similarity']:.3f}")
        
        # Overall score (average of all metrics)
        overall = np.mean(list(metrics.values()))
        print(f"\n  Overall Score:     {overall:.3f}")
        
        print("\n" + "=" * 60)
    
    def save_report(self, evaluation_results: Dict[str, Any], output_path: str):
        """
        Save evaluation report to JSON file.
        
        Args:
            evaluation_results: Results from evaluate_dataset()
            output_path: Path to save JSON report
        """
        with open(output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"\n✓ Report saved to: {output_path}")


def main():
    """Main CLI for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate RAG system performance")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to JSON dataset file with test questions"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to evaluate"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        help="Ground truth answer (for single question evaluation)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["similarity", "mmr"],
        default="similarity",
        help="Retrieval method to use (default: similarity)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save evaluation report (JSON)"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = RAGEvaluator(config_path=args.config)
    
    # Single question evaluation
    if args.question:
        print("\nEvaluating single question...")
        print("=" * 60)
        print(f"Question: {args.question}")
        
        metrics = evaluator.evaluate_single(
            question=args.question,
            ground_truth=args.ground_truth,
            method=args.method
        )
        
        print(f"\nAnswer: {metrics['answer']}")
        print("\nMetrics:")
        print("-" * 60)
        print(f"  Answer Relevancy:  {metrics['answer_relevancy']:.3f}")
        print(f"  Context Relevancy: {metrics['context_relevancy']:.3f}")
        print(f"  Groundedness:      {metrics['groundedness']:.3f}")
        print(f"  Faithfulness:      {metrics['faithfulness']:.3f}")
        
        if "ground_truth_similarity" in metrics:
            print(f"  Ground Truth Sim:  {metrics['ground_truth_similarity']:.3f}")
        
        print(f"\nSources: {', '.join(metrics['sources'])}")
    
    # Dataset evaluation
    elif args.dataset:
        results = evaluator.evaluate_dataset(
            dataset_path=args.dataset,
            method=args.method
        )
        
        evaluator.print_report(results)
        
        # Save report if output path provided
        if args.output:
            evaluator.save_report(results, args.output)
    
    else:
        parser.print_help()
        print("\nError: Either --dataset or --question must be provided")


if __name__ == "__main__":
    main()