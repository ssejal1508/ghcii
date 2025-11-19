import os
import pandas as pd
import numpy as np
from src.data_generator import TransactionDataGenerator
from src.ml_classifier import MLTransactionClassifier
from src.hybrid_system import SmartLabelSystem
from src.bias_detector import BiasDetector
from src.evaluator import SystemEvaluator

def setup_directories():
    """Create necessary directories"""
    dirs = ['data', 'models', 'reports', 'config']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def main():
    print("="*80)
    print("SMARTLABEL - AUTONOMOUS TRANSACTION CATEGORIZER")
    print("GHCI Round 2 Hackathon Submission")
    print("="*80)
    
    # Setup
    setup_directories()
    
    # Step 1: Generate Dataset
    print("\n[1/6] Generating transaction dataset...")
    generator = TransactionDataGenerator('config/categories.json')
    df = generator.generate_dataset(n_samples=50000)
    df.to_csv('data/transactions.csv', index=False)
    print(f"✓ Generated {len(df)} transactions")
    print(f"✓ Category distribution:\n{df['category'].value_counts()}")
    
    # Step 2: Train ML Model
    print("\n[2/6] Training machine learning model...")
    ml_classifier = MLTransactionClassifier()
    eval_results = ml_classifier.train(df, test_size=0.2)
    ml_classifier.save_model('models/')
    
    print(f"\n✓ Model trained with Macro F1: {eval_results['macro_f1']:.4f}")
    
    # Step 3: Initialize Hybrid System
    print("\n[3/6] Initializing hybrid classification system...")
    smart_label = SmartLabelSystem('config/categories.json', confidence_threshold=0.80)
    smart_label.ml_classifier = ml_classifier  # Use the trained classifier
    print("✓ Hybrid system ready")
    
    # Step 4: Run Predictions
    print("\n[4/6] Running predictions on test set...")
    test_df = df.sample(n=1000, random_state=42)
    
    predictions = []
    confidences = []
    needs_feedback_count = 0
    
    for idx, row in test_df.iterrows():
        txn = row.to_dict()
        result = smart_label.classify_transaction(txn)
        predictions.append(result['category'])
        confidences.append(result['confidence'])
        if result['needs_feedback']:
            needs_feedback_count += 1
    
    test_df['predicted_category'] = predictions
    test_df['confidence'] = confidences
    
    print(f"✓ Classified {len(test_df)} transactions")
    print(f"✓ {needs_feedback_count} transactions flagged for user review (confidence < 80%)")
    
    # Step 5: Evaluate System
    print("\n[5/6] Evaluating system performance...")
    evaluator = SystemEvaluator()
    
    # Get unique categories
    categories = sorted(df['category'].unique())
    
    eval_results = evaluator.evaluate(
        y_true=test_df['category'].values,
        y_pred=test_df['predicted_category'].values,
        categories=categories
    )
    
    evaluator.save_report('reports/evaluation_report.json')
    evaluator.generate_markdown_report('reports/EVALUATION_REPORT.md')
    
    # Step 6: Bias Analysis
    print("\n[6/6] Running bias detection analysis...")
    
    # Prepare dataframe for bias analysis with correct column names
    bias_analysis_df = test_df.copy()
    bias_analysis_df['true_category'] = bias_analysis_df['category']  # Add true_category column
    
    bias_detector = BiasDetector()
    bias_results = bias_detector.analyze_bias(bias_analysis_df)
    fairness_report = bias_detector.generate_fairness_report()
    
    # Save fairness report
    import json
    with open('reports/fairness_report.json', 'w') as f:
        json.dump(fairness_report, f, indent=2, default=str)
    print("✓ Fairness report saved")
    
    # Final Summary
    print("\n" + "="*80)
    print("SYSTEM READY FOR DEMONSTRATION")
    print("="*80)
    print(f"\n📊 Final Macro F1 Score: {eval_results['overall_metrics']['macro_f1']:.4f}")
    print(f"🎯 Target Achievement: {'✅ PASSED' if eval_results['overall_metrics']['macro_f1'] >= 0.90 else '⚠️ NEEDS IMPROVEMENT'}")
    print(f"📈 Overall Accuracy: {eval_results['overall_metrics']['accuracy']:.4f}")
    print(f"🔍 Transactions needing review: {needs_feedback_count}/{len(test_df)} ({needs_feedback_count/len(test_df)*100:.1f}%)")
    
    print("\n📁 Generated Files:")
    print("  • data/transactions.csv - Complete dataset")
    print("  • models/ - Trained ML models")
    print("  • reports/evaluation_report.json - Detailed metrics")
    print("  • reports/EVALUATION_REPORT.md - Human-readable report")
    print("  • reports/fairness_report.json - Bias analysis")
    print("  • reports/confusion_matrix.png - Visual confusion matrix")
    print("  • reports/f1_scores.png - Per-category F1 scores")
    
    print("\n" + "="*80)
    
    # Demo: Show sample predictions with explanations
    print("\n📋 SAMPLE PREDICTIONS WITH EXPLANATIONS:\n")
    
    sample_transactions = test_df.head(5)
    for idx, row in sample_transactions.iterrows():
        result = smart_label.classify_transaction(row.to_dict())
        print(f"\nTransaction: {row['merchant']}")
        print(f"Amount: ₹{row['amount']}")
        print(f"True Category: {row['category']}")
        print(f"Predicted: {result['category']} ({result['confidence']:.2%} confidence)")
        print(f"Reasoning: {', '.join(result['explanation']['reasoning'][:2])}")
        print(f"Needs Review: {'Yes ⚠️' if result['needs_feedback'] else 'No ✓'}")
        print("-" * 60)

if __name__ == "__main__":
    main()
