import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                             f1_score, precision_recall_fscore_support)
import json
from datetime import datetime

class SystemEvaluator:
    def __init__(self):
        self.results = {}
        
    def evaluate(self, y_true, y_pred, y_pred_proba=None, categories=None):
        """
        Comprehensive evaluation
        """
        print("\n" + "="*80)
        print("SMARTLABEL EVALUATION REPORT")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # 1. Overall Metrics
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        
        print("\n📊 OVERALL PERFORMANCE METRICS")
        print("-" * 80)
        print(f"Macro F1 Score:     {macro_f1:.4f} {'✓ PASS' if macro_f1 >= 0.90 else '✗ BELOW TARGET'}")
        print(f"Micro F1 Score:     {micro_f1:.4f}")
        print(f"Weighted F1 Score:  {weighted_f1:.4f}")
        print(f"Overall Accuracy:   {(y_true == y_pred).mean():.4f}")
        
        self.results['overall_metrics'] = {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'weighted_f1': weighted_f1,
            'accuracy': (y_true == y_pred).mean()
        }
        
        # 2. Per-Class Metrics
        print("\n📈 PER-CATEGORY PERFORMANCE")
        print("-" * 80)
        
        if categories is None:
            categories = sorted(list(set(y_true) | set(y_pred)))
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=categories, zero_division=0
        )
        
        per_class_df = pd.DataFrame({
            'Category': categories,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        print(per_class_df.to_string(index=False))
        
        # Identify best and worst performing categories
        best_cat = per_class_df.loc[per_class_df['F1-Score'].idxmax()]
        worst_cat = per_class_df.loc[per_class_df['F1-Score'].idxmin()]
        
        print(f"\n🏆 Best Performance:  {best_cat['Category']} (F1: {best_cat['F1-Score']:.3f})")
        print(f"⚠️  Worst Performance: {worst_cat['Category']} (F1: {worst_cat['F1-Score']:.3f})")
        
        self.results['per_class_metrics'] = per_class_df.to_dict('records')
        
        # 3. Confusion Matrix
        print("\n🔀 CONFUSION MATRIX ANALYSIS")
        print("-" * 80)
        
        cm = confusion_matrix(y_true, y_pred, labels=categories)
        self.results['confusion_matrix'] = cm.tolist()
        
        # Find most common misclassifications
        misclass = []
        for i, true_cat in enumerate(categories):
            for j, pred_cat in enumerate(categories):
                if i != j and cm[i][j] > 0:
                    misclass.append({
                        'true': true_cat,
                        'predicted': pred_cat,
                        'count': int(cm[i][j])
                    })
        
        misclass.sort(key=lambda x: x['count'], reverse=True)
        
        print("Top 5 Misclassifications:")
        for mc in misclass[:5]:
            print(f"  • {mc['true']:30s} → {mc['predicted']:30s} ({mc['count']} times)")
        
        self.results['top_misclassifications'] = misclass[:10]
        
        # 4. Confidence Analysis (if probabilities provided)
        if y_pred_proba is not None:
            self._analyze_confidence(y_true, y_pred, y_pred_proba)
        
        # 5. Generate visualizations
        self._generate_visualizations(cm, categories, per_class_df)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        
        return self.results
    
    def _analyze_confidence(self, y_true, y_pred, y_pred_proba):
        """Analyze prediction confidence"""
        print("\n🎯 CONFIDENCE ANALYSIS")
        print("-" * 80)
        
        # Calculate confidence for predictions
        max_probs = np.max(y_pred_proba, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = (y_true == y_pred)
        
        correct_confidence = max_probs[correct_mask]
        incorrect_confidence = max_probs[~correct_mask]
        
        print(f"Average Confidence (Correct):   {correct_confidence.mean():.3f}")
        print(f"Average Confidence (Incorrect): {incorrect_confidence.mean():.3f}")
        
        # Confidence bins
        bins = [0, 0.5, 0.7, 0.8, 0.9, 1.0]
        bin_labels = ['<50%', '50-70%', '70-80%', '80-90%', '90-100%']
        
        print("\nAccuracy by Confidence Level:")
        for i, label in enumerate(bin_labels):
            mask = (max_probs >= bins[i]) & (max_probs < bins[i+1])
            if mask.sum() > 0:
                acc = (y_true[mask] == y_pred[mask]).mean()
                count = mask.sum()
                print(f"  {label:10s}: {acc:.3f} ({count} predictions)")
        
        self.results['confidence_analysis'] = {
            'avg_confidence_correct': float(correct_confidence.mean()),
            'avg_confidence_incorrect': float(incorrect_confidence.mean()),
            'high_confidence_errors': int((max_probs[~correct_mask] > 0.8).sum())
        }
    
    def _generate_visualizations(self, cm, categories, per_class_df):
        """Generate evaluation visualizations"""
        try:
            # 1. Confusion Matrix Heatmap
            plt.figure(figsize=(14, 12))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=categories, yticklabels=categories,
                       cbar_kws={'label': 'Count'})
            plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Category', fontsize=12)
            plt.ylabel('True Category', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
            print("\n✓ Confusion matrix saved to reports/confusion_matrix.png")
            plt.close()
            
            # 2. Per-Category F1 Scores
            plt.figure(figsize=(12, 8))
            colors = ['green' if f1 >= 0.90 else 'orange' if f1 >= 0.80 else 'red' 
                     for f1 in per_class_df['F1-Score']]
            
            plt.barh(per_class_df['Category'], per_class_df['F1-Score'], color=colors)
            plt.xlabel('F1 Score', fontsize=12)
            plt.title('F1 Score by Category', fontsize=16, fontweight='bold')
            plt.axvline(x=0.90, color='green', linestyle='--', label='Target (0.90)', linewidth=2)
            plt.legend()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig('reports/f1_scores.png', dpi=300, bbox_inches='tight')
            print("✓ F1 scores chart saved to reports/f1_scores.png")
            plt.close()
            
        except Exception as e:
            print(f"Note: Could not generate visualizations: {e}")
    
    def save_report(self, filepath='reports/evaluation_report.json'):
        """Save detailed evaluation report"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n✓ Detailed report saved to {filepath}")
    
    def generate_markdown_report(self, filepath='reports/EVALUATION_REPORT.md'):
        """Generate human-readable markdown report"""
        md = f"""# SmartLabel Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

- **Macro F1 Score:** {self.results['overall_metrics']['macro_f1']:.4f}
- **Target Achievement:** {'PASSED (>=0.90)' if self.results['overall_metrics']['macro_f1'] >= 0.90 else 'BELOW TARGET'}
- **Overall Accuracy:** {self.results['overall_metrics']['accuracy']:.4f}

---

## Per-Category Performance

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
"""
        
        for row in self.results['per_class_metrics']:
            md += f"| {row['Category']} | {row['Precision']:.3f} | {row['Recall']:.3f} | {row['F1-Score']:.3f} | {row['Support']} |\n"
        
        md += f"""
---

## Key Insights

### Top Misclassifications

"""
        for mc in self.results['top_misclassifications'][:5]:
            md += f"- **{mc['true']}** -> {mc['predicted']} ({mc['count']} times)\n"
        
        md += """
---

## Visualizations

![Confusion Matrix](confusion_matrix.png)

![F1 Scores](f1_scores.png)

---

## Recommendations

1. Review and improve categories with F1 < 0.85
2. Add more training data for frequently misclassified patterns
3. Fine-tune confidence thresholds for optimal active learning
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md)
        
        print(f"✓ Markdown report saved to {filepath}")
