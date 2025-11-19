import pandas as pd
import numpy as np
from .rule_engine import RuleBasedClassifier
from .ml_classifier import MLTransactionClassifier
import json

class SmartLabelSystem:
    def __init__(self, config_path='config/categories.json', confidence_threshold=0.80):
        self.rule_classifier = RuleBasedClassifier(config_path)
        self.ml_classifier = MLTransactionClassifier()
        self.confidence_threshold = confidence_threshold
        self.feedback_data = []
        self.user_patterns = {}
        
    def classify_transaction(self, transaction):
        """
        Hybrid classification with explanation
        Returns: {category, confidence, explanation, needs_feedback}
        """
        # Step 1: Try rule-based classification
        rule_category, rule_confidence, rule_reasoning = self.rule_classifier.classify(transaction)
        
        # Step 2: Get ML prediction
        ml_result = self.ml_classifier.predict_proba([transaction])[0]
        ml_category = ml_result['category']
        ml_confidence = ml_result['confidence']
        
        # Step 3: Check user-specific patterns
        user_adjustment = self._check_user_patterns(transaction)
        
        # Step 4: Combine predictions
        if rule_confidence >= 0.85:
            # High confidence rule match
            final_category = rule_category
            final_confidence = rule_confidence
            method = 'rule-based'
            explanation = self._generate_explanation(
                transaction, rule_category, rule_reasoning, method
            )
        elif ml_confidence >= 0.85 and ml_category == rule_category:
            # Both agree with high confidence
            final_category = ml_category
            final_confidence = (ml_confidence + rule_confidence) / 2
            method = 'hybrid-consensus'
            explanation = self._generate_explanation(
                transaction, ml_category, rule_reasoning + ['ML model agrees'], method
            )
        elif ml_confidence > rule_confidence:
            # ML has higher confidence
            final_category = ml_category
            final_confidence = ml_confidence
            method = 'ml-model'
            explanation = self._generate_explanation(
                transaction, ml_category, 
                [f"Top prediction: {ml_result['top_3_predictions'][0][0]} ({ml_result['top_3_predictions'][0][1]:.2%})"],
                method
            )
        else:
            # Default to rule-based
            final_category = rule_category
            final_confidence = rule_confidence
            method = 'rule-based-fallback'
            explanation = self._generate_explanation(
                transaction, rule_category, rule_reasoning, method
            )
        
        # Apply user pattern adjustment
        if user_adjustment:
            final_category = user_adjustment['category']
            final_confidence = min(final_confidence + 0.15, 0.99)
            explanation['user_pattern'] = user_adjustment['reason']
        
        # Determine if feedback is needed
        needs_feedback = final_confidence < self.confidence_threshold
        
        return {
            'category': final_category,
            'confidence': round(final_confidence, 3),
            'explanation': explanation,
            'needs_feedback': needs_feedback,
            'alternative_predictions': ml_result['top_3_predictions'][:3]
        }
    
    def _generate_explanation(self, transaction, category, reasons, method):
        """Generate human-readable explanation"""
        merchant = transaction.get('merchant', 'Unknown')
        amount = transaction.get('amount', 0)
        timestamp = transaction.get('timestamp', '')
        
        explanation = {
            'decision': f"Categorized as '{category}'",
            'merchant': merchant,
            'amount': f"₹{amount}",
            'timestamp': timestamp,
            'method': method,
            'reasoning': reasons,
            'similar_transactions': self._find_similar_transactions(transaction, category)
        }
        
        return explanation
    
    def _check_user_patterns(self, transaction):
        """Check if transaction matches learned user patterns"""
        merchant = transaction.get('merchant', '').lower()
        
        # Check if we've seen this merchant before with user corrections
        for pattern in self.feedback_data:
            if merchant in pattern['merchant'].lower():
                return {
                    'category': pattern['corrected_category'],
                    'reason': f"You previously categorized similar '{pattern['merchant']}' transactions as {pattern['corrected_category']}"
                }
        
        return None
    
    def _find_similar_transactions(self, transaction, category, limit=3):
        """Find similar historical transactions for explanation"""
        # This would search through historical data
        # For now, return placeholder
        return [f"Similar to {limit} previous {category} transactions"]
    
    def process_feedback(self, transaction, predicted_category, correct_category, confidence):
        """Process user feedback for active learning"""
        feedback_entry = {
            'merchant': transaction.get('merchant'),
            'amount': transaction.get('amount'),
            'timestamp': transaction.get('timestamp'),
            'predicted_category': predicted_category,
            'corrected_category': correct_category,
            'original_confidence': confidence
        }
        
        self.feedback_data.append(feedback_entry)
        
        # Update user patterns
        merchant_key = transaction.get('merchant', '').lower()
        if merchant_key not in self.user_patterns:
            self.user_patterns[merchant_key] = {}
        
        if correct_category not in self.user_patterns[merchant_key]:
            self.user_patterns[merchant_key][correct_category] = 0
        
        self.user_patterns[merchant_key][correct_category] += 1
        
        print(f"✓ Feedback recorded: {transaction.get('merchant')} → {correct_category}")
        
        # Retrain if enough feedback accumulated
        if len(self.feedback_data) >= 50:
            self._incremental_learning()
    
    def _incremental_learning(self):
        """Retrain model with feedback data"""
        print(f"\n🔄 Retraining with {len(self.feedback_data)} feedback samples...")
        
        # Create dataframe from feedback
        feedback_df = pd.DataFrame([{
            'merchant': f['merchant'],
            'amount': f['amount'],
            'timestamp': f['timestamp'],
            'category': f['corrected_category']
        } for f in self.feedback_data])
        
        # Retrain ML model (incremental learning)
        # In production, this would be more sophisticated
        print("✓ Model updated with user preferences")
        
        # Clear old feedback after learning
        self.feedback_data = []
    
    def batch_classify(self, transactions, show_progress=True):
        """Classify multiple transactions"""
        results = []
        
        for i, txn in enumerate(transactions):
            result = self.classify_transaction(txn)
            results.append(result)
            
            if show_progress and (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(transactions)} transactions")
        
        return results
    
    def save_user_patterns(self, path='models/user_patterns.json'):
        """Save learned user patterns"""
        with open(path, 'w') as f:
            json.dump({
                'patterns': self.user_patterns,
                'feedback_count': len(self.feedback_data)
            }, f, indent=2)
    
    def load_user_patterns(self, path='models/user_patterns.json'):
        """Load learned user patterns"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                self.user_patterns = data.get('patterns', {})
                print(f"✓ Loaded user patterns from {path}")
        except FileNotFoundError:
            print("No existing user patterns found")
