import json
import re
from datetime import datetime

class RuleBasedClassifier:
    def __init__(self, config_path='config/categories.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.categories = self.config['categories']
        
        # Define UPI/transfer patterns
        self.transfer_patterns = [
            'upi/', 'upi-', 'gpay', 'phonepe', 'paytm', 'bhim',
            'google pay', 'amazon pay', 'mobikwik', 'freecharge',
            'imps', 'neft', 'rtgs', 'transfer'
        ]
        
        # Define known merchant keywords across ALL categories
        self.known_merchants = set()
        for category, config in self.categories.items():
            if category != 'Transfer - Personal':
                self.known_merchants.update([kw.lower() for kw in config['keywords']])
        
    def classify(self, transaction):
        """
        Classify transaction using rules with enhanced features
        Returns: (category, confidence, reasoning)
        """
        merchant = transaction.get('merchant', '').lower()
        amount = transaction.get('amount', 0)
        hour = transaction.get('hour', None)
        is_weekend = transaction.get('is_weekend', 0)
        is_recurring = transaction.get('is_recurring', 0)
        merchant_visit_count = transaction.get('merchant_visit_count', 0)
        payment_method = transaction.get('payment_method', '')
        mcc_code = transaction.get('merchant_category_code', '')
        is_verified_merchant = transaction.get('is_verified_merchant', 1)
        merchant_has_upi_prefix = transaction.get('merchant_has_upi_prefix', 0)
        
        # CRITICAL: Enhanced transfer detection logic
        # Only classify as transfer if:
        # 1. Starts with UPI pattern AND
        # 2. Has NO merchant keywords from other categories AND
        # 3. Has MCC code 6012 (financial) AND
        # 4. Merchant name is very short (< 20 chars after UPI prefix)
        
        is_pure_transfer = False
        merchant_lower = merchant.lower()
        
        # Extract the merchant name after UPI prefix
        merchant_core = merchant_lower
        for pattern in ['upi/', 'gpay-', 'paytm-', 'phonepe-']:
            if merchant_lower.startswith(pattern):
                merchant_core = merchant_lower.replace(pattern, '').strip()
                break
        
        # Check if it's a pure transfer
        if any(merchant_lower.startswith(p) for p in ['upi/', 'gpay-', 'paytm-', 'phonepe-']):
            # Check if merchant core contains ANY known merchant keywords
            has_known_merchant = any(keyword in merchant_core for keyword in self.known_merchants if len(keyword) > 3)
            
            # Check MCC code
            is_financial_mcc = mcc_code == '6012'
            
            # Check if very generic (short name, no meaningful words)
            is_generic = len(merchant_core.replace('/', '').strip()) < 15
            
            # It's a pure transfer ONLY if:
            # - No known merchant keywords AND
            # - (Financial MCC OR very generic name)
            if not has_known_merchant and (is_financial_mcc or is_generic):
                is_pure_transfer = True
        
        if is_pure_transfer:
            return 'Transfer - Personal', 0.95, ['Pure UPI/payment transfer with no merchant context']
        
        matches = []
        
        for category, config in self.categories.items():
            # Skip transfer category in normal matching
            if category == 'Transfer - Personal':
                continue
                
            score = 0
            reasons = []
            
            # Keyword matching with fuzzy logic (weighted more heavily)
            keyword_matched = False
            for keyword in config['keywords']:
                if keyword.lower() in merchant_core:  # Search in core merchant name
                    score += 70  # Increased from 60
                    reasons.append(f"matched keyword '{keyword}'")
                    keyword_matched = True
                    break
            
            # If no keyword match but merchant is in the name somewhere, give partial credit
            if not keyword_matched:
                for keyword in config['keywords']:
                    if keyword.lower() in merchant_lower and len(keyword) > 4:
                        score += 40
                        reasons.append(f"partial match '{keyword}'")
                        break
            
            # MCC code matching (very important)
            expected_mcc = self._get_expected_mcc(category)
            if mcc_code in expected_mcc:
                score += 35  # Increased from 30
                reasons.append(f"MCC code {mcc_code} matches category")
            
            # Amount range check
            amt_range = config['amount_range']
            if amt_range[0] <= amount <= amt_range[1]:
                score += 20
                reasons.append(f"amount Rs.{amount} in typical range")
            elif amount < amt_range[0] * 0.5 or amount > amt_range[1] * 2:
                score -= 30
            
            # Time-based scoring (enhanced with weekend check)
            if hour is not None and config['typical_hours']:
                if hour in config['typical_hours']:
                    score += 15
                    reasons.append(f"transaction at {hour}:00 matches pattern")
                
                # Weekend bonuses for certain categories
                if is_weekend and category in ['Entertainment', 'Food - Dining Out', 'Shopping - Non-Essentials']:
                    score += 10
                    reasons.append("weekend transaction matches category pattern")
            
            # Recurring transaction bonus
            if is_recurring and category in ['Bills & Utilities', 'Education', 'Health & Fitness']:
                score += 15
                reasons.append("recurring transaction matches category")
            
            # Merchant visit frequency
            if merchant_visit_count > 5 and category in ['Food - Delivery', 'Food - Groceries', 'Transportation']:
                score += 10
                reasons.append(f"frequently visited merchant ({merchant_visit_count} times)")
            
            # Verified merchant bonus
            if is_verified_merchant:
                score += 5
            
            if score > 0:
                matches.append({
                    'category': category,
                    'score': score,
                    'reasons': reasons
                })
        
        if not matches:
            # Last resort: check if it looks like a transfer
            for pattern in self.transfer_patterns:
                if pattern in merchant_lower:
                    return 'Transfer - Personal', 0.4, ['Contains transfer keywords but no merchant match']
            return 'Other', 0.3, ['No strong rule matches']
        
        # Sort by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        best_match = matches[0]
        
        # Convert score to confidence (0-1)
        # Adjusted scoring to be more conservative
        max_possible_score = 155  # 70 + 35 + 20 + 15 + 10 + 5
        confidence = min(best_match['score'] / max_possible_score, 0.99)
        
        # Boost confidence if score is very high
        if best_match['score'] >= 100:
            confidence = min(confidence * 1.1, 0.99)
        
        return best_match['category'], confidence, best_match['reasons']
    
    def _get_expected_mcc(self, category):
        """Get expected MCC codes for a category"""
        mcc_map = {
            'Food - Dining Out': ['5812', '5814'],
            'Food - Delivery': ['5814'],
            'Food - Groceries': ['5411', '5422'],
            'Transportation': ['4121', '5542', '5172'],
            'Entertainment': ['7832', '7841', '7929'],
            'Shopping - Essentials': ['5311', '5399', '5912'],
            'Shopping - Non-Essentials': ['5621', '5651', '5699'],
            'Education': ['8211', '8220', '5942'],
            'Bills & Utilities': ['4814', '4900', '9399'],
            'Health & Fitness': ['7997', '8011', '8099'],
            'Transfer - Personal': ['6012'],
            'Other': ['9999'],
        }
        return mcc_map.get(category, [])
    
    def batch_classify(self, transactions):
        """Classify multiple transactions"""
        results = []
        for txn in transactions:
            category, confidence, reasoning = self.classify(txn)
            results.append({
                'category': category,
                'confidence': confidence,
                'reasoning': reasoning,
                'method': 'rule-based'
            })
        return results
