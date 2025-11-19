import pandas as pd
import numpy as np
from datetime import datetime
import random
import hashlib

class FeatureExtractor:
    """Extract comprehensive features from raw transaction data"""
    
    def __init__(self):
        self.city_suffixes = ['Bangalore', 'Mumbai', 'Delhi', 'Pune', 'Hyderabad', 
                             'Chennai', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Lucknow',
                             'Kochi', 'Chandigarh', 'Indore', 'Nagpur', 'Bhopal']
    
    def extract_from_transaction(self, transaction):
        """Extract all 70+ features from a raw transaction"""
        
        # Get basic fields
        merchant = transaction.get('merchant', '')
        amount = float(transaction.get('amount', 0))
        
        # Parse timestamp
        timestamp_str = transaction.get('timestamp', '')
        if timestamp_str:
            try:
                dt = pd.to_datetime(timestamp_str)
            except:
                dt = pd.Timestamp.now()
        else:
            dt = pd.Timestamp.now()
        
        # Payment details
        payment_method = transaction.get('payment_method', 'UPI')
        payment_app = self._infer_payment_app(payment_method, merchant)
        card_type = self._infer_card_type(payment_method)
        
        # Location inference
        city = self._infer_city(merchant, transaction.get('city', 'Bangalore'))
        merchant_city = self._infer_merchant_city(merchant)
        is_same_city = 1 if city == merchant_city else 0
        distance = self._estimate_distance(is_same_city)
        
        # User behavior inference - FIX: ensure merchant_visit_count is never None
        visit_count_raw = transaction.get('visit_count', None)
        merchant_visit_count = visit_count_raw if visit_count_raw is not None else self._estimate_visit_count(merchant, amount)
        
        is_recurring = self._infer_recurring(merchant, amount)
        is_first_time = 1 if merchant_visit_count == 1 else 0
        
        # MCC code inference
        mcc_code = transaction.get('mcc', self._infer_mcc(merchant, amount))
        
        # Merchant trust
        is_verified, merchant_rating = self._infer_merchant_trust(merchant)
        
        # Build complete feature set
        features = {
            # Basic transaction info
            'transaction_id': transaction.get('transaction_id', self._generate_transaction_id()),
            'merchant': merchant,
            'amount': amount,
            'timestamp': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'upi_reference': transaction.get('upi_reference', self._generate_upi_reference()),
            'category': transaction.get('category', 'Unknown'),
            
            # Temporal features
            'year': dt.year,
            'month': dt.month,
            'day': dt.day,
            'hour': dt.hour,
            'minute': dt.minute,
            'day_of_week': dt.dayofweek,
            'is_weekend': 1 if dt.dayofweek >= 5 else 0,
            'is_month_start': 1 if dt.day <= 5 else 0,
            'is_month_end': 1 if dt.day >= 25 else 0,
            
            # Amount features
            'amount_rounded': round(amount, 0),
            'amount_log': round(np.log1p(amount), 4),
            'is_round_amount': 1 if amount % 10 == 0 else 0,
            'amount_last_digit': int(amount) % 10,
            'amount_decimal': round(amount % 1, 2),
            'is_large_transaction': 1 if amount > 5000 else 0,
            'is_micro_transaction': 1 if amount < 100 else 0,
            
            # Merchant text features
            'merchant_length': len(merchant),
            'merchant_word_count': len(merchant.split()),
            'merchant_uppercase_count': sum(1 for c in merchant if c.isupper()),
            'merchant_lowercase_count': sum(1 for c in merchant if c.islower()),
            'merchant_digit_count': sum(1 for c in merchant if c.isdigit()),
            'merchant_special_char_count': sum(1 for c in merchant if not c.isalnum() and c != ' '),
            'merchant_has_upi_prefix': 1 if merchant.lower().startswith(('upi', 'gpay', 'paytm', 'phonepe')) else 0,
            'merchant_has_numbers': 1 if any(c.isdigit() for c in merchant) else 0,
            'merchant_has_city': 1 if any(city_name.lower() in merchant.lower() for city_name in self.city_suffixes) else 0,
            'merchant_has_pvt_ltd': 1 if 'pvt' in merchant.lower() or 'ltd' in merchant.lower() else 0,
            'merchant_all_caps': 1 if merchant.isupper() else 0,
            'merchant_first_char': merchant[0] if merchant else '',
            'merchant_last_char': merchant[-1] if merchant else '',
            
            # Payment features
            'payment_method': payment_method,
            'payment_app': payment_app,
            'card_type': card_type,
            'bank_name': transaction.get('bank_name', self._infer_bank(payment_app)),
            'is_online': 1 if payment_method in ['UPI', 'Net Banking', 'Wallet'] else 0,
            'is_contactless': 1 if payment_method == 'UPI' else 0,
            'requires_otp': 1 if amount > 2000 or payment_method in ['Credit Card', 'Net Banking'] else 0,
            
            # Location features
            'city': city,
            'location_type': self._infer_location_type(merchant, dt.hour),
            'merchant_city': merchant_city,
            'is_same_city': is_same_city,
            'distance_from_home_km': distance,
            
            # User behavior features - FIX: use merchant_visit_count which is guaranteed to be int
            'is_recurring': is_recurring,
            'merchant_visit_count': merchant_visit_count,
            'days_since_last_transaction': transaction.get('days_since_last', self._estimate_days_since_last(is_recurring)),
            'transaction_sequence_number': transaction.get('sequence_number', random.randint(1, 1000)),
            'time_since_last_txn_hours': transaction.get('hours_since_last', self._estimate_hours_since_last(merchant_visit_count)),
            'is_first_time_merchant': is_first_time,
            'merchant_category_frequency': merchant_visit_count + random.randint(0, 10),  # Now safe
            'avg_monthly_spend_category': self._estimate_monthly_spend(merchant, amount),
            'is_unusual_time': 1 if dt.hour < 6 or dt.hour > 23 else 0,
            
            # Transaction context
            'description': transaction.get('description', f'Payment to {merchant}'),
            'narration': transaction.get('narration', f'Paid to {merchant}'),
            'merchant_category_code': mcc_code,
            'currency': 'INR',
            'exchange_rate': 1.0,
            'tax_amount': transaction.get('tax_amount', self._estimate_tax(merchant, amount)),
            'tip_amount': transaction.get('tip_amount', self._estimate_tip(merchant, amount, dt.hour)),
            'discount_amount': transaction.get('discount_amount', self._estimate_discount(merchant, amount)),
            'cashback_amount': transaction.get('cashback_amount', self._estimate_cashback(payment_method, amount)),
            
            # Risk features
            'risk_score': self._calculate_risk_score(amount, dt.hour, is_same_city, is_first_time),
            'is_verified_merchant': is_verified,
            'merchant_rating': merchant_rating,
            'transaction_status': transaction.get('status', 'Success'),
            'failure_reason': transaction.get('failure_reason', 'N/A')
        }
        
        return features
    
    # Helper methods for intelligent inference
    
    def _infer_payment_app(self, payment_method, merchant):
        """Infer payment app from merchant name or payment method"""
        merchant_lower = merchant.lower()
        
        if 'gpay' in merchant_lower or 'google pay' in merchant_lower:
            return 'GPay'
        elif 'phonepe' in merchant_lower:
            return 'PhonePe'
        elif 'paytm' in merchant_lower:
            return 'Paytm'
        elif 'amazon pay' in merchant_lower:
            return 'Amazon Pay'
        elif payment_method == 'UPI':
            return random.choice(['GPay', 'PhonePe', 'Paytm'])
        else:
            return 'Bank App'
    
    def _infer_card_type(self, payment_method):
        """Infer card type"""
        if payment_method in ['Credit Card', 'Debit Card']:
            return random.choice(['Visa', 'Mastercard', 'Rupay'])
        return 'N/A'
    
    def _infer_bank(self, payment_app):
        """Infer bank name"""
        return random.choice(['SBI', 'HDFC', 'ICICI', 'Axis', 'Kotak'])
    
    def _infer_city(self, merchant, default='Bangalore'):
        """Infer city from merchant name"""
        for city in self.city_suffixes:
            if city.lower() in merchant.lower():
                return city
        return default
    
    def _infer_merchant_city(self, merchant):
        """Infer merchant's city"""
        for city in self.city_suffixes:
            if city.lower() in merchant.lower():
                return city
        return random.choice(self.city_suffixes)
    
    def _estimate_distance(self, is_same_city):
        """Estimate distance"""
        if is_same_city:
            return round(random.uniform(0, 15), 2)
        return round(random.uniform(20, 50), 2)
    
    def _estimate_visit_count(self, merchant, amount):
        """Estimate how many times user visited this merchant"""
        merchant_lower = merchant.lower()
        
        # Frequent merchants
        if any(word in merchant_lower for word in ['swiggy', 'zomato', 'uber', 'ola']):
            return random.randint(10, 50)
        # Occasional merchants
        elif any(word in merchant_lower for word in ['amazon', 'flipkart', 'bigbasket']):
            return random.randint(5, 20)
        # Rare merchants
        else:
            return random.randint(1, 10)
    
    def _infer_recurring(self, merchant, amount):
        """Infer if transaction is recurring"""
        merchant_lower = merchant.lower()
        
        # Subscriptions and utilities are recurring
        recurring_keywords = ['netflix', 'prime', 'spotify', 'jio', 'airtel', 
                            'electricity', 'gas', 'water', 'broadband']
        
        if any(word in merchant_lower for word in recurring_keywords):
            return 1
        
        # Bills with round amounts are often recurring
        if amount % 100 == 0 and 200 <= amount <= 2000:
            return random.choice([1, 0])
        
        return 0
    
    def _infer_mcc(self, merchant, amount):
        """Infer MCC code from merchant name"""
        merchant_lower = merchant.lower()
        
        # Food delivery
        if any(word in merchant_lower for word in ['swiggy', 'zomato', 'delivery']):
            return '5814'
        # Restaurants
        elif any(word in merchant_lower for word in ['restaurant', 'cafe', 'pizza', 'burger']):
            return '5812'
        # Transportation
        elif any(word in merchant_lower for word in ['uber', 'ola', 'rapido', 'metro']):
            return '4121'
        # Entertainment
        elif any(word in merchant_lower for word in ['netflix', 'prime', 'hotstar', 'bookmyshow', 'pvr']):
            return '7832'
        # Groceries
        elif any(word in merchant_lower for word in ['dmart', 'bigbasket', 'grocery', 'kirana']):
            return '5411'
        # Shopping
        elif any(word in merchant_lower for word in ['amazon', 'flipkart', 'myntra']):
            return '5311'
        # Utilities
        elif any(word in merchant_lower for word in ['jio', 'airtel', 'electricity', 'gas']):
            return '4814'
        # Education
        elif any(word in merchant_lower for word in ['udemy', 'coursera', 'byju', 'unacademy']):
            return '8211'
        # Health
        elif any(word in merchant_lower for word in ['gym', 'hospital', 'pharmacy', 'clinic']):
            return '7997'
        # Default
        else:
            return '9999'
    
    def _infer_merchant_trust(self, merchant):
        """Infer merchant verification and rating"""
        merchant_lower = merchant.lower()
        
        major_brands = ['swiggy', 'zomato', 'amazon', 'flipkart', 'uber', 'ola',
                       'netflix', 'prime', 'bookmyshow', 'bigbasket', 'myntra']
        
        if any(brand in merchant_lower for brand in major_brands):
            return 1, round(random.uniform(4.2, 5.0), 1)
        else:
            return random.choice([1, 1, 0]), round(random.uniform(3.5, 4.8), 1)
    
    def _infer_location_type(self, merchant, hour):
        """Infer transaction location type"""
        merchant_lower = merchant.lower()
        
        if any(word in merchant_lower for word in ['swiggy', 'zomato', 'amazon', 'flipkart']):
            return 'Online'
        elif any(word in merchant_lower for word in ['restaurant', 'cafe', 'dhaba']):
            return 'Restaurant'
        elif any(word in merchant_lower for word in ['dmart', 'grocery', 'kirana']):
            return 'Store'
        else:
            return 'Online'
    
    def _estimate_days_since_last(self, is_recurring):
        """Estimate days since last transaction"""
        if is_recurring:
            return random.randint(28, 32)  # Monthly recurring
        return random.randint(1, 30)
    
    def _estimate_hours_since_last(self, visit_count):
        """Estimate hours since last transaction"""
        if visit_count is None or visit_count > 20:
            return round(random.uniform(12, 72), 2)  # Frequent user
        return round(random.uniform(24, 168), 2)  # Occasional user
    
    def _estimate_monthly_spend(self, merchant, amount):
        """Estimate monthly spend in this category"""
        merchant_lower = merchant.lower()
        
        if any(word in merchant_lower for word in ['swiggy', 'zomato']):
            return round(random.uniform(3000, 10000), 2)
        elif any(word in merchant_lower for word in ['uber', 'ola']):
            return round(random.uniform(1000, 5000), 2)
        elif any(word in merchant_lower for word in ['dmart', 'grocery']):
            return round(random.uniform(5000, 15000), 2)
        else:
            return round(amount * random.uniform(2, 10), 2)
    
    def _estimate_tax(self, merchant, amount):
        """Estimate GST/tax"""
        merchant_lower = merchant.lower()
        
        if any(word in merchant_lower for word in ['swiggy', 'zomato', 'restaurant']):
            return round(amount * 0.05, 2) if random.random() > 0.5 else 0
        elif any(word in merchant_lower for word in ['netflix', 'gym', 'hospital']):
            return round(amount * 0.18, 2) if random.random() > 0.3 else 0
        return 0
    
    def _estimate_tip(self, merchant, amount, hour):
        """Estimate tip amount"""
        merchant_lower = merchant.lower()
        
        if 'swiggy' in merchant_lower or 'zomato' in merchant_lower:
            if hour >= 22 or hour <= 6:
                return round(random.uniform(20, 50), 2) if random.random() > 0.4 else 0
            return round(random.uniform(10, 30), 2) if random.random() > 0.6 else 0
        return 0
    
    def _estimate_discount(self, merchant, amount):
        """Estimate discount"""
        merchant_lower = merchant.lower()
        
        if any(word in merchant_lower for word in ['amazon', 'flipkart', 'myntra', 'swiggy']):
            return round(amount * random.uniform(0.05, 0.20), 2) if random.random() > 0.7 else 0
        return 0
    
    def _estimate_cashback(self, payment_method, amount):
        """Estimate cashback"""
        if payment_method == 'UPI':
            return round(min(50, amount * 0.02), 2) if random.random() > 0.85 else 0
        elif payment_method == 'Credit Card':
            return round(amount * 0.01, 2) if random.random() > 0.80 else 0
        return 0
    
    def _calculate_risk_score(self, amount, hour, is_same_city, is_first_time):
        """Calculate risk score"""
        risk = 0.0
        
        if amount > 10000:
            risk += 0.3
        elif amount > 5000:
            risk += 0.2
        
        if hour < 6 or hour > 23:
            risk += 0.2
        
        if not is_same_city:
            risk += 0.1
        
        if is_first_time:
            risk += 0.15
        
        risk += random.uniform(0, 0.1)
        
        return round(min(risk, 1.0), 4)
    
    def _generate_transaction_id(self):
        """Generate transaction ID"""
        timestamp = datetime.now().timestamp()
        unique = f"{timestamp}{random.randint(1000, 9999)}"
        return hashlib.md5(unique.encode()).hexdigest()[:16].upper()
    
    def _generate_upi_reference(self):
        """Generate UPI reference"""
        return f"UPI{random.randint(100000000, 999999999)}"
