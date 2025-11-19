import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import re
import hashlib

class TransactionDataGenerator:
    def __init__(self, config_path='config/categories.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.categories = self.config['categories']
        
        # Diverse merchant database
        self.merchant_database = {
            'Food - Dining Out': [
                # National chains
                'Dominos Pizza', 'Pizza Hut', 'KFC', 'McDonalds', 'Burger King', 'Subway',
                'Cafe Coffee Day', 'Starbucks', 'Barista', 'Chaayos',
                # Regional restaurants
                'Saravana Bhavan', 'MTR Foods', 'Kamat Restaurant', 'Udupi Krishna',
                'Barbeque Nation', 'Mainland China', 'Social Offline', 'The Beer Cafe',
                # Local eateries
                'Sharma Dhaba', 'Punjabi Rasoi', 'Sagar Ratna', 'Haldirams',
                'Annapurna Hotel', 'City Palace Restaurant', 'Royal Biryani House',
                'Truffles Ice Cream', 'Corner House', 'Naturals Ice Cream',
                # Street food style
                'Chaat Corner', 'Dosa Plaza', 'Rolls Mania', 'Momos Point'
            ],
            'Food - Delivery': [
                'Swiggy', 'Zomato', 'Dunzo Daily', 'Zepto', 'Blinkit', 'Instamart',
                'Uber Eats', 'Food Panda', 'Box8', 'Faasos', 'Behrouz Biryani',
                'Oven Story Pizza', 'Wow Momo', 'Freshmenu'
            ],
            'Food - Groceries': [
                # Modern retail
                'DMart', 'Reliance Fresh', 'Star Bazaar', 'More Megastore', 'Spencer Retail',
                'Big Bazaar', 'Hypercity', '24 Seven', 
                # Online
                'BigBasket', 'Grofers', 'Amazon Fresh', 'JioMart',
                # Traditional
                'Ramesh Kirana Store', 'Gupta Provision Store', 'Metro Cash & Carry',
                'Nilgiris', 'Heritage Fresh', 'Apna Bazaar'
            ],
            'Transportation': [
                # Ride hailing
                'Uber India', 'Ola Cabs', 'Rapido Bike', 'Meru Cabs', 'BluSmart',
                # Fuel
                'Indian Oil Petrol', 'HP Petrol Pump', 'Bharat Petroleum', 'Shell Petrol',
                'Reliance Petrol', 'Essar Oil',
                # Others
                'BMTC Bus Pass', 'Delhi Metro', 'Mumbai Metro', 'Fastag Recharge',
                'Park Plus Parking', 'EasyMyTrip Cabs'
            ],
            'Entertainment': [
                'BookMyShow', 'PVR Cinemas', 'INOX Movies', 'Carnival Cinemas',
                'Netflix India', 'Amazon Prime Video', 'Disney Hotstar', 'Sony Liv',
                'Spotify Premium', 'YouTube Premium', 'Gaana Plus',
                'PlayStation Store', 'Steam Games', 'Xbox Store', 'Google Play Games'
            ],
            'Shopping - Essentials': [
                # E-commerce
                'Amazon India', 'Flipkart', 'Snapdeal', 'Myntra', 'Ajio',
                'Meesho', 'FirstCry', 'Nykaa',
                # Books & Stationery
                'Crossword Bookstore', 'Landmark Books', 'Odyssey', 'Sapna Book House',
                'Campus Sutra Store', 'University Book Depot',
                # Medical
                'Apollo Pharmacy', 'MedPlus', 'Netmeds', 'PharmEasy', '1mg',
                'Wellness Forever', 'Guardian Pharmacy'
            ],
            'Shopping - Non-Essentials': [
                'Lifestyle Stores', 'Westside', 'Pantaloons', 'Max Fashion',
                'Zara India', 'H&M India', 'Uniqlo', 'Decathlon Sports',
                'Titan World', 'Tanishq Jewellers', 'Malabar Gold',
                'Lenskart', 'Forest Essentials', 'The Body Shop',
                'MAC Cosmetics', 'Sephora India'
            ],
            'Education': [
                'Udemy', 'Coursera', 'Unacademy Plus', 'Byjus Classes', 'Vedantu',
                'Khan Academy', 'Toppr', 'WhiteHat Jr',
                'British Council', 'IDP Education', 'Manya Princeton',
                'Delhi University', 'Anna University Fees', 'IIT Madras',
                'VIT University', 'Manipal University',
                'City Coaching Center', 'Brilliant Tutorials', 'Allen Career Institute'
            ],
            'Bills & Utilities': [
                # Telecom
                'Jio Recharge', 'Airtel Postpaid', 'Vi Vodafone', 'BSNL Mobile',
                'ACT Fibernet', 'Hathway Broadband', 'Tikona Wifi',
                # Utilities
                'BESCOM Electricity', 'MSEDCL Bill', 'Bangalore Water Board',
                'Indane Gas', 'Bharat Gas', 'HP Gas Cylinder',
                # DTH
                'Tata Sky', 'Dish TV', 'Airtel Digital TV', 'Sun Direct'
            ],
            'Health & Fitness': [
                'Cult Fit Gym', 'Gold Gym', 'Anytime Fitness', 'Fitness First',
                'The Yoga Institute', 'Art of Living', 'Patanjali Wellness',
                'Apollo Hospitals', 'Fortis Healthcare', 'Max Hospital',
                'Dr Lal PathLabs', 'Thyrocare', 'Metropolis Healthcare',
                'City Clinic', 'Skin Care Center', 'Dental Care Hospital'
            ],
            'Transfer - Personal': [
                'UPI Transfer', 'Google Pay', 'PhonePe', 'Paytm', 'BHIM UPI',
                'Amazon Pay', 'Mobikwik', 'Freecharge', 'IMPS Transfer', 'NEFT Payment'
            ]
        }
        
        # Regional variations
        self.city_suffixes = ['Bangalore', 'Mumbai', 'Delhi', 'Pune', 'Hyderabad', 
                             'Chennai', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Lucknow',
                             'Kochi', 'Chandigarh', 'Indore', 'Nagpur', 'Bhopal']
        
    def generate_dataset(self, n_samples=50000, real_data_path=None):
        """Generate synthetic + real transaction data"""
        transactions = []
        
        # Load real data if available
        if real_data_path:
            real_df = pd.read_csv(real_data_path)
            transactions.extend(real_df.to_dict('records'))
            n_samples -= len(real_df)
        
        # Generate synthetic data
        for _ in range(n_samples):
            category = random.choice(list(self.categories.keys()))
            transaction = self._generate_transaction(category)
            transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        return df
    
    def _generate_transaction(self, category):
        """Generate a single realistic transaction with comprehensive features"""
        cat_config = self.categories[category]
        
        # Generate timestamp first to use for context
        timestamp = self._generate_timestamp(cat_config['typical_hours'])
        dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        
        # Generate merchant name with category-specific logic
        merchant = self._generate_merchant_name(category, cat_config)
        
        # Generate amount with realistic patterns based on category and time
        amount = self._generate_realistic_amount(category, dt, merchant)
        
        # Generate payment method based on category and amount
        payment_method, payment_app, card_type = self._generate_payment_details(category, amount, dt)
        
        # Generate location details with consistency
        city, merchant_city, is_same_city, distance = self._generate_location_details()
        
        # Generate user behavior patterns
        is_recurring, merchant_visit_count, is_first_time = self._generate_user_behavior(category, merchant)
        
        # Add UPI reference patterns
        upi_ref = self._generate_upi_reference()
        
        # Generate MCC based on category
        mcc_code = self._generate_mcc(category)
        
        # Generate verification and rating based on merchant type
        is_verified, merchant_rating = self._generate_merchant_trust(merchant, category)
        
        # Generate comprehensive transaction data
        return {
            # Basic transaction info (Columns 1-6)
            'transaction_id': self._generate_transaction_id(),
            'merchant': merchant,
            'amount': amount,
            'timestamp': timestamp,
            'upi_reference': upi_ref,
            'category': category,
            
            # Temporal features (Columns 7-15)
            'year': dt.year,
            'month': dt.month,
            'day': dt.day,
            'hour': dt.hour,
            'minute': dt.minute,
            'day_of_week': dt.weekday(),  # 0=Monday, 6=Sunday
            'is_weekend': 1 if dt.weekday() >= 5 else 0,
            'is_month_start': 1 if dt.day <= 5 else 0,
            'is_month_end': 1 if dt.day >= 25 else 0,
            
            # Amount-based features (Columns 16-22)
            'amount_rounded': round(amount, 0),
            'amount_log': round(np.log1p(amount), 4),
            'is_round_amount': 1 if amount % 10 == 0 else 0,
            'amount_last_digit': int(amount) % 10,
            'amount_decimal': round(amount % 1, 2),
            'is_large_transaction': 1 if amount > 5000 else 0,
            'is_micro_transaction': 1 if amount < 100 else 0,
            
            # Merchant text features (Columns 23-35)
            'merchant_length': len(merchant),
            'merchant_word_count': len(merchant.split()),
            'merchant_uppercase_count': sum(1 for c in merchant if c.isupper()),
            'merchant_lowercase_count': sum(1 for c in merchant if c.islower()),
            'merchant_digit_count': sum(1 for c in merchant if c.isdigit()),
            'merchant_special_char_count': sum(1 for c in merchant if not c.isalnum() and c != ' '),
            'merchant_has_upi_prefix': 1 if merchant.lower().startswith(('upi', 'gpay', 'paytm', 'phonepe')) else 0,
            'merchant_has_numbers': 1 if any(c.isdigit() for c in merchant) else 0,
            'merchant_has_city': 1 if any(city.lower() in merchant.lower() for city in self.city_suffixes) else 0,
            'merchant_has_pvt_ltd': 1 if 'pvt' in merchant.lower() or 'ltd' in merchant.lower() else 0,
            'merchant_all_caps': 1 if merchant.isupper() else 0,
            'merchant_first_char': merchant[0] if merchant else '',
            'merchant_last_char': merchant[-1] if merchant else '',
            
            # Payment method features (Columns 36-42)
            'payment_method': payment_method,
            'payment_app': payment_app,
            'card_type': card_type,
            'bank_name': random.choice(['SBI', 'HDFC', 'ICICI', 'Axis', 'Kotak', 'PNB', 'BOB', 'Canara']),
            'is_online': 1 if payment_method in ['UPI', 'Net Banking', 'Wallet'] else random.choice([0, 1]),
            'is_contactless': 1 if payment_method == 'UPI' else random.choice([0, 1]),
            'requires_otp': 1 if amount > 2000 or payment_method in ['Credit Card', 'Net Banking'] else random.choice([0, 1]),
            
            # Location features (Columns 43-47)
            'city': city,
            'location_type': self._get_location_type(category, dt.hour),
            'merchant_city': merchant_city,
            'is_same_city': is_same_city,
            'distance_from_home_km': distance,
            
            # User behavior features (Columns 48-56)
            'is_recurring': is_recurring,
            'merchant_visit_count': merchant_visit_count,
            'days_since_last_transaction': random.randint(0, 30) if is_recurring else random.randint(0, 90),
            'transaction_sequence_number': random.randint(1, 1000),
            'time_since_last_txn_hours': round(random.uniform(1, 48) if merchant_visit_count > 10 else random.uniform(1, 168), 2),
            'is_first_time_merchant': is_first_time,
            'merchant_category_frequency': merchant_visit_count + random.randint(0, 20),
            'avg_monthly_spend_category': self._get_avg_monthly_spend(category),
            'is_unusual_time': 1 if dt.hour < 6 or dt.hour > 23 else 0,
            
            # Transaction context (Columns 57-65)
            'description': self._generate_description(merchant, amount),
            'narration': self._generate_narration(merchant, category),
            'merchant_category_code': mcc_code,
            'currency': 'INR',
            'exchange_rate': 1.0,
            'tax_amount': self._calculate_tax(amount, category),
            'tip_amount': self._calculate_tip(amount, category, dt.hour),
            'discount_amount': self._calculate_discount(amount, category, dt.day),
            'cashback_amount': self._calculate_cashback(amount, payment_method),
            
            # Risk and fraud indicators (Columns 66-70)
            'risk_score': self._calculate_risk_score(amount, dt.hour, is_same_city, is_first_time),
            'is_verified_merchant': is_verified,
            'merchant_rating': merchant_rating,
            'transaction_status': self._get_transaction_status(amount, payment_method),
            'failure_reason': 'N/A',
        }
    
    def _generate_transaction_id(self):
        """Generate unique transaction ID"""
        timestamp = datetime.now().timestamp()
        random_part = random.randint(1000, 9999)
        unique_string = f"{timestamp}{random_part}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16].upper()
    
    def _generate_narration(self, merchant, category):
        """Generate transaction narration"""
        templates = [
            f"Paid to {merchant}",
            f"{merchant} - Purchase",
            f"Payment for {category.lower()}",
            f"Online payment - {merchant}",
            f"POS transaction at {merchant}",
        ]
        return random.choice(templates)
    
    def _generate_mcc(self, category):
        """Generate Merchant Category Code based on category"""
        mcc_map = {
            'Food - Dining Out': random.choice(['5812', '5814']),  # Restaurants
            'Food - Delivery': '5814',  # Fast Food
            'Food - Groceries': random.choice(['5411', '5422']),  # Grocery stores
            'Transportation': random.choice(['4121', '5542', '5172']),  # Taxi, Auto, Fuel
            'Entertainment': random.choice(['7832', '7841', '7929']),  # Movies, Entertainment
            'Shopping - Essentials': random.choice(['5311', '5399', '5912']),  # Department stores
            'Shopping - Non-Essentials': random.choice(['5621', '5651', '5699']),  # Clothing, accessories
            'Education': random.choice(['8211', '8220', '5942']),  # Schools, courses, books
            'Bills & Utilities': random.choice(['4814', '4900', '9399']),  # Telecom, utilities
            'Health & Fitness': random.choice(['7997', '8011', '8099']),  # Gym, doctors
            'Transfer - Personal': '6012',  # Financial institutions
            'Other': '9999',
        }
        return mcc_map.get(category, '9999')
    
    def _generate_merchant_name(self, category, config):
        """Generate realistic merchant names with diversity and variations"""
        # Use diverse merchant database
        if category in self.merchant_database:
            base_merchant = random.choice(self.merchant_database[category])
        else:
            keywords = config['keywords']
            if not keywords:
                return "MISCELLANEOUS TRANSACTION"
            base_merchant = random.choice(keywords).upper()
        
        # Add realistic variations (70% chance)
        if random.random() < 0.7:
            variations = [
                lambda x: x,  # Original
                lambda x: x.upper(),  # All caps
                lambda x: x.replace(' ', ''),  # Remove spaces
                lambda x: x.replace(' ', '-'),  # Hyphenate
                lambda x: x + ' ' + random.choice(['PVT LTD', 'PRIVATE LIMITED', 'INDIA', 
                                                   'SERVICES', 'TECH', 'ONLINE']),
                lambda x: 'UPI-' + x,
                lambda x: x + '-' + ''.join(random.choices('0123456789', k=4)),
                lambda x: self._add_typo(x),  # Add typos
                lambda x: x + ' ' + random.choice(self.city_suffixes),
                lambda x: x + ' - ' + random.choice(self.city_suffixes),
                lambda x: 'PAYTM-' + x if random.random() > 0.5 else 'GPAY-' + x,
            ]
            
            variation_func = random.choice(variations)
            merchant = variation_func(base_merchant)
        else:
            merchant = base_merchant
        
        # Add UPI reference pattern (30% chance)
        if random.random() < 0.3:
            merchant = f"UPI/{merchant}/{random.randint(100000, 999999)}"
        
        return merchant
    
    def _add_typo(self, text):
        """Add realistic typos"""
        if len(text) < 5 or random.random() > 0.3:
            return text
        
        typo_types = [
            lambda t, i: t[:i] + t[i+1:],  # Delete char
            lambda t, i: t[:i] + t[i] + t[i] + t[i+1:],  # Duplicate char
            lambda t, i: t[:i] + random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + t[i+1:],  # Replace
        ]
        
        pos = random.randint(1, len(text)-2)
        typo_func = random.choice(typo_types)
        return typo_func(text, pos)
    
    def _generate_timestamp(self, typical_hours):
        """Generate realistic timestamps"""
        days_ago = random.randint(0, 365)
        hour = random.choice(typical_hours + list(range(24)))  # Some variation
        minute = random.randint(0, 59)
        
        base_date = datetime.now() - timedelta(days=days_ago)
        timestamp = base_date.replace(hour=hour, minute=minute, second=0)
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    def _generate_upi_reference(self):
        """Generate UPI reference number"""
        prefixes = ['UPI', 'IMPS', 'NEFT', 'RTGS']
        ref = f"{random.choice(prefixes)}{random.randint(100000000, 999999999)}"
        return ref
    
    def _generate_description(self, merchant, amount):
        """Generate transaction description"""
        templates = [
            f"Payment to {merchant}",
            f"UPI/{merchant}/Rs.{amount}",
            f"{merchant} - Online Payment",
            f"Debit Card Transaction - {merchant}",
        ]
        return random.choice(templates)
    
    def _generate_realistic_amount(self, category, dt, merchant):
        """Generate realistic amounts based on context"""
        cat_config = self.categories[category]
        amount_range = cat_config['amount_range']
        
        # Base amount
        if category == 'Food - Delivery':
            # Lunch smaller, dinner larger
            if 12 <= dt.hour <= 14:
                amount = round(np.random.uniform(150, 400), 2)
            elif 19 <= dt.hour <= 22:
                amount = round(np.random.uniform(250, 800), 2)
            else:
                amount = round(np.random.uniform(100, 500), 2)
                
        elif category == 'Food - Dining Out':
            # Weekend dinners more expensive
            if dt.weekday() >= 5 and dt.hour >= 19:
                amount = round(np.random.uniform(500, 2000), 2)
            else:
                amount = round(np.random.uniform(200, 1000), 2)
                
        elif category == 'Transportation':
            # Uber/Ola = random, Petrol = round amounts
            if any(word in merchant.lower() for word in ['uber', 'ola', 'rapido']):
                amount = round(np.random.uniform(50, 500), 2)
            elif 'petrol' in merchant.lower():
                amount = round(random.choice([500, 1000, 1500, 2000]), 2)  # Round amounts
            else:
                amount = round(np.random.uniform(20, 300), 2)
                
        elif category == 'Bills & Utilities':
            # Bills are usually round amounts
            base = random.choice([299, 399, 499, 599, 799, 999, 1499, 1999, 2499])
            amount = float(base)
            
        elif category == 'Entertainment':
            # Subscriptions = fixed, movies = variable
            if any(word in merchant.lower() for word in ['netflix', 'prime', 'hotstar', 'spotify']):
                amount = random.choice([149.0, 199.0, 299.0, 499.0, 999.0])
            else:
                amount = round(np.random.uniform(200, 1500), 2)
                
        elif category == 'Education':
            # Course fees tend to be round numbers
            amount = round(random.choice([499, 999, 1999, 4999, 9999, 19999, 49999]), 2)
            
        elif category == 'Food - Groceries':
            # Weekly grocery shopping
            amount = round(np.random.uniform(500, 5000), 2)
            # Big basket orders on weekends
            if dt.weekday() >= 5:
                amount = round(np.random.uniform(1000, 3000), 2)
                
        else:
            # Default random within range
            amount = round(np.random.uniform(amount_range[0], amount_range[1]), 2)
        
        return amount
    
    def _generate_payment_details(self, category, amount, dt):
        """Generate realistic payment method based on context"""
        # UPI dominates for small amounts
        if amount < 500:
            payment_method = random.choice(['UPI', 'UPI', 'UPI', 'Debit Card', 'Wallet'])
        elif amount < 2000:
            payment_method = random.choice(['UPI', 'UPI', 'Debit Card', 'Credit Card'])
        elif amount < 10000:
            payment_method = random.choice(['Credit Card', 'Debit Card', 'UPI', 'Net Banking'])
        else:
            payment_method = random.choice(['Net Banking', 'Credit Card', 'Debit Card'])
        
        # Payment app based on method
        if payment_method == 'UPI':
            payment_app = random.choice(['GPay', 'PhonePe', 'Paytm', 'Amazon Pay', 'BHIM'])
        elif payment_method in ['Credit Card', 'Debit Card']:
            payment_app = 'Bank App'
        else:
            payment_app = random.choice(['Bank App', 'Paytm', 'PhonePe'])
        
        # Card type
        if payment_method in ['Credit Card', 'Debit Card']:
            card_type = random.choice(['Visa', 'Mastercard', 'Rupay', 'Amex'])
        else:
            card_type = 'N/A'
        
        return payment_method, payment_app, card_type
    
    def _generate_location_details(self):
        """Generate realistic location with consistency"""
        city = random.choice(self.city_suffixes)
        
        # 80% transactions in same city
        if random.random() < 0.8:
            merchant_city = city
            is_same_city = 1
            distance = round(random.uniform(0, 15), 2)  # Within city
        else:
            merchant_city = random.choice([c for c in self.city_suffixes if c != city])
            is_same_city = 0
            distance = round(random.uniform(20, 50), 2)  # Different city
        
        return city, merchant_city, is_same_city, distance
    
    def _generate_user_behavior(self, category, merchant):
        """Generate realistic user behavior patterns"""
        # Recurring transactions
        if category in ['Bills & Utilities', 'Education']:
            is_recurring = random.choice([1, 1, 1, 0])  # 75% recurring
        elif category in ['Food - Delivery', 'Transportation']:
            is_recurring = random.choice([1, 0, 0, 0])  # 25% recurring
        else:
            is_recurring = 0
        
        # Merchant visit count - frequent for food delivery, low for education
        if category in ['Food - Delivery', 'Food - Groceries']:
            merchant_visit_count = random.randint(5, 50)
            is_first_time = 0
        elif category in ['Education', 'Health & Fitness']:
            merchant_visit_count = random.randint(1, 10)
            is_first_time = random.choice([1, 0, 0, 0])
        else:
            merchant_visit_count = random.randint(1, 20)
            is_first_time = random.choice([1, 0, 0])
        
        return is_recurring, merchant_visit_count, is_first_time
    
    def _generate_merchant_trust(self, merchant, category):
        """Generate merchant verification and rating"""
        # Major brands are always verified with high ratings
        major_brands = ['swiggy', 'zomato', 'amazon', 'flipkart', 'uber', 'ola', 
                       'netflix', 'bookmyshow', 'bigbasket']
        
        if any(brand in merchant.lower() for brand in major_brands):
            is_verified = 1
            merchant_rating = round(random.uniform(4.0, 5.0), 1)
        else:
            is_verified = random.choice([1, 1, 1, 0])  # 75% verified
            merchant_rating = round(random.uniform(3.0, 5.0), 1)
        
        return is_verified, merchant_rating
    
    def _get_location_type(self, category, hour):
        """Get realistic location type based on category and time"""
        if category in ['Food - Delivery', 'Shopping - Essentials', 'Shopping - Non-Essentials', 'Entertainment']:
            return 'Online'
        elif category == 'Food - Dining Out':
            return 'Restaurant'
        elif category in ['Food - Groceries', 'Bills & Utilities']:
            if 9 <= hour <= 21:
                return random.choice(['Store', 'Online'])
            else:
                return 'Online'
        elif category == 'Transportation':
            return random.choice(['Street', 'Station', 'Home'])
        else:
            return random.choice(['Online', 'Office', 'Home'])
    
    def _get_avg_monthly_spend(self, category):
        """Get realistic monthly spend by category"""
        monthly_spend_map = {
            'Food - Delivery': round(random.uniform(2000, 8000), 2),
            'Food - Dining Out': round(random.uniform(1500, 5000), 2),
            'Food - Groceries': round(random.uniform(3000, 10000), 2),
            'Transportation': round(random.uniform(1000, 5000), 2),
            'Entertainment': round(random.uniform(500, 2000), 2),
            'Shopping - Essentials': round(random.uniform(2000, 8000), 2),
            'Shopping - Non-Essentials': round(random.uniform(1000, 6000), 2),
            'Education': round(random.uniform(500, 20000), 2),
            'Bills & Utilities': round(random.uniform(1500, 5000), 2),
            'Health & Fitness': round(random.uniform(1000, 5000), 2),
            'Transfer - Personal': round(random.uniform(500, 15000), 2),
            'Other': round(random.uniform(500, 5000), 2),
        }
        return monthly_spend_map.get(category, 2000)
    
    def _calculate_tax(self, amount, category):
        """Calculate realistic GST"""
        # 18% GST on services, 5% on food
        if category in ['Food - Delivery', 'Food - Dining Out']:
            return round(amount * 0.05, 2) if random.random() > 0.3 else 0
        elif category in ['Entertainment', 'Education', 'Health & Fitness']:
            return round(amount * 0.18, 2) if random.random() > 0.2 else 0
        else:
            return 0
    
    def _calculate_tip(self, amount, category, hour):
        """Calculate realistic tips"""
        if category == 'Food - Delivery':
            # More tips during late night
            if hour >= 22 or hour <= 6:
                return round(random.uniform(20, 50), 2) if random.random() > 0.5 else 0
            else:
                return round(random.uniform(10, 30), 2) if random.random() > 0.7 else 0
        elif category == 'Food - Dining Out':
            return round(amount * random.uniform(0.05, 0.15), 2) if random.random() > 0.6 else 0
        else:
            return 0
    
    def _calculate_discount(self, amount, category, day):
        """Calculate realistic discounts"""
        # More discounts during month-end sales
        if day >= 25:
            discount_chance = 0.4
        else:
            discount_chance = 0.7
        
        if category in ['Shopping - Essentials', 'Shopping - Non-Essentials', 'Food - Delivery']:
            return round(amount * random.uniform(0.05, 0.25), 2) if random.random() > discount_chance else 0
        else:
            return 0
    
    def _calculate_cashback(self, amount, payment_method):
        """Calculate realistic cashback"""
        if payment_method == 'UPI':
            # UPI cashback campaigns
            return round(random.uniform(5, min(50, amount * 0.05)), 2) if random.random() > 0.85 else 0
        elif payment_method == 'Credit Card':
            # Credit card rewards
            return round(amount * random.uniform(0.01, 0.05), 2) if random.random() > 0.80 else 0
        else:
            return 0
    
    def _calculate_risk_score(self, amount, hour, is_same_city, is_first_time):
        """Calculate transaction risk score"""
        risk = 0.0
        
        # High amount = higher risk
        if amount > 10000:
            risk += 0.3
        elif amount > 5000:
            risk += 0.2
        
        # Unusual hours
        if hour < 6 or hour > 23:
            risk += 0.2
        
        # Different city
        if not is_same_city:
            risk += 0.1
        
        # First time merchant
        if is_first_time:
            risk += 0.15
        
        # Add some randomness
        risk += random.uniform(0, 0.15)
        
        return round(min(risk, 1.0), 4)
    
    def _get_transaction_status(self, amount, payment_method):
        """Get realistic transaction status"""
        # Higher amounts have slightly more failures
        if amount > 10000:
            return random.choice(['Success', 'Success', 'Success', 'Success', 'Pending', 'Failed'])
        else:
            return random.choice(['Success', 'Success', 'Success', 'Success', 'Success', 'Pending'])
    
if __name__ == "__main__":
    generator = TransactionDataGenerator()
    df = generator.generate_dataset(n_samples=50000)
    df.to_csv('data/transactions.csv', index=False)
    print(f"Generated {len(df)} transactions")
    print(df['category'].value_counts())
