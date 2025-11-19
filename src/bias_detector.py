import pandas as pd
import numpy as np
from collections import defaultdict

class BiasDetector:
    def __init__(self):
        self.bias_metrics = {}
        
    def analyze_bias(self, predictions_df):
        """
        Analyze model for potential biases with enhanced features
        predictions_df should have: merchant, amount, true_category, predicted_category, confidence, plus new features
        """
        print("\n" + "="*60)
        print("BIAS ANALYSIS REPORT")
        print("="*60)
        
        # 1. Amount-based bias
        self._analyze_amount_bias(predictions_df)
        
        # 2. Merchant type bias (local vs brands)
        self._analyze_merchant_bias(predictions_df)
        
        # 3. Confidence distribution across categories
        self._analyze_confidence_bias(predictions_df)
        
        # 4. Error rate by category
        self._analyze_category_bias(predictions_df)
        
        # 5. NEW: Time-based bias analysis
        self._analyze_time_bias(predictions_df)
        
        # 6. NEW: Payment method bias
        self._analyze_payment_method_bias(predictions_df)
        
        # 7. NEW: Geographic bias
        self._analyze_geographic_bias(predictions_df)
        
        return self.bias_metrics
    
    def _analyze_amount_bias(self, df):
        """Check if accuracy varies by transaction amount"""
        print("\n1. AMOUNT-BASED BIAS ANALYSIS")
        print("-" * 60)
        
        # Create amount bins
        df['amount_bin'] = pd.cut(df['amount'], 
                                   bins=[0, 100, 500, 1000, 5000, float('inf')],
                                   labels=['<₹100', '₹100-500', '₹500-1K', '₹1K-5K', '>₹5K'])
        
        bias_data = []
        for bin_name in df['amount_bin'].cat.categories:
            bin_df = df[df['amount_bin'] == bin_name]
            if len(bin_df) > 0:
                accuracy = (bin_df['true_category'] == bin_df['predicted_category']).mean()
                avg_confidence = bin_df['confidence'].mean()
                count = len(bin_df)
                
                bias_data.append({
                    'amount_range': bin_name,
                    'count': count,
                    'accuracy': accuracy,
                    'avg_confidence': avg_confidence
                })
                
                print(f"{bin_name:12s}: Accuracy={accuracy:.3f}, "
                      f"Avg Confidence={avg_confidence:.3f}, Count={count}")
        
        self.bias_metrics['amount_bias'] = bias_data
        
        # Check for significant bias
        accuracies = [d['accuracy'] for d in bias_data]
        if max(accuracies) - min(accuracies) > 0.1:
            print(f"\n⚠️  WARNING: Significant amount-based bias detected "
                  f"(accuracy range: {min(accuracies):.3f} - {max(accuracies):.3f})")
        else:
            print(f"\n✓ Amount-based bias is within acceptable range")
    
    def _analyze_merchant_bias(self, df):
        """Check bias towards brand names vs local merchants"""
        print("\n2. MERCHANT TYPE BIAS ANALYSIS")
        print("-" * 60)
        
        # Define major brands
        major_brands = ['swiggy', 'zomato', 'amazon', 'flipkart', 'uber', 'ola', 
                        'netflix', 'prime', 'bookmyshow', 'domino', 'mcdonald', 'kfc']
        
        df['is_major_brand'] = df['merchant'].str.lower().apply(
            lambda x: any(brand in x for brand in major_brands)
        )
        
        for is_brand in [True, False]:
            label = "Major Brands" if is_brand else "Local Merchants"
            subset = df[df['is_major_brand'] == is_brand]
            
            if len(subset) > 0:
                accuracy = (subset['true_category'] == subset['predicted_category']).mean()
                avg_confidence = subset['confidence'].mean()
                count = len(subset)
                
                print(f"{label:20s}: Accuracy={accuracy:.3f}, "
                      f"Avg Confidence={avg_confidence:.3f}, Count={count}")
        
        brand_acc = (df[df['is_major_brand']]['true_category'] == 
                     df[df['is_major_brand']]['predicted_category']).mean()
        local_acc = (df[~df['is_major_brand']]['true_category'] == 
                     df[~df['is_major_brand']]['predicted_category']).mean()
        
        if abs(brand_acc - local_acc) > 0.05:
            print(f"\n⚠️  WARNING: Bias towards {'brands' if brand_acc > local_acc else 'local merchants'} detected")
        else:
            print(f"\n✓ No significant merchant type bias")
        
        self.bias_metrics['merchant_bias'] = {
            'brand_accuracy': brand_acc,
            'local_accuracy': local_acc
        }
    
    def _analyze_confidence_bias(self, df):
        """Analyze confidence distribution across categories"""
        print("\n3. CONFIDENCE DISTRIBUTION BY CATEGORY")
        print("-" * 60)
        
        confidence_by_category = df.groupby('predicted_category')['confidence'].agg(['mean', 'std', 'count'])
        
        print(confidence_by_category.to_string())
        
        # Check for over-confident or under-confident categories
        for category, row in confidence_by_category.iterrows():
            if row['mean'] > 0.95 and row['count'] > 50:
                print(f"\n⚠️  Category '{category}' may be over-confident (avg: {row['mean']:.3f})")
            elif row['mean'] < 0.60 and row['count'] > 50:
                print(f"\n⚠️  Category '{category}' may be under-confident (avg: {row['mean']:.3f})")
        
        self.bias_metrics['confidence_distribution'] = confidence_by_category.to_dict()
    
    def _analyze_category_bias(self, df):
        """Analyze error rates by category"""
        print("\n4. ERROR RATE BY CATEGORY")
        print("-" * 60)
        
        for category in df['true_category'].unique():
            cat_df = df[df['true_category'] == category]
            if len(cat_df) > 0:
                accuracy = (cat_df['true_category'] == cat_df['predicted_category']).mean()
                error_rate = 1 - accuracy
                count = len(cat_df)
                
                status = "✓" if error_rate < 0.15 else "⚠️"
                print(f"{status} {category:30s}: Error Rate={error_rate:.3f}, Count={count}")
        
        print("\n" + "="*60)
    
    def _analyze_time_bias(self, df):
        """Check if accuracy varies by time of day"""
        print("\n5. TIME-BASED BIAS ANALYSIS")
        print("-" * 60)
        
        if 'hour' not in df.columns:
            print("Hour data not available")
            return
        
        # Create time bins
        df['time_period'] = pd.cut(df['hour'], 
                                    bins=[0, 6, 12, 18, 24],
                                    labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'],
                                    include_lowest=True)
        
        for period in df['time_period'].cat.categories:
            period_df = df[df['time_period'] == period]
            if len(period_df) > 0:
                accuracy = (period_df['true_category'] == period_df['predicted_category']).mean()
                count = len(period_df)
                print(f"{str(period):20s}: Accuracy={accuracy:.3f}, Count={count}")
        
        print("\n✓ Time-based bias analysis complete")
    
    def _analyze_payment_method_bias(self, df):
        """Check if accuracy varies by payment method"""
        print("\n6. PAYMENT METHOD BIAS ANALYSIS")
        print("-" * 60)
        
        if 'payment_method' not in df.columns:
            print("Payment method data not available")
            return
        
        for method in df['payment_method'].unique():
            method_df = df[df['payment_method'] == method]
            if len(method_df) > 10:
                accuracy = (method_df['true_category'] == method_df['predicted_category']).mean()
                count = len(method_df)
                print(f"{method:20s}: Accuracy={accuracy:.3f}, Count={count}")
        
        print("\n✓ Payment method bias analysis complete")
    
    def _analyze_geographic_bias(self, df):
        """Check if accuracy varies by location"""
        print("\n7. GEOGRAPHIC BIAS ANALYSIS")
        print("-" * 60)
        
        if 'city' not in df.columns:
            print("City data not available")
            return
        
        city_stats = []
        for city in df['city'].unique():
            city_df = df[df['city'] == city]
            if len(city_df) > 10:
                accuracy = (city_df['true_category'] == city_df['predicted_category']).mean()
                count = len(city_df)
                city_stats.append({'city': city, 'accuracy': accuracy, 'count': count})
        
        # Show top 5 cities
        city_stats.sort(key=lambda x: x['count'], reverse=True)
        for stat in city_stats[:5]:
            print(f"{stat['city']:20s}: Accuracy={stat['accuracy']:.3f}, Count={stat['count']}")
        
        print("\n✓ Geographic bias analysis complete")
    
    def generate_fairness_report(self):
        """Generate comprehensive fairness report"""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'metrics': self.bias_metrics,
            'recommendations': []
        }
        
        # Generate recommendations based on detected biases
        if 'amount_bias' in self.bias_metrics:
            amount_data = self.bias_metrics['amount_bias']
            accuracies = [d['accuracy'] for d in amount_data]
            if max(accuracies) - min(accuracies) > 0.1:
                report['recommendations'].append(
                    "Consider adding more training data for amount ranges with lower accuracy"
                )
        
        if 'merchant_bias' in self.bias_metrics:
            mb = self.bias_metrics['merchant_bias']
            if abs(mb['brand_accuracy'] - mb['local_accuracy']) > 0.05:
                report['recommendations'].append(
                    "Balance training data between major brands and local merchants"
                )
        
        return report
