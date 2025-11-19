from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import random
import os
from src.hybrid_system import SmartLabelSystem
from src.ml_classifier import MLTransactionClassifier
import json

app = Flask(__name__)

# Initialize the SmartLabel system
print("Loading SmartLabel AI system...")
smart_label = SmartLabelSystem('config/categories.json', confidence_threshold=0.80)

# Load the trained model
try:
    ml_classifier = MLTransactionClassifier()
    ml_classifier.load_model('models/')
    smart_label.ml_classifier = ml_classifier
    print("✓ AI model loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Could not load model - {e}")
    print("Please run 'python main.py' first to train the model")

# Store user feedback
feedback_log = []

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_transaction():
    """Classify a single transaction"""
    try:
        data = request.json
        
        # Parse timestamp
        timestamp_str = data.get('timestamp', '')
        if timestamp_str:
            dt = pd.to_datetime(timestamp_str)
        else:
            dt = pd.Timestamp.now()
        
        merchant = data.get('merchant', '')
        amount = float(data.get('amount', 0))
        
        # Create a COMPLETE transaction object with ALL required features
        transaction = {
            # Basic info
            'transaction_id': 'WEB' + str(pd.Timestamp.now().timestamp())[:10],
            'merchant': merchant,
            'amount': amount,
            'timestamp': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'upi_reference': f'UPI{random.randint(100000000, 999999999)}',
            'category': 'Unknown',  # To be predicted
            
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
            'merchant_has_city': 0,  # Default
            'merchant_has_pvt_ltd': 1 if 'pvt' in merchant.lower() or 'ltd' in merchant.lower() else 0,
            'merchant_all_caps': 1 if merchant.isupper() else 0,
            'merchant_first_char': merchant[0] if merchant else '',
            'merchant_last_char': merchant[-1] if merchant else '',
            
            # Payment features
            'payment_method': data.get('payment_method', 'UPI'),
            'payment_app': data.get('payment_app', 'GPay'),
            'card_type': 'N/A',
            'bank_name': 'HDFC',
            'is_online': 1,
            'is_contactless': 1 if data.get('payment_method', 'UPI') == 'UPI' else 0,
            'requires_otp': 1 if amount > 2000 else 0,
            
            # Location features
            'city': 'Bangalore',
            'location_type': 'Online',
            'merchant_city': 'Bangalore',
            'is_same_city': 1,
            'distance_from_home_km': 5.0,
            
            # User behavior
            'is_recurring': 0,
            'merchant_visit_count': data.get('visit_count', 1),
            'days_since_last_transaction': 7,
            'transaction_sequence_number': 100,
            'time_since_last_txn_hours': 24.0,
            'is_first_time_merchant': 0,
            'merchant_category_frequency': 10,
            'avg_monthly_spend_category': 5000.0,
            'is_unusual_time': 1 if dt.hour < 6 or dt.hour > 23 else 0,
            
            # Transaction context
            'description': f'Payment to {merchant}',
            'narration': f'Paid to {merchant}',
            'merchant_category_code': data.get('mcc', '9999'),
            'currency': 'INR',
            'exchange_rate': 1.0,
            'tax_amount': 0.0,
            'tip_amount': 0.0,
            'discount_amount': 0.0,
            'cashback_amount': 0.0,
            
            # Risk features
            'risk_score': 0.2,
            'is_verified_merchant': 1,
            'merchant_rating': 4.5,
            'transaction_status': 'Success',
            'failure_reason': 'N/A'
        }
        
        # Classify
        result = smart_label.classify_transaction(transaction)
        
        return jsonify({
            'success': True,
            'category': result['category'],
            'confidence': result['confidence'],
            'needs_feedback': result['needs_feedback'],
            'explanation': result['explanation'],
            'alternatives': result['alternative_predictions']
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/classify_bulk', methods=['POST'])
def classify_bulk():
    """Classify multiple transactions from CSV upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Read CSV
        df = pd.read_csv(file)
        
        results = []
        for idx, row in df.iterrows():
            transaction = row.to_dict()
            result = smart_label.classify_transaction(transaction)
            
            results.append({
                'merchant': transaction.get('merchant', ''),
                'amount': transaction.get('amount', 0),
                'predicted_category': result['category'],
                'confidence': result['confidence'],
                'needs_review': result['needs_feedback']
            })
        
        # Save results
        results_df = pd.DataFrame(results)
        output_path = 'data/classified_transactions.csv'
        results_df.to_csv(output_path, index=False)
        
        return jsonify({
            'success': True,
            'total': len(results),
            'needs_review': sum(1 for r in results if r['needs_review']),
            'download_url': '/download/classified'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for a transaction"""
    try:
        data = request.json
        
        transaction = {
            'merchant': data.get('merchant'),
            'amount': data.get('amount'),
            'timestamp': data.get('timestamp')
        }
        
        predicted = data.get('predicted_category')
        correct = data.get('correct_category')
        confidence = data.get('confidence')
        
        # Process feedback
        smart_label.process_feedback(transaction, predicted, correct, confidence)
        
        # Log feedback
        feedback_log.append({
            'merchant': transaction['merchant'],
            'predicted': predicted,
            'correct': correct,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'message': 'Thank you for your feedback! SmartLabel is learning...',
            'total_feedback': len(feedback_log)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/stats')
def get_stats():
    """Get system statistics"""
    try:
        with open('config/categories.json', 'r') as f:
            config = json.load(f)
        
        return jsonify({
            'success': True,
            'total_categories': len(config['categories']),
            'categories': list(config['categories'].keys()),
            'feedback_received': len(feedback_log),
            'model_version': '1.0.0',
            'confidence_threshold': smart_label.confidence_threshold
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/download/classified')
def download_classified():
    """Download classified transactions"""
    path = 'data/classified_transactions.csv'
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

@app.route('/download/template')
def download_template():
    """Download CSV template"""
    template = pd.DataFrame({
        'merchant': ['Swiggy', 'Uber India', 'Netflix India'],
        'amount': [350.50, 245.00, 199.00],
        'timestamp': ['2024-01-15 20:30:00', '2024-01-15 09:15:00', '2024-01-01 10:00:00'],
        'payment_method': ['UPI', 'UPI', 'Credit Card'],
        'hour': [20, 9, 10]
    })
    
    path = 'data/template.csv'
    template.to_csv(path, index=False)
    return send_file(path, as_attachment=True)

@app.route('/favicon.ico')
def favicon():
    """Serve favicon"""
    return '', 204

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create templates directory if not exists
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("\n" + "="*60)
    print("🚀 SMARTLABEL - AUTONOMOUS TRANSACTION CATEGORIZER")
    print("="*60)
    print("\n✓ Server starting on http://localhost:5000")
    print("✓ Open your browser and navigate to the URL above")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
