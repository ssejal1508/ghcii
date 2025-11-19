from flask import Flask, render_template, request, jsonify, send_file, session
import pandas as pd
import numpy as np
import random
import os
from src.hybrid_system import SmartLabelSystem
from src.ml_classifier import MLTransactionClassifier
from src.feature_extractor import FeatureExtractor
from src.embedding_generator import EmbeddingGenerator
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'smartlabel_secret_key_2024'  # For session management

# Initialize the SmartLabel system
print("Loading SmartLabel AI system...")
smart_label = SmartLabelSystem('config/categories.json', confidence_threshold=0.80)
feature_extractor = FeatureExtractor()
embedding_generator = EmbeddingGenerator()

# Load the trained model
try:
    ml_classifier = MLTransactionClassifier()
    ml_classifier.load_model('models/')
    smart_label.ml_classifier = ml_classifier
    
    # Load embeddings
    embedding_generator.load('models/embeddings.pkl')
    print("✓ AI model and embeddings loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Could not load model - {e}")
    print("Please run 'python main.py' first to train the model")

# Store user feedback and transaction history
feedback_log = []
transaction_history = []  # Store all classified transactions

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_transaction():
    """Classify a single transaction"""
    try:
        data = request.json
        
        # Extract comprehensive features from raw transaction
        raw_transaction = {
            'merchant': data.get('merchant', ''),
            'amount': data.get('amount', 0),
            'timestamp': data.get('timestamp', ''),
            'payment_method': data.get('payment_method', 'UPI'),
            'payment_app': data.get('payment_app', ''),
            'city': data.get('city', ''),
            'visit_count': data.get('visit_count', None),
            'mcc': data.get('mcc', ''),
        }
        
        # Use feature extractor to intelligently extract all 70+ features
        transaction = feature_extractor.extract_from_transaction(raw_transaction)
        
        # Generate embeddings
        merchant = transaction.get('merchant', '')
        description = transaction.get('description', '')
        embedding = embedding_generator.generate_embeddings(merchant, description)
        
        # Find similar transactions from history
        similar_transactions = []
        if len(transaction_history) > 0:
            history_embeddings = [t['embedding'] for t in transaction_history]
            similar_indices = embedding_generator.find_similar_transactions(
                embedding, history_embeddings, top_k=3
            )
            similar_transactions = [
                {
                    'merchant': transaction_history[idx]['merchant'],
                    'category': transaction_history[idx]['category'],
                    'amount': transaction_history[idx]['amount'],
                    'similarity': round(sim * 100, 1)
                }
                for idx, sim in similar_indices if sim > 0.3
            ]
        
        # Classify
        result = smart_label.classify_transaction(transaction)
        
        # Store in history
        history_entry = {
            'id': len(transaction_history) + 1,
            'merchant': merchant,
            'amount': transaction.get('amount', 0),
            'category': result['category'],
            'confidence': result['confidence'],
            'timestamp': transaction.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            'payment_method': transaction.get('payment_method', 'UPI'),
            'embedding': embedding.tolist(),
            'needs_review': result['needs_feedback']
        }
        transaction_history.append(history_entry)
        
        # Keep only last 100 transactions in memory
        if len(transaction_history) > 100:
            transaction_history.pop(0)
        
        return jsonify({
            'success': True,
            'category': result['category'],
            'confidence': result['confidence'],
            'needs_feedback': result['needs_feedback'],
            'explanation': result['explanation'],
            'alternatives': result['alternative_predictions'],
            'similar_transactions': similar_transactions,
            'transaction_id': history_entry['id']
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
            # Extract features from each transaction
            raw_transaction = row.to_dict()
            transaction = feature_extractor.extract_from_transaction(raw_transaction)
            
            # Classify
            result = smart_label.classify_transaction(transaction)
            
            results.append({
                'merchant': raw_transaction.get('merchant', ''),
                'amount': raw_transaction.get('amount', 0),
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
        import traceback
        traceback.print_exc()
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

@app.route('/history', methods=['GET'])
def get_history():
    """Get transaction history"""
    try:
        # Get last N transactions
        limit = int(request.args.get('limit', 20))
        
        recent_transactions = transaction_history[-limit:]
        recent_transactions.reverse()  # Most recent first
        
        # Remove embeddings from response (too large)
        response_data = [
            {
                'id': t['id'],
                'merchant': t['merchant'],
                'amount': t['amount'],
                'category': t['category'],
                'confidence': t['confidence'],
                'timestamp': t['timestamp'],
                'payment_method': t['payment_method'],
                'needs_review': t['needs_review']
            }
            for t in recent_transactions
        ]
        
        return jsonify({
            'success': True,
            'transactions': response_data,
            'total': len(transaction_history)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/history/clear', methods=['POST'])
def clear_history():
    """Clear transaction history"""
    try:
        transaction_history.clear()
        return jsonify({
            'success': True,
            'message': 'History cleared successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Get spending analytics from history"""
    try:
        if len(transaction_history) == 0:
            return jsonify({
                'success': True,
                'analytics': {
                    'total_transactions': 0,
                    'total_spent': 0,
                    'by_category': {},
                    'by_payment_method': {},
                    'avg_confidence': 0
                }
            })
        
        # Calculate analytics
        total_spent = sum(t['amount'] for t in transaction_history)
        avg_confidence = sum(t['confidence'] for t in transaction_history) / len(transaction_history)
        
        # By category
        by_category = {}
        for t in transaction_history:
            cat = t['category']
            if cat not in by_category:
                by_category[cat] = {'count': 0, 'amount': 0}
            by_category[cat]['count'] += 1
            by_category[cat]['amount'] += t['amount']
        
        # By payment method
        by_payment = {}
        for t in transaction_history:
            pm = t['payment_method']
            if pm not in by_payment:
                by_payment[pm] = {'count': 0, 'amount': 0}
            by_payment[pm]['count'] += 1
            by_payment[pm]['amount'] += t['amount']
        
        return jsonify({
            'success': True,
            'analytics': {
                'total_transactions': len(transaction_history),
                'total_spent': round(total_spent, 2),
                'by_category': by_category,
                'by_payment_method': by_payment,
                'avg_confidence': round(avg_confidence, 3),
                'needs_review_count': sum(1 for t in transaction_history if t['needs_review'])
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

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
