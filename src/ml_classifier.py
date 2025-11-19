import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import json
from datetime import datetime

class MLTransactionClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),
            analyzer='char_wb'
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Store categorical columns for consistent encoding
        self.mcc_categories = None
        self.payment_categories = None
        self.payment_app_categories = None
        
    def extract_features(self, df, fit=False):
        """Extract features from transactions - now using 70 comprehensive features"""
        features = {}
        
        # 1. Text features from merchant name (500 features from TF-IDF)
        if fit:
            merchant_features = self.vectorizer.fit_transform(df['merchant'].fillna(''))
        else:
            merchant_features = self.vectorizer.transform(df['merchant'].fillna(''))
        features['merchant_tfidf'] = merchant_features.toarray()
        
        # 2. Numerical and categorical features
        numerical = []
        
        # Amount features (7 features)
        numerical.append(df['amount'].values.reshape(-1, 1))
        numerical.append(df['amount_log'].values.reshape(-1, 1))
        numerical.append(df['amount_rounded'].values.reshape(-1, 1))
        numerical.append(df['is_round_amount'].values.reshape(-1, 1))
        numerical.append(df['amount_last_digit'].values.reshape(-1, 1))
        numerical.append(df['is_large_transaction'].values.reshape(-1, 1))
        numerical.append(df['is_micro_transaction'].values.reshape(-1, 1))
        
        # Time features (9 features)
        numerical.append(df['hour'].values.reshape(-1, 1))
        numerical.append(df['day_of_week'].values.reshape(-1, 1))
        numerical.append(df['is_weekend'].values.reshape(-1, 1))
        numerical.append(df['is_month_start'].values.reshape(-1, 1))
        numerical.append(df['is_month_end'].values.reshape(-1, 1))
        numerical.append(df['month'].values.reshape(-1, 1))
        numerical.append(df['day'].values.reshape(-1, 1))
        numerical.append(df['minute'].values.reshape(-1, 1))
        numerical.append(df['is_unusual_time'].values.reshape(-1, 1))
        
        # Merchant text features (11 features - excluding char features)
        numerical.append(df['merchant_length'].values.reshape(-1, 1))
        numerical.append(df['merchant_word_count'].values.reshape(-1, 1))
        numerical.append(df['merchant_uppercase_count'].values.reshape(-1, 1))
        numerical.append(df['merchant_lowercase_count'].values.reshape(-1, 1))
        numerical.append(df['merchant_digit_count'].values.reshape(-1, 1))
        numerical.append(df['merchant_special_char_count'].values.reshape(-1, 1))
        numerical.append(df['merchant_has_upi_prefix'].values.reshape(-1, 1))
        numerical.append(df['merchant_has_numbers'].values.reshape(-1, 1))
        numerical.append(df['merchant_has_city'].values.reshape(-1, 1))
        numerical.append(df['merchant_has_pvt_ltd'].values.reshape(-1, 1))
        numerical.append(df['merchant_all_caps'].values.reshape(-1, 1))
        
        # Payment method features (3 features)
        numerical.append(df['is_online'].values.reshape(-1, 1))
        numerical.append(df['is_contactless'].values.reshape(-1, 1))
        numerical.append(df['requires_otp'].values.reshape(-1, 1))
        
        # Location features (2 features)
        numerical.append(df['is_same_city'].values.reshape(-1, 1))
        numerical.append(df['distance_from_home_km'].values.reshape(-1, 1))
        
        # User behavior features (8 features)
        numerical.append(df['is_recurring'].values.reshape(-1, 1))
        numerical.append(df['merchant_visit_count'].values.reshape(-1, 1))
        numerical.append(df['days_since_last_transaction'].values.reshape(-1, 1))
        numerical.append(df['transaction_sequence_number'].values.reshape(-1, 1))
        numerical.append(df['time_since_last_txn_hours'].values.reshape(-1, 1))
        numerical.append(df['is_first_time_merchant'].values.reshape(-1, 1))
        numerical.append(df['merchant_category_frequency'].values.reshape(-1, 1))
        numerical.append(np.log1p(df['avg_monthly_spend_category'].values).reshape(-1, 1))
        
        # Transaction context features (4 features)
        numerical.append(df['tax_amount'].values.reshape(-1, 1))
        numerical.append(df['tip_amount'].values.reshape(-1, 1))
        numerical.append(df['discount_amount'].values.reshape(-1, 1))
        numerical.append(df['cashback_amount'].values.reshape(-1, 1))
        
        # Risk features (3 features)
        numerical.append(df['risk_score'].values.reshape(-1, 1))
        numerical.append(df['is_verified_merchant'].values.reshape(-1, 1))
        numerical.append(df['merchant_rating'].values.reshape(-1, 1))
        
        # Categorical encoding with consistency
        if fit:
            # Store categories for later
            self.mcc_categories = df['merchant_category_code'].unique().tolist()
            self.payment_categories = df['payment_method'].unique().tolist()
            self.payment_app_categories = df['payment_app'].unique().tolist()
        
        # One-hot encoding with consistent categories
        mcc_dummies = pd.get_dummies(df['merchant_category_code'], prefix='mcc')
        # Ensure all expected columns exist
        for cat in self.mcc_categories:
            col = f'mcc_{cat}'
            if col not in mcc_dummies.columns:
                mcc_dummies[col] = 0
        mcc_dummies = mcc_dummies[[f'mcc_{cat}' for cat in self.mcc_categories]]
        
        payment_dummies = pd.get_dummies(df['payment_method'], prefix='payment')
        for cat in self.payment_categories:
            col = f'payment_{cat}'
            if col not in payment_dummies.columns:
                payment_dummies[col] = 0
        payment_dummies = payment_dummies[[f'payment_{cat}' for cat in self.payment_categories]]
        
        payment_app_dummies = pd.get_dummies(df['payment_app'], prefix='app')
        for cat in self.payment_app_categories:
            col = f'app_{cat}'
            if col not in payment_app_dummies.columns:
                payment_app_dummies[col] = 0
        payment_app_dummies = payment_app_dummies[[f'app_{cat}' for cat in self.payment_app_categories]]
        
        # Combine numerical features
        numerical_features = np.hstack(numerical)
        
        # Combine with categorical features
        categorical_features = np.hstack([
            mcc_dummies.values,
            payment_dummies.values,
            payment_app_dummies.values
        ])
        
        # Scale numerical features
        combined_numerical_categorical = np.hstack([numerical_features, categorical_features])
        
        if fit:
            features['numerical'] = self.scaler.fit_transform(combined_numerical_categorical)
        else:
            features['numerical'] = self.scaler.transform(combined_numerical_categorical)
        
        # Combine all features (TF-IDF + Numerical + Categorical)
        combined = np.hstack([features['merchant_tfidf'], features['numerical']])
        
        print(f"Feature matrix shape: {combined.shape} (rows × features)")
        
        return combined
    
    def train(self, df, test_size=0.2):
        """Train the model"""
        print("Extracting features...")
        X = self.extract_features(df, fit=True)
        y = self.label_encoder.fit_transform(df['category'])
        
        print("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print(f"\nMacro F1 Score: {macro_f1:.4f}")
        
        # Detailed classification report
        categories = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=categories)
        print("\nClassification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        self.is_fitted = True
        
        return {
            'macro_f1': macro_f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'categories': categories.tolist(),
            'test_predictions': {
                'y_true': y_test,
                'y_pred': y_pred
            }
        }
    
    def predict_proba(self, transactions):
        """Predict with probability scores"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction. Call train() first or load a saved model.")
        
        df = pd.DataFrame(transactions)
        X = self.extract_features(df, fit=False)
        
        probabilities = self.model.predict_proba(X)
        predictions = self.model.predict(X)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            category = self.label_encoder.classes_[pred]
            confidence = probs[pred]
            
            # Get top 3 predictions
            top_3_idx = np.argsort(probs)[-3:][::-1]
            top_3 = [(self.label_encoder.classes_[idx], probs[idx]) for idx in top_3_idx]
            
            results.append({
                'category': category,
                'confidence': float(confidence),
                'top_3_predictions': top_3,
                'method': 'ml-model'
            })
        
        return results
    
    def get_feature_importance(self):
        """Get feature importance for explainability"""
        importance = self.model.feature_importances_
        return importance
    
    def save_model(self, path='models/'):
        """Save trained model"""
        joblib.dump(self.model, f'{path}classifier.pkl')
        joblib.dump(self.vectorizer, f'{path}vectorizer.pkl')
        joblib.dump(self.scaler, f'{path}scaler.pkl')
        joblib.dump(self.label_encoder, f'{path}label_encoder.pkl')
        
        # Save categorical columns info
        categorical_info = {
            'mcc_categories': self.mcc_categories,
            'payment_categories': self.payment_categories,
            'payment_app_categories': self.payment_app_categories
        }
        with open(f'{path}categorical_info.json', 'w') as f:
            json.dump(categorical_info, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path='models/'):
        """Load trained model"""
        self.model = joblib.load(f'{path}classifier.pkl')
        self.vectorizer = joblib.load(f'{path}vectorizer.pkl')
        self.scaler = joblib.load(f'{path}scaler.pkl')
        self.label_encoder = joblib.load(f'{path}label_encoder.pkl')
        
        # Load categorical columns info
        with open(f'{path}categorical_info.json', 'r') as f:
            categorical_info = json.load(f)
        self.mcc_categories = categorical_info['mcc_categories']
        self.payment_categories = categorical_info['payment_categories']
        self.payment_app_categories = categorical_info['payment_app_categories']
        
        self.is_fitted = True
        print(f"Model loaded from {path}")
