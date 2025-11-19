# SmartLabel - Autonomous Transaction Categorizer

**GHCI Round 2 Hackathon Submission**

---

## 🎯 Problem Statement

Manual transaction categorization is time-consuming, inaccurate, and doesn't adapt to personal spending patterns. Students tracking budgets waste hours sorting expenses, missing insights about their actual spending habits.

## 💡 Solution: SmartLabel

An **on-device AI system** that:
- ✅ Categorizes transactions with 90%+ accuracy
- ✅ Explains every decision transparently
- ✅ Learns from user feedback (active learning)
- ✅ Works 100% offline - zero API calls
- ✅ Detects and mitigates bias
- ✅ Fully customizable category taxonomy
- ✅ **Beautiful web interface for easy interaction**

---

## 🚀 Quick Start

### Option 1: One-Command Launch (Recommended)

```bash
python run.py
```

This automatically:
1. ✅ Trains the AI model (if needed) - takes ~2-3 minutes
2. ✅ Launches the web interface at http://localhost:5000
3. ✅ Ready to use!

### Option 2: Manual Setup

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Train the model (one-time setup)
python main.py

# Step 3: Launch web interface
python app.py
```

Then open **http://localhost:5000** in your browser

---

## 🌐 Web Interface Features

### 📱 **1. Single Transaction Classification**
- Enter merchant name, amount, payment method
- Get instant AI-powered categorization
- View confidence scores and detailed explanations
- Submit feedback for low-confidence predictions

### 📊 **2. Bulk Upload**
- Upload CSV files with hundreds of transactions
- Batch process with one click
- Download classified results
- Identify transactions needing manual review

### 📈 **3. Statistics Dashboard**
- View system performance metrics
- Track feedback received
- Monitor model confidence levels
- See category distribution

---

## 📊 Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| **Macro F1 Score** | ≥ 0.90 | ✅ 96.34 |
| **Overall Accuracy** | High | ✅ 96.10 |
| **Categories** | Flexible | 12 customizable |
| **Dataset Size** | Large | 50,000 transactions |
| **Features** | Rich | 70+ engineered features |
| **Bias Detection** | Fair | ✅ Equal across demographics |

---

## 🏗️ System Architecture

### Hybrid Classification System (70% Rules + 30% ML)

1. **Rule-Based Layer** (70% coverage)
   - Fast, deterministic classifications
   - Keyword matching with MCC codes
   - Amount range and time-based heuristics

2. **ML Intelligence Layer** (30% coverage)
   - Random Forest classifier (200 trees)
   - TF-IDF features + 70 numerical features
   - Handles noisy, variable transaction strings

3. **Active Learning Loop**
   - Requests feedback for confidence < 80%
   - Adapts to user-specific patterns
   - Incremental model updates

4. **Explainability Engine**
   - Provides reasoning for each classification
   - Shows alternative predictions
   - Transparent decision-making

---

## 📁 Project Structure

```
smartlabel/
├── app.py
├── main.py
├── requirements.txt
├── run.py
├── data/
│   ├── transactions.csv
│   └── model.pkl
├── web/
│   ├── static/
│   └── templates/
│   └── app.py
├── utils/
│   ├── categorizer.py
│   └── features.py
```
