# SmartLabel Evaluation Report

**Generated:** 2025-11-20 00:04:04

---

## Executive Summary

- **Macro F1 Score:** 0.9634
- **Target Achievement:** PASSED (>=0.90)
- **Overall Accuracy:** 0.9610

---

## Per-Category Performance

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Bills & Utilities | 0.986 | 0.971 | 0.978 | 70 |
| Education | 1.000 | 0.989 | 0.994 | 87 |
| Entertainment | 1.000 | 0.968 | 0.984 | 93 |
| Food - Delivery | 1.000 | 0.882 | 0.937 | 93 |
| Food - Dining Out | 0.904 | 0.938 | 0.920 | 80 |
| Food - Groceries | 1.000 | 0.938 | 0.968 | 80 |
| Health & Fitness | 0.961 | 0.986 | 0.973 | 74 |
| Other | 1.000 | 1.000 | 1.000 | 81 |
| Shopping - Essentials | 1.000 | 0.942 | 0.970 | 86 |
| Shopping - Non-Essentials | 0.987 | 0.961 | 0.974 | 77 |
| Transfer - Personal | 0.798 | 1.000 | 0.888 | 99 |
| Transportation | 0.987 | 0.963 | 0.975 | 80 |

---

## Key Insights

### Top Misclassifications

- **Food - Delivery** -> Food - Dining Out (8 times)
- **Food - Dining Out** -> Transfer - Personal (5 times)
- **Food - Groceries** -> Transfer - Personal (4 times)
- **Shopping - Essentials** -> Transfer - Personal (4 times)
- **Entertainment** -> Transfer - Personal (3 times)

---

## Visualizations

![Confusion Matrix](confusion_matrix.png)

![F1 Scores](f1_scores.png)

---

## Recommendations

1. Review and improve categories with F1 < 0.85
2. Add more training data for frequently misclassified patterns
3. Fine-tune confidence thresholds for optimal active learning
