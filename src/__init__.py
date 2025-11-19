"""
SmartLabel - Autonomous Transaction Categorizer
"""

__version__ = "1.0.0"

from .data_generator import TransactionDataGenerator
from .rule_engine import RuleBasedClassifier
from .ml_classifier import MLTransactionClassifier
from .hybrid_system import SmartLabelSystem
from .bias_detector import BiasDetector
from .evaluator import SystemEvaluator

__all__ = [
    'TransactionDataGenerator',
    'RuleBasedClassifier',
    'MLTransactionClassifier',
    'SmartLabelSystem',
    'BiasDetector',
    'SystemEvaluator'
]
