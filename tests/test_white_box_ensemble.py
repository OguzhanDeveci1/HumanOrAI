"""
WHITE BOX TEST CASE 2: Ensemble Prediction Logic Testing
Test ID: WB-TC-002
Risk Level: CRITICAL
Test Type: Statement Coverage + Logic Testing

Tests:
- Line 227: predictions_list extraction
- Line 228: Majority voting logic (>= 3 rule)
- Line 229: Ensemble label assignment
- Line 230: Confidence averaging
- Line 231: Vote count calculation
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock dependencies BEFORE importing app
sys.modules['h2o'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()


class TestEnsembleVotingLogic(unittest.TestCase):
    """White Box Test Case 2: Ensemble Voting Algorithm Statement Coverage"""

    def create_mock_predictor(self, predictions):
        """
        Helper: Mock predictor oluştur
        predictions: dict of {model_name: (prediction, confidence)}
        """
        from app import HumanOrAIPredictor

        predictor = HumanOrAIPredictor()
        predictor.is_initialized = True
        predictor.models = {
            'BERT': {'model': MagicMock(), 'tokenizer': MagicMock()},
            'RoBERTa': {'model': MagicMock(), 'tokenizer': MagicMock()},
            'DRF': MagicMock(),
            'GBM': MagicMock(),
            'GLM': MagicMock()
        }
        predictor.tfidf_vectorizer = MagicMock()
        predictor.h2o_column_names = ['col1', 'col2']
        predictor.device = 'cpu'

        # Mock individual prediction methods
        predictor.predict_bert = MagicMock(return_value=predictions.get('BERT', (1, 0.85)))
        predictor.predict_roberta = MagicMock(return_value=predictions.get('RoBERTa', (1, 0.90)))

        return predictor

    @patch('app.h2o.H2OFrame')
    def test_ensemble_unanimous_human(self, mock_h2o_frame):
        """
        Test Scenario 1: Tüm modeller HUMAN tahmin ediyor (5/5)
        Tests: Line 227-228, 229-230, 232
        """
        print("\n=== Scenario 1: Unanimous HUMAN (5/5) ===")

        from app import HumanOrAIPredictor

        # Arrange: 5/5 HUMAN predictions
        predictor = self.create_mock_predictor({
            'BERT': (1, 85.5),
            'RoBERTa': (1, 90.2)
        })

        # Mock H2O predictions (DRF, GBM, GLM all predict HUMAN=1)
        mock_h2o_pred = MagicMock()
        mock_h2o_pred.as_data_frame.return_value = pd.DataFrame({
            'predict': [1],
            'p0': [0.2],
            'p1': [0.8]
        })
        predictor.models['DRF'].predict = MagicMock(return_value=mock_h2o_pred)
        predictor.models['GBM'].predict = MagicMock(return_value=mock_h2o_pred)
        predictor.models['GLM'].predict = MagicMock(return_value=mock_h2o_pred)

        predictor.tfidf_vectorizer.transform = MagicMock(
            return_value=MagicMock(toarray=lambda: np.array([[0.1, 0.2]]))
        )

        # Mock H2O frame
        mock_h2o_frame.return_value = MagicMock()

        # Act
        result = predictor.predict_all("Test text for unanimous human")

        # Assert: Line 227-232 coverage
        ensemble = result['ensemble']
        self.assertEqual(ensemble['prediction'], 1, "Should predict HUMAN")
        self.assertEqual(ensemble['label'], 'HUMAN')
        self.assertEqual(ensemble['vote_count'], 5, "All 5 models voted HUMAN")
        self.assertEqual(ensemble['total_models'], 5)

        # Line 230: Average confidence calculation
        individual_results = result['individual_results']
        confidences = [r['confidence'] for r in individual_results.values()]
        expected_avg = round(sum(confidences) / len(confidences), 2)
        self.assertEqual(ensemble['confidence'], expected_avg)

        print(f"✅ Ensemble Result: {ensemble['label']}")
        print(f"✅ Vote Count: {ensemble['vote_count']}/5")
        print(f"✅ Avg Confidence: {ensemble['confidence']}%")

    @patch('app.h2o.H2OFrame')
    def test_ensemble_majority_human(self, mock_h2o_frame):
        """
        Test Scenario 2: 3/5 HUMAN (majority voting - edge case)
        Tests: Line 228 (sum(predictions_list) >= 3)
        """
        print("\n=== Scenario 2: Majority HUMAN (3/5) ===")

        from app import HumanOrAIPredictor

        # Arrange: 3 HUMAN, 2 AI
        predictor = self.create_mock_predictor({
            'BERT': (1, 75.0),      # HUMAN
            'RoBERTa': (0, 65.0)    # AI
        })

        # DRF: HUMAN (1), GBM: HUMAN (1), GLM: AI (0)
        mock_h2o_human = MagicMock()
        mock_h2o_human.as_data_frame.return_value = pd.DataFrame({
            'predict': [1], 'p0': [0.3], 'p1': [0.7]
        })

        mock_h2o_ai = MagicMock()
        mock_h2o_ai.as_data_frame.return_value = pd.DataFrame({
            'predict': [0], 'p0': [0.6], 'p1': [0.4]
        })

        predictor.models['DRF'].predict = MagicMock(return_value=mock_h2o_human)
        predictor.models['GBM'].predict = MagicMock(return_value=mock_h2o_human)
        predictor.models['GLM'].predict = MagicMock(return_value=mock_h2o_ai)

        predictor.tfidf_vectorizer.transform = MagicMock(
            return_value=MagicMock(toarray=lambda: np.array([[0.1, 0.2]]))
        )

        mock_h2o_frame.return_value = MagicMock()

        # Act
        result = predictor.predict_all("Test text for majority")

        # Assert: Line 228 - exactly >= 3 condition
        ensemble = result['ensemble']
        vote_count = sum([
            1 if r['prediction'] == 1 else 0
            for r in result['individual_results'].values()
        ])

        self.assertEqual(vote_count, 3, "Should have exactly 3 HUMAN votes")
        self.assertEqual(ensemble['prediction'], 1, "Majority rule: HUMAN wins")
        self.assertEqual(ensemble['label'], 'HUMAN')

        print(f"✅ Vote Count: {vote_count}/5 HUMAN")
        print(f"✅ Ensemble: {ensemble['label']} (majority wins)")

    @patch('app.h2o.H2OFrame')
    def test_ensemble_ai_wins(self, mock_h2o_frame):
        """
        Test Scenario 3: 2/5 HUMAN, 3/5 AI (AI wins)
        Tests: Line 228 else branch (sum < 3)
        """
        print("\n=== Scenario 3: AI Wins (2/5 HUMAN, 3/5 AI) ===")

        from app import HumanOrAIPredictor

        # Arrange: 2 HUMAN, 3 AI
        predictor = self.create_mock_predictor({
            'BERT': (1, 60.0),      # HUMAN
            'RoBERTa': (0, 80.0)    # AI
        })

        # DRF: HUMAN, GBM: AI, GLM: AI
        mock_h2o_ai = MagicMock()
        mock_h2o_ai.as_data_frame.return_value = pd.DataFrame({
            'predict': [0], 'p0': [0.7], 'p1': [0.3]
        })

        mock_h2o_human = MagicMock()
        mock_h2o_human.as_data_frame.return_value = pd.DataFrame({
            'predict': [1], 'p0': [0.4], 'p1': [0.6]
        })

        predictor.models['DRF'].predict = MagicMock(return_value=mock_h2o_human)
        predictor.models['GBM'].predict = MagicMock(return_value=mock_h2o_ai)
        predictor.models['GLM'].predict = MagicMock(return_value=mock_h2o_ai)

        predictor.tfidf_vectorizer.transform = MagicMock(
            return_value=MagicMock(toarray=lambda: np.array([[0.1, 0.2]]))
        )

        mock_h2o_frame.return_value = MagicMock()

        # Act
        result = predictor.predict_all("AI generated text")

        # Assert: sum < 3, so prediction = 0 (AI)
        ensemble = result['ensemble']
        self.assertEqual(ensemble['prediction'], 0, "AI should win")
        self.assertEqual(ensemble['label'], 'AI')
        self.assertEqual(ensemble['vote_count'], 2, "Only 2 voted HUMAN")

        print(f"✅ Vote Count: {ensemble['vote_count']}/5 HUMAN")
        print(f"✅ Ensemble: {ensemble['label']} (AI wins)")

    @patch('app.h2o.H2OFrame')
    def test_ensemble_unanimous_ai(self, mock_h2o_frame):
        """
        Test Scenario 4: 0/5 HUMAN, 5/5 AI (Unanimous AI)
        Tests: Edge case - all models predict AI
        """
        print("\n=== Scenario 4: Unanimous AI (0/5) ===")

        from app import HumanOrAIPredictor

        # Arrange: 0 HUMAN, 5 AI
        predictor = self.create_mock_predictor({
            'BERT': (0, 88.0),      # AI
            'RoBERTa': (0, 92.0)    # AI
        })

        # All H2O models: AI
        mock_h2o_ai = MagicMock()
        mock_h2o_ai.as_data_frame.return_value = pd.DataFrame({
            'predict': [0], 'p0': [0.85], 'p1': [0.15]
        })

        predictor.models['DRF'].predict = MagicMock(return_value=mock_h2o_ai)
        predictor.models['GBM'].predict = MagicMock(return_value=mock_h2o_ai)
        predictor.models['GLM'].predict = MagicMock(return_value=mock_h2o_ai)

        predictor.tfidf_vectorizer.transform = MagicMock(
            return_value=MagicMock(toarray=lambda: np.array([[0.1, 0.2]]))
        )

        mock_h2o_frame.return_value = MagicMock()

        # Act
        result = predictor.predict_all("Clear AI generated content")

        # Assert
        ensemble = result['ensemble']
        self.assertEqual(ensemble['prediction'], 0, "Should predict AI")
        self.assertEqual(ensemble['label'], 'AI')
        self.assertEqual(ensemble['vote_count'], 0, "Zero voted HUMAN")

        print(f"✅ Vote Count: {ensemble['vote_count']}/5 HUMAN")
        print(f"✅ Ensemble: {ensemble['label']} (unanimous AI)")


if __name__ == '__main__':
    unittest.main(verbosity=2)
