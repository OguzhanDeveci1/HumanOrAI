"""
WHITE BOX TEST CASE 1: Model Initialization Code Path Testing
Test ID: WB-TC-001
Risk Level: HIGH
Test Type: Code Path Coverage (Decision Coverage)

Tests:
- Line 28-29: is_initialized flag check
- Line 34-42: BERT model loading path
- Line 44-52: RoBERTa model loading path
- Line 57-64: TF-IDF pickle/joblib fallback
- Line 66-79: H2O models loading
- Line 82-88: Column names extraction
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock dependencies BEFORE importing app
sys.modules['h2o'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['joblib'] = MagicMock()


class TestModelInitializationCodePath(unittest.TestCase):
    """White Box Test Case 1: Model Initialization Code Paths"""

    def setUp(self):
        """Test öncesi hazırlık"""
        self.torch_mock = MagicMock()
        self.torch_mock.device.return_value = 'cpu'
        self.torch_mock.cuda.is_available.return_value = False

    @patch('app.h2o')
    @patch('app.BertForSequenceClassification')
    @patch('app.BertTokenizer')
    @patch('app.RobertaForSequenceClassification')
    @patch('app.RobertaTokenizer')
    @patch('builtins.open', new_callable=mock_open, read_data=b'tfidf_data')
    @patch('pickle.load')
    @patch('app.torch')
    def test_initialize_all_code_paths(self, mock_torch, mock_pickle,
                                       mock_file, mock_roberta_tok,
                                       mock_roberta_model, mock_bert_tok,
                                       mock_bert_model, mock_h2o):
        """
        Test Case 1: Tüm model yükleme kod yollarını test et

        Code Coverage Areas:
        - Line 28-29: is_initialized check
        - Line 34-42: BERT loading path
        - Line 44-52: RoBERTa loading path
        - Line 57-64: TF-IDF pickle/joblib fallback
        - Line 66-79: H2O models loading
        - Line 82-88: Column names extraction
        """
        from app import HumanOrAIPredictor

        # Arrange: Mock setup
        mock_torch.device.return_value = 'cpu'
        mock_torch.cuda.is_available.return_value = False

        # Mock BERT
        mock_bert_instance = MagicMock()
        mock_bert_instance.to.return_value = mock_bert_instance
        mock_bert_instance.eval.return_value = mock_bert_instance
        mock_bert_model.from_pretrained.return_value = mock_bert_instance
        mock_bert_tok.from_pretrained.return_value = MagicMock()

        # Mock RoBERTa
        mock_roberta_instance = MagicMock()
        mock_roberta_instance.to.return_value = mock_roberta_instance
        mock_roberta_instance.eval.return_value = mock_roberta_instance
        mock_roberta_model.from_pretrained.return_value = mock_roberta_instance
        mock_roberta_tok.from_pretrained.return_value = MagicMock()

        mock_pickle.return_value = MagicMock()

        mock_h2o.init.return_value = None

        # Mock H2O model with varimp
        mock_drf_model = MagicMock()
        mock_varimp_df = MagicMock()
        mock_varimp_df.tolist.return_value = ['col1', 'col2', 'col3']
        mock_drf_model.varimp.return_value = {'variable': mock_varimp_df}
        mock_h2o.load_model.return_value = mock_drf_model

        predictor = HumanOrAIPredictor(models_dir='test_models')

        # Act: İlk initialize çağrısı
        print("\n--- First initialize() call ---")
        self.assertFalse(predictor.is_initialized)
        predictor.initialize()

        # Assert: İlk çağrı sonuçları
        self.assertTrue(predictor.is_initialized)  # Line 90 coverage
        self.assertIn('BERT', predictor.models)    # Line 37-40 coverage
        self.assertIn('RoBERTa', predictor.models) # Line 47-50 coverage
        self.assertIn('DRF', predictor.models)     # Line 73 coverage
        self.assertIn('GBM', predictor.models)     # Line 76 coverage
        self.assertIn('GLM', predictor.models)     # Line 79 coverage

        # Assert: Device selection (Line 25)
        self.assertEqual(predictor.device, 'cpu')

        # Act: İkinci initialize çağrısı (idempotency test)
        print("--- Second initialize() call (early return test) ---")
        call_count_before = mock_bert_model.from_pretrained.call_count
        predictor.initialize()
        call_count_after = mock_bert_model.from_pretrained.call_count

        # Assert: İkinci çağrı erken döndü (Line 28-29 branch coverage)
        self.assertEqual(call_count_before, call_count_after,
                        "Second initialize should return early without loading models")

        print("\n[PASS] Test PASSED: All code paths covered")
        print(f"   - is_initialized flag: OK")
        print(f"   - BERT loading path: OK")
        print(f"   - RoBERTa loading path: OK")
        print(f"   - TF-IDF loading path: OK")
        print(f"   - H2O models loading path: OK")
        print(f"   - Early return on second call: OK")

    @patch('app.h2o')
    @patch('app.BertForSequenceClassification')
    @patch('app.BertTokenizer')
    @patch('app.RobertaForSequenceClassification')
    @patch('app.RobertaTokenizer')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    @patch('app.torch')
    def test_tfidf_pickle_fallback_path(self, mock_torch, mock_pickle,
                                        mock_file, mock_roberta_tok,
                                        mock_roberta_model, mock_bert_tok,
                                        mock_bert_model, mock_h2o):
        """
        Test Case 1.1: TF-IDF pickle failure -> joblib fallback (Line 60-64)
        """
        from app import HumanOrAIPredictor

        # Arrange: Mock models
        mock_torch.device.return_value = 'cpu'
        mock_torch.cuda.is_available.return_value = False

        mock_bert_instance = MagicMock()
        mock_bert_instance.to.return_value = mock_bert_instance
        mock_bert_instance.eval.return_value = mock_bert_instance
        mock_bert_model.from_pretrained.return_value = mock_bert_instance
        mock_bert_tok.from_pretrained.return_value = MagicMock()

        mock_roberta_instance = MagicMock()
        mock_roberta_instance.to.return_value = mock_roberta_instance
        mock_roberta_instance.eval.return_value = mock_roberta_instance
        mock_roberta_model.from_pretrained.return_value = mock_roberta_instance
        mock_roberta_tok.from_pretrained.return_value = MagicMock()

        # Pickle başarısız olacak
        mock_pickle.side_effect = Exception("Pickle load failed")

        # Joblib mock (runtime import için sys.modules kullanıldı)
        # sys.modules['joblib'] zaten yukarıda mock'landı
        mock_joblib_load = MagicMock(return_value=MagicMock())
        sys.modules['joblib'].load = mock_joblib_load

        mock_h2o.init.return_value = None
        mock_h2o.load_model.return_value = MagicMock()

        predictor = HumanOrAIPredictor()

        # Act
        predictor.initialize()

        # Assert: Joblib fallback çalıştı mı?
        mock_joblib_load.assert_called_once()
        print("\n[PASS] Pickle fallback path tested: joblib.load was called")


if __name__ == '__main__':
    unittest.main(verbosity=2)
