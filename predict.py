import os
import torch
import pickle
import h2o
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

class HumanOrAIPredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.predictions = {}
        self.h2o_column_names = None  # Store H2O expected column names

        print("Loading models...")
        self._load_models()
        print("All models loaded successfully!\n")

    def _load_models(self):
        # Load BERT model
        print("Loading BERT model...")
        bert_path = os.path.join(self.models_dir, 'bert_model')
        self.models['BERT'] = {
            'model': BertForSequenceClassification.from_pretrained(bert_path),
            'tokenizer': BertTokenizer.from_pretrained(bert_path)
        }
        self.models['BERT']['model'].eval()

        # Load RoBERTa model
        print("Loading RoBERTa model...")
        roberta_path = os.path.join(self.models_dir, 'roberta_model')
        self.models['RoBERTa'] = {
            'model': RobertaForSequenceClassification.from_pretrained(roberta_path),
            'tokenizer': RobertaTokenizer.from_pretrained(roberta_path)
        }
        self.models['RoBERTa']['model'].eval()

        # Load TF-IDF vectorizer for H2O models
        print("Loading TF-IDF vectorizer...")
        tfidf_path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
        try:
            with open(tfidf_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
        except Exception as e:
            print(f"Error loading with pickle: {e}")
            print("Trying with joblib...")
            import joblib
            self.tfidf_vectorizer = joblib.load(tfidf_path)

        # Initialize H2O
        print("Initializing H2O...")
        h2o.init()

        # Load H2O models (DRF, GBM, GLM)
        print("Loading H2O DRF model...")
        drf_path = os.path.join(self.models_dir, 'DRF_1_AutoML_4_20251221_72446')
        self.models['DRF'] = h2o.load_model(drf_path)

        print("Loading H2O GBM model...")
        gbm_path = os.path.join(self.models_dir, 'GBM_1_AutoML_4_20251221_72446')
        self.models['GBM'] = h2o.load_model(gbm_path)

        print("Loading H2O GLM model...")
        glm_path = os.path.join(self.models_dir, 'GLM_1_AutoML_4_20251221_72446')
        self.models['GLM'] = h2o.load_model(glm_path)

        # Get expected column names from one of the models
        try:
            varimp = self.models['DRF'].varimp(use_pandas=True)
            if varimp is not None and len(varimp) > 0:
                self.h2o_column_names = varimp['variable'].tolist()
                print(f"Model expects {len(self.h2o_column_names)} features")
        except Exception as e:
            print(f"Could not get column names: {e}")

    def predict_bert(self, text):
        """Predict using BERT model"""
        tokenizer = self.models['BERT']['tokenizer']
        model = self.models['BERT']['model']

        inputs = tokenizer(text, return_tensors='pt', truncation=True,
                          max_length=512, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()

        return prediction, confidence

    def predict_roberta(self, text):
        """Predict using RoBERTa model"""
        tokenizer = self.models['RoBERTa']['tokenizer']
        model = self.models['RoBERTa']['model']

        inputs = tokenizer(text, return_tensors='pt', truncation=True,
                          max_length=512, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()

        return prediction, confidence

    def predict_h2o_model(self, text, model_name):
        """Predict using H2O models (DRF, GBM, GLM)"""
        # Transform text using TF-IDF
        tfidf_features = self.tfidf_vectorizer.transform([text]).toarray()
        n_features = tfidf_features.shape[1]

        # Use stored column names if available
        if self.h2o_column_names and len(self.h2o_column_names) >= n_features:
            # Use the expected column names from model training
            df = pd.DataFrame(tfidf_features, columns=self.h2o_column_names[:n_features])
        else:
            # Fallback: let pandas auto-generate column names
            df = pd.DataFrame(tfidf_features)

        # Convert to H2O frame
        h2o_frame = h2o.H2OFrame(df)

        # Get prediction
        model = self.models[model_name]
        predictions = model.predict(h2o_frame)

        # Extract prediction and confidence
        pred_df = predictions.as_data_frame()
        prediction = int(pred_df['predict'][0])
        confidence = float(pred_df['p1'][0]) if prediction == 1 else float(pred_df['p0'][0])

        return prediction, confidence

    def predict_all(self, text):
        """Make predictions using all 5 models"""
        print("="*70)
        print("HUMAN OR AI PREDICTION RESULTS")
        print("="*70)
        print(f"\nInput text preview: {text[:200]}...\n")
        print("-"*70)

        results = {}

        # BERT prediction
        print("\n1. BERT Model:")
        pred, conf = self.predict_bert(text)
        results['BERT'] = {'prediction': pred, 'confidence': conf}
        label = "HUMAN" if pred == 1 else "AI"
        print(f"   Prediction: {label} ({pred})")
        print(f"   Confidence: {conf*100:.2f}%")

        # RoBERTa prediction
        print("\n2. RoBERTa Model:")
        pred, conf = self.predict_roberta(text)
        results['RoBERTa'] = {'prediction': pred, 'confidence': conf}
        label = "HUMAN" if pred == 1 else "AI"
        print(f"   Prediction: {label} ({pred})")
        print(f"   Confidence: {conf*100:.2f}%")

        # DRF prediction
        print("\n3. DRF (Distributed Random Forest) Model:")
        pred, conf = self.predict_h2o_model(text, 'DRF')
        results['DRF'] = {'prediction': pred, 'confidence': conf}
        label = "HUMAN" if pred == 1 else "AI"
        print(f"   Prediction: {label} ({pred})")
        print(f"   Confidence: {conf*100:.2f}%")

        # GBM prediction
        print("\n4. GBM (Gradient Boosting Machine) Model:")
        pred, conf = self.predict_h2o_model(text, 'GBM')
        results['GBM'] = {'prediction': pred, 'confidence': conf}
        label = "HUMAN" if pred == 1 else "AI"
        print(f"   Prediction: {label} ({pred})")
        print(f"   Confidence: {conf*100:.2f}%")

        # GLM prediction
        print("\n5. GLM (Generalized Linear Model):")
        pred, conf = self.predict_h2o_model(text, 'GLM')
        results['GLM'] = {'prediction': pred, 'confidence': conf}
        label = "HUMAN" if pred == 1 else "AI"
        print(f"   Prediction: {label} ({pred})")
        print(f"   Confidence: {conf*100:.2f}%")

        # Ensemble prediction (majority voting)
        print("\n" + "="*70)
        predictions_list = [r['prediction'] for r in results.values()]
        ensemble_pred = 1 if sum(predictions_list) >= 3 else 0
        ensemble_label = "HUMAN" if ensemble_pred == 1 else "AI"
        avg_confidence = np.mean([r['confidence'] for r in results.values()])

        print("ENSEMBLE PREDICTION (Majority Vote):")
        print(f"   Final Prediction: {ensemble_label} ({ensemble_pred})")
        print(f"   Average Confidence: {avg_confidence*100:.2f}%")
        print(f"   Vote Count: {sum(predictions_list)} out of 5 models predicted HUMAN")
        print("="*70 + "\n")

        return results, ensemble_pred, avg_confidence

    def cleanup(self):
        """Cleanup H2O cluster"""
        h2o.cluster().shutdown()


def main():
    # Get input text from user
    print("Human or AI Text Classifier")
    print("="*70)
    print("\nEnter the English article/text to classify:")
    print("(Press Enter twice when done)\n")

    lines = []
    while True:
        line = input()
        if line == "" and len(lines) > 0:
            break
        lines.append(line)

    text = "\n".join(lines)

    if not text.strip():
        print("Error: No text provided!")
        return

    # Initialize predictor
    predictor = HumanOrAIPredictor()

    # Make predictions
    results, ensemble_pred, avg_conf = predictor.predict_all(text)

    # Cleanup
    predictor.cleanup()


if __name__ == "__main__":
    main()
