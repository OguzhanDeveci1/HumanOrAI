from flask import Flask, render_template, request, jsonify
import os
import torch
import pickle
import h2o
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import numpy as np
import pandas as pd
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Disable Flask logging for cleaner output
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class HumanOrAIPredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.is_initialized = False
        self.h2o_column_names = None  # Store H2O expected column names
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initialize(self):
        if self.is_initialized:
            return

        print("Initializing models...")
        print(f"Using device: {self.device}")

        # Load BERT model
        print("Loading BERT...")
        bert_path = os.path.join(self.models_dir, 'bert_model')
        self.models['BERT'] = {
            'model': BertForSequenceClassification.from_pretrained(bert_path),
            'tokenizer': BertTokenizer.from_pretrained(bert_path)
        }
        self.models['BERT']['model'].to(self.device)
        self.models['BERT']['model'].eval()

        # Load RoBERTa model
        print("Loading RoBERTa...")
        roberta_path = os.path.join(self.models_dir, 'roberta_model')
        self.models['RoBERTa'] = {
            'model': RobertaForSequenceClassification.from_pretrained(roberta_path),
            'tokenizer': RobertaTokenizer.from_pretrained(roberta_path)
        }
        self.models['RoBERTa']['model'].to(self.device)
        self.models['RoBERTa']['model'].eval()

        # Load TF-IDF vectorizer
        print("Loading TF-IDF...")
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
        h2o.init(verbose=False)

        # Load H2O models
        print("Loading H2O models...")
        drf_path = os.path.join(self.models_dir, 'DRF_1_AutoML_4_20251221_72446')
        self.models['DRF'] = h2o.load_model(drf_path)

        gbm_path = os.path.join(self.models_dir, 'GBM_1_AutoML_4_20251221_72446')
        self.models['GBM'] = h2o.load_model(gbm_path)

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

        self.is_initialized = True
        print("All models loaded successfully!\n")

    def predict_bert(self, text):
        tokenizer = self.models['BERT']['tokenizer']
        model = self.models['BERT']['model']

        inputs = tokenizer(text, return_tensors='pt', truncation=True,
                          max_length=512, padding=True)

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()

        return prediction, confidence

    def predict_roberta(self, text):
        tokenizer = self.models['RoBERTa']['tokenizer']
        model = self.models['RoBERTa']['model']

        inputs = tokenizer(text, return_tensors='pt', truncation=True,
                          max_length=512, padding=True)

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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

        # Convert to H2O frame with optimizations
        h2o_frame = h2o.H2OFrame(df, column_types=['numeric'] * n_features)

        # Get prediction
        model = self.models[model_name]
        predictions = model.predict(h2o_frame)

        pred_df = predictions.as_data_frame()
        prediction = int(pred_df['predict'][0])
        confidence = float(pred_df['p1'][0]) if prediction == 1 else float(pred_df['p0'][0])

        return prediction, confidence

    def predict_all(self, text):
        results = {}

        # BERT
        print("Running BERT...")
        pred, conf = self.predict_bert(text)
        results['BERT'] = {
            'prediction': pred,
            'confidence': round(conf * 100, 2),
            'label': 'HUMAN' if pred == 1 else 'AI'
        }

        # RoBERTa
        print("Running RoBERTa...")
        pred, conf = self.predict_roberta(text)
        results['RoBERTa'] = {
            'prediction': pred,
            'confidence': round(conf * 100, 2),
            'label': 'HUMAN' if pred == 1 else 'AI'
        }

        # H2O models - optimize by creating H2O frame once
        print("Running H2O models...")
        tfidf_features = self.tfidf_vectorizer.transform([text]).toarray()
        n_features = tfidf_features.shape[1]

        # Create H2O frame once for all H2O models
        if self.h2o_column_names and len(self.h2o_column_names) >= n_features:
            df = pd.DataFrame(tfidf_features, columns=self.h2o_column_names[:n_features])
        else:
            df = pd.DataFrame(tfidf_features)

        h2o_frame = h2o.H2OFrame(df, column_types=['numeric'] * n_features)

        # DRF
        print("  - DRF...")
        predictions = self.models['DRF'].predict(h2o_frame)
        pred_df = predictions.as_data_frame()
        pred = int(pred_df['predict'][0])
        conf = float(pred_df['p1'][0]) if pred == 1 else float(pred_df['p0'][0])
        results['DRF'] = {
            'prediction': pred,
            'confidence': round(conf * 100, 2),
            'label': 'HUMAN' if pred == 1 else 'AI'
        }

        # GBM
        print("  - GBM...")
        predictions = self.models['GBM'].predict(h2o_frame)
        pred_df = predictions.as_data_frame()
        pred = int(pred_df['predict'][0])
        conf = float(pred_df['p1'][0]) if pred == 1 else float(pred_df['p0'][0])
        results['GBM'] = {
            'prediction': pred,
            'confidence': round(conf * 100, 2),
            'label': 'HUMAN' if pred == 1 else 'AI'
        }

        # GLM
        print("  - GLM...")
        predictions = self.models['GLM'].predict(h2o_frame)
        pred_df = predictions.as_data_frame()
        pred = int(pred_df['predict'][0])
        conf = float(pred_df['p1'][0]) if pred == 1 else float(pred_df['p0'][0])
        results['GLM'] = {
            'prediction': pred,
            'confidence': round(conf * 100, 2),
            'label': 'HUMAN' if pred == 1 else 'AI'
        }

        # Ensemble
        predictions_list = [r['prediction'] for r in results.values()]
        ensemble_pred = 1 if sum(predictions_list) >= 3 else 0
        ensemble_label = 'HUMAN' if ensemble_pred == 1 else 'AI'
        avg_confidence = round(np.mean([r['confidence'] for r in results.values()]), 2)
        vote_count = sum(predictions_list)

        ensemble = {
            'prediction': ensemble_pred,
            'label': ensemble_label,
            'confidence': avg_confidence,
            'vote_count': vote_count,
            'total_models': 5
        }

        return {
            'individual_results': results,
            'ensemble': ensemble
        }

# Initialize predictor
predictor = HumanOrAIPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Initialize models on first request
        if not predictor.is_initialized:
            predictor.initialize()

        data = request.get_json()
        text = data.get('text', '')

        if not text.strip():
            return jsonify({'error': 'Please provide text to analyze'}), 400

        # Make predictions
        results = predictor.predict_all(text)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Human or AI Classifier Web Application...")
    print("Server will be available at: http://localhost:5000")
    print("-" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
