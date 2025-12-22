"""
Model dosyalarını kontrol eden yardımcı script
"""
import os
import pickle
import joblib
import sys

def check_file(filepath, file_type):
    print(f"\n{'='*60}")
    print(f"Checking: {file_type}")
    print(f"Path: {filepath}")
    print(f"{'='*60}")

    if not os.path.exists(filepath):
        print("❌ File does not exist!")
        return False

    print(f"✓ File exists")
    print(f"  Size: {os.path.getsize(filepath):,} bytes")

    return True

def check_pickle_file(filepath):
    print("\nTrying to load with pickle...")
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        print("✓ Successfully loaded with pickle")
        print(f"  Type: {type(obj)}")
        if hasattr(obj, 'get_params'):
            print(f"  Params: {obj.get_params()}")
        return True
    except Exception as e:
        print(f"❌ Failed with pickle: {e}")
        return False

def check_joblib_file(filepath):
    print("\nTrying to load with joblib...")
    try:
        obj = joblib.load(filepath)
        print("✓ Successfully loaded with joblib")
        print(f"  Type: {type(obj)}")
        if hasattr(obj, 'get_params'):
            print(f"  Params: {obj.get_params()}")
        return True
    except Exception as e:
        print(f"❌ Failed with joblib: {e}")
        return False

def check_h2o_model(model_path):
    print(f"\n{'='*60}")
    print(f"Checking H2O Model: {os.path.basename(model_path)}")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print("❌ Directory does not exist!")
        return False

    print("✓ Directory exists")

    try:
        import h2o
        print("\nInitializing H2O...")
        h2o.init(verbose=False)

        print(f"Loading model from {model_path}...")
        model = h2o.load_model(model_path)
        print("✓ Successfully loaded H2O model")
        print(f"  Model type: {type(model)}")

        return True
    except Exception as e:
        print(f"❌ Failed to load H2O model: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("MODEL FILES DIAGNOSTIC TOOL")
    print("="*60)

    models_dir = 'models'

    # Check TF-IDF vectorizer
    tfidf_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    if check_file(tfidf_path, "TF-IDF Vectorizer"):
        pickle_ok = check_pickle_file(tfidf_path)
        if not pickle_ok:
            joblib_ok = check_joblib_file(tfidf_path)
            if joblib_ok:
                print("\n💡 Solution: Use joblib.load() instead of pickle.load()")

    # Check BERT model
    bert_path = os.path.join(models_dir, 'bert_model')
    if check_file(bert_path, "BERT Model Directory"):
        config_path = os.path.join(bert_path, 'config.json')
        model_path = os.path.join(bert_path, 'model.safetensors')
        check_file(config_path, "BERT config.json")
        check_file(model_path, "BERT model.safetensors")

    # Check RoBERTa model
    roberta_path = os.path.join(models_dir, 'roberta_model')
    if check_file(roberta_path, "RoBERTa Model Directory"):
        config_path = os.path.join(roberta_path, 'config.json')
        model_path = os.path.join(roberta_path, 'model.safetensors')
        check_file(config_path, "RoBERTa config.json")
        check_file(model_path, "RoBERTa model.safetensors")

    # Check H2O models
    h2o_models = [
        'DRF_1_AutoML_4_20251221_72446',
        'GBM_1_AutoML_4_20251221_72446',
        'GLM_1_AutoML_4_20251221_72446'
    ]

    for model_name in h2o_models:
        model_path = os.path.join(models_dir, model_name)
        check_h2o_model(model_path)

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
