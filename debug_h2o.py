"""
H2O model sütun isimlerini kontrol etme scripti
"""
import h2o
import os

# Initialize H2O
print("Initializing H2O...")
h2o.init(verbose=False)

models_dir = 'models'

# Load one of the models
model_path = os.path.join(models_dir, 'DRF_1_AutoML_4_20251221_72446')
print(f"\nLoading model from: {model_path}")

model = h2o.load_model(model_path)

print("\n" + "="*60)
print("MODEL INFORMATION")
print("="*60)

# Get model details
print(f"\nModel type: {type(model)}")
print(f"Model algorithm: {model.algo}")

# Get training frame column names
print("\n" + "="*60)
print("EXPECTED COLUMN NAMES")
print("="*60)

# Try to get the column names the model was trained with
try:
    params = model.params
    print(f"\nModel parameters keys: {list(params.keys())[:10]}...")  # First 10 keys
except Exception as e:
    print(f"Could not get params: {e}")

# Get model training metrics
try:
    print("\n" + "="*60)
    print("MODEL METRICS")
    print("="*60)
    print(f"\nTraining metrics: {model.model_performance()}")
except Exception as e:
    print(f"Could not get metrics: {e}")

# Try to extract feature names
try:
    print("\n" + "="*60)
    print("FEATURE NAMES")
    print("="*60)

    # Get variable importances which shows feature names
    varimp = model.varimp(use_pandas=True)
    if varimp is not None and len(varimp) > 0:
        print(f"\nNumber of features: {len(varimp)}")
        print(f"\nFirst 10 features:")
        print(varimp.head(10))

        # Extract all feature names
        feature_names = varimp['variable'].tolist()
        print(f"\nAll feature names (first 20):")
        for i, name in enumerate(feature_names[:20]):
            print(f"  {i}: {name}")
    else:
        print("No variable importance available")

except Exception as e:
    print(f"Error getting variable importance: {e}")

# Try to get model details
try:
    print("\n" + "="*60)
    print("MODEL DETAILS")
    print("="*60)

    details = str(model)
    lines = details.split('\n')
    print('\n'.join(lines[:30]))  # Print first 30 lines

except Exception as e:
    print(f"Error getting model details: {e}")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
print("\nUse the feature names shown above to create your DataFrame.")
print("Example: If features are 'feature_0', 'feature_1', etc.")
print("Then use: feature_names = [f'feature_{i}' for i in range(n_features)]")

h2o.cluster().shutdown()
