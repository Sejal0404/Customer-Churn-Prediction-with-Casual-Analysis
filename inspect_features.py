import joblib
import os

# Load the feature names
try:
    feature_names = joblib.load('models/feature_names_linear.pkl')
    print("Feature names loaded successfully:")
    print(f"Number of features: {len(feature_names)}")
    print("Features:")
    for i, feature in enumerate(feature_names):
        print(f"  {i+1}. {feature}")
except Exception as e:
    print(f"Error loading feature names: {e}")

# Also check if there are other feature files
print("\nChecking for other feature files:")
files_to_check = [
    'models/feature_names_tree.pkl',
    'models/feature_names.pkl'
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        try:
            features = joblib.load(file_path)
            print(f"\n{file_path}: {len(features)} features")
            print(f"First 5: {features[:5]}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    else:
        print(f"{file_path}: File not found")
