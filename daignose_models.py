# diagnose_models.py
from pathlib import Path
import joblib

def check_directory(directory, name):
    print(f'\n=== {name} directory: {directory} ===')
    files = list(directory.glob('*.pkl'))
    print(f'Found {len(files)} .pkl files:')
    
    for file in files:
        print(f'  {file.name}')
        try:
            obj = joblib.load(file)
            print(f'    Type: {type(obj).__name__}')
            if hasattr(obj, 'classes_'):
                print(f'    Classes: {obj.classes_}')
            if hasattr(obj, 'n_features_in_'):
                print(f'    Features expected: {obj.n_features_in_}')
            if isinstance(obj, list):
                print(f'    List length: {len(obj)}')
                if len(obj) > 0:
                    print(f'    First element: {type(obj[0]).__name__}')
        except Exception as e:
            print(f'    Error loading: {e}')
    return files

# Check both directories
results_files = check_directory(Path('results'), 'Results')
models_files = check_directory(Path('models'), 'Models')

print('\n=== RECOMMENDATION ===')
if results_files and not models_files:
    print('Copy files from results/ to models/ directory')
elif not results_files and not models_files:
    print('No model files found. Run: python main.py')
else:
    print('Files exist in both directories. Check config.py matches actual filenames')