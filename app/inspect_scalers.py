import os, pickle
# compute path relative to this script (repo root is parent of app/)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
p = os.path.join(REPO_ROOT, 'outputs', 'models', 'bilstm_scalers.pkl')
print('Checking path:', os.path.abspath(p), os.path.exists(p))
if os.path.exists(p):
    with open(p,'rb') as f:
        obj = pickle.load(f)
    try:
        keys = list(obj.keys()) if hasattr(obj,'keys') else None
        print('Type of obj:', type(obj))
        print('Number of keys:', len(keys) if keys is not None else 'N/A')
        print('Sample keys (first 40):', keys[:40])
        print('Has 1510830 (int)?', 1510830 in obj)
        print("Has '1510830' (str)?", '1510830' in obj)
    except Exception as e:
        print('Error inspecting pickle:', e)
else:
    print('Scalers file not found')
