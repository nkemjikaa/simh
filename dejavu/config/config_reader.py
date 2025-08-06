import yaml
import os

def read(filename):
    full_path = os.path.abspath(filename)
    print(f"🔍 Reading config from: {full_path}")
    with open(full_path, 'r') as f:
        data = yaml.safe_load(f)
        print("✅ Loaded config:", data)
        return data