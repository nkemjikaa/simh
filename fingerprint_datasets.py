from dejavu import Dejavu
from dejavu.config.config_reader import read
import os

# Load config
def main():
    config = read("dejavu.cnf.MySQL.yaml")
    djv = Dejavu(config)

    dataset_path = "simh_datasets"
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(".wav"):
                filepath = os.path.join(root, file)
                print(f"🎧 Fingerprinting: {filepath}")
                try:
                    djv.fingerprint_file(filepath)
                    print(f"✅ Done: {filepath}")
                except Exception as e:
                    print(f"❌ Error with {filepath}: {e}")

    print("🎉 Finished fingerprinting all files.")

if __name__ == "__main__":
    main()
