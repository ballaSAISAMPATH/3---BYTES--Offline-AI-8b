# download_models.py
import os
from sentence_transformers import SentenceTransformer

def download_all_models():
    print("📥 Downloading embedding model...")

    target_path = "./models/embeddings/sentence-transformers/all-MiniLM-L6-v2"

    os.makedirs(target_path, exist_ok=True)

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./models/embeddings"
    )

    model.save(target_path)

    print("✅ Model saved to:", target_path)
    print("🎉 All models downloaded successfully!")
    return True

if __name__ == "__main__":
    download_all_models()
