# download_models.py
import os
from sentence_transformers import SentenceTransformer

def download_all_models():
    print("📥 Downloading embedding model for offline use...")
    print("🌐 Make sure you have internet connection for this step.")

    # Correct path structure for offline loading
    target_path = "./models/embeddings/sentence-transformers/all-MiniLM-L6-v2"
    os.makedirs(target_path, exist_ok=True)

    try:
        print("📥 Downloading all-MiniLM-L6-v2...")
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./models/embeddings"
        )

        # Save the fully downloaded model
        model.save(target_path)

        print("✅ Embedding model downloaded and saved to:")
        print("   ", target_path)

        print("🎉 All models downloaded successfully!")
        print("🔒 You can now run completely offline.")
        return True

    except Exception as e:
        print(f"❌ Failed to download embedding model: {e}")
        return False


if __name__ == "__main__":
    success = download_all_models()
    if success:
        print("\n✅ Setup complete! You can now run offline.")
    else:
        print("\n❌ Setup failed. Check your internet connection.")
