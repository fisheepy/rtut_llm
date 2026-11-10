import faiss
import numpy as np
import json
import openai
import os
import glob
from flask import Flask, request, jsonify
from docx import Document
from tqdm import tqdm
from flask_cors import CORS  # ✅ Enable CORS for frontend access
import openai._exceptions 
import boto3

# ✅ Load S3 Credentials from Heroku Environment Variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = 'us-east-2'

# ✅ Initialize S3 Client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# ✅ Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")
ROY_BOT_MODEL = os.getenv("ROY_BOT_MODEL", "gpt-6-luna")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets the `/backend/server/llm/` path
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(BASE_DIR, "metadata_store.json")

# ✅ Initialize FAISS Index (384 dimensions for MiniLM embeddings)
D = 1536  # ✅ Set FAISS index dimension to match OpenAI embedding size
index = faiss.IndexFlatL2(D)  # L2 distance metric

metadata_store = []  # Stores metadata like policy names

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "FAISS Server is running."

# ✅ Function to Upload Files to S3
def upload_to_s3(file_path, s3_key):
    try:
        s3.upload_file(file_path, AWS_BUCKET_NAME, s3_key)
        print(f"✅ Uploaded {file_path} to S3 bucket {AWS_BUCKET_NAME}")
    except Exception as e:
        print(f"❌ Failed to upload {file_path} to S3: {e}")

# ✅ Function to Download Files from S3
def download_from_s3(s3_key, local_path):
    try:
        s3.download_file(AWS_BUCKET_NAME, s3_key, local_path)
        print(f"✅ Downloaded {s3_key} from S3 to {local_path}")
    except Exception as e:
        print(f"⚠️ Failed to download {s3_key} from S3: {e}")

# ✅ Load FAISS and Metadata on Startup
def initialize_faiss():
    global index, metadata_store
    print("🔄 Initializing FAISS...")

    # Download Index and Metadata from S3
    download_from_s3("faiss_index.bin", INDEX_PATH)
    download_from_s3("metadata_store.json", METADATA_PATH)

    # Load FAISS Index
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        print("✅ FAISS Index Loaded from S3")
    else:
        print("⚠️ No FAISS index found. Run `/train` API to train with documents.")

    # Load Metadata
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata_store = json.load(f)
            print(f"✅ Loaded {len(metadata_store)} policies from metadata store.")
        except json.JSONDecodeError:
            print("❌ Failed to load metadata_store.json.")
            metadata_store = []
    else:
        metadata_store = []

# ✅ Call `initialize_faiss()` Immediately (Works for Both Gunicorn & Local)
initialize_faiss()

# ✅ Function to Get Text Embeddings
def get_embedding(text):
    """Get OpenAI embedding for a given text with error handling."""
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except openai._exceptions.AuthenticationError:
        print("❌ OpenAI API Key is invalid!")
        return None
    except openai._exceptions.RateLimitError:
        print("❌ OpenAI Rate Limit exceeded. Try again later.")
        return None
    except openai._exceptions.OpenAIError as e:
        print(f"❌ OpenAI Error: {e}")
        return None

# ✅ Function to Extract Text from `.docx`
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return text

# ✅ Function to Train FAISS from `.docx` Policies
def train_faiss_from_documents(directory):
    global index, metadata_store
    doc_files = glob.glob(os.path.join(directory, "*.docx"))

    print(f"📄 Found {len(doc_files)} policy documents. Processing...")

    all_embeddings = []
    all_metadata = []

    for file_path in tqdm(doc_files, desc="Processing Documents"):
        policy_text = extract_text_from_docx(file_path)
        policy_name = os.path.basename(file_path)

        # ✅ Split into chunks
        chunks = [policy_text[i:i+500] for i in range(0, len(policy_text), 500)]

        for chunk in chunks:
            embedding = get_embedding(chunk)
            all_embeddings.append(embedding)
            all_metadata.append({"name": policy_name, "text": chunk})  # ✅ Store metadata

    if all_embeddings:
        index.add(np.array(all_embeddings, dtype=np.float32))  # ✅ Add all at once
        metadata_store.extend(all_metadata)  # ✅ Ensure metadata is stored!

    # ✅ Save FAISS index
    faiss.write_index(index, INDEX_PATH)

    # ✅ Save metadata to JSON
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, ensure_ascii=False, indent=4)

    print(f"✅ FAISS Training Complete! Indexed {len(all_embeddings)} vectors and {len(metadata_store)} policies.")

# ✅ API Endpoint to Train FAISS Automatically
@app.route("/train", methods=["POST"])
def train():
    directory = request.json.get("directory")
    if not directory or not os.path.exists(directory):
        return jsonify({"error": "Invalid directory path"}), 400

    train_faiss_from_documents(directory)
    return jsonify({"message": "Training complete!"})

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "num_vectors": index.ntotal,
        "num_policies": len(metadata_store),
        "index_size": index.d,
        "sample_metadata": metadata_store[:5]  # ✅ Print first 5 stored policies
    })

# ✅ API Endpoint to Search FAISS for Relevant Policies
@app.route("/search", methods=["POST"])
def search():
    global index, metadata_store

    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "Missing search query"}), 400

    # Convert user query to an embedding
    query_embedding = np.array([get_embedding(query)], dtype=np.float32)

    # Retrieve the top 3 most relevant documents
    distances, indices = index.search(query_embedding, k=3)

    results = []
    for idx in indices[0]:
        if idx < len(metadata_store):
            results.append(metadata_store[idx])

    return jsonify({"results": results})

@app.route("/chat", methods=["POST"])
def chat():
    global index, metadata_store

    data = request.json
    question = data.get("question")
    style = data.get("style", "friendly").lower()  # Default to friendly

    if not question:
        return jsonify({"error": "Missing question"}), 400

    # Get embedding for user query
    query_embedding = get_embedding(question)
    if query_embedding is None:
        return jsonify({"error": "Failed to generate embedding. Check OpenAI API key."}), 500

    # Search FAISS index
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k=3)
    results = [metadata_store[idx] for idx in indices[0] if idx < len(metadata_store)]

    # Construct context from retrieved docs
    context = "\n\n".join([f"**{r['name']}**: {r['text']}" for r in results])

    # Define system prompts by style
    SYSTEM_PROMPTS = {
        "friendly": "You are Roy Bot, a friendly and helpful chatbot that assists employees in understanding company policies. Always respond in a warm, approachable, and cheerful tone.",
        "professional": "You are Roy Bot, a professional HR policy advisor. Provide precise, clear, and formal answers with references to relevant company policies. Avoid casual language.",
        "strict": "You are Roy Bot, a no-nonsense safety and compliance advisor. Answer concisely, directly, and in a serious tone. Emphasize the importance of following rules.",
        "default": "You are Roy Bot, a helpful chatbot that assists employees in understanding company policies."
    }

    # Pick the appropriate system prompt
    system_prompt = SYSTEM_PROMPTS.get(style, SYSTEM_PROMPTS["default"])

    try:
        response = openai.responses.create(
            model=ROY_BOT_MODEL,
            instructions=system_prompt,
            input=f"Hi Roy Bot, I have a question: {question}\n\nHere are some related policies:\n{context}",
        )

        answer = (response.output_text or "").strip()
        if not answer:
            raise RuntimeError("OpenAI returned an empty answer")

        return jsonify({
            "answer": f"🤖 Roy Bot: {answer}",
            "referenced_documents": list(set([r["name"] for r in results]))
        })

    except openai._exceptions.OpenAIError as e:
        print(f"❌ OpenAI Error ({type(e).__name__}) using {ROY_BOT_MODEL}: {e}")
        return jsonify({"error": "Roy Bot is temporarily unavailable. Please try again shortly."}), 500
    except Exception as e:
        print(f"❌ Roy Bot Error ({type(e).__name__}) using {ROY_BOT_MODEL}: {e}")
        return jsonify({"error": "Oops! Something went wrong. Try again later. 😞"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))  # ✅ Use Heroku's assigned port
    print(f"🚀 FAISS Server Listening on: http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
