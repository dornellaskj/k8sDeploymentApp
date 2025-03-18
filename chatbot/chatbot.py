from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load raw data from the file
with open('show_data.txt', 'r') as f:
    show_data = f.read()

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Optimized for embeddings

# Function to convert text to embeddings
def get_embeddings(text):
    return model.encode(text, convert_to_numpy=True)

# Load and preprocess data
with open("show_data.txt", "r") as f:
    show_data_chunks = f.read().split("\n\n")  # Splitting into chunks

# Convert each chunk to embeddings
embeddings = np.array([get_embeddings(chunk) for chunk in show_data_chunks])

# Create FAISS index
dim = embeddings.shape[1]  # Get embedding dimension
index = faiss.IndexFlatL2(dim)
index.add(embeddings)  # Add embeddings to FAISS


# Function to search the knowledge base
def search(query, threshold=10.0):  # Adjust the threshold as needed
    query_embedding = get_embeddings(query).reshape(1, -1)
    D, I = index.search(query_embedding, k=1)  # k=1 for the closest result

    # Check if the closest match is above the threshold (too far away)
    if D[0][0] > threshold:
        return "No relevant results found."

    return show_data_chunks[I[0][0]]

# API route for the chatbot
@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Get user input from the request
        data = request.json
        user_input = data.get("message", "").strip()

        # Check if message is empty
        if not user_input:
            return jsonify({"response": "Please provide a message."})

        # Search for the most relevant response using the search function
        response = search(user_input)

        # Return the response in JSON format
        return jsonify({"response": response})

    except Exception as e:
        # Handle any exceptions and return an error message
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)