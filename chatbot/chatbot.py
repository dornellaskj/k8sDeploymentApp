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

# Split raw data into chunks (e.g., by paragraphs)
show_data_chunks = show_data.split('\n\n')  # Split by double newlines

# Convert each chunk to embeddings
embeddings = np.array([get_embeddings(chunk) for chunk in show_data_chunks])

# Ensure embeddings are valid
if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
    raise ValueError("Embeddings could not be generated.")

# Create a FAISS index
dim = embeddings.shape[1]  # Get the dimensionality of the embeddings
index = faiss.IndexFlatL2(dim)  # Create the FAISS index with L2 distance
index.add(embeddings)  # Add embeddings to the FAISS index

# Function to search the knowledge base
def search(query):
    query_embedding = get_embeddings(query).reshape(1, -1)
    D, I = index.search(query_embedding, k=1)  # k=1 for the closest result
    
    if I[0][0] == -1:  # If no match is found
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