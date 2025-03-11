from transformers import AutoTokenizer, AutoModel
import faiss
import torch
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load raw data from the file
with open('show_data.txt', 'r') as f:
    show_data = f.read()

# Load a pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"  # Switching to BERT from DistilBERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to convert text to embeddings
def get_embeddings(text):
    # Tokenize the input text and prepare it for the model
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        # Get model outputs
        outputs = model(**inputs)
    # Use the mean of the last hidden state for embeddings
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Mean pooling

# Split raw data into chunks (e.g., by paragraphs)
show_data_chunks = show_data.split('\n\n')  # Split by double newlines

# Convert each chunk to embeddings
embeddings = [get_embeddings(chunk) for chunk in show_data_chunks]
embeddings = np.array(embeddings)

# Create a FAISS index
dim = embeddings.shape[1]  # Get the dimensionality of the embeddings
index = faiss.IndexFlatL2(dim)  # Create the FAISS index with L2 distance
index.add(embeddings)  # Add embeddings to the FAISS index

# Function to search the knowledge base
def search(query):
    # Get the embedding of the query
    query_embedding = get_embeddings(query).reshape(1, -1)
    # Perform a search in the FAISS index
    D, I = index.search(query_embedding, k=1)  # k=1 for the closest result
    # Retrieve the corresponding chunk of raw data
    answer = show_data_chunks[I[0][0]]
    return answer

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
