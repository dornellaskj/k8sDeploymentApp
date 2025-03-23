from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load raw data from the file
with open('show_data.txt', 'r') as f:
    show_data = f.read()

modelName = 'facebook/blenderbot-400M-distill'
model = AutoModelForSeq2SeqLM.from_pretrained(modelName)
tokenizer = AutoTokenizer.from_pretrained(modelName)
inputs = []
show_data_chunks = []
# Function to convert text to embeddings
def get_embeddings(text):
    return tokenizer.encode_plus(text, return_tensors='pt')

# Load and preprocess data
with open("show_data.txt", "r") as f:
    show_data_chunks = f.read().split("\n\n")  # Splitting into chunks

# Convert each chunk to embeddings
for chunk in show_data_chunks:
    inputs.append(get_embeddings(chunk))

# Add this part after creating the embeddings
print(f"Total number of inputs created: {len(inputs)}")


# Function to search the knowledge base
def search(query):    
    query_embedding = get_embeddings(query)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens = True).strip()

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