from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model and tokenizer
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data["message"]
    
    # Tokenize the input and generate a response
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    response_ids = model.generate(input_ids, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
