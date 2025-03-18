# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install torch transformers flask
RUN pip install sentence-transformers

# Install FAISS GPU version
RUN pip install faiss-gpu

# Add your chatbot script (e.g., chatbot.py) to the container
COPY chatbot.py .
COPY show_data.txt .
# Expose the port on which the app will run
EXPOSE 5000

# Run the application
CMD ["python", "chatbot.py"]
