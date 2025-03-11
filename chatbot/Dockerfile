# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install torch transformers flask

# Install FAISS GPU version
RUN pip install faiss-gpu

# Add your chatbot script (e.g., chatbot.py) to the container
COPY chatbot.py .

# Expose the port on which the app will run
EXPOSE 5000

# Run the application
CMD ["python", "chatbot.py"]
