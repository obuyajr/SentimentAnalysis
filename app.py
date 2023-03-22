import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from flask import Flask, jsonify, request

# Load the pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Set the device to use for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the CSV file
data = pd.read_csv("data.csv")

# Initialize the Flask application
app = Flask(__name__)

# Define a function to perform sentiment analysis on a given text
def predict_sentiment(text):
    # Tokenize the text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Move the inputs to the appropriate device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Perform inference
    outputs = model(input_ids, attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).tolist()

    # Return the predicted sentiment label
    return predictions[0]

# Define a route to perform sentiment analysis on the CSV data
@app.route("/sentiment", methods=["POST"])
def predict_sentiment_csv():
    # Load the CSV data
    data = pd.read_csv(request.files["file"])

    # Perform sentiment analysis on each row
    data["sentiment"] = data["text"].apply(predict_sentiment)

    # Return the CSV data with the predicted sentiment labels
    return data.to_csv(index=False)

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
