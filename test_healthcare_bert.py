import torch
from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained('./healthcare_bert')
tokenizer = BertTokenizer.from_pretrained('./healthcare_bert')

# Load the label encoder
df = pd.read_csv('medical_texts.csv')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['label'])

# Function to make predictions
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return le.inverse_transform([predicted_class])[0]

# Test the model with some examples
test_texts = [
    "Patient complains of persistent cough and difficulty breathing.",
    "Individual reports frequent urination and increased thirst.",
    "Person experiences severe headache with sensitivity to light.",
    "Patient presents with elevated blood pressure readings.",
]

for text in test_texts:
    prediction = predict(text)
    print(f"Text: {text}")
    print(f"Predicted diagnosis: {prediction}\n")

# Interactive testing
while True:
    user_input = input("Enter a medical description (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    prediction = predict(user_input)
    print(f"Predicted diagnosis: {prediction}\n")