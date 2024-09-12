import pandas as pd
import random

# Sample medical conditions and their associated symptoms
conditions = {
    "Hypertension": ["high blood pressure", "headache", "shortness of breath", "nosebleeds"],
    "Diabetes": ["increased thirst", "frequent urination", "blurred vision", "fatigue"],
    "Asthma": ["wheezing", "coughing", "chest tightness", "shortness of breath"],
    "Migraine": ["severe headache", "nausea", "sensitivity to light", "vision changes"]
}

# Generate sample data
data = []
for _ in range(1000):  # Generate 1000 samples
    condition = random.choice(list(conditions.keys()))
    symptoms = conditions[condition]
    text = f"Patient presents with {', '.join(random.sample(symptoms, k=random.randint(1, len(symptoms))))}"
    data.append({"text": text, "label": condition})

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv("medical_texts.csv", index=False)
print("Sample dataset created: medical_texts.csv");