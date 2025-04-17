import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("indian_law_model")
    model = BertForSequenceClassification.from_pretrained("indian_law_model")
    label_classes = np.load("indian_law_model/label_classes.npy", allow_pickle=True)
    return tokenizer, model, label_classes

tokenizer, model, label_classes = load_model_and_tokenizer()

st.title("🧑‍⚖️ Indian Legal Question Classifier")
st.write("Enter a legal question and this app will classify it using a fine-tuned BERT model.")

user_input = st.text_area("Enter your legal question here:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=1).item()
            predicted_label = label_classes[pred_id]
        
        st.success(f"Predicted Legal Response: {predicted_label}")

        # Show attention visualization
        st.subheader("🧠 Attention Heatmap (Head 0)")
        attention = outputs.attentions[-1][0][0].detach().numpy()  # (head, tokens, tokens)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, cmap="viridis", ax=ax)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        st.pyplot(fig)
