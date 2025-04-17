🧑‍⚖️ **Indian Legal Query Classifier using Fine-Tuned BERT**
This project demonstrates how to fine-tune a pre-trained BERT model to classify Indian legal queries using the Indian Law dataset and deploy it using Streamlit for real-time predictions and visualization.

📌 **Project Features**
-> Fine-tunes BERT on the Indian Law Q&A dataset
-> Classifies legal questions into appropriate response categories
-> Visualizes model attention with heatmaps
-> Interactive UI built with Streamlit

📂 **Folder Structure**
legal-query-classifier/
├── app.py                   # Streamlit frontend
├── requirements.txt         # Required Python packages
├── indian_law_model/        # Saved model & tokenizer
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── label_classes.npy

✅**Step-by-Step Setup**
🚀 Part 1: Train the Model in Google Colab
-> Open this notebook(ModelTraining.ipynb) in Google Colab and run the training code.
-> Save the model artifacts:
    #Save model and tokenizer
    model.save_pretrained("indian_law_model")
    tokenizer.save_pretrained("indian_law_model")
    
    #Save label classes for decoding predictions
    import numpy as np
    np.save("indian_law_model/label_classes.npy", le.classes_)

-> Download the model folder:
    from google.colab import files
    import shutil
    shutil.make_archive("indian_law_model", 'zip', "indian_law_model")
    files.download("indian_law_model.zip")

💻 Part 2: Set Up Streamlit App Locally
-> Extract the indian_law_model.zip into your project folder.
-> Create a new Python file named app.py and paste the code.
-> Create a requirements.txt file:
    streamlit
    torch
    transformers
    scikit-learn
    seaborn
    matplotlib
    numpy
-> Install dependencies:
    pip install -r requirements.txt
-> Run the app:
    streamlit run app.py
