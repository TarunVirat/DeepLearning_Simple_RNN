# 🎬 IMDB Sentiment Analysis with Simple RNN

This project uses a **Simple Recurrent Neural Network (RNN)** to classify IMDB movie reviews as **positive** or **negative**.  
It includes Jupyter notebooks for training & evaluation, and a **Streamlit app** for interactive predictions.

---

## 📂 Files in this Repo

- `main.py` → Streamlit app for real-time sentiment prediction  
- `simplernn.ipynb` → Training notebook for RNN (IMDB dataset)  
- `embedding.ipynb` → Experiments with embeddings & preprocessing  
- `prediction.ipynb` → Notebook for testing predictions  
- `requirements.txt` → Dependencies file  
- `README.md` → Project documentation  

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/simple_rnn_imdb.git
cd simple_rnn_imdb
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv venv
.env\Scriptsctivate      # On Windows
source venv/bin/activate     # On macOS/Linux
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Streamlit App Locally
```bash
streamlit run main.py
```
Then open `http://localhost:8501/` in your browser.

---

## 🌐 Live App
Try the deployed version here:  
👉 [IMDB Sentiment App](https://deeplearningsimplernn-wzvr5mql9a5jez9d34mvkr.streamlit.app/)

---

## 🧩 How It Works
- Loads the IMDB dataset (built into Keras)  
- Trains a SimpleRNN on a subset of reviews  
- Tokenizes user input into word indices  
- Predicts sentiment (Positive/Negative) and confidence  

---

## 🧪 Example
**Input:**  
```
The movie was excellent, with strong acting and a great story.
```

**Output:**  
```
Sentiment: Positive 😀
Confidence: 0.91
```

---

## 📦 Requirements
See `requirements.txt`. Main dependencies:
- tensorflow==2.20.0  
- numpy  
- pandas  
- scikit-learn  
- matplotlib  
- streamlit  
- scikeras  

---

## ✨ Features
- End-to-end sentiment analysis pipeline  
- SimpleRNN model trained on IMDB dataset  
- Live demo via Streamlit Cloud  
- Easy reproducibility  

---

## 📌 Future Improvements
- Use **LSTM / GRU** for better accuracy  
- Add caching to avoid retraining on every refresh  
- Improve text preprocessing (cleaning, stopwords, etc.)  

---

## 👤 Author
**Tarun Peela**  
GitHub: [@TarunVirat](https://github.com/TarunVirat)
