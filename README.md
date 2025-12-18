# Tailyo-Service
# 🚀 ML-Powered Service Recommendation System

This project implements an ML-powered service recommendation system that helps users identify the most relevant services based on their preferences such as business type, budget, language, location, and optional keywords. The system uses similarity-based machine learning techniques and provides explainable recommendations through an interactive Streamlit web interface.

---

## 📌 Project Overview

With the increasing number of digital services available, selecting the right service manually becomes difficult and time-consuming. This project addresses the problem by processing user preferences, encoding service features, ranking services based on similarity, and presenting clear, interpretable recommendations.

---

## 🧠 Key Features

- Content-based recommendation using cosine similarity
- One-Hot Encoding for categorical features
- TF-IDF Vectorization for text-based features
- Match score and match quality generation
- Human-readable recommendation explanations
- Streamlit-based interactive user interface

---

## 🏗️ System Architecture

The system follows a five-stage modular pipeline:

1. User Input and Data Preprocessing
2. Feature Encoding
3. Filtering and Similarity Computation
4. Ranking and Explanation Generation
5. Result Visualization

---
## ⚙️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- SciPy
- Streamlit
- Joblib

---

## ▶️ How to Run the Application

1. Clone this repository:
   ```bash
   git clone https://github.com/harsharchandran4105/Tailyo_service.git
   cd Tailyo Service

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
