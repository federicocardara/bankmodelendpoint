# 🧠 Bank Marketing Prediction

Version: **1.0.0**

This repository contains two Dockerized Python applications:

| Folder       | Purpose                                                    |
|--------------|------------------------------------------------------------|
| `train/`     | Trains and evaluates multiple models using `bank-full.csv` |
| `endpoint/`  | Serves the best model via a REST API using Flask           |

The goal is to predict whether a customer will subscribe to a product **before making a phone call**.

## ⚙️ Application 1 – Model Training (`train/`)

### 🔍 What it does:

- Loads the dataset `bank-full.csv`
- Preprocesses the data:
  - **MinMaxScaler** for numerical features
  - **OneHotEncoder** for categorical features
- Trains 4 models:
  - Random Forest
  - Logistic Regression
  - Support Vector Machine (SVM)
  - XGBoost
- Selects and saves the **best model** based on accuracy

## Structure

train/ 
├── Dockerfile 
├── docker-compose.yml │ 
├── train_model.py │ 
├── requirements.txt │ 
└── data/ 
    └── bank-full.csv │  
└── results/ 

### 📦 Output:

- Saved in `train/results/best_model.pkl`

### ▶️ Build & Run:

```bash
cd train
docker compose up --build

# 🌐 Bank Marketing API – Model Inference Service

This project exposes the **best trained model** (from the `train/` project) as a **RESTful API** using Flask.

## 🧩 Structure

endpoint/
├── Dockerfile
├── docker-compose.yml
├── model_endpoint.py
├── requirements.txt


## 🚀 Features

- Loads `best_model.pkl` (a scikit-learn pipeline including preprocessing)
- Exposes a **POST** endpoint at `/predict`
- Returns the prediction (`0` or `1`) as JSON

## ⚙️ Setup Instructions

### ✅ Step 1: Copy the trained model

Ensure you've already trained the model using the pipeline in the `train/` folder.

Then copy it:

```bash
cp ../train/results/best_model.pkl model/
