# 🔮 Bank Customer Churn – MLOps Lab

## 📦 Dataset
The dataset used is [Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction/data).  
It contains customer records from a U.S. bank and the goal is to predict whether a customer will leave the bank or not (`Exited` column).

Dataset columns include:
- Customer demographics (e.g., `Age`, `Gender`, `Geography`)
- Account information (e.g., `Balance`, `Tenure`, `NumOfProducts`)
- Target variable: `Exited`

CSV file is placed under the `dataset/` folder.

---

## ⚙️ How to Reproduce

1. 📦 **Create a virtual environment**

```bash
   conda create -n churn_prediction python=3.12 -y
   conda activate churn_prediction
   pip install -r requirements.txt
```

---

2. 📥 **Download the dataset**

Download the CSV from Kaggle

Place the file in: dataset/bank-churn.csv

---

3. 🚀 **Run experiments**

```bash
# Random Forest
python src/train.py --model rf --n_estimators 300 --max_depth 8

# Extra Trees
python src/train.py --model et --n_estimators 400 --max_depth 10

# Logistic Regression
python src/train.py --model lr --c 0.5

# Gradient Boosting
python src/train.py --model gb --n_estimators 200 --learning_rate 0.05

```

---

4. 🔍 **Start MLflow UI**

```bash
    mlflow ui
```
Open http://127.0.0.1:5000 in your browser.

---

## 🗂️ Registry

models were registered in the MLflow Model Registry:

✅ Staging: (best F1 & AUC)

✅ Production: (high stability & speed)

![Registry Screenshot](lab1.png)

---

```bash
src/
│
├── churn/                 
│   ├── data.py            # load & split
│   ├── features.py        # preprocessing pipeline
│   ├── models.py          # model factory
│   ├── train.py           # CLIs -> calls churn.train()
│   └── __init__.py         
│   
│── train.py

```
---
