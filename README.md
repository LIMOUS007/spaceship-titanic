# 🚀 Spaceship Titanic – ML Classification Project  

**Dataset:** Kaggle – Spaceship Titanic  
**Goal:** Predict whether a passenger was transported to an alternate dimension  

---

## 📌 Problem Overview  

The Spaceship Titanic dataset contains information about passengers aboard a damaged interstellar ship.  
The task is to predict whether a passenger was **Transported (True/False)** based on their travel details, spending behavior, and cabin information.

This is a **binary classification problem**, evaluated using **accuracy**.

---

## 🧠 Approach Summary  

We followed a structured machine learning workflow:

### 1️⃣ Data Cleaning & Preprocessing  
- Combined train and test data  
- Extracted useful features from `Cabin`  
  - Deck  
  - Cabin Number  
  - Side (Port/Starboard)  
- Extracted group-based features from `PassengerId`  
  - Group ID  
  - Group Size  
- Handled missing values using:
  - Median for numerical columns  
  - Mode for categorical columns  

---

### 2️⃣ Feature Engineering  

New features created:

| Feature | Meaning |
|--------|---------|
| `TotalSpending` | Sum of all spending columns |
| `SpentAnything` | Whether passenger spent money |
| `Cryo_x_Spend` | Interaction between CryoSleep and spending |
| `AvgSpendPerPerson` | Spending normalized by group size |

---

### 3️⃣ Models Tried  

We trained and compared multiple classifiers:

- Logistic Regression  
- Random Forest  
- K-Nearest Neighbors  
- Support Vector Machine (SVM)  
- Gradient Boosting  
- Extra Trees  
- AdaBoost  

---

### 4️⃣ Best Model – Ensemble  

The final submission used an **ensemble model** combining:

- Extra Trees  
- Gradient Boosting  

Final Public Score on Kaggle: **0.80102**

This improved performance compared to individual models.

---

## 🧪 How to Run the Code  


```bash
pip install pandas numpy scikit-learn
python spaceship_titanic.py 
```

📈 Results
Model	Accuracy
Extra Trees	0.805
Gradient Boosting	0.796
Ensemble	0.808 (validation)
Kaggle Score	0.79869