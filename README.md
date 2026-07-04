# Spaceship Titanic Predictor

Predicts whether a passenger aboard the Spaceship Titanic was transported to an alternate dimension.

**Dataset:** Kaggle – Spaceship Titanic
**Goal:** Predict whether a passenger was Transported (True/False)
**Type:** Binary classification — evaluated with accuracy

---

## Approach

### 1. Data Cleaning & Preprocessing
- Combined train and test data
- Extracted features from `Cabin`: Deck, Cabin Number, Side (Port/Starboard)
- Extracted group-based features from `PassengerId`: Group ID, Group Size
- Handled missing values: median for numerical columns, mode for categorical

### 2. Feature Engineering

| Feature | Meaning |
|---------|---------|
| `TotalSpending` | Sum of all spending columns |
| `SpentAnything` | Whether the passenger spent money |
| `Cryo_x_Spend` | Interaction between CryoSleep and spending |
| `AvgSpendPerPerson` | Spending normalized by group size |

---

## Models compared

- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gradient Boosting
- Extra Trees
- AdaBoost

Final submission: an **ensemble** of Extra Trees + Gradient Boosting.

---

## Results

| Model | Accuracy |
|-------|----------|
| Extra Trees | 0.805 |
| Gradient Boosting | 0.796 |
| Ensemble (validation) | 0.808 |
| Ensemble (Kaggle public) | 0.79869 |

---

## How to run

```bash
pip install -r requirements.txt
python spaceship-titanic.py
```

**Outputs:**
- `submission_ensemble.csv`