import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
train = pd.read_csv("spaceship-titanic/spaceship_titanic_train.csv")
test = pd.read_csv("spaceship-titanic/spaceship_titanic_test.csv")
train_id = train["PassengerId"]
test_id = test["PassengerId"]
y = train["Transported"]
train = train.drop("Transported", axis = 1)
all_data = pd.concat([train, test], axis = 0)
all_data[["Deck", "Num", "Side"]] = all_data["Cabin"].str.split("/", expand=True)
all_data["Num"] = all_data["Num"].astype(float)
all_data["Group"] = all_data["PassengerId"].str.split("_").str[0].astype(int)
all_data["GroupSize"] = all_data.groupby("Group")["PassengerId"].transform("count")
num_col = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Num"]
all_data[num_col] = all_data[num_col].fillna(all_data[num_col].median())
spend_cols = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
all_data["TotalSpending"] = all_data[spend_cols].sum(axis = 1)
cat_col = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
for col in cat_col:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
all_data["CryoSleep"] = all_data["CryoSleep"].astype(int)
all_data["VIP"] = all_data["VIP"].astype(int)
all_data["SpentAnything"] = (all_data["TotalSpending"] > 0).astype(int)
all_data["Cryo_x_Spend"] = all_data["CryoSleep"] * all_data["TotalSpending"]
all_data["AvgSpendPerPerson"] = all_data["TotalSpending"] / all_data["GroupSize"]
all_data = all_data.drop(["PassengerId", "Cabin", "Name"], axis = 1)
all_data = pd.get_dummies(all_data, drop_first=True)
x = all_data.iloc[ : len(train), : ]
test = all_data.iloc[len(train):,:]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2, stratify = y)
lr = LogisticRegression(max_iter=10000)
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
rf = RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_split=5, min_samples_leaf=2, random_state= 42)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
model = LogisticRegression(max_iter=2000)
model.fit(x_train_scaled, y_train)
y_pred_sc = model.predict(x_test_scaled)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train_scaled, y_train)
y_pred_kn = knn.predict(x_test_scaled)
svm = SVC(kernel= "rbf", C=1, gamma="scale")
svm.fit(x_train_scaled, y_train)
y_pred_sv = svm.predict(x_test_scaled)
gm = GradientBoostingClassifier(random_state=42)
gm.fit(x_train, y_train)
y_pred_gm = gm.predict(x_test)
et = ExtraTreesClassifier(n_estimators=1000, random_state=42)
et.fit(x_train, y_train)
y_pred_et = et.predict(x_test)
et2 = ExtraTreesClassifier(
    n_estimators=1500,
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
et2.fit(x_train, y_train)
y_pred_et2 = et2.predict(x_test)
ada = AdaBoostClassifier(n_estimators=300, random_state=42)
ada.fit(x_train, y_train)
y_pred_ad = ada.predict(x_test)
test_scaled = scaler.transform(test)
prob_et = et.predict_proba(x_test)[:, 1]
prob_et2 = et2.predict_proba(x_test)[:, 1]
prob_gm = gm.predict_proba(x_test)[:, 1]
ensemble_pred = (prob_et + prob_gm) / 2 > 0.5
ensemble_pred2 = (prob_et2 + prob_gm) / 2 > 0.5
results = {
    "Logistic Regression (Unscaled)": pred,
    "Logistic Regression (Scaled)": y_pred_sc,
    "Random Forest": y_pred_rf,
    "K-Nearest Neighbors": y_pred_kn,
    "Support Vector Machine": y_pred_sv,
    "Gradient Boosting": y_pred_gm,
    "Extra Trees": y_pred_et,
    "Extra Trees 2": y_pred_et2,
    "AdaBoost": y_pred_ad,
    "Ensemble": ensemble_pred,
    "Ensemble 2": ensemble_pred2
}

print("--- MODEL PERFORMANCE COMPARISON ---\n")

for name, y_pred in results.items():
    print(f"Model: {name}")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 30)


best_model1 = et2.predict_proba(test)[:, 1]
best_model2 = gm.predict_proba(test)[:, 1]
best_ensemble_pred = (best_model1 + best_model2) / 2 > 0.5
submission_ens = pd.DataFrame({
    "PassengerId": test_id,
    "Transported": best_ensemble_pred
})
submission_ens.to_csv("submission_ensemble.csv", index=False)