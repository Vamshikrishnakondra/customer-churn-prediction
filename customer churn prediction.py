
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Customer Churn Prediction",
                   page_icon="📊",
                   layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["AvgCharges"] = df["TotalCharges"] / (df["tenure"] + 1)
    return df

def preprocess(df):
    d = df.copy()
    enc = LabelEncoder()
    for c in d.select_dtypes(include="object").columns:
        d[c] = enc.fit_transform(d[c])
    scaler = StandardScaler()
    num = ["tenure","MonthlyCharges","TotalCharges","AvgCharges"]
    d[num] = scaler.fit_transform(d[num])
    X = d.drop(["customerID","Churn"], axis=1)
    y = d["Churn"]
    return train_test_split(X,y,test_size=0.2,
                            random_state=42,
                            stratify=y)

df = load_data()

st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Choose Analysis",
    ["EDA","Visualizations","Train Models",
     "Feature Importance","Model Comparison"]
)

if option == "EDA":
    st.subheader("Dataset")
    st.dataframe(df.head())
    st.write(df.describe())
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Churn", ax=ax)
    st.pyplot(fig)

elif option == "Visualizations":
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Contract", hue="Churn", ax=ax)
    plt.xticks(rotation=30)
    st.pyplot(fig)

elif option == "Train Models":
    X_train,X_test,y_train,y_test = preprocess(df)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train,y_train)

    pred = lr.predict(X_test)
    prob = lr.predict_proba(X_test)[:,1]

    st.write("Accuracy:", accuracy_score(y_test,pred))
    st.text(classification_report(y_test,pred))
    st.write("ROC-AUC:", roc_auc_score(y_test,prob))

    sm = SMOTE(random_state=42)
    X_sm,y_sm = sm.fit_resample(X_train,y_train)
    lr.fit(X_sm,y_sm)

elif option == "Feature Importance":
    X_train,X_test,y_train,y_test = preprocess(df)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train,y_train)

    feat = pd.DataFrame({
        "Feature":X_train.columns,
        "Importance":rf.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.dataframe(feat)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=feat.head(10),
                x="Importance", y="Feature", ax=ax)
    st.pyplot(fig)

elif option == "Model Comparison":

    st.header("📊 Model Comparison")

    X_train,X_test,y_train,y_test = preprocess(df)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train,y_train)
        pred = model.predict(X_test)

        if hasattr(model,"predict_proba"):
            score = model.predict_proba(X_test)[:,1]
        else:
            score = model.decision_function(X_test)

        results.append({
            "Model":name,
            "Accuracy":accuracy_score(y_test,pred),
            "ROC-AUC":roc_auc_score(y_test,score)
        })

    results = pd.DataFrame(results).sort_values(
        "ROC-AUC", ascending=False)

    st.dataframe(results, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=results,
                x="Accuracy",
                y="Model",
                ax=ax)
    st.pyplot(fig)
