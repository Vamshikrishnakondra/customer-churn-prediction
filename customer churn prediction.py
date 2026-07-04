import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")

st.write("Upload the Telco Customer Churn Dataset")

#uploaded_file = st.file_uploader("WA_Fn-UseC_-Telco-Customer-Churn.csv", type=["csv"])

if True:

    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    st.success("Dataset Loaded Successfully")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write("Shape:", df.shape)

    st.subheader("Dataset Information")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    # -----------------------------
    # Data Cleaning
    # -----------------------------

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # -----------------------------
    # Sidebar
    # -----------------------------

    option = st.sidebar.selectbox(
        "Choose Analysis",
        [
            "EDA",
            "Visualizations",
            "Train Models",
            "Feature Importance",
            "Model Comparison"
        ]
    )

    # =====================================================
    # EDA
    # =====================================================

    if option == "EDA":

        st.header("Exploratory Data Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Gender Count")
            st.write(df['gender'].value_counts())

            st.write("Partner Count")
            st.write(df['Partner'].value_counts())

        with col2:
            st.write("Dependents Count")
            st.write(df['Dependents'].value_counts())

            st.write("Senior Citizen")
            st.write(df['SeniorCitizen'].value_counts())

        st.write("### Churn Distribution")

        fig, ax = plt.subplots()

        sns.countplot(x='Churn', data=df, ax=ax)

        st.pyplot(fig)

    # =====================================================
    # VISUALIZATION
    # =====================================================

    elif option == "Visualizations":

        st.header("Visualizations")

        fig, ax = plt.subplots()

        sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=ax)

        st.pyplot(fig)

        fig, ax = plt.subplots()

        sns.countplot(x='Contract', hue='Churn', data=df, ax=ax)

        plt.xticks(rotation=30)

        st.pyplot(fig)

        fig, ax = plt.subplots()

        df["Churn"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            explode=[0,0.1],
            colors=["orange","skyblue"],
            shadow=True,
            startangle=90,
            ax=ax
        )

        ax.set_ylabel("")

        st.pyplot(fig)

        fig, ax = plt.subplots()

        sns.histplot(df["MonthlyCharges"], bins=30, kde=True, ax=ax)

        st.pyplot(fig)

    # =====================================================
    # TRAIN MODELS
    # =====================================================

    elif option == "Train Models":

        st.header("Machine Learning Models")

        df_ml = df.copy()

        encoder = LabelEncoder()

        cat_cols = df_ml.select_dtypes(include="object").columns

        for col in cat_cols:
            df_ml[col] = encoder.fit_transform(df_ml[col])

        scaler = StandardScaler()

        cols = ['tenure','MonthlyCharges','TotalCharges']

        df_ml[cols] = scaler.fit_transform(df_ml[cols])

        X = df_ml.drop(["customerID","Churn"], axis=1)

        y = df_ml["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        st.subheader("Logistic Regression")

        lr = LogisticRegression(max_iter=1000)

        lr.fit(X_train,y_train)

        pred = lr.predict(X_test)

        prob = lr.predict_proba(X_test)[:,1]

        st.write("Accuracy:", accuracy_score(y_test,pred))

        st.text(classification_report(y_test,pred))

        st.write("ROC-AUC:", roc_auc_score(y_test,prob))

        st.subheader("Random Forest")

        rf = RandomForestClassifier(random_state=42)

        rf.fit(X_train,y_train)

        pred_rf = rf.predict(X_test)

        st.write("Accuracy:",accuracy_score(y_test,pred_rf))

        st.text(classification_report(y_test,pred_rf))

        st.subheader("SMOTE")

        sm = SMOTE()

        X_sm,y_sm = sm.fit_resample(X_train,y_train)

        lr.fit(X_sm,y_sm)

        pred_sm = lr.predict(X_test)

        prob_sm = lr.predict_proba(X_test)[:,1]

        st.write("Accuracy:",accuracy_score(y_test,pred_sm))

        st.write("ROC-AUC:",roc_auc_score(y_test,prob_sm))

        st.subheader("ROC Curve")

        fpr,tpr,_ = roc_curve(y_test,prob)

        fig,ax = plt.subplots()

        ax.plot(fpr,tpr,label="Logistic")

        ax.plot([0,1],[0,1],'--')

        ax.legend()

        st.pyplot(fig)

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    elif option == "Feature Importance":

        st.header("Top Important Features")

        df_ml = df.copy()

        encoder = LabelEncoder()

        cat_cols = df_ml.select_dtypes(include="object").columns

        for col in cat_cols:
            df_ml[col]=encoder.fit_transform(df_ml[col])

        scaler=StandardScaler()

        cols=['tenure','MonthlyCharges','TotalCharges']

        df_ml[cols]=scaler.fit_transform(df_ml[cols])

        X=df_ml.drop(["customerID","Churn"],axis=1)

        y=df_ml["Churn"]

        X_train,X_test,y_train,y_test=train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        rf=RandomForestClassifier(random_state=42)

        rf.fit(X_train,y_train)

        feature=pd.DataFrame({

            "Feature":X.columns,

            "Importance":rf.feature_importances_

        })

        feature=feature.sort_values(by="Importance",ascending=False)

        st.dataframe(feature)

        fig,ax=plt.subplots(figsize=(8,5))

        sns.barplot(
            x="Importance",
            y="Feature",
            data=feature.head(10),
            ax=ax
        )

        st.pyplot(fig)

    # =====================================================
    # MODEL COMPARISON
    # =====================================================

  elif option == "Model Comparison":

    st.header("📊 Model Comparison")

    df_ml = df.copy()

    # Convert TotalCharges to numeric
    df_ml["TotalCharges"] = pd.to_numeric(df_ml["TotalCharges"], errors="coerce")
    df_ml["TotalCharges"].fillna(df_ml["TotalCharges"].median(), inplace=True)

    # Create AvgCharges BEFORE scaling (optional)
    if "AvgCharges" not in df_ml.columns:
        df_ml["AvgCharges"] = df_ml["TotalCharges"] / (df_ml["tenure"] + 1)

    # Replace inf values
    df_ml.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_ml.fillna(df_ml.median(numeric_only=True), inplace=True)

    # Label Encoding
    encoder = LabelEncoder()

    for col in df_ml.select_dtypes(include="object").columns:
        df_ml[col] = encoder.fit_transform(df_ml[col])

    # Scale Numerical Columns
    scaler = StandardScaler()

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgCharges"]

    df_ml[num_cols] = scaler.fit_transform(df_ml[num_cols])

    # Features & Target
    X = df_ml.drop(["customerID", "Churn"], axis=1)
    y = df_ml["Churn"]

    # Final safety check
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(numeric_only=True), inplace=True)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            prob = model.decision_function(X_test)

        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, prob)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "ROC-AUC": round(auc, 4)
        })

    result = pd.DataFrame(results)
    result = result.sort_values(by="ROC-AUC", ascending=False)

    st.dataframe(result, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=result,
        x="Accuracy",
        y="Model",
        ax=ax
    )

    ax.set_title("Model Accuracy Comparison")

    st.pyplot(fig)
