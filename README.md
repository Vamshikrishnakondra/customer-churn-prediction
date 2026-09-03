📊 Customer Churn Prediction Dashboard

An interactive Machine Learning and Data Science web application built with Python and Streamlit to analyze customer churn and compare multiple classification algorithms.

The application performs Exploratory Data Analysis (EDA), data preprocessing, machine learning model training, feature importance analysis, and model comparison through an easy-to-use dashboard.

🚀 Live Project Overview

Customer churn prediction helps businesses identify customers who are likely to stop using their services.

This project uses the Telco Customer Churn dataset to:

Analyze customer behavior and churn patterns
Preprocess numerical and categorical features
Handle missing values
Address class imbalance using SMOTE
Train multiple machine learning models
Compare models using Accuracy and ROC-AUC
Identify important features influencing churn
Visualize important customer patterns through an interactive dashboard
✨ Features
📌 1. Exploratory Data Analysis

The EDA section provides:

Dataset preview
Statistical summary
Churn distribution
Basic understanding of customer data
📊 2. Data Visualizations

The dashboard includes visualizations such as:

Monthly charges vs. churn
Contract type vs. churn
Churn distribution
Feature importance
🤖 3. Machine Learning Models

The project compares several classification algorithms:

Logistic Regression
Random Forest
Decision Tree
Gradient Boosting
Support Vector Machine (SVM)
⚖️ 4. Handling Class Imbalance

The dataset contains an imbalance between churned and non-churned customers.

SMOTE (Synthetic Minority Oversampling Technique) is used to generate synthetic samples for the minority class.

🔍 5. Feature Importance

Random Forest is used to identify the features that contribute most to customer churn.

This helps answer questions such as:

Which customer characteristics are most strongly associated with churn?

📈 6. Model Comparison

Models are evaluated using:

Accuracy
ROC-AUC

The dashboard displays the results in both a table and a visualization.

🛠️ Tech Stack
Technology	Purpose
Python	Programming language
Pandas	Data manipulation
NumPy	Numerical computation
Scikit-learn	Machine learning
Imbalanced-learn	SMOTE
Matplotlib	Data visualization
Seaborn	Statistical visualization
Streamlit	Interactive dashboard
🧠 Machine Learning Workflow
Raw Customer Data
        ↓
Data Cleaning
        ↓
Missing Value Handling
        ↓
Feature Engineering
        ↓
Categorical Encoding
        ↓
Feature Scaling
        ↓
Train/Test Split
        ↓
SMOTE
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Feature Importance
        ↓
Model Comparison
        ↓
Streamlit Dashboard

📂 Project Structure
customer-churn-prediction/
│
├── app.py
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
├── README.md
└── screenshots/
    ├── eda.png
    ├── visualizations.png
    ├── model-comparison.png
    └── feature-importance.png

📊 Dataset

The project uses the Telco Customer Churn dataset, which contains customer information such as:

Customer demographics
Tenure
Contract type
Monthly charges
Total charges
Internet services
Payment methods
Churn status
Target Variable

Churn

Yes → Customer left the service
No → Customer remained with the service
⚙️ Feature Engineering

A new feature called AvgCharges is created:

AvgCharges = TotalCharges / (tenure + 1)


This feature provides an approximate measure of the customer's average charges over their relationship with the company.

The +1 prevents division by zero for customers with zero tenure.

🔄 Data Preprocessing

The following preprocessing steps are performed:

Convert TotalCharges to numeric format
Handle missing values using the median
Create the AvgCharges feature
Encode categorical variables using LabelEncoder
Standardize numerical features using StandardScaler
Remove unnecessary customerID
Split the dataset into training and testing sets
Apply SMOTE to address class imbalance
📏 Model Evaluation

The models are evaluated using two major metrics.

Accuracy

Measures the percentage of correctly classified customers.

Accuracy =
Correct Predictions / Total Predictions

ROC-AUC

Measures how well the model distinguishes between:

Customers likely to churn
Customers likely to stay

A higher ROC-AUC generally indicates better classification performance.

📸 Dashboard Screenshots

Add screenshots of your Streamlit application here.

EDA

Visualizations

Model Comparison

Feature Importance

💻 Installation
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git

2. Navigate to the project
cd customer-churn-prediction

3. Create a virtual environment
python -m venv venv

4. Activate the environment

Windows:

venv\Scripts\activate


macOS/Linux:

source venv/bin/activate

5. Install dependencies
pip install -r requirements.txt

6. Run the Streamlit application
streamlit run app.py


The application will open in your browser.

📦 requirements.txt
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn

📌 Key Learning Outcomes

Through this project, I gained practical experience in:

Exploratory Data Analysis
Data preprocessing
Feature engineering
Categorical encoding
Feature scaling
Handling imbalanced datasets
SMOTE
Classification algorithms
Model evaluation
ROC-AUC analysis
Feature importance
Data visualization
Streamlit application development
🔮 Future Improvements

Possible improvements include:

Add customer-level churn prediction
Add interactive prediction form
Add probability of churn
Add SHAP explainability
Add hyperparameter tuning
Add cross-validation
Add XGBoost/LightGBM
Deploy the application online
Add database integration
Add automated model retraining
👨‍💻 Author

Your Name

Aspiring Data Scientist | Machine Learning | Generative AI

📌 GitHub: https://github.com/YOUR_USERNAME

⭐ If you found this project useful

Consider giving the repository a ⭐ on GitHub!

CUSTOMER CHURN PREDICTION STREAMLIT
https://customer-churn-prediction-hobv6hfz8otvozty4gyaf2.streamlit.app/
