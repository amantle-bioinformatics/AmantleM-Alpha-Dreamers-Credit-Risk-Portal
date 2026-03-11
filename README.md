# Alpha-Dreamers-Credit-Risk-Portal

This project identifies and predicts the high risk loan borrowers at a banking instituion called Alpha Dreamers Banking Consortium based on its features. Users will enter the features needed to apply for a loan and the system will return whether the applicant is a high risk or low risk loan creditor to the loan department. This will help the loan department determine if the applicant is eligible for a loan.
It includes a Logistic Regression model built with Python and a web-based interface for real time predictions.

**My successfully deployed web app link**:
https://amantlem-alpha-dreamers-credit-risk-app-mgpwgmdcvd7i928azut3y4.streamlit.app/

## 📈 Project Workflow
The project follows a structured 6-step pipeline:

***Dataset**: Sourcing and understanding the CSEdata.csv data.
***Data Cleaning**: Handling missing values, outliers, and encoding categorical variables (e.g., converting "yes/no" to numerical values).
***Model Training**: Training a regression model (such as Logistic Regression or Random Forest) using Scikit-Learn.
***Save Model**: Exporting the trained model using pickle or joblib for future use.
***Build Web App**: Developing an interactive user interface using Streamlit.
***Deploy**: Hosting the application (e.g., on Streamlit Cloud or Render) so users can access it via a URL.

---

The dataset (`CSEdata.csv`) contains **252000 records** with the following features:

| Column Name | Type |
| --- | --- |
| `Income` | Numeric |
| `Age` | Numeric |
| `Experience` | Numeric |
| `Married/Single` | Categorical |
| `House_Ownership` | Categorical |
| `Car_Ownership` | Categorical |
| `Profession` | Categorical |
| `CITY` | Categorical |
| `STATE` | Categorical |
| `CURRENT_JOB_YRS` | Numeric |
| `CURRENT_HOUSE_YRS` | Numeric |

---

## 🛠️ Tech Stack

* **Language**: Python 3.x
* **Data Analysis**: Pandas, NumPy
* **Machine Learning**: Scikit-learn (Logistic Regression, Random Forest, etc.)
* **Save Model**: Pickle or Joblib
* **Web Framework**: [Streamlit](https://streamlit.io/) (Recommended for beginners) or Flask
* **Environment**: Jupyter Notebook/Google Colab (for Cleaning and Training) and Streamlit Cloud or Render (for Deployment)

---

## 📂 Project Structure

```text
├── data/
│   └── CSEdata.csv             # Raw dataset
├── models/scaler
│   └── credit_risk_model.joblib         # Saved trained model
    └── scaler.joblib                    #Saved translator
├── credit_risk_app.py                      # Streamlit/Web application code
├── requirements.txt            # List of libraries to install
└── README.md                   # Project documentation

```

## 💡 Key Learnings

* **Handling Categorical Data**: Use `Binary Mapping`, `One_HotEncoder` or `TargetEncoder` to convert categorical data into numeric data so the computer can understand them.
* **Feature Scaling**: Use `StandardScaler` to make numeric data comparable
* **Model Evaluation**: We use a weighted Logistic Regression (`class=`balanced``) to balance the data and **Evaluation Metrics**  to check how accurate, precise and recall our price predictions are.
* **Model Persistence (The "Pickle" or "Joblib" Moment)**:This is often the biggest hurdle for moving to web development. In Colab, the model lives in the computer's temporary memory. Once you close the tab, the model is gone. The Solution is to learn how to Serialize (save) your model as a .pkl or .joblib file. This teaches that a Machine Learning model is just a "weight file" that can be moved from the cloud (Colab) to a local app (Streamlit).
* **Designing for the User (Web UI)**: When you build the web app, you stop thinking like a mathematician and start thinking like a developer. A user won't type 1 for "Yes" and 0 for "No." You learn to build Input Widgets (dropdowns and sliders) that translate user-friendly choices back into the binary data your model expects.
