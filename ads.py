import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def train_models(X_single, X_multiple, y):
    lin_model_single = LinearRegression()
    lin_model_single.fit(X_single, y)

    svr_model_single = SVR(kernel='linear')
    svr_model_single.fit(X_single, y)

    lin_model_multiple = None
    svr_model_multiple = None
    
    if X_multiple is not None:
        lin_model_multiple = LinearRegression()
        lin_model_multiple.fit(X_multiple, y)

        svr_model_multiple = SVR(kernel='linear')
        svr_model_multiple.fit(X_multiple, y)

    return lin_model_single, svr_model_single, lin_model_multiple, svr_model_multiple

def predict_sales(model, feature_values):
    return model.predict([feature_values])[0]

def main():
    st.title("Sales Prediction Based on Ad Spend")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        st.write("Dataset Preview:")
        st.write(data.head())

        if 'TV' in data.columns and 'sales' in data.columns:
            X_single = data[['TV']]
            y = data['sales']

            lin_model_single, svr_model_single, lin_model_multiple, svr_model_multiple = train_models(X_single, 
                    data[['TV', 'radio', 'newspaper']] if 'radio' in data.columns and 'newspaper' in data.columns else None, 
                    y)

            model_choice = st.selectbox("Choose the model", ["Linear Regression", "SVR"])
            feature_choice = st.selectbox("Choose the feature type", ["Single", "Multiple"])

            if feature_choice == "Single":
                tv_spend = st.slider("TV Ad Spend (in $)", min_value=0, max_value=int(data['TV'].max()), value=100)

                if model_choice == "Linear Regression":
                    model = lin_model_single
                else:
                    model = svr_model_single

                predicted_sales = predict_sales(model, [tv_spend])
                st.write(f"Predicted Sales: ${predicted_sales:.2f} million for TV ad spend of ${tv_spend}.")

            elif feature_choice == "Multiple":
                if lin_model_multiple and svr_model_multiple:
                    tv_spend = st.slider("TV Ad Spend (in $)", min_value=0, max_value=int(data['TV'].max()), value=100)
                    radio_spend = st.slider("Radio Ad Spend (in $)", min_value=0, max_value=int(data['radio'].max()), value=50)
                    newspaper_spend = st.slider("Newspaper Ad Spend (in $)", min_value=0, max_value=int(data['newspaper'].max()), value=30)

                    if model_choice == "Linear Regression":
                        model = lin_model_multiple
                    else:
                        model = svr_model_multiple

                    predicted_sales = predict_sales(model, [tv_spend, radio_spend, newspaper_spend])
                    st.write(f"Predicted Sales: ${predicted_sales:.2f} million for TV ad spend of ${tv_spend}, Radio ad spend of ${radio_spend}, and Newspaper ad spend of ${newspaper_spend}.")
                else:
                    st.write("Error: The dataset must contain 'radio' and 'newspaper' columns for multiple feature prediction.")
        else:
            st.write("Error: The dataset must contain 'TV' and 'sales' columns.")
    else:
        st.write("Please upload a CSV file.")

if __name__ == "__main__":
    main()
