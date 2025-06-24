import pandas as pd

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

import streamlit as st

import base64



st.set_page_config(page_title="Air Quality Dashboard", layout="wide")





# Load and preprocess dataset

@st.cache_data



def load_data():

    df = pd.read_csv("Air_Quality.csv")

    df.drop(columns=['CO2','Date', 'City'], inplace=True)

    df.fillna(method='ffill', inplace=True)

    df['AQI_Category'] = (df['AQI'] > 100).astype(int)

    return df



df = load_data()

od = df.copy()

X = df.drop(['AQI', 'AQI_Category'], axis=1)

y = df['AQI_Category']

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)



# Sidebar controls

st.sidebar.title("📂 Sections")

section = st.sidebar.radio("Navigate", [

    "Data Overview", "Visualizations", "Model: Logistic Regression",

    "Model: Polynomial Regression", "Model: Random Forest","Prediction Panel"])



# Data Overview

if section == "Data Overview":

    st.subheader("🧾 Data Preview")

    st.write(od)

    st.subheader('Data Description')

    st.dataframe(df.describe(include='all'))

    st.write("Shape:", df.shape)

    st.write("Missing Values:")

    st.dataframe(df.isnull().sum().reset_index().rename(columns={0: 'Missing Count', 'index': 'Column'}))

elif section == "Visualizations":

    st.subheader("📊 Visualization Section")

    viz_option = st.sidebar.selectbox("Choose a visualization:",

        ("AQI Category Distribution", 

         "Feature Correlation Heatmap", 

         "NO2 vs PM2.5", 

         "SO2 Distribution",

         "Hourly AQI Trends",

         "CO vs O3 Scatter Plot",

         "Pollutant Violin Plots"))



    if viz_option == "AQI Category Distribution":

        fig1 = px.pie(df, names='AQI_Category', title='Good vs Poor Air Quality')

        st.plotly_chart(fig1, use_container_width=True)



    elif viz_option == "Feature Correlation Heatmap":

        fig2 = px.imshow(df.drop(['AQI_Category'], axis=1).corr(), text_auto=True)

        st.plotly_chart(fig2, use_container_width=True)



    elif viz_option == "NO2 vs PM2.5":

        fig3 = px.scatter(df, x='NO2', y='PM2.5', color='AQI_Category', title='NO2 vs PM2.5')

        st.plotly_chart(fig3, use_container_width=True)



    elif viz_option == "SO2 Distribution":

        fig4 = px.histogram(df, x='SO2', color='AQI_Category', nbins=40)

        st.plotly_chart(fig4, use_container_width=True)

        


        

        

    elif viz_option == "CO vs O3 Scatter Plot":

        fig8 = px.scatter(df, x='CO', y='O3', color='AQI_Category',

                         title='CO vs O3 Levels',

                         marginal_x="histogram", 

                         marginal_y="histogram")

        st.plotly_chart(fig8, use_container_width=True)

        

    elif viz_option == "AQI Distribution by Hour":

        df['Hour'] = pd.to_datetime(df['Date']).dt.hour

        fig9 = px.box(df, x='Hour', y='AQI', 

                     title='AQI Distribution by Hour of Day',

                     color='Hour')

        st.plotly_chart(fig9, use_container_width=True)

        

    elif viz_option == "Pollutant Violin Plots":

        pollutants = ['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']

        selected_pollutant = st.selectbox('Select pollutant:', pollutants)

        

        fig10 = px.violin(df, y=selected_pollutant, box=True, 

                         points="all",

                         title=f'Distribution of {selected_pollutant} Levels')

        st.plotly_chart(fig10, use_container_width=True)



elif section == "Model: Logistic Regression":

    st.subheader("📈 Logistic Regression")

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    st.text(classification_report(y_test, preds))

    st.write(f"Accuracy: {accuracy_score(y_test, preds):.2f}")

    cm = confusion_matrix(y_test, preds)

    fig = px.imshow(cm, text_auto=True, x=['Good', 'Poor'], y=['Good', 'Poor'])

    st.plotly_chart(fig, use_container_width=True)



elif section == "Model: Polynomial Regression":

    st.subheader("🧮 Polynomial Logistic Regression")

    poly = PolynomialFeatures(degree=2)

    X_train_poly = poly.fit_transform(X_train)

    X_test_poly = poly.transform(X_test)

    poly_model = LogisticRegression(max_iter=1000)

    poly_model.fit(X_train_poly, y_train)

    preds = poly_model.predict(X_test_poly)

    st.text(classification_report(y_test, preds))

    st.write(f"Accuracy: {accuracy_score(y_test, preds):.2f}")

    cm = confusion_matrix(y_test, preds)

    fig = px.imshow(cm, text_auto=True, x=['Good', 'Poor'], y=['Good', 'Poor'])

    st.plotly_chart(fig, use_container_width=True)



elif section == "Model: Random Forest":

    st.subheader("🌲 Random Forest Classifier")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    st.text(classification_report(y_test, y_pred))

    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    cm = confusion_matrix(y_test, y_pred)

    fig = px.imshow(cm, text_auto=True, x=['Good', 'Poor'], y=['Good', 'Poor'])

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Feature Importance")

    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values('Importance')

    fig_imp = px.bar(feature_importance, x='Importance', y='Feature', orientation='h')

    st.plotly_chart(fig_imp, use_container_width=True)





elif section == "Prediction Panel":

    st.subheader("🎯 Predict AQI Category")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            CO = st.number_input("CO (ppm)", min_value=0.0)

            NO2 = st.number_input("NO2 (ppb)", min_value=0.0)

            SO2 = st.number_input("SO2 (ppb)", min_value=0.0)

            O3 = st.number_input("O3 (ppb)", min_value=0.0)

        with col2:

            PM25 = st.number_input("PM2.5 (ug/m3)", min_value=0.0)

            PM10 = st.number_input("PM10 (ug/m3)", min_value=0.0)



        submitted = st.form_submit_button("Predict AQI")



    if submitted:

        user_data = pd.DataFrame({

            'CO': [CO],

            'NO2': [NO2],

            'SO2': [SO2],

            'O3': [O3],

            'PM2.5': [PM25],

            'PM10': [PM10],

        })

        user_scaled = scaler.transform(user_data)



        log_model = LogisticRegression(max_iter=1000)

        log_model.fit(X_train, y_train)

        log_pred = log_model.predict(user_scaled)[0]

        log_prob = log_model.predict_proba(user_scaled)[0][1]



        poly = PolynomialFeatures(degree=2)

        poly_model = LogisticRegression(max_iter=1000)

        X_poly = poly.fit_transform(X_train)

        poly_model.fit(X_poly, y_train)

        user_poly = poly.transform(user_scaled)

        poly_pred = poly_model.predict(user_poly)[0]

        poly_prob = poly_model.predict_proba(user_poly)[0][1]



        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        rf.fit(X_train, y_train)

        rf_pred = rf.predict(user_scaled)[0]

        rf_prob = rf.predict_proba(user_scaled)[0][1]



        st.subheader("Model Predictions")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric("Logistic Regression", "Poor" if log_pred else "Good", f"{log_prob:.2%}")

        with col2:

            st.metric("Polynomial Regression", "Poor" if poly_pred else "Good", f"{poly_prob:.2%}")

        with col3:

            st.metric("Random Forest", "Poor" if rf_pred else "Good", f"{rf_prob:.2%}")



        if sum([log_pred, poly_pred, rf_pred]) >= 2:

            st.error("⚠ Final Verdict: POOR AIR QUALITY")

        else:

            st.success("✅ Final Verdict: Good Air Quality")
