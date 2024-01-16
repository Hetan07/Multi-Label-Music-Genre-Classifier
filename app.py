import streamlit as st
import joblib
import pandas as pd
import numpy as np
import xgboost

# Local Imports
import feature_extraction
import audio_splitting

st.set_page_config(layout="wide")
# Vars
fields_df = ['Chromagram Short-Time Fourier Transform (Chroma-STFT)',
             'Root Mean Square Energy (RMS)',
             'Spectral Centroid',
             'Spectral Bandwidth',
             'Spectral Rolloff',
             'Zero Crossing Rate',
             'Harmony',
             'Percussion',
             'Tempo',
             'Mel-Frequency Cepstral Coefficients (MFCC-1)',
             'MFCC-2',
             'MFCC-3',
             'MFCC-4',
             'MFCC-5',
             'MFCC-6',
             'MFCC-7',
             'MFCC-8',
             'MFCC-9',
             'MFCC-10',
             'MFCC-11',
             'MFCC-12',
             'MFCC-13',
             'MFCC-14',
             'MFCC-15',
             'MFCC-16',
             'MFCC-17',
             'MFCC-18',
             'MFCC-19',
             'MFCC-20', ]

st.title("Music Genre Classifier")
st.write("A single-label music genre classifier based and trained on the GTZAN Dataset available for use on "
         "Kaggle. All the models have been trained on that dataset.")

st.write("Prediction of following genres")

class_names = ["Blues", "Classical", "Country", "Disco", "HipHop",
               "Jazz", "Metal", "Pop", "Reggae", "Rock"]

class_indices = {i: class_name for i, class_name in enumerate(class_names)}

col1, col2 = st.columns(2)
s = ''
with col1:
    for i in class_names[:5]:
        s += "- " + i + "\n"
    st.markdown(s)

s = ''

with col2:
    for i in class_names[5:]:
        s += "- " + i + "\n"
    st.markdown(s)

st.divider()
# Upload music file
st.subheader("Upload a music file")
uploaded_file = st.file_uploader("Upload a music file", type=["mp3", "wav", "ogg"], label_visibility="collapsed")

st.divider()
if uploaded_file is not None:
    # User selects a model
    all_models = ["K-Nearest Neighbors",
                  "Logistic Regression",
                  "Support Vector Machines",
                  "Neural Network",
                  "XGB Classifier"]

    features_list, val_list = audio_splitting.split_audio(uploaded_file)
    features = feature_extraction.scale(features_list)

    feature_copy = features_list
    feature_copy.insert(19, "-")
    st.header("Feature Extraction")

    st.write("The given audio sample is processed using the librosa library to get the features extracted used by the "
             "models for genre prediction. Following is the dataframe with each of the feature extracted and "
             "corresponding mean and variance of the feature")

    col3, col4 = st.columns([0.6,0.4])
    with col3:

        # Features Dataframe
        df = pd.DataFrame({
            "name": fields_df,
            "Mean": feature_copy[2::2],
            "Variance": feature_copy[3::2]
        })

        st.dataframe(
            df,
            column_config={
                "name": "Features",
                "Mean": "Mean of Feature",
                "Variance": "Variance of Feature"
            },
            use_container_width=True
        )

    with col4:

        col1, col2 = st.columns([0.55, 0.45])

        col1.subheader("Select a model")
        with col1:
            model_name = st.selectbox("Select a model", all_models, label_visibility="collapsed")

            # Load the selected model
            if model_name == "K-Nearest Neighbors":
                model = joblib.load("./models/knn.pkl")
            elif model_name == "Logistic Regression":
                model = joblib.load("./models/logistic.pkl")
            elif model_name == "Support Vector Machines":
                model = joblib.load("./models/svm.pkl")
            elif model_name == "Neural Network":
                model = joblib.load("./models/nn.pkl")
            elif model_name == "XGB Classifier":
                model = joblib.load("./models/xgb.pkl")
        col2.subheader("Predicted genre")

        # Reshape the features to match the expected shape for prediction
        reshaped_features = features.reshape(1, -1)

        if model_name == "XGB Classifier":
            predicted_indices = model.predict(reshaped_features)
            predicted_labels = [class_indices[i] for i in predicted_indices]
            with col2:
                st.metric("Predicted Genre:", str(predicted_labels[0]), label_visibility="collapsed")
        else:
            predicted_label = model.predict(features)[0]
            with col2:
                st.metric("Predicted Genre:", str(predicted_label).capitalize(), label_visibility="collapsed")
