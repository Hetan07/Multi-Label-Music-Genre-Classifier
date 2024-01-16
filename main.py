import streamlit as st
import joblib
import numpy as np
import xgboost
# from sklearn.ensemble import GradientBoostingClassifier
# from tensorflow.keras.models import load_model
import tensorflow
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from tensorflow import keras
from keras.models import load_model
# Local Imports
import feature_extraction1
import audio_splitting2

# Create a Streamlit web app
st.title("Music Genre Classifier")

# Upload music file
uploaded_file = st.file_uploader("Upload a music file", type=["mp3", "wav"])

if uploaded_file is not None:
    # User selects a model
    all_models = ["K-Nearest Neighbors - (Single Label)", "Logistic Regression - (Single Label)", "Support Vector Machines - (Single Label)",
                  "Neural Network - (Single Label)",
                  "XGB Classifier - (Single Label)", "Convolutional Recurrent Neural Network - (Multi Label)", "XGB - (Multi Label)",
                  "Neural Network - (Multi Label)","Batch Normalization - (Multi Label)"]
    model_name = st.selectbox("Select a model", all_models)
    st.write(f"Predicition of following genres")
    multi_class_names = ["Metal", "Jazz", "Blues", "R&B", "Classical", "Reggae", "Rap & Hip-Hop", "Punk", "Rock",
                         "Country", "Bebop", "Pop", "Soul", "Dance & Electronic", "Folk"]

    st.write(multi_class_names)

    # Load the selected model
    if model_name == "K-Nearest Neighbors - (Single Label)":
        model = joblib.load("./models/knn.pkl")
    elif model_name == "Logistic Regression - (Single Label)":
        model = joblib.load("./models/logistic.pkl")
    elif model_name == "Support Vector Machines - (Single Label)":
        model = joblib.load("./models/svm.pkl")
    elif model_name == "Neural Network - (Single Label)":
        model = joblib.load("./models/nn.pkl")
    elif model_name == "XGB Classifier - (Single Label)":
        model = joblib.load("./models/xgb.pkl")
    elif model_name == "XGB - (Multi Label)":
        model = joblib.load("./models/xgb_mlb.pkl")
    elif model_name == "Convolutional Recurrent Neural Network - (Multi Label)":
        model = tensorflow.keras.models.load_model("./models/model_crnn1.h5", compile=False)
        model.compile(loss=binary_crossentropy,
                      optimizer=Adam(),
                      metrics=['accuracy'])
    elif model_name == "Neural Network - (Multi Label)":
        model = tensorflow.keras.models.load_model("./models/model_nn.h5", compile=False)
        model.compile(loss=binary_crossentropy,
                      optimizer=Adam(),
                      metrics=['accuracy'])
    elif model_name == "Batch Normalization - (Multi Label)":
        model = tensorflow.keras.models.load_model("./models/model_bn.h5", compile=False)
        model.compile(loss=binary_crossentropy,
                      optimizer=Adam(),
                      metrics=['accuracy'])
    class_names = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

    xgb_multi_class_names = ["Rock", "Rap & Hip-Hop", "Soul", "Classical", "Dance & Electronic", "Blues","Jazz",
                             "Country","Bebop","Folk","Reggae","R&B","Punk","Metal","Pop"]

    xmulti_class_names = ["Metal", "Blues", "Reggae", "Jazz", "Rock", "Folk", "Classical", "Dance & Electronic",
                         "Punk","Bebop", "Pop", "R&B", "Country", "Rap & Hip-Hop", "Soul"]
    class_indices = {i: class_name for i, class_name in enumerate(class_names)}

    features_list = audio_splitting2.split_audio(uploaded_file)
    features = feature_extraction1.scale(features_list)

    st.write(features)
    # Reshape the features to match the expected shape for prediction
    reshaped_features = features.reshape(1, -1)
    if model_name == "XGB - (Multi Label)":
        # Predict labels for the input features
        predicted_indices = model.predict(reshaped_features)
        print(predicted_indices)
        predicted_labels = []
        for i in range(0,len(predicted_indices[0])):
            if predicted_indices[0][i]==1.0:
                predicted_labels.append(xgb_multi_class_names[i])
        if predicted_labels:
            st.write(f"Predicted Genres: {', '.join(predicted_labels)}")
        else:
            st.write("No genres predicted for this input.")
    if model_name == "XGB Classifier - (Single Label)":
        predicted_indices = model.predict(reshaped_features)
        predicted_labels = [class_indices[i] for i in predicted_indices]
        st.write(f"Predicted Genre: {predicted_labels[0]}")
    elif model_name == "Convolutional Recurrent Neural Network - (Multi Label)"\
            or model_name == "Neural Network - (Multi Label)"\
            or model_name == "Batch Normalization - (Multi Label)":
        predicted_probabilities = model.predict(reshaped_features)

        # Set a threshold for class prediction (e.g., 0.5)
        threshold = 0.3
        print(predicted_probabilities)
        probabilities = []
        if model_name == "Convolutional Recurrent Neural Network - (Multi Label)":
            predicted_labels = [class_name for i, class_name in enumerate(multi_class_names) if
                                predicted_probabilities[0][i] >= threshold]
            probabilities = [(class_name,predicted_probabilities[0][i]*100) for i, class_name in enumerate(multi_class_names)]

        else:
            predicted_labels = [class_name for i,class_name in enumerate(xmulti_class_names) if
                                predicted_probabilities[0][i] >= threshold]
            probabilities = [(class_name,predicted_probabilities[0][i]*100) for i, class_name in enumerate(xmulti_class_names)]

        if predicted_labels:
            st.write(f"All probabilities are:")
            st.write(probabilities)
            st.write(f"Predicted Genres: {', '.join(predicted_labels)}")
        else:
            st.write("No genre predicted above the threshold.")
    else:
        predicted_label = model.predict(reshaped_features)[0]
        st.write(f"Predicted Genre: {predicted_label}")
