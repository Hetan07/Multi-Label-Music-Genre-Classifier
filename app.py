import joblib
import pandas as pd
import streamlit as st
import tensorflow
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

import audio_splitting
# Local Imports
import feature_extraction

st.set_page_config(layout="wide")


def display(model_name,col2):
    xgb_multi_class_names = ["Rock", "Rap & Hip-Hop", "Soul", "Classical", "Dance & Electronic", "Blues", "Jazz",
                             "Country", "Bebop", "Folk", "Reggae", "R&B", "Punk", "Metal", "Pop"]
    # if you are interested to know why there are two different lists at the same time, the order of genres in which
    # xgb was trained and others were trained are different
    xmulti_class_names = ["Metal", "Blues", "Reggae", "Jazz", "Rock", "Folk", "Classical", "Dance & Electronic",
                          "Punk", "Bebop", "Pop", "R&B", "Country", "Rap & Hip-Hop", "Soul"]

    if model_name == "XGB - (Multi Label)":
        predicted_indices = model.predict(reshaped_features)
        predicted_labels = []
        for i in range(0, len(predicted_indices[0])):
            if predicted_indices[0][i] == 1.0:
                predicted_labels.append(xgb_multi_class_names[i])
        if predicted_labels:
            labels = ', '.join(predicted_labels)
            with col2:
                st.metric(f"Predicted Genres: ",labels,label_visibility='collapsed')
        else:
            with col2:
                st.caption("No genres predicted for this input.")

    elif model_name == "XGB Classifier - (Single Label)":
        predicted_indices = model.predict(reshaped_features)
        predicted_labels = [class_indices[i] for i in predicted_indices]
        with col2:
            st.metric("Predicted Genre:", str(predicted_labels[0]), label_visibility="collapsed")

    elif model_name == "Convolutional Recurrent Neural Network - (Multi Label)" \
            or model_name == "Neural Network - (Multi Label)" \
            or model_name == "Convolutional Neural Network - (Multi Label)" \
            or model_name == "Batch Normalization - (Multi Label)":
        predicted_probabilities = model.predict(reshaped_features)
        threshold = 0.3
        print(predicted_probabilities)
        probabilities = []
        if model_name == "Convolutional Recurrent Neural Network - (Multi Label)":
            predicted_labels = [class_name for i, class_name in enumerate(multi_class_names) if
                                predicted_probabilities[0][i] >= threshold]
            probabilities = [(class_name, predicted_probabilities[0][i] * 100) for i, class_name in
                             enumerate(multi_class_names)]

        else:
            predicted_labels = [class_name for i, class_name in enumerate(xmulti_class_names) if
                                predicted_probabilities[0][i] >= threshold]
            probabilities = [(class_name, predicted_probabilities[0][i] * 100) for i, class_name in
                             enumerate(xmulti_class_names)]

        if predicted_labels:
            with col2:
                st.metric(f"Predicted Genres:",str(', '.join(predicted_labels)))
                st.caption("Below is a list of the probability of the sample being classified into each genre. Any "
                           "probability value below the threshold value (=0.35) is interpreted as the sample not being of "
                           "that genre ")
                df = pd.DataFrame(probabilities,columns=["Genre","Probabilities"])
                st.dataframe(df,hide_index=True,use_container_width=True)
        else:
            st.write("No genre predicted above the threshold.")

    else:
        predicted_label = model.predict(reshaped_features)[0]
        with col2:
            st.metric("Predicted Genres:", str(predicted_label).capitalize(), label_visibility="collapsed")


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

url_single_label = "https://huggingface.co/spaces/Hetan07/Single_Label_Music_Genre_Classifier"
url_github = "https://github.com/Hetan07/Multi-Label-Music-Genre-Classifier"
url_docs = "https://librosa.org/doc/latest/index.html"

st.title("Multi-Label Music Genre Classifier")
st.write("A multi-label music genre classifier based on the extension of my previous [project](%s). "
         "The source files have been provided both on HuggingFace and on [Github](%s). "
         "Dataset had to be created specifically, as none was available with the features and multi-labels tags for "
         "each audio."
         "All the models have been trained on the created dataset." % (url_single_label, url_github))

st.divider()
st.subheader('On Dataset Creation')

with st.expander("See explanation"):
    s = 'The work done for creating the dataset were\n' \
        '- Downloading the appropriate songs taken randomly from the MuMu dataset in sampled manner from ~80 genres (' \
        'tags)\n' \
        '- Data Cleaning which included to clean and replace the download songs as many of them were things such as album ' \
        'intros, interludes or skits\n' \
        '- There were also issues where the song required was not available on any platform and so had to appropriately ' \
        'replaced for another proper track or I had to manually search and download\n' \
        '- Each file had to properly checked to prevent any distortion or disturbances\n' \
        '- Applying feature extraction on each downloaded song using the librosa library\n' \
        '- Reducing the labels from ~80 to around ~15\n' \
        '\nIn the end I decided to have feature extraction work on 3 second samples and thus have around ~24000 samples. ' \
        'I have linked the actual dataset created from all the steps if anyone wishes to work upon it further\n'

    st.markdown(s)
st.divider()

st.subheader("Prediction of following genres")

multi_class_names = ["Bebop", "Blues", "Classical", "Country", "Dance & Electronic", "Folk", "Jazz", "Metal",
                     "Pop", "Punk", "R&B", "Rap & Hip-Hop", "Reggae", "Rock", "Soul"]

class_names = ["Blues", "Classical", "Country", "Disco", "HipHop",
               "Jazz", "Metal", "Pop", "Reggae", "Rock"]

class_indices = {i: class_name for i, class_name in enumerate(class_names)}

col1, col2 = st.columns(2)
s = ''
with col1:
    for i in multi_class_names[:7]:
        s += "- " + i + "\n"
    st.markdown(s)

s = ''
with col2:
    for i in multi_class_names[8:]:
        s += "- " + i + "\n"
    st.markdown(s)

st.divider()
# Upload music file
st.subheader("Upload a music file")
uploaded_file = st.file_uploader("Upload a music file", type=["mp3", "wav", "ogg"], label_visibility="collapsed")

st.divider()
if uploaded_file is not None:
    # User selects a model
    all_models = ["K-Nearest Neighbors - (Single Label)",
                  "Logistic Regression - (Single Label)",
                  "Support Vector Machines - (Single Label)",
                  "Neural Network - (Single Label)",
                  "XGB Classifier - (Single Label)",
                  "Convolutional Recurrent Neural Network - (Multi Label)",
                  "Convolutional Neural Network - (Multi Label)",
                  "XGB - (Multi Label)",
                  "Neural Network - (Multi Label)",
                  "Neural Network with Batch Normalization - (Multi Label)"]

    features_list, val_list = audio_splitting.split_audio(uploaded_file)
    features = feature_extraction.scale(features_list)

    feature_copy = features_list
    feature_copy.insert(19, "-")
    st.header("Feature Extraction")

    st.write("The given audio sample is processed using the librosa library to get the features extracted used by the "
             "models for genre prediction. Following is the dataframe with each of the feature extracted and "
             "corresponding mean and variance of the feature. The docs of [*librosa*](%s) library can be referred for a more "
             "indepth explanation and implementation specifics." % url_docs)

    col3, col4 = st.columns([0.55, 0.45])
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
    st.caption("Note: Harmonic and Percussion values generally have mean in the power of -1e5 or -1e6 and thus "
               "are represented as 0.\nAlso, for the feature 'tempo' variance has not been added to keep up with the "
               "consistency as presented in the original GTZAN dataset")

    st.divider()

    col1, col2 = st.columns([0.45, 0.55])

    col1.subheader("Select a model")
    with col1:
        model_name = st.selectbox("Select a model", all_models, label_visibility="collapsed")

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
        elif model_name == "Neural Network with Batch Normalization - (Multi Label)":
            model = tensorflow.keras.models.load_model("./models/model_bn.h5", compile=False)
            model.compile(loss=binary_crossentropy,
                          optimizer=Adam(),
                          metrics=['accuracy'])
        elif model_name == "Convolutional Neural Network - (Multi Label)":
            model = tensorflow.keras.models.load_model("./models/model_cnn.h5",compile=False)
            model.compile(loss=binary_crossentropy,
                          optimizer=Adam(),
                          metrics=['accuracy'])
    col2.subheader("Predicted genre")

    # Reshape the features to match the expected shape for prediction
    reshaped_features = features.reshape(1, -1)
    display(model_name,col2)
