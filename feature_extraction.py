import librosa
import numpy as np
import joblib
import soundfile as sf
scaler = joblib.load("./models/std_scaler(1).pkl")


def load_audio_from_uploaded_file(uploaded_file):
    # Use the soundfile library to read the audio data and sample rate
    audio_data, sample_rate = sf.read(uploaded_file)

    return audio_data, sample_rate


# sample_audio,sr = librosa.load(r"classical.00000.wav",sr = 44100)
Fields = ['name', 'length', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
          'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var',
          'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
          'harmony_mean', 'harmony_var', 'percussive_mean', 'percussive_var', 'tempo',
          'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var',
          'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var',
          'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean',
          'mfcc12_var',
          'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean',
          'mfcc16_var',
          'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean',
          'mfcc20_var']

short_field = Fields[2:]


def all_feature_extraction(audio_path, sample_rate=22050):
    data_list = []
    val_field = []
    audio_df, sr = librosa.load(audio_path, sr=22050)
    data_list.append(audio_path)
    data_list.append(len(audio_df))

    # 1. Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=audio_df, hop_length=512)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)

    val_field.append(chroma_stft)
    data_list.append(chroma_stft_mean)
    data_list.append(chroma_stft_var)

    # 2. RMS
    rms = librosa.feature.rms(y=audio_df)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)
    data_list.append(rms_mean)
    data_list.append(rms_var)

    spectral_centroid = librosa.feature.spectral_centroid(y=audio_df)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_var = np.var(spectral_centroid)
    data_list.append(spectral_centroid_mean)
    data_list.append(spectral_centroid_var)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_df)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_var = np.var(spectral_bandwidth)
    data_list.append(spectral_bandwidth_mean)
    data_list.append(spectral_bandwidth_var)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_df)
    spectral_rolloff_mean = np.mean(spectral_rolloff)
    spectral_rolloff_var = np.var(spectral_rolloff)
    data_list.append(spectral_rolloff_mean)
    data_list.append(spectral_rolloff_var)

    zcr = librosa.feature.zero_crossing_rate(y=audio_df)
    zcr_mean = np.mean(zcr)
    zcr_var = np.var(zcr)
    data_list.append(zcr_mean)
    data_list.append(zcr_var)

    harmonic, percussive = librosa.effects.hpss(y=audio_df)
    harmonic_mean = np.mean(harmonic)
    harmonic_var = np.var(harmonic)
    percussive_mean = np.mean(percussive)
    percussive_var = np.var(percussive)
    data_list.append(harmonic_mean)
    data_list.append(harmonic_var)
    data_list.append(percussive_mean)
    data_list.append(percussive_var)

    tempo = librosa.feature.tempo(y=audio_df)
    tempo = np.mean(tempo)
    data_list.append(tempo)
    mfccs = librosa.feature.mfcc(y=audio_df, sr=sr)
    row_means = np.mean(mfccs, axis=1)
    row_vars = np.var(mfccs, axis=1)
    mfcc_means = {}
    mfcc_vars = {}
    for i in range(1, 21):
        variable_name = f'mfcc{i}'
        mfcc_means[variable_name] = row_means[i - 1]  # You can initialize with values if needed
        mfcc_vars[variable_name] = row_vars[i - 1]
    # Convert the dictionary values to a list
    mfcc_list = [value for value in zip(mfcc_means.values(), mfcc_vars.values())]

    for mean, var in mfcc_list:
        data_list.append(mean)
        data_list.append(var)

    return [data_list,val_field]

def scale(initial_features):
    final_features = initial_features[2:]
    final_features = np.array(final_features)
    # Apply the loaded scaler to your single data point
    scaled_data_point = scaler.transform([final_features])
    return scaled_data_point
