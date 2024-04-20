# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/03/24 19:09:23
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0054: Research Project
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   None
"""

# here put the import lib

# here put the import lib
from pydub import AudioSegment

from transformers import AutoFeatureExtractor
import noisereduce as nr
import os
import random
import numpy as np
from matplotlib.colors import Normalize
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    silhouette_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
    auc,
)
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from models.speech.AlexNet import AlexNet
from models.speech.CNN import CNN
from models.speech.baselines import Baselines as SpeechBase
from models.speech.MLP import MLP
from models.speech.RNN import RNN
from models.speech.LSTM import LSTM
from models.speech.GMM import GMM
from models.speech.KMeans import KMeans
import numpy as np
from scipy.io import wavfile
import soundfile
import librosa

from models.speech.wav2vec import Wav2Vec


def get_padding(data, max_length):
    if max_length > data.shape[1]:
        pad_width = max_length - data.shape[1]
        data = np.pad(data, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        data = data[:, :max_length]

    return data


def get_features(
    dataset,
    method,
    filename,
    features="mfcc",
    n_mfcc=40,
    n_mels=128,
    max_length=109,
    window=None,
    sr=16000
):
    with soundfile.SoundFile(filename) as sound_file:
        X = sound_file.read(dtype="float32")  # 121715,2    # 62462  # 58124  # 45456,
        if dataset == "eNTERFACE" and len(X.shape) == 2:
            X = X[:, 1]
        sample_rate = sound_file.samplerate
        if sr != 16000:  # resample
            X = librosa.resample(X, orig_sr=sample_rate,target_sr=sr)

        if window != None:
            X = X[
                int(window[0] * sample_rate) : int(window[1] * sample_rate)
            ]  # 0,3 48000  # RAVDESS 59793

        result = np.array([])

        if method in ["CNN", "AlexNet"]:
            mfccs = librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=n_mfcc
            )  # 40,122 length=122
            print(mfccs.shape[1])
            mfccs = get_padding(mfccs, max_length)
            stft = np.abs(librosa.stft(X))  # spectrum
            chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
            chroma = get_padding(chroma, max_length)
            mel = librosa.feature.melspectrogram(
                y=X, sr=sample_rate, n_mels=n_mels, fmax=8000
            )  # 12,109
            mel = get_padding(mel, max_length)

        else:
            mfccs = np.mean(
                librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0
            )  # (40,121715)  # 40
            stft = np.abs(librosa.stft(X))  # spectrum
            chroma = np.mean(
                librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0
            )  # 12, 121715
            mel = np.mean(
                librosa.feature.melspectrogram(
                    y=X, sr=sample_rate, n_mels=n_mels, fmax=8000
                ).T,
                axis=0,
            )  # 128,121715

        if method in ["CNN", "AlexNet"]:
            if features == "mfcc":
                result = mfccs
            elif features == "chroma":
                result = chroma
            elif features == "mel":
                result = mel
            elif features == "all":
                result = np.vstack((mfccs, chroma))
                result = np.vstack((result, mel))
        else:
            if features == "mfcc":
                result = np.hstack((result, mfccs))  # 40,89954
            elif features == "chroma":
                result = np.hstack((result, chroma))
            elif features == "mel":
                result = np.hstack((result, mel))
            elif features == "all":
                result = np.hstack((result, mfccs))
                result = np.hstack((result, chroma))
                result = np.hstack((result, mel))
    sound_file.close()
    return result, X  # 40,


def print_features(data, features, n_mfcc, n_mels):
    if features in ["all", "mfcc"]:
        # Check chromagram feature values
        mfcc_features = np.array(data)[:, :n_mfcc]  # 1440,180  (2798,40)  (1440,40,150)
        mfcc_min = np.min(mfcc_features)
        mfcc_max = np.max(mfcc_features)
        # stack all features into a single series so we don't get a mean of means or stdev of stdevs
        mfcc_mean = mfcc_features.mean()
        mfcc_stdev = mfcc_features.std()
        print(
            f"{n_mfcc} MFCC features:       \
        min = {mfcc_min:.3f}, \
        max = {mfcc_max:.3f}, \
        mean = {mfcc_mean:.3f}, \
        deviation = {mfcc_stdev:.3f}"
        )

    if features in ["all", "chroma"]:
        # Check chroma feature values
        chroma_features = np.array(data)[:, n_mfcc : n_mfcc + 12]
        chroma_min = np.min(chroma_features)
        chroma_max = np.max(chroma_features)
        # stack all features into a single series so we don't get a mean of means or stdev of stdevs
        chroma_mean = chroma_features.mean()
        chroma_stdev = chroma_features.std()
        print(
            f"\n11 chroma features:             \
        min = {chroma_min:.3f},\
        max = {chroma_max:.3f},\
        mean = {chroma_mean:.3f},\
        deviation = {chroma_stdev:.3f}"
        )

    if features in ["all", "mel"]:
        # Check mel spectrogram feature values
        mel_features = np.array(data)[:, n_mfcc + 12 :]
        mel_min = np.min(mel_features)
        mel_max = np.max(mel_features)
        # stack all features into a single series so we don't get a mean of means or stdev of stdevs
        mel_mean = mel_features.mean()
        mel_stdev = mel_features.std()
        print(
            f"\n{n_mels} Mel Spectrogram features: \
        min = {mel_min:.3f}, \
        max = {mel_max:.3f}, \
        mean = {mel_mean:.3f}, \
        deviation = {mel_stdev:.3f}"
        )


def transform_feature(method, x, features, n_mfcc, n_mels, scaled):
    print_features(x, features, n_mfcc, n_mels)
    if method in ["AlexNet", "CNN"]:
        n, f, a = np.array(x).shape
        x = np.array(x).reshape(n, -1)
    if scaled == "minmax":
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
    elif scaled == "standard":
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    if method in ["AlexNet", "CNN"]:
        x = x.reshape(n, f, a)
    return x


def get_reverse(
    x,
    y,
    audio,
    lengths,
    path,
    dataset,
    method,
    features,
    n_mfcc,
    n_mels,
    max_length,
    sample,
    window,
    sr
):
    sample_index = random.sample([i for i in range(np.array(x).shape[0])], sample)
    for i in sample_index:
        sound = AudioSegment.from_file(path[i], format="wav")
        reversed_sound = sound.reverse()
        name = path[i].split(".")[0].split("/")[-1]
        if not os.path.exists(f"datasets/speech/{dataset}_reverse/"):
            os.makedirs(f"datasets/speech/{dataset}_reverse/")
        reversed_sound.export(
            f"datasets/speech/{dataset}_reverse/{name}_reverse.wav", format="wav"
        )
        feature, X = get_features(
            dataset,
            method,
            f"datasets/speech/{dataset}_reverse/{name}_reverse.wav",
            features,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            window=window,
            sr=sr
        )
        x.append(feature)
        y.append(y[i])
        audio.append(X)
        lengths.append(len(X))
    return x, y, audio, lengths


def get_noise(
    x,
    y,
    audio,
    lengths,
    path,
    dataset,
    method,
    features,
    n_mfcc,
    n_mels,
    max_length,
    sample,
    window,
    sr
):
    sample_index = random.sample([i for i in range(np.array(x).shape[0])], sample)
    for i in sample_index:
        with soundfile.SoundFile(path[i]) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate

        random_values = (
            np.random.rand(len(X))
            if dataset != "eNTERFACE"
            else np.random.rand(len(X), 2)
        )
        X = X + 2e-2 * random_values
        name = path[i].split(".")[0].split("/")[-1]
        if not os.path.exists(f"datasets/speech/{dataset}_noise/"):
            os.makedirs(f"datasets/speech/{dataset}_noise/")
        soundfile.write(
            f"datasets/speech/{dataset}_noise/{name}_noise.wav", X, sample_rate
        )

        feature, X = get_features(
            dataset,
            method,
            f"datasets/speech/{dataset}_noise/{name}_noise.wav",
            features,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            window=window,
            sr=sr
        )
        x.append(feature)
        y.append(y[i])
        audio.append(X)
        lengths.append(len(X))
    return x, y, audio, lengths


def get_denoise(
    x,
    y,
    audio,
    lengths,
    emotion_map,
    dataset,
    method,
    features,
    n_mfcc,
    n_mels,
    max_length,
    window,
    sr
):
    # sample_index = random.sample([i for i in range(np.array(x).shape[0])], sample)
    # for i in sample_index:
    for i in os.listdir(f"datasets/speech/{dataset}_noise/"):
        # audio = AudioSegment.from_file(path[i], format="wav")
        au = AudioSegment.from_file(
            os.path.join(f"datasets/speech/{dataset}_noise/", i), format="wav"
        )
        samples = np.array(au.get_array_of_samples())
        reduced_noise = nr.reduce_noise(samples, sr=au.frame_rate)
        reduced_audio = AudioSegment(
            reduced_noise.tobytes(),
            frame_rate=au.frame_rate,
            sample_width=au.sample_width,
            channels=au.channels,
        )
        # name = path[i].split(".")[0].split("/")[-1]
        name = i.split(".")[0][:-6]
        if not os.path.exists(f"datasets/speech/{dataset}_denoise/"):
            os.makedirs(f"datasets/speech/{dataset}_denoise/")
        reduced_audio.export(
            f"datasets/speech/{dataset}_denoise/{name}_denoise.wav", format="wav"
        )

        feature, X = get_features(
            dataset,
            method,
            f"datasets/speech/{dataset}_denoise/{name}_denoise.wav",
            features,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            window=window,
            sr=sr
        )

        x.append(feature)
        if dataset == "RAVDESS":
            emotion = emotion_map[name.split("-")[2]]
        elif dataset == "TESS":
            emotion = emotion_map[name.split("_")[-1].split(".")[0].lower()]
        elif dataset == "SAVEE":
            emotion = emotion_map[name.split(".")[0].split("_")[-1][:-2]]
        elif dataset == "CREMA":
            emotion = emotion_map[name.split("_")[2].lower()]
        elif dataset == "EmoDB":
            emotion = emotion_map[name.split(".")[0][-2]]
        else:
            if name[1] == "_":
                label = name.split(".")[0].split("_")[-2]
            else:
                label = name.split(".")[0].split("_")[1]
            emotion = emotion_map[label]
        y.append(emotion)
        audio.append(X)
        lengths.append(len(X))
    return x, y, audio, lengths


def load_RAVDESS(
    method,
    features,
    n_mfcc,
    n_mels,
    scaled,
    max_length,
    reverse,
    noise,
    denoise,
    window=None,
    sr=16000
):
    x, y, category, path, audio, lengths = [], [], [], [], [], []
    emotion_map = {
        "01": 0,  # 'neutral'
        "02": 1,  # 'calm'
        "03": 2,  # 'happy'
        "04": 3,  # 'sad'
        "05": 4,  # 'angry'
        "06": 5,  # 'fearful'
        "07": 6,  # 'disgust'
        "08": 7,  # 'surprised'
    }

    category_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }
    for file in glob.glob("datasets/speech/RAVDESS/Actor_*/*.wav"):
        # print(file)
        file_name = os.path.basename(file)
        emotion = emotion_map[file_name.split("-")[2]]
        feature, X = get_features(
            "RAVDESS",
            method,
            file,
            features,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            window=window,
            sr=sr
        )
        x.append(feature)
        y.append(emotion)
        path.append(file)
        category.append(category_map[file_name.split("-")[2]])
        audio.append(X)
        lengths.append(len(X))
        if category.count(category_map[file_name.split("-")[2]]) == 1:
            visual4feature(file, "RAVDESS", category_map[file_name.split("-")[2]])

    visual4label("speech", "RAVDESS", category)
    print(np.array(x).shape)  # (864,40), (288,40), (288,40)

    if method != "wav2vec":
        if scaled != None:
            x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

    if reverse == True:
        x, y, audio, lengths = get_reverse(
            x,
            y,
            audio,
            lengths,
            path,
            "RAVDESS",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
            sr
        )

    if noise == True:
        x, y, audio, lengths = get_noise(
            x,
            y,
            audio,
            lengths,
            path,
            "RAVDESS",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
            sr
        )

    if denoise == True:
        x, y, audio, lengths = get_denoise(
            x,
            y,
            audio,
            lengths,
            emotion_map,
            "RAVDESS",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            window,
            sr
        )

    length = None if method != "wav2vec" else max(lengths)

    if method != "wav2vec":
        X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
            np.array(x), y, test_size=0.4, random_state=9
        )  # 3:2
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base", return_attention_mask=True
        )
        X = feature_extractor(
            audio,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=length,
            truncation=True,
            padding=True,
        )
        X_train, X_left, ytrain, yleft = train_test_split(  # 54988
            np.array(X["input_values"]), y, test_size=0.4, random_state=9
        )  # 3:2

    X_val, X_test, yval, ytest = train_test_split(
        X_left, yleft, test_size=0.5, random_state=9
    )  # 1:1
    # (1680, 40), (560, 40), (560, 40), (1680,)
    return X_train, ytrain, X_val, yval, X_test, ytest, length
    # 864, 84351; 864


def load_TESS(
    method,
    features,
    n_mfcc,
    n_mels,
    scaled,
    max_length,
    reverse,
    noise,
    denoise,
    window=None,
    sr=16000
):
    x, y, category, path, audio, lengths = [], [], [], [], [], []
    emotion_map = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "neutral": 4,
        "ps": 5,
        "sad": 6,
    }
    for dirname, _, filenames in os.walk("datasets/speech/TESS"):
        for filename in filenames:
            feature, X = get_features(
                "TESS",
                method,
                os.path.join(dirname, filename),
                features,
                n_mfcc=n_mfcc,
                n_mels=n_mels,
                max_length=max_length,
                window=window,
                sr=sr
            )
            label = filename.split("_")[-1].split(".")[0]
            emotion = emotion_map[label.lower()]
            x.append(feature)
            y.append(emotion)
            path.append(os.path.join(dirname, filename))
            category.append(label.lower())
            audio.append(X)
            lengths.append(len(X))
            if category.count(label.lower()) == 1:
                visual4feature(os.path.join(dirname, filename), "TESS", label.lower())

        if len(y) == 2800:
            break

    visual4label("speech", "TESS", category)

    if method != "wav2vec":
        if scaled != None:
            x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

    if reverse == True:
        x, y, audio, lengths = get_reverse(
            x,
            y,
            audio,
            lengths,
            path,
            "TESS",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            500,
            window,
            sr
        )

    if noise == True:
        x, y, audio, lengths = get_noise(
            x,
            y,
            audio,
            lengths,
            path,
            "TESS",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
            sr
        )

    if denoise == True:
        x, y, audio, lengths = get_denoise(
            x,
            y,
            audio,
            lengths,
            emotion_map,
            "TESS",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            window,
            sr
        )

    length = None if method != "wav2vec" else max(lengths)

    if method != "wav2vec":
        X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
            np.array(x), y, test_size=0.4, random_state=9
        )  # 3:2
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base", return_attention_mask=True
        )
        X = feature_extractor(
            audio,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=length,
            truncation=True,
            padding=True,
        )  # (1440, 84351)
        X_train, X_left, ytrain, yleft = train_test_split(
            np.array(X["input_values"]), y, test_size=0.4, random_state=9
        )  # 3:2

    X_val, X_test, yval, ytest = train_test_split(
        X_left, yleft, test_size=0.5, random_state=9
    )  # 1:1
    # (1680, 40), (560, 40), (560, 40), (1680,)
    return X_train, ytrain, X_val, yval, X_test, ytest, length


def load_SAVEE(
    method,
    features,
    n_mfcc,
    n_mels,
    scaled,
    max_length,
    reverse,
    noise,
    denoise,
    window=None,
    sr=16000
):
    x, y, category, paths, audio, lengths = [], [], [], [], [], []
    emotion_map = {
        "a": 0,  # angry
        "d": 1,  # digust
        "f": 2,  # fear
        "h": 3,  # happiness
        "n": 4,  # neutral
        "sa": 5,  # sadness
        "su": 6,  # surprise
    }
    category_map = {
        "a": "angry",  # angry
        "d": "disgust",
        "f": "fear",
        "h": "happiness",
        "n": "neutral",
        "sa": "sadness",
        "su": "surprise",
    }
    path = "datasets/speech/SAVEE"
    for file in os.listdir(path):
        feature, X = get_features(
            "SAVEE",
            method,
            os.path.join(path, file),
            features,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            window=window,
            sr=sr
        )
        label = file.split(".")[0].split("_")[-1][:-2]
        emotion = emotion_map[label]
        x.append(feature)
        y.append(emotion)
        paths.append(os.path.join(path, file))
        category.append(category_map[label])
        audio.append(X)
        lengths.append(len(X))
        if category.count(category_map[label]) == 1:
            visual4feature(os.path.join(path, file), "SAVEE", category_map[label])

    visual4label("speech", "SAVEE", category)

    if method != "wav2vec":
        if scaled != None:
            x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

    if reverse == True:
        x, y, audio, lengths = get_reverse(
            x,
            y,
            audio,
            lengths,
            paths,
            "SAVEE",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
            sr
        )  # sample need to evaluate

    if noise == True:
        x, y, audio, lengths = get_noise(
            x,
            y,
            audio,
            lengths,
            paths,
            "SAVEE",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
            sr
        )

    if denoise == True:
        x, y, audio, lengths = get_denoise(
            x,
            y,
            audio,
            lengths,
            emotion_map,
            "SAVEE",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            window,
            sr
        )

    length = None if method != "wav2vec" else max(lengths)

    if method != "wav2vec":
        X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
            np.array(x), y, test_size=0.4, random_state=9
        )  # 3:2
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base", return_attention_mask=True
        )
        X = feature_extractor(
            audio,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=length,
            truncation=True,
            padding=True,
        )
        X_train, X_left, ytrain, yleft = train_test_split(
            np.array(X["input_values"]), y, test_size=0.4, random_state=9
        )  # 3:2

    X_val, X_test, yval, ytest = train_test_split(
        X_left, yleft, test_size=0.5, random_state=9
    )  # 1:1
    # (1680, 40), (560, 40), (560, 40), (1680,)
    return X_train, ytrain, X_val, yval, X_test, ytest, length


def load_CREMA(
    method,
    features,
    n_mfcc,
    n_mels,
    scaled,
    max_length,
    reverse,
    noise,
    denoise,
    window=None,
    sr=16000
):
    x, y, category, paths, audio, lengths = [], [], [], [], [], []
    emotion_map = {
        "ang": 0,  # angry
        "dis": 1,  # disgust
        "fea": 2,  # fear
        "hap": 3,  # happiness
        "neu": 4,  # neutral
        "sad": 5,  # sadness
    }
    category_map = {
        "ang": "angry",
        "dis": "disgust",
        "fea": "fear",
        "hap": "happiness",
        "neu": "neutral",
        "sad": "sadness",
    }
    path = "datasets/speech/CREMA-D"
    for file in os.listdir(path):
        feature, X = get_features(
            "CREMA",
            method,
            os.path.join(path, file),
            features,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            window=window,
            sr=sr
        )
        label = file.split("_")[2]
        emotion = emotion_map[label.lower()]
        x.append(feature)
        y.append(emotion)
        paths.append(os.path.join(path, file))
        category.append(category_map[label.lower()])
        audio.append(X)  # list of array
        lengths.append(len(X))
        if category.count(category_map[label.lower()]) == 1:
            visual4feature(
                os.path.join(path, file), "CREMA-D", category_map[label.lower()]
            )

    visual4label("speech", "CREMA", category)

    if method != "wav2vec":
        if scaled != None:
            x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

    if reverse == True:
        x, y, audio, lengths = get_reverse(
            x,
            y,
            audio,
            lengths,
            paths,
            "CREMA",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            500,
            window,
            sr
        )  # sample need to evaluate

    if noise == True:
        x, y, audio, lengths = get_noise(
            x,
            y,
            audio,
            lengths,
            paths,
            "CREMA",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
            sr
        )

    if denoise == True:
        x, y, audio, lengths = get_denoise(
            x,
            y,
            audio,
            lengths,
            emotion_map,
            "CREMA",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            window,
            sr
        )

    length = None if method != "wav2vec" else max(lengths)

    if method != "wav2vec":
        X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
            np.array(x), y, test_size=0.4, random_state=9
        )  # 3:2
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base", return_attention_mask=True
        )
        X = feature_extractor(
            audio,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=length,
            truncation=True,
            padding=True,
        )
        X_train, X_left, ytrain, yleft = train_test_split(
            np.array(X["input_values"]), y, test_size=0.4, random_state=9
        )  # 3:2

    X_val, X_test, yval, ytest = train_test_split(
        X_left, yleft, test_size=0.5, random_state=9
    )  # 1:1
    # (4465, 40), (1488, 40), (1489, 40), (1680,)
    return X_train, ytrain, X_val, yval, X_test, ytest, length


def load_EmoDB(
    method,
    features,
    n_mfcc,
    n_mels,
    scaled,
    max_length,
    reverse,
    noise,
    denoise,
    window=None,
    sr=16000
):
    x, y, category, paths, audio, lengths = [], [], [], [], [], []
    emotion_map = {
        "W": 0,  # angry
        "L": 1,  # boredom
        "E": 2,  # disgust
        "A": 3,  # anxiety/fear
        "F": 4,  # happiness
        "T": 5,  # sadness
        "N": 6,
    }
    category_map = {
        "W": "angry",
        "L": "boredom",
        "E": "disgust",
        "A": "fear",  # fear
        "F": "happiness",
        "T": "sadness",
        "N": "neutral",
    }
    path = "datasets/speech/EmoDB"
    for file in os.listdir(path):
        feature, X = get_features(
            "EmoDB",
            method,
            os.path.join(path, file),
            features,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            window=window,
            sr=sr
        )
        label = file.split(".")[0][-2]
        emotion = emotion_map[label]
        x.append(feature)
        y.append(emotion)
        paths.append(os.path.join(path, file))
        category.append(category_map[label])
        audio.append(X)
        lengths.append(len(X))
        if category.count(category_map[label]) == 1:
            visual4feature(os.path.join(path, file), "EmoDB", category_map[label])

    visual4label("speech", "EmoDB", category)

    if method != "wav2vec":
        if scaled != None:
            x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

    if reverse == True:
        x, y, audio, lengths = get_reverse(
            x,
            y,
            audio,
            lengths,
            paths,
            "EmoDB",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            500,
            window,
            sr
        )  # sample need to evaluate

    if noise == True:
        x, y, audio, lengths = get_noise(
            x,
            y,
            audio,
            lengths,
            paths,
            "EmoDB",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
            sr
        )

    if denoise == True:
        x, y, audio, lengths = get_denoise(
            x,
            y,
            audio,
            lengths,
            emotion_map,
            "EmoDB",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            window,
            sr
        )

    length = None if method != "wav2vec" else max(lengths)

    if method != "wav2vec":
        X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
            np.array(x), y, test_size=0.4, random_state=9
        )  # 3:2
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base", return_attention_mask=True
        )
        X = feature_extractor(
            audio,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=length,
            truncation=True,
            padding=True,
        )
        X_train, X_left, ytrain, yleft = train_test_split(
            np.array(X["input_values"]), y, test_size=0.4, random_state=9
        )  # 3:2

    X_val, X_test, yval, ytest = train_test_split(
        X_left, yleft, test_size=0.5, random_state=9
    )  # 1:1
    # (1680, 40), (560, 40), (560, 40), (1680,)
    return X_train, ytrain, X_val, yval, X_test, ytest, length


def load_eNTERFACE(
    method,
    features,
    n_mfcc,
    n_mels,
    scaled,
    max_length,
    reverse,
    noise,
    denoise,
    window=None,
    sr=16000
):
    x, y, category, paths, audio, lengths = [], [], [], [], [], []
    emotion_map = {
        "an": 0,  # angry
        "di": 1,  # disgust
        "fe": 2,  # fear
        "ha": 3,  # happiness
        "sa": 4,  # sadness
        "su": 5,  # surprise
    }

    category_map = {
        "an": "angry",
        "di": "disgust",
        "fe": "fear",
        "ha": "happiness",
        "sa": "sadness",
        "su": "surprise",
    }
    path = "datasets/speech/eNTERFACE05"
    for file in os.listdir(path):
        feature, X = get_features(
            "eNTERFACE",
            method,
            os.path.join(path, file),
            features,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            window=window,
            sr=sr
        )
        if file[1] == "_":
            label = file.split(".")[0].split("_")[-2]
        else:
            label = file.split(".")[0].split("_")[1]
        emotion = emotion_map[label]
        x.append(feature)
        y.append(emotion)
        paths.append(os.path.join(path, file))
        category.append(category_map[label])
        audio.append(X)
        lengths.append(len(X))
        if category.count(category_map[label]) == 1:
            visual4feature(os.path.join(path, file), "eNTERFACE", category_map[label])

    visual4label("speech", "eNTERFACE05", category)

    if method != "wav2vec":
        if scaled != None:
            x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

    if reverse == True:
        x, y, audio, lengths = get_reverse(
            x,
            y,
            audio,
            lengths,
            paths,
            "eNTERFACE",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            500,
            window,
            sr
        )  # sample need to evaluate

    if noise == True:
        x, y, audio, lengths = get_noise(
            x,
            y,
            audio,
            lengths,
            paths,
            "eNTERFACE",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
            sr
        )

    if denoise == True:
        x, y, audio, lengths = get_denoise(
            x,
            y,
            audio,
            lengths,
            emotion_map,
            "eNTERFACE",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            window,
            sr
        )

    length = None if method != "wav2vec" else max(lengths)

    if method != "wav2vec":
        X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
            np.array(x), y, test_size=0.4, random_state=9
        )  # 3:2
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base", return_attention_mask=True
        )
        X = feature_extractor(
            audio,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=length,
            truncation=True,
            padding=True,
        )

        X_train, X_left, ytrain, yleft = train_test_split(
            np.array(X), y, test_size=0.4, random_state=9
        )  # 3:2

    X_val, X_test, yval, ytest = train_test_split(
        X_left, yleft, test_size=0.5, random_state=9
    )  # 1:1
    # (1680, 40), (560, 40), (560, 40), (1680,)
    return X_train, ytrain, X_val, yval, X_test, ytest, length


def load_cross_corpus(
    method,
    features,
    n_mfcc,
    n_mels,
    scaled,
    max_length,
    reverse,
    noise,
    denoise,
    window=None,
    corpus=None,
    sr=16000
):  # ["RAVDESS", "TESS"]

    lengths = []
    (
        X_train,
        ytrain,
        X_val,
        yval,
        X_test,
        ytest,
        y_train_corpus,
        y_test_corpus,
        train_corpus_audio,
        test_corpus_audio,
    ) = (None, None, None, None, None, None, None, None, None, None)
    emotion_map = {
        ["01", "02", "neutral", "n", "neu", "L", "N"]: 1,  # neutral
        ["03", "08", "happy", "ps", "h", "su", "hap", "F", "ha", "su"]: 2,  # positive
        [
            "04",
            "05",
            "06",
            "07",
            "angry",
            "disgust",
            "fear",
            "sad",
            "a",
            "d",
            "f",
            "sa",
            "ang",
            "dis",
            "fea",
            "sad",
            "W",
            "E",
            "A",
            "T",
            "an",
            "di",
            "fe",
            "sa",
        ]: 3,
    }
    for index, cor in enumerate(corpus):
        # index=0 train, index=1 test
        x, y, paths, audio = [], [], [], []
        if cor == "RAVDESS":
            for file in glob.glob("datasets/speech/RAVDESS/Actor_*/*.wav"):
                # print(file)
                file_name = os.path.basename(file)
                for k, i in enumerate(emotion_map.keys()):
                    if file_name.split("-")[2] in i:
                        emotion = emotion_map[i]
                feature, X = get_features(
                    cor,
                    method,
                    file,
                    features,
                    n_mfcc=n_mfcc,
                    n_mels=n_mels,
                    max_length=max_length,
                    window=window,
                    sr=sr
                )
                x.append(feature)
                y.append(emotion)
                paths.append(file)
                audio.append(X)
                lengths.append(len(X))
        elif cor == "TESS":
            for dirname, _, filenames in os.walk("datasets/speech/TESS"):
                for filename in filenames:
                    feature, X = get_features(
                        cor,
                        method,
                        os.path.join(dirname, filename),
                        features,
                        n_mfcc=n_mfcc,
                        n_mels=n_mels,
                        max_length=max_length,
                        window=window,
                        sr=sr
                    )
                    label = filename.split("_")[-1].split(".")[0]
                    for k, i in enumerate(emotion_map.keys()):
                        if file_name.split("-")[2] in i:
                            emotion = emotion_map[i]
                    x.append(feature)
                    y.append(emotion)
                    paths.append(os.path.join(dirname, filename))
                    audio.append(X)
                    lengths.append(len(X))
        else:
            path = f"datasets/speech/{cor}"
            for file in os.listdir(path):
                feature, X = get_features(
                    cor,
                    method,
                    os.path.join(path, file),
                    features,
                    n_mfcc=n_mfcc,
                    n_mels=n_mels,
                    max_length=max_length,
                    window=window,
                    sr=sr
                )
                label = file.split(".")[0].split("_")[-1][:-2]
                for k, i in enumerate(emotion_map.keys()):
                    if label in i:
                        emotion = emotion_map[i]
                x.append(feature)
                y.append(emotion)
                paths.append(os.path.join(path, file))
                audio.append(X)
                lengths.append(len(X))

        if method != "wav2vec":
            if scaled != None:
                x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

        if reverse == True:
            x, y, audio, lengths = get_reverse(
                x,
                y,
                audio,
                lengths,
                paths,
                cor,
                method,
                features,
                n_mfcc,
                n_mels,
                max_length,
                100,
                window,
                sr
            )

        if noise == True:
            x, y, audio, lengths = get_noise(
                x,
                y,
                audio,
                lengths,
                paths,
                cor,
                method,
                features,
                n_mfcc,
                n_mels,
                max_length,
                100,
                window,
                sr
            )

        if denoise == True:
            x, y, audio, lengths = get_denoise(
                x,
                y,
                audio,
                lengths,
                emotion_map,
                cor,
                method,
                features,
                n_mfcc,
                n_mels,
                max_length,
                window,
                sr
            )

        new_label = []
        for i in y:
            if i not in [1, 2, 3]:
                new_label.append(i)
            else:
                for k, j in enumerate(emotion_map.keys()):
                    if label in j:
                        emotion = emotion_map[j]
                new_label.append(emotion)
        y = new_label

        if method != "wav2vec":
            if index == 0:  # train corpus
                # split into train and val
                sample_index = random.sample(
                    [i for i in range(np.array(x).shape[0])], 1200
                )
                X_train, X_val, ytrain, yval = train_test_split(  # 2800, 1680, 1120
                    np.array(x)[sample_index, :],
                    np.array(y)[sample_index, :].tolist(),
                    test_size=0.25,
                    random_state=9,
                )  # 3:1
            elif index == 1:
                sample_index = random.sample(
                    [i for i in range(np.array(x).shape[0])], 300
                )
                X_test = np.array(x)[sample_index, :]
                ytest = np.array(y)[sample_index, :].tolist()
        else:
            if index == 0:
                train_corpus_audio = audio
                y_train_corpus = y
            else:
                test_corpus_audio = audio
                y_test_corpus = y

    length = None
    if method == "wav2vec":
        length = max(lengths)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base", return_attention_mask=True
        )
        X_train_corpus = feature_extractor(
            train_corpus_audio,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=length,
            truncation=True,
            padding=True,
        )
        sample_index = random.sample([i for i in range(np.array(x).shape[0])], 1200)
        X_train, X_val, ytrain, yval = train_test_split(  # 2800, 1680, 1120
            np.array(X_train_corpus["input_values"])[sample_index, :],
            y_train_corpus,
            test_size=0.25,
            random_state=9,
        )  # 3:1

        X_test_corpus = feature_extractor(
            test_corpus_audio,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=length,
            truncation=True,
            padding=True,
        )

        sample_index = random.sample([i for i in range(np.array(x).shape[0])], 300)
        X_test = np.array(X_test_corpus["input_values"])[sample_index, :]
        ytest = np.array(y_test_corpus)[sample_index, :].tolist()

    return X_train, ytrain, X_val, yval, X_test, ytest, length
    # 900, 300, 300, train_corpus 1200, test_corpus 300


def load_mix_corpus(
    method,
    features,
    n_mfcc,
    n_mels,
    scaled,
    max_length,
    reverse,
    noise,
    denoise,
    window=None,
    corpus=None,
    sr=16000
):  # cc: mix, corpus: with only one string as the testing set

    # 900+300, 900 train 300 val, 300 test
    # 900/5=180, 300/5=60, 1200/5=240
    # 300
    datasets = ["RAVDESS", "TESS", "SAVEE", "CREMA-D", "EmoDB"]
    emotion_map = {
        ("01", "02", "neutral", "n", "neu", "L", "N"): 1,  # neutral
        ("03", "08", "happy", "ps", "h", "su", "hap", "F", "ha", "su"): 2,  # positive
        (
            "04",
            "05",
            "06",
            "07",
            "angry",
            "disgust",
            "fear",
            "sad",
            "a",
            "d",
            "f",
            "sa",
            "ang",
            "dis",
            "fea",
            "sad",
            "W",
            "E",
            "A",
            "T",
            "an",
            "di",
            "fe",
            "sa",
        ): 3,  # tuple can be hashed but list cannot
    }
    lengths, paths, audio, X_train, ytrain, X_val, yval, X_test, ytest = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for index, dataset in enumerate(datasets):
        x, y = [], []  # for each dataset
        if dataset == "RAVDESS":
            for file in glob.glob("datasets/speech/RAVDESS/Actor_*/*.wav"):
                # print(file)
                file_name = os.path.basename(file)
                for k, i in enumerate(emotion_map.keys()):
                    if file_name.split("-")[2] in i:  # tuple
                        emotion = emotion_map[i]
                feature, X = get_features(
                    dataset,
                    method,
                    file,
                    features,
                    n_mfcc=n_mfcc,
                    n_mels=n_mels,
                    max_length=max_length,
                    window=window,
                    sr=sr
                )
                x.append(feature)
                y.append(emotion)
                paths.append(file)
                audio.append(X)
                lengths.append(len(X))
        elif dataset == "TESS":
            for dirname, _, filenames in os.walk("datasets/speech/TESS"):
                for filename in filenames:
                    feature, X = get_features(
                        dataset,
                        method,
                        os.path.join(dirname, filename),
                        features,
                        n_mfcc=n_mfcc,
                        n_mels=n_mels,
                        max_length=max_length,
                        window=window,
                        sr=sr
                    )
                    label = filename.split("_")[-1].split(".")[0]
                    for k, i in enumerate(emotion_map.keys()):
                        if file_name.split("-")[2] in i:
                            emotion = emotion_map[i]
                    x.append(feature)
                    y.append(emotion)
                    paths.append(os.path.join(dirname, filename))
                    audio.append(X)
                    lengths.append(len(X))
        else:
            path = f"datasets/speech/{dataset}"
            for file in os.listdir(path):
                feature, X = get_features(
                    dataset,
                    method,
                    os.path.join(path, file),
                    features,
                    n_mfcc=n_mfcc,
                    n_mels=n_mels,
                    max_length=max_length,
                    window=window,
                    sr=sr
                )
                label = file.split(".")[0].split("_")[-1][:-2]
                for k, i in enumerate(emotion_map.keys()):
                    if label in i:
                        emotion = emotion_map[i]
                x.append(feature)
                y.append(emotion)
                paths.append(os.path.join(path, file))
                audio.append(X)
                lengths.append(len(X))

        # first reverse for each dataset, then get
        sample_index = random.sample(
            [i for i in range(np.array(x).shape[0])], 240
        )  # 1440, 40
        # 240, 40
        X_train = (
            np.array(x)[sample_index, :]
            if len(X_train) == 0
            else np.concatenate((X_train, np.array(x)[sample_index, :]), axis=0)
        )
        ytrain = (
            np.array(y)[sample_index].tolist()
            if len(ytrain) == 0
            else (ytrain + np.array(y)[sample_index].tolist())
        )

        if method != "wav2vec":
            if scaled != None:
                X_train = transform_feature(
                    method, X_train, features, n_mfcc, n_mels, scaled
                )

        if reverse == True:
            X_train, ytrain, audio, lengths = get_reverse(
                X_train,
                ytrain,
                audio,
                lengths,
                paths,
                "mix",
                method,
                features,
                n_mfcc,
                n_mels,
                max_length,
                100,
                window,
                sr
            )

        if noise == True:
            X_train, ytrain, audio, lengths = get_noise(
                X_train,
                ytrain,
                audio,
                lengths,
                paths,
                "mix",
                method,
                features,
                n_mfcc,
                n_mels,
                max_length,
                100,
                window,
                sr
            )

        if denoise == True:
            X_train, ytrain, audio, lengths = get_denoise(
                X_train,
                ytrain,
                audio,
                lengths,
                emotion_map,
                "mix",
                method,
                features,
                n_mfcc,
                n_mels,
                max_length,
                window,
                sr
            )

        if dataset == corpus[0]:  # testing set
            left_index = [
                i for i in range(np.array(x).shape[0]) if i not in sample_index
            ]
            test_index = random.sample(left_index, 300)
            X_test = np.array(x)[test_index, :]

    # after traversing all dataset
    length = None
    if method == "wav2vec":
        length = max(lengths)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base", return_attention_mask=True
        )
        X_train = feature_extractor(
            X_train,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=length,
            truncation=True,
            padding=True,
        )
        X_test = feature_extractor(
            X_test,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=length,
            truncation=True,
            padding=True,
        )
        X_test = np.array(X_test["input_values"])

    X_train, X_val, ytrain, yval = train_test_split(  # 2800, 1680, 1120
        X_train, ytrain, test_size=0.25, random_state=9
    )  # 3:1

    return X_train, ytrain, X_val, yval, X_test, ytest, length


def load_finetune_corpus(  # train with one, finetune with the other, test with the other
    method,
    features,
    n_mfcc,
    n_mels,
    scaled,
    max_length,
    reverse,
    noise,
    denoise,
    window=None,
    corpus=None,
    sr=16000
):  # ["RAVDESS", "TESS"]  train, finetune/test

    lengths = []
    (
        X_train,
        ytrain,
        X_val,
        yval,
        X_test,
        ytest,
        Xtune_train,
        ytune_train,
        Xtune_val,
        ytune_val,
        y_train_corpus,
        y_test_corpus,
        train_corpus_audio,
        finetune_corpus_audio,
    ) = (None, None, None, None, None, None, None, None, None, None)
    emotion_map = {
        ["01", "02", "neutral", "n", "neu", "L", "N"]: 1,  # neutral
        ["03", "08", "happy", "ps", "h", "su", "hap", "F", "ha", "su"]: 2,  # positive
        [
            "04",
            "05",
            "06",
            "07",
            "angry",
            "disgust",
            "fear",
            "sad",
            "a",
            "d",
            "f",
            "sa",
            "ang",
            "dis",
            "fea",
            "sad",
            "W",
            "E",
            "A",
            "T",
            "an",
            "di",
            "fe",
            "sa",
        ]: 3,
    }
    for index, cor in enumerate(corpus):
        # index=0 train, index=1 test
        x, y, paths, audio = [], [], [], []
        if cor == "RAVDESS":
            for file in glob.glob("datasets/speech/RAVDESS/Actor_*/*.wav"):
                # print(file)
                file_name = os.path.basename(file)
                for k, i in enumerate(emotion_map.keys()):
                    if file_name.split("-")[2] in i:
                        emotion = emotion_map[i]
                feature, X = get_features(
                    cor,
                    method,
                    file,
                    features,
                    n_mfcc=n_mfcc,
                    n_mels=n_mels,
                    max_length=max_length,
                    window=window,
                    sr=sr
                )
                x.append(feature)
                y.append(emotion)
                paths.append(file)
                audio.append(X)
                lengths.append(len(X))
        elif cor == "TESS":
            for dirname, _, filenames in os.walk("datasets/speech/TESS"):
                for filename in filenames:
                    feature, X = get_features(
                        cor,
                        method,
                        os.path.join(dirname, filename),
                        features,
                        n_mfcc=n_mfcc,
                        n_mels=n_mels,
                        max_length=max_length,
                        window=window,
                        sr=sr
                    )
                    label = filename.split("_")[-1].split(".")[0]
                    for k, i in enumerate(emotion_map.keys()):
                        if file_name.split("-")[2] in i:
                            emotion = emotion_map[i]
                    x.append(feature)
                    y.append(emotion)
                    paths.append(os.path.join(dirname, filename))
                    audio.append(X)
                    lengths.append(len(X))
        else:
            path = f"datasets/speech/{cor}"
            for file in os.listdir(path):
                feature, X = get_features(
                    cor,
                    method,
                    os.path.join(path, file),
                    features,
                    n_mfcc=n_mfcc,
                    n_mels=n_mels,
                    max_length=max_length,
                    window=window,
                    sr=sr
                )
                label = file.split(".")[0].split("_")[-1][:-2]
                for k, i in enumerate(emotion_map.keys()):
                    if label in i:
                        emotion = emotion_map[i]
                x.append(feature)
                y.append(emotion)
                paths.append(os.path.join(path, file))
                audio.append(X)
                lengths.append(len(X))

        if method != "wav2vec":
            if scaled != None:
                x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

        if reverse == True:
            x, y, audio, lengths = get_reverse(
                x,
                y,
                audio,
                lengths,
                paths,
                cor,
                method,
                features,
                n_mfcc,
                n_mels,
                max_length,
                100,
                window,
                sr
            )

        if noise == True:
            x, y, audio, lengths = get_noise(
                x,
                y,
                audio,
                lengths,
                paths,
                cor,
                method,
                features,
                n_mfcc,
                n_mels,
                max_length,
                100,
                window,
                sr
            )

        if denoise == True:
            x, y, audio, lengths = get_denoise(
                x,
                y,
                audio,
                lengths,
                emotion_map,
                cor,
                method,
                features,
                n_mfcc,
                n_mels,
                max_length,
                window,
                sr
            )

        new_label = []
        for i in y:
            if i not in [1, 2, 3]:
                new_label.append(i)
            else:
                for k, j in enumerate(emotion_map.keys()):
                    if label in j:
                        emotion = emotion_map[j]
                new_label.append(emotion)
        y = new_label

        # train: 1200 (900+300, train+val)
        # finetune: 600 train, 200 val, 200 test
        if method != "wav2vec":
            if index == 0:  # train corpus
                # split into train and val
                sample_index = random.sample(
                    [i for i in range(np.array(x).shape[0])], 1200
                )
                X_train, X_val, ytrain, yval = train_test_split(  # 2800, 1680, 1120
                    np.array(x)[sample_index, :], y, test_size=0.25, random_state=9
                )  # 3:1  # train + val
            elif index == 1:  # finetune
                sample_index = random.sample(
                    [i for i in range(np.array(x).shape[0])], 1000
                )
                Xtune_train, X_left, ytune_train, yleft = train_test_split(
                    np.array(x)[sample_index, :], y, test_size=0.4, random_state=9
                )  # 3:2

                Xtune_val, X_test, ytune_val, ytest = train_test_split(
                    X_left, yleft, test_size=0.5, random_state=9
                )  # 1:1
        else:
            if index == 0:
                train_corpus_audio = audio
            else:
                finetune_corpus_audio = audio

    length = None
    if method == "wav2vec":
        length = max(lengths)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base", return_attention_mask=True
        )
        X_train_corpus = feature_extractor(
            train_corpus_audio,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=length,
            truncation=True,
            padding=True,
        )

        sample_index = random.sample([i for i in range(np.array(x).shape[0])], 1200)
        X_train, X_val, ytrain, yval = train_test_split(  # 2800, 1680, 1120
            np.array(X_train_corpus["input_values"])[sample_index, :],
            y,
            test_size=0.25,
            random_state=9,
        )  # 3:1  # train + val

        X_finetune_corpus = feature_extractor(
            finetune_corpus_audio,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=length,
            truncation=True,
            padding=True,
        )

        sample_index = random.sample(
            [i for i in range(np.array(X_finetune_corpus).shape[0])], 1000
        )
        Xtune_train, X_left, ytune_train, yleft = train_test_split(
            np.array(X_finetune_corpus["input_values"])[sample_index, :],
            y,
            test_size=0.4,
            random_state=9,
        )  # 3:2

        Xtune_val, X_test, ytune_val, ytest = train_test_split(
            X_left, yleft, test_size=0.5, random_state=9
        )  # 1:1

    return (
        X_train,
        ytrain,
        X_val,
        yval,
        X_test,
        ytest,
        length,
        Xtune_train,
        ytune_train,
        Xtune_val,
        ytune_val,
    )
    # 900, 300, 300, train_corpus 1200, test_corpus 300


"""
description: This function is used for loading data from preprocessed dataset into model input.
param {*} task: task Aor B
param {*} path: preprocessed dataset path
param {*} method: selected model for experiment
param {*} batch_size: batch size of NNs
return {*}: loaded model input 
"""


def load_data(
    task,
    method,
    cc,
    dataset,
    features,
    n_mfcc=40,
    n_mels=128,
    scaled=None,
    max_length=109,
    batch_size=16,
    reverse=False,
    noise=False,
    denoise=False,
    window=None,
    corpus=None,
    sr=16000
):
    if task == "speech":
        if corpus == None:
            if dataset == "RAVDESS":
                X_train, ytrain, X_val, yval, X_test, ytest, length = load_RAVDESS(
                    method,
                    features=features,
                    n_mfcc=n_mfcc,
                    n_mels=n_mels,
                    scaled=scaled,
                    max_length=max_length,
                    reverse=reverse,
                    noise=noise,
                    denoise=denoise,
                    window=window,
                    sr=sr
                )
            elif dataset == "TESS":
                X_train, ytrain, X_val, yval, X_test, ytest, length = load_TESS(
                    method,
                    features=features,
                    n_mfcc=n_mfcc,
                    n_mels=n_mels,
                    scaled=scaled,
                    max_length=max_length,
                    reverse=reverse,
                    noise=noise,
                    denoise=denoise,
                    window=window,
                    sr=sr
                )
            elif dataset == "SAVEE":
                X_train, ytrain, X_val, yval, X_test, ytest, length = load_SAVEE(
                    method,
                    features=features,
                    n_mfcc=n_mfcc,
                    n_mels=n_mels,
                    scaled=scaled,
                    max_length=max_length,
                    reverse=reverse,
                    noise=noise,
                    denoise=denoise,
                    window=window,
                    sr=sr
                )
            elif dataset == "CREMA-D":
                X_train, ytrain, X_val, yval, X_test, ytest, length = load_CREMA(
                    method,
                    features=features,
                    n_mfcc=n_mfcc,
                    n_mels=n_mels,
                    scaled=scaled,
                    max_length=max_length,
                    reverse=reverse,
                    noise=noise,
                    denoise=denoise,
                    window=window,
                    sr=sr
                )
            elif dataset == "EmoDB":
                X_train, ytrain, X_val, yval, X_test, ytest, length = load_EmoDB(
                    method,
                    features=features,
                    n_mfcc=n_mfcc,
                    n_mels=n_mels,
                    scaled=scaled,
                    max_length=max_length,
                    reverse=reverse,
                    noise=noise,
                    denoise=denoise,
                    window=window,
                    sr=sr
                )
            elif dataset == "eNTERFACE":
                X_train, ytrain, X_val, yval, X_test, ytest, length = load_eNTERFACE(
                    method,
                    features=features,
                    n_mfcc=n_mfcc,
                    n_mels=n_mels,
                    scaled=scaled,
                    max_length=max_length,
                    reverse=reverse,
                    noise=noise,
                    denoise=denoise,
                    window=window,
                    sr=sr
                )
        else:
            if cc == "mix":
                X_train, ytrain, X_val, yval, X_test, ytest, length = load_mix_corpus(
                    method,
                    features,
                    n_mfcc,
                    n_mels,
                    scaled,
                    max_length,
                    reverse,
                    noise,
                    denoise,
                    window=None,
                    corpus=corpus,
                    sr=sr
                )
            elif cc == "finetune":
                (
                    X_train,
                    ytrain,
                    X_val,
                    yval,
                    X_test,
                    ytest,
                    length,
                    Xtune_train,
                    ytune_train,
                    Xtune_val,
                    ytune_val,
                ) = load_finetune_corpus(
                    method,
                    features,
                    n_mfcc,
                    n_mels,
                    scaled,
                    max_length,
                    reverse,
                    noise,
                    denoise,
                    window=None,
                    corpus=corpus,
                    sr=sr
                )
            elif cc == "cross":
                X_train, ytrain, X_val, yval, X_test, ytest, length = load_cross_corpus(
                    method,
                    features,
                    n_mfcc,
                    n_mels,
                    scaled,
                    max_length,
                    reverse,
                    noise,
                    denoise,
                    window=None,
                    corpus=corpus,
                    sr=sr
                )

        shape = np.array(X_train).shape[1]
        num_classes = len(set(ytrain))
        if method in ["SVM", "KNN", "DT", "RF", "NB", "LSTM", "CNN", "AlexNet", "GMM"]:
            if (
                cc != "finetune"
            ):  # cc of finetune can only be used in CNN, AlexNet, LSTM
                return X_train, ytrain, X_val, yval, X_test, ytest, shape, num_classes
            else:  # finetune
                return (
                    X_train,
                    ytrain,
                    X_val,
                    yval,
                    X_test,
                    ytest,
                    shape,
                    num_classes,
                    Xtune_train,
                    ytune_train,
                    Xtune_val,
                    ytune_val,
                    
                )
        elif method in ["MLP", "RNN", "wav2vec"]:
            train_ds = tf.data.Dataset.from_tensor_slices(
                (X_train, np.array(ytrain).astype(int))
            ).batch(batch_size)
            val_ds = tf.data.Dataset.from_tensor_slices(
                (X_val, np.array(yval).astype(int))
            ).batch(batch_size)
            test_ds = tf.data.Dataset.from_tensor_slices(
                (X_test, np.array(ytest).astype(int))
            ).batch(batch_size)
            if method == "wav2vec":
                return train_ds, val_ds, test_ds, num_classes, length
            else:
                return train_ds, val_ds, test_ds, shape, num_classes

    # file = os.listdir(path)
    # Xtest, ytest, Xtrain, ytrain, Xval, yval = [], [], [], [], [], []

    # # divide into train/validation/test dataset
    # for index, f in enumerate(file):
    #     if not os.path.isfile(os.path.join(path, f)):
    #         continue
    #     else:
    #         img = cv2.imread(os.path.join(path, f))
    #         if task == "A" and method in [
    #             "LR",
    #             "KNN",
    #             "SVM",
    #             "DT",
    #             "NB",
    #             "RF",
    #             "ABC",
    #             "KMeans",
    #         ]:
    #             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         if "test" in f:
    #             Xtest.append(img)
    #             ytest.append(f.split("_")[1][0])
    #         elif "train" in f:
    #             Xtrain.append(img)
    #             ytrain.append(f.split("_")[1][0])
    #         elif "val" in f:
    #             Xval.append(img)
    #             yval.append(f.split("_")[1][0])

    # if method in ["LR", "KNN", "SVM", "DT", "NB", "RF", "ABC", "KMeans"]:  # baselines
    #     if task == "A":
    #         n, h, w = np.array(Xtrain).shape
    #         Xtrain = np.array(Xtrain).reshape(
    #             n, h * w
    #         )  # need to reshape gray picture into two-dimensional ones
    #         Xval = np.array(Xval).reshape(len(Xval), h * w)
    #         Xtest = np.array(Xtest).reshape(len(Xtest), h * w)
    #     elif task == "B":
    #         n, h, w, c = np.array(Xtrain).shape
    #         Xtrain = np.array(Xtrain).reshape(n, h * w * c)
    #         Xval = np.array(Xval).reshape(len(Xval), h * w * c)
    #         Xtest = np.array(Xtest).reshape(len(Xtest), h * w * c)

    #         # shuffle dataset
    #         Xtrain, ytrain = shuffle(Xtrain, ytrain, random_state=42)
    #         Xval, yval = shuffle(Xval, yval, random_state=42)
    #         Xtest, ytest = shuffle(Xtest, ytest, random_state=42)

    #         # use PCA for task B to reduce dimensionality
    #         pca = PCA(n_components=64)
    #         Xtrain = pca.fit_transform(Xtrain)
    #         Xval = pca.fit_transform(Xval)
    #         Xtest = pca.fit_transform(Xtest)

    #     return Xtrain, ytrain, Xtest, ytest, Xval, yval

    # else:  # pretrained or customized
    #     n, h, w, c = np.array(Xtrain).shape
    #     Xtrain = np.array(Xtrain)
    #     Xval = np.array(Xval)
    #     Xtest = np.array(Xtest)

    #     """
    #         Notice that due to large size of task B dataset, part of train and validation data is sampled for
    #         pretrained network. However, all test data are used for performance measurement in testing procedure.
    #     """
    #     if task == "B":
    #         sample_index = random.sample([i for i in range(Xtrain.shape[0])], 40000)
    #         Xtrain = Xtrain[sample_index, :, :, :]
    #         ytrain = np.array(ytrain)[sample_index].tolist()

    #         sample_index_val = random.sample([i for i in range(Xval.shape[0])], 5000)
    #         Xval = Xval[sample_index_val, :]
    #         yval = np.array(yval)[sample_index_val].tolist()

    #         sample_index_test = random.sample([i for i in range(Xtest.shape[0])], 7180)
    #         Xtest = Xtest[sample_index_test, :]
    #         ytest = np.array(ytest)[sample_index_test].tolist()

    #     if method in [
    #         "CNN",
    #         "MLP",
    #         "EnsembleNet",
    #     ]:  # customized, loaded data with batches
    #         train_ds = tf.data.Dataset.from_tensor_slices(
    #             (Xtrain, np.array(ytrain).astype(int))
    #         ).batch(batch_size)
    #         val_ds = tf.data.Dataset.from_tensor_slices(
    #             (Xval, np.array(yval).astype(int))
    #         ).batch(batch_size)
    #         test_ds = tf.data.Dataset.from_tensor_slices(
    #             (Xtest, np.array(ytest).astype(int))
    #         ).batch(batch_size)
    #         normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)  # normalization
    #         train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    #         val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    #         test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    #         return train_ds, val_ds, test_ds
    #     else:
    #         return Xtrain, ytrain, Xtest, ytest, Xval, yval


"""
description: This function is used for loading selected model.
param {*} task: task A or B
param {*} method: selected model
param {*} multilabel: whether configuring multilabels setting (can only be used with MLP/CNN in task B)
param {*} lr: learning rate for adjustment and tuning
return {*}: constructed model
"""


def load_model(
    task,
    method,
    features="mfcc",
    cc="single",
    shape=None,
    num_classes=None,
    dataset="RAVDESS",
    max_length=109,
    bidirectional=False,
    epochs=10,
    lr=0.001,
    batch_size=16,
    cv=False
):
    if task == "speech":
        if method in ["SVM", "DT", "RF", "NB", "KNN"]:
            model = SpeechBase(method)
        elif method == "MLP":
            model = MLP(
                task, method, features, cc, shape, num_classes, dataset, epochs, lr
            )
        elif method == "RNN":
            model = RNN(
                task,
                method,
                features,
                cc,
                shape,
                num_classes,
                dataset,
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
            )
        elif method == "LSTM":
            model = LSTM(
                task,
                method,
                features,
                cc,
                shape,
                num_classes,
                dataset,
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                cv=cv
            )
        elif method == "CNN":
            model = CNN(
                task,
                method,
                features,
                cc,
                shape,
                num_classes,
                dataset,
                length=max_length,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                cv=cv
            )
        elif method == "AlexNet":
            model = AlexNet(
                task,
                method,
                features,
                cc,
                shape,
                num_classes,
                dataset,
                length=max_length,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                cv=cv
            )
        elif method == "GMM":
            model = GMM(task, method, features, cc, dataset, num_classes)
        elif method == "KMeans":
            model = KMeans(task, method, features, cc, dataset, num_classes)
        elif method == "wav2vec":
            model = Wav2Vec(
                task,
                method,
                features,
                cc,
                num_classes,
                dataset,
                max_length,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )

    return model


"""
description: This function is used for visualizing confusion matrix.
param {*} task: task A or B
param {*} method: selected model
param {*} ytrain: train ground truth
param {*} yval: validation ground truth
param {*} ytest: test ground truth
param {*} train_pred: train prediction
param {*} val_pred: validation prediction
param {*} test_pred: test prediction
"""


def visual4cm(
    task,
    method,
    features,
    cc,
    dataset,
    ytrain,
    yval,
    ytest,
    train_pred,
    val_pred,
    test_pred,
    ytune_train,
    tune_train_pred,
    ytune_val,
    tune_val_pred,
):
    # confusion matrix
    if cc != "finetune":
        cms = {
            "train": confusion_matrix(ytrain, train_pred),
            "val": confusion_matrix(yval, val_pred),
            "test": confusion_matrix(ytest, test_pred),
        }
    else:
        cms = {
            "train": confusion_matrix(ytrain, train_pred),
            "val": confusion_matrix(yval, val_pred),
            "finetune_train": confusion_matrix(ytune_train, tune_train_pred),
            "finetune_val": confusion_matrix(ytune_val, tune_val_pred),
            "test": confusion_matrix(ytest, test_pred),
        }

    items = (
        ["train", "val", "test"]
        if cc != "finetune"
        else ["train", "val", "finetune_train", "finetune_val", "test"]
    )
    fig, axes = plt.subplots(1, len(items), figsize=(20, 5), sharey="row")
    for index, mode in enumerate(items):
        disp = ConfusionMatrixDisplay(
            cms[mode], display_labels=sorted(list(set(ytrain)))
        )
        disp.plot(ax=axes[index])
        disp.ax_.set_title(mode)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel("")
        if index != 0:
            disp.ax_.set_ylabel("")

    fig.text(0.45, 0.05, "Predicted label", ha="center")
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    fig.colorbar(disp.im_, ax=axes)

    if not os.path.exists(f"outputs/{task}/confusion_matrix/"):
        os.makedirs(f"outputs/{task}/confusion_matrix/")
    fig.savefig(
        f"outputs/{task}/confusion_matrix/{method}_{features}_{cc}_{dataset}.png"
    )
    plt.close()


"""
description: This function is used for visualizing auc roc curves.
param {*} task: task A or B
param {*} method: selected model
param {*} ytrain: train ground truth
param {*} yval: validation ground truth
param {*} ytest: test ground truth
param {*} train_pred: train prediction
param {*} val_pred: validation prediction
param {*} test_pred: test prediction
"""


# def visual4auc(task, method, ytrain, yval, ytest, train_pred, val_pred, test_pred):
#     # roc curves
#     rocs = {
#         "train": roc_curve(
#             np.array(ytrain).astype(int),
#             train_pred.astype(int),
#             pos_label=1,
#             drop_intermediate=True,
#         ),
#         "val": roc_curve(
#             np.array(yval).astype(int),
#             val_pred.astype(int),
#             pos_label=1,
#             drop_intermediate=True,
#         ),
#         "test": roc_curve(
#             np.array(ytest).astype(int),
#             test_pred.astype(int),
#             pos_label=1,
#             drop_intermediate=True,
#         ),
#     }

#     colors = list(mcolors.TABLEAU_COLORS.keys())

#     plt.figure(figsize=(10, 6))
#     for index, mode in enumerate(["train", "val", "test"]):
#         plt.plot(
#             rocs[mode][0],
#             rocs[mode][1],
#             lw=1,
#             label="{}(AUC={:.3f})".format(mode, auc(rocs[mode][0], rocs[mode][1])),
#             color=mcolors.TABLEAU_COLORS[colors[index]],
#         )
#     plt.plot([0, 1], [0, 1], "--", lw=1, color="grey")
#     plt.axis("square")
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.xlabel("False Positive Rate", fontsize=10)
#     plt.ylabel("True Positive Rate", fontsize=10)
#     plt.title(f"ROC Curve for {method}", fontsize=10)
#     plt.legend(loc="lower right", fontsize=5)

#     if not os.path.exists("outputs/images/roc_curve/"):
#         os.makedirs("outputs/images/roc_curve/")
#     plt.savefig(f"outputs/images/roc_curve/{method}_task{task}.png")
#     plt.close()


"""
description: This function is used for visualizing decision trees.
param {*} method: selected model
param {*} model: constructed tree model
"""


def visual4tree(task, method, features, cc, dataset, model):
    plt.figure(figsize=(100, 15))
    class_names = (
        ["pneumonia", "non-pneumonia"]
        if task == "A"
        else ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
    )
    tree.plot_tree(
        model, class_names=class_names, filled=True, rounded=True, fontsize=5
    )
    if not os.path.exists(f"outputs/{task}/trees/"):
        os.makedirs(f"outputs/{task}/trees/")
    plt.savefig(f"outputs/{task}/trees/{method}_{features}_{cc}_{dataset}.png")
    plt.close()


"""
description: This function is used for calculating metrics performance including accuracy, precision, recall, f1-score.
param {*} task: task A or B
param {*} y: ground truth
param {*} pred: predicted labels
"""


def get_metrics(task, y, pred):
    average = "macro"
    result = {
        "acc": round(
            accuracy_score(np.array(y).astype(int), pred.astype(int)) * 100, 4
        ),
        "pre": round(
            precision_score(np.array(y).astype(int), pred.astype(int), average=average)
            * 100,
            4,
        ),
        "rec": round(
            recall_score(np.array(y).astype(int), pred.astype(int), average=average)
            * 100,
            4,
        ),
        "f1": round(
            f1_score(np.array(y).astype(int), pred.astype(int), average=average) * 100,
            4,
        ),
    }
    return result


"""
description: This function is used for visualizing hyperparameter selection for grid search models.
param {*} task: task A or B
param {*} method: selected model
param {*} scores: mean test score for cross validation of different parameter combinations
"""


def hyperpara_selection(task, method, feature, cc, dataset, scores):
    plt.figure(figsize=(8, 5))
    plt.plot(scores, c="g", marker="D", markersize=5)
    plt.xlabel("Params")
    plt.ylabel("Accuracy")
    plt.title(f"Params for {method}")
    if not os.path.exists(f"outputs/{task}/hyperpara_selection/"):
        os.makedirs(f"outputs/{task}/hyperpara_selection/")
    plt.savefig(
        f"outputs/{task}/hyperpara_selection/{method}_{feature}_{cc}_{dataset}.png"
    )
    plt.close()


"""
description: This function is used for visualizing dataset label distribution.
param {*} task: task A or B
param {*} data: npz data
"""


# def visual4label(task, data):
#     fig, ax = plt.subplots(
#         nrows=1, ncols=3, figsize=(6, 3), subplot_kw=dict(aspect="equal"), dpi=600
#     )

#     for index, mode in enumerate(["train", "val", "test"]):
#         pie_data = [
#             np.count_nonzero(data[f"{mode}_labels"].flatten() == i)
#             for i in range(len(set(data[f"{mode}_labels"].flatten().tolist())))
#         ]
#         labels = [
#             f"label {i}"
#             for i in sorted(list(set(data[f"{mode}_labels"].flatten().tolist())))
#         ]
#         wedges, texts, autotexts = ax[index].pie(
#             pie_data,
#             autopct=lambda pct: f"{pct:.2f}%\n({int(np.round(pct/100.*np.sum(pie_data))):d})",
#             textprops=dict(color="w"),
#         )
#         if index == 2:
#             ax[index].legend(
#                 wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)
#             )
#         size = 6 if task == "A" else 3
#         plt.setp(autotexts, size=size, weight="bold")
#         ax[index].set_title(mode)
#     plt.tight_layout()

#     if not os.path.exists("outputs/images/"):
#         os.makedirs("outputs/images/")
#     fig.savefig(f"outputs/images/label_distribution_task{task}.png")
#     plt.close()


def visaul4curves(task, method, feature, cc, dataset, train_res, val_res, epochs):
    acc = train_res["train_acc"]
    val_acc = val_res["val_acc"]
    epochs = list(range(epochs))

    plt.figure(figsize=(10, 8))
    plt.plot(epochs, acc, label="train accuracy")
    plt.plot(epochs, val_acc, label="val accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    if not os.path.exists(f"outputs/{task}/nn_curves/"):
        os.makedirs(f"outputs/{task}/nn_curves/")
    plt.savefig(f"outputs/{task}/nn_curves/{method}_{feature}_{cc}_{dataset}_acc.png")
    plt.close()

    loss = train_res["train_loss"]
    val_loss = val_res["val_loss"]
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, loss, label="train loss")
    plt.plot(epochs, val_loss, label="val loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    if not os.path.exists(f"outputs/{task}/nn_curves/"):
        os.makedirs(f"outputs/{task}/nn_curves/")
    plt.savefig(f"outputs/{task}/nn_curves/{method}_{feature}_{cc}_{dataset}_loss.png")
    plt.close()


def visual4feature(filename, dataset, emotion):
    with soundfile.SoundFile(filename) as sound_file:
        data = sound_file.read(dtype="float32")
        if dataset == "eNTERFACE":
            data = data[:, 1]
        sr = sound_file.samplerate
    sound_file.close()

    # waveform, spectrum (specshow)
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(
        data, sr=sr, color="blue"
    )  # visualize wave in the time domain
    if not os.path.exists(f"outputs/speech/features/"):
        os.makedirs(f"outputs/speech/features/")
    plt.savefig(f"outputs/speech/features/{dataset}_{emotion}_waveform.png")
    plt.close()

    x = librosa.stft(
        data
    )  # frequency domain: The STFT represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short overlapping windows.
    xdb = librosa.amplitude_to_db(
        abs(x)
    )  # Convert an amplitude spectrogram to dB-scaled spectrogram.
    plt.figure(figsize=(11, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(
        xdb, sr=sr, x_axis="time", y_axis="hz"
    )  # visualize wave in the frequency domain
    plt.colorbar()
    if not os.path.exists(f"outputs/speech/features/"):
        os.makedirs(f"outputs/speech/features/")
    plt.savefig(f"outputs/speech/features/{dataset}_{emotion}_spectrum.png")
    plt.close()

    # mfcc spectrum
    mfc_coefficients = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mfc_coefficients, x_axis="time", norm=Normalize(vmin=-30, vmax=30)
    )
    plt.colorbar()
    plt.yticks(())
    plt.ylabel("MFC Coefficient")
    plt.title(f"{emotion} MFC Coefficients")
    plt.tight_layout()
    if not os.path.exists(f"outputs/speech/features/"):
        os.makedirs(f"outputs/speech/features/")
    plt.savefig(f"outputs/speech/features/{dataset}_{emotion}_mfcc.png")
    plt.close()

    # mel spectrum
    melspectrogram = librosa.feature.melspectrogram(
        y=data, sr=sr, n_mels=128, fmax=8000
    )
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.power_to_db(S=melspectrogram, ref=np.mean),
        y_axis="mel",
        fmax=8000,
        x_axis="time",
        norm=Normalize(vmin=-20, vmax=20),
    )
    plt.colorbar(format="%+2.0f dB", label="Amplitude")
    plt.ylabel("Mels")
    plt.title(f"{emotion} Mel spectrogram")
    plt.tight_layout()
    if not os.path.exists(f"outputs/speech/features/"):
        os.makedirs(f"outputs/speech/features/")
    plt.savefig(f"outputs/speech/features/{dataset}_{emotion}_mels.png")
    plt.close()

    # chroma spectrum
    chromagram = librosa.feature.chroma_stft(y=data, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chromagram, y_axis="chroma", x_axis="time")
    plt.colorbar(label="Relative Intensity")
    plt.title(f"{emotion} Chromagram")
    plt.tight_layout()
    if not os.path.exists(f"outputs/speech/features/"):
        os.makedirs(f"outputs/speech/features/")
    plt.savefig(f"outputs/speech/features/{dataset}_{emotion}_chroma.png")
    plt.close()


def visual4label(task, dataset, category):

    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(6, 3), subplot_kw=dict(aspect="equal"), dpi=600
    )

    pie_data = [category.count(i) for i in sorted(list(set(category)))]
    labels = sorted(list(set(category)))
    wedges, texts, autotexts = ax.pie(
        pie_data,
        autopct=lambda pct: f"{pct:.2f}%\n({int(np.round(pct/100.*np.sum(pie_data))):d})",
        textprops=dict(color="w"),
    )
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    size = 6
    plt.setp(autotexts, size=size, weight="bold")
    ax.set_title(f"Emotion distribution of {dataset}")

    plt.tight_layout()

    if not os.path.exists(f"outputs/{task}/emotion_labels/"):
        os.makedirs(f"outputs/{task}/emotion_labels/")
    fig.savefig(f"outputs/{task}/emotion_labels/{dataset}.png")
    plt.close()
