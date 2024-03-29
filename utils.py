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
import noisereduce as nr
import os
import cv2
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
import numpy as np
from scipy.io import wavfile
import soundfile
import librosa


def get_padding(data, max_length):
    if max_length > data.shape[1]:
        pad_width = max_length - data.shape[1]
        data = np.pad(data, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        data = data[:, :max_length]

    return data


def get_features(
    method,
    filename,
    features="mfcc",
    n_mfcc=40,
    n_mels=128,
    max_length=109,
    window=None,
):
    with soundfile.SoundFile(filename) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        if window != None:
            X = X[int(window[0] * sample_rate) : int(window[1] * sample_rate)]

        result = np.array([])

        if method in ["CNN", "AlexNet"]:
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc)
            mfccs = get_padding(mfccs, max_length)
            stft = np.abs(librosa.stft(X))  # spectrum
            chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
            chroma = get_padding(chroma, max_length)
            mel = librosa.feature.melspectrogram(
                y=X, sr=sample_rate, n_mels=n_mels, fmax=8000
            )
            mel = get_padding(mel, max_length)

        else:
            mfccs = np.mean(
                librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0
            )
            stft = np.abs(librosa.stft(X))  # spectrum
            chroma = np.mean(
                librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0
            )
            mel = np.mean(
                librosa.feature.melspectrogram(
                    y=X, sr=sample_rate, n_mels=n_mels, fmax=8000
                ).T,
                axis=0,
            )

        if features == "mfcc":
            result = np.hstack((result, mfccs))
        elif features == "chroma":
            result = np.hstack((result, chroma))
        elif features == "mel":
            result = np.hstack((result, mel))
        elif features == "all":
            result = np.hstack((result, mfccs))
            result = np.hstack((result, chroma))
            result = np.hstack((result, mel))
    sound_file.close()
    return result


def print_features(data, features, n_mfcc, n_mels):
    if features == "all" or "mfcc":
        # Check chromagram feature values
        mfcc_features = data[:, :n_mfcc]
        mfcc_min = np.min(mfcc_features)
        mfcc_max = np.max(mfcc_features)
        # stack all features into a single series so we don't get a mean of means or stdev of stdevs
        mfcc_mean = mfcc_features.stack().mean()
        mfcc_stdev = mfcc_features.stack().std()
        print(
            f"{n_mfcc} MFCC features:       \
        min = {mfcc_min:.3f}, \
        max = {mfcc_max:.3f}, \
        mean = {mfcc_mean:.3f}, \
        deviation = {mfcc_stdev:.3f}"
        )

    if features == "all" or "chroma":
        # Check chroma feature values
        chroma_features = data[:, n_mfcc:11]
        chroma_min = np.min(chroma_features)
        chroma_max = np.max(chroma_features)
        # stack all features into a single series so we don't get a mean of means or stdev of stdevs
        chroma_mean = chroma_features.stack().mean()
        chroma_stdev = chroma_features.stack().std()
        print(
            f"\n11 chroma features:             \
        min = {chroma_min:.3f},\
        max = {chroma_max:.3f},\
        mean = {chroma_mean:.3f},\
        deviation = {chroma_stdev:.3f}"
        )

    if features == "all" or "mel":
        # Check mel spectrogram feature values
        mel_features = data[:, n_mfcc + 11 :]
        mel_min = np.min(mel_features)
        mel_max = np.max(mel_features)
        # stack all features into a single series so we don't get a mean of means or stdev of stdevs
        mel_mean = mel_features.stack().mean()
        mel_stdev = mel_features.stack().std()
        print(
            f"\n{n_mels} Mel Spectrogram features: \
        min = {mel_min:.3f}, \
        max = {mel_max:.3f}, \
        mean = {mel_mean:.3f}, \
        deviation = {mel_stdev:.3f}"
        )


def transform_feature(x, features, n_mfcc, n_mels, scaled):
    print_features(x, features, n_mfcc, n_mels)
    if scaled == "minmax":
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
    elif scaled == "standard":
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    return x


def get_reverse(
    x, y, path, dataset, method, features, n_mfcc, n_mels, max_length, sample, window
):
    sample_index = random.sample([i for i in range(x.shape[0])], sample)
    for i in sample_index:
        sound = AudioSegment.from_file(path[i], format="wav")
        reversed_sound = sound.reverse()
        name = path[i].split(".")[0].split("/")[-1]
        if not os.path.exists(f"datasets/{dataset}_reverse/"):
            os.makedirs(f"datasets/{dataset}_reverse/")
        reversed_sound.export(
            f"datasets/{dataset}_reverse/{name}_reverse.wav", format="wav"
        )
        feature = get_features(
            method,
            f"datasets/{dataset}_reverse/{name}_reverse.wav",
            features,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            window=window,
        )
        x.append(feature)
        y.append(y[i])
    return x, y


def get_noise(
    x, y, path, dataset, method, features, n_mfcc, n_mels, max_length, sample, window
):
    sample_index = random.sample([i for i in range(x.shape[0])], sample)
    for i in sample_index:
        with soundfile.SoundFile(path[i]) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate

        random_values = np.random.rand(len(X))
        X = X + 2e-2 * random_values
        name = path[i].split(".")[0].split("/")[-1]
        if not os.path.exists(f"datasets/{dataset}_noise/"):
            os.makedirs(f"datasets/{dataset}_noise/")
        soundfile.write(f"datasets/{dataset}_noise/{name}_noise.wav", X, sample_rate)

        feature = get_features(
            method,
            f"datasets/{dataset}_noise/{name}_noise.wav",
            features,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            window=window,
        )
        x.append(feature)
        y.append(y[i])
    return x, y


def get_denoise(
    x, y, path, dataset, method, features, n_mfcc, n_mels, max_length, sample, window
):
    sample_index = random.sample([i for i in range(x.shape[0])], sample)
    for i in sample_index:
        audio = AudioSegment.from_file(path[i], format="wav")
        samples = np.array(audio.get_array_of_samples())
        reduced_noise = nr.reduce_noise(samples, sr=audio.frame_rate)
        reduced_audio = AudioSegment(
            reduced_noise.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels,
        )
        name = path[i].split(".")[0].split("/")[-1]
        if not os.path.exists(f"datasets/{dataset}_denoise/"):
            os.makedirs(f"datasets/{dataset}_denoise/")
        reduced_audio.export(
            f"datasets/{dataset}_denoise/{name}_denoise.wav", format="wav"
        )

        feature = get_features(
            method,
            f"datasets/{dataset}_noise/{name}_denoise.wav",
            features,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            window=window,
        )
        x.append(feature)
        y.append(y[i])
    return x, y


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
):
    x, y, category, path = [], [], [], []
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
        feature = get_features(
            method,
            file,
            features,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            window=window,
        )
        x.append(feature)
        y.append(emotion)
        path.append(file)
        category.append(category_map[file_name.split("-")[2]])
        if (
            category_map[file_name.split("-")[2]] not in list(set(category))
            or len(category) == 0
        ):
            visual4feature(file, "RAVDESS", category_map[file_name.split("-")[2]])

    visual4label("speech", "RAVDESS", category)
    print(np.array(x).shape)  # (864,40), (288,40), (288,40)

    if scaled != None:
        x = transform_feature(x, features, n_mfcc, n_mels, scaled)

    if reverse == True:
        x, y = get_reverse(
            x,
            y,
            path,
            "RAVDESS",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
        )

    if noise == True:
        x, y = get_noise(
            x,
            y,
            path,
            "RAVDESS",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
        )

    if denoise == True:
        x, y = get_denoise(
            x,
            y,
            path,
            "RAVDESS",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
        )

    X_train, X_left, ytrain, yleft = train_test_split(
        np.array(x), y, test_size=0.4, random_state=9
    )  # 3:2
    X_val, X_test, yval, ytest = train_test_split(
        X_left, yleft, test_size=0.5, random_state=9
    )  # 1:1
    return X_train, ytrain, X_val, yval, X_test, ytest


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
):
    x, y, category, path = [], [], [], []
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
            feature = get_features(
                method,
                os.path.join(dirname, filename),
                features,
                n_mfcc=n_mfcc,
                n_mels=n_mels,
                max_length=max_length,
                window=window,
            )
            label = filename.split("_")[-1].split(".")[0]
            emotion = emotion_map[label.lower()]
            x.append(feature)
            y.append(emotion)
            path.append(os.path.join(dirname, filename))
            category.append(label.lower())
            if label.lower() not in list(set(category)) or len(category) == 0:
                visual4feature(os.path.join(dirname, filename), "TESS", label.lower())

        if len(y) == 2800:
            break

    visual4label("speech", "TESS", category)

    if scaled != None:
        x = transform_feature(x, features, n_mfcc, n_mels, scaled)

    if reverse == True:
        x, y = get_reverse(
            x,
            y,
            path,
            "TESS",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            500,
            window,
        )

    if noise == True:
        x, y = get_noise(
            x,
            y,
            path,
            "TESS",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
        )

    if denoise == True:
        x, y = get_denoise(
            x,
            y,
            path,
            "TESS",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
        )

    X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
        np.array(x), y, test_size=0.4, random_state=9
    )  # 3:2
    X_val, X_test, yval, ytest = train_test_split(
        X_left, yleft, test_size=0.5, random_state=9
    )  # 1:1
    # (1680, 40), (560, 40), (560, 40), (1680,)
    return X_train, ytrain, X_val, yval, X_test, ytest


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
):
    x, y, category, paths = [], [], [], []
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
        feature = get_features(
            method,
            os.path.join(path, file),
            features,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            window=window,
        )
        label = file.split("_")[-1][:-2]
        emotion = emotion_map[label]
        x.append(feature)
        y.append(emotion)
        paths.append(os.path.join(path, file))
        category.append(category_map[label])
        if category_map[label] not in list(set(category)) or len(category) == 0:
            visual4feature(os.path.join(path, file), "SAVEE", category_map[label])

    visual4label("speech", "SAVEE", category)

    if scaled != None:
        x = transform_feature(x, features, n_mfcc, n_mels, scaled)

    if reverse == True:
        x, y = get_reverse(
            x,
            y,
            paths,
            "SAVEE",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            500,
            window,
        )  # sample need to evaluate

    if noise == True:
        x, y = get_noise(
            x,
            y,
            path,
            "SAVEE",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
        )

    if denoise == True:
        x, y = get_denoise(
            x,
            y,
            path,
            "SAVEE",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
        )

    X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
        np.array(x), y, test_size=0.4, random_state=9
    )  # 3:2
    X_val, X_test, yval, ytest = train_test_split(
        X_left, yleft, test_size=0.5, random_state=9
    )  # 1:1
    # (1680, 40), (560, 40), (560, 40), (1680,)
    return X_train, ytrain, X_val, yval, X_test, ytest


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
):
    x, y, category, paths = [], [], [], []
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
    path = "datasets/speech/CREAM-D"
    for file in os.listdir(path):
        feature = get_features(
            method,
            os.path.join(path, file),
            features,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            window=window,
        )
        label = file.split("_")[2]
        emotion = emotion_map[label.lower()]
        x.append(feature)
        y.append(emotion)
        paths.append(os.path.join(path, file))
        category.append(category_map[label.lower()])
        if category_map[label.lower()] not in list(set(category)) or len(category) == 0:
            visual4feature(
                os.path.join(path, file), "SAVEE", category_map[label.lower()]
            )

    visual4label("speech", "CREMA", category)

    if scaled != None:
        x = transform_feature(x, features, n_mfcc, n_mels, scaled)

    if reverse == True:
        x, y = get_reverse(
            x,
            y,
            paths,
            "CREMA",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            500,
            window,
        )  # sample need to evaluate

    if noise == True:
        x, y = get_noise(
            x,
            y,
            path,
            "CREMA",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
        )

    if denoise == True:
        x, y = get_denoise(
            x,
            y,
            path,
            "CREMA",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
        )

    X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
        np.array(x), y, test_size=0.4, random_state=9
    )  # 3:2
    X_val, X_test, yval, ytest = train_test_split(
        X_left, yleft, test_size=0.5, random_state=9
    )  # 1:1
    # (1680, 40), (560, 40), (560, 40), (1680,)
    return X_train, ytrain, X_val, yval, X_test, ytest


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
):
    x, y, category, paths = [], [], [], []
    emotion_map = {
        "W": 0,  # angry
        "L": 1,  # boredom
        "E": 2,  # disgust
        "A": 3,  # anxiety/fear
        "F": 4,  # happiness
        "T": 5,  # sadness
    }
    category_map = {
        "W": "angry",
        "L": "boredom",
        "E": "disgust",
        "A": "anxiety/fear",
        "F": "happiness",
        "T": "sadness",
    }
    path = "datasets/speech/EmoDB"
    for file in os.listdir(path):
        feature = get_features(
            method,
            os.path.join(path, file),
            features,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            window=window,
        )
        label = file.split(".")[0][-2]
        emotion = emotion_map[label]
        x.append(feature)
        y.append(emotion)
        paths.append(os.path.join(path, file))
        category.append(category_map[label])
        if category_map[label] not in list(set(category)) or len(category) == 0:
            visual4feature(os.path.join(path, file), "SAVEE", category_map[label])

    visual4label("speech", "EmoDB", category)
    if scaled != None:
        x = transform_feature(x, features, n_mfcc, n_mels, scaled)

    if reverse == True:
        x, y = get_reverse(
            x,
            y,
            paths,
            "EmoDB",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            500,
            window,
        )  # sample need to evaluate

    if noise == True:
        x, y = get_noise(
            x,
            y,
            path,
            "EmoDB",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
        )

    if denoise == True:
        x, y = (
            get_denoise(
                x,
                y,
                path,
                "EmoDB",
                method,
                features,
                n_mfcc,
                n_mels,
                max_length,
                300,
                window,
            ),
        )

    X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
        np.array(x), y, test_size=0.4, random_state=9
    )  # 3:2
    X_val, X_test, yval, ytest = train_test_split(
        X_left, yleft, test_size=0.5, random_state=9
    )  # 1:1
    # (1680, 40), (560, 40), (560, 40), (1680,)
    return X_train, ytrain, X_val, yval, X_test, ytest


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
):
    x, y, category, paths = [], [], [], []
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
    path = "datasets/speech/eNTERFACE"
    for file in os.listdir(path):
        feature = get_features(
            method,
            os.path.join(path, file),
            features,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            window=window,
        )
        label = file.split(".")[0].split("_")[-2]
        emotion = emotion_map[label]
        x.append(feature)
        y.append(emotion)
        paths.append(os.path.join(path, file))
        category.append(category_map[label])
        if category_map[label] not in list(set(category)) or len(category) == 0:
            visual4feature(os.path.join(path, file), "SAVEE", category_map[label])

    visual4label("speech", "eNTERFACE05", category)

    if scaled != None:
        x = transform_feature(x, features, n_mfcc, n_mels, scaled)

    if reverse == True:
        x, y = get_reverse(
            x,
            y,
            paths,
            "eNTERFACE",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            500,
            window,
        )  # sample need to evaluate

    if noise == True:
        x, y = get_noise(
            x,
            y,
            path,
            "eNTERFACE",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
        )

    if denoise == True:
        x, y = get_denoise(
            x,
            y,
            path,
            "eNTERFACE",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
        )

    X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
        np.array(x), y, test_size=0.4, random_state=9
    )  # 3:2
    X_val, X_test, yval, ytest = train_test_split(
        X_left, yleft, test_size=0.5, random_state=9
    )  # 1:1
    # (1680, 40), (560, 40), (560, 40), (1680,)
    return X_train, ytrain, X_val, yval, X_test, ytest


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
):
    if task == "speech":
        if dataset == "RAVDESS":
            X_train, ytrain, X_val, yval, X_test, ytest = load_RAVDESS(
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
            )
        elif dataset == "TESS":
            X_train, ytrain, X_val, yval, X_test, ytest = load_TESS(
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
            )
        elif dataset == "SAVEE":
            X_train, ytrain, X_val, yval, X_test, ytest = load_SAVEE(
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
            )
        elif dataset == "CREMA-D":
            X_train, ytrain, X_val, yval, X_test, ytest = load_CREMA(
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
            )
        elif dataset == "EmoDB":
            X_train, ytrain, X_val, yval, X_test, ytest = load_EmoDB(
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
            )
        elif dataset == "eINTERFACE":
            X_train, ytrain, X_val, yval, X_test, ytest = load_eNTERFACE(
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
            )

        shape = np.array(X_train).shape[1]
        num_classes = len(set(ytrain))
        if method in ["SVM", "KNN", "DT", "RF", "NB", "LSTM", "CNN", "AlexNet", "GMM"]:
            return X_train, ytrain, X_val, yval, X_test, ytest, shape, num_classes
        elif method in ["MLP", "RNN"]:
            train_ds = tf.data.Dataset.from_tensor_slices(
                (X_train, np.array(ytrain).astype(int))
            ).batch(batch_size)
            val_ds = tf.data.Dataset.from_tensor_slices(
                (X_val, np.array(yval).astype(int))
            ).batch(batch_size)
            test_ds = tf.data.Dataset.from_tensor_slices(
                (X_test, np.array(ytest).astype(int))
            ).batch(batch_size)
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
    features,
    cc,
    shape,
    num_classes,
    dataset,
    max_length=109,
    bidirectional=False,
    epochs=10,
    lr=0.001,
    batch_size=16,
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
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
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
                bidirectional=bidirectional,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )
        elif method == "GMM":
            model = GMM(
                task,
                method,
                features,
                cc,
                dataset,
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
):
    # confusion matrix
    cms = {
        "train": confusion_matrix(ytrain, train_pred),
        "val": confusion_matrix(yval, val_pred),
        "test": confusion_matrix(ytest, test_pred),
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey="row")
    for index, mode in enumerate(["train", "val", "test"]):
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
