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
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# from dtaidistance import dtw
from scipy.stats import pearsonr
import pandas as pd
import math
from pydub import AudioSegment
from audio_similarity import AudioSimilarity
import cv2
from patchify import patchify
import seaborn as sns
import csv
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from imblearn.metrics import classification_report_imbalanced
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from scipy.signal import butter, lfilter

from scipy.signal import wiener
import mediapipe as mp
import dlib

# from transformers import AutoFeatureExtractor
import noisereduce as nr
import cv2
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
from sklearn.metrics import (
    balanced_accuracy_score,
    class_likelihood_ratios,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
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
from scipy.io import wavfile
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from models.speech.AlexNet import AlexNet
from models.speech.CNN import CNN as speech_CNN
from models.speech.baselines import Baselines as SpeechBase
from models.speech.MLP import MLP as speech_MLP
from models.speech.RNN import RNN
from models.speech.LSTM import LSTM
from models.speech.GMM import GMM
from models.speech.KMeans import KMeans
from models.image.CNN import CNN as image_CNN
from models.image.MLP import MLP as image_MLP
from models.image.Inception import Inception
from models.image.ViT import ViT
import numpy as np
from scipy.io import wavfile
import soundfile
import librosa
import numpy as np
import librosa
import soundfile as sf
import numpy as np
import scipy.signal as signal
import soundfile as sf


random.seed(123)


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
    sr=16000,
):
    with soundfile.SoundFile(filename) as sound_file:
        X = sound_file.read(dtype="float32")  # 121715,2    # 62462  # 58124  # 45456,
        if dataset == "eNTERFACE" and len(X.shape) == 2:
            X = X[:, 1]
        sample_rate = sound_file.samplerate
        # print(sample_rate)
        if sr != 16000:  # resample
            X = librosa.resample(X, orig_sr=sample_rate, target_sr=sr)

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
    sr,
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
            sr=sr,
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
    sr,
    noise,
):
    sample_index = random.sample([i for i in range(np.array(x).shape[0])], sample)
    if noise == "white":
        avg_snr = 0
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

            signal_power = np.mean(X**2)
            noise_power = np.mean((2e-2 * random_values) ** 2)
            snr = 10 * np.log10(signal_power / noise_power)
            avg_snr += snr
        avg_snr /= len(sample_index)
        print("Average Signal-to-Noise Ratio (SNR):", avg_snr, "dB")

    elif noise == "buzz":
        avg_snr = 0
        for i in sample_index:
            ori_audio = AudioSegment.from_file(path[i])
            duration, frequency, amplitude, sample_rate = 10, 100, 20000, 16000
            t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
            buzzing_wave = amplitude * np.sin(2 * np.pi * frequency * t)
            buzzing_wave = buzzing_wave.astype(np.int16)
            buzzing_noise = AudioSegment(
                buzzing_wave.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1,
            )
            audio_with_noise = ori_audio.overlay(buzzing_noise)
            signal_power = np.mean(np.array(ori_audio.get_array_of_samples()) ** 2)
            noise_power = np.mean(np.array(buzzing_noise.get_array_of_samples()) ** 2)
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                avg_snr += snr
            else:
                snr = float("-inf")  # Set SNR to negative infinity
        avg_snr = avg_snr / len(sample_index)
        print("Average Signal-to-Noise Ratio (SNR):", avg_snr, "dB")
    elif noise == "bubble":
        avg_snr = 0
        for i in sample_index:
            # Load original audio file
            original_audio, sr = librosa.load(path[i], sr=None)
            # Generate bubble noise with the same duration and sample rate as the original audio
            duration = len(original_audio) / sr
            bubble_frequency_range = (1000, 5000)
            bubble_duration_range = (0.05, 0.5)
            amplitude_range = (0.05, 0.1)
            # Generate random parameters for each bubble
            num_bubbles = int(
                duration * np.random.uniform(1, 10)
            )  # Adjust number of bubbles based on duration
            frequencies = np.random.uniform(*bubble_frequency_range, size=num_bubbles)
            durations = np.random.uniform(*bubble_duration_range, size=num_bubbles)
            amplitudes = np.random.uniform(*amplitude_range, size=num_bubbles)
            # Generate bubble noise signal
            t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
            bubble_noise = np.zeros_like(t)
            for freq, dur, amp in zip(frequencies, durations, amplitudes):
                envelope = signal.gaussian(
                    int(dur * sample_rate), int(dur * sample_rate / 4)
                )
                bubble = amp * np.sin(
                    2 * np.pi * freq * np.linspace(0, dur, int(dur * sample_rate))
                )
                start_idx = np.random.randint(0, len(t) - len(bubble))
                bubble_noise[start_idx : start_idx + len(bubble)] += bubble * envelope
            noisy_audio = original_audio + bubble_noise

            # Calculate SNR
            signal_power = np.sum(original_audio**2) / len(original_audio)
            noise_power = np.sum(bubble_noise**2) / len(bubble_noise)
            snr = 10 * np.log10(signal_power / noise_power)
            avg_anr += snr
        avg_snr = avg_snr / len(sample_index)
        print("Average Signal-to-Noise Ratio (SNR): {:.2f} dB".format(avg_snr))
    elif noise == "cocktail":
        for i in sample_index:
            original_audio, sample_rate = librosa.load(path[i], sr=None)
            reversed_audio = original_audio[::-1]  # mix the audio with its reverse
            mixed_audio = original_audio + reversed_audio
            mixed_audio /= np.max(np.abs(mixed_audio))
            # # Save the mixed audio
            # mixed_audio_path = "mixed_audio_with_reverse.wav"
            # sf.write(mixed_audio_path, mixed_audio, sample_rate)

    name = path[i].split(".")[0].split("/")[-1]
    if not os.path.exists(f"datasets/speech/{dataset}_noise/"):
        os.makedirs(f"datasets/speech/{dataset}_noise/")
    if noise == "white":
        soundfile.write(
            f"datasets/speech/{dataset}_noise/{name}_noise.wav", X, sample_rate
        )
    elif noise == "buzz":
        audio_with_noise.export(
            f"datasets/speech/{dataset}_noise/{name}_noise.wav", format="wav"
        )
    elif noise == "bubble":
        soundfile.write(
            f"datasets/speech/{dataset}_noise/{name}_noise.wav", noisy_audio, sr
        )
    elif noise == "cocktail":
        soundfile.write(
            f"datasets/speech/{dataset}_noise/{name}_noise.wav",
            mixed_audio,
            sample_rate,
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
        sr=sr,
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
    sr,
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
            sr=sr,
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
    sr=16000,
    split=None,
):
    x, y, category, path, audio, lengths, corr, corr_emo, similarities = (
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

    # if we use original class for ranging split
    # emotion_map = {
    #     "01": 0,  # 'neutral'
    #     "02": 1,  # 'calm'
    #     "03": 2,  # 'happy'
    #     "04": 3,  # 'sad'
    #     "05": 4,  # 'angry'
    #     "06": 5,  # 'fearful'
    #     "07": 6,  # 'disgust'
    #     "08": 7,  # 'surprised'
    # }

    # this is for ranging split with 3 classes
    if split == None:
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
    else:
        emotion_map = {
            ("01", "02", "neutral", "n", "neu", "L", "N"): 0,  # neutral
            (
                "03",
                "08",
                "happy",
                "ps",
                "h",
                "su",
                "hap",
                "F",
                "ha",
                "su",
                "happiness",
            ): 1,  # positive
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
                "anger",
                "sadness",
            ): 2,
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
        if split == None:
            emotion = emotion_map[file_name.split("-")[2]]
        else:
            for k, i in enumerate(emotion_map.keys()):
                if file_name.split("-")[2] in i:
                    emotion = emotion_map[i]
        feature, X = get_features(
            "RAVDESS",
            method,
            file,
            features,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            max_length=max_length,
            window=window,
            sr=sr,
        )
        x.append(feature)
        y.append(emotion)
        path.append(file)
        category.append(category_map[file_name.split("-")[2]])
        audio.append(X)
        lengths.append(len(X))
        if category.count(category_map[file_name.split("-")[2]]) == 1:
            visual4feature(file, "RAVDESS", category_map[file_name.split("-")[2]])
            corr.append(file)
            corr_emo.append(category_map[file_name.split("-")[2]])
        if category.count(category_map[file_name.split("-")[2]]) == 2:
            index = category.index(
                category_map[file_name.split("-")[2]]
            )  # find the index of the first one
            visual4corr3(
                "RAVDESS", path[index], file, category_map[file_name.split("-")[2]]
            )
            # visual4corr4(
            #     "RAVDESS", path[index], file, category_map[file_name.split("-")[2]]
            # )
            visual4corrMAV(
                "RAVDESS", path[index], file, category_map[file_name.split("-")[2]]
            )

    #         similarity = signal_similarity(
    #             "RAVDESS", path[index], file, category_map[file_name.split("-")[2]]
    #         )
    #         similarity["score"]["emotion"] = category_map[file_name.split("-")[2]]
    #         similarities.append(similarity["score"])

    # filename = f"outputs/speech/signal_similarity/RAVDESS_score.csv"
    # with open(filename, mode="w", newline="") as file:
    #     writer = csv.DictWriter(file, fieldnames=similarities[0].keys())
    #     writer.writeheader()
    #     for row in similarities:
    #         writer.writerow(row)

    visual4corr2("RAVDESS", corr, corr_emo)
    visual4label("speech", "RAVDESS", category)
    print(np.array(x).shape)  # (864,40), (288,40), (288,40)
    if method not in ["AlexNet", "CNN"]:
        visual4corr(x, "RAVDESS")

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
            sr,
        )

    if noise != None:
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
            sr,
            noise,
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
            sr,
        )

    length = None if method != "wav2vec" else max(lengths)

    if split == None:
        if method != "wav2vec":
            X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
                np.array(x), y, test_size=0.4, random_state=9
            )  # 3:2
        else:
            # feature_extractor = AutoFeatureExtractor.from_pretrained(
            #     "facebook/wav2vec2-base", return_attention_mask=True
            # )
            # X = feature_extractor(
            #     audio,
            #     sampling_rate=feature_extractor.sampling_rate,
            #     max_length=length,
            #     truncation=True,
            #     padding=True,
            # )
            # X_train, X_left, ytrain, yleft = train_test_split(  # 54988
            #     np.array(X["input_values"]), y, test_size=0.4, random_state=9
            # )  # 3:2
            pass

        X_val, X_test, yval, ytest = train_test_split(
            X_left, yleft, test_size=0.5, random_state=9
        )  # 1:1
        # (1680, 40), (560, 40), (560, 40), (1680,)
    else:  # split != None can only be used for AlexNet as single-corpus split experiment
        """
        # this one is used for original ranging split of proportion
        X_train, X_test, ytrain, ytest = train_test_split(  # 2800, 1680, 1120
            np.array(x), y, test_size=split, random_state=9
        )  # 0.25
        X_train, X_val, ytrain, yval = train_test_split(
            X_train, ytrain, test_size=0.5, random_state=9
        )  # 1:1 for train : val

        """

        # this is the new one after modification, which is used for corresponding split for single-corpus
        # for cross-corpus-split-size

        # test
        random.seed(123)
        test_index = random.sample(
            [i for i in range(np.array(x).shape[0])], 200  # 200 fixed for testing size
        )  # 1440, 40
        left_index = [i for i in range(np.array(x).shape[0]) if i not in test_index]
        X_test = np.array(x)[test_index, :]
        ytest = np.array(y)[test_index].tolist()
        X_left = np.array(x)[left_index, :]
        yleft = np.array(y)[left_index].tolist()

        # train/val
        random.seed(123)
        train_index = random.sample(
            [i for i in range(np.array(X_left).shape[0])], int(split * 200)
        )  # train + val
        X_train, X_val, ytrain, yval = train_test_split(
            X_left[train_index, :],
            np.array(yleft)[train_index].tolist(),
            test_size=0.5,
            random_state=9,
        )  # 1:1 for train : val

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
    sr=16000,
    split=None,
):
    x, y, category, path, audio, lengths, corr, corr_emo, similarities = (
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

    # original class of ranging split
    # emotion_map = {
    #     "angry": 0,
    #     "disgust": 1,
    #     "fear": 2,
    #     "happy": 3,
    #     "neutral": 4,
    #     "ps": 5,
    #     "sad": 6,
    # }

    if split == None:
        emotion_map = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "neutral": 4,
            "ps": 5,
            "sad": 6,
        }
    else:
        emotion_map = {
            ("01", "02", "neutral", "n", "neu", "L", "N"): 0,  # neutral
            (
                "03",
                "08",
                "happy",
                "ps",
                "h",
                "su",
                "hap",
                "F",
                "ha",
                "su",
                "happiness",
            ): 1,  # positive
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
                "anger",
                "sadness",
            ): 2,
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
                sr=sr,
            )
            label = filename.split("_")[-1].split(".")[0]
            if split == None:
                emotion = emotion_map[label.lower()]
            else:
                for k, i in enumerate(emotion_map.keys()):
                    label = filename.split("_")[-1].split(".")[0]
                    if label in i:
                        emotion = emotion_map[i]

            x.append(feature)
            y.append(emotion)
            path.append(os.path.join(dirname, filename))
            category.append(label.lower())
            audio.append(X)
            lengths.append(len(X))
            if category.count(label.lower()) == 1:
                visual4feature(os.path.join(dirname, filename), "TESS", label.lower())
                corr.append(os.path.join(dirname, filename))
                corr_emo.append(label.lower())
            if category.count(label.lower()) == 2:
                index = category.index(label.lower())  # find the index of the first one
                visual4corr3(
                    "TESS", path[index], os.path.join(dirname, filename), label.lower()
                )
                # visual4corr4(
                #     "TESS", path[index], os.path.join(dirname, filename), label.lower()
                # )
                visual4corrMAV(
                    "TESS", path[index], os.path.join(dirname, filename), label.lower()
                )
                # similarity = signal_similarity(
                #     "TESS", path[index], os.path.join(dirname, filename), label.lower()
                # )
                # similarity["score"]["emotion"] = label.lower()
                # similarities.append(similarity["score"])

        if len(y) == 2800:
            break

    # filename = f"outputs/speech/signal_similarity/TESS_score.csv"
    # with open(filename, mode="w", newline="") as file:
    #     writer = csv.DictWriter(file, fieldnames=similarities[0].keys())
    #     writer.writeheader()
    #     for row in similarities:
    #         writer.writerow(row)

    visual4corr2("TESS", corr, corr_emo)
    visual4label("speech", "TESS", category)
    print(np.array(x).shape)
    if method not in ["AlexNet", "CNN"]:
        visual4corr(x, "TESS")

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
            sr,
        )

    if noise != None:
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
            sr,
            noise,
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
            sr,
        )

    length = None if method != "wav2vec" else max(lengths)

    if split == None:
        if method != "wav2vec":
            X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
                np.array(x), y, test_size=0.4, random_state=9
            )  # 3:2
        else:
            # feature_extractor = AutoFeatureExtractor.from_pretrained(
            #     "facebook/wav2vec2-base", return_attention_mask=True
            # )
            # X = feature_extractor(
            #     audio,
            #     sampling_rate=feature_extractor.sampling_rate,
            #     max_length=length,
            #     truncation=True,
            #     padding=True,
            # )  # (1440, 84351)
            # X_train, X_left, ytrain, yleft = train_test_split(
            #     np.array(X["input_values"]), y, test_size=0.4, random_state=9
            # )  # 3:2
            pass

        X_val, X_test, yval, ytest = train_test_split(
            X_left, yleft, test_size=0.5, random_state=9
        )  # 1:1
        # (1680, 40), (560, 40), (560, 40), (1680,)
    else:  # split != None can only be used for AlexNet as single-corpus split experiment
        """
        # this one is used for original ranging split of proportion
        X_train, X_test, ytrain, ytest = train_test_split(  # 2800, 1680, 1120
            np.array(x), y, test_size=split, random_state=9
        )  # 0.25
        X_train, X_val, ytrain, yval = train_test_split(
            X_train, ytrain, test_size=0.5, random_state=9
        )  # 1:1 for train : val

        """

        # this is the new one after modification, which is used for corresponding split for single-corpus
        # for cross-corpus-split-size

        # test
        random.seed(123)
        test_index = random.sample(
            [i for i in range(np.array(x).shape[0])], 200  # 200 fixed for testing size
        )  # 1440, 40
        left_index = [i for i in range(np.array(x).shape[0]) if i not in test_index]
        X_test = np.array(x)[test_index, :]
        ytest = np.array(y)[test_index].tolist()
        X_left = np.array(x)[left_index, :]
        yleft = np.array(y)[left_index].tolist()

        # train/val
        random.seed(123)
        train_index = random.sample(
            [i for i in range(np.array(X_left).shape[0])], int(split * 200)
        )  # train + val
        X_train, X_val, ytrain, yval = train_test_split(
            X_left[train_index, :],
            np.array(yleft)[train_index].tolist(),
            test_size=0.5,
            random_state=9,
        )  # 1:1 for train : val

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
    sr=16000,
    split=None,
):
    x, y, category, paths, audio, lengths, corr, corr_emo, similarities = (
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

    # original class ranging split
    # emotion_map = {
    #     "a": 0,  # angry
    #     "d": 1,  # digust
    #     "f": 2,  # fear
    #     "h": 3,  # happiness
    #     "n": 4,  # neutral
    #     "sa": 5,  # sadness
    #     "su": 6,  # surprise
    # }

    if split == None:
        emotion_map = {
            "a": 0,  # angry
            "d": 1,  # digust
            "f": 2,  # fear
            "h": 3,  # happiness
            "n": 4,  # neutral
            "sa": 5,  # sadness
            "su": 6,  # surprise
        }
    else:
        emotion_map = {
            ("01", "02", "neutral", "n", "neu", "L", "N"): 0,  # neutral
            (
                "03",
                "08",
                "happy",
                "ps",
                "h",
                "su",
                "hap",
                "F",
                "ha",
                "su",
                "happiness",
            ): 1,  # positive
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
                "anger",
                "sadness",
            ): 2,
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
            sr=sr,
        )
        label = file.split(".")[0].split("_")[-1][:-2]
        if split == None:
            emotion = emotion_map[label]
        else:
            for k, i in enumerate(emotion_map.keys()):
                if label in i:
                    emotion = emotion_map[i]
        x.append(feature)
        y.append(emotion)
        paths.append(os.path.join(path, file))
        category.append(category_map[label])
        audio.append(X)
        lengths.append(len(X))
        if category.count(category_map[label]) == 1:
            visual4feature(os.path.join(path, file), "SAVEE", category_map[label])
            corr.append(os.path.join(path, file))
            corr_emo.append(category_map[label])
        if category.count(category_map[label]) == 2:
            index = category.index(
                category_map[label]
            )  # find the index of the first one
            visual4corr3(
                "SAVEE", paths[index], os.path.join(path, file), category_map[label]
            )
            # visual4corr4(
            #     "SAVEE", paths[index], os.path.join(path, file), category_map[label]
            # )
            visual4corrMAV(
                "SAVEE", paths[index], os.path.join(path, file), category_map[label]
            )
    #         similarity = signal_similarity(
    #             "SAVEE", paths[index], os.path.join(path, file), category_map[label]
    #         )
    #         similarity["score"]["emotion"] = category_map[label]
    #         similarities.append(similarity["score"])

    # filename = f"outputs/speech/signal_similarity/SAVEE_score.csv"
    # with open(filename, mode="w", newline="") as file:
    #     writer = csv.DictWriter(file, fieldnames=similarities[0].keys())
    #     writer.writeheader()
    #     for row in similarities:
    #         writer.writerow(row)

    visual4corr2("SAVEE", corr, corr_emo)
    visual4label("speech", "SAVEE", category)
    print(np.array(x).shape)
    if method not in ["AlexNet", "CNN"]:
        visual4corr(x, "SAVEE")

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
            sr,
        )  # sample need to evaluate

    if noise != None:
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
            sr,
            noise,
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
            sr,
        )

    length = None if method != "wav2vec" else max(lengths)

    if split == None:
        if method != "wav2vec":
            X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
                np.array(x), y, test_size=0.4, random_state=9
            )  # 3:2
        else:
            # feature_extractor = AutoFeatureExtractor.from_pretrained(
            #     "facebook/wav2vec2-base", return_attention_mask=True
            # )
            # X = feature_extractor(
            #     audio,
            #     sampling_rate=feature_extractor.sampling_rate,
            #     max_length=length,
            #     truncation=True,
            #     padding=True,
            # )
            # X_train, X_left, ytrain, yleft = train_test_split(
            #     np.array(X["input_values"]), y, test_size=0.4, random_state=9
            # )  # 3:2
            pass

        X_val, X_test, yval, ytest = train_test_split(
            X_left, yleft, test_size=0.5, random_state=9
        )  # 1:1
        # (1680, 40), (560, 40), (560, 40), (1680,)
    else:  # split != None can only be used for AlexNet as single-corpus split experiment
        """
        # this one is used for original ranging split of proportion
        X_train, X_test, ytrain, ytest = train_test_split(  # 2800, 1680, 1120
            np.array(x), y, test_size=split, random_state=9
        )  # 0.25
        X_train, X_val, ytrain, yval = train_test_split(
            X_train, ytrain, test_size=0.5, random_state=9
        )  # 1:1 for train : val

        """

        # this is the new one after modification, which is used for corresponding split for single-corpus
        # for cross-corpus-split-size

        # test
        random.seed(123)
        test_index = random.sample(
            [i for i in range(np.array(x).shape[0])], 200  # 200 fixed for testing size
        )  # 1440, 40
        left_index = [i for i in range(np.array(x).shape[0]) if i not in test_index]
        X_test = np.array(x)[test_index, :]
        ytest = np.array(y)[test_index].tolist()
        X_left = np.array(x)[left_index, :]
        yleft = np.array(y)[left_index].tolist()

        # train/val
        random.seed(123)
        train_index = random.sample(
            [i for i in range(np.array(X_left).shape[0])], int(split * 200)
        )  # train + val
        X_train, X_val, ytrain, yval = train_test_split(
            X_left[train_index, :],
            np.array(yleft)[train_index].tolist(),
            test_size=0.5,
            random_state=9,
        )  # 1:1 for train : val

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
    sr=16000,
    split=None,
):
    x, y, category, paths, audio, lengths, corr, corr_emo, similarities = (
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
    # original class ranging split
    # emotion_map = {
    #     "ang": 0,  # angry
    #     "dis": 1,  # disgust
    #     "fea": 2,  # fear
    #     "hap": 3,  # happiness
    #     "neu": 4,  # neutral
    #     "sad": 5,  # sadness
    # }

    if split == None:
        emotion_map = {
            "ang": 0,  # angry
            "dis": 1,  # disgust
            "fea": 2,  # fear
            "hap": 3,  # happiness
            "neu": 4,  # neutral
            "sad": 5,  # sadness
        }
    else:
        emotion_map = {
            ("01", "02", "neutral", "n", "neu", "L", "N"): 0,  # neutral
            (
                "03",
                "08",
                "happy",
                "ps",
                "h",
                "su",
                "hap",
                "F",
                "ha",
                "su",
                "happiness",
            ): 1,  # positive
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
                "anger",
                "sadness",
            ): 2,
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
            sr=sr,
        )
        label = file.split("_")[2]
        # emotion = emotion_map[label.lower()]

        if split == None:
            emotion = emotion_map[label.lower()]
        else:
            for k, i in enumerate(emotion_map.keys()):
                if label.lower() in i:
                    emotion = emotion_map[i]
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
            corr.append(os.path.join(path, file))
            corr_emo.append(category_map[label.lower()])
        if category.count(category_map[label.lower()]) == 2:
            index = category.index(
                category_map[label.lower()]
            )  # find the index of the first one
            visual4corr3(
                "CREMA",
                paths[index],
                os.path.join(path, file),
                category_map[label.lower()],
            )
            # visual4corr4(
            #     "CREMA",
            #     paths[index],
            #     os.path.join(path, file),
            #     category_map[label.lower()],
            # )
            visual4corrMAV(
                "CREMA",
                paths[index],
                os.path.join(path, file),
                category_map[label.lower()],
            )
    #         similarity = signal_similarity(
    #             "CREMA",
    #             paths[index],
    #             os.path.join(path, file),
    #             category_map[label.lower()],
    #         )
    #         similarity["score"]["emotion"] = category_map[label.lower()]
    #         similarities.append(similarity["score"])

    # filename = f"outputs/speech/signal_similarity/CREMA-D_score.csv"
    # with open(filename, mode="w", newline="") as file:
    #     writer = csv.DictWriter(file, fieldnames=similarities[0].keys())
    #     writer.writeheader()
    #     for row in similarities:
    #         writer.writerow(row)

    visual4corr2("CREMA", corr, corr_emo)
    visual4label("speech", "CREMA", category)
    print(np.array(x).shape)
    if method not in ["AlexNet", "CNN"]:
        visual4corr(x, "CREMA-D")

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
            sr,
        )  # sample need to evaluate

    if noise != None:
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
            sr,
            noise,
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
            sr,
        )

    length = None if method != "wav2vec" else max(lengths)

    if split == None:
        if method != "wav2vec":
            X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
                np.array(x), y, test_size=0.4, random_state=9
            )  # 3:2
        else:
            # feature_extractor = AutoFeatureExtractor.from_pretrained(
            #     "facebook/wav2vec2-base", return_attention_mask=True
            # )
            # X = feature_extractor(
            #     audio,
            #     sampling_rate=feature_extractor.sampling_rate,
            #     max_length=length,
            #     truncation=True,
            #     padding=True,
            # )
            # X_train, X_left, ytrain, yleft = train_test_split(
            #     np.array(X["input_values"]), y, test_size=0.4, random_state=9
            # )  # 3:2
            pass

        X_val, X_test, yval, ytest = train_test_split(
            X_left, yleft, test_size=0.5, random_state=9
        )  # 1:1
        # (4465, 40), (1488, 40), (1489, 40), (1680,)
    else:  # split != None can only be used for AlexNet as single-corpus split experiment
        """
        # this one is used for original ranging split of proportion
        X_train, X_test, ytrain, ytest = train_test_split(  # 2800, 1680, 1120
            np.array(x), y, test_size=split, random_state=9
        )  # 0.25
        X_train, X_val, ytrain, yval = train_test_split(
            X_train, ytrain, test_size=0.5, random_state=9
        )  # 1:1 for train : val

        """

        # this is the new one after modification, which is used for corresponding split for single-corpus
        # for cross-corpus-split-size

        # test
        random.seed(123)
        test_index = random.sample(
            [i for i in range(np.array(x).shape[0])], 200  # 200 fixed for testing size
        )  # 1440, 40
        left_index = [i for i in range(np.array(x).shape[0]) if i not in test_index]
        X_test = np.array(x)[test_index, :]
        ytest = np.array(y)[test_index].tolist()
        X_left = np.array(x)[left_index, :]
        yleft = np.array(y)[left_index].tolist()

        # train/val
        random.seed(123)
        train_index = random.sample(
            [i for i in range(np.array(X_left).shape[0])], int(split * 200)
        )  # train + val
        X_train, X_val, ytrain, yval = train_test_split(
            X_left[train_index, :],
            np.array(yleft)[train_index].tolist(),
            test_size=0.5,
            random_state=9,
        )  # 1:1 for train : val

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
    sr=16000,
    split=None,
):
    x, y, category, paths, audio, lengths, corr, corr_emo, similarities = (
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
    # original class ranging split

    # emotion_map = {
    #     "W": 0,  # angry
    #     "L": 1,  # boredom
    #     "E": 2,  # disgust
    #     "A": 3,  # anxiety/fear
    #     "F": 4,  # happiness
    #     "T": 5,  # sadness
    #     "N": 6,
    # }

    if split == None:
        emotion_map = {
            "W": 0,  # angry
            "L": 1,  # boredom
            "E": 2,  # disgust
            "A": 3,  # anxiety/fear
            "F": 4,  # happiness
            "T": 5,  # sadness
            "N": 6,
        }
    else:
        emotion_map = {
            ("01", "02", "neutral", "n", "neu", "L", "N"): 0,  # neutral
            (
                "03",
                "08",
                "happy",
                "ps",
                "h",
                "su",
                "hap",
                "F",
                "ha",
                "su",
                "happiness",
            ): 1,  # positive
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
                "anger",
                "sadness",
            ): 2,
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
            sr=sr,
        )
        label = file.split(".")[0][-2]
        if split == None:
            emotion = emotion_map[label]
        else:
            for k, i in enumerate(emotion_map.keys()):
                if label in i:
                    emotion = emotion_map[i]
        x.append(feature)
        y.append(emotion)
        paths.append(os.path.join(path, file))
        category.append(category_map[label])
        audio.append(X)
        lengths.append(len(X))
        if category.count(category_map[label]) == 1:
            visual4feature(os.path.join(path, file), "EmoDB", category_map[label])
            corr.append(os.path.join(path, file))
            corr_emo.append(category_map[label])
        if category.count(category_map[label]) == 2:
            index = category.index(
                category_map[label]
            )  # find the index of the first one
            visual4corr3(
                "EmoDB", paths[index], os.path.join(path, file), category_map[label]
            )
            # visual4corr4(
            #     "EmoDB", paths[index], os.path.join(path, file), category_map[label]
            # )
            visual4corrMAV(
                "EmoDB", paths[index], os.path.join(path, file), category_map[label]
            )
    #         similarity = signal_similarity(
    #             "EmoDB", paths[index], os.path.join(path, file), category_map[label]
    #         )
    #         similarity["score"]["emotion"] = category_map[label]
    #         similarities.append(similarity["score"])

    # filename = f"outputs/speech/signal_similarity/EmoDB_score.csv"
    # with open(filename, mode="w", newline="") as file:
    #     writer = csv.DictWriter(file, fieldnames=similarities[0].keys())
    #     writer.writeheader()
    #     for row in similarities:
    #         writer.writerow(row)

    visual4corr2("EmoDB", corr, corr_emo)
    visual4label("speech", "EmoDB", category)
    print(np.array(x).shape)
    if method not in ["AlexNet", "CNN"]:
        visual4corr(x, "EmoDB")

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
            sr,
        )  # sample need to evaluate

    if noise != None:
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
            sr,
            noise,
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
            sr,
        )

    length = None if method != "wav2vec" else max(lengths)

    if split == None:
        if method != "wav2vec":
            X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
                np.array(x), y, test_size=0.4, random_state=9
            )  # 3:2
        else:
            # feature_extractor = AutoFeatureExtractor.from_pretrained(
            #     "facebook/wav2vec2-base", return_attention_mask=True
            # )
            # X = feature_extractor(
            #     audio,
            #     sampling_rate=feature_extractor.sampling_rate,
            #     max_length=length,
            #     truncation=True,
            #     padding=True,
            # )
            # X_train, X_left, ytrain, yleft = train_test_split(
            #     np.array(X["input_values"]), y, test_size=0.4, random_state=9
            # )  # 3:2
            pass

        X_val, X_test, yval, ytest = train_test_split(
            X_left, yleft, test_size=0.5, random_state=9
        )  # 1:1
        # (1680, 40), (560, 40), (560, 40), (1680,)
    else:  # split != None can only be used for AlexNet as single-corpus split experiment
        """
        # this one is used for original ranging split of proportion
        X_train, X_test, ytrain, ytest = train_test_split(  # 2800, 1680, 1120
            np.array(x), y, test_size=split, random_state=9
        )  # 0.25
        X_train, X_val, ytrain, yval = train_test_split(
            X_train, ytrain, test_size=0.5, random_state=9
        )  # 1:1 for train : val

        """

        # this is the new one after modification, which is used for corresponding split for single-corpus
        # for cross-corpus-split-size

        # test
        random.seed(123)
        test_index = random.sample(
            [i for i in range(np.array(x).shape[0])], 200  # 200 fixed for testing size
        )  # 1440, 40
        left_index = [i for i in range(np.array(x).shape[0]) if i not in test_index]
        X_test = np.array(x)[test_index, :]
        ytest = np.array(y)[test_index].tolist()
        X_left = np.array(x)[left_index, :]
        yleft = np.array(y)[left_index].tolist()

        # train/val
        random.seed(123)
        train_index = random.sample(
            [i for i in range(np.array(X_left).shape[0])], int(split * 200)
        )  # train + val
        X_train, X_val, ytrain, yval = train_test_split(
            X_left[train_index, :],
            np.array(yleft)[train_index].tolist(),
            test_size=0.5,
            random_state=9,
        )  # 1:1 for train : val

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
    sr=16000,
    split=None,
):
    x, y, category, paths, audio, lengths, corr, corr_emo, similarities = (
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
    # original class ranging split
    # emotion_map = {
    #     "an": 0,  # angry
    #     "di": 1,  # disgust
    #     "fe": 2,  # fear
    #     "ha": 3,  # happiness
    #     "sa": 4,  # sadness
    #     "su": 5,  # surprise
    # }

    if split == None:
        emotion_map = {
            "an": 0,  # angry
            "di": 1,  # disgust
            "fe": 2,  # fear
            "ha": 3,  # happiness
            "sa": 4,  # sadness
            "su": 5,  # surprise
        }
    else:
        emotion_map = {
            # ("01", "02", "neutral", "n", "neu", "L", "N"): 0,  # neutral
            (
                "03",
                "08",
                "happy",
                "ps",
                "h",
                "su",
                "hap",
                "F",
                "ha",
                "su",
                "happiness",
            ): 0,  # positive
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
                "anger",
                "sadness",
            ): 1,
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
            sr=sr,
        )
        if file[1] == "_":
            label = file.split(".")[0].split("_")[-2]
        else:
            label = file.split(".")[0].split("_")[1]

        # emotion = emotion_map[label]
        if split == None:
            emotion = emotion_map[label]
        else:
            for k, i in enumerate(emotion_map.keys()):
                if label in i:
                    emotion = emotion_map[i]

        x.append(feature)
        y.append(emotion)
        paths.append(os.path.join(path, file))
        category.append(category_map[label])
        audio.append(X)
        lengths.append(len(X))
        if category.count(category_map[label]) == 1:
            visual4feature(os.path.join(path, file), "eNTERFACE", category_map[label])
            corr.append(os.path.join(path, file))
            corr_emo.append(category_map[label])
        if category.count(category_map[label]) == 2:
            index = category.index(
                category_map[label]
            )  # find the index of the first one
            visual4corr3(
                "eNTERFACE", paths[index], os.path.join(path, file), category_map[label]
            )
            # visual4corr4(
            #     "eNTERFACE", paths[index], os.path.join(path, file), category_map[label]
            # )
            visual4corrMAV(
                "eNTERFACE", paths[index], os.path.join(path, file), category_map[label]
            )
    #         similarity = signal_similarity(
    #             "eNTERFACE", paths[index], os.path.join(path, file), category_map[label]
    #         )
    #         similarity["score"]["emotion"] = category_map[label]
    #         similarities.append(similarity["score"])

    # filename = f"outputs/speech/signal_similarity/eNTERFACE_score.csv"
    # with open(filename, mode="w", newline="") as file:
    #     writer = csv.DictWriter(file, fieldnames=similarities[0].keys())
    #     writer.writeheader()
    #     for row in similarities:
    #         writer.writerow(row)

    visual4corr2("eNTERFACE", corr, corr_emo)
    visual4label("speech", "eNTERFACE05", category)
    print(np.array(x).shape)
    if method not in ["AlexNet", "CNN"]:
        visual4corr(x, "eNTERFACE")

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
            sr,
        )  # sample need to evaluate

    if noise != None:
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
            sr,
            noise,
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
            sr,
        )

    length = None if method != "wav2vec" else max(lengths)

    if split == None:
        if method != "wav2vec":
            X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
                np.array(x), y, test_size=0.4, random_state=9
            )  # 3:2
        else:
            # feature_extractor = AutoFeatureExtractor.from_pretrained(
            #     "facebook/wav2vec2-base", return_attention_mask=True
            # )
            # X = feature_extractor(
            #     audio,
            #     sampling_rate=feature_extractor.sampling_rate,
            #     max_length=length,
            #     truncation=True,
            #     padding=True,
            # )

            # X_train, X_left, ytrain, yleft = train_test_split(
            #     np.array(X), y, test_size=0.4, random_state=9
            # )  # 3:2
            pass

        X_val, X_test, yval, ytest = train_test_split(
            X_left, yleft, test_size=0.5, random_state=9
        )  # 1:1
        # (1680, 40), (560, 40), (560, 40), (1680,)
    else:  # split != None can only be used for AlexNet as single-corpus split experiment
        """
        # this one is used for original ranging split of proportion
        X_train, X_test, ytrain, ytest = train_test_split(  # 2800, 1680, 1120
            np.array(x), y, test_size=split, random_state=9
        )  # 0.25
        X_train, X_val, ytrain, yval = train_test_split(
            X_train, ytrain, test_size=0.5, random_state=9
        )  # 1:1 for train : val

        """

        # this is the new one after modification, which is used for corresponding split for single-corpus
        # for cross-corpus-split-size

        # test
        random.seed(123)
        test_index = random.sample(
            [i for i in range(np.array(x).shape[0])], 200  # 200 fixed for testing size
        )  # 1440, 40
        left_index = [i for i in range(np.array(x).shape[0]) if i not in test_index]
        X_test = np.array(x)[test_index, :]
        ytest = np.array(y)[test_index].tolist()
        X_left = np.array(x)[left_index, :]
        yleft = np.array(y)[left_index].tolist()

        # train/val
        random.seed(123)
        train_index = random.sample(
            [i for i in range(np.array(X_left).shape[0])], int(split * 200)
        )  # train + val
        X_train, X_val, ytrain, yval = train_test_split(
            X_left[train_index, :],
            np.array(yleft)[train_index].tolist(),
            test_size=0.5,
            random_state=9,
        )  # 1:1 for train : val

    return X_train, ytrain, X_val, yval, X_test, ytest, length


def load_AESDD(
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
    sr=16000,
    split=None,
):
    x, y, category, path, audio, lengths, corr, corr_emo, similarities = (
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

    # original class of ranging split
    # emotion_map = {
    #     "anger": 0,
    #     "disgust": 1,
    #     "fear": 2,
    #     "happiness": 3,
    #     "sadness": 4,
    # }

    if split == None:
        emotion_map = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "happiness": 3,
            "sadness": 4,
        }
    else:
        emotion_map = {
            # ("01", "02", "neutral", "n", "neu", "L", "N"): 0,  # neutral
            (
                "03",
                "08",
                "happy",
                "ps",
                "h",
                "su",
                "hap",
                "F",
                "ha",
                "su",
                "happiness",
            ): 0,  # positive
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
                "anger",
                "sadness",
            ): 1,
        }

    for dirname, _, filenames in os.walk("datasets/speech/AESDD"):
        for filename in filenames:
            feature, X = get_features(
                "AESDD",
                method,
                os.path.join(dirname, filename),
                features,
                n_mfcc=n_mfcc,
                n_mels=n_mels,
                max_length=max_length,
                window=window,
                sr=sr,
            )
            # label = filename.split("_")[-1].split(".")[0]
            label = dirname.split("/")[-1]

            # original ranging spllit
            # emotion = emotion_map[label]

            if split == None:
                emotion = emotion_map[label]
            else:
                for k, i in enumerate(emotion_map.keys()):
                    # label = filename.split("_")[-1].split(".")[0]
                    if label in i:
                        emotion = emotion_map[i]

            x.append(feature)
            y.append(emotion)
            path.append(os.path.join(dirname, filename))
            category.append(label)
            audio.append(X)
            lengths.append(len(X))
            if category.count(label) == 1:
                visual4feature(os.path.join(dirname, filename), "AESDD", label)
                corr.append(os.path.join(dirname, filename))
                corr_emo.append(label)
            if category.count(label) == 2:
                index = category.index(label)  # find the index of the first one
                visual4corr3(
                    "AESDD", path[index], os.path.join(dirname, filename), label
                )
                # visual4corr4(
                #     "AESDD", path[index], os.path.join(dirname, filename), label
                # )
                visual4corrMAV(
                    "AESDD", path[index], os.path.join(dirname, filename), label
                )
    #             similarity = signal_similarity(
    #                 "AESDD", path[index], os.path.join(dirname, filename), label
    #             )
    #             similarity["score"]["emotion"] = label
    #             similarities.append(similarity["score"])
    #     # if len(y) == 2800:
    #     #     break
    # filename = f"outputs/speech/signal_similarity/AESDD_score.csv"
    # with open(filename, mode="w", newline="") as file:
    #     writer = csv.DictWriter(file, fieldnames=similarities[0].keys())
    #     writer.writeheader()
    #     for row in similarities:
    #         writer.writerow(row)

    visual4corr2("AESDD", corr, corr_emo)
    visual4label("speech", "AESDD", category)
    print(np.array(x).shape)
    if method not in ["AlexNet", "CNN"]:
        visual4corr(x, "AESDD")

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
            "AESDD",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            500,
            window,
            sr,
        )

    if noise != None:
        x, y, audio, lengths = get_noise(
            x,
            y,
            audio,
            lengths,
            path,
            "AESDD",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            300,
            window,
            sr,
            noise,
        )

    if denoise == True:
        x, y, audio, lengths = get_denoise(
            x,
            y,
            audio,
            lengths,
            emotion_map,
            "AESDD",
            method,
            features,
            n_mfcc,
            n_mels,
            max_length,
            window,
            sr,
        )

    length = None if method != "wav2vec" else max(lengths)

    if split == None:
        if method != "wav2vec":
            X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
                np.array(x), y, test_size=0.4, random_state=9
            )  # 3:2
        else:
            # feature_extractor = AutoFeatureExtractor.from_pretrained(
            #     "facebook/wav2vec2-base", return_attention_mask=True
            # )
            # X = feature_extractor(
            #     audio,
            #     sampling_rate=feature_extractor.sampling_rate,
            #     max_length=length,
            #     truncation=True,
            #     padding=True,
            # )  # (1440, 84351)
            # X_train, X_left, ytrain, yleft = train_test_split(
            #     np.array(X["input_values"]), y, test_size=0.4, random_state=9
            # )  # 3:2
            pass

        X_val, X_test, yval, ytest = train_test_split(
            X_left, yleft, test_size=0.5, random_state=9
        )  # 1:1
        # (1680, 40), (560, 40), (560, 40), (1680,)
    else:  # split != None can only be used for AlexNet as single-corpus split experiment
        """
        # this one is used for original ranging split of proportion
        X_train, X_test, ytrain, ytest = train_test_split(  # 2800, 1680, 1120
            np.array(x), y, test_size=split, random_state=9
        )  # 0.25
        X_train, X_val, ytrain, yval = train_test_split(
            X_train, ytrain, test_size=0.5, random_state=9
        )  # 1:1 for train : val

        """

        # this is the new one after modification, which is used for corresponding split for single-corpus
        # for cross-corpus-split-size

        # test
        random.seed(123)
        test_index = random.sample(
            [i for i in range(np.array(x).shape[0])], 200  # 200 fixed for testing size
        )  # 1440, 40
        left_index = [i for i in range(np.array(x).shape[0]) if i not in test_index]
        X_test = np.array(x)[test_index, :]
        ytest = np.array(y)[test_index].tolist()
        X_left = np.array(x)[left_index, :]
        yleft = np.array(y)[left_index].tolist()

        # train/val
        random.seed(123)
        train_index = random.sample(
            [i for i in range(np.array(X_left).shape[0])], int(split * 200)
        )  # train + val
        X_train, X_val, ytrain, yval = train_test_split(
            X_left[train_index, :],
            np.array(yleft)[train_index].tolist(),
            test_size=0.5,
            random_state=9,
        )  # 1:1 for train : val

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
    sr=16000,
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
        ("01", "02", "neutral", "n", "neu", "L", "N"): 0,  # neutral
        ("03", "08", "happy", "ps", "h", "su", "hap", "F", "ha", "su"): 1,  # positive
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
        ): 2,
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
                    sr=sr,
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
                        sr=sr,
                    )
                    for k, i in enumerate(emotion_map.keys()):
                        label = filename.split("_")[-1].split(".")[0]
                        if label in i:
                            emotion = emotion_map[i]
                    x.append(feature)
                    y.append(emotion)
                    paths.append(os.path.join(dirname, filename))
                    audio.append(X)
                    lengths.append(len(X))
        else:
            path = (
                f"datasets/speech/{cor}"
                if cor != "eNTERFACE"
                else f"datasets/speech/eNTERFACE05"
            )
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
                    sr=sr,
                )
                if cor == "SAVEE":
                    label = file.split(".")[0].split("_")[-1][:-2]
                elif cor == "CREMA-D":
                    label = file.split("_")[2].lower()
                elif cor == "EmoDB":
                    label = file.split(".")[0][-2]
                elif cor == "eNTERFACE":
                    if file[1] == "_":
                        label = file.split(".")[0].split("_")[-2]
                    else:
                        label = file.split(".")[0].split("_")[1]
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
                sr,
            )

        if noise != None:
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
                sr,
                noise,
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
                sr,
            )

        # new_label = []
        # for i in y:
        #     if i not in [1, 2, 3]:
        #         new_label.append(i)
        #     else:
        #         for k, j in enumerate(emotion_map.keys()):
        #             if label in j:
        #                 emotion = emotion_map[j]
        #         new_label.append(emotion)
        # y = new_label
        if method != "wav2vec":
            if index == 0:  # train corpus
                # split into train and val
                random.seed(123)
                sample_index = random.sample(
                    [i for i in range(np.array(x).shape[0])], 1200
                )
                X_train, X_val, ytrain, yval = train_test_split(  # 2800, 1680, 1120
                    np.array(x)[sample_index, :],
                    np.array(y)[sample_index].tolist(),
                    test_size=0.25,
                    random_state=9,
                )  # 3:1
            elif index == 1:
                random.seed(123)
                # sample_index = random.sample(
                #     [i for i in range(np.array(x).shape[0])], 300
                # )
                # X_test = np.array(x)[sample_index, :]
                # ytest = np.array(y)[sample_index].tolist()
                X_test = np.array(x)
                ytest = y
        else:
            if index == 0:
                train_corpus_audio = audio
                y_train_corpus = y
            else:
                test_corpus_audio = audio
                y_test_corpus = y

    length = None
    # if method == "wav2vec":
    #     length = max(lengths)
    #     feature_extractor = AutoFeatureExtractor.from_pretrained(
    #         "facebook/wav2vec2-base", return_attention_mask=True
    #     )
    #     X_train_corpus = feature_extractor(
    #         train_corpus_audio,
    #         sampling_rate=feature_extractor.sampling_rate,
    #         max_length=length,
    #         truncation=True,
    #         padding=True,
    #     )
    #     random.seed(123)
    #     sample_index = random.sample([i for i in range(np.array(x).shape[0])], 1200)
    #     X_train, X_val, ytrain, yval = train_test_split(  # 2800, 1680, 1120
    #         np.array(X_train_corpus["input_values"])[sample_index, :],
    #         y_train_corpus,
    #         test_size=0.25,
    #         random_state=9,
    #     )  # 3:1

    #     X_test_corpus = feature_extractor(
    #         test_corpus_audio,
    #         sampling_rate=feature_extractor.sampling_rate,
    #         max_length=length,
    #         truncation=True,
    #         padding=True,
    #     )
    #     random.seed(123)
    #     # sample_index = random.sample([i for i in range(np.array(x).shape[0])], 300)
    #     # X_test = np.array(X_test_corpus["input_values"])[sample_index, :]
    #     # ytest = np.array(y_test_corpus)[sample_index, :].tolist()
    #     X_test = np.array(X_test_corpus["input_values"])
    #     ytest = y_test_corpus.tolist()

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
    sr=16000,
):  # cc: mix, corpus: with only one string as the testing set

    # 900+300, 900 train 300 val, 300 test
    # 900/5=180, 300/5=60, 1200/5=240
    # 300
    datasets = ["RAVDESS", "TESS", "SAVEE", "CREMA-D", "EmoDB"]
    emotion_map = {
        ("01", "02", "neutral", "n", "neu", "L", "N"): 0,  # neutral
        ("03", "08", "happy", "ps", "h", "su", "hap", "F", "ha", "su"): 1,  # positive
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
        ): 2,  # tuple can be hashed but list cannot
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
                    sr=sr,
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
                        sr=sr,
                    )
                    for k, i in enumerate(emotion_map.keys()):
                        label = filename.split("_")[-1].split(".")[0]
                        if label in i:
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
                    sr=sr,
                )
                if dataset == "SAVEE":
                    label = file.split(".")[0].split("_")[-1][:-2]
                elif dataset == "CREMA-D":
                    label = file.split("_")[2].lower()
                elif dataset == "EmoDB":
                    label = file.split(".")[0][-2]
                elif dataset == "eNTERFACE":
                    if file[1] == "_":
                        label = file.split(".")[0].split("_")[-2]
                    else:
                        label = file.split(".")[0].split("_")[1]
                for k, i in enumerate(emotion_map.keys()):
                    if label in i:
                        emotion = emotion_map[i]
                x.append(feature)
                y.append(emotion)
                paths.append(os.path.join(path, file))
                audio.append(X)
                lengths.append(len(X))

        # first reverse for each dataset, then get
        random.seed(123)
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
                sr,
            )

        if noise != None:
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
                sr,
                noise,
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
                sr,
            )

        if (dataset == corpus[0]) or (corpus[0] == "eNTERFACE"):  # testing set
            # if dataset == corpus[0]:  # testing set
            left_index = [
                i for i in range(np.array(x).shape[0]) if i not in sample_index
            ]
            # random.seed(123)
            # test_index = (
            #     left_index if dataset == "SAVEE" else random.sample(left_index, 300)
            # )
            # the left index of SAVEE is not enough for 300
            X_test = np.array(x)[left_index, :]  # 300,40
            ytest = np.array(y)[left_index].tolist()

    # after traversing all dataset
    length = None
    # if method == "wav2vec":
    #     length = max(lengths)
    #     feature_extractor = AutoFeatureExtractor.from_pretrained(
    #         "facebook/wav2vec2-base", return_attention_mask=True
    #     )
    #     X_train = feature_extractor(
    #         X_train,
    #         sampling_rate=feature_extractor.sampling_rate,
    #         max_length=length,
    #         truncation=True,
    #         padding=True,
    #     )
    #     X_test = feature_extractor(
    #         X_test,
    #         sampling_rate=feature_extractor.sampling_rate,
    #         max_length=length,
    #         truncation=True,
    #         padding=True,
    #     )
    #     X_test = np.array(X_test["input_values"])

    X_train, X_val, ytrain, yval = train_test_split(  # 2800, 1680, 1120
        X_train, ytrain, test_size=0.25, random_state=9
    )  # 3:1  # 1200, 40  # 300,4

    return X_train, ytrain, X_val, yval, X_test, ytest, length


def load_mix3_corpus(
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
    sr=16000,
):  # cc: mix, corpus: with only one string as the testing set

    # 900+300, 900 train 300 val, 300 test
    # 900/5=180, 300/5=60, 1200/5=240
    # 300
    datasets = ["RAVDESS", "TESS", "CREMA-D"]
    emotion_map = {
        ("01", "02", "neutral", "n", "neu", "L", "N"): 0,  # neutral
        ("03", "08", "happy", "ps", "h", "su", "hap", "F", "ha", "su"): 1,  # positive
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
        ): 2,  # tuple can be hashed but list cannot
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
                    sr=sr,
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
                        sr=sr,
                    )
                    for k, i in enumerate(emotion_map.keys()):
                        label = filename.split("_")[-1].split(".")[0]
                        if label in i:
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
                    sr=sr,
                )
                # if dataset == "SAVEE":
                #     label = file.split(".")[0].split("_")[-1][:-2]
                # elif dataset == "CREMA-D":
                #     label = file.split("_")[2].lower()
                # elif dataset == "EmoDB":
                #     label = file.split(".")[0][-2]
                # elif dataset == "eNTERFACE":
                #     if file[1] == "_":
                #         label = file.split(".")[0].split("_")[-2]
                #     else:
                #         label = file.split(".")[0].split("_")[1]
                if dataset == "CREMA-D":
                    label = file.split("_")[2].lower()

                for k, i in enumerate(emotion_map.keys()):
                    if label in i:
                        emotion = emotion_map[i]
                x.append(feature)
                y.append(emotion)
                paths.append(os.path.join(path, file))
                audio.append(X)
                lengths.append(len(X))

        # first reverse for each dataset, then get
        # 200 for CREMA-D, RAVDESS, TESS of each
        random.seed(123)
        sample_index = random.sample([i for i in range(np.array(x).shape[0])], 200)

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
                sr,
            )

        if noise != None:
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
                sr,
                noise,
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
                sr,
            )

        # if (dataset == corpus[0]) or (corpus[0] == "eNTERFACE"):  # testing set
        if dataset == corpus[0]:  # testing set
            left_index = [
                i for i in range(np.array(x).shape[0]) if i not in sample_index
            ]
            random.seed(123)
            test_index = random.sample(left_index, 200)  # 200:200:200  -- 200
            X_test = np.array(x)[test_index, :]  # 300,40
            ytest = np.array(y)[test_index].tolist()

    # after traversing all dataset
    length = None
    # if method == "wav2vec":
    #     length = max(lengths)
    #     feature_extractor = AutoFeatureExtractor.from_pretrained(
    #         "facebook/wav2vec2-base", return_attention_mask=True
    #     )
    #     X_train = feature_extractor(
    #         X_train,
    #         sampling_rate=feature_extractor.sampling_rate,
    #         max_length=length,
    #         truncation=True,
    #         padding=True,
    #     )
    #     X_test = feature_extractor(
    #         X_test,
    #         sampling_rate=feature_extractor.sampling_rate,
    #         max_length=length,
    #         truncation=True,
    #         padding=True,
    #     )
    #     X_test = np.array(X_test["input_values"])

    X_train, X_val, ytrain, yval = train_test_split(  # 2800, 1680, 1120
        X_train, ytrain, test_size=0.5, random_state=9
    )  # 3:1  # 1200, 40  # 300,4

    return X_train, ytrain, X_val, yval, X_test, ytest, length


def load_split_corpus(
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
    sr=16000,
    split=None,
):  # cc: mix, corpus: with only one string as the testing set

    # 900+300, 900 train 300 val, 300 test
    # 900/5=180, 300/5=60, 1200/5=240
    # 300
    # datasets = ["RAVDESS", "TESS", "SAVEE", "CREMA-D", "EmoDB"]
    emotion_map = {
        ("01", "02", "neutral", "n", "neu", "L", "N"): 0,  # neutral
        ("03", "08", "happy", "ps", "h", "su", "hap", "F", "ha", "su"): 1,  # positive
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
        ): 2,  # tuple can be hashed but list cannot
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
    for index, cor in enumerate(corpus):
        x, y = [], []  # for each dataset
        if cor == "RAVDESS":
            for file in glob.glob("datasets/speech/RAVDESS/Actor_*/*.wav"):
                # print(file)
                file_name = os.path.basename(file)
                for k, i in enumerate(emotion_map.keys()):
                    if file_name.split("-")[2] in i:  # tuple
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
                    sr=sr,
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
                        sr=sr,
                    )
                    for k, i in enumerate(emotion_map.keys()):
                        label = filename.split("_")[-1].split(".")[0]
                        if label in i:
                            emotion = emotion_map[i]
                    x.append(feature)
                    y.append(emotion)
                    paths.append(os.path.join(dirname, filename))
                    audio.append(X)
                    lengths.append(len(X))
        else:
            # if cor == "eNTERFACE":
            #     cor = "eNTERFACE05"
            path = (
                f"datasets/speech/{cor}"
                if cor != "eNTERFACE"
                else f"datasets/speech/eNTERFACE05"
            )
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
                    sr=sr,
                )
                if cor == "SAVEE":
                    label = file.split(".")[0].split("_")[-1][:-2]
                elif cor == "CREMA-D":
                    label = file.split("_")[2].lower()
                elif cor == "EmoDB":
                    label = file.split(".")[0][-2]
                elif cor == "eNTERFACE":
                    if file[1] == "_":
                        label = file.split(".")[0].split("_")[-2]
                    else:
                        label = file.split(".")[0].split("_")[1]
                for k, i in enumerate(emotion_map.keys()):
                    if label in i:
                        emotion = emotion_map[i]
                x.append(feature)
                y.append(emotion)
                paths.append(os.path.join(path, file))
                audio.append(X)
                lengths.append(len(X))

        # first reverse for each dataset, then get
        random.seed(123)
        if index == 0:  # train
            sample_index = random.sample(
                [i for i in range(np.array(x).shape[0])], int(1000 * (1 - split))
            )  # 1440, 40
        elif index == 1:  # test, split is test size
            sample_index = random.sample(
                [i for i in range(np.array(x).shape[0])], int(1000 * split)
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
                sr,
            )

        if noise != None:
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
                sr,
                noise,
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
                sr,
            )

        if (cor == corpus[1]) or (corpus[1] == "eNTERFACE"):  # testing set
            # if dataset == corpus[0]:  # testing set
            left_index = [
                i for i in range(np.array(x).shape[0]) if i not in sample_index
            ]
            # random.seed(123)
            # test_index = (
            #     left_index if dataset == "SAVEE" else random.sample(left_index, 300)
            # )
            # the left index of SAVEE is not enough for 300
            X_test = np.array(x)[left_index, :]  # 300,40
            ytest = np.array(y)[left_index].tolist()

    # after traversing all dataset
    length = None
    # if method == "wav2vec":
    #     length = max(lengths)
    #     feature_extractor = AutoFeatureExtractor.from_pretrained(
    #         "facebook/wav2vec2-base", return_attention_mask=True
    #     )
    #     X_train = feature_extractor(
    #         X_train,
    #         sampling_rate=feature_extractor.sampling_rate,
    #         max_length=length,
    #         truncation=True,
    #         padding=True,
    #     )
    #     X_test = feature_extractor(
    #         X_test,
    #         sampling_rate=feature_extractor.sampling_rate,
    #         max_length=length,
    #         truncation=True,
    #         padding=True,
    #     )
    #     X_test = np.array(X_test["input_values"])

    X_train, X_val, ytrain, yval = train_test_split(  # 2800, 1680, 1120
        X_train, ytrain, test_size=0.5, random_state=9
    )  # 3:1  # 1200, 40  # 300,4

    return X_train, ytrain, X_val, yval, X_test, ytest, length


# load_split_corpus is for the original cross-corpus ranging different split for proportion
# while load_split_corpus_size is for size
def load_split_corpus_size(
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
    sr=16000,
    split=None,  # split is the size of testing set within the mixture
):  # cc: mix, corpus: with only one string as the testing set

    # 900+300, 900 train 300 val, 300 test
    # 900/5=180, 300/5=60, 1200/5=240
    # 300
    # datasets = ["RAVDESS", "TESS", "SAVEE", "CREMA-D", "EmoDB"]
    if ("eNTERFACE" in set(corpus)) and ("AESDD" in set(corpus)):
        emotion_map = {
            # ("01", "02", "neutral", "n", "neu", "L", "N"): 0,  # neutral
            (
                "03",
                "08",
                "happy",
                "ps",
                "h",
                "su",
                "hap",
                "F",
                "ha",
                "su",
                "happiness",
            ): 0,  # positive
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
                "anger",
                "sadness",
            ): 1,  # tuple can be hashed but list cannot
        }
    else:
        emotion_map = {
            ("01", "02", "neutral", "n", "neu", "L", "N"): 0,  # neutral
            (
                "03",
                "08",
                "happy",
                "ps",
                "h",
                "su",
                "hap",
                "F",
                "ha",
                "su",
                "happiness",
            ): 1,  # positive
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
                "anger",
                "sadness",
            ): 2,  # tuple can be hashed but list cannot
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
    for index, cor in enumerate(corpus):
        x, y = [], []  # for each dataset
        if cor == "RAVDESS":
            for file in glob.glob("datasets/speech/RAVDESS/Actor_*/*.wav"):
                # print(file)
                file_name = os.path.basename(file)
                for k, i in enumerate(emotion_map.keys()):
                    if file_name.split("-")[2] in i:  # tuple
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
                    sr=sr,
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
                        sr=sr,
                    )
                    for k, i in enumerate(emotion_map.keys()):
                        label = filename.split("_")[-1].split(".")[0]
                        if label in i:
                            emotion = emotion_map[i]
                    x.append(feature)
                    y.append(emotion)
                    paths.append(os.path.join(dirname, filename))
                    audio.append(X)
                    lengths.append(len(X))
        elif cor == "AESDD":
            for dirname, _, filenames in os.walk("datasets/speech/AESDD"):
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
                        sr=sr,
                    )
                    label = dirname.split("/")[-1]
                    for k, i in enumerate(emotion_map.keys()):
                        if label in i:
                            emotion = emotion_map[i]
                    x.append(feature)
                    y.append(emotion)
                    paths.append(os.path.join(dirname, filename))
                    audio.append(X)
                    lengths.append(len(X))
        else:
            # if cor == "eNTERFACE":
            #     cor = "eNTERFACE05"
            path = (
                f"datasets/speech/{cor}"
                if cor != "eNTERFACE"
                else f"datasets/speech/eNTERFACE05"
            )
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
                    sr=sr,
                )
                if cor == "SAVEE":
                    label = file.split(".")[0].split("_")[-1][:-2]
                elif cor == "CREMA-D":
                    label = file.split("_")[2].lower()
                elif cor == "EmoDB":
                    label = file.split(".")[0][-2]
                elif cor == "eNTERFACE":
                    if file[1] == "_":
                        label = file.split(".")[0].split("_")[-2]
                    else:
                        label = file.split(".")[0].split("_")[1]
                for k, i in enumerate(emotion_map.keys()):
                    if label in i:
                        emotion = emotion_map[i]
                x.append(feature)
                y.append(emotion)
                paths.append(os.path.join(path, file))
                audio.append(X)
                lengths.append(len(X))

        # first reverse for each dataset, then get
        random.seed(123)
        if index == 0:  # train
            sample_index = random.sample(
                [i for i in range(np.array(x).shape[0])], int(1000 * (1 - split))
            )  # 1440, 40
        elif index == 1:  # test, split is test size
            sample_index = random.sample(
                [i for i in range(np.array(x).shape[0])], int(1000 * split)
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
                sr,
            )

        if noise != None:
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
                sr,
                noise,
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
                sr,
            )

        if (cor == corpus[1]) or (corpus[1] == "eNTERFACE"):  # testing set
            # if dataset == corpus[0]:  # testing set
            left_index = [
                i for i in range(np.array(x).shape[0]) if i not in sample_index
            ]
            random.seed(123)
            test_index = random.sample(left_index, 200)  # fixed 200 for testing set
            X_test = np.array(x)[test_index, :]  # 300,40
            ytest = np.array(y)[test_index].tolist()

    # after traversing all dataset
    length = None
    # if method == "wav2vec":
    #     length = max(lengths)
    #     feature_extractor = AutoFeatureExtractor.from_pretrained(
    #         "facebook/wav2vec2-base", return_attention_mask=True
    #     )
    #     X_train = feature_extractor(
    #         X_train,
    #         sampling_rate=feature_extractor.sampling_rate,
    #         max_length=length,
    #         truncation=True,
    #         padding=True,
    #     )
    #     X_test = feature_extractor(
    #         X_test,
    #         sampling_rate=feature_extractor.sampling_rate,
    #         max_length=length,
    #         truncation=True,
    #         padding=True,
    #     )
    #     X_test = np.array(X_test["input_values"])

    X_train, X_val, ytrain, yval = train_test_split(  # 2800, 1680, 1120
        X_train, ytrain, test_size=0.5, random_state=9
    )  # 3:1  # 1200, 40  # 300,4

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
    sr=16000,
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
        train_corpus_audio,
        finetune_corpus_audio,
    ) = (None, None, None, None, None, None, None, None, None, None, None, None)
    emotion_map = {
        ("01", "02", "neutral", "n", "neu", "L", "N"): 0,  # neutral
        ("03", "08", "happy", "ps", "h", "su", "hap", "F", "ha", "su"): 1,  # positive
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
        ): 2,
    }
    for index, cor in enumerate(corpus):
        # index=0 train, index=1 test
        x, y, paths, audio = [], [], [], []
        if cor == "RAVDESS":
            for file in glob.glob("datasets/speech/RAVDESS/Actor_*/*.wav"):
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
                    sr=sr,
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
                        sr=sr,
                    )
                    # label = filename.split("_")[-1].split(".")[0]
                    for k, i in enumerate(emotion_map.keys()):
                        label = filename.split("_")[-1].split(".")[0]
                        if label in i:
                            emotion = emotion_map[i]
                    x.append(feature)
                    y.append(emotion)
                    paths.append(os.path.join(dirname, filename))
                    audio.append(X)
                    lengths.append(len(X))
        else:
            if cor == "eNTERFACE":
                path = f"datasets/speech/eNTERFACE05"
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
                    sr=sr,
                )
                if cor == "SAVEE":
                    label = file.split(".")[0].split("_")[-1][:-2]
                elif cor == "CREMA-D":
                    label = file.split("_")[2].lower()
                elif cor == "EmoDB":
                    label = file.split(".")[0][-2]
                elif cor == "eNTERFACE":
                    if file[1] == "_":
                        label = file.split(".")[0].split("_")[-2]
                    else:
                        label = file.split(".")[0].split("_")[1]
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
                sr,
            )

        if noise != None:
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
                sr,
                noise,
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
                sr,
            )

        # new_label = []
        # for i in y:
        #     if i not in [1, 2, 3]:
        #         new_label.append(i)
        #     else:
        #         for k, j in enumerate(emotion_map.keys()):
        #             if label in j:
        #                 emotion = emotion_map[j]
        #         new_label.append(emotion)
        # y = new_label

        # train: 1200 (900+300, train+val)
        # finetune: 600 train, 200 val, 200 test
        if method != "wav2vec":
            if index == 0:  # train corpus
                # split into train and val
                random.seed(123)
                sample_index = random.sample(
                    [i for i in range(np.array(x).shape[0])], 1200
                )
                X_train, X_val, ytrain, yval = train_test_split(  # 2800, 1680, 1120
                    np.array(x)[sample_index, :],
                    np.array(y)[sample_index],
                    test_size=0.25,
                    random_state=9,
                )  # 3:1  # train + val
            elif index == 1:  # finetune
                random.seed(123)
                # sample_index = random.sample(
                #     [i for i in range(np.array(x).shape[0])], 1000
                # )
                Xtune_train, X_left, ytune_train, yleft = train_test_split(
                    # np.array(x)[sample_index, :],
                    # np.array(y)[sample_index],
                    np.array(x),
                    y,
                    test_size=0.4,
                    random_state=9,
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
    # if method == "wav2vec":
    #     length = max(lengths)
    #     feature_extractor = AutoFeatureExtractor.from_pretrained(
    #         "facebook/wav2vec2-base", return_attention_mask=True
    #     )
    #     X_train_corpus = feature_extractor(
    #         train_corpus_audio,
    #         sampling_rate=feature_extractor.sampling_rate,
    #         max_length=length,
    #         truncation=True,
    #         padding=True,
    #     )

    #     random.seed(123)
    #     sample_index = random.sample([i for i in range(np.array(x).shape[0])], 1200)
    #     X_train, X_val, ytrain, yval = train_test_split(  # 2800, 1680, 1120
    #         np.array(X_train_corpus["input_values"])[sample_index, :],
    #         y,
    #         test_size=0.25,
    #         random_state=9,
    #     )  # 3:1  # train + val

    #     X_finetune_corpus = feature_extractor(
    #         finetune_corpus_audio,
    #         sampling_rate=feature_extractor.sampling_rate,
    #         max_length=length,
    #         truncation=True,
    #         padding=True,
    #     )

    #     random.seed(123)
    #     # sample_index = random.sample(
    #     #     [i for i in range(np.array(X_finetune_corpus).shape[0])], 1000
    #     # )
    #     Xtune_train, X_left, ytune_train, yleft = train_test_split(
    #         # np.array(X_finetune_corpus["input_values"])[sample_index, :],
    #         np.array(X_finetune_corpus["input_values"]),
    #         y,
    #         test_size=0.4,
    #         random_state=9,
    #     )  # 3:2

    #     Xtune_val, X_test, ytune_val, ytest = train_test_split(
    #         X_left, yleft, test_size=0.5, random_state=9
    #     )  # 1:1

    return (
        X_train,
        ytrain,  # 900
        X_val,
        yval,  # 300
        X_test,
        ytest,  # 200
        length,
        Xtune_train,
        ytune_train,  # 600
        Xtune_val,  # 200
        ytune_val,
    )
    # 900, 300, 300, train_corpus 1200, test_corpus 300


# def get_face_landmarks(image, draw=False, static_image_mode=True):
def get_face_landmarks(image, landmark="xyz", write=None, dataset="CK"):
    if landmark == "xyz":
        image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
        )
        image_rows, image_cols, _ = image.shape
        results = face_mesh.process(image_input_rgb)
        image_landmarks = []

        if results.multi_face_landmarks:
            ls_single_face = results.multi_face_landmarks[0].landmark
            xs_ = []
            ys_ = []
            zs_ = []
            for (
                idx
            ) in ls_single_face:  # every single landmark get three coordinates xyz
                xs_.append(idx.x)
                ys_.append(idx.y)
                zs_.append(idx.z)
                x = int(idx.x * image.shape[1])
                y = int(idx.y * image.shape[0])
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # 468x2
            for j in range(len(xs_)):  # get landmarks of the face
                image_landmarks.append(xs_[j] - min(xs_))
                image_landmarks.append(ys_[j] - min(ys_))
                image_landmarks.append(zs_[j] - min(zs_))

        if write != None:
            if not os.path.exists(f"outputs/image/landmarks/"):
                os.makedirs(f"outputs/image/landmarks/")
            cv2.imwrite(f"outputs/image/landmarks/{dataset}_468_{write}.JPG", image)
        face_mesh.close()
        res = image_landmarks

    elif landmark == "5":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        res = []
        detector = dlib.get_frontal_face_detector()
        faces = detector(image, 1)
        predictor_5 = dlib.shape_predictor(
            "models/image/shape_predictor_5_face_landmarks.dat"
        )
        for face in faces:
            # Get the landmarks/parts for the face
            image_landmarks = predictor_5(image, face)

            # Loop over the landmark points
            tmp_x = []
            tmp_y = []
            for n in range(0, 5):  # 5 landmarks with x,y, 134 features in total
                x = image_landmarks.part(n).x
                y = image_landmarks.part(n).y
                tmp_x.append(x)
                tmp_y.append(y)
            for n in range(0, 5):
                res.append(tmp_x[n] - min(tmp_x))
                res.append(tmp_y[n] - min(tmp_y))
                cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

        if (write != None) and (len(faces) != 0):
            if not os.path.exists(f"outputs/image/landmarks/"):
                os.makedirs(f"outputs/image/landmarks/")
            cv2.imwrite(
                f"outputs/image/landmarks/{dataset}_{landmark}_{write}.JPG", image
            )

    elif landmark == "68":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        res = []
        detector = dlib.get_frontal_face_detector()
        faces = detector(image, 1)
        predictor_68 = dlib.shape_predictor(
            "models/image/shape_predictor_68_face_landmarks.dat"
        )
        for face in faces:
            # Get the landmarks/parts for the face
            image_landmarks = predictor_68(image, face)

            # Loop over the landmark points
            tmp_x = []
            tmp_y = []
            for n in range(0, 68):  # 5 landmarks with x,y, 134 features in total
                x = image_landmarks.part(n).x
                y = image_landmarks.part(n).y
                tmp_x.append(x)
                tmp_y.append(y)
            for n in range(0, 68):
                res.append(tmp_x[n] - min(tmp_x))
                res.append(tmp_y[n] - min(tmp_y))
                cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
        if (write != None) and (len(faces) != 0):
            if not os.path.exists(f"outputs/image/landmarks/"):
                os.makedirs(f"outputs/image/landmarks/")
            cv2.imwrite(
                f"outputs/image/landmarks/{dataset}_{landmark}_{write}.JPG", image
            )

    # elif landmark == "468":
    #     res = []
    #     mp_face_mesh = mp.solutions.face_mesh
    #     face_mesh = mp_face_mesh.FaceMesh(
    #         static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
    #     )
    #     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     results = face_mesh.process(rgb_image)
    #     if results.multi_face_landmarks:
    #         for face_landmarks in results.multi_face_landmarks:
    #             for landmark in face_landmarks.landmark:
    #                 x = int(landmark.x * image.shape[1])
    #                 y = int(landmark.y * image.shape[0])
    #                 res.append(x)
    #                 res.append(y)
    #                 cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # 468x2

    #     if write != None:
    #         if not os.path.exists(f"outputs/image/landmarks/"):
    #             os.makedirs(f"outputs/image/landmarks/")
    #         cv2.imwrite(f"outputs/image/landmarks/CK_468_{write}.JPG", image)
    #     face_mesh.close()

    # elif landmark == "facenet":
    #     # model = VGGFace(
    #     #     model="resnet50",
    #     #     include_top=False,
    #     #     input_shape=(224, 224, 3),
    #     #     pooling="avg",
    #     # )
    #     # img = cv2.resize(image, (224, 224))
    #     # img_array = image_process.img_to_array(img)
    #     # img_array = np.expand_dims(img_array, axis=0)
    #     # img_array = preprocess_input(img_array)
    #     # image_landmarks = model.predict(img_array)
    #     # # print(features.shape)  # Should print (1, 2048)
    #     result = DeepFace.represent(img_path=image, model_name='Facenet',enforce_detection=False)
    #     print(result)

    return res


def load_CK(method, landmark=None, split=None, process=None):
    X, y, category = [], [], []
    if split == None:
        emotion_map = {
            "anger": 0,
            "contempt": 1,
            "disgust": 2,
            "fear": 3,
            "happy": 4,
            "sadness": 5,
            "surprise": 6,
        }
    else:
        emotion_map = {
            ("anger", "disgust", "fear", "sadness", "sad", "2", "3", "5", "6"): 0,
            ("happy", "surprise", "4", "1"): 1,
            ("contempt", "neutral", "7"): 2,
        }

    for dirname, _, filenames in os.walk("datasets/image/CK"):
        for filename in filenames:
            img = cv2.imread(os.path.join(dirname, filename))
            # print(img)
            # print(filename)
            # print(dirname)
            label = dirname.split("/")[-1]
            if landmark == None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if len(X) == 0:
                    if not os.path.exists("outputs/image/process/"):
                        os.makedirs("outputs/image/process/")
                    cv2.imwrite(f"outputs/image/process/CK_ori.JPG", img)

                if process == "blur":  # gaussian blur
                    img = cv2.GaussianBlur(img, (7, 7), 0)
                elif process == "noise":  # salt & pepper
                    img = random_noise(img, mode="s&p")
                    img = (255 * img).astype(np.uint8)
                elif process == "sobel":
                    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                    sobelx = cv2.convertScaleAbs(sobelx)
                    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                    sobely = cv2.convertScaleAbs(sobely)
                    img = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
                    # img = cv2.addWeighted(img, 1, sobel, 1, 0)
                elif process == "equal":
                    img = cv2.equalizeHist(img)
                elif process == "filter":
                    img = wiener(img, mysize=(15, 15))
                    img = np.clip(img, 0, 255).astype(np.uint8)

                X.append(img)  # grayscale
                if len(X) == 1:
                    cv2.imwrite(f"outputs/image/process/CK_{process}.JPG", img)

                if split != None:
                    for k, i in enumerate(emotion_map.keys()):
                        if label in i:  # tuple
                            emotion = emotion_map[i]
                    y.append(emotion)
                else:
                    category.append(label)
                    y.append(emotion_map[label])

            else:  # MLP
                write = len(X) if len(X) <= 5 else None
                face_landmarks = get_face_landmarks(
                    img, landmark=landmark, write=write, dataset="CK"
                )
                X.append(face_landmarks)  # 136 for 68
                y.append(emotion_map[label])  # anger

    visual4label("image", "CK", category)
    X, y = shuffle(X, y, random_state=42)  # shuffle
    print(len(X))
    num_classes = len(set(y))
    if landmark == None:
        n, h, w = np.array(X).shape  # expected: 48x48x1, (35887, 48, 48, 3)
        # n, h, w, d = np.array(X).shape  # expected: 48x48x1, (35887, 48, 48, 3)
        if method == "MLP":
            X = np.array(X).reshape((n, h * w)).tolist()
            h = h * w
    else:
        h = np.array(X).shape[1]

    if split == None:
        X_train, X_left, ytrain, yleft = train_test_split(
            np.array(X),
            y,
            test_size=0.4,
            random_state=9,
        )  # 3:2

        X_val, X_test, yval, ytest = train_test_split(
            X_left, yleft, test_size=0.5, random_state=9
        )  # 1:1
    else:
        random.seed(123)
        test_index = random.sample(
            [i for i in range(np.array(X).shape[0])], 200  # 200 fixed for testing size
        )  # 1440, 40
        left_index = [i for i in range(np.array(X).shape[0]) if i not in test_index]
        X_test = np.array(X)[test_index, :, :]
        ytest = np.array(y)[test_index].tolist()
        X_left = np.array(X)[left_index, :, :]
        yleft = np.array(y)[left_index].tolist()

        # train/val
        random.seed(123)
        if split == 4:
            X_train, X_val, ytrain, yval = train_test_split(
                X_left,
                np.array(yleft).tolist(),
                test_size=0.5,
                random_state=9,
            )  # 1:1 for train : val
        else:
            train_index = random.sample(
                [i for i in range(np.array(X_left).shape[0])], int(split * 200)
            )  # train + val
            X_train, X_val, ytrain, yval = train_test_split(
                X_left[train_index, :, :],
                np.array(yleft)[train_index].tolist(),
                test_size=0.5,
                random_state=9,
            )  # 1:1 for train : val

    # if landmark == False:
    #     # np.array() format for X now
    #     return X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes  # shape 48
    # if landmark == True:
    #     np.savetxt(
    #         "outputs/image/landmark_CK.txt",
    #         np.asarray(X),
    #     )

    return X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes


def load_FER(method, landmark=None, split=None, process=None):
    X, y, category = [], [], []
    if split == None:
        emotion_map = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "neutral": 4,
            "sad": 5,
            "surprise": 6,
        }
    else:
        emotion_map = {
            ("anger", "disgust", "fear", "sadness", "sad", "2", "3", "5", "6"): 0,
            ("happy", "surprise", "4", "1"): 1,
            ("contempt", "neutral", "7"): 2,
        }
    path = "datasets/image/FER"
    for firstdir in os.listdir(path):
        first_path = os.path.join(path, firstdir)
        for secdir in os.listdir(first_path):
            sec_path = os.path.join(first_path, secdir)
            for file in os.listdir(sec_path):
                img = cv2.imread(os.path.join(sec_path, file))
                if landmark == None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if len(X) == 0:
                        if not os.path.exists("outputs/image/process/"):
                            os.makedirs("outputs/image/process/")
                        cv2.imwrite(f"outputs/image/process/FER_ori.JPG", img)

                    if process == "blur":  # gaussian blur
                        img = cv2.GaussianBlur(img, (7, 7), 0)
                    elif process == "noise":  # salt & pepper
                        img = random_noise(img, mode="s&p")
                        img = (255 * img).astype(np.uint8)
                    elif process == "sobel":
                        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                        sobelx = cv2.convertScaleAbs(sobelx)
                        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                        sobely = cv2.convertScaleAbs(sobely)
                        img = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
                        # img = cv2.addWeighted(img, 1, sobel, 1, 0)
                    elif process == "equal":
                        img = cv2.equalizeHist(img)
                    elif process == "filter":
                        img = wiener(img, mysize=(15, 15))
                        img = np.clip(img, 0, 255).astype(np.uint8)

                    X.append(img)  # grayscale
                    if len(X) == 1:
                        cv2.imwrite(f"outputs/image/process/FER_{process}.JPG", img)

                    if split != None:
                        for k, i in enumerate(emotion_map.keys()):
                            if secdir in i:  # tuple
                                emotion = emotion_map[i]
                        y.append(emotion)
                    else:
                        category.append(secdir)
                        y.append(emotion_map[secdir])
                else:  # MLP
                    write = len(X) if len(X) <= 5 else None
                    face_landmarks = get_face_landmarks(
                        img, landmark=landmark, write=write, dataset="FER"
                    )
                    if landmark == "xyz":
                        if len(face_landmarks) == 1404:  # 468 x 3 = 1404
                            X.append(face_landmarks)
                            y.append(emotion_map[secdir])  # anger
                    else:
                        if len(face_landmarks) != 0:
                            X.append(face_landmarks)  # 136 for 68
                            y.append(emotion_map[secdir])  # anger

    X, y = shuffle(X, y, random_state=42)  # shuffle
    print(len(X))
    visual4label("image", "FER", category)
    num_classes = len(set(y))
    if landmark == None:
        n, h, w = np.array(X).shape  # expected: 48x48x1, (35887, 48, 48, 3)
        # n, h, w, d = np.array(X).shape  # expected: 48x48x1, (35887, 48, 48, 3)
        if method == "MLP":
            X = np.array(X).reshape((n, h * w)).tolist()
            h = h * w
    else:
        # X = X[:, :1404]
        h = np.array(X).shape[1]

    if split == None:
        X_train, X_left, ytrain, yleft = train_test_split(
            np.array(X),
            y,
            test_size=0.4,
            random_state=9,
        )  # 3:2

        X_val, X_test, yval, ytest = train_test_split(
            X_left, yleft, test_size=0.5, random_state=9
        )  # 1:1
    else:
        random.seed(123)
        test_index = random.sample(
            [i for i in range(np.array(X).shape[0])], 200  # 200 fixed for testing size
        )  # 1440, 40
        left_index = [i for i in range(np.array(X).shape[0]) if i not in test_index]
        X_test = np.array(X)[test_index, :, :]
        ytest = np.array(y)[test_index].tolist()
        X_left = np.array(X)[left_index, :, :]
        yleft = np.array(y)[left_index].tolist()

        # train/val
        random.seed(123)
        train_index = random.sample(
            [i for i in range(np.array(X_left).shape[0])], int(split * 200)
        )  # train + val
        X_train, X_val, ytrain, yval = train_test_split(
            X_left[train_index, :, :],
            np.array(yleft)[train_index].tolist(),
            test_size=0.5,
            random_state=9,
        )  # 1:1 for train : val

    # if landmark == False:
    #     # np.array() format for X now
    #     return X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes  # shape 48
    # if landmark == True:
    #     np.savetxt(
    #         "outputs/image/landmark_FER.txt",
    #         np.asarray(X),
    #     )

    return X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes


def load_RAF(method, landmark=None, split=None, process=None):
    X, y, category = [], [], []
    if split == None:
        emotion_map = {
            "1": 0,
            "2": 1,
            "3": 2,
            "4": 3,
            "5": 4,
            "6": 5,
            "7": 6,
        }
        category_map = {
            "1": "surpised",
            "2": "fearful",
            "3": "disgusted",
            "4": "happy",
            "5": "sad",
            "6": "angry",
            "7": "neutral",
        }
    else:
        emotion_map = {
            ("anger", "disgust", "fear", "sadness", "sad", "2", "3", "5", "6"): 0,
            ("happy", "surprise", "4", "1"): 1,
            ("contempt", "neutral", "7"): 2,
        }

    path = "datasets/image/RAF"
    for firstdir in os.listdir(path):
        first_path = os.path.join(path, firstdir)
        for secdir in os.listdir(first_path):
            sec_path = os.path.join(first_path, secdir)
            for file in os.listdir(sec_path):
                img = cv2.imread(os.path.join(sec_path, file))
                if landmark == None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if len(X) == 0:
                        if not os.path.exists("outputs/image/process/"):
                            os.makedirs("outputs/image/process/")
                        cv2.imwrite(f"outputs/image/process/RAF_ori.JPG", img)

                    if process == "blur":  # gaussian blur
                        img = cv2.GaussianBlur(img, (7, 7), 0)
                    elif process == "noise":  # salt & pepper
                        img = random_noise(img, mode="s&p")
                        img = (255 * img).astype(np.uint8)
                    elif process == "sobel":
                        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                        sobelx = cv2.convertScaleAbs(sobelx)
                        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                        sobely = cv2.convertScaleAbs(sobely)
                        img = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
                        # img = cv2.addWeighted(img, 1, sobel, 1, 0)
                    elif process == "equal":
                        img = cv2.equalizeHist(img)
                    elif process == "filter":
                        img = wiener(img, mysize=(15, 15))
                        img = np.clip(img, 0, 255).astype(np.uint8)

                    X.append(img)  # grayscale
                    if len(X) == 1:
                        cv2.imwrite(f"outputs/image/process/RAF_{process}.JPG", img)
                    if split != None:
                        for k, i in enumerate(emotion_map.keys()):
                            if secdir in i:  # tuple
                                emotion = emotion_map[i]
                        y.append(emotion)
                    else:
                        category.append(category_map[secdir])
                        y.append(emotion_map[secdir])
                else:  # MLP
                    write = len(X) if len(X) <= 5 else None
                    face_landmarks = get_face_landmarks(
                        img, landmark=landmark, write=write, dataset="RAF"
                    )
                    if landmark == "xyz":
                        if len(face_landmarks) == 1404:  # 468 x 3 = 1404
                            X.append(face_landmarks)
                            y.append(emotion_map[secdir])  # anger
                    else:
                        if len(face_landmarks) != 0:
                            X.append(face_landmarks)  # 136 for 68
                            y.append(emotion_map[secdir])  # anger

    X, y = shuffle(X, y, random_state=42)  # shuffle
    num_classes = len(set(y))
    print(len(X))
    visual4label("image", "RAF", category)
    if landmark == None:
        n, h, w = np.array(X).shape  # expected: 48x48x1, (35887, 48, 48, 3)
        # n, h, w, d = np.array(X).shape  # expected: 48x48x1, (35887, 48, 48, 3)
        print(n, h, w)
        if method == "MLP":
            X = np.array(X).reshape((n, h * w)).tolist()
            h = h * w
    else:
        h = np.array(X).shape[1]

    if split == None:
        X_train, X_left, ytrain, yleft = train_test_split(
            np.array(X),
            y,
            test_size=0.4,
            random_state=9,
        )  # 3:2

        X_val, X_test, yval, ytest = train_test_split(
            X_left, yleft, test_size=0.5, random_state=9
        )  # 1:1
    else:
        random.seed(123)
        test_index = random.sample(
            [i for i in range(np.array(X).shape[0])], 200  # 200 fixed for testing size
        )  # 1440, 40
        left_index = [i for i in range(np.array(X).shape[0]) if i not in test_index]
        X_test = np.array(X)[test_index, :, :]
        ytest = np.array(y)[test_index].tolist()
        X_left = np.array(X)[left_index, :, :]
        yleft = np.array(y)[left_index].tolist()

        # train/val
        random.seed(123)
        train_index = random.sample(
            [i for i in range(np.array(X_left).shape[0])], int(split * 200)
        )  # train + val
        X_train, X_val, ytrain, yval = train_test_split(
            X_left[train_index, :, :],
            np.array(yleft)[train_index].tolist(),
            test_size=0.5,
            random_state=9,
        )  # 1:1 for train : val

    # if landmark == False:
    #     # np.array() format for X now
    #     return X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes  # shape 48
    # if landmark == True:
    #     np.savetxt(
    #         "outputs/image/landmark_RAF.txt",
    #         np.asarray(X),
    #     )

    return X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes


def load_split_corpus_size_image(
    method,
    corpus=None,
    split=None,  # split is the size of testing set within the mixture
    process=None,
):  # cc: mix, corpus: with only one string as the testing set

    # 900+300, 900 train 300 val, 300 test
    # 900/5=180, 300/5=60, 1200/5=240
    # 300
    # datasets = ["RAVDESS", "TESS", "SAVEE", "CREMA-D", "EmoDB"]
    emotion_map = {
        ("anger", "disgust", "fear", "sadness", "sad", "2", "3", "5", "6"): 0,
        ("happy", "surprise", "4", "1"): 1,
        ("contempt", "neutral", "7"): 2,
    }

    X_train, ytrain, X_val, yval, X_test, ytest = [], [], [], [], [], []

    for index, cor in enumerate(corpus):
        x, y = [], []  # for each dataset
        if cor == "CK":
            for dirname, _, filenames in os.walk("datasets/image/CK"):
                for filename in filenames:
                    img = cv2.imread(os.path.join(dirname, filename))
                    label = dirname.split("/")[-1]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    if index == 0:  # only train
                        if process == "blur":  # gaussian blur
                            img = cv2.GaussianBlur(img, (7, 7), 0)
                        elif process == "noise":  # salt & pepper
                            img = random_noise(img, mode="s&p")
                            img = (255 * img).astype(np.uint8)
                        elif process == "sobel":
                            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                            sobelx = cv2.convertScaleAbs(sobelx)
                            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                            sobely = cv2.convertScaleAbs(sobely)
                            img = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
                            # img = cv2.addWeighted(img, 1, sobel, 1, 0)
                        elif process == "equal":
                            img = cv2.equalizeHist(img)
                        elif process == "filter":
                            img = wiener(img, mysize=(15, 15))
                            img = np.clip(img, 0, 255).astype(np.uint8)

                        elif (
                            process == "assi"
                        ):  # which means to assimilate both as common image
                            img = cv2.equalizeHist(img)
                            img = cv2.GaussianBlur(img, (3, 3), 0)
                            # sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                            # sobelx = cv2.convertScaleAbs(sobelx)
                            # sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                            # sobely = cv2.convertScaleAbs(sobely)
                            # sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
                            # img = cv2.addWeighted(img, 1, sobel, 1, 0)
                    if index == 1:
                        if process == "assi":
                            img = cv2.equalizeHist(img)
                            img = cv2.GaussianBlur(img, (3, 3), 0)
                            # sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                            # sobelx = cv2.convertScaleAbs(sobelx)
                            # sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                            # sobely = cv2.convertScaleAbs(sobely)
                            # sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
                            # img = cv2.addWeighted(img, 1, sobel, 1, 0)

                    x.append(img)  # grayscale

                    for k, i in enumerate(emotion_map.keys()):
                        if label in i:  # tuple
                            emotion = emotion_map[i]
                    y.append(emotion)
        else:
            path = "datasets/image/RAF" if cor == "RAF" else "datasets/image/FER"
            for firstdir in os.listdir(path):
                first_path = os.path.join(path, firstdir)
                for secdir in os.listdir(first_path):
                    sec_path = os.path.join(first_path, secdir)
                    for file in os.listdir(sec_path):
                        img = cv2.imread(os.path.join(sec_path, file))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        if cor == "RAF":
                            img = cv2.resize(img, (48, 48))
                        if index == 0:  # only
                            if process == "blur":  # gaussian blur
                                img = cv2.GaussianBlur(img, (7, 7), 0)
                            elif process == "noise":  # salt & pepper
                                img = random_noise(img, mode="s&p")
                                img = (255 * img).astype(np.uint8)
                            elif process == "sobel":
                                sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                                sobelx = cv2.convertScaleAbs(sobelx)
                                sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                                sobely = cv2.convertScaleAbs(sobely)
                                img = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
                                # img = cv2.addWeighted(img, 1, sobel, 1, 0)
                            elif process == "equal":
                                img = cv2.equalizeHist(img)
                            elif process == "filter":
                                img = wiener(img, mysize=(15, 15))
                                img = np.clip(img, 0, 255).astype(np.uint8)
                            elif (
                                process == "assi"
                            ):  # which means to assimilate both as common image
                                img = cv2.equalizeHist(img)
                                img = cv2.GaussianBlur(img, (3, 3), 0)
                                # sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                                # sobelx = cv2.convertScaleAbs(sobelx)
                                # sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                                # sobely = cv2.convertScaleAbs(sobely)
                                # sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
                                # img = cv2.addWeighted(img, 1, sobel, 1, 0)

                        if index == 1:
                            if process == "assi":
                                img = cv2.equalizeHist(img)
                                img = cv2.GaussianBlur(img, (3, 3), 0)
                                # sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                                # sobelx = cv2.convertScaleAbs(sobelx)
                                # sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                                # sobely = cv2.convertScaleAbs(sobely)
                                # sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
                                # img = cv2.addWeighted(img, 1, sobel, 1, 0)

                        x.append(img)  # grayscale
                        for k, i in enumerate(emotion_map.keys()):
                            if secdir in i:  # tuple
                                emotion = emotion_map[i]
                        y.append(emotion)

        # first reverse for each dataset, then get
        x, y = shuffle(x, y, random_state=42)  # shuffle
        num_classes = len(set(y))
        n, h, w = np.array(x).shape

        random.seed(123)
        if index == 0:  # train
            sample_index = random.sample(
                [i for i in range(np.array(x).shape[0])], int(1000 * (1 - split))
            )  # 1440, 40
        elif index == 1:  # test, split is test size
            if (split == 0.8) and (cor == "CK"):  # CK is tested
                sample_index = random.sample(
                    [i for i in range(np.array(x).shape[0])], int(len(x) - 200)
                )  # 1440, 40
            else:
                sample_index = random.sample(
                    [i for i in range(np.array(x).shape[0])], int(1000 * split)
                )  # 1440, 40

        # 240, 40
        X_train = (
            np.array(x)[sample_index, :, :]
            if len(X_train) == 0
            else np.concatenate((X_train, np.array(x)[sample_index, :, :]), axis=0)
        )
        ytrain = (
            np.array(y)[sample_index].tolist()
            if len(ytrain) == 0
            else (ytrain + np.array(y)[sample_index].tolist())
        )

        if cor == corpus[1]:  # testing set
            # if dataset == corpus[0]:  # testing set
            left_index = [
                i for i in range(np.array(x).shape[0]) if i not in sample_index
            ]
            random.seed(123)
            test_index = random.sample(left_index, 200)  # fixed 200 for testing set
            X_test = np.array(x)[test_index, :, :]  # 300,40
            ytest = np.array(y)[test_index].tolist()

    X_train, X_val, ytrain, yval = train_test_split(  # 2800, 1680, 1120
        X_train, ytrain, test_size=0.5, random_state=9
    )  # 3:1  # 1200, 40  # 300,4

    return X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes


def load_mix3_corpus_image(
    method,
    corpus=None,
):  # cc: mix, corpus: with only one string as the testing set

    # 900+300, 900 train 300 val, 300 test
    # 900/5=180, 300/5=60, 1200/5=240
    # 300
    datasets = ["CK", "FER", "RAF"]
    emotion_map = {
        ("anger", "disgust", "fear", "sadness", "sad", "2", "3", "5", "6"): 0,
        ("happy", "surprise", "4", "1"): 1,
        ("contempt", "neutral", "7"): 2,
    }

    X_train, ytrain, X_val, yval, X_test, ytest = [], [], [], [], [], []

    for index, dataset in enumerate(datasets):
        x, y = [], []  # for each dataset
        if dataset == "CK":
            for dirname, _, filenames in os.walk("datasets/image/CK"):
                for filename in filenames:
                    img = cv2.imread(os.path.join(dirname, filename))
                    label = dirname.split("/")[-1]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    x.append(img)  # grayscale
                    for k, i in enumerate(emotion_map.keys()):
                        if label in i:  # tuple
                            emotion = emotion_map[i]
                    y.append(emotion)
        else:
            path = "datasets/image/RAF" if dataset == "RAF" else "datasets/image/FER"
            for firstdir in os.listdir(path):
                first_path = os.path.join(path, firstdir)
                for secdir in os.listdir(first_path):
                    sec_path = os.path.join(first_path, secdir)
                    for file in os.listdir(sec_path):
                        img = cv2.imread(os.path.join(sec_path, file))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        if dataset == "RAF":
                            img = cv2.resize(img, (48, 48))
                        x.append(img)  # grayscale
                        for k, i in enumerate(emotion_map.keys()):
                            if secdir in i:  # tuple
                                emotion = emotion_map[i]
                        y.append(emotion)

        # first reverse for each dataset, then get
        x, y = shuffle(x, y, random_state=42)  # shuffle
        num_classes = len(set(y))
        n, h, w = np.array(x).shape

        random.seed(123)
        sample_index = random.sample([i for i in range(np.array(x).shape[0])], 200)

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

        # if (dataset == corpus[0]) or (corpus[0] == "eNTERFACE"):  # testing set
        if dataset == corpus[0]:  # testing set
            left_index = [
                i for i in range(np.array(x).shape[0]) if i not in sample_index
            ]
            random.seed(123)
            test_index = random.sample(left_index, 200)  # 200:200:200  -- 200
            X_test = np.array(x)[test_index, :]  # 300,40
            ytest = np.array(y)[test_index].tolist()

    X_train, X_val, ytrain, yval = train_test_split(  # 2800, 1680, 1120
        X_train, ytrain, test_size=0.5, random_state=9
    )  # 3:1  # 1200, 40  # 300,4

    return X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes


def load_patches(X, dataset):
    """
    description: This method is used for ViT to get image patches.
    param {*} X: input images
    return {*}: patches
    """
    if dataset in ["CK", "FER"]:
        X_patches = patchify(X[0], (8, 8), 8)  # 100 for 10x10x3, step
        X_save = np.reshape(X_patches, (36, 8, 8, 1))
        if not os.path.exists("outputs/image/patches/"):
            os.makedirs("outputs/image/patches/")
        for index, patch in enumerate(X_save):
            cv2.imwrite(f"outputs/image/patches/{dataset}_{index}.JPG", patch)

        X_patches = np.reshape(X_patches, (1, 36, 8 * 8 * 1))
        for x in X[1:]:
            patches = patchify(x, (8, 8), 8)
            patches = np.reshape(patches, (1, 36, 8 * 8 * 1))
            X_patches = np.concatenate(((np.array(X_patches)), patches), axis=0)

    elif dataset == "RAF":
        X_patches = patchify(
            X[0], (25, 25), 25
        )  # 100 for 10x10x3, step  100x100 -- 25x25 16
        X_save = np.reshape(X_patches, (16, 25, 25, 1))
        if not os.path.exists("outputs/image/patches/"):
            os.makedirs("outputs/image/patches/")
        for index, patch in enumerate(X_save):
            cv2.imwrite(f"outputs/image/patches/{dataset}_{index}.JPG", patch)

        X_patches = np.reshape(X_patches, (1, 16, 25 * 25 * 1))
        for x in X[1:]:
            patches = patchify(x, (25, 25), 25)
            patches = np.reshape(patches, (1, 16, 25 * 25 * 1))
            X_patches = np.concatenate(((np.array(X_patches)), patches), axis=0)

    return X_patches.astype(np.int64)


# def sample_ViT(X, y, n):
#     """
#     description: Due to large size of patches, this method is used for sampling from patches to
#     reduce dimensionality.
#     param {*} X: input data
#     param {*} y: input label
#     param {*} n: class name
#     return {*}: sampled data and label
#     """
#     ViT_index, ViT_label = [], []
#     for i in range(12):
#         class_index = [index for index, j in enumerate(y) if label_map[n[index]] == i]
#         ViT_index += random.sample(class_index, 100)
#     ViT_sample = [i for index, i in enumerate(X) if index in ViT_index]
#     ViT_label = [i for index, i in enumerate(y) if index in ViT_index]
#     return ViT_sample, ViT_label


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
    sr=16000,
    landmark=None,
    split=None,
    process=None,
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
                    sr=sr,
                    split=split,
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
                    sr=sr,
                    split=split,
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
                    sr=sr,
                    split=split,
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
                    sr=sr,
                    split=split,
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
                    sr=sr,
                    split=split,
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
                    sr=sr,
                    split=split,
                )
            elif dataset == "AESDD":
                X_train, ytrain, X_val, yval, X_test, ytest, length = load_AESDD(
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
                    sr=sr,
                    split=split,
                )
        else:
            if split == None:
                if cc == "mix":
                    # # use for original design of mixture of 5
                    # X_train, ytrain, X_val, yval, X_test, ytest, length = (
                    #     load_mix_corpus(
                    #         method,
                    #         features,
                    #         n_mfcc,
                    #         n_mels,
                    #         scaled,
                    #         max_length,
                    #         reverse,
                    #         noise,
                    #         denoise,
                    #         window=None,
                    #         corpus=corpus,
                    #         sr=sr,
                    #     )
                    # )

                    # used for new design of ranging split case of 3
                    X_train, ytrain, X_val, yval, X_test, ytest, length = (
                        load_mix3_corpus(
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
                            sr=sr,
                        )
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
                        sr=sr,
                    )
                elif cc == "cross":
                    X_train, ytrain, X_val, yval, X_test, ytest, length = (
                        load_cross_corpus(
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
                            sr=sr,
                        )
                    )
            else:
                # X_train, ytrain, X_val, yval, X_test, ytest, length = load_split_corpus(
                #     method,
                #     features,
                #     n_mfcc,
                #     n_mels,
                #     scaled,
                #     max_length,
                #     reverse,
                #     noise,
                #     denoise,
                #     window=None,
                #     corpus=corpus,
                #     sr=sr,
                #     split=split,
                # )
                X_train, ytrain, X_val, yval, X_test, ytest, length = (
                    load_split_corpus_size(
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
                        sr=sr,
                        split=split,
                    )
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

    elif task == "image":
        if corpus == None:
            if dataset == "CK":
                X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes = load_CK(
                    method, landmark, split, process
                )
            elif dataset == "FER":
                X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes = load_FER(
                    method, landmark, split, process
                )
            elif dataset == "RAF":
                X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes = load_RAF(
                    method, landmark, split, process
                )
            if method == "ViT":
                # Xtrain, ytrain = sample_ViT(Xtrain, ytrain, ntrain)
                # Xval, yval = sample_ViT(Xval, yval, nval)
                # Xtest, ytest = sample_ViT(Xtest, ytest, ntest)

                X_train = load_patches(X_train, dataset)
                X_val = load_patches(X_val, dataset)
                X_test = load_patches(X_test, dataset)
        else:  # corpus != None
            if split == None:
                if cc == "mix":
                    X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes = (
                        load_mix3_corpus_image(method, corpus)
                    )
            else:
                X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes = (
                    load_split_corpus_size_image(method, corpus, split, process)
                )

        if method in ["CNN", "Inception", "MLP"]:
            return X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes
        elif method in ["ViT"]:
            return X_train, ytrain, X_val, yval, X_test, ytest, num_classes


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
    cv=False,
    h=None,
    landmark=None,
):
    if task == "speech":
        if method in ["SVM", "DT", "RF", "NB", "KNN"]:
            model = SpeechBase(method)
        elif method == "MLP":
            model = speech_MLP(
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
                cv=cv,
            )
        elif method == "CNN":
            model = speech_CNN(
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
                cv=cv,
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
                cv=cv,
            )
        elif method == "GMM":
            model = GMM(task, method, features, cc, dataset, num_classes)
        elif method == "KMeans":
            model = KMeans(task, method, features, cc, dataset, num_classes)
        elif method == "wav2vec":
            # model = Wav2Vec(
            #     task,
            #     method,
            #     features,
            #     cc,
            #     num_classes,
            #     dataset,
            #     max_length,
            #     epochs=epochs,
            #     lr=lr,
            #     batch_size=batch_size,
            # )
            pass
    elif task == "image":
        if method == "MLP":
            model = image_MLP(
                task,
                method,
                cc,
                h,
                num_classes,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                landmark=landmark,
            )
        elif method == "CNN":
            model = image_CNN(
                task,
                method,
                cc,
                h,
                num_classes,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )
        elif method == "Inception":
            model = Inception(
                task,
                method,
                cc,
                h,
                num_classes,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )
        elif method == "ViT":
            model = ViT(
                task,
                dataset,
                method,
                cc,
                num_classes,
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
    ytune_train=None,
    tune_train_pred=None,
    ytune_val=None,
    tune_val_pred=None,
    corpus=None,
    split=None,
    landmark=None,
    process=None,
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

    if task == "speech":
        if (corpus != None) and (split != None):
            fig.savefig(
                f"outputs/{task}/confusion_matrix/{method}_{features}_cross_{corpus[0]}_{corpus[1]}_{split}.png"
            )
        elif (corpus == None) and (split != None):
            fig.savefig(
                f"outputs/{task}/confusion_matrix/{method}_{features}_{cc}_{dataset}_{split}.png"
            )
        else:
            fig.savefig(
                f"outputs/{task}/confusion_matrix/{method}_{features}_{cc}_{dataset}.png"
            )
    elif task == "image":
        if landmark == None:
            if (corpus != None) and (split != None):  # split cross
                if process == None:
                    fig.savefig(
                        f"outputs/{task}/confusion_matrix/{method}_raw_cross_{corpus[0]}_{corpus[1]}_{split}.png"
                    )
                else:
                    fig.savefig(
                        f"outputs/{task}/confusion_matrix/{method}_raw_cross_{corpus[0]}_{corpus[1]}_{split}_{process}.png"
                    )
            elif (corpus == None) and (split != None):  # split single
                fig.savefig(
                    f"outputs/{task}/confusion_matrix/{method}_raw_{cc}_{dataset}_{split}.png"
                )
            elif (corpus != None) and (split == None):  # mix3
                fig.savefig(
                    f"outputs/{task}/confusion_matrix/{method}_raw_{cc}_{corpus[0]}.png"
                )
            else:  # general
                if process == None:
                    fig.savefig(
                        f"outputs/{task}/confusion_matrix/{method}_raw_{cc}_{dataset}.png"
                    )
                else:
                    fig.savefig(
                        f"outputs/{task}/confusion_matrix/{method}_raw_{cc}_{dataset}_{process}.png"
                    )
        else:
            fig.savefig(
                f"outputs/{task}/confusion_matrix/{method}_landmarks_{cc}_{dataset}.png"
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


def get_imbalance(task, y, pred):
    # mcc  [-1,1]
    # geometric mean  [0,1]
    # discriminant power
    # balanced accuracy  [0,1]
    # kappa [-1,1]
    # youden's index
    # likelihoods
    # auroc
    # classification report imbalance & IBA
    sen = sensitivity_score(np.array(y).astype(int), pred.astype(int), average="macro")
    spe = specificity_score(np.array(y).astype(int), pred.astype(int), average="macro")
    X = sen / (1 - sen)
    Y = spe / (1 - spe)
    LR_pos = sen / (1 - spe)
    LR_neg = (1 - sen) / spe
    result = {
        "mcc": round(matthews_corrcoef(np.array(y).astype(int), pred.astype(int)), 4),
        "g-mean": round(
            geometric_mean_score(
                np.array(y).astype(int), pred.astype(int), average="macro"
            )
            * 100,
            4,
        ),
        "discriminant-power": round(
            (math.sqrt(3) * (math.log(X) + math.log(Y))) / math.pi, 4
        ),
        "balanced-accuracy": round(
            balanced_accuracy_score(np.array(y).astype(int), pred.astype(int)) * 100,
            4,
        ),
        "kappa": round(
            cohen_kappa_score(np.array(y).astype(int), pred.astype(int)),
            4,
        ),
        "yoden": round(
            spe + sen - 1,
            4,
        ),
        # "likelihood" : class_likelihood_ratios(np.array(y).astype(int), pred.astype(int)),
        "likelihood": [round(LR_pos, 4), round(LR_neg, 4)],
    }
    print(classification_report_imbalanced(np.array(y).astype(int), pred.astype(int)))

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


def visual4corr(feature, dataset):
    # print(feature)
    # print(np.array(feature).shape)
    fea = []
    for i in feature:
        fea.append(i.tolist())
    corr_matrix = np.corrcoef(fea, rowvar=False)
    # Finally if we use the option rowvar=False,
    # the columns are now being treated as the variables and
    # we will find the column-wise Pearson correlation
    # coefficients between variables in xarr and yarr.

    # print(corr_matrix)
    # print(np.array(corr_matrix).shape)
    plt.figure(figsize=(10, 8))
    hm = sns.heatmap(data=corr_matrix, annot=False)
    plt.title(f"Correlation matrix of {dataset}")
    if not os.path.exists(f"outputs/speech/corr_matrix/"):
        os.makedirs(f"outputs/speech/corr_matrix/")
    plt.savefig(f"outputs/speech/corr_matrix/{dataset}.png")
    plt.close()

    # txt
    filename = f"outputs/speech/corr_matrix/corr.csv"
    if not os.path.isfile(filename):
        with open(filename, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(dataset)
            for i in corr_matrix:
                writer.writerow(i)
    else:
        with open(filename, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow("\n")
            writer.writerow(dataset)
            for i in corr_matrix:
                writer.writerow(i)


def visual4corr2(dataset, corr, corr_emo):
    for index_i, i in enumerate(corr):
        for index_j, j in enumerate(corr):
            if index_i != index_j:
                sample_rate, data1 = wavfile.read(i)
                sample_rate, data2 = wavfile.read(j)
                if dataset == "eNTERFACE":
                    data1 = data1[:, 0]
                    data2 = data2[:, 0]
                data1 = data1.astype(float)
                data2 = data2.astype(float)
                data1 /= np.max(np.abs(data1))
                data2 /= np.max(np.abs(data2))

                correlation = np.correlate(data1, data2, mode="full")
                lags = np.arange(-len(data1) + 1, len(data2))

                # Plot the cross-correlation
                plt.figure(figsize=(10, 5))
                fig, (ax1, ax2, ax_corr) = plt.subplots(3, 1, sharex=True)
                ax1.plot(data1)
                ax2.plot(data2)
                plt.title(
                    f"Cross-Correlation of {corr_emo[index_i]} and {corr_emo[index_j]} in {dataset}"
                )
                ax_corr.plot(lags, correlation)
                plt.xlabel("Lag")
                plt.ylabel("Correlation")
                if not os.path.exists(f"outputs/speech/corr_signal/"):
                    os.makedirs(f"outputs/speech/corr_signal/")
                fig.savefig(
                    f"outputs/speech/corr_signal/{dataset}_{corr_emo[index_i]}_{corr_emo[index_j]}.png"
                )
                plt.close()


def visual4corr3(dataset, file1, file2, emotion):
    sample_rate, data1 = wavfile.read(file1)
    sample_rate, data2 = wavfile.read(file2)
    if dataset == "eNTERFACE":
        data1 = data1[:, 0]
        data2 = data2[:, 0]
    data1 = data1.astype(float)
    data2 = data2.astype(float)
    data1 /= np.max(np.abs(data1))
    data2 /= np.max(np.abs(data2))

    cross_correlation = np.correlate(
        data1, data2, mode="full"
    )  # the amplitude is sum of correlation multiplication
    cross_lags = np.arange(-len(data1) + 1, len(data2))
    auto_correlation1 = np.correlate(data1, data1, mode="full")
    auto_lags1 = np.arange(-len(data1) + 1, len(data1))
    auto_correlation2 = np.correlate(data2, data2, mode="full")
    auto_lags2 = np.arange(-len(data2) + 1, len(data2))

    print(
        emotion,
        np.max(auto_correlation1),
        np.max(auto_correlation2),
        np.max(cross_correlation),
    )

    # Plot the cross-correlation
    plt.figure(figsize=(10, 16))
    fig, (ax1, ax2, ax3, ax4, ax_corr) = plt.subplots(5, 1, sharex=True)
    ax1.plot(data1)
    ax2.plot(data2)
    ax3.plot(auto_lags1, auto_correlation1)
    ax4.plot(auto_lags2, auto_correlation2)
    # plt.title(f"Cross-Correlation of {emotion} in {dataset}")
    ax_corr.plot(cross_lags, cross_correlation)
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    if not os.path.exists(f"outputs/speech/corr_signal3/"):
        os.makedirs(f"outputs/speech/corr_signal3/")
    fig.savefig(f"outputs/speech/corr_signal3/{dataset}_{emotion}.png")
    plt.close()

    # most artificially to most real
    # (RAVDESS/TESS), (SAVEE, EmoDB, AESDD, CREMA-D), eNTERFACE05


def visual4corr4(dataset, file1, file2, emotion):
    sample_rate, data1 = wavfile.read(file1)
    sample_rate, data2 = wavfile.read(file2)
    # print(sample_rate)
    if dataset == "eNTERFACE":
        data1 = data1[:, 0]
        data2 = data2[:, 0]

    # low pass filter
    # Parameters
    fs = 1000.0  # Sample rate (Hz)  # 500
    # t = np.arange(0, 1.0, 1.0/fs)  # Time vector
    cutoff = 5.0  # Desired cutoff frequency of the filter (Hz)  # 10
    order = 5  # Order of the filter
    # small cutoff can have a obvious difference between two signal
    # large cutoff will still have similar signal
    # frequency cannot be too small

    # Generate the signal
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    fdata1 = lfilter(b, a, data1)
    fdata2 = lfilter(b, a, data2)

    fdata1 = fdata1.astype(float)
    fdata2 = fdata2.astype(float)
    fdata1 /= np.max(np.abs(fdata1))
    fdata2 /= np.max(np.abs(fdata2))

    cross_correlation = np.correlate(
        fdata1, fdata2, mode="full"
    )  # the amplitude is sum of correlation multiplication
    cross_lags = np.arange(-len(fdata1) + 1, len(fdata2))
    auto_correlation1 = np.correlate(fdata1, fdata1, mode="full")
    auto_lags1 = np.arange(-len(fdata1) + 1, len(fdata1))
    auto_correlation2 = np.correlate(fdata2, fdata2, mode="full")
    auto_lags2 = np.arange(-len(fdata2) + 1, len(fdata2))

    # Plot the cross-correlation
    plt.figure(figsize=(10, 16))
    fig, (ax1, ax2, ax3, ax4, ax_corr) = plt.subplots(5, 1, sharex=True)
    ax1.plot(fdata1)
    ax2.plot(fdata2)
    ax3.plot(auto_lags1, auto_correlation1)
    ax4.plot(auto_lags2, auto_correlation2)
    # plt.title(f"Cross-Correlation of {emotion} in {dataset}")
    ax_corr.plot(cross_lags, cross_correlation)
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    if not os.path.exists(f"outputs/speech/corr_signal_filter/"):
        os.makedirs(f"outputs/speech/corr_signal_filter/")
    fig.savefig(f"outputs/speech/corr_signal_filter/{dataset}_{emotion}.png")
    plt.close()

    # spectrogram of cross correlation
    melspectrogram = librosa.feature.melspectrogram(
        y=cross_correlation, sr=sample_rate, n_mels=128, fmax=8000
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
    if not os.path.exists(f"outputs/speech/corr_signal_filter/"):
        os.makedirs(f"outputs/speech/corr_signal_filter/")
    plt.savefig(f"outputs/speech/corr_signal_filter/{dataset}_{emotion}_mels.png")
    plt.close()

    # most artificially to most real
    # (RAVDESS/TESS), (SAVEE, EmoDB, AESDD, CREMA-D), eNTERFACE0

    # moving average
    window_size = 100  # Window size for moving average
    ma_data1 = np.convolve(data1, np.ones(window_size) / window_size, mode="valid")
    ma_data2 = np.convolve(data2, np.ones(window_size) / window_size, mode="valid")
    # the larger the window size, the better

    ma_data1 = ma_data1.astype(float)
    ma_data2 = ma_data2.astype(float)
    ma_data1 /= np.max(np.abs(ma_data1))
    ma_data2 /= np.max(np.abs(ma_data2))

    cross_correlation = np.correlate(
        ma_data1, ma_data2, mode="full"
    )  # the amplitude is sum of correlation multiplication
    cross_lags = np.arange(-len(ma_data1) + 1, len(ma_data2))
    auto_correlation1 = np.correlate(ma_data1, ma_data1, mode="full")
    auto_lags1 = np.arange(-len(ma_data1) + 1, len(ma_data1))
    auto_correlation2 = np.correlate(ma_data2, ma_data2, mode="full")
    auto_lags2 = np.arange(-len(ma_data2) + 1, len(ma_data2))

    # Plot the cross-correlation
    plt.figure(figsize=(10, 16))
    fig, (ax1, ax2, ax3, ax4, ax_corr) = plt.subplots(5, 1, sharex=True)
    ax1.plot(ma_data1)
    ax2.plot(ma_data2)
    ax3.plot(auto_lags1, auto_correlation1)
    ax4.plot(auto_lags2, auto_correlation2)
    # plt.title(f"Cross-Correlation of {emotion} in {dataset}")
    ax_corr.plot(cross_lags, cross_correlation)
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    if not os.path.exists(f"outputs/speech/corr_signal_ma/"):
        os.makedirs(f"outputs/speech/corr_signal_ma/")
    fig.savefig(f"outputs/speech/corr_signal_ma/{dataset}_{emotion}.png")
    plt.close()


def signal_similarity(dataset, file1, file2, emotion):
    sample_rate, data1 = wavfile.read(file1)
    sample_rate, data2 = wavfile.read(file2)
    print(sample_rate)
    # if dataset == "eNTERFACE":
    #     data1 = data1[:, 0]
    #     data2 = data2[:, 0]

    weights = {
        "zcr_similarity": 0.2,
        "rhythm_similarity": 0.2,
        "chroma_similarity": 0.2,
        "energy_envelope_similarity": 0.1,
        "spectral_contrast_similarity": 0.1,
        "perceptual_similarity": 0.2,
    }

    # Create an instance of the AudioSimilarity class

    audio_similarity = AudioSimilarity(
        file1, file2, sample_rate, weights, verbose=True, sample_size=1
    )
    similarity = {
        # "zcr":audio_similarity.zcr_similarity(),
        # "rhythm":audio_similarity.rhythm_similarity(),
        # "chroma":audio_similarity.chroma_similarity(),
        # "energy":audio_similarity.energy_envelope_similarity(),
        # "spectral":audio_similarity.spectral_contrast_similarity(),
        # "perceptual":audio_similarity.perceptual_similarity(),
        "score": audio_similarity.stent_weighted_audio_similarity(metrics="all")
    }
    print(f"Stent Weighted Audio Similarity of {emotion}: {similarity}")

    # plt.figure(figsize=(10, 16))
    audio_similarity.plot(
        metrics=None,
        option="all",
        figsize=(10, 7),
        color1="red",
        color2="green",
        dpi=600,
        savefig=False,
        fontsize=6,
        label_fontsize=8,
        title_fontsize=14,
        alpha=0.5,
        title="Audio Similarity Metrics",
    )
    if not os.path.exists(f"outputs/speech/signal_similarity/"):
        os.makedirs(f"outputs/speech/signal_similarity/")
    plt.savefig(f"outputs/speech/signal_similarity/{dataset}_{emotion}.png")
    plt.close()

    return similarity

    # # dwt, too long
    # # Generate dummy data for the time series
    # np.random.seed(0)
    # time_series_a = np.cumsum(data1)
    # time_series_b = np.cumsum(data2)

    # # Calculate DTW distance and obtain the warping paths (no need for the C library)
    # distance, paths = dtw.warping_paths(time_series_a, time_series_b, use_c=False)
    # best_path = dtw.best_path(paths)

    # dtw_score = distance / len(best_path)

    # plt.figure(figsize=(12, 8))

    # # Original Time Series Plot
    # ax1 = plt.subplot2grid((2, 2), (0, 0))
    # ax1.plot(time_series_a, label='Series A', color='blue')
    # ax1.plot(time_series_b, label='Series B', linestyle='--',color='orange')
    # ax1.set_title('Original Time Series')
    # ax1.legend()

    # # Shortest Path Plot (Cost Matrix with the path)
    # # In this example, only the path is plotted, not the entire cost matrix.

    # ax2 = plt.subplot2grid((2, 2), (0, 1))
    # ax2.plot(np.array(best_path)[:, 0], np.array(best_path)[:, 1], 'green', marker='o', linestyle='-')
    # ax2.set_title('Shortest Path (Best Path)')
    # ax2.set_xlabel('Series A')
    # ax2.set_ylabel('Series B')
    # ax2.grid(True)

    # # Point-to-Point Comparison Plot
    # ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    # ax3.plot(time_series_a, label='Series A', color='blue', marker='o')
    # ax3.plot(time_series_b, label='Series B', color='orange', marker='x', linestyle='--')
    # for a, b in best_path:
    #     ax3.plot([a, b], [time_series_a[a], time_series_b[b]], color='grey', linestyle='-', linewidth=1, alpha = 0.5)
    # ax3.set_title('Point-to-Point Comparison After DTW Alignment')
    # ax3.legend()

    # plt.tight_layout()
    # if not os.path.exists(f"outputs/speech/signal_similarity/"):
    #     os.makedirs(f"outputs/speech/signal_similarity/")
    # plt.savefig(f"outputs/speech/signal_similarity/{dataset}_{emotion}_dtw.png")
    # plt.close()

    # # Create a DataFrame to display the similarity score and correlation coefficient
    # print(f"DTW Similarity Score: {dtw_score}")

    # lower score, greater similarity


def visual4corrMAV(dataset, file1, file2, emotion):
    sample_rate, data1 = wavfile.read(file1)
    sample_rate, data2 = wavfile.read(file2)
    # print(sample_rate)
    if dataset == "eNTERFACE":
        data1 = data1[:, 0]
        data2 = data2[:, 0]

    ori_data1 = data1
    ori_data2 = data2
    # first calculate for envelope absolute
    # Compute the analytic signal using the Hilbert transform
    data1 = signal.hilbert(data1)
    data1 = np.abs(data1)
    data2 = signal.hilbert(data2)
    data2 = np.abs(data2)

    # then moving average
    window_size = 1000  # Window size for moving average  # 1000, smaller window, better describe the shape
    ma_data1 = np.convolve(data1, np.ones(window_size) / window_size, mode="valid")
    ma_data2 = np.convolve(data2, np.ones(window_size) / window_size, mode="valid")
    # the larger the window size, the better

    # then low pass filter
    # low pass filter
    # Parameters
    fs = 1000.0  # Sample rate (Hz)  # 500
    # t = np.arange(0, 1.0, 1.0/fs)  # Time vector
    cutoff = 5.0  # Desired cutoff frequency of the filter (Hz)  # 10
    order = 5  # Order of the filter
    # small cutoff can have a obvious difference between two signal
    # large cutoff will still have similar signal
    # frequency cannot be too small

    # Generate the signal
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    fdata1 = lfilter(b, a, ma_data1)
    fdata2 = lfilter(b, a, ma_data2)

    fdata1 = fdata1.astype(float)
    fdata2 = fdata2.astype(float)
    fdata1 /= np.max(np.abs(fdata1))
    fdata2 /= np.max(np.abs(fdata2))

    cross_correlation = np.correlate(
        fdata1, fdata2, mode="full"
    )  # the amplitude is sum of correlation multiplication
    cross_lags = np.arange(-len(fdata1) + 1, len(fdata2))
    auto_correlation1 = np.correlate(fdata1, fdata1, mode="full")
    auto_lags1 = np.arange(-len(fdata1) + 1, len(fdata1))
    auto_correlation2 = np.correlate(fdata2, fdata2, mode="full")
    auto_lags2 = np.arange(-len(fdata2) + 1, len(fdata2))
    # print(
    #     emotion,
    #     np.max(auto_correlation1),
    #     np.max(auto_correlation2),
    #     np.max(cross_correlation),
    # )

    # Plot the cross-correlation
    plt.figure(figsize=(10, 16))
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax_corr) = plt.subplots(7, 1, sharex=True)
    ax1.plot(ori_data1)
    ax2.plot(ori_data2)
    ax3.plot(fdata1)
    ax4.plot(fdata2)
    ax5.plot(auto_lags1, auto_correlation1)
    ax6.plot(auto_lags2, auto_correlation2)
    # plt.title(f"Cross-Correlation of {emotion} in {dataset}")
    ax_corr.plot(cross_lags, cross_correlation)
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    if not os.path.exists(f"outputs/speech/corr_signal_MAV/"):
        os.makedirs(f"outputs/speech/corr_signal_MAV/")
    fig.savefig(f"outputs/speech/corr_signal_MAV/{dataset}_{emotion}.png")
    plt.close()
