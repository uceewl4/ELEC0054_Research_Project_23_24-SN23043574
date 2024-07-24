# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/07/24 07:07:34
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0054: Research Project
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :  This file includes all utility functions for this project including data loading, 
            model loading, visualization and experimental setups.
"""

# here put the import lib
import os
import csv
import cv2
import math
import dlib
import glob
import pickle
import random
import librosa
import soundfile
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import mediapipe as mp
import tensorflow as tf
import noisereduce as nr
from sklearn import tree
from scipy.io import wavfile
import scipy.signal as signal
from patchify import patchify
from pydub import AudioSegment
from scipy.signal import wiener
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.utils import shuffle
import matplotlib.colors as mcolors
from skimage.util import random_noise
from scipy.interpolate import interp1d
from matplotlib.colors import Normalize
from scipy.signal import butter, lfilter
from sklearn.metrics import accuracy_score
from audio_similarity import AudioSimilarity
from sklearn.preprocessing import MinMaxScaler
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score

# from transformers import AutoFeatureExtractor
from sklearn.neural_network import MLPClassifier
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import StandardScaler
from imblearn.metrics import classification_report_imbalanced

from models.image.ViT import ViT
from models.speech.RNN import RNN
from models.speech.GMM import GMM
from models.speech.LSTM import LSTM
from models.speech.KMeans import KMeans
from models.speech.DBSCAN import DBSCAN
from models.speech.AlexNet import AlexNet
from models.image.Xception import Xception
from models.image.CNN import CNN as image_CNN
from models.image.MLP import MLP as image_MLP
from models.speech.CNN import CNN as speech_CNN
from models.speech.MLP import MLP as speech_MLP
from models.speech.baselines import Baselines as SpeechBase
from sklearn.metrics import (
    auc,
    roc_curve,
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    silhouette_score,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    class_likelihood_ratios,
)

random.seed(123)


def get_padding(data, max_length):
    """
    description: This function pads the audio length as consistent length for CNN scenario.
    param {*} data: audio
    param {*} max_length: length needed to be padded
    return {*}: padded audio
    """
    if max_length > data.shape[1]:  # pad
        pad_width = max_length - data.shape[1]
        data = np.pad(data, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:  # truncate
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
    """
    description: This function gets features for an individual audio.
    param {*} dataset: name of dataset
    param {*} method: name of model selected
    param {*} filename: path for audio file
    param {*} features: feature selected
    param {*} n_mfcc: number of mfcc features
    param {*} n_mels: number of mels features
    param {*} window: window selected
    param {*} sr: sampling rate
    return {*}: features and original audio loaded
    """
    with soundfile.SoundFile(filename) as sound_file:
        X = sound_file.read(dtype="float32")  # 121715,2 (eNTERFACE-05)
        if dataset == "eNTERFACE" and len(X.shape) == 2:
            X = X[:, 1]  # use first channel
        sample_rate = sound_file.samplerate
        # print(sample_rate)
        if sr != 16000:  # resample
            X = librosa.resample(X, orig_sr=sample_rate, target_sr=sr)

        # window
        if window != None:
            X = X[
                int(window[0] * sample_rate) : int(window[1] * sample_rate)
            ]  # 0, 3 -- 48000

        result = np.array([])

        if method in ["CNN", "AlexNet"]:
            mfccs = librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=n_mfcc
            )  # 40,122 length=122
            # print(mfccs.shape[1])
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
            )  # (40, 121715)  # 40
            stft = np.abs(librosa.stft(X))  # spectrum
            chroma = np.mean(
                librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0
            )  # (12, 121715)
            mel = np.mean(
                librosa.feature.melspectrogram(
                    y=X, sr=sample_rate, n_mels=n_mels, fmax=8000
                ).T,
                axis=0,
            )  # (128, 121715)

        # concate features involved
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

    return result, X


def print_features(data, features, n_mfcc, n_mels):
    """
    description: This function is used for print statistics for features of audio like min, max, mean, std, etc.
    param {*} data: features extracted
    param {*} features: name of feature selected
    param {*} n_mfcc: number of mfcc
    param {*} n_mels: number of mels
    """
    if features in ["all", "mfcc"]:
        # Check mfcc feature values
        mfcc_features = np.array(data)[:, :n_mfcc]
        mfcc_min = np.min(mfcc_features)
        mfcc_max = np.max(mfcc_features)
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
    """
    description: This function transforms features as standardization or min-max normalization.
    param {*} method: name of method
    param {*} x: features extracted
    param {*} features: name of feature selected
    param {*} n_mfcc: number of mfcc feature
    param {*} n_mels: number of mel feature
    param {*} scaled: min-max normalization or standardization
    return {*}: scaled features
    """
    print_features(x, features, n_mfcc, n_mels)  # print statistics
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
    """
    description: This function produces reversed audio.
    param {*} x: array of features
    param {*} y: array of labels
    param {*} audio: array of audio
    param {*} lengths: array of audio length
    param {*} path: array of audio path
    param {*} dataset: dataset name
    param {*} method: model name
    param {*} features: features selected
    param {*} n_mfcc: number of mfcc features
    param {*} n_mels: number of mel features
    param {*} max_length: max length for CNN
    param {*} sample: number of samples
    param {*} window: window selected
    param {*} sr: sampling rate
    return {*}: sampled reversed features, labels, audios and audio lengths
    """
    sample_index = random.sample(
        [i for i in range(np.array(x).shape[0])], sample
    )  # sampling audio for reverse
    for i in sample_index:
        sound = AudioSegment.from_file(path[i], format="wav")
        reversed_sound = sound.reverse()  # reverse

        # save the reversed audio
        name = path[i].split(".")[0].split("/")[-1]
        if not os.path.exists(f"datasets/speech/{dataset}_reverse/"):
            os.makedirs(f"datasets/speech/{dataset}_reverse/")
        reversed_sound.export(
            f"datasets/speech/{dataset}_reverse/{name}_reverse.wav", format="wav"
        )

        # get features for reverse audio
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
    """
    description: This function produces audio with noise.
    param {*} x: array of features
    param {*} y: array of labels
    param {*} audio: array of audio
    param {*} lengths: array of audio length
    param {*} path: array of audio path
    param {*} dataset: dataset name
    param {*} method: model name
    param {*} features: features selected
    param {*} n_mfcc: number of mfcc features
    param {*} n_mels: number of mel features
    param {*} max_length: max length for CNN
    param {*} sample: number of samples
    param {*} window: window selected
    param {*} sr: sampling rate
    param {*} noise: type of noise produced
    return {*}: sampled noisy features, labels, audios and audio lengths
    """
    sample_index = random.sample(
        [i for i in range(np.array(x).shape[0])], sample
    )  # sample

    if noise == "white":  # white noise
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
    elif noise == "buzz":  # buzzing noise
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
                snr = float("-inf")
        avg_snr = avg_snr / len(sample_index)
        print("Average Signal-to-Noise Ratio (SNR):", avg_snr, "dB")
    elif noise == "bubble":  # bubble noise
        avg_snr = 0
        for i in sample_index:
            original_audio, sr = librosa.load(path[i], sr=None)
            duration = len(original_audio) / sr
            bubble_frequency_range = (1000, 5000)
            bubble_duration_range = (0.05, 0.5)
            amplitude_range = (0.05, 0.1)
            num_bubbles = int(duration * np.random.uniform(1, 10))
            frequencies = np.random.uniform(*bubble_frequency_range, size=num_bubbles)
            durations = np.random.uniform(*bubble_duration_range, size=num_bubbles)
            amplitudes = np.random.uniform(*amplitude_range, size=num_bubbles)
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

            signal_power = np.sum(original_audio**2) / len(original_audio)
            noise_power = np.sum(bubble_noise**2) / len(bubble_noise)
            snr = 10 * np.log10(signal_power / noise_power)
            avg_anr += snr
        avg_snr = avg_snr / len(sample_index)
        print("Average Signal-to-Noise Ratio (SNR): {:.2f} dB".format(avg_snr))
    elif noise == "cocktail":  # cocktail effect
        for i in sample_index:
            original_audio, sample_rate = librosa.load(path[i], sr=None)
            reversed_audio = original_audio[::-1]
            mixed_audio = original_audio + reversed_audio
            mixed_audio /= np.max(np.abs(mixed_audio))

    # save noisy audio
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

    # get features for noisy audio
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
    """
    description: This function produces audio with noise.
    param {*} x: array of features
    param {*} y: array of labels
    param {*} audio: array of audio
    param {*} lengths: array of audio length
    param {*} emotion_map: mapping between emotion and label
    param {*} dataset: dataset name
    param {*} method: model name
    param {*} features: features selected
    param {*} n_mfcc: number of mfcc features
    param {*} n_mels: number of mel features
    param {*} max_length: max length for CNN
    param {*} window: window selected
    param {*} sr: sampling rate
    return {*}: reconstructed features, labels, audios and audio lengths
    """

    for i in os.listdir(f"datasets/speech/{dataset}_noise/"):
        # open noisy audio
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

        # save reconstructed audio
        name = i.split(".")[0][:-6]
        if not os.path.exists(f"datasets/speech/{dataset}_denoise/"):
            os.makedirs(f"datasets/speech/{dataset}_denoise/")
        reduced_audio.export(
            f"datasets/speech/{dataset}_denoise/{name}_denoise.wav", format="wav"
        )

        # get reconstructed features
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
    """
    description: This function load dataset for RAVDESS.
    param {*} method: model name
    param {*} features: features selected
    param {*} n_mfcc: number of mfcc features
    param {*} n_mels: number of mel features
    param {*} scaled: min-max normalization or standardization
    param {*} max_length: max length for CNN
    param {*} reverse: whether reverse
    param {*} noise: whether adding noise
    param {*} denoise: whether denoise
    param {*} window: window selected
    param {*} sr: sampling rate
    param {*} split: single-corpus split for 4/3/2.5/2/1
    return {*}: training, validation and testing set, audio length
    """
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

    if split == None:  # single-corpus emotion mapping
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
    else:  # three-class cross-corpus emotion mapping
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
            ): 2,  # negative
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

    # load dataset
    for file in glob.glob("datasets/speech/RAVDESS/Actor_*/*.wav"):
        file_name = os.path.basename(file)
        if split == None:
            emotion = emotion_map[file_name.split("-")[2]]
        else:
            for k, i in enumerate(emotion_map.keys()):
                if file_name.split("-")[2] in i:
                    emotion = emotion_map[i]

        # get features
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

        # visualization
        if category.count(category_map[file_name.split("-")[2]]) == 1:
            visual4feature(file, "RAVDESS", category_map[file_name.split("-")[2]])
            corr.append(file)
            corr_emo.append(category_map[file_name.split("-")[2]])
        if category.count(category_map[file_name.split("-")[2]]) == 2:
            index = category.index(
                category_map[file_name.split("-")[2]]
            )  # find the index of the first one
            visual4corr_signal(
                "RAVDESS", path[index], file, category_map[file_name.split("-")[2]]
            )
            # visual4corr_filter_ma(
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

    visual4label("speech", "RAVDESS", category)
    print(np.array(x).shape)
    if method not in ["AlexNet", "CNN"]:
        visual4corrmatrix(x, "RAVDESS")

    # scaled
    if method != "wav2vec":
        if scaled != None:
            x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

    # reverse
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

    # noise
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

    # denoise
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
    if split == None:  # single-corpus
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
        )
    else:  # cross-corpus corresponding single-corpus setup
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
        )
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
    """
    description: This function load dataset for TESS.
    param {*} method: model name
    param {*} features: features selected
    param {*} n_mfcc: number of mfcc features
    param {*} n_mels: number of mel features
    param {*} scaled: min-max normalization or standardization
    param {*} max_length: max length for CNN
    param {*} reverse: whether reverse
    param {*} noise: whether adding noise
    param {*} denoise: whether denoise
    param {*} window: window selected
    param {*} sr: sampling rate
    param {*} split: single-corpus split for 4/3/2.5/2/1
    return {*}: training, validation and testing set, audio length
    """
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

    if split == None:  # single-corpus
        emotion_map = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "neutral": 4,
            "ps": 5,
            "sad": 6,
        }
    else:  # 3-class single-corpus
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
            ): 2,  # negatiive
        }

    for dirname, _, filenames in os.walk("datasets/speech/TESS"):
        for filename in filenames:
            # get features
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

            # visualization
            if category.count(label.lower()) == 1:
                visual4feature(os.path.join(dirname, filename), "TESS", label.lower())
                corr.append(os.path.join(dirname, filename))
                corr_emo.append(label.lower())
            if category.count(label.lower()) == 2:
                index = category.index(label.lower())  # find the index of the first one
                visual4corr_signal(
                    "TESS", path[index], os.path.join(dirname, filename), label.lower()
                )
                # visual4corr_filter_ma(
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

    visual4label("speech", "TESS", category)
    print(np.array(x).shape)
    if method not in ["AlexNet", "CNN"]:
        visual4corrmatrix(x, "TESS")

    # scaled
    if method != "wav2vec":
        if scaled != None:
            x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

    # reverse
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

    # noise
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

    # denoise
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

    if split == None:  # single-corpus
        if method != "wav2vec":
            X_train, X_left, ytrain, yleft = train_test_split(
                np.array(x), y, test_size=0.4, random_state=9
            )
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

    else:  # cross-corpus corresponding single-corpus cases
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
        )
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
    """
    description: This function load dataset for SAVEE.
    param {*} method: model name
    param {*} features: features selected
    param {*} n_mfcc: number of mfcc features
    param {*} n_mels: number of mel features
    param {*} scaled: min-max normalization or standardization
    param {*} max_length: max length for CNN
    param {*} reverse: whether reverse
    param {*} noise: whether adding noise
    param {*} denoise: whether denoise
    param {*} window: window selected
    param {*} sr: sampling rate
    param {*} split: single-corpus split for 4/3/2.5/2/1
    return {*}: training, validation and testing set, audio length
    """
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

    if split == None:  # single-corpus
        emotion_map = {
            "a": 0,  # angry
            "d": 1,  # digust
            "f": 2,  # fear
            "h": 3,  # happiness
            "n": 4,  # neutral
            "sa": 5,  # sadness
            "su": 6,  # surprise
        }
    else:  # single-corpus 3-class
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
            ): 2,  # negative
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
        # get features
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

        # visualization
        if category.count(category_map[label]) == 1:
            visual4feature(os.path.join(path, file), "SAVEE", category_map[label])
            corr.append(os.path.join(path, file))
            corr_emo.append(category_map[label])
        if category.count(category_map[label]) == 2:
            index = category.index(
                category_map[label]
            )  # find the index of the first one
            visual4corr_signal(
                "SAVEE", paths[index], os.path.join(path, file), category_map[label]
            )
            # visual4corr_filter_ma(
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

    visual4label("speech", "SAVEE", category)
    print(np.array(x).shape)
    if method not in ["AlexNet", "CNN"]:
        visual4corrmatrix(x, "SAVEE")

    # scaled
    if method != "wav2vec":
        if scaled != None:
            x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

    # reverse
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
        )

    # noise
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

    # denoise
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

    if split == None:  # single-corpus
        if method != "wav2vec":
            X_train, X_left, ytrain, yleft = train_test_split(
                np.array(x), y, test_size=0.4, random_state=9
            )
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

    else:  # cross-corpus corresponding single-corpus case
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
        )
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
    """
    description: This function load dataset for CREMA-D.
    param {*} method: model name
    param {*} features: features selected
    param {*} n_mfcc: number of mfcc features
    param {*} n_mels: number of mel features
    param {*} scaled: min-max normalization or standardization
    param {*} max_length: max length for CNN
    param {*} reverse: whether reverse
    param {*} noise: whether adding noise
    param {*} denoise: whether denoise
    param {*} window: window selected
    param {*} sr: sampling rate
    param {*} split: single-corpus split for 4/3/2.5/2/1
    return {*}: training, validation and testing set, audio length
    """
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

    if split == None:  # single-corpus
        emotion_map = {
            "ang": 0,  # angry
            "dis": 1,  # disgust
            "fea": 2,  # fear
            "hap": 3,  # happiness
            "neu": 4,  # neutral
            "sad": 5,  # sadness
        }
    else:  # single-corpus 3-class
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
            ): 2,  # negative
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

        # visualization
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
            visual4corr_signal(
                "CREMA",
                paths[index],
                os.path.join(path, file),
                category_map[label.lower()],
            )
            # visual4corr_filter_ma(
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

    visual4label("speech", "CREMA", category)
    print(np.array(x).shape)
    if method not in ["AlexNet", "CNN"]:
        visual4corrmatrix(x, "CREMA-D")

    # scaled
    if method != "wav2vec":
        if scaled != None:
            x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

    # reverse
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
        )

    # noise
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

    # denoise
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

    if split == None:  # single-corpus
        if method != "wav2vec":
            X_train, X_left, ytrain, yleft = train_test_split(
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
    else:  # cross-corpus corresponding single-corpus
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
        )
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
    """
    description: This function load dataset for EmoDB.
    param {*} method: model name
    param {*} features: features selected
    param {*} n_mfcc: number of mfcc features
    param {*} n_mels: number of mel features
    param {*} scaled: min-max normalization or standardization
    param {*} max_length: max length for CNN
    param {*} reverse: whether reverse
    param {*} noise: whether adding noise
    param {*} denoise: whether denoise
    param {*} window: window selected
    param {*} sr: sampling rate
    param {*} split: single-corpus split for 4/3/2.5/2/1
    return {*}: training, validation and testing set, audio length
    """
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

    if split == None:  # single-corpus
        emotion_map = {
            "W": 0,  # angry
            "L": 1,  # boredom
            "E": 2,  # disgust
            "A": 3,  # anxiety/fear
            "F": 4,  # happiness
            "T": 5,  # sadness
            "N": 6,
        }
    else:  # 3-class single-corpus
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
            ): 2,  # negative
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
        # get features
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

        # visualization
        if category.count(category_map[label]) == 1:
            visual4feature(os.path.join(path, file), "EmoDB", category_map[label])
            corr.append(os.path.join(path, file))
            corr_emo.append(category_map[label])
        if category.count(category_map[label]) == 2:
            index = category.index(
                category_map[label]
            )  # find the index of the first one
            visual4corr_signal(
                "EmoDB", paths[index], os.path.join(path, file), category_map[label]
            )
            # visual4corr_filter_ma(
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

    visual4label("speech", "EmoDB", category)
    print(np.array(x).shape)
    if method not in ["AlexNet", "CNN"]:
        visual4corrmatrix(x, "EmoDB")

    # scaled
    if method != "wav2vec":
        if scaled != None:
            x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

    # reverse
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
        )

    # noise
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

    # denoise
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

    if split == None:  # single-corpus
        if method != "wav2vec":
            X_train, X_left, ytrain, yleft = train_test_split(
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
    else:  # cross-corpus corresponding single-corpus
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
        )
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
    """
    description: This function load dataset for eNTERFACE-05.
    param {*} method: model name
    param {*} features: features selected
    param {*} n_mfcc: number of mfcc features
    param {*} n_mels: number of mel features
    param {*} scaled: min-max normalization or standardization
    param {*} max_length: max length for CNN
    param {*} reverse: whether reverse
    param {*} noise: whether adding noise
    param {*} denoise: whether denoise
    param {*} window: window selected
    param {*} sr: sampling rate
    param {*} split: single-corpus split for 4/3/2.5/2/1
    return {*}: training, validation and testing set, audio length
    """
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

    if split == None:  # single-corpus
        emotion_map = {
            "an": 0,  # angry
            "di": 1,  # disgust
            "fe": 2,  # fear
            "ha": 3,  # happiness
            "sa": 4,  # sadness
            "su": 5,  # surprise
        }
    else:  # 3-class single-corpus
        emotion_map = {
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
            ): 1,  # negative
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
        # get features
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

        # visualization
        if category.count(category_map[label]) == 1:
            visual4feature(os.path.join(path, file), "eNTERFACE", category_map[label])
            corr.append(os.path.join(path, file))
            corr_emo.append(category_map[label])
        if category.count(category_map[label]) == 2:
            index = category.index(
                category_map[label]
            )  # find the index of the first one
            visual4corr_signal(
                "eNTERFACE", paths[index], os.path.join(path, file), category_map[label]
            )
            # visual4corr_filter_ma(
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

    visual4label("speech", "eNTERFACE05", category)
    print(np.array(x).shape)
    if method not in ["AlexNet", "CNN"]:
        visual4corrmatrix(x, "eNTERFACE")

    # scaled
    if method != "wav2vec":
        if scaled != None:
            x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

    # reverse
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
        )

    # noise
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

    # denoise
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

    if split == None:  # single-corpus
        if method != "wav2vec":
            X_train, X_left, ytrain, yleft = train_test_split(
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
    else:  # cross-corpus corresponding single-corpus
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
        )
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
    """
    description: This function load dataset for AESDD.
    param {*} method: model name
    param {*} features: features selected
    param {*} n_mfcc: number of mfcc features
    param {*} n_mels: number of mel features
    param {*} scaled: min-max normalization or standardization
    param {*} max_length: max length for CNN
    param {*} reverse: whether reverse
    param {*} noise: whether adding noise
    param {*} denoise: whether denoise
    param {*} window: window selected
    param {*} sr: sampling rate
    param {*} split: single-corpus split for 4/3/2.5/2/1
    return {*}: training, validation and testing set, audio length
    """
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

    if split == None:  # single-corpus
        emotion_map = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "happiness": 3,
            "sadness": 4,
        }
    else:  # 3-class single-corpus
        emotion_map = {
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
            # get features
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
            label = dirname.split("/")[-1]

            if split == None:
                emotion = emotion_map[label]
            else:
                for k, i in enumerate(emotion_map.keys()):
                    if label in i:
                        emotion = emotion_map[i]

            x.append(feature)
            y.append(emotion)
            path.append(os.path.join(dirname, filename))
            category.append(label)
            audio.append(X)
            lengths.append(len(X))

            # visualization
            if category.count(label) == 1:
                visual4feature(os.path.join(dirname, filename), "AESDD", label)
                corr.append(os.path.join(dirname, filename))
                corr_emo.append(label)
            if category.count(label) == 2:
                index = category.index(label)  # find the index of the first one
                visual4corr_signal(
                    "AESDD", path[index], os.path.join(dirname, filename), label
                )
                # visual4corr_filter_ma(
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

    visual4label("speech", "AESDD", category)
    print(np.array(x).shape)
    if method not in ["AlexNet", "CNN"]:
        visual4corrmatrix(x, "AESDD")

    # scaled
    if method != "wav2vec":
        if scaled != None:
            x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

    # reverse
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

    # noise
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

    # denoise
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

    if split == None:  # single-corpus
        if method != "wav2vec":
            X_train, X_left, ytrain, yleft = train_test_split(
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
    else:  # cross-corpus corresponding single-corpus
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
        )
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
    """
    description: This function loads dataset for cross-corpus experiments of Case 1.
    param {*} method: model name
    param {*} features: features selected
    param {*} n_mfcc: number of mfcc features
    param {*} n_mels: number of mel features
    param {*} scaled: min-max normalization or standardization
    param {*} max_length: max length for CNN
    param {*} reverse: whether reverse
    param {*} noise: whether adding noise
    param {*} denoise: whether denoise
    param {*} window: window selected
    param {*} corpus: two corpus where the first one is training corpus and the second one is testing corpus
    param {*} sr: sampling rate
    return {*}: training, validation and testing set, audio length
    """
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
        ): 2,  # negative
    }
    # load dataset for each
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

                # get features
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

        # scaled
        if method != "wav2vec":
            if scaled != None:
                x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

        # reverse
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

        # noise
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

        # denoise
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

        if method != "wav2vec":
            # 900, 300, 300, train_corpus 1200, test_corpus 300
            if index == 0:  # train corpus
                # split into train and val
                random.seed(123)
                sample_index = random.sample(
                    [i for i in range(np.array(x).shape[0])], 1200
                )
                X_train, X_val, ytrain, yval = train_test_split(
                    np.array(x)[sample_index, :],
                    np.array(y)[sample_index].tolist(),
                    test_size=0.25,
                    random_state=9,
                )  # 3:1
            elif index == 1:  # test corpus
                random.seed(123)
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

    return X_train, ytrain, X_val, yval, X_test, ytest, length


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
    # 1200/5=240
    """
    description: This function loads dataset for cross-corpus experiments of Case 2.
    param {*} method: model name
    param {*} features: features selected
    param {*} n_mfcc: number of mfcc features
    param {*} n_mels: number of mel features
    param {*} scaled: min-max normalization or standardization
    param {*} max_length: max length for CNN
    param {*} reverse: whether reverse
    param {*} noise: whether adding noise
    param {*} denoise: whether denoise
    param {*} window: window selected
    param {*} corpus: one corpus which is the testing corpus when the mixture is the training corpus
    param {*} sr: sampling rate
    return {*}: training, validation and testing set, audio length
    """
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
        ): 2,  # negative
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
    # load data
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

        # mixture of train
        random.seed(123)
        sample_index = random.sample([i for i in range(np.array(x).shape[0])], 240)
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

        # scaled
        if method != "wav2vec":
            if scaled != None:
                X_train = transform_feature(
                    method, X_train, features, n_mfcc, n_mels, scaled
                )

        # reverse
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

        # noise
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

        # denoise
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
            left_index = [
                i for i in range(np.array(x).shape[0]) if i not in sample_index
            ]
            X_test = np.array(x)[left_index, :]
            ytest = np.array(y)[left_index].tolist()

    length = None

    X_train, X_val, ytrain, yval = train_test_split(
        X_train, ytrain, test_size=0.25, random_state=9
    )  # 3:1  # 900:300

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
):
    # 1:1:1 (200/200/200)
    """
    description: This function loads dataset for cross-corpus experiments of modified setup with mixture of 3 corpus.
    param {*} method: model name
    param {*} features: features selected
    param {*} n_mfcc: number of mfcc features
    param {*} n_mels: number of mel features
    param {*} scaled: min-max normalization or standardization
    param {*} max_length: max length for CNN
    param {*} reverse: whether reverse
    param {*} noise: whether adding noise
    param {*} denoise: whether denoise
    param {*} window: window selected
    param {*} corpus: the testing corpus under mixture of 3 as training for 1:1:1 (200/200/200)
    param {*} sr: sampling rate
    return {*}: training, validation and testing set, audio length
    """
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
        ): 2,  # negative
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
    # load dataset
    for index, dataset in enumerate(datasets):
        x, y = [], []  # for each dataset
        if dataset == "RAVDESS":
            for file in glob.glob("datasets/speech/RAVDESS/Actor_*/*.wav"):
                file_name = os.path.basename(file)
                for k, i in enumerate(emotion_map.keys()):
                    if file_name.split("-")[2] in i:
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

        # scaled
        if method != "wav2vec":
            if scaled != None:
                X_train = transform_feature(
                    method, X_train, features, n_mfcc, n_mels, scaled
                )

        # reverse
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

        # noise
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

        # denoise
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

        if dataset == corpus[0]:  # testing set
            left_index = [
                i for i in range(np.array(x).shape[0]) if i not in sample_index
            ]
            random.seed(123)
            test_index = random.sample(left_index, 200)  # 200:200:200  -- 200
            X_test = np.array(x)[test_index, :]
            ytest = np.array(y)[test_index].tolist()

    length = None

    X_train, X_val, ytrain, yval = train_test_split(
        X_train, ytrain, test_size=0.5, random_state=9
    )

    return X_train, ytrain, X_val, yval, X_test, ytest, length


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
    split=None,
):
    """
    description: This function loads dataset for cross-corpus experiments of modified setup.
    param {*} method: model name
    param {*} features: features selected
    param {*} n_mfcc: number of mfcc features
    param {*} n_mels: number of mel features
    param {*} scaled: min-max normalization or standardization
    param {*} max_length: max length for CNN
    param {*} reverse: whether reverse
    param {*} noise: whether adding noise
    param {*} denoise: whether denoise
    param {*} window: window selected
    param {*} corpus: two corpus where the first one is training corpus and the second one is testing corpus
    param {*} sr: sampling rate
    param {*} split: ranging split of mixture for training 0.8/0.6/0.5/0.4/0.2
    return {*}: training, validation and testing set, audio length
    """

    if ("eNTERFACE" in set(corpus)) and ("AESDD" in set(corpus)):
        emotion_map = {
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
            ): 1,  # negative
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
            ): 2,  # negative
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
    # load dataset
    for index, cor in enumerate(corpus):
        x, y = [], []  # for each dataset
        if cor == "RAVDESS":
            for file in glob.glob("datasets/speech/RAVDESS/Actor_*/*.wav"):
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

        # mixture of training 1000*(1-split) for train corpus, 1000*split for test corpus
        random.seed(123)
        if index == 0:  # train in mixture
            sample_index = random.sample(
                [i for i in range(np.array(x).shape[0])], int(1000 * (1 - split))
            )
        elif index == 1:  # test in mixture
            sample_index = random.sample(
                [i for i in range(np.array(x).shape[0])], int(1000 * split)
            )

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

        # scaled
        if method != "wav2vec":
            if scaled != None:
                X_train = transform_feature(
                    method, X_train, features, n_mfcc, n_mels, scaled
                )

        # reverse
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

        # noise
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

        # denoise
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

        if (cor == corpus[1]) or (
            corpus[1] == "eNTERFACE"
        ):  # testing set fixed 200 samples
            left_index = [
                i for i in range(np.array(x).shape[0]) if i not in sample_index
            ]
            random.seed(123)
            test_index = random.sample(left_index, 200)
            X_test = np.array(x)[test_index, :]
            ytest = np.array(y)[test_index].tolist()

    length = None

    X_train, X_val, ytrain, yval = train_test_split(
        X_train, ytrain, test_size=0.5, random_state=9
    )

    return X_train, ytrain, X_val, yval, X_test, ytest, length


def load_finetune_corpus(
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
    """
    description: This function loads dataset for cross-corpus experiments of Case 3.
    param {*} method: model name
    param {*} features: features selected
    param {*} n_mfcc: number of mfcc features
    param {*} n_mels: number of mel features
    param {*} scaled: min-max normalization or standardization
    param {*} max_length: max length for CNN
    param {*} reverse: whether reverse
    param {*} noise: whether adding noise
    param {*} denoise: whether denoise
    param {*} window: window selected
    param {*} corpus: two corpus where the first one is training corpus and the second one is finetuning/testing corpus
    param {*} sr: sampling rate
    return {*}: training, validation and testing set, audio length
    """

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
        ): 2,  # negative
    }
    # load dataset
    for index, cor in enumerate(corpus):
        # index=0 train, index=1 finetune/test
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

        # scaled
        if method != "wav2vec":
            if scaled != None:
                x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

        # reverse
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

        # noise
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

        # denoise
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

        # train: 1200 (900+300, train+val)
        # finetune: 600 train, 200 val, 200 test
        if method != "wav2vec":
            if index == 0:  # train corpus
                # split into train and val
                random.seed(123)
                sample_index = random.sample(
                    [i for i in range(np.array(x).shape[0])], 1200
                )
                X_train, X_val, ytrain, yval = train_test_split(
                    np.array(x)[sample_index, :],
                    np.array(y)[sample_index],
                    test_size=0.25,
                    random_state=9,
                )  # 3:1  # train + val
            elif index == 1:  # finetune
                random.seed(123)
                Xtune_train, X_left, ytune_train, yleft = train_test_split(
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


def get_face_landmarks(image, landmark="xyz", write=None, dataset="CK"):
    """
    description: This function gets facial landmarks from raw images.
    param {*} image: raw image
    param {*} landmark: landmark choices as xyz, 5, 68
    param {*} dataset: name of dataset
    return {*}: landmarks extracted
    """
    if landmark == "xyz":  # 468 landmarks
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
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # 468x3=1404
            for j in range(len(xs_)):  # get landmarks of the face
                image_landmarks.append(xs_[j] - min(xs_))
                image_landmarks.append(ys_[j] - min(ys_))
                image_landmarks.append(zs_[j] - min(zs_))

        # save
        if write != None:
            if not os.path.exists(f"outputs/image/landmarks/"):
                os.makedirs(f"outputs/image/landmarks/")
            cv2.imwrite(f"outputs/image/landmarks/{dataset}_468_{write}.JPG", image)
        face_mesh.close()
        res = image_landmarks

    elif landmark == "5":  # 5 landmarks
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        res = []
        detector = dlib.get_frontal_face_detector()
        faces = detector(image, 1)
        predictor_5 = dlib.shape_predictor(
            "models/image/shape_predictor_5_face_landmarks.dat"
        )
        for face in faces:
            image_landmarks = predictor_5(image, face)

            # Loop over the landmark points
            tmp_x = []
            tmp_y = []
            for n in range(0, 5):  # 5 landmarks with x,y, 10 features in total
                x = image_landmarks.part(n).x
                y = image_landmarks.part(n).y
                tmp_x.append(x)
                tmp_y.append(y)
            for n in range(0, 5):
                res.append(tmp_x[n] - min(tmp_x))
                res.append(tmp_y[n] - min(tmp_y))
                cv2.circle(image, (tmp_x[n], tmp_y[n]), 1, (255, 0, 0), -1)

        # save
        if (write != None) and (len(faces) != 0):
            if not os.path.exists(f"outputs/image/landmarks/"):
                os.makedirs(f"outputs/image/landmarks/")
            cv2.imwrite(
                f"outputs/image/landmarks/{dataset}_{landmark}_{write}.JPG", image
            )

    elif landmark == "68":  # 68 landmarks
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        res = []
        detector = dlib.get_frontal_face_detector()
        faces = detector(image, 1)
        predictor_68 = dlib.shape_predictor(
            "models/image/shape_predictor_68_face_landmarks.dat"
        )
        for face in faces:
            image_landmarks = predictor_68(image, face)

            # Loop over the landmark points
            tmp_x = []
            tmp_y = []
            for n in range(0, 68):  # 68 landmarks with x,y, 134 features in total
                x = image_landmarks.part(n).x
                y = image_landmarks.part(n).y
                tmp_x.append(x)
                tmp_y.append(y)
            for n in range(0, 68):
                res.append(tmp_x[n] - min(tmp_x))
                res.append(tmp_y[n] - min(tmp_y))
                cv2.circle(image, (tmp_x[n], tmp_y[n]), 1, (255, 0, 0), -1)

        # save
        if (write != None) and (len(faces) != 0):
            if not os.path.exists(f"outputs/image/landmarks/"):
                os.makedirs(f"outputs/image/landmarks/")
            cv2.imwrite(
                f"outputs/image/landmarks/{dataset}_{landmark}_{write}.JPG", image
            )

    return res


def load_CK(method, landmark=None, split=None, process=None):
    """
    description: This function loads dataset for CK+.
    param {*} method: name of model
    param {*} landmark: choices of landmark
    param {*} split: split for 3-class single-corpus
    param {*} process: sobel/equal/filter/blur/assi/noise
    return {*}: training, validation and testing set, dimension, number of classes
    """
    X, y, category = [], [], []
    if split == None:  # single-corpus
        emotion_map = {
            "anger": 0,
            "contempt": 1,
            "disgust": 2,
            "fear": 3,
            "happy": 4,
            "sadness": 5,
            "surprise": 6,
        }
    else:  # 3-class single-corpus
        emotion_map = {
            ("anger", "disgust", "fear", "sadness", "sad", "2", "3", "5", "6"): 0,
            ("happy", "surprise", "4", "1"): 1,
            ("contempt", "neutral", "7"): 2,
        }

    # load dataset
    for dirname, _, filenames in os.walk("datasets/image/CK"):
        for filename in filenames:
            img = cv2.imread(os.path.join(dirname, filename))
            label = dirname.split("/")[-1]

            if landmark == None:  # raw image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if len(X) == 0:
                    if not os.path.exists("outputs/image/process/"):
                        os.makedirs("outputs/image/process/")
                    cv2.imwrite(f"outputs/image/process/CK_ori.JPG", img)

                # processing
                if process == "blur":  # gaussian blur
                    img = cv2.GaussianBlur(img, (7, 7), 0)
                elif process == "noise":  # salt & pepper
                    img = random_noise(img, mode="s&p")
                    img = (255 * img).astype(np.uint8)
                elif process == "sobel":  # sobel's kernel
                    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                    sobelx = cv2.convertScaleAbs(sobelx)
                    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                    sobely = cv2.convertScaleAbs(sobely)
                    img = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
                elif process == "equal":  # histogram equalization
                    img = cv2.equalizeHist(img)
                elif process == "filter":  # wiener noise filter
                    img = wiener(img, mysize=(15, 15))
                    img = np.clip(img, 0, 255).astype(np.uint8)

                X.append(img)
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

            else:  # landmarks
                write = len(X) if len(X) <= 5 else None
                face_landmarks = get_face_landmarks(
                    img, landmark=landmark, write=write, dataset="CK"
                )
                X.append(face_landmarks)
                y.append(emotion_map[label])

    visual4label("image", "CK", category)
    X, y = shuffle(X, y, random_state=42)  # shuffle
    print(len(X))
    num_classes = len(set(y))
    if landmark == None:
        n, h, w = np.array(X).shape
        if method == "MLP":
            X = np.array(X).reshape((n, h * w)).tolist()
            h = h * w
    else:
        h = np.array(X).shape[1]

    # train test split
    if split == None:  # single-corpus
        X_train, X_left, ytrain, yleft = train_test_split(
            np.array(X),
            y,
            test_size=0.4,
            random_state=9,
        )  # 3:2

        X_val, X_test, yval, ytest = train_test_split(
            X_left, yleft, test_size=0.5, random_state=9
        )  # 1:1
    else:  # cross-corpus corresponding 3-class single-corpus
        random.seed(123)
        test_index = random.sample(
            [i for i in range(np.array(X).shape[0])], 200  # 200 fixed for testing size
        )
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

    return X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes


def load_FER(method, landmark=None, split=None, process=None):
    """
    description: This function loads dataset for FER-2013.
    param {*} method: name of model
    param {*} landmark: choices of landmark
    param {*} split: split for 3-class single-corpus
    param {*} process: sobel/equal/filter/blur/assi/noise
    return {*}: training, validation and testing set, dimension, number of classes
    """
    X, y, category = [], [], []
    if split == None:  # single-corpus
        emotion_map = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "neutral": 4,
            "sad": 5,
            "surprise": 6,
        }
    else:  # 3-class single-corpus
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
                if landmark == None:  # raw image
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if len(X) == 0:
                        if not os.path.exists("outputs/image/process/"):
                            os.makedirs("outputs/image/process/")
                        cv2.imwrite(f"outputs/image/process/FER_ori.JPG", img)

                    # processing
                    if process == "blur":  # gaussian blur
                        img = cv2.GaussianBlur(img, (7, 7), 0)
                    elif process == "noise":  # salt & pepper
                        img = random_noise(img, mode="s&p")
                        img = (255 * img).astype(np.uint8)
                    elif process == "sobel":  # sobel's kernel
                        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                        sobelx = cv2.convertScaleAbs(sobelx)
                        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                        sobely = cv2.convertScaleAbs(sobely)
                        img = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
                    elif process == "equal":  # histogram equalization
                        img = cv2.equalizeHist(img)
                    elif process == "filter":  # wiener noise filter
                        img = wiener(img, mysize=(15, 15))
                        img = np.clip(img, 0, 255).astype(np.uint8)

                    X.append(img)
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
                else:  # landmarks
                    write = len(X) if len(X) <= 5 else None
                    face_landmarks = get_face_landmarks(
                        img, landmark=landmark, write=write, dataset="FER"
                    )
                    if landmark == "xyz":
                        if len(face_landmarks) == 1404:
                            X.append(face_landmarks)
                            y.append(emotion_map[secdir])
                    else:
                        if len(face_landmarks) != 0:
                            X.append(face_landmarks)
                            y.append(emotion_map[secdir])

    X, y = shuffle(X, y, random_state=42)  # shuffle
    print(len(X))
    visual4label("image", "FER", category)
    num_classes = len(set(y))
    if landmark == None:
        n, h, w = np.array(X).shape
        if method == "MLP":
            X = np.array(X).reshape((n, h * w)).tolist()
            h = h * w
    else:
        h = np.array(X).shape[1]

    if split == None:  # single-corpus
        X_train, X_left, ytrain, yleft = train_test_split(
            np.array(X),
            y,
            test_size=0.4,
            random_state=9,
        )  # 3:2

        X_val, X_test, yval, ytest = train_test_split(
            X_left, yleft, test_size=0.5, random_state=9
        )  # 1:1
    else:  # cross-corpus corresponding 3-class single-corpus
        random.seed(123)
        test_index = random.sample(
            [i for i in range(np.array(X).shape[0])], 200  # 200 fixed for testing size
        )
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

    return X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes


def load_RAF(method, landmark=None, split=None, process=None):
    """
    description: This function loads dataset for RAF-DB.
    param {*} method: name of model
    param {*} landmark: choices of landmark
    param {*} split: split for 3-class single-corpus
    param {*} process: sobel/equal/filter/blur/assi/noise
    return {*}: training, validation and testing set, dimension, number of classes
    """
    X, y, category = [], [], []
    if split == None:  # single-corpus
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
    else:  # 3-class single-corpus
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
                if landmark == None:  # raw image
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if len(X) == 0:
                        if not os.path.exists("outputs/image/process/"):
                            os.makedirs("outputs/image/process/")
                        cv2.imwrite(f"outputs/image/process/RAF_ori.JPG", img)

                    # processing
                    if process == "blur":  # gaussian blur
                        img = cv2.GaussianBlur(img, (7, 7), 0)
                    elif process == "noise":  # salt & pepper
                        img = random_noise(img, mode="s&p")
                        img = (255 * img).astype(np.uint8)
                    elif process == "sobel":  # sobel's kernel
                        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                        sobelx = cv2.convertScaleAbs(sobelx)
                        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                        sobely = cv2.convertScaleAbs(sobely)
                        img = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
                    elif process == "equal":  # histogram equalization
                        img = cv2.equalizeHist(img)
                    elif process == "filter":  # wiener noise filtering
                        img = wiener(img, mysize=(15, 15))
                        img = np.clip(img, 0, 255).astype(np.uint8)

                    X.append(img)
                    if len(X) == 1:
                        cv2.imwrite(f"outputs/image/process/RAF_{process}.JPG", img)
                    if split != None:
                        for k, i in enumerate(emotion_map.keys()):
                            if secdir in i:
                                emotion = emotion_map[i]
                        y.append(emotion)
                    else:
                        category.append(category_map[secdir])
                        y.append(emotion_map[secdir])
                else:  # landmarks
                    write = len(X) if len(X) <= 5 else None
                    face_landmarks = get_face_landmarks(
                        img, landmark=landmark, write=write, dataset="RAF"
                    )
                    if landmark == "xyz":
                        if len(face_landmarks) == 1404:
                            X.append(face_landmarks)
                            y.append(emotion_map[secdir])
                    else:
                        if len(face_landmarks) != 0:
                            X.append(face_landmarks)
                            y.append(emotion_map[secdir])

    X, y = shuffle(X, y, random_state=42)  # shuffle
    num_classes = len(set(y))
    print(len(X))
    visual4label("image", "RAF", category)
    if landmark == None:
        n, h, w = np.array(X).shape
        if method == "MLP":
            X = np.array(X).reshape((n, h * w)).tolist()
            h = h * w
    else:
        h = np.array(X).shape[1]

    if split == None:  # single-corpus
        X_train, X_left, ytrain, yleft = train_test_split(
            np.array(X),
            y,
            test_size=0.4,
            random_state=9,
        )  # 3:2

        X_val, X_test, yval, ytest = train_test_split(
            X_left, yleft, test_size=0.5, random_state=9
        )  # 1:1
    else:  # cross-corpus corresponding 3-class single-corpus
        random.seed(123)
        test_index = random.sample(
            [i for i in range(np.array(X).shape[0])], 200  # 200 fixed for testing size
        )
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

    return X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes


def load_split_corpus_size_image(
    method,
    corpus=None,
    split=None,
    process=None,
):
    """
    description: This function constructs cross-corpus experiments for the mixture of 2 corpus.
    param {*} method: name of model
    param {*} corpus: two corpus of the mixture, the former one is training set while the latter one is testing set
    param {*} split: split of size for mixture of two corpus with 0.8/0.6/0.5/0.4/0.2
    param {*} process: process of assimilating corpus as same framework
    return {*}: training, validation and testing, dimension and number of classes
    """
    emotion_map = {
        ("anger", "disgust", "fear", "sadness", "sad", "2", "3", "5", "6"): 0,
        ("happy", "surprise", "4", "1"): 1,
        ("contempt", "neutral", "7"): 2,
    }

    X_train, ytrain, X_val, yval, X_test, ytest = [], [], [], [], [], []

    # load each dataset
    for index, cor in enumerate(corpus):
        x, y = [], []  # for each dataset
        if cor == "CK":
            for dirname, _, filenames in os.walk("datasets/image/CK"):
                for filename in filenames:
                    img = cv2.imread(os.path.join(dirname, filename))
                    label = dirname.split("/")[-1]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    if index == 0:
                        # processing
                        if process == "blur":
                            img = cv2.GaussianBlur(img, (7, 7), 0)
                        elif process == "noise":
                            img = random_noise(img, mode="s&p")
                            img = (255 * img).astype(np.uint8)
                        elif process == "sobel":
                            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                            sobelx = cv2.convertScaleAbs(sobelx)
                            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                            sobely = cv2.convertScaleAbs(sobely)
                            img = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
                        elif process == "equal":
                            img = cv2.equalizeHist(img)
                        elif process == "filter":
                            img = wiener(img, mysize=(15, 15))
                            img = np.clip(img, 0, 255).astype(np.uint8)

                        elif process == "assi":
                            img = cv2.equalizeHist(img)
                            img = cv2.GaussianBlur(img, (3, 3), 0)
                    if index == 1:
                        if process == "assi":
                            img = cv2.equalizeHist(img)
                            img = cv2.GaussianBlur(img, (3, 3), 0)

                    x.append(img)
                    for k, i in enumerate(emotion_map.keys()):
                        if label in i:
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
                        if index == 0:
                            # processing
                            if process == "blur":
                                img = cv2.GaussianBlur(img, (7, 7), 0)
                            elif process == "noise":
                                img = random_noise(img, mode="s&p")
                                img = (255 * img).astype(np.uint8)
                            elif process == "sobel":
                                sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                                sobelx = cv2.convertScaleAbs(sobelx)
                                sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                                sobely = cv2.convertScaleAbs(sobely)
                                img = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
                            elif process == "equal":
                                img = cv2.equalizeHist(img)
                            elif process == "filter":
                                img = wiener(img, mysize=(15, 15))
                                img = np.clip(img, 0, 255).astype(np.uint8)
                            elif process == "assi":
                                img = cv2.equalizeHist(img)
                                img = cv2.GaussianBlur(img, (3, 3), 0)

                        if index == 1:
                            if process == "assi":
                                img = cv2.equalizeHist(img)
                                img = cv2.GaussianBlur(img, (3, 3), 0)
                        x.append(img)
                        for k, i in enumerate(emotion_map.keys()):
                            if secdir in i:
                                emotion = emotion_map[i]
                        y.append(emotion)

        x, y = shuffle(x, y, random_state=42)  # shuffle
        num_classes = len(set(y))
        n, h, w = np.array(x).shape

        # train test split
        random.seed(123)
        if index == 0:  # train in mixture
            sample_index = random.sample(
                [i for i in range(np.array(x).shape[0])], int(1000 * (1 - split))
            )
        elif index == 1:  # test in mixture
            if (split == 0.8) and (cor == "CK"):
                sample_index = random.sample(
                    [i for i in range(np.array(x).shape[0])], int(len(x) - 200)
                )
            else:
                sample_index = random.sample(
                    [i for i in range(np.array(x).shape[0])], int(1000 * split)
                )

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
            left_index = [
                i for i in range(np.array(x).shape[0]) if i not in sample_index
            ]
            random.seed(123)
            test_index = random.sample(left_index, 200)  # fixed 200 for testing set
            X_test = np.array(x)[test_index, :, :]
            ytest = np.array(y)[test_index].tolist()

    X_train, X_val, ytrain, yval = train_test_split(
        X_train, ytrain, test_size=0.5, random_state=9
    )

    return X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes


def load_mix3_corpus_image(
    method,
    corpus=None,
):
    """
    description: This function constructs cross-corpus experiments for the mixture of 3 corpus.
    param {*} method: name of model
    param {*} corpus: testing corpus of 1:1:1 (200/200/200)
    return {*}: training, validation and testing, dimension and number of classes
    """
    datasets = ["CK", "FER", "RAF"]
    emotion_map = {
        ("anger", "disgust", "fear", "sadness", "sad", "2", "3", "5", "6"): 0,
        ("happy", "surprise", "4", "1"): 1,
        ("contempt", "neutral", "7"): 2,
    }

    X_train, ytrain, X_val, yval, X_test, ytest = [], [], [], [], [], []

    # load dataset
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
                        x.append(img)
                        for k, i in enumerate(emotion_map.keys()):
                            if secdir in i:
                                emotion = emotion_map[i]
                        y.append(emotion)

        x, y = shuffle(x, y, random_state=42)  # shuffle
        num_classes = len(set(y))
        n, h, w = np.array(x).shape

        # 200 samples for each corpus in mixture
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

        if dataset == corpus[0]:  # testing set
            left_index = [
                i for i in range(np.array(x).shape[0]) if i not in sample_index
            ]
            random.seed(123)
            test_index = random.sample(left_index, 200)  # 1:1:1 (200:200:200)
            X_test = np.array(x)[test_index, :]
            ytest = np.array(y)[test_index].tolist()

    X_train, X_val, ytrain, yval = train_test_split(
        X_train, ytrain, test_size=0.5, random_state=9
    )

    return X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes


def load_patches(X, dataset):
    """
    description: This method is used for ViT to get image patches.
    param {*} X: input images
    return {*}: patches
    """
    if dataset in ["CK", "FER"]:
        X_patches = patchify(X[0], (8, 8), 8)
        X_save = np.reshape(X_patches, (36, 8, 8, 1))

        # save patches
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
        X_patches = patchify(X[0], (25, 25), 25)
        X_save = np.reshape(X_patches, (16, 25, 25, 1))

        # save patches
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
    """
    description: This function is the general function for loading dataset.
    param {*} task: image or speech
    param {*} method: model name
    param {*} cc: mix/cross/finetune
    parma {*} dataset: name of dataset
    param {*} features: features selected
    param {*} n_mfcc: number of mfcc features
    param {*} n_mels: number of mel features
    param {*} scaled: min-max normalization or standardization
    param {*} max_length: max length for CNN
    param {*} reverse: whether reverse
    param {*} noise: whether adding noise
    param {*} denoise: whether denoise
    param {*} window: window selected
    param {*} corpus: depend on cross-corpus experiment setups
    param {*} sr: sampling rate
    param {*} landmark: landmark choices
    param {*} split: cross-corpus 0.8/0.6/0.5/0.4/0.2, single-corpus 4/3/2.5/2/1
    param {*} process: processing choices of operations
    return {*}: training, validation and testing set, audio length
    """
    if task == "speech":  # load speech dataset
        if corpus == None:  # single-corpus
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
                    # # use for original design of Case 2
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

                    # used for modified setup mixture of 3
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
                elif cc == "finetune":  # cross-corpus Case 3
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
                elif cc == "cross":  # cross-corpus Case 1
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
            else:  # modified setup of cross-corpus
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
            if cc != "finetune":
                return X_train, ytrain, X_val, yval, X_test, ytest, shape, num_classes
            else:
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

    elif task == "image":  # load image dataset
        if corpus == None:  # single-corpus
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
                X_train = load_patches(X_train, dataset)
                X_val = load_patches(X_val, dataset)
                X_test = load_patches(X_test, dataset)
        else:
            if split == None:  # cross-corpus mixture of 3
                if cc == "mix":
                    X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes = (
                        load_mix3_corpus_image(method, corpus)
                    )
            else:  # cross-corpus mixture of 2
                X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes = (
                    load_split_corpus_size_image(method, corpus, split, process)
                )

        if method in ["CNN", "Xception", "MLP"]:
            return X_train, ytrain, X_val, yval, X_test, ytest, h, num_classes
        elif method in ["ViT"]:
            return X_train, ytrain, X_val, yval, X_test, ytest, num_classes


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
    """
    description: This function is the general function for model loading.
    param {*} task: image or speech
    param {*} method: model name
    param {*} features: features selected
    param {*} cc: mix/cross/finetune
    param {*} shape: shape of audio input
    param {*} num_classes: number of classes
    parma {*} dataset: name of dataset
    param {*} max_length: max length for CNN
    param {*} bidirectional: whether construct bidirectional for RNN
    param {*} epochs: number of epochs for NN
    param {*} lr: learning rate for NN
    param {*} batch_size: batch size for NN
    param {*} cv: whether cross-validation
    param {*} h: dimension of image
    param {*} landmark: landmark choices
    return {*}: constructed model
    """
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
        elif method == "DBSCAN":
            model = DBSCAN(task, method, features, cc, dataset, num_classes)
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
        elif method == "Xception":
            model = Xception(
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
    """
    description: This function is used for visualizing confusion matrix.
    param {*} task: speech/image
    param {*} method: selected model
    param {*} features: features selected
    param {*} cc: mix/finetune/cross
    param {*} dataset: name of dataset
    param {*} ytrain: train ground truth
    param {*} yval: validation ground truth
    param {*} ytest: test ground truth
    param {*} train_pred: train prediction
    param {*} val_pred: validation prediction
    param {*} test_pred: test prediction
    param {*} ytune_train: finetuning train ground truth
    param {*} ytune_val: finetuning validation ground truth
    param {*} tune_train_pred: finetuning train prediction
    param {*} tune_val_pred: finetuning validation prediction
    param {*} corpus: depend on cross-corpus experiment setups
    param {*} split: cross-corpus 0.8/0.6/0.5/0.4/0.2, single-corpus 4/3/2.5/2/1
    param {*} landmark: landmark choices
    param {*} process: processing choices of operations
    """

    if cc != "finetune":
        cms = {
            "train": confusion_matrix(ytrain, train_pred),
            "val": confusion_matrix(yval, val_pred),
            "test": confusion_matrix(ytest, test_pred),
        }
    else:  # finetune
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

    # save
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


def visual4tree(task, method, features, cc, dataset, model):
    """
    description: This function is used for visualizing decision trees.
    param {*} task: speech/image
    param {*} method: selected model
    param {*} features: features selected
    param {*} cc: mix/finetune/cross
    param {*} dataset: name of dataset
    param {*} model: constructed tree model
    """
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


def get_metrics(task, y, pred):
    """
    description: This function is used for calculating balanced metrics.
    param {*} task: speech/image
    param {*} y: ground truth
    param {*} pred: predicted labels
    """
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
    """
    description: This function is used for calculating imbalanced metrics.
    param {*} task: speech/image
    param {*} y: ground truth
    param {*} pred: predicted labels
    """

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
        "likelihood": [round(LR_pos, 4), round(LR_neg, 4)],
    }
    print(classification_report_imbalanced(np.array(y).astype(int), pred.astype(int)))

    return result


def hyperpara_selection(task, method, feature, cc, dataset, scores):
    """
    description: This function is used for visualizing hyperparameter selection for grid search models.
    param {*} task: speech/image
    param {*} method: selected model
    param {*} scores: mean test score for cross validation of different parameter combinations
    """
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


def visaul4curves(task, method, feature, cc, dataset, train_res, val_res, epochs):
    """
    description: This function visualizes curves of convergence for loss and accuracy along epochs.
    param {*} task: speech/image
    param {*} method: name of model
    param {*} feature: feature selected
    param {*} cc: mix/finetune/cross
    param {*} dataset: name of dataset
    param {*} train_res: loss and accuracy stored along epochs in training
    param {*} val_res: loss and accuracy stored along epochs in training
    param {*} epochs: epochs for NN
    """
    acc = train_res["train_acc"]
    val_acc = val_res["val_acc"]
    epochs = list(range(epochs))

    # accuracy
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

    # loss
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
    """
    description: This function visualizes features for audio as waveform, mel-spectrogram, chromogram, etc.
    param {*} filename: name for audio
    param {*} dataset: name for dataset
    param {*} emotion: name for emotion
    """
    with soundfile.SoundFile(filename) as sound_file:
        data = sound_file.read(dtype="float32")
        if dataset == "eNTERFACE":
            data = data[:, 1]
        sr = sound_file.samplerate
    sound_file.close()

    # waveform
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(
        data, sr=sr, color="blue"
    )  # visualize wave in the time domain
    if not os.path.exists(f"outputs/speech/features/"):
        os.makedirs(f"outputs/speech/features/")
    plt.savefig(f"outputs/speech/features/{dataset}_{emotion}_waveform.png")
    plt.close()

    # spectrum
    x = librosa.stft(data)
    # frequency domain: The STFT represents a signal in the time-frequency domain
    # by computing discrete Fourier transforms (DFT) over short overlapping windows.
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
    """
    description: This function visualizes label proportion.
    param {*} task: image/speech
    param {*} dataset: name of dataset
    param {*} category: category mapping for emotion and label
    """
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

    # save
    if not os.path.exists(f"outputs/{task}/emotion_labels/"):
        os.makedirs(f"outputs/{task}/emotion_labels/")
    fig.savefig(f"outputs/{task}/emotion_labels/{dataset}.png")
    plt.close()


def visual4corrmatrix(feature, dataset):
    """
    description: This function visualizes correlation matrix for each dataset of 40 mfcc features.
    param {*} feature: features
    param {*} dataset: name of dataset
    """
    fea = []
    for i in feature:
        fea.append(i.tolist())
    corr_matrix = np.corrcoef(fea, rowvar=False)
    # Finally if we use the option rowvar=False,
    # the columns are now being treated as the variables and
    # we will find the column-wise Pearson correlation
    # coefficients between variables in xarr and yarr.

    plt.figure(figsize=(10, 8))
    hm = sns.heatmap(data=corr_matrix, annot=False)
    plt.title(f"Correlation matrix of {dataset}")
    if not os.path.exists(f"outputs/speech/corr_matrix/"):
        os.makedirs(f"outputs/speech/corr_matrix/")
    plt.savefig(f"outputs/speech/corr_matrix/{dataset}.png")
    plt.close()

    # store the matrix as txt
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


def visual4corr_signal(dataset, file1, file2, emotion):
    """
    description: This function visualizes auto-correlation, cross-correlation for two signal of same emotion.
    param {*} dataset: name of dataset
    param {*} file1: file of signal 1
    param {*} file2: file of signal 2
    param {*} emotion: emotion name
    """
    # read audio data
    sample_rate, data1 = wavfile.read(file1)
    sample_rate, data2 = wavfile.read(file2)
    if dataset == "eNTERFACE":
        data1 = data1[:, 0]
        data2 = data2[:, 0]
    data1 = data1.astype(float)
    data2 = data2.astype(float)
    data1 /= np.max(np.abs(data1))
    data2 /= np.max(np.abs(data2))

    # auto-correlation and cross-correlation
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

    # plot
    plt.figure(figsize=(10, 16))
    fig, (ax1, ax2, ax3, ax4, ax_corr) = plt.subplots(5, 1, sharex=True)
    ax1.plot(data1)
    ax2.plot(data2)
    ax3.plot(auto_lags1, auto_correlation1)
    ax4.plot(auto_lags2, auto_correlation2)
    ax_corr.plot(cross_lags, cross_correlation)
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    if not os.path.exists(f"outputs/speech/corr_signal/"):
        os.makedirs(f"outputs/speech/corr_signal/")
    fig.savefig(f"outputs/speech/corr_signal/{dataset}_{emotion}.png")
    plt.close()

    # most artificially to most real
    # (RAVDESS/TESS), (SAVEE, EmoDB, AESDD, CREMA-D), eNTERFACE05


def visual4corr_filter_ma(dataset, file1, file2, emotion):
    """
    description: This function visualizes auto-correlation, cross-correlation for two signal of same emotion after filtering and MA.
    param {*} dataset: name of dataset
    param {*} file1: file of signal 1
    param {*} file2: file of signal 2
    param {*} emotion: emotion name
    """
    sample_rate, data1 = wavfile.read(file1)
    sample_rate, data2 = wavfile.read(file2)
    if dataset == "eNTERFACE":
        data1 = data1[:, 0]
        data2 = data2[:, 0]

    # low pass filter
    fs = 1000.0
    cutoff = 5.0
    order = 5
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    fdata1 = lfilter(b, a, data1)
    fdata2 = lfilter(b, a, data2)

    fdata1 = fdata1.astype(float)
    fdata2 = fdata2.astype(float)
    fdata1 /= np.max(np.abs(fdata1))
    fdata2 /= np.max(np.abs(fdata2))

    cross_correlation = np.correlate(fdata1, fdata2, mode="full")
    cross_lags = np.arange(-len(fdata1) + 1, len(fdata2))
    auto_correlation1 = np.correlate(fdata1, fdata1, mode="full")
    auto_lags1 = np.arange(-len(fdata1) + 1, len(fdata1))
    auto_correlation2 = np.correlate(fdata2, fdata2, mode="full")
    auto_lags2 = np.arange(-len(fdata2) + 1, len(fdata2))

    # plot
    plt.figure(figsize=(10, 16))
    fig, (ax1, ax2, ax3, ax4, ax_corr) = plt.subplots(5, 1, sharex=True)
    ax1.plot(fdata1)
    ax2.plot(fdata2)
    ax3.plot(auto_lags1, auto_correlation1)
    ax4.plot(auto_lags2, auto_correlation2)
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

    # moving average
    window_size = 100  # Window size for moving average
    ma_data1 = np.convolve(data1, np.ones(window_size) / window_size, mode="valid")
    ma_data2 = np.convolve(data2, np.ones(window_size) / window_size, mode="valid")
    # the larger the window size, the better
    ma_data1 = ma_data1.astype(float)
    ma_data2 = ma_data2.astype(float)
    ma_data1 /= np.max(np.abs(ma_data1))
    ma_data2 /= np.max(np.abs(ma_data2))

    cross_correlation = np.correlate(ma_data1, ma_data2, mode="full")
    cross_lags = np.arange(-len(ma_data1) + 1, len(ma_data2))
    auto_correlation1 = np.correlate(ma_data1, ma_data1, mode="full")
    auto_lags1 = np.arange(-len(ma_data1) + 1, len(ma_data1))
    auto_correlation2 = np.correlate(ma_data2, ma_data2, mode="full")
    auto_lags2 = np.arange(-len(ma_data2) + 1, len(ma_data2))

    # plot
    plt.figure(figsize=(10, 16))
    fig, (ax1, ax2, ax3, ax4, ax_corr) = plt.subplots(5, 1, sharex=True)
    ax1.plot(ma_data1)
    ax2.plot(ma_data2)
    ax3.plot(auto_lags1, auto_correlation1)
    ax4.plot(auto_lags2, auto_correlation2)
    ax_corr.plot(cross_lags, cross_correlation)
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    if not os.path.exists(f"outputs/speech/corr_signal_ma/"):
        os.makedirs(f"outputs/speech/corr_signal_ma/")
    fig.savefig(f"outputs/speech/corr_signal_ma/{dataset}_{emotion}.png")
    plt.close()


def signal_similarity(dataset, file1, file2, emotion):
    """
    description: This function visualizes signal similarity for two signal of same emotion.
    param {*} dataset: name of dataset
    param {*} file1: file of signal 1
    param {*} file2: file of signal 2
    param {*} emotion: emotion name
    """
    sample_rate, data1 = wavfile.read(file1)
    sample_rate, data2 = wavfile.read(file2)
    # print(sample_rate)

    weights = {
        "zcr_similarity": 0.2,
        "rhythm_similarity": 0.2,
        "chroma_similarity": 0.2,
        "energy_envelope_similarity": 0.1,
        "spectral_contrast_similarity": 0.1,
        "perceptual_similarity": 0.2,
    }

    audio_similarity = AudioSimilarity(
        file1, file2, sample_rate, weights, verbose=True, sample_size=1
    )
    similarity = {
        "score": audio_similarity.stent_weighted_audio_similarity(metrics="all")
    }
    print(f"Stent Weighted Audio Similarity of {emotion}: {similarity}")

    # plot
    audio_similarity.plot(
        metrics=None,
        option="all",
        figsize=(20, 8),
        color1="red",
        color2="green",
        dpi=1000,
        savefig=False,
        fontsize=20,
        label_fontsize=20,
        title_fontsize=20,
        alpha=0.5,
        title="Audio Similarity Metrics",
    )
    if not os.path.exists(f"outputs/speech/signal_similarity/"):
        os.makedirs(f"outputs/speech/signal_similarity/")
    plt.savefig(f"outputs/speech/signal_similarity/{dataset}_{emotion}.png")
    plt.close()

    return similarity


def visual4corrMAV(dataset, file1, file2, emotion):
    """
    description: This function visualizes auto-correlation, cross-correlation for two signal of same emotion after MAV.
    param {*} dataset: name of dataset
    param {*} file1: file of signal 1
    param {*} file2: file of signal 2
    param {*} emotion: emotion name
    """
    sample_rate, data1 = wavfile.read(file1)
    sample_rate, data2 = wavfile.read(file2)
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
    fs = 1000.0
    cutoff = 5.0
    order = 5
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    fdata1 = lfilter(b, a, ma_data1)
    fdata2 = lfilter(b, a, ma_data2)

    fdata1 = fdata1.astype(float)
    fdata2 = fdata2.astype(float)
    fdata1 /= np.max(np.abs(fdata1))
    fdata2 /= np.max(np.abs(fdata2))

    cross_correlation = np.correlate(fdata1, fdata2, mode="full")
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

    # plot
    plt.figure(figsize=(10, 16))
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax_corr) = plt.subplots(7, 1, sharex=True)
    ax1.plot(ori_data1)
    ax2.plot(ori_data2)
    ax3.plot(fdata1)
    ax4.plot(fdata2)
    ax5.plot(auto_lags1, auto_correlation1)
    ax6.plot(auto_lags2, auto_correlation2)
    ax_corr.plot(cross_lags, cross_correlation)
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    if not os.path.exists(f"outputs/speech/corr_signal_MAV/"):
        os.makedirs(f"outputs/speech/corr_signal_MAV/")
    fig.savefig(f"outputs/speech/corr_signal_MAV/{dataset}_{emotion}.png")
    plt.close()
