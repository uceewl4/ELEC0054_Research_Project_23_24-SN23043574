# """
# Author: uceewl4 uceewl4@ucl.ac.uk
# Date: 2024-03-25 22:12:23
# LastEditors: uceewl4 uceewl4@ucl.ac.uk
# LastEditTime: 2024-03-28 21:12:37
# FilePath: /ELEC0054_Research_Project_23_24-SN23043574/tmp.py
# Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
# """

# from pydub import AudioSegment

# # 加载WAV文件
# sound = AudioSegment.from_file(
#     "datasets/speech/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav", format="wav"
# )

# # 倒放音频
# reversed_sound = sound.reverse()

# # 导出倒放后的音频到新文件
# reversed_sound.export("tmp.wav", format="wav")


import numpy as np
from scipy.io import wavfile
import soundfile

import librosa

with soundfile.SoundFile(
    "datasets/speech/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
) as sound_file:
    X = sound_file.read(dtype="float32")
    sample_rate = sound_file.samplerate

random_values = np.random.rand(len(X))
print(X)
print(random_values)
X = X + 2e-2 * random_values
soundfile.write("tmp.wav", X, sample_rate)

# # import librosa

# # import scipy.signal as signal

# # import numpy as np

# # y, sr = librosa.load("tmp.wav")
# # from pydub import AudioSegment

# # D = librosa.amplitude_to_db(np.abs(librosa.stft(y)) ** 2, ref=np.max)
# # # 应用Wiener滤波器进行去噪
# # D_denoised = signal.wiener(D, 1)

# # # 将去噪后的功率谱转换回音频信号
# # y_denoised = librosa.istft(np.exp(librosa.db_to_amplitude(D_denoised)))
# # soundfile.write("tmp2.wav", y_denoised, sr)

# import numpy as np
# import librosa
# import soundfile as sf
# from scipy.signal import wiener

# # # Load audio file
# # audio_file = "tmp.wav"
# # y, sr = librosa.load(audio_file, sr=None)
# # # Estimate noise spectrum using a specific window size
# # window_size = 40960  # Adjust the window size as needed  # 1024
# # noise_spectrum = np.abs(librosa.stft(y, n_fft=window_size, window="hann")).mean(axis=1)
# # # Apply Wiener filter for noise reduction
# # filtered_audio = wiener(y)

# # # Save the filtered audio
# # output_file = "tmp2.wav"
# # sf.write(output_file, filtered_audio, sr)
# # print("Filtered audio saved as", output_file)

# # import numpy as np
# # import librosa
# # import soundfile as sf

# # # Load audio file
# # audio_file = "tmp.wav"
# # y, sr = librosa.load(audio_file, sr=None)

# # # Compute the short-time Fourier transform (STFT)
# # stft_matrix = librosa.stft(y)

# # # Estimate the magnitude spectrum of the noise
# # noise_threshold = 0.05  # Adjust this threshold as needed
# # noise_spectrum = np.abs(stft_matrix)[
# #     :, np.abs(stft_matrix).max(axis=1) < noise_threshold * np.abs(stft_matrix).max()
# # ]

# # # Compute the mean noise spectrum
# # mean_noise_spectrum = np.mean(noise_spectrum, axis=1)

# # # Subtract the estimated noise spectrum from the magnitude spectrum of the original signal
# # cleaned_stft = np.maximum(np.abs(stft_matrix) - mean_noise_spectrum[:, np.newaxis], 0)

# # # Inverse STFT to obtain the cleaned audio signal
# # cleaned_audio = librosa.istft(cleaned_stft)

# # # Save the cleaned audio
# # output_file = "tmp2.wav"
# # sf.write(output_file, cleaned_audio, sr)
# # print("Cleaned audio saved as", output_file)

# import numpy as np
# import soundfile as sf
# from scipy.signal import spectrogram

# # Load audio file
# audio_file = "tmp.wav"
# y, sr = sf.read(audio_file)

# # Compute the spectrogram of the noisy signal
# frequencies, times, spectrogram_data = spectrogram(y, fs=sr, window="hann")

# # Estimate the noise spectrum (e.g., using a minimum statistics approach)
# noise_spectrum = np.min(spectrogram_data, axis=1)

# # Apply spectral subtraction
# alpha = 20  # Adjust as needed (experiment with different values)
# cleaned_spectrogram = np.maximum(
#     spectrogram_data - alpha * noise_spectrum[:, np.newaxis], 0
# )

# # Reconstruct the cleaned signal from the modified spectrogram
# cleaned_signal = np.fft.irfft(cleaned_spectrogram, axis=0)

# # Ensure the cleaned signal has the same length as the original signal
# cleaned_signal = cleaned_signal[: len(y)]
# print(y)

# # Save the cleaned audio
# # output_file = "tmp2.wav"
# # sf.write(output_file, cleaned_signal, sr)
# print(cleaned_signal)
# soundfile.write("tmp3.wav", cleaned_signal.flatten(), sr)

# print("Cleaned audio saved as", output_file)


from pydub import AudioSegment
import noisereduce as nr

# 加载音频文件
audio = AudioSegment.from_file("tmp.wav")

samples = np.array(audio.get_array_of_samples())

# Reduce noise
reduced_noise = nr.reduce_noise(samples, sr=audio.frame_rate)

# Convert reduced noise signal back to audio
reduced_audio = AudioSegment(
    reduced_noise.tobytes(),
    frame_rate=audio.frame_rate,
    sample_width=audio.sample_width,
    channels=audio.channels,
)

# Save reduced audio to file
reduced_audio.export("tmp2.wav", format="wav")


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
                x = transform_feature(x, features, n_mfcc, n_mels, scaled)

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
                Xtune_train, X_left, ytune_train, y_left = train_test_split(
                    np.array(x)[sample_index, :], y, test_size=0.4, random_state=9
                )  # 3:2

                Xtune_val, X_test, ytune_val, ytest = train_test_split(
                    X_left, yleft, test_size=0.5, random_state=9
                )  # 1:1
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
