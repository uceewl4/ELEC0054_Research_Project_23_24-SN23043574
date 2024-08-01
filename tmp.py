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


# from matplotlib import pyplot as plt
# from matplotlib.colors import Normalize
# import numpy as np
# from scipy.io import wavfile
# import soundfile

# import librosa


# def visual4feature(data, sr, name):

#     # waveform, spectrum (specshow)
#     plt.figure(figsize=(10, 4))
#     librosa.display.waveshow(
#         data, sr=sr, color="blue"
#     )  # visualize wave in the time domain
#     plt.savefig(f"tmp_waveform_{name}.png")
#     plt.close()

#     x = librosa.stft(
#         data
#     )  # frequency domain: The STFT represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short overlapping windows.
#     xdb = librosa.amplitude_to_db(
#         abs(x)
#     )  # Convert an amplitude spectrogram to dB-scaled spectrogram.
#     plt.figure(figsize=(11, 4))
#     librosa.display.specshow(
#         xdb, sr=sr, x_axis="time", y_axis="hz"
#     )  # visualize wave in the frequency domain
#     plt.colorbar()
#     plt.savefig(f"tmp_spectrum_{name}.png")
#     plt.close()

#     # mfcc spectrum
#     mfc_coefficients = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(
#         mfc_coefficients, x_axis="time", norm=Normalize(vmin=-30, vmax=30)
#     )
#     plt.colorbar()
#     plt.yticks(())
#     plt.ylabel("MFC Coefficient")
#     plt.tight_layout()
#     plt.savefig(f"tmp_mfcc_{name}.png")
#     plt.close()

#     # mel spectrum
#     melspectrogram = librosa.feature.melspectrogram(
#         y=data, sr=sr, n_mels=128, fmax=8000
#     )
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(
#         librosa.power_to_db(S=melspectrogram, ref=np.mean),
#         y_axis="mel",
#         fmax=8000,
#         x_axis="time",
#         norm=Normalize(vmin=-20, vmax=20),
#     )
#     plt.colorbar(format="%+2.0f dB", label="Amplitude")
#     plt.ylabel("Mels")
#     plt.tight_layout()
#     plt.savefig(f"tmp_mels_{name}.png")
#     plt.close()

#     # chroma spectrum
#     chromagram = librosa.feature.chroma_stft(y=data, sr=sr)
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(chromagram, y_axis="chroma", x_axis="time")
#     plt.colorbar(label="Relative Intensity")
#     plt.tight_layout()
#     plt.savefig(f"tmp_chroma_{name}.png")
#     plt.close()


# with soundfile.SoundFile("datasets/speech/eNTERFACE05/s_3_sa_1.wav") as sound_file:
#     X = sound_file.read(dtype="float32")
#     sample_rate = sound_file.samplerate
# print(sample_rate)
# print(X)
# print(np.array(X).shape)
# Y = X[:, 0]
# soundfile.write("single_channel_eNTERFACE.wav", Y, sample_rate)
# Z = X[:, 1]
# soundfile.write("single_channel_eNTERFACE2.wav", Z, sample_rate)

# visual4feature(Y, sample_rate, "eNTERFACE")
# visual4feature(Z, sample_rate, "eNTERFACE2")


# # random_values = np.random.rand(len(X))
# # print(X)
# # print(random_values)
# # X = X + 2e-2 * random_values
# # soundfile.write("tmp.wav", X, sample_rate)

# # from pydub import AudioSegment

# # # 加载音频文件
# # audio = AudioSegment.from_file("datasets/speech/EmoDB/03a01Fa.wav", format="wav")
# # new_sample_rate = 3
# # audio.set_frame_rate(new_sample_rate)
# # audio.export("tmp2.wav", format="wav")
# # import librosa
# # y, sr = librosa.load('datasets/speech/EmoDB/03a01Fa.wav')

# # # 设置新的采样率
# # new_sr = 16  # 例如，将采样率改为22050Hz

# # # 重新采样音频
# # y_resampled = librosa.resample(y, orig_sr=sr,target_sr=new_sr)

# # # 保存新的音频文件
# # soundfile.write("tmp.wav", y_resampled, new_sr)
# # # import librosa

# # # import scipy.signal as signal

# # # import numpy as np

# # # y, sr = librosa.load("tmp.wav")
# # # from pydub import AudioSegment

# # # D = librosa.amplitude_to_db(np.abs(librosa.stft(y)) ** 2, ref=np.max)
# # # # 应用Wiener滤波器进行去噪
# # # D_denoised = signal.wiener(D, 1)

# # # # 将去噪后的功率谱转换回音频信号
# # # y_denoised = librosa.istft(np.exp(librosa.db_to_amplitude(D_denoised)))
# # # soundfile.write("tmp2.wav", y_denoised, sr)

# # import numpy as np
# # import librosa
# # import soundfile as sf
# # from scipy.signal import wiener

# # # # Load audio file
# # # audio_file = "tmp.wav"
# # # y, sr = librosa.load(audio_file, sr=None)
# # # # Estimate noise spectrum using a specific window size
# # # window_size = 40960  # Adjust the window size as needed  # 1024
# # # noise_spectrum = np.abs(librosa.stft(y, n_fft=window_size, window="hann")).mean(axis=1)
# # # # Apply Wiener filter for noise reduction
# # # filtered_audio = wiener(y)

# # # # Save the filtered audio
# # # output_file = "tmp2.wav"
# # # sf.write(output_file, filtered_audio, sr)
# # # print("Filtered audio saved as", output_file)

# # # import numpy as np
# # # import librosa
# # # import soundfile as sf

# # # # Load audio file
# # # audio_file = "tmp.wav"
# # # y, sr = librosa.load(audio_file, sr=None)

# # # # Compute the short-time Fourier transform (STFT)
# # # stft_matrix = librosa.stft(y)

# # # # Estimate the magnitude spectrum of the noise
# # # noise_threshold = 0.05  # Adjust this threshold as needed
# # # noise_spectrum = np.abs(stft_matrix)[
# # #     :, np.abs(stft_matrix).max(axis=1) < noise_threshold * np.abs(stft_matrix).max()
# # # ]

# # # # Compute the mean noise spectrum
# # # mean_noise_spectrum = np.mean(noise_spectrum, axis=1)

# # # # Subtract the estimated noise spectrum from the magnitude spectrum of the original signal
# # # cleaned_stft = np.maximum(np.abs(stft_matrix) - mean_noise_spectrum[:, np.newaxis], 0)

# # # # Inverse STFT to obtain the cleaned audio signal
# # # cleaned_audio = librosa.istft(cleaned_stft)

# # # # Save the cleaned audio
# # # output_file = "tmp2.wav"
# # # sf.write(output_file, cleaned_audio, sr)
# # # print("Cleaned audio saved as", output_file)

# # import numpy as np
# # import soundfile as sf
# # from scipy.signal import spectrogram

# # # Load audio file
# # audio_file = "tmp.wav"
# # y, sr = sf.read(audio_file)

# # # Compute the spectrogram of the noisy signal
# # frequencies, times, spectrogram_data = spectrogram(y, fs=sr, window="hann")

# # # Estimate the noise spectrum (e.g., using a minimum statistics approach)
# # noise_spectrum = np.min(spectrogram_data, axis=1)

# # # Apply spectral subtraction
# # alpha = 20  # Adjust as needed (experiment with different values)
# # cleaned_spectrogram = np.maximum(
# #     spectrogram_data - alpha * noise_spectrum[:, np.newaxis], 0
# # )

# # # Reconstruct the cleaned signal from the modified spectrogram
# # cleaned_signal = np.fft.irfft(cleaned_spectrogram, axis=0)

# # # Ensure the cleaned signal has the same length as the original signal
# # cleaned_signal = cleaned_signal[: len(y)]
# # print(y)

# # # Save the cleaned audio
# # # output_file = "tmp2.wav"
# # # sf.write(output_file, cleaned_signal, sr)
# # print(cleaned_signal)
# # soundfile.write("tmp3.wav", cleaned_signal.flatten(), sr)

# # print("Cleaned audio saved as", output_file)

# import numpy as np
# from pydub import AudioSegment

# # # 加载音频文件
# # audio = AudioSegment.from_file(
# #     "datasets/speech/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
# # )
# # duration, frequency, amplitude = 10, 100, 20000
# # t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
# # buzzing_wave = amplitude * np.sin(2 * np.pi * frequency * t)
# # buzzing_wave = buzzing_wave.astype(np.int16)
# # buzzing_noise = AudioSegment(
# #     buzzing_wave.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1
# # )
# # audio_with_noise = audio.overlay(buzzing_noise)
# # audio_with_noise.export("tmp2.wav", format="wav")


# # import numpy as np
# # import librosa
# # import soundfile as sf
# # import numpy as np
# # import scipy.signal as signal
# # import soundfile as sf


# # def add_bubble_noise(signal, noise_level=0.1):
# #     """
# #     Adds bubble-like noise to a signal.

# #     Parameters:
# #         signal (ndarray): The input signal.
# #         noise_level (float): The level of noise to be added. Should be between 0 and 1.

# #     Returns:
# #         ndarray: Signal with added bubble noise.
# #     """
# #     noise = np.random.normal(0, noise_level, len(signal))
# #     return signal + noise


# # # Load audio file
# # audio_file = "datasets/speech/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
# # signal, sr = librosa.load(audio_file, sr=None)

# # # Add bubble noise to the audio signal
# # noisy_signal = add_bubble_noise(
# #     signal, noise_level=0.01
# # )  # Adjust noise level as needed

# # # Save the noisy audio file
# # output_file = "tmp4.wav"
# # sf.write(output_file, noisy_signal, sr)

# # print("Bubble noise added to the audio file successfully!")

# # with soundfile.SoundFile(
# #     "datasets/speech/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
# # ) as sound_file:
# #     X = sound_file.read(dtype="float32")
# #     sample_rate = sound_file.samplerate

# #     random_values = np.random.rand(len(X))
# #     #     if dataset != "eNTERFACE"
# #     #     else np.random.rand(len(X), 2)
# #     # )
# #     X = X + 2e-2 * random_values

# # # if noise == "white":
# # soundfile.write(f"tmp5.wav", X, sample_rate)


# # import numpy as np
# # import librosa
# # import soundfile as sf


# # Load original audio file
# # original_audio_file = "datasets/speech/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
# # original_audio, sr = librosa.load(original_audio_file, sr=None)
# # # Generate bubble noise with the same duration and sample rate as the original audio
# # duration = len(original_audio) / sr
# # bubble_frequency_range = (1000, 5000)
# # bubble_duration_range = (0.05, 0.5)
# # amplitude_range = (0.05, 0.1)
# # # Generate random parameters for each bubble
# # num_bubbles = int(
# #     duration * np.random.uniform(1, 10)
# # )  # Adjust number of bubbles based on duration
# # frequencies = np.random.uniform(*bubble_frequency_range, size=num_bubbles)
# # durations = np.random.uniform(*bubble_duration_range, size=num_bubbles)
# # amplitudes = np.random.uniform(*amplitude_range, size=num_bubbles)
# # # Generate bubble noise signal
# # t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
# # bubble_noise = np.zeros_like(t)
# # for freq, dur, amp in zip(frequencies, durations, amplitudes):
# #     envelope = signal.gaussian(int(dur * sample_rate), int(dur * sample_rate / 4))
# #     bubble = amp * np.sin(
# #         2 * np.pi * freq * np.linspace(0, dur, int(dur * sample_rate))
# #     )
# #     start_idx = np.random.randint(0, len(t) - len(bubble))
# #     bubble_noise[start_idx : start_idx + len(bubble)] += bubble * envelope
# # noisy_audio = original_audio + bubble_noise
# # output_file = "tmp4.wav"
# # sf.write(output_file, noisy_audio, sr)
# # print("Bubble noise added to the original audio file and saved successfully!")
# # # Calculate SNR
# # signal_power = np.sum(original_audio**2) / len(original_audio)
# # noise_power = np.sum(bubble_noise**2) / len(bubble_noise)
# # snr = 10 * np.log10(signal_power / noise_power)

# # print("Signal-to-Noise Ratio (SNR): {:.2f} dB".format(snr))


# # ori_audio = AudioSegment.from_file(
# #     "datasets/speech/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
# # )
# # duration, frequency, amplitude, sample_rate = 10, 100, 20000, 16000
# # t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
# # buzzing_wave = amplitude * np.sin(2 * np.pi * frequency * t)
# # buzzing_wave = buzzing_wave.astype(np.int16)
# # buzzing_noise = AudioSegment(
# #     buzzing_wave.tobytes(),
# #     frame_rate=sample_rate,
# #     sample_width=2,
# #     channels=1,
# # )
# # audio_with_noise = ori_audio.overlay(buzzing_noise)
# # P_signal = np.mean(np.array(ori_audio.get_array_of_samples()) ** 2)

# # # Calculate noise power (mean squared amplitude)
# # P_noise = np.mean(np.array(buzzing_noise.get_array_of_samples()) ** 2)

# # # Calculate SNR in dB
# # SNR = 10 * np.log10(P_signal / P_noise)
# # print("Signal-to-Noise Ratio (SNR):", SNR, "dB")


# # with soundfile.SoundFile(
# #     "datasets/speech/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
# # ) as sound_file:
# #     X = sound_file.read(dtype="float32")
# #     sample_rate = sound_file.samplerate

# # random_values = (
# #     np.random.rand(len(X))
# #     # if dataset != "eNTERFACE"
# #     # else np.random.rand(len(X), 2)
# # )
# # X_noisy = X + 2e-2 * random_values

# # # Calculate signal power
# # signal_power = np.mean(X**2)

# # # Calculate noise power
# # noise_power = np.mean((2e-2 * random_values) ** 2)

# # # Calculate SNR in dB
# # SNR = 10 * np.log10(signal_power / noise_power)
# # print("Signal-to-Noise Ratio (SNR):", SNR, "dB")


# # import numpy as np
# # import librosa
# # import soundfile as sf

# # # Load the audio file
# # audio_file_path = "datasets/speech/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
# # audio_data, sample_rate = librosa.load(audio_file_path, sr=None)

# # # Parameters for babble noise
# # duration = len(audio_data) / sample_rate  # Duration of the audio in seconds
# # num_samples = len(audio_data)  # Total number of samples
# # num_sources = 1  # Number of sources contributing to the babble noise

# # # Generate babble noise
# # babble = np.zeros(num_samples)
# # for _ in range(num_sources):
# #     frequency = np.random.uniform(
# #         100, 1000
# #     )  # Random frequency between 100 Hz and 1000 Hz
# #     amplitude = np.random.uniform(0.1, 0.5)  # Random amplitude
# #     phase = np.random.uniform(0, 2 * np.pi)  # Random phase
# #     source = amplitude * np.sin(
# #         2 * np.pi * frequency * np.arange(num_samples) / sample_rate + phase
# #     )
# #     babble += source

# # # Adjust babble noise to match the amplitude of the audio signal
# # babble *= np.max(np.abs(audio_data)) / np.max(np.abs(babble))

# # # Mix audio and babble noise
# # mixed_audio = audio_data + babble

# # # Save the mixed audio
# # output_file_path = "mixed_audio.wav"
# # sf.write(output_file_path, mixed_audio, sample_rate)

# # print("Mixed audio saved successfully!")


# # import numpy as np
# # import librosa
# # import soundfile as sf


# # def generate_voice(duration, sample_rate, frequency, amplitude):
# #     phase = np.random.uniform(0, 2 * np.pi)  # Random phase
# #     voice = amplitude * np.sin(
# #         2 * np.pi * frequency * np.arange(duration * sample_rate) / sample_rate + phase
# #     )
# #     return voice


# # # Load the audio file
# # audio_file_path = "datasets/speech/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
# # audio_data, sample_rate = librosa.load(audio_file_path, sr=None)

# # # Parameters
# # duration = len(audio_data) / sample_rate  # Duration of the audio in seconds
# # num_samples = len(audio_data)  # Total number of samples
# # num_voices = 2  # Number of voices

# # # Define parameters for each voice
# # voice_params = [
# #     {"frequency": 300, "amplitude": 0.3},
# #     {"frequency": 500, "amplitude": 0.4},
# # ]

# # # Generate babble noise with two voices speaking together
# # babble = np.zeros(num_samples)
# # for params in voice_params:
# #     voice = generate_voice(
# #         duration, sample_rate, params["frequency"], params["amplitude"]
# #     )
# #     start_idx = np.random.randint(0, len(babble) - len(voice))
# #     babble[start_idx : start_idx + len(voice)] += voice

# # # Adjust babble noise to match the amplitude of the audio signal
# # babble *= np.max(np.abs(audio_data)) / np.max(np.abs(babble))

# # # Mix audio and babble noise
# # mixed_audio = audio_data + babble

# # # Save the mixed audio
# # output_file_path = "mixed_audio_with_babble.wav"
# # sf.write(output_file_path, mixed_audio, sample_rate)

# # # print("Mixed audio with babble noise saved successfully!")
# # import numpy as np
# # import librosa
# # import soundfile as sf

# # # Load the original audio file
# # original_audio_path = "datasets/speech/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
# # original_audio, sample_rate = librosa.load(original_audio_path, sr=None)

# # # Reverse the original audio
# # reversed_audio = original_audio[::-1]

# # # Mix the original audio with its reverse
# # mixed_audio = original_audio + reversed_audio

# # # Normalize mixed audio
# # mixed_audio /= np.max(np.abs(mixed_audio))

# # # Save the mixed audio
# # mixed_audio_path = "mixed_audio_with_reverse.wav"
# # sf.write(mixed_audio_path, mixed_audio, sample_rate)

# # print("Mixed audio with reverse added saved successfully!")


# def load_AESDD(
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
#     sr=16000,
#     split=None,
# ):
#     x, y, category, path, audio, lengths = [], [], [], [], [], []

#     # original class of ranging split
#     # emotion_map = {
#     #     "angry": 0,
#     #     "disgust": 1,
#     #     "fear": 2,
#     #     "happy": 3,
#     #     "neutral": 4,
#     #     "ps": 5,
#     #     "sad": 6,
#     # }

#     if split == None:
#         emotion_map = {
#             "anger": 0,
#             "disgust": 1,
#             "fear": 2,
#             "happiness": 3,
#             "sadness": 4,
#         }
#     else:
#         emotion_map = {
#             ("01", "02", "neutral", "n", "neu", "L", "N"): 0,  # neutral
#             (
#                 "03",
#                 "08",
#                 "happy",
#                 "ps",
#                 "h",
#                 "su",
#                 "hap",
#                 "F",
#                 "ha",
#                 "su",
#                 "happiness",
#             ): 1,  # positive
#             (
#                 "04",
#                 "05",
#                 "06",
#                 "07",
#                 "angry",
#                 "disgust",
#                 "fear",
#                 "sad",
#                 "a",
#                 "d",
#                 "f",
#                 "sa",
#                 "ang",
#                 "dis",
#                 "fea",
#                 "sad",
#                 "W",
#                 "E",
#                 "A",
#                 "T",
#                 "an",
#                 "di",
#                 "fe",
#                 "sa",
#                 "anger",
#                 "sadness",
#             ): 2,
#         }

#     for dirname, _, filenames in os.walk("datasets/speech/AESDD"):
#         for filename in filenames:
#             feature, X = get_features(
#                 "AESDD",
#                 method,
#                 os.path.join(dirname, filename),
#                 features,
#                 n_mfcc=n_mfcc,
#                 n_mels=n_mels,
#                 max_length=max_length,
#                 window=window,
#                 sr=sr,
#             )
#             # label = filename.split("_")[-1].split(".")[0]
#             if split == None:
#                 emotion = emotion_map[dirname]
#             else:
#                 for k, i in enumerate(emotion_map.keys()):
#                     # label = filename.split("_")[-1].split(".")[0]
#                     if dirname in i:
#                         emotion = emotion_map[i]

#             x.append(feature)
#             y.append(emotion)
#             path.append(os.path.join(dirname, filename))
#             category.append(dirname)
#             audio.append(X)
#             lengths.append(len(X))
#             if category.count(dirname) == 1:
#                 visual4feature(os.path.join(dirname, filename), "AESDD", dirname)

#         # if len(y) == 2800:
#         #     break

#     visual4label("speech", "AESDD", category)
#     print(np.array(x).shape)

#     if method != "wav2vec":
#         if scaled != None:
#             x = transform_feature(method, x, features, n_mfcc, n_mels, scaled)

#     if reverse == True:
#         x, y, audio, lengths = get_reverse(
#             x,
#             y,
#             audio,
#             lengths,
#             path,
#             "AESDD",
#             method,
#             features,
#             n_mfcc,
#             n_mels,
#             max_length,
#             500,
#             window,
#             sr,
#         )

#     if noise != None:
#         x, y, audio, lengths = get_noise(
#             x,
#             y,
#             audio,
#             lengths,
#             path,
#             "AESDD",
#             method,
#             features,
#             n_mfcc,
#             n_mels,
#             max_length,
#             300,
#             window,
#             sr,
#             noise,
#         )

#     if denoise == True:
#         x, y, audio, lengths = get_denoise(
#             x,
#             y,
#             audio,
#             lengths,
#             emotion_map,
#             "AESDD",
#             method,
#             features,
#             n_mfcc,
#             n_mels,
#             max_length,
#             window,
#             sr,
#         )

#     length = None if method != "wav2vec" else max(lengths)

#     if split == None:
#         if method != "wav2vec":
#             X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
#                 np.array(x), y, test_size=0.4, random_state=9
#             )  # 3:2
#         else:
#             feature_extractor = AutoFeatureExtractor.from_pretrained(
#                 "facebook/wav2vec2-base", return_attention_mask=True
#             )
#             X = feature_extractor(
#                 audio,
#                 sampling_rate=feature_extractor.sampling_rate,
#                 max_length=length,
#                 truncation=True,
#                 padding=True,
#             )  # (1440, 84351)
#             X_train, X_left, ytrain, yleft = train_test_split(
#                 np.array(X["input_values"]), y, test_size=0.4, random_state=9
#             )  # 3:2

#         X_val, X_test, yval, ytest = train_test_split(
#             X_left, yleft, test_size=0.5, random_state=9
#         )  # 1:1
#         # (1680, 40), (560, 40), (560, 40), (1680,)
#     else:  # split != None can only be used for AlexNet as single-corpus split experiment
#         """
#         # this one is used for original ranging split of proportion
#         X_train, X_test, ytrain, ytest = train_test_split(  # 2800, 1680, 1120
#             np.array(x), y, test_size=split, random_state=9
#         )  # 0.25
#         X_train, X_val, ytrain, yval = train_test_split(
#             X_train, ytrain, test_size=0.5, random_state=9
#         )  # 1:1 for train : val

#         """

#         # this is the new one after modification, which is used for corresponding split for single-corpus
#         # for cross-corpus-split-size

#         # test
#         random.seed(123)
#         test_index = random.sample(
#             [i for i in range(np.array(x).shape[0])], 200  # 200 fixed for testing size
#         )  # 1440, 40
#         left_index = [i for i in range(np.array(x).shape[0]) if i not in test_index]
#         X_test = np.array(x)[test_index, :]
#         ytest = np.array(y)[test_index].tolist()
#         X_left = np.array(x)[left_index, :]
#         yleft = np.array(y)[left_index].tolist()

#         # train/val
#         random.seed(123)
#         train_index = random.sample(
#             [i for i in range(np.array(X_left).shape[0])], int(split * 200)
#         )  # train + val
#         X_train, X_val, ytrain, yval = train_test_split(
#             X_left[train_index, :],
#             np.array(yleft)[train_index].tolist(),
#             test_size=0.5,
#             random_state=9,
#         )  # 1:1 for train : val

#     return X_train, ytrain, X_val, yval, X_test, ytest, length

# from pydub import AudioSegment
# import cv2
# import os
# import random
# import numpy as np
# from matplotlib.colors import Normalize
# import librosa
# import soundfile
# import os, glob, pickle
# import numpy as np
# import numpy as np
# from scipy.io import wavfile
# import soundfile
# import librosa
# import numpy as np
# import librosa
# import soundfile as sf
# import numpy as np
# import scipy.signal as signal
# import soundfile as sf


# random.seed(123)

# for dirname, _, filenames in os.walk("datasets/speech/AESDD"):
#     for filename in filenames:
#         sound = AudioSegment.from_file(os.path.join(dirname, filename), format="wav")
#         reversed_sound = sound.reverse()
#         label = dirname.split("/")[-1]
#         name = filename.split(".")[0]
#         if not os.path.exists(f"datasets/speech/AESDD_reverse/{label}"):
#             os.makedirs(f"datasets/speech/AESDD_reverse/{label}")
#         reversed_sound.export(
#             f"datasets/speech/AESDD_reverse/{label}/{name}_reverse.wav",
#             format="wav",
#         )


# # importing the modules
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # generating 2-D 10x10 matrix of random numbers
# # from 1 to 100
# data = np.random.randint(low=1, high=100, size=(40, 10))

# print(data)
# # plotting the heatmap
# hm = sns.heatmap(data=data, annot=True)

# # displaying the plotted heatmap
# plt.show()

# mcc
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html


# # Function to read audio file
# def read_audio(filename):
#     sample_rate, data = wavfile.read(filename)
#     return sample_rate, data


# # Function to normalize audio data
# def normalize(data):
#     return data / np.max(np.abs(data))


# # Read the two audio files
# sample_rate1, data1 = read_audio("datasets/speech/TESS/OAF_angry/OAF_back_angry.wav")
# sample_rate2, data2 = read_audio(
#     "datasets/speech/TESS/OAF_disgust/OAF_back_disgust.wav"
# )

# # Ensure both audio files have the same sample rate
# if sample_rate1 != sample_rate2:
#     raise ValueError("Sample rates of the two audio files do not match.")

# # Normalize the audio data
# data1 = normalize(data1)
# data2 = normalize(data2)

# # Compute the cross-correlation
# correlation = np.correlate(data1, data2, mode="full")
# lags = np.arange(-len(data1) + 1, len(data2))

# # Plot the cross-correlation
# plt.figure(figsize=(10, 5))
# fig, (ax1, ax2, ax_corr) = plt.subplots(3, 1, sharex=True)
# ax_corr.plot(lags, correlation)
# ax1.plot(data1)
# ax2.plot(data2)
# plt.title("Cross-Correlation of Two Audio Signals")
# plt.xlabel("Lag")
# plt.ylabel("Correlation")
# plt.savefig("corr.png")


import cv2
import numpy as np
from keras.models import model_from_json

# here put the import lib
import os
import cv2
import smtplib
import numpy as np
from PIL import Image
from io import BytesIO
import streamlit as st
from email.header import Header
from email.mime.text import MIMEText

# from streamlit_option_menu import option_menu
# from keras.models import Sequential
# import tensorflow as tf
# from tensorflow.keras import Model
import cv2
import numpy as np
from keras.models import model_from_json

# from utils import get_features, load_data, load_model
# import cv2
# import mediapipe as mp
# import streamlit as st
# import sounddevice as sd
# import numpy as np
# import scipy.io.wavfile as wav


# emotion_dict = {
#     0: "Anger",
#     1: "Contempt",
#     2: "Disgust",
#     3: "Fear",
#     4: "Happy",
#     5: "Sadness",
#     6: "Surprise",
# }

# emotion_model = tf.keras.models.load_model("outputs/image/models/CNN.h5")
# print("Loaded model from disk")

# # start the webcam feed
# # cap = cv2.VideoCapture(0)

# # pass here your video path
# # you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
# cap = cv2.VideoCapture(
#     "/Users/anlly/Desktop/ucl/Final_Project/Automatic_Emotion_Detection_Trials/Emotion_detection_with_CNN-main/tmp.mp4"
# )
# while True:
#     # Find haar cascade to draw bounding box around face
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, (1280, 720))

#     if not ret:
#         break
#     face_detector = cv2.CascadeClassifier(
#         "/Users/anlly/Desktop/ucl/Final_Project/ELEC0054_Research_Project_23_24-SN23043574/outputs/image/models/haarcascades/haarcascade_frontalface_default.xml",
#     )  # crop the face
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # detect faces available on camera
#     num_faces = face_detector.detectMultiScale(
#         gray_frame, scaleFactor=1.3, minNeighbors=5
#     )

#     print(num_faces)
#     # take each face available on the camera and Preprocess it
#     for x, y, w, h in num_faces:  # face bounding boxes, rectangle
#         cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
#         roi_gray_frame = gray_frame[
#             y : y + h, x : x + w
#         ]  # crop each face into one gray frame
#         cropped_image = np.expand_dims(
#             np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0
#         )  # resize into the images

#         # predict the emotions
#         emotion_prediction = emotion_model.predict(cropped_image)
#         maxindex = int(np.argmax(emotion_prediction))
#         print(maxindex)
#         cv2.putText(
#             frame,
#             emotion_dict[maxindex],
#             (x + 5, y - 20),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (255, 0, 0),
#             2,
#             cv2.LINE_AA,
#         )

#         face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)

#     cv2.imshow("Emotion Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()


# def process(file):
#     image = cv2.imread(file)
#     cv2.imwrite(f"outputs/tmp/{file.split(".")[0].split("/")[-1]}.png", image)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image = cv2.equalizeHist(image)
#     image = cv2.GaussianBlur(image, (3, 3), 0)

#     # sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
#     # sobelx = cv2.convertScaleAbs(sobelx)
#     # sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
#     # sobely = cv2.convertScaleAbs(sobely)
#     # sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
#     # image = cv2.addWeighted(image, 1, sobel, 1, 0)

#     # image = cv2.equalizeHist(image)
#     print(file.split(".")[0])
#     print(file.split(".")[0].split("/")[-1])
#     cv2.imwrite(f"outputs/tmp/{file.split(".")[0].split("/")[-1]}_assi.png", image)


# image = process("datasets/image/CK/anger/S011_004_00000020.png")
# image = process("datasets/image/FER/train/angry/Training_364963.jpg")
# image = process("datasets/image/RAF/train/1/train_00852_aligned.jpg")
# image = process("datasets/image/CK/contempt/S147_002_00000012.png")
# image = process("datasets/image/FER/train/fear/Training_737388.jpg")
# image = process("datasets/image/RAF/train/5/train_00199_aligned.jpg")


import cv2
import numpy as np

# # Function to calculate the variance of the Laplacian
# def variance_of_laplacian(image):
#     # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
#     return laplacian_var

# # Function to apply Gaussian blur iteratively
# def apply_gaussian_blur_to_target(image, target_blur, max_iterations=10, epsilon=100):
#     current_image = image.copy()
#     current_blur = variance_of_laplacian(current_image)
#     print(f"Current Blur: {current_blur}")
#     ksize = 3

#     iteration = 0
#     while (current_blur - target_blur) > epsilon and iteration < max_iterations:
#         blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
#         current_blur = variance_of_laplacian(blurred_image)
#         current_image = blurred_image
#         iteration += 1
#         ksize += 2  # Increment kernel size to increase blur effect
#         print(f"Iteration: {iteration}, Kernel Size: {ksize}, Current Blur: {current_blur}")

#     return current_image

# # Load an image
# def process(file):
#     image = cv2.imread(file)

#     # Define the target blur extent
#     target_blur = 500.0  # Example target value, adjust as needed

#     # Apply Gaussian blur to reach the target blur extent
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image = cv2.equalizeHist(image)
#     blurred_image = apply_gaussian_blur_to_target(image, target_blur)

#     # Display the original and blurred images
#     # cv2.imshow("Original Image", image)
#     # cv2.imshow("Blurred Image", blurred_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     # Save the blurred image if needed
#     cv2.imwrite(f"outputs/tmp/{file.split(".")[0].split("/")[-1]}_assi_2.png", blurred_image)


# import cv2
# import numpy as np

# # Function to calculate the variance of the Laplacian
# def variance_of_laplacian(image):
#     # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
#     return laplacian_var

# # Function to apply Gaussian blur with a specified blur extent parameter
# def apply_gaussian_blur_with_extent(image, extent):
#     ksize = int(max(3, (extent * 10) // 2 * 2 + 1))  # Ensure ksize is odd and at least 3
#     sigma = extent * 2
#     blurred_image = cv2.GaussianBlur(image, (ksize, ksize), sigma)
#     return blurred_image

# def process(file):
#     # Load an image
#     image = cv2.imread(file)

#     # Measure the current variance of the Laplacian
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image = cv2.equalizeHist(image)

#     current_laplacian_var = variance_of_laplacian(image)
#     print(f"Current Variance of Laplacian: {current_laplacian_var}")

#     # Define the target blur extent using variance of Laplacian
#     target_blur = 2000.0  # Example target value, adjust as needed

#     # Calculate the extent parameter to reach the target variance of Laplacian
#     # Assuming a linear relationship, adjust the coefficient as needed
#     extent = np.sqrt(current_laplacian_var / target_blur)

#     # Apply Gaussian blur with the calculated extent

#     blurred_image = apply_gaussian_blur_with_extent(image, extent)

#     # Measure the final variance of the Laplacian
#     final_laplacian_var = variance_of_laplacian(blurred_image)
#     print(f"Final Variance of Laplacian Score: {final_laplacian_var}")

#     # Save the blurred image if needed
#     cv2.imwrite(f"outputs/tmp/{file.split(".")[0].split("/")[-1]}_assi_2.png", blurred_image)


import cv2
import numpy as np


# # Function to calculate the variance of the Laplacian
# def variance_of_laplacian(image):
#     # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
#     return laplacian_var


# # Function to apply Gaussian blur with a specified kernel size and sigma
# def apply_gaussian_blur(image, ksize, sigma):
#     blurred_image = cv2.GaussianBlur(image, (ksize, ksize), sigma)
#     return blurred_image


# # Function to determine the kernel size and sigma based on the target blur extent
# def calculate_blur_parameters(current_laplacian_var, target_blur):
#     # Determine the required extent to achieve the target blur extent
#     extent = np.sqrt(current_laplacian_var / target_blur)
#     # if current_laplacian_var < target_min:
#     #     # Increase blur if the current variance is less than the target minimum
#     #     extent = np.sqrt(target_min / current_laplacian_var)
#     # elif current_laplacian_var > target_max:
#     #     # Decrease blur if the current variance is more than the target maximum
#     #     extent = np.sqrt(current_laplacian_var / target_max)
#     # else:
#     #     # No adjustment needed if already within the range
#     #     extent = 1

#     # Map the extent parameter to a kernel size (must be an odd number)
#     ksize = int(
#         max(3, (extent * 10) // 2 * 2 + 1)
#     )  # Ensure ksize is odd and at least 3

#     # Map the extent parameter to a sigma value
#     sigma = extent * 2

#     return ksize, sigma


# def process(file):
#     # Load an image
#     image = cv2.imread(file)
#     image = cv2.resize(image, (48, 48))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image = cv2.equalizeHist(image)

#     # Define the target blur extent using variance of Laplacian
#     target_blur = 100000.0  # Example target value, adjust as needed
#     # target_min = 500
#     # target_max = 100000

#     # Calculate the current variance of the Laplacian
#     current_laplacian_var = variance_of_laplacian(image)
#     print(f"Current Variance of Laplacian: {current_laplacian_var}")

#     # Calculate the Gaussian blur parameters to reach the target blur extent
#     ksize, sigma = calculate_blur_parameters(current_laplacian_var, target_blur)

#     # Apply Gaussian blur with the calculated parameters
#     blurred_image = apply_gaussian_blur(image, ksize, sigma)

#     # Measure the final variance of the Laplacian
#     final_laplacian_var = variance_of_laplacian(blurred_image)
#     print(f"Final Variance of Laplacian Score: {final_laplacian_var}")

#     # Save the blurred image if needed
#     cv2.imwrite(
#         f"outputs/tmp/{file.split('.')[0].split('/')[-1]}_assi_2.png", blurred_image
#     )


# image = process("datasets/image/CK/disgust/S005_001_00000010.png")
# image = process("datasets/image/FER/train/disgust/Training_1070239.jpg")
# image = process("datasets/image/RAF/train/2/train_09375_aligned.jpg")
# image = process("datasets/image/CK/disgust/S131_010_00000017.png")
# image = process("datasets/image/FER/train/disgust/Training_3566649.jpg")
# image = process("datasets/image/RAF/train/2/train_00027_aligned.jpg")
# image = process("datasets/image/CK/anger/S011_004_00000020.png")
# image = process("datasets/image/FER/train/angry/Training_364963.jpg")
# image = process("datasets/image/RAF/train/1/train_00852_aligned.jpg")
# image = process("datasets/image/CK/contempt/S147_002_00000012.png")
# image = process("datasets/image/FER/train/fear/Training_737388.jpg")
# image = process("datasets/image/RAF/train/5/train_00199_aligned.jpg")


# image = cv2.resize(image, (48,48))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.equalizeHist(image)
# target_blur = 100000.0  # Example target value, adjust as needed
# current_laplacian_var = variance_of_laplacian(image)
# print(f"Current Variance of Laplacian: {current_laplacian_var}")

# extent = np.sqrt(current_laplacian_var / target_blur)
# ksize = int(max(3, (extent * 10) // 2 * 2 + 1))  # Ensure ksize is odd and at least 3
# sigma = extent * 2
# blurred_image = cv2.GaussianBlur(image, (ksize, ksize), sigma)
# final_laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
# print(f"Final Variance of Laplacian Score: {final_laplacian_var}")


def process(file):
    # noise extent: dct
    # image = cv2.imread(file)
    # image = cv2.resize(image, (48, 48))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # dct1 = cv2.dct(np.float32(image) / 255.0)
    # high_freq1 = dct1[4:, 4:]
    # noise_dct1 = np.std(high_freq1)
    # print(f"Previous Estimated Noise Level (DCT): {noise_dct1}")

    # image = cv2.equalizeHist(image)
    # image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    # dct2 = cv2.dct(np.float32(image) / 255.0)
    # high_freq2 = dct2[4:, 4:]
    # noise_dct2 = np.std(high_freq2)
    # print(f"Current Estimated Noise Level (DCT): {noise_dct2}")

    # cv2.imwrite(
    #     f"outputs/tmp/{file.split('.')[0].split('/')[-1]}_assi_111.png",
    #     image,
    # )

    # detail extent: entropy

    # image = cv2.imread(file)
    # image = cv2.resize(image, (48, 48))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # hist_norm = hist.ravel() / hist.sum()
    # entropy_value = -np.sum(hist_norm * np.log2(hist_norm + (hist_norm == 0)))
    # print(f"Previous Enrtopy Value: {entropy_value}")

    # image = cv2.equalizeHist(image)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # hist_norm = hist.ravel() / hist.sum()
    # entropy_value = -np.sum(hist_norm * np.log2(hist_norm + (hist_norm == 0)))
    # print(f"Current Enrtopy Value: {entropy_value}")

    # cv2.imwrite(
    #     f"outputs/tmp/{file.split('.')[0].split('/')[-1]}_assi_112.png",
    #     image,
    # )

    # clahe and gamma

    # image = cv2.imread(file)
    # image = cv2.resize(image, (48, 48))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # mean_intensity = np.mean(image)
    # print(f"Previous Brightness Value: {mean_intensity}")

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # image = clahe.apply(image)
    # gamma = 1.5
    # invGamma = 1.0 / gamma
    # table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype(
    #     "uint8"
    # )
    # image = cv2.LUT(image, table)

    # mean_intensity = np.mean(image)
    # print(f"Current Brightness Value: {mean_intensity}")

    # cv2.imwrite(
    #     f"outputs/tmp/{file.split('.')[0].split('/')[-1]}_assi_113.png",
    #     image,
    # )

    # edge detection
    image = cv2.imread(file)
    image = cv2.resize(image, (48, 48))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges_before = cv2.Canny(image, 100, 200)
    edge_pixels = np.count_nonzero(edges_before)
    total_pixels = image.size
    edge_density_before = edge_pixels / total_pixels
    print(f"Edge Density before Laplacian: {edge_density_before}")

    # Apply the Laplacian operator
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
    laplacian = np.clip(laplacian, 0, 255).astype("uint8")
    image = cv2.addWeighted(image, 1, laplacian, 1, 0)

    edges_after = cv2.Canny(image, 100, 200)
    edge_pixels = np.count_nonzero(edges_after)
    total_pixels = image.size
    edge_density_after = edge_pixels / total_pixels
    print(f"Edge Density after Laplacian: {edge_density_after}")

    cv2.imwrite(
        f"outputs/tmp/{file.split('.')[0].split('/')[-1]}_assi_114.png",
        image,
    )


image = process("datasets/image/CK/disgust/S005_001_00000010.png")
image = process("datasets/image/FER/train/disgust/Training_1070239.jpg")
image = process("datasets/image/RAF/train/2/train_09375_aligned.jpg")
image = process("datasets/image/CK/disgust/S131_010_00000017.png")
image = process("datasets/image/FER/train/disgust/Training_3566649.jpg")
image = process("datasets/image/RAF/train/2/train_00027_aligned.jpg")
image = process("datasets/image/CK/anger/S011_004_00000020.png")
image = process("datasets/image/FER/train/angry/Training_364963.jpg")
image = process("datasets/image/RAF/train/1/train_00852_aligned.jpg")
image = process("datasets/image/CK/contempt/S147_002_00000012.png")
image = process("datasets/image/FER/train/fear/Training_737388.jpg")
image = process("datasets/image/RAF/train/5/train_00199_aligned.jpg")
