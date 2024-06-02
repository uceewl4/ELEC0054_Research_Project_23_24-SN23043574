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


# importing the modules
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# generating 2-D 10x10 matrix of random numbers
# from 1 to 100
data = np.random.randint(low=1, high=100, size=(40, 10))

print(data)
# plotting the heatmap
hm = sns.heatmap(data=data, annot=True)

# displaying the plotted heatmap
plt.show()

# mcc
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
