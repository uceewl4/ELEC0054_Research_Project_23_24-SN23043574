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
