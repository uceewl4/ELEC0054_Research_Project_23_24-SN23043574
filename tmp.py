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
print(sample_rate)
# random_values = np.random.rand(len(X))
# print(X)
# print(random_values)
# X = X + 2e-2 * random_values
# soundfile.write("tmp.wav", X, sample_rate)

# from pydub import AudioSegment

# # 加载音频文件
# audio = AudioSegment.from_file("datasets/speech/EmoDB/03a01Fa.wav", format="wav")
# new_sample_rate = 3
# audio.set_frame_rate(new_sample_rate)
# audio.export("tmp2.wav", format="wav")
# import librosa
# y, sr = librosa.load('datasets/speech/EmoDB/03a01Fa.wav')

# # 设置新的采样率
# new_sr = 16  # 例如，将采样率改为22050Hz

# # 重新采样音频
# y_resampled = librosa.resample(y, orig_sr=sr,target_sr=new_sr)

# # 保存新的音频文件
# soundfile.write("tmp.wav", y_resampled, new_sr)
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

import numpy as np
from pydub import AudioSegment

# # 加载音频文件
# audio = AudioSegment.from_file(
#     "datasets/speech/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
# )
# duration, frequency, amplitude = 10, 100, 20000
# t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
# buzzing_wave = amplitude * np.sin(2 * np.pi * frequency * t)
# buzzing_wave = buzzing_wave.astype(np.int16)
# buzzing_noise = AudioSegment(
#     buzzing_wave.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1
# )
# audio_with_noise = audio.overlay(buzzing_noise)
# audio_with_noise.export("tmp2.wav", format="wav")


import numpy as np
import librosa
import soundfile as sf
import numpy as np
import scipy.signal as signal
import soundfile as sf


def add_bubble_noise(signal, noise_level=0.1):
    """
    Adds bubble-like noise to a signal.

    Parameters:
        signal (ndarray): The input signal.
        noise_level (float): The level of noise to be added. Should be between 0 and 1.

    Returns:
        ndarray: Signal with added bubble noise.
    """
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise


# # Load audio file
# audio_file = "datasets/speech/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
# signal, sr = librosa.load(audio_file, sr=None)

# # Add bubble noise to the audio signal
# noisy_signal = add_bubble_noise(
#     signal, noise_level=0.01
# )  # Adjust noise level as needed

# # Save the noisy audio file
# output_file = "tmp4.wav"
# sf.write(output_file, noisy_signal, sr)

# print("Bubble noise added to the audio file successfully!")

with soundfile.SoundFile(
    "datasets/speech/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
) as sound_file:
    X = sound_file.read(dtype="float32")
    sample_rate = sound_file.samplerate

    random_values = np.random.rand(len(X))
    #     if dataset != "eNTERFACE"
    #     else np.random.rand(len(X), 2)
    # )
    X = X + 2e-2 * random_values

# if noise == "white":
soundfile.write(f"tmp5.wav", X, sample_rate)


import numpy as np
import librosa
import soundfile as sf


# Load original audio file
# original_audio_file = "datasets/speech/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
# original_audio, sr = librosa.load(original_audio_file, sr=None)
# # Generate bubble noise with the same duration and sample rate as the original audio
# duration = len(original_audio) / sr
# bubble_frequency_range = (1000, 5000)
# bubble_duration_range = (0.05, 0.5)
# amplitude_range = (0.05, 0.1)
# # Generate random parameters for each bubble
# num_bubbles = int(
#     duration * np.random.uniform(1, 10)
# )  # Adjust number of bubbles based on duration
# frequencies = np.random.uniform(*bubble_frequency_range, size=num_bubbles)
# durations = np.random.uniform(*bubble_duration_range, size=num_bubbles)
# amplitudes = np.random.uniform(*amplitude_range, size=num_bubbles)
# # Generate bubble noise signal
# t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
# bubble_noise = np.zeros_like(t)
# for freq, dur, amp in zip(frequencies, durations, amplitudes):
#     envelope = signal.gaussian(int(dur * sample_rate), int(dur * sample_rate / 4))
#     bubble = amp * np.sin(
#         2 * np.pi * freq * np.linspace(0, dur, int(dur * sample_rate))
#     )
#     start_idx = np.random.randint(0, len(t) - len(bubble))
#     bubble_noise[start_idx : start_idx + len(bubble)] += bubble * envelope
# noisy_audio = original_audio + bubble_noise
# output_file = "tmp4.wav"
# sf.write(output_file, noisy_audio, sr)
# print("Bubble noise added to the original audio file and saved successfully!")
# # Calculate SNR
# signal_power = np.sum(original_audio**2) / len(original_audio)
# noise_power = np.sum(bubble_noise**2) / len(bubble_noise)
# snr = 10 * np.log10(signal_power / noise_power)

# print("Signal-to-Noise Ratio (SNR): {:.2f} dB".format(snr))


ori_audio = AudioSegment.from_file(
    "datasets/speech/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
)
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
P_signal = np.mean(np.array(ori_audio.get_array_of_samples()) ** 2)

# Calculate noise power (mean squared amplitude)
P_noise = np.mean(np.array(buzzing_noise.get_array_of_samples()) ** 2)

# Calculate SNR in dB
SNR = 10 * np.log10(P_signal / P_noise)
print("Signal-to-Noise Ratio (SNR):", SNR, "dB")


# with soundfile.SoundFile(
#     "datasets/speech/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
# ) as sound_file:
#     X = sound_file.read(dtype="float32")
#     sample_rate = sound_file.samplerate

# random_values = (
#     np.random.rand(len(X))
#     # if dataset != "eNTERFACE"
#     # else np.random.rand(len(X), 2)
# )
# X_noisy = X + 2e-2 * random_values

# # Calculate signal power
# signal_power = np.mean(X**2)

# # Calculate noise power
# noise_power = np.mean((2e-2 * random_values) ** 2)

# # Calculate SNR in dB
# SNR = 10 * np.log10(signal_power / noise_power)
# print("Signal-to-Noise Ratio (SNR):", SNR, "dB")
