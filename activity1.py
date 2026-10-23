import speech_recognition as sr
import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

def record_audio(filename="my_voice.wav", duration=5, sample_rate=44100):
    """1. Audio Capture & Digitization"""
    print(f"\n[LISTENING] Please speak now... Recording for {duration} seconds")
    
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    
    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk)
    
    frames = []
    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    print("[DONE] Recording saved")
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

def transcribe_audio(filename):
    """2. Speech Recognition with Google Speech-to-Text"""
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data)
            print(f"\n[TRANSCRIPTION]: {text}")
            return text
        except sr.UnknownValueError:
            print("Google could not understand audio")
            return "Could not understand"
        except sr.RequestError as e:
            print(f"Google API Error: {e}")
            return f"Error: {e}"

def analyze_voice_properties(filename):
    """3. Measuring and Comparing Voice Properties"""
    with wave.open(filename, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        duration = n_frames / float(sample_rate)
        audio_bytes = wf.readframes(n_frames)
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    max_amplitude = np.max(np.abs(audio_data))
    rms_energy = np.sqrt(np.mean(audio_data**2))
    zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
    volume_db = 20 * np.log10(rms_energy) if rms_energy > 0 else 0

    print("\n--- Voice Properties ---")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Duration: {duration:.2f} sec")
    print(f"Total Samples: {n_frames}")
    print(f"Max Amplitude: {max_amplitude}")
    print(f"RMS Energy (Loudness): {rms_energy:.2f}")
    print(f"Volume: {volume_db:.2f} dB")
    print(f"Zero Crossings: {zero_crossings} (indicates pitch/voiced)")

    return audio_data, sample_rate, duration, rms_energy, volume_db

def plot_waveform(audio_data, sample_rate, transcript):
    """4. Plot Waveform"""
    time_axis = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, audio_data, color='#4A90E2', linewidth=0.5)
    plt.title(f'Say & See - Waveform | Transcribed: "{transcript}"', fontsize=12)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 1, 2)
    plt.specgram(audio_data, Fs=sample_rate, cmap='viridis')
    plt.title("Voice Spectrogram - Frequency Properties")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    
    plt.tight_layout()
    plt.savefig("voice_analysis.png")
    plt.show()
    print("\n[PLOT] Saved as voice_analysis.png")

def compare_two_voices(file1, file2):
    """Extra: Comparing Voice Properties (for project requirement)"""
    data1, sr1, dur1, rms1, db1 = analyze_voice_properties(file1)
    data2, sr2, dur2, rms2, db2 = analyze_voice_properties(file2)
    
    print("\n--- COMPARISON ---")
    print(f"File 1: {dur1:.2f}s, Loudness {db1:.2f} dB")
    print(f"File 2: {dur2:.2f}s, Loudness {db2:.2f} dB")
    if rms1 > rms2:
        print("Voice 1 is LOUDER and has more energy")
    else:
        print("Voice 2 is LOUDER and has more energy")
if __name__ == "__main__":
    FILENAME = "recording.wav"
    record_audio(FILENAME, duration=5)
    text = transcribe_audio(FILENAME)
    audio_data, sr_rate, dur, rms, db = analyze_voice_properties(FILENAME)
    plot_waveform(audio_data, sr_rate, text)