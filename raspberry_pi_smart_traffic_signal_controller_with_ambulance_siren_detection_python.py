import pyaudio
import numpy as np
import RPi.GPIO as GPIO
import time

# GPIO Setup
GPIO.setmode(GPIO.BCM)
# Define GPIO pins for traffic lights (example pins)
RED = 17
YELLOW = 27
GREEN = 22
GPIO.setup(RED, GPIO.OUT)
GPIO.setup(YELLOW, GPIO.OUT)
GPIO.setup(GREEN, GPIO.OUT)

# Audio Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Siren frequency range in Hz
SIREN_LOW = 650
SIREN_HIGH = 1700

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

def detect_siren(data):
    # Convert audio data to numpy array
    audio_data = np.frombuffer(data, dtype=np.int16)
    fft_result = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(fft_result), 1.0/RATE)

    # Get magnitude and corresponding frequencies
    magnitude = np.abs(fft_result)
    peak_freq = abs(freqs[np.argmax(magnitude)])

    if SIREN_LOW <= peak_freq <= SIREN_HIGH:
        return True
    return False

def set_signal(red, yellow, green):
    GPIO.output(RED, red)
    GPIO.output(YELLOW, yellow)
    GPIO.output(GREEN, green)

def give_way_to_ambulance():
    print("Ambulance detected! Turning GREEN for 10 seconds.")
    set_signal(False, False, True)
    time.sleep(10)
    set_signal(True, False, False)

try:
    set_signal(True, False, False)  # Default RED
    print("Listening for siren...")
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        if detect_siren(data):
            give_way_to_ambulance()

except KeyboardInterrupt:
    print("Exiting...")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    GPIO.cleanup()
