#!/usr/bin/env python3
"""
Smart Traffic Signal Controller with Ambulance Siren Detection

Assumptions
-----------
- One microphone per approach (N, E, S, W). Plug 4 USB mics, or map to 1–4 ALSA input devices.
- When a siren is detected strongest on one approach, that approach turns GREEN; all others RED.
- Simple state machine with a MIN_GREEN time, then returns to normal cycle if no siren persists.
- Detection is spectral + modulation based, robust to general traffic noise.

Hardware
--------
- Raspberry Pi (any with GPIO).
- 12/24V traffic lights driven through relays or MOSFET drivers (GPIO is logic-only!).

Audio libs
----------
- Uses `sounddevice` (PortAudio) for low-latency capture.
- Uses NumPy/SciPy for signal processing.

Configure below before running:
- EDIT `AUDIO_DEVICES` to match your system (use `python3 -m sounddevice` to list).
- EDIT `LIGHT_PINS` to match your wiring.

Run
---
python3 ambulance_traffic_controller.py

"""
from __future__ import annotations
import os
import time
import math
import queue
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter, resample
import sounddevice as sd
try:
    import RPi.GPIO as GPIO
except Exception:
    # For development/testing on non-Pi
    class _MockGPIO:
        BCM = 'BCM'; OUT = 'OUT'
        def setmode(self, *_): pass
        def setwarnings(self, *_): pass
        def setup(self, *_): pass
        def output(self, *args):
            # print("GPIO:", args)
            pass
        def cleanup(self): pass
    GPIO = _MockGPIO()

# ===================== USER CONFIG ===================== #
# Order: North, East, South, West
AUDIO_DEVICES = {
    'N': 2,  # ALSA/PortAudio device index for North mic
    'E': 3,  # East mic
    'S': 4,  # South mic
    'W': 5,  # West mic
}

SAMPLE_RATE = 16000         # Hz (16 kHz is fine for sirens)
FRAME_SIZE = 2048           # samples per analysis frame (~128 ms)
HOP_SIZE = 1024             # analysis hop (~64 ms)
BUFFER_SECONDS = 3.0        # rolling analysis window length

# Siren characteristics (approx): sweeping tones 600–3000 Hz with 0.5–8 Hz modulation
SIREN_BAND_LOW = 500
SIREN_BAND_HIGH = 3200
MOD_MIN = 0.5               # Hz (sweep rate lower bound)
MOD_MAX = 8.0               # Hz (sweep rate upper bound)

# Decision thresholds (tune on-site)
MIN_CONFIDENCE = 0.65       # siren detection score threshold per approach
MIN_FRAMES_TRIGGER = 6      # consecutive positive frames to trigger (debounce)
COOLDOWN_SECONDS = 5.0      # cooldown after losing siren before releasing priority

# Traffic light timing
YELLOW_TIME = 2.5           # seconds
MIN_GREEN_TIME = 12.0       # minimum hold of ambulance green
MAX_AMBULANCE_HOLD = 90.0   # hard upper bound safety

# GPIO pin mapping per approach and lamp color (BCM numbering)
LIGHT_PINS = {
    'N': {'R': 5,  'Y': 6,  'G': 13},
    'E': {'R': 16, 'Y': 19, 'G': 20},
    'S': {'R': 21, 'Y': 26, 'G': 12},
    'W': {'R': 7,  'Y': 8,  'G': 25},
}

# Normal cycle fallback if no siren (simple fixed cycle)
NORMAL_CYCLE_ORDER = ['N', 'E', 'S', 'W']
NORMAL_GREEN = 12.0
NORMAL_YELLOW = 2.5
# =================== END USER CONFIG =================== #

@dataclass
class DetectionResult:
    confidence: float
    dominant_hz: float
    snr_db: float

class SirenDetector:
    """Detects ambulance sirens via spectral energy + modulation in siren band.

    Algorithm summary (per frame):
      1) Band-pass filter to 500–3200 Hz.
      2) FFT to find dominant frequency and band energy.
      3) Maintain history of dominant frequency; compute its modulation rate via
         autocorrelation. A strong periodic sweep within 0.5–8 Hz boosts confidence.
      4) Combine SNR of the dominant tone with modulation consistency => confidence [0..1].
    """
    def __init__(self, fs: int, frame: int, hop: int):
        self.fs = fs
        self.frame = frame
        self.hop = hop
        self.hist_len = int(BUFFER_SECONDS * fs / hop)
        self.dom_hist = deque(maxlen=self.hist_len)
        self.energy_hist = deque(maxlen=self.hist_len)
        self._bp_b, self._bp_a = butter(4, [SIREN_BAND_LOW/(fs/2), SIREN_BAND_HIGH/(fs/2)], btype='band')
        # Hann window for FFT
        self.win = np.hanning(frame)

    def process(self, x: np.ndarray) -> DetectionResult:
        # x: mono audio frame (FRAME_SIZE)
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        # Band-pass
        y = lfilter(self._bp_b, self._bp_a, x)
        # FFT magnitude
        X = np.fft.rfft(self.win * y)
        mag = np.abs(X)
        freqs = np.fft.rfftfreq(len(y), 1.0/self.fs)
        # Focus on siren band
        band_mask = (freqs >= SIREN_BAND_LOW) & (freqs <= SIREN_BAND_HIGH)
        band_mag = mag[band_mask]
        band_freqs = freqs[band_mask]
        if band_mag.size == 0:
            self.dom_hist.append(0.0)
            self.energy_hist.append(0.0)
            return DetectionResult(0.0, 0.0, -np.inf)
        # Dominant frequency & SNR estimate
        pk_idx = int(np.argmax(band_mag))
        dom_f = float(band_freqs[pk_idx])
        dom_pwr = float(band_mag[pk_idx]**2)
        noise_pwr = float(np.mean(np.delete(band_mag**2, pk_idx))) + 1e-12
        snr = 10.0 * math.log10(dom_pwr / noise_pwr)
        # Update history
        self.dom_hist.append(dom_f)
        self.energy_hist.append(float(np.mean(band_mag)))

        # Need enough history for modulation estimate
        conf_mod = 0.0
        mod_hz = 0.0
        if len(self.dom_hist) > int(1.0 * self.fs / self.hop):
            f = np.asarray(self.dom_hist)
            f = f - np.mean(f)
            if np.std(f) > 1.0:  # has some movement
                # Autocorrelation
                ac = np.correlate(f, f, mode='full')[len(f)-1:]
                # Ignore lag 0, search up to ~2 seconds
                max_lag = int(min(len(f)-1, 2.0 * self.fs / self.hop))
                if max_lag > 3:
                    ac_seg = ac[1:max_lag]
                    lag = int(np.argmax(ac_seg)) + 1
                    mod_hz = (self.fs / self.hop) / lag
                    # Score if modulation lies within expected range
                    if MOD_MIN <= mod_hz <= MOD_MAX:
                        # Normalize autocorr peak
                        conf_mod = float(np.clip(ac_seg[lag-1] / (ac[0] + 1e-9), 0.0, 1.0))

        # Confidence blend: SNR + modulation consistency
        # Map SNR 0..25 dB roughly to 0..1
        conf_snr = np.clip((snr - 0.0) / 25.0, 0.0, 1.0)
        confidence = 0.35*conf_snr + 0.65*conf_mod
        return DetectionResult(float(confidence), dom_f, float(snr))

class AudioWorker(threading.Thread):
    def __init__(self, name: str, device_index: int, result_queue: queue.Queue):
        super().__init__(daemon=True)
        self.name = name
        self.device_index = device_index
        self.result_queue = result_queue
        self.detector = SirenDetector(SAMPLE_RATE, FRAME_SIZE, HOP_SIZE)
        self._stop = threading.Event()
        self._stream = None
        self._buf = np.zeros(0, dtype=np.float32)

    def callback(self, indata, frames, time_info, status):
        if status:
            # print(f"{self.name} stream status: {status}")
            pass
        # Accumulate and analyze in HOP_SIZE strides
        self._buf = np.concatenate([self._buf, indata.copy().astype(np.float32).flatten()])
        while len(self._buf) >= FRAME_SIZE:
            frame = self._buf[:FRAME_SIZE]
            self._buf = self._buf[HOP_SIZE:]
            res = self.detector.process(frame)
            self.result_queue.put((self.name, time.time(), res))

    def run(self):
        with sd.InputStream(device=self.device_index, channels=1, samplerate=SAMPLE_RATE,
                            blocksize=HOP_SIZE, callback=self.callback):
            while not self._stop.is_set():
                time.sleep(0.05)

    def stop(self):
        self._stop.set()

class TrafficLights:
    def __init__(self, pins_map):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        self.pins = pins_map
        for d in self.pins:
            for col in self.pins[d].values():
                GPIO.setup(col, GPIO.OUT)
        self.all_red()

    def set(self, direction: str, color: str):
        # Turn off others in that direction first
        for c in ['R','Y','G']:
            GPIO.output(self.pins[direction][c], 1 if c==color else 0)

    def all_red(self):
        for d in self.pins:
            GPIO.output(self.pins[d]['R'], 1)
            GPIO.output(self.pins[d]['Y'], 0)
            GPIO.output(self.pins[d]['G'], 0)

    def green_only(self, direction: str):
        for d in self.pins:
            if d == direction:
                GPIO.output(self.pins[d]['R'], 0)
                GPIO.output(self.pins[d]['Y'], 0)
                GPIO.output(self.pins[d]['G'], 1)
            else:
                GPIO.output(self.pins[d]['R'], 1)
                GPIO.output(self.pins[d]['Y'], 0)
                GPIO.output(self.pins[d]['G'], 0)

    def yellow_then_red(self, direction: str, yellow_time: float):
        GPIO.output(self.pins[direction]['G'], 0)
        GPIO.output(self.pins[direction]['Y'], 1)
        time.sleep(yellow_time)
        GPIO.output(self.pins[direction]['Y'], 0)
        GPIO.output(self.pins[direction]['R'], 1)

    def cleanup(self):
        self.all_red()
        GPIO.cleanup()

class Controller:
    def __init__(self):
        self.q = queue.Queue()
        self.workers = [AudioWorker(k, v, self.q) for k, v in AUDIO_DEVICES.items()]
        self.lights = TrafficLights(LIGHT_PINS)
        self.debounce_counts = {k: 0 for k in AUDIO_DEVICES}
        self.last_positive = {k: 0.0 for k in AUDIO_DEVICES}
        self.priority_dir: str | None = None
        self.priority_since = 0.0
        self.current_cycle_index = 0

    def start(self):
        for w in self.workers:
            w.start()
        try:
            self.loop()
        finally:
            for w in self.workers:
                w.stop()
            self.lights.cleanup()

    def loop(self):
        self.lights.all_red()
        last_change = 0.0
        # enter normal cycle initially
        current_dir = NORMAL_CYCLE_ORDER[self.current_cycle_index]
        self.lights.green_only(current_dir)
        last_change = time.time()
        phase = 'GREEN'

        while True:
            # Process detection results non-blocking
            try:
                while True:
                    dname, ts, res = self.q.get_nowait()
                    if res.confidence >= MIN_CONFIDENCE:
                        self.debounce_counts[dname] += 1
                        self.last_positive[dname] = ts
                    else:
                        self.debounce_counts[dname] = max(0, self.debounce_counts[dname]-1)
            except queue.Empty:
                pass

            # Determine if any direction has valid siren
            now = time.time()
            candidates = []
            for d in AUDIO_DEVICES:
                # still "active" if within cooldown window
                active = (now - self.last_positive[d]) <= COOLDOWN_SECONDS
                if self.debounce_counts[d] >= MIN_FRAMES_TRIGGER or active:
                    # weight by how recently seen & debounce count
                    score = self.debounce_counts[d] + max(0.0, COOLDOWN_SECONDS - (now - self.last_positive[d]))
                    candidates.append((score, d))
            candidates.sort(reverse=True)

            if candidates:
                best_dir = candidates[0][1]
                if self.priority_dir != best_dir:
                    # Switch to ambulance priority
                    if phase == 'GREEN' and current_dir != best_dir:
                        # safe change: yellow current, then red, then green priority
                        self.lights.yellow_then_red(current_dir, YELLOW_TIME)
                    current_dir = best_dir
                    self.lights.green_only(current_dir)
                    self.priority_dir = best_dir
                    self.priority_since = now
                    phase = 'GREEN'
                    last_change = now
                else:
                    # we are already serving ambulance; hold until limits
                    if (now - self.priority_since) > MAX_AMBULANCE_HOLD:
                        # fail-safe release
                        self.priority_dir = None
                        self._advance_cycle()
                        current_dir = NORMAL_CYCLE_ORDER[self.current_cycle_index]
                        self.lights.green_only(current_dir)
                        phase = 'GREEN'
                        last_change = now
            else:
                # No active siren; run normal cycle
                if self.priority_dir is not None:
                    # release priority after MIN_GREEN_TIME
                    if now - self.priority_since >= MIN_GREEN_TIME:
                        self.priority_dir = None
                        self._advance_cycle()
                        current_dir = NORMAL_CYCLE_ORDER[self.current_cycle_index]
                        self.lights.green_only(current_dir)
                        phase = 'GREEN'
                        last_change = now
                else:
                    # Regular timing
                    if phase == 'GREEN' and (now - last_change) >= NORMAL_GREEN:
                        self.lights.yellow_then_red(current_dir, NORMAL_YELLOW)
                        self._advance_cycle()
                        current_dir = NORMAL_CYCLE_ORDER[self.current_cycle_index]
                        self.lights.green_only(current_dir)
                        phase = 'GREEN'
                        last_change = time.time()

            time.sleep(0.03)

    def _advance_cycle(self):
        self.current_cycle_index = (self.current_cycle_index + 1) % len(NORMAL_CYCLE_ORDER)


def main():
    print("Listing PortAudio devices (tip: run `python3 -m sounddevice` for detailed list)...")
    try:
        devs = sd.query_devices()
        for i, d in enumerate(devs):
            if int(d.get('max_input_channels', 0)) > 0:
                print(f"Input device {i}: {d['name']}")
    except Exception as e:
        print("Could not list devices:", e)
    print("\nConfigured devices:", AUDIO_DEVICES)
    ctrl = Controller()
    try:
        ctrl.start()
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == '__main__':
    main()
