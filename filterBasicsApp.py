import numpy as np
import streamlit as st
from scipy.signal import butter, filtfilt, freqz
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

st.set_page_config(page_title="Filterung", layout="wide")

st.title("Filterung von Signalen im Zeit- und Frequenzbereich")

st.markdown("""
Wähle drei Frequenzkomponenten für ein Signal und stelle danach verschiedene Filter ein, um zu sehen,
wie Tiefpass, Hochpass, Bandpass oder Bandsperre das Signal im Zeit- und Frequenzbereich beeinflussen.
""")

st.sidebar.header("Parameter")

fs = st.sidebar.slider("Abtastfrequenz fs (Hz)", 200, 1000, 700, 50)
duration = st.sidebar.slider("Signal-Dauer (s)", 0.2, 2.0, 0.5, 0.1)

freq1 = st.sidebar.slider("Komponente 1 (Hz)", 1, 200, 10, 1)
freq2 = st.sidebar.slider("Komponente 2 (Hz)", 1, 400, 150, 1)
freq3 = st.sidebar.slider("Komponente 3 (Hz)", 1, 400, 300, 1)

amp1 = st.sidebar.slider("Amplitude 1", 0.0, 1.0, 0.3, 0.01)
amp2 = st.sidebar.slider("Amplitude 2", 0.0, 1.0, 0.2, 0.01)
amp3 = st.sidebar.slider("Amplitude 3", 0.0, 1.0, 0.25, 0.01)

filter_type = st.sidebar.selectbox(
    "Filtertyp",
    ["Tiefpass", "Hochpass", "Bandpass", "Bandsperre"],
    index=0,
)
order = st.sidebar.slider("Filterordnung", 1, 8, 4, 1)

cutoff1 = st.sidebar.slider("Grenzfrequenz 1 (Hz)", 1, 400, 100, 1)
if filter_type in ["Bandpass", "Bandsperre"]:
    cutoff2 = st.sidebar.slider("Grenzfrequenz 2 (Hz)", 1, 400, 250, 1)
else:
    cutoff2 = None

np.random.seed(42)

@st.cache_data
def build_signal(fs_value, duration_value, freq1_value, freq2_value, freq3_value, amp1_value, amp2_value, amp3_value):
    t_values = np.linspace(0, duration_value, int(fs_value * duration_value), endpoint=False)
    signal_values = (
        amp1_value * np.sin(2 * np.pi * freq1_value * t_values)
        + amp2_value * np.sin(2 * np.pi * freq2_value * t_values)
        + amp3_value * np.sin(2 * np.pi * freq3_value * t_values)
    )
    return t_values, signal_values

@st.cache_data
def design_filter(filter_name, cutoff_low, cutoff_high, fs_value, order_value):
    nyq = 0.5 * fs_value
    if filter_name == "Tiefpass":
        b, a = butter(order_value, cutoff_low / nyq, btype="lowpass")
    elif filter_name == "Hochpass":
        b, a = butter(order_value, cutoff_low / nyq, btype="highpass")
    elif filter_name == "Bandpass":
        b, a = butter(order_value, [cutoff_low / nyq, cutoff_high / nyq], btype="bandpass")
    else:
        b, a = butter(order_value, [cutoff_low / nyq, cutoff_high / nyq], btype="bandstop")
    return b, a


def compute_fft(x_values):
    fft_values = fft(x_values)
    frequencies = fftfreq(len(x_values), d=1 / fs)
    positive_mask = frequencies >= 0
    positive_frequencies = frequencies[positive_mask]
    positive_magnitude = np.abs(fft_values[positive_mask]) * 2 / len(x_values)
    return positive_frequencies, positive_magnitude


t, signal = build_signal(fs, duration, freq1, freq2, freq3, amp1, amp2, amp3)

b, a = design_filter(filter_type, cutoff1, cutoff2, fs, order)
filtered_signal = filtfilt(b, a, signal)
w, h = freqz(b, a, worN=8000)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Zeitbereich")
    fig_time, ax_time = plt.subplots(figsize=(8, 4))
    ax_time.plot(t, signal, label="Originalsignal", color="0.4", alpha=0.8)
    ax_time.plot(t, filtered_signal, label="Gefiltertes Signal", color="tab:orange", linewidth=2)
    ax_time.set_xlabel("Zeit (s)")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_title("Originalsignal und gefiltertes Signal")
    ax_time.grid(True)
    ax_time.legend()
    st.pyplot(fig_time)

with col2:
    st.subheader("Frequenzbereich")
    fig_freq, ax_freq = plt.subplots(figsize=(8, 4))
    orig_freq, orig_mag = compute_fft(signal)
    filt_freq, filt_mag = compute_fft(filtered_signal)
    ax_freq.plot(orig_freq, orig_mag, label="Original", color="0.4", alpha=0.8)
    ax_freq.plot(filt_freq, filt_mag, label="Gefiltert", color="tab:orange", linewidth=2)
    ax_freq.set_xlim(0, fs / 2)
    ax_freq.set_xlabel("Frequenz (Hz)")
    ax_freq.set_ylabel("Amplitude")
    ax_freq.set_title("Frequenzspektrum")
    ax_freq.grid(True)
    ax_freq.legend()
    st.pyplot(fig_freq)

st.subheader("Frequenzgang des Filters")
left_col, right_col = st.columns([1, 1])

with left_col:
    fig_resp, ax_resp = plt.subplots(figsize=(3.8, 1.8))
    frequencies_hz = (w / (2 * np.pi)) * fs
    response_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    ax_resp.semilogx(frequencies_hz, response_db)
    ax_resp.set_xlim(1, fs / 2)
    ax_resp.set_xlabel("Frequenz (Hz)")
    ax_resp.set_ylabel("Dämpfung (dB)")
    ax_resp.set_title("Bode-Diagramm des Filters")
    ax_resp.grid(True, which="both", ls="--", alpha=0.5)
    st.pyplot(fig_resp)

with right_col:
    st.markdown("")

st.caption(
    "Hinweis: Ein Tiefpass lässt tiefe Frequenzen durch, ein Hochpass hohe. "
    "Ein Bandpass lässt nur einen Frequenzbereich durch, während eine Bandsperre genau diesen Bereich unterdrückt."
)
