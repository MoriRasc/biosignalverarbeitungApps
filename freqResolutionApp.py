import numpy as np
import streamlit as st
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

st.set_page_config(page_title="Frequenzauflösung", layout="wide")

st.title("Frequenzauflösung der FFT")

st.markdown("""
Diese App zeigt, wie die Frequenzauflösung eines Spektrums von der Signal-Länge abhängt und nicht von Abtastfrequenz oder FFT-Länge.
Die beiden engen Frequenzkomponenten bei 248 Hz und 250 Hz sind besonders gut geeignet, um den Effekt sichtbar zu machen.
""")

st.sidebar.header("Parameter")

fs = st.sidebar.slider("Abtastfrequenz fs (Hz)", 600.0, 4000.0, 2000.0, 100.0)
duration = st.sidebar.slider("Dauer des aufgenommenen Signals (s)", 0.1, 1.0, 0.1, 0.05)
zero_padding_factor = st.sidebar.slider("Zero-Padding-Faktor", 1, 8, 1, 1)

f0 = 200.0
f1 = 248.0
f2 = 250.0

n = int(duration * fs)
t = np.linspace(0, duration, n, endpoint=False)

signal = np.sin(2 * np.pi * f0 * t) + np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)


def frequency_resolution(fs_value, n_points):
    return fs_value / n_points


@st.cache_data
def compute_fft(signal_values, fs_value, target_length):
    padded_signal = np.zeros(target_length)
    padded_signal[: len(signal_values)] = signal_values
    fft_values = fft(padded_signal)
    frequencies = fftfreq(target_length, d=1 / fs_value)
    positive_mask = frequencies >= 0
    positive_frequencies = frequencies[positive_mask]
    positive_magnitude = np.abs(fft_values[positive_mask]) * 2 / target_length
    return positive_frequencies, positive_magnitude


resolution = frequency_resolution(fs, n)
fft_len = max(n * zero_padding_factor, n)
frequencies, magnitudes = compute_fft(signal, fs, fft_len)
frequencies_base, magnitudes_base = compute_fft(signal, fs, n)

st.subheader("Signal und Spektrum")
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.markdown("**Signal**")
    st.code(f"x(t) = sin(2π·{f0:.0f}t) + sin(2π·{f1:.0f}t) + sin(2π·{f2:.0f}t)", language="text")
    st.markdown("**Frequenzparameter**")
    st.write(f"- f₀ = {f0:.0f} Hz")
    st.write(f"- f₁ = {f1:.0f} Hz")
    st.write(f"- f₂ = {f2:.0f} Hz")

with row1_col2:
    st.caption("Zeitbereich")
    fig_time, ax_time = plt.subplots(figsize=(5.2, 2.2))
    ax_time.plot(t, signal, color="tab:blue")
    ax_time.set_xlabel("Zeit (s)")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_title("Signal")
    ax_time.grid(True)
    fig_time.tight_layout()
    st.pyplot(fig_time)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.caption("FFT ohne Zero-Padding")
    fig_fft, ax_fft = plt.subplots(figsize=(5.2, 2.2))
    ax_fft.plot(frequencies_base, magnitudes_base, label="Ohne Zero-Padding", color="tab:blue", alpha=0.8)
    ax_fft.set_xlim(190, 270)
    ax_fft.set_ylim(0, 2)
    ax_fft.set_xlabel("Frequenz (Hz)")
    ax_fft.set_ylabel("Amplitude")
    ax_fft.set_title("Originale FFT")
    ax_fft.grid(True)
    fig_fft.tight_layout()
    st.pyplot(fig_fft)

with row2_col2:
    st.caption("FFT mit Zero-Padding")
    fig_fft_padded, ax_fft_padded = plt.subplots(figsize=(5.2, 2.2))
    ax_fft_padded.plot(frequencies, magnitudes, label="Mit Zero-Padding", color="tab:orange")
    ax_fft_padded.set_xlim(190, 270)
    ax_fft_padded.set_ylim(0, 0.5)
    ax_fft_padded.set_xlabel("Frequenz (Hz)")
    ax_fft_padded.set_ylabel("Amplitude")
    ax_fft_padded.set_title("FFT mit Zero-Padding")
    ax_fft_padded.grid(True)
    fig_fft_padded.tight_layout()
    st.pyplot(fig_fft_padded)

st.caption(
    "Die Amplitude erscheint bei Zero-Padding kleiner, weil die FFT-Magnitude durch die größere FFT-Länge geteilt wird. "
    "Das ist eine Skalierungsänderung, nicht eine Änderung der tatsächlichen Frequenzinformation."
)

st.metric("Anzahl Samples", n)
st.metric("Frequenzauflösung Δf", f"{resolution:.3f} Hz")
st.metric("FFT-Länge", fft_len)

st.subheader("3. Interpretation")

st.write(
    f"Die Frequenzauflösung der FFT sind durch die Abstände bestimmt: Δf = fs / N = fs / (fs x Dauer) = 1 / Dauer."
)

st.write(
    "Wenn zwei Frequenzen sehr nahe beieinander liegen, können sie nur dann als getrennte Peaks sichtbar werden, wenn die Auflösung klein genug ist. "
    "Zero-Padding hilft dabei, die Darstellung zu glätten, aber es kann eng benachbarte Frequenzen nicht wirklich unterscheiden, wenn die ursprüngliche Messdauer zu kurz ist."
)
