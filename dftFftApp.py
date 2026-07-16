import math
import cmath
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter

st.set_page_config(page_title="DFT/FFT und Resampling", layout="wide")
st.title("DFT, FFT und Resampling")

st.markdown("""
Diese App basiert auf dem Notebook zur DFT/FFT und zeigt, wie sich Downsampling und Upsampling
auf das Signal und sein Spektrum auswirken.
""")


@st.cache_data
def compute_dft(x):
    n = len(x)
    X = np.zeros(n, dtype=complex)
    for k in range(n):
        for m in range(n):
            angle = -2j * math.pi * k * m / n
            X[k] += x[m] * cmath.exp(angle)
    return X


@st.cache_data
def compute_idft(X):
    n = len(X)
    x = np.zeros(n, dtype=complex)
    for m in range(n):
        for k in range(n):
            angle = 2j * math.pi * k * m / n
            x[m] += X[k] * cmath.exp(angle)
        x[m] /= n
    return x


@st.cache_data
def make_signal(fs, duration, freqs, amplitudes):
    t = np.arange(0, duration, 1 / fs)
    x = np.zeros_like(t, dtype=float)
    for freq, amp in zip(freqs, amplitudes):
        x += amp * np.sin(2 * np.pi * freq * t)
    return t, x


@st.cache_data
def spectrum(x, fs):
    X = fft(x)
    freqs = fftfreq(len(x), d=1 / fs)
    mask = freqs >= 0
    return freqs[mask], np.abs(X[mask]) / len(x)


@st.cache_data
def lowpass_filter(data, cutoff, fs):
    b, a = butter(4, cutoff / (0.5 * fs), btype="low")
    return lfilter(b, a, data)


@st.cache_data
def alias_frequency(freq, nyquist):
    if freq <= nyquist:
        return freq
    wrapped = np.mod(freq, 2 * nyquist)
    if wrapped > nyquist:
        wrapped = 2 * nyquist - wrapped
    return wrapped


st.sidebar.header("Parameter")
fs = 100
frequency_1 = 1
frequency_2 = 4
frequency_3 = 7
amplitude_1 = 3.0
amplitude_2 = 1.0
amplitude_3 = 0.5
duration = 5.0
freqs = [frequency_1, frequency_2, frequency_3]
amplitudes = [amplitude_1, amplitude_2, amplitude_3]

operation = st.sidebar.selectbox(
    "Operation",
    ["Downsampling", "Upsampling (Interpolation)"],
)
factor = st.sidebar.slider("Faktor", 2, 15, 5, 1)


t, x = make_signal(fs, duration, freqs, amplitudes)

st.subheader("1. Originalsignal")
left_top, right_top = st.columns(2)

with left_top:
    fig_time, ax_time = plt.subplots(figsize=(6, 2.8))
    ax_time.plot(t, x, color="tab:blue")
    ax_time.set_xlabel("Zeit (s)")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_title("Signal x(t)")
    ax_time.grid(True)
    fig_time.tight_layout()
    st.pyplot(fig_time)

with right_top:
    st.markdown("**Originalsignalwerte**")
    st.code(
        f"x(t) = {amplitude_1:.1f}·sin(2π·{frequency_1}t) + {amplitude_2:.1f}·sin(2π·{frequency_2}t) + {amplitude_3:.1f}·sin(2π·{frequency_3}t)",
        language="text",
    )
    st.write("**Frequenzkomponenten:**")
    st.write(f"- {frequency_1} Hz")
    st.write(f"- {frequency_2} Hz")
    st.write(f"- {frequency_3} Hz")
    st.write (f"**Abtastfrequenz:** {fs} Hz")

    
freqs_fft, mags_fft = spectrum(x, fs)

st.subheader("2. Spektrum")
left_top_spec, right_top_spec = st.columns(2)

with left_top_spec:
    fig_spec, ax_spec = plt.subplots(figsize=(6, 2.8))
    ax_spec.stem(freqs_fft, mags_fft, linefmt='-', markerfmt=' ', basefmt='-')
    ax_spec.set_xlabel("Frequenz (Hz)")
    ax_spec.set_ylabel("Amplitude")
    ax_spec.set_title("DFT/FFT des Signals")
    ax_spec.set_xlim(0, fs / 2 + 5)
    ax_spec.grid(True)
    fig_spec.tight_layout()
    st.pyplot(fig_spec)

with right_top_spec:
    st.markdown("**Interpretation**")
    st.write("Die Peaks im Spektrum zeigen die vorhandenen Frequenzanteile des Signals.")
    st.write(f"**Maximale sichtbare Frequenz:** {fs / 2:.1f} Hz")

    

if operation == "Downsampling":
    new_fs = fs / factor
    t_down = t[::factor]
    x_down = x[::factor]
    freqs_down, mags_down = spectrum(x_down, new_fs)

    st.subheader("3. Downsampling")
    top_left, top_right = st.columns(2)
    with top_left:
        fig_down_time, ax_down_time = plt.subplots(figsize=(6, 2.8))
        ax_down_time.plot(t, x, color="gray", alpha=0.4, label="Original")
        ax_down_time.plot(t_down, x_down, "o-", color="tab:red", label="Downsampled")
        ax_down_time.set_xlabel("Zeit (s)")
        ax_down_time.set_ylabel("Amplitude")
        ax_down_time.set_title("Signal nach Downsampling")
        ax_down_time.grid(True)
        ax_down_time.legend()
        fig_down_time.tight_layout()
        st.pyplot(fig_down_time)

    with top_right:
        fig_down_spec, ax_down_spec = plt.subplots(figsize=(6, 2.8))
        ax_down_spec.stem(freqs_down, mags_down, linefmt='-', markerfmt=' ', basefmt='-')
        ax_down_spec.set_xlabel("Frequenz (Hz)")
        ax_down_spec.set_ylabel("Amplitude")
        ax_down_spec.set_title("Spektrum nach Downsampling")
        ax_down_spec.set_xlim(0, min(new_fs / 2 , 10))
        ax_down_spec.grid(True)
        fig_down_spec.tight_layout()
        st.pyplot(fig_down_spec)

    bottom_left, bottom_right = st.columns(2)

    nyquist_new = new_fs / 2
    aggressive_freqs = [f for f in freqs if f > nyquist_new]
    if aggressive_freqs:
        expected_aliases = [f"{f:.1f} Hz -> {alias_frequency(f, nyquist_new):.1f} Hz" for f in aggressive_freqs]
        expected_text = "Die Komponenten oberhalb der neuen Nyquist-Frequenz werden mit Aliasing zurück in den sichtbaren Bereich gefaltet: " + ", ".join(expected_aliases)
    else:
        expected_text = "Keine Aliasing-Effekte erwartet, weil alle Komponenten unterhalb der neuen Nyquist-Frequenz liegen."

    with bottom_left:
        X_down = compute_dft(x_down)
        x_recon = compute_idft(X_down).real
        fig_recon, ax_recon = plt.subplots(figsize=(6, 2.8))
        ax_recon.plot(t_down, x_recon, color="tab:green", label="Rekonstruiert")
        ax_recon.plot(t, x, color="gray", linestyle="--", label="Original")
        ax_recon.set_xlabel("Zeit (s)")
        ax_recon.set_ylabel("Amplitude")
        ax_recon.set_title("Rekonstruktion aus dem downsampled Spektrum")
        ax_recon.grid(True)
        ax_recon.legend()
        fig_recon.tight_layout()
        st.pyplot(fig_recon)

    with bottom_right:
        st.info(f"Abtastfrequenz nach Downsampling: {new_fs:.1f} Hz\n\n{expected_text}")

elif operation == "Upsampling (Interpolation)":
    new_fs = fs * factor
    t_up = np.arange(0, duration, 1 / new_fs)
    x_up = np.zeros(len(t_up))
    x_up[::factor] = x
    freqs_up, mags_up = spectrum(x_up, new_fs)

    st.subheader("3. Upsampling")
    top_left, top_right = st.columns(2)
    with top_left:
        fig_up_time, ax_up_time = plt.subplots(figsize=(6, 2.8))
        ax_up_time.plot(t, x, color="gray", alpha=0.4, label="Original")
        ax_up_time.plot(t_up, x_up, "-", color="tab:green", label="Upsampled")
        ax_up_time.set_xlabel("Zeit (s)")
        ax_up_time.set_ylabel("Amplitude")
        ax_up_time.set_title("Signal nach Upsampling")
        ax_up_time.grid(True)
        ax_up_time.legend()
        fig_up_time.tight_layout()
        st.pyplot(fig_up_time)

    with top_right:
        fig_up_spec, ax_up_spec = plt.subplots(figsize=(6, 2.8))
        ax_up_spec.stem(freqs_up, mags_up, linefmt='-', markerfmt=' ', basefmt='-')
        ax_up_spec.set_xlabel("Frequenz (Hz)")
        ax_up_spec.set_ylabel("Amplitude")
        ax_up_spec.set_title("Spektrum nach Upsampling")
        ax_up_spec.set_xlim(0, max(new_fs / 2 + 5, 20))
        ax_up_spec.grid(True)
        fig_up_spec.tight_layout()
        st.pyplot(fig_up_spec)

    bottom_left, bottom_right = st.columns(2)

    cutoff = max(freqs) + 5
    x_filtered = lowpass_filter(x_up, cutoff, new_fs)
    x_filtered_gain = x_filtered * 10

    with bottom_left:
        fig_filter, ax_filter = plt.subplots(figsize=(6, 2.8))
        ax_filter.plot(t_up, x_up, color="gray", alpha=0.4, label="Upsampled")
        ax_filter.plot(t_up, x_filtered_gain, color="tab:purple", label="Tiefpass-gefiltert ×10")
        ax_filter.set_xlabel("Zeit (s)")
        ax_filter.set_ylabel("Amplitude")
        ax_filter.set_title("Gefiltertes Signal nach Upsampling")
        ax_filter.grid(True)
        ax_filter.legend()
        fig_filter.tight_layout()
        st.pyplot(fig_filter)

    with bottom_right:
        st.info(
            f"Abtastfrequenz nach Upsampling: {new_fs:.1f} Hz. "
            "Das Upsampling erzeugt zusätzliche Spektralabbildungen. Ein Tiefpass ist nötig, um das ursprüngliche Spektrum zu erhalten. "
            "Die Verstärkung um den Faktor 10 ist nötig, weil der Tiefpass die Amplitude reduziert, sodass das gefilterte Signal sonst kleiner als das Original erscheint."
        )

