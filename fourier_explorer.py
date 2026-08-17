"""
Fourier Series Explorer
========================
An interactive Streamlit app for teaching Fourier decomposition/composition.

Run with:
    pip install streamlit numpy scipy matplotlib
    streamlit run fourier_explorer.py

Place this file in the same folder as demo_audio.wav (the bundled predefined
audio clip) before running.

What it does
------------
1. Pick a classic periodic waveform (square, sawtooth, triangle) and use a
   slider to control how many harmonics are summed. You immediately see:
     - the time-domain approximation converging to the true wave (Gibbs effect)
     - the exact mathematical formula, rendered in LaTeX, with the current
       partial sum spelled out term by term
     - a frequency spectrum (stem plot) showing which harmonics are "on"
2. Pick "Audio" to decompose a predefined WAV clip (or your own uploaded
   WAV) into frequency components. A slider controls how many of the
   *lowest* frequencies are kept, and you can listen to the reconstruction
   as it sharpens with more frequencies added.
"""

import io
import os
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from scipy.io import wavfile

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="Fourier-Reihen-Explorer", layout="wide")
st.title("Fourier-Reihen-Explorer")
st.caption(
    "Erfahre, wie jedes periodische Signal als Summe einfacher Sinus- und "
    "Kosinuswellen aufgebaut ist — passe die Anzahl der Frequenzen an und "
    "beobachte, wie die Approximation konvergiert."
)

WAVE_OPTIONS = ["Rechteckswelle", "Sägezahnschwingung", "Dreieckswelle", "Audio"]

DEFAULT_WAV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_audio.wav")

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("Steuerung")
wave_type = st.sidebar.selectbox("Wellenform", WAVE_OPTIONS)

# Map displayed (German) options to the internal English keys used
# by the coefficient/true_wave functions. Keep display labels in German.
DISPLAY_TO_INTERNAL = {
    "Rechteckswelle": "Square wave",
    "Sägezahnschwingung": "Sawtooth wave",
    "Dreieckswelle": "Triangle wave",
    "Audio": "Audio",
}
internal_wave_type = DISPLAY_TO_INTERNAL.get(wave_type, wave_type)

is_audio_mode = wave_type == "Audio"

if not is_audio_mode:
    n_terms = st.sidebar.slider("Anzahl der Harmonischen (N)", 1, 100, 5)
    show_components = st.sidebar.checkbox("Einzelne Harmonische anzeigen", False)
    max_spectrum_n = st.sidebar.slider("Im Spektrum angezeigte Harmonische", 10, 100, 30)
else:
    uploaded_file = st.sidebar.file_uploader(
        "Vordefiniertes Audio ersetzen (optional WAV-Upload)", type=["wav"]
    )
    TARGET_SR = 44100  # Standard-Audiosamplingrate

# --------------------------------------------------------------------------
# Fourier series math for the classic waveforms
# --------------------------------------------------------------------------
def fourier_coeffs(wave_type: str, n_max: int):
    """Return cosine (a) and sine (b) coefficient arrays, index 0..n_max.

    All three waveforms are odd functions, so a_n = 0 for every n; only the
    sine (b_n) coefficients are non-zero. Index 0 is unused (kept for
    readability so array index == harmonic number).
    """
    a = np.zeros(n_max + 1)
    b = np.zeros(n_max + 1)
    for n in range(1, n_max + 1):
        if wave_type == "Square wave":
            if n % 2 == 1:
                b[n] = 4 / (np.pi * n)
        elif wave_type == "Sawtooth wave":
            b[n] = (2 / np.pi) * ((-1) ** (n + 1)) / n
        elif wave_type == "Triangle wave":
            if n % 2 == 1:
                k = (n - 1) // 2
                b[n] = (8 / np.pi**2) * ((-1) ** k) / (n**2)
    return a, b


def reconstruct(x, a, b, n_max):
    """Sum harmonics 1..n_max at points x. Returns (total, list_of_components)."""
    total = np.zeros_like(x)
    components = []
    for n in range(1, n_max + 1):
        comp = a[n] * np.cos(n * x) + b[n] * np.sin(n * x)
        total += comp
        components.append(comp)
    return total, components


def true_wave(x, wave_type):
    if wave_type == "Square wave":
        return np.sign(np.sin(x))
    if wave_type == "Sawtooth wave":
        return x / np.pi
    if wave_type == "Triangle wave":
        return (2 / np.pi) * np.arcsin(np.sin(x))
    return None


GENERAL_FORMULA = {
    "Square wave": r"f(x) \approx \frac{4}{\pi}\sum_{n=1,3,5,\dots}^{N}\frac{1}{n}\sin(nx)",
    "Sawtooth wave": r"f(x) \approx \frac{2}{\pi}\sum_{n=1}^{N}\frac{(-1)^{n+1}}{n}\sin(nx)",
    "Triangle wave": r"f(x) \approx \frac{8}{\pi^2}\sum_{n=1,3,5,\dots}^{N}\frac{(-1)^{(n-1)/2}}{n^2}\sin(nx)",
}


def partial_sum_latex(a, b, n_max, max_terms_shown=4):
    """Build a LaTeX string spelling out the first few non-zero terms."""
    terms = []
    for n in range(1, n_max + 1):
        if abs(b[n]) < 1e-12 and abs(a[n]) < 1e-12:
            continue
        if len(terms) >= max_terms_shown:
            terms.append(r"\dots")
            break
        coeff = b[n] if abs(b[n]) > abs(a[n]) else a[n]
        func = r"\sin" if abs(b[n]) > abs(a[n]) else r"\cos"
        terms.append(rf"{coeff:.3f}\,{func}({n}x)")
    return "f(x) \\approx " + " + ".join(terms).replace("+ -", "- ")


# --------------------------------------------------------------------------
# MODE 1: classic periodic waveforms
# --------------------------------------------------------------------------
if not is_audio_mode:
    x = np.linspace(-np.pi, np.pi, 2000)
    a, b = fourier_coeffs(internal_wave_type, max(n_terms, max_spectrum_n))
    approx, components = reconstruct(x, a, b, n_terms)
    true_y = true_wave(x, internal_wave_type)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Zeitbereich: Approximation vs. reale Wellenform")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, true_y, color="black", linewidth=2, label="Reale Wellenform")
        ax.plot(x, approx, color="crimson", linewidth=2, label=f"Approximation (N={n_terms})")
        if show_components:
            for n, comp in enumerate(components, start=1):
                if abs(a[n]) > 1e-12 or abs(b[n]) > 1e-12:
                    ax.plot(x, comp, linewidth=0.8, alpha=0.5, label=f"n={n}")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("x")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{wave_type}: Gibbs-Phänomen mit wachsendem N")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Die Mathematik")
        st.latex(GENERAL_FORMULA[internal_wave_type])
        st.markdown(f"**Aktuelle Teilsumme (N = {n_terms}):**")
        st.latex(partial_sum_latex(a, b, n_terms))
        rms_error = np.sqrt(np.mean((true_y - approx) ** 2))
        st.metric("RMS-Fehler zur echten Welle", f"{rms_error:.4f}")
    st.caption(
    "Tipp: Bewege den Schieberegler langsam und beobachte, wie das Gibbs-'Ringing' "
    "nahe scharfer Kanten schmaler wird, aber selbst bei großen N nie ganz verschwindet."
)
    st.subheader("Frequenzspektrum")
    st.caption(
        "Jeder Balken steht für die Amplitude einer Harmonischen. Harmonische bis N (rot) "
        "sind in der Approximation oben enthalten; die restlichen (grau) sind noch nicht hinzugefügt."
    )
    amps = np.sqrt(a**2 + b**2)
    ns = np.arange(1, max_spectrum_n + 1)
    colors = ["crimson" if n <= n_terms else "lightgray" for n in ns]
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.bar(ns, amps[1 : max_spectrum_n + 1], color=colors, width=0.6)
    ax2.set_xlabel("Harmonische n (Frequenz = n × Grundfrequenz)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Amplitudenspektrum")
    ax2.grid(True, alpha=0.3, axis="y")
    st.pyplot(fig2)
    plt.close(fig2)

# --------------------------------------------------------------------------
# MODE 2: audio decomposition and reconstruction
# --------------------------------------------------------------------------
else:
    st.subheader("Audio in Frequenzen zerlegen")
    st.caption(
        "Standardmäßig wird ein vordefiniertes Audioclip geladen — lade in der Seitenleiste "
        "dein eigenes WAV hoch, um einen anderen Klang zu verwenden. Das Signal wird "
        "per FFT in Frequenzkomponenten zerlegt; der Schieberegler bestimmt, wie viele "
        "der *niedrigsten* Frequenzen bei der Rekonstruktion erhalten bleiben, sodass "
        "höhere Frequenzen mit steigendem Wert hinzugefügt werden."
    )

    @st.cache_data(show_spinner=False)
    def load_wav(source, target_sr):
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        sr, data = wavfile.read(source)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float64)
        peak = np.max(np.abs(data))
        if peak > 0:
            data = data / peak
        if sr != target_sr:
            n_target = max(1, int(round(data.size * target_sr / sr)))
            data = scipy.signal.resample(data, n_target)
            sr = target_sr
        return sr, data

    @st.cache_data(show_spinner=False)
    def compute_spectrum(signal, sample_rate):
        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(signal.size, d=1 / sample_rate)
        magnitudes = np.abs(spectrum)
        return spectrum, freqs, magnitudes

    if uploaded_file is not None:
        source_bytes = uploaded_file.read()
        source, source_label = source_bytes, uploaded_file.name
    else:
        source, source_label = DEFAULT_WAV_PATH, "predefined demo clip"

    sample_rate, signal = load_wav(source, TARGET_SR)
    st.caption(f"Verwendet **{source_label}** — {len(signal)/sample_rate:.2f}s bei {sample_rate} Hz")

    # --- FFT decomposition ---
    spectrum, freqs, magnitudes = compute_spectrum(signal, sample_rate)
    total_bins = len(freqs)

    # Default range picked in Hz (not bin count), since how many bins that
    # corresponds to depends on clip duration and sample rate.
    if "low_freq" not in st.session_state:
        st.session_state.low_freq = 0.0
    if "high_freq" not in st.session_state:
        st.session_state.high_freq = min(1000.0, float(freqs[-1]))

    # --- interactive spectrum plot: drag across it to set the low/high range ---
    st.subheader("Frequenzbereich mit zwei Cursorn auswählen")
    st.caption(
        "Nutze den Schieberegler unten, um den x-Achsenbereich zu vergrößern, und ziehe "
        "danach horizontal im Diagramm, um den zu behaltenden Frequenzbereich auszuwählen. "
        "Plotly-Zoom und -Pan sind deaktiviert, damit die Auswahl leicht angepasst werden kann."
    )

    max_display_freq = min(20000.0, sample_rate / 2)
    zoom_min, zoom_max = st.slider(
        "Zoombereich des Diagramms (Hz)",
        0.0,
        float(max_display_freq),
        value=(0.0, float(max_display_freq)),
        step=50.0,
    )

    low_cutoff = float(st.session_state.low_freq)
    high_cutoff = float(st.session_state.high_freq)
    low_cutoff = max(0.0, min(low_cutoff, freqs[-1]))
    high_cutoff = max(low_cutoff, min(high_cutoff, freqs[-1]))
    st.session_state.low_freq = low_cutoff
    st.session_state.high_freq = high_cutoff

    kept_mask = (freqs >= low_cutoff) & (freqs <= high_cutoff)

    spec_fig = go.Figure()
    spec_fig.add_trace(
        go.Scatter(
            x=freqs[kept_mask], y=magnitudes[kept_mask], mode="lines",
            line=dict(color="crimson", width=1.5), name="Kept", showlegend=False,
        )
    )
    spec_fig.add_trace(
        go.Scatter(
            x=freqs[~kept_mask], y=magnitudes[~kept_mask], mode="lines",
            line=dict(color="lightgray", width=1.5), name="Dropped", showlegend=False,
        )
    )
    spec_fig.add_vline(x=low_cutoff, line_dash="dash", line_color="crimson")
    spec_fig.add_vline(x=high_cutoff, line_dash="dash", line_color="crimson")
    spec_fig.update_layout(
        xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",
        height=350, margin=dict(l=40, r=20, t=20, b=40),
        dragmode="select",
        xaxis=dict(range=[zoom_min, zoom_max], fixedrange=True),
        yaxis=dict(fixedrange=True),
        modebar_remove=["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d"],
    )

    drag_event = st.plotly_chart(
        spec_fig, use_container_width=True, key="spectrum_chart",
        on_select="rerun", selection_mode="box",
    )

    selection = getattr(drag_event, "selection", None)
    boxes = selection.get("box", []) if selection else []
    if boxes:
        x_range = boxes[0]["x"]
        new_low = float(min(x_range))
        new_high = float(max(x_range))
        new_low = max(0.0, min(new_low, freqs[-1]))
        new_high = max(new_low, min(new_high, freqs[-1]))
        if (
            abs(new_low - st.session_state.low_freq) > 1e-6
            or abs(new_high - st.session_state.high_freq) > 1e-6
        ):
            st.session_state.low_freq = new_low
            st.session_state.high_freq = new_high
            st.rerun()

    kept_bins = np.where(kept_mask)[0]
    n_components = kept_bins.size
    if n_components == 0:
        st.warning("Derzeit sind keine Frequenzbereiche ausgewählt; ziehe einen größeren Bereich, damit hörbare Komponenten erhalten bleiben.")
    else:
        st.caption(
            f"Es werden {n_components} von {total_bins} Frequenzkomponenten "
            f"zwischen {low_cutoff:.0f} Hz und {high_cutoff:.0f} Hz beibehalten."
        )

    filtered_spectrum = np.zeros_like(spectrum)
    filtered_spectrum[kept_mask] = spectrum[kept_mask]
    reconstructed = np.fft.irfft(filtered_spectrum, n=signal.size)
    peak = np.max(np.abs(reconstructed))
    if peak > 0:
        reconstructed = reconstructed / peak

    # --- audio player: reconstructed only ---
    def to_wav_bytes(sig, sr):
        pcm = np.int16(np.clip(sig, -1, 1) * 32767)
        buf = io.BytesIO()
        wavfile.write(buf, sr, pcm)
        return buf.getvalue()

    if n_components > 0:
        top_freq = freqs[kept_bins[-1]]
    else:
        top_freq = 0.0
    st.markdown(
        f"**Rekonstruiertes Audio — {n_components} von {total_bins} Frequenzkomponenten, "
        f"von {low_cutoff:.0f} Hz bis {top_freq:.0f} Hz**"
    )
    st.audio(to_wav_bytes(reconstructed, sample_rate), format="audio/wav")

    st.subheader("Zeitbereich (vergrößert)")
    zoom_samples = min(int(sample_rate), signal.size)  # erste 20 ms
    t_axis = np.arange(zoom_samples) / sample_rate
    fig3, ax3 = plt.subplots(figsize=(10, 3.5))
    ax3.plot(t_axis, signal[:zoom_samples], color="black", linewidth=1.5, label="Original")
    ax3.plot(t_axis, reconstructed[:zoom_samples], color="crimson", linewidth=1.5, label="Rekonstruiert")
    ax3.set_xlabel("Zeit (s)")
    ax3.set_ylabel("Amplitude")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
    plt.close(fig3)

    st.latex(
        r"x(t) \approx \text{IFFT}\big(X_k \cdot \mathbb{1}[k \le K]\big),"
        r"\quad X_k = \text{FFT}(x(t)),\quad K = \text{cutoff bin}"
    )

st.sidebar.markdown("---")
