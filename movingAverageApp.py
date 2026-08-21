import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Randprobleme bei der Faltung", layout="wide")


@st.cache_data
def build_example_signal():
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, 400)
    clean = np.sin(2 * np.pi * 6 * t) + 0.35 * np.sin(2 * np.pi * 18 * t)
    noise = 0.65 * rng.normal(0, 1, size=len(t))
    signal = clean + noise
    return t, signal


@st.cache_data
def build_window(window_type, window_size):
    if window_type == "Rechteck":
        kernel = np.ones(window_size) / window_size
    elif window_type == "Dreieck":
        kernel = np.bartlett(window_size)
        kernel = kernel / np.sum(kernel)
    elif window_type == "Hamming":
        kernel = np.hamming(window_size)
        kernel = kernel / np.sum(kernel)
    else:
        x_axis = np.linspace(-3, 3, window_size)
        kernel = np.exp(-(x_axis ** 2) / 2)
        kernel = kernel / np.sum(kernel)
    return kernel


st.title("Moving Average als Faltung")
st.markdown(
    """
    Ein Moving Average ist ein einfacher Filter, bei dem ein Fenster über das Signal wandert und an jeder Stelle
    die Werte im Fenster gemittelt werden. Mathematisch ist das eine Faltung mit einem kompakten, meist gleichgewichteten Kern.
    """
)

with st.sidebar:
    st.header("Filterparameter")
    window_size = st.slider("Fensterbreite", 3, 21, 7, 2)
    window_type = st.selectbox(
        "Fensterform",
        ["Rechteck", "Dreieck", "Hamming", "Gauss"],
        index=0,
    )


t, original_signal = build_example_signal()
kernel = build_window(window_type, window_size)
filtered_ma = np.convolve(original_signal, kernel, mode="same")

st.subheader("1) Rauschhaftes Signal und schrittweise Glättung")
st.markdown(
    r"""
    Das Signal ist jetzt deutlich verrauscht. Ein Moving Average arbeitet dadurch, dass ein Fenster über das Signal wandert:

    $$
    y[n] = \sum_{k=0}^{M-1} x[n+k] \cdot h[k]
    $$

    Das Fenster nimmt an jeder Position genau die Werte aus seinem Bereich und bildet daraus einen Mittelwert.
    """
)

# smaller zoomed-in segment for a clearer view of the moving average effect
segment_start = 130
segment_end = 220
window_positions = [segment_start + 8, segment_start + 20, segment_start + 32, segment_start + 44, segment_start + 56]
fig1, axes1 = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
for ax in axes1.flat:
    ax.set_axis_off()

for idx, pos in enumerate(window_positions):
    start = max(0, pos - window_size // 2)
    end = min(len(original_signal), start + window_size)
    segment = original_signal[start:end]
    if len(segment) < window_size:
        continue
    ax = axes1[idx // 2, idx % 2]
    ax.set_axis_on()
    segment_t = t[start:end]
    ax.plot(t[segment_start:segment_end], original_signal[segment_start:segment_end], color="0.6", linewidth=1.5, label="Signal")
    ax.plot(t[:pos + 1], filtered_ma[:pos + 1], color="tab:green", linewidth=2, alpha=0.9, label="bis hier gefiltert")
    ax.axvspan(t[start], t[end - 1], color="tab:blue", alpha=0.18)
    ax.plot(segment_t, segment, color="tab:orange", linewidth=2)
    ax.scatter(segment_t, segment, color="tab:orange", s=20)
    mean_value = np.mean(segment)
    ax.hlines(mean_value, t[start], t[end - 1], colors="tab:red", linestyles="--", linewidth=2, label=f"Mittelwert = {mean_value:.3f}")
    ax.set_title(f"Fenster bei t={t[pos]:.3f}s")
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(t[segment_start], t[segment_end])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
st.pyplot(fig1)

st.markdown(
    """
    An jeder Position wird der Mittelwert des Fensters als neuer Wert ausgerechnet. Dadurch werden die großen, schnellen Sprünge
    im verrauschten Signal deutlich reduziert, während die langsamere Grundform erhalten bleibt.
    """
)

st.subheader("2) Gewichtung des Fensters")

fig2, axes2 = plt.subplots(2, 1, figsize=(11, 7))
axes2[0].stem(np.arange(len(kernel)), kernel, basefmt=" ", linefmt="C0", markerfmt="C0o")
axes2[0].set_title(f"Systemantwort / Fenster: {window_type}")
axes2[0].set_xlabel("n")
axes2[0].set_ylabel("Gewicht")
axes2[0].grid(True, alpha=0.3)

axes2[1].plot(t, original_signal, label="Originalsignal", color="0.5", alpha=0.8)
axes2[1].plot(t, filtered_ma, label=f"Gefiltert ({window_type}, M={window_size})", color="tab:orange", linewidth=2)
axes2[1].set_xlabel("Zeit (s)")
axes2[1].set_ylabel("Amplitude")
axes2[1].set_title("Glättung durch Moving Average")
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)

st.write(f"Filterkern: {np.round(kernel, 3)}")

freq_axis = np.fft.rfftfreq(1024, d=1 / 400)
mag_response = np.abs(np.fft.rfft(kernel, n=1024))
mag_response = mag_response / np.max(mag_response)

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(freq_axis, mag_response, color="tab:blue", linewidth=2)
ax3.set_title("Normierte Systemantwort")
ax3.set_xlabel("Frequenz (Hz)")
ax3.set_ylabel("|H(f)|")
ax3.set_xlim(0, 200)
ax3.grid(True, alpha=0.3)
st.pyplot(fig3)

st.subheader("3) Vorteile des Moving Average")
st.markdown(
    """
    - Der Filter ist einfach zu implementieren.
    - Er reduziert Rauschen wirksam, indem er lokale Schwankungen ausgleicht.
    - Die Wirkung ist leicht kontrollierbar, denn die Fensterbreite bestimmt direkt die Glättungsstärke.
    
    """
)

st.subheader("4) Nachteile des Moving Average")
st.markdown(
    """
    - Durch die Dämpfung hoher Frequenzen reduziert der Filter zwar das Rauschen, glättet dabei aber auch echte, hochfrequente Signalanteile wie schnelle Transienten.
    - Die Vergrößerung des Filterfensters optimiert die Glättungsleistung, führt jedoch zu einem Verlust an zeitlicher Auflösung und erhöhter Gruppenlaufzeit. Er führt eine Phasenverschiebung in den Signalformen ein, da die Faltung kausal ist und ausschließlich auf vergangenen Zuständen beruht.
    - Mit zunehmender Fensterbreite steigt der Rechenaufwand erheblich, da für jeden Ausgangswert eine größere Anzahl an vergangenen Datenpunkten verarbeitet werden muss, was die Berechnungszeit im laufenden Betrieb verlängert.
    - Während hochfrequente Komponenten gedämpft werden, bleiben niederfrequente Signale erhalten.

    Folglich ist dieser Filterart hochwirksam zur Rauschunterdrückung, zeigt jedoch eine schlechte Leistung bei der Erhaltung geometrischer oder transienter Details.
    """
)



