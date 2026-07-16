import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Quantisierung", layout="wide")

st.title("Quantisierung von Signalen")

st.markdown("""
Quantisierung ist der Prozess, kontinuierliche Werte auf eine kleinere Menge diskreter Werte abzubilden.
Dabei entsteht ein Quantisierungsfehler, weil ein analoges Signal nur näherungsweise durch digitale Stufen dargestellt werden kann.
""")

st.sidebar.header("Parameter")

f = st.sidebar.slider("Signalfrequenz f (Hz)", 1, 20, 5, 1)
amplitude = st.sidebar.slider("Signalamplitude", 0.5, 3.0, 1.5, 0.1)
noise_std_dev = st.sidebar.slider("Rauschstandardabweichung σ", 0.001, 0.5, 0.01, 0.001)
FS = st.sidebar.slider("Full Scale FS", 0.5, 5.0, 2.0, 0.1)
num_bits = st.sidebar.slider("Quantisierungsbits", 2, 8, 4, 1)

fs = 1000
Duration = 1.0
t = np.linspace(0, Duration, int(fs * Duration), endpoint=False)

np.random.seed(42)
signal_clean = amplitude * np.sin(2 * np.pi * f * t)
noise = np.random.normal(0, noise_std_dev, size=t.shape)
signal_noisy = signal_clean + noise


def quantize(signal, num_bits, FS):
    """Mid-tread quantizer with 0 as a valid level."""
    min_int = -(2 ** (num_bits - 1))
    max_int = (2 ** (num_bits - 1)) - 1

    scaled = signal * max_int / FS
    quantized_int = np.round(scaled).astype(int)
    quantized_int = np.clip(quantized_int, min_int, max_int)

    quantized = quantized_int / max_int * FS
    bit_strings = [format(val & (2**num_bits - 1), f'0{num_bits}b') for val in quantized_int]
    return quantized, quantized_int, bit_strings


@st.cache_data
def compute_quantization(signal, bits, full_scale):
    quantized_signal, quantized_int, bit_strings = quantize(signal, bits, full_scale)
    error = signal - quantized_signal
    return quantized_signal, quantized_int, bit_strings, error


def calculate_lsb(bit_depth, full_scale):
    return full_scale / (2**bit_depth)


def min_bit_depth_for_snr(snr_db):
    return math.ceil((snr_db - 1.76) / 6.02)


quantized_signal, quantized_int, bit_strings, quant_error = compute_quantization(signal_noisy, num_bits, FS)
snr_db = 20 * math.log10(amplitude / noise_std_dev)
lsb = calculate_lsb(num_bits, FS)
min_bits = min_bit_depth_for_snr(snr_db)

st.subheader("Schritt 1: Quantisierung und Kennwerte")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("SNR", f"{snr_db:.1f} dB")
with col2:
    st.metric("LSB-Wert", f"{lsb:.4f}")
with col3:
    st.metric("Mindestbittiefe", f"{min_bits} Bits")

st.write(
    "Der LSB-Wert beschreibt die kleinste Änderung, die der Quantisierer noch unterscheiden kann. "
    "Die Schrittweite des Quantisierers (LSB) sollte so gewählt werden, dass sie knapp unterhalb "
    "des analogen Grundrauschens liegt; eine feinere Quantisierung würde lediglich wertlose "
    "Rauschdaten digitalisieren, während eine zu grobe Quantisierung zu Signalverzerrungen führt."
)


st.subheader("Schritt 2: Originalsignal und quantisiertes Signal")

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(t, signal_noisy, label="Original", alpha=0.7, color="tab:blue")
axes[0].plot(t, quantized_signal, label=f"{num_bits}-bit Quantisiert", color="tab:orange")
axes[0].set_title(f"Signal mit {num_bits}-bit-Quantisierung")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(t, quant_error, color="tab:red")
axes[1].set_title("Quantisierungsfehler")
axes[1].set_xlabel("Zeit (s)")
axes[1].set_ylabel("Fehler")
axes[1].grid(True)

st.pyplot(fig)

st.subheader("Schritt 3: Interpretation")

if lsb > noise_std_dev:
    st.info(
        f"Der LSB-Wert ({lsb:.4f}) ist größer als die Rauschstandardabweichung ({noise_std_dev:.4f}). "
        "Damit ist die Quantisierung grober als das vorhandene Rauschen."
    )
else:
    st.info(
        f"Der LSB-Wert ({lsb:.4f}) ist kleiner oder gleich der Rauschstandardabweichung ({noise_std_dev:.4f}). "
        "Damit können kleinere Signaländerungen noch aufgelöst werden."
    )

st.write(
    f"Für den gewählten Parametersatz würde eine Mindestbittiefe von {min_bits} Bits für die gewünschte "
    "Signalqualität sinnvoll erscheinen. ")
st.write(
    "Wenn das Signal stark verrauscht ist, hilft eine höhere Bittiefe nicht automatisch weiter: "
    "Dann werden auch kleine Rauschschwankungen mitquantisiert. Das verbessert"
    "nicht unbedingt die Qualität des eigentlichen Signals, und macht den Aufbau teurer und speicherintensiver."
)

st.subheader("Schritt 4: Vollskalen-Effekt")

st.write(
    "Ein größerer Full-Scale-Bereich verteilt die verfügbaren Stufen über einen größeren Bereich. "
    "Dadurch wird der Schrittabstand zwischen benachbarten Quantisierungsstufen größer, also die Quantisierung bei gleichem Signal grober. "
    "Ein kleinerer FS macht die Stufen enger und damit feiner, erhöht aber die Gefahr von Übersteuerung, wenn die Signalamplitude den Bereich übersteigt."
)
