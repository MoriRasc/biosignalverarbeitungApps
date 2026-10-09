
import math

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Quantisierung", layout="wide")

st.title("Quantisierung von Signalen")

st.markdown("""
Quantisierung bildet kontinuierliche Amplitudenwerte auf eine endliche Menge
digitaler Stufen ab. Dabei entsteht ein Quantisierungsfehler. Zusätzliches
analoges Rauschen ist davon zu unterscheiden: Es ist bereits im Signal vorhanden,
bevor der ADC quantisiert.
""")

st.sidebar.header("Parameter")

f = st.sidebar.slider("Signalfrequenz f (Hz)", 1, 20, 5, 1)
amplitude = st.sidebar.slider(
    "Signalamplitude (Spitzenwert)", 0.5, 3.0, 1.5, 0.1
)
noise_std_dev = st.sidebar.slider(
    "Analoges Rauschen σ (Standardabweichung)",
    0.001, 0.5, 0.01, 0.001
)
FS = st.sidebar.slider(
    "Full-Scale-Spitzenwert FS (V)", 0.5, 5.0, 2.0, 0.1
)
num_bits = st.sidebar.slider("Quantisierungsbits", 2, 8, 4, 1)

fs = 1000
duration = 1.0
t = np.arange(0, duration, 1 / fs)

rng = np.random.default_rng(42)
signal_clean = amplitude * np.sin(2 * np.pi * f * t)
noise = rng.normal(0, noise_std_dev, size=t.shape)
signal_noisy = signal_clean + noise


def quantize(signal, num_bits, full_scale_peak):
    """
    Uniform mid-tread quantizer with zero as a valid level.

    Input range: -FS to +FS.
    Quantization step: 2*FS / 2**num_bits.

    Signed two's-complement codes are asymmetric at the endpoints:
    the negative endpoint is representable, but the positive endpoint
    is one LSB below +FS.
    """
    levels = 2**num_bits
    lsb = (2 * full_scale_peak) / levels

    min_code = -(2 ** (num_bits - 1))
    max_code = (2 ** (num_bits - 1)) - 1

    quantized_int = np.rint(signal / lsb).astype(int)
    quantized_int = np.clip(
        quantized_int, min_code, max_code
    )

    quantized = quantized_int * lsb

    bit_strings = [
        format(code & (levels - 1), f"0{num_bits}b")
        for code in quantized_int
    ]

    return quantized, quantized_int, bit_strings


@st.cache_data
def compute_quantization(signal, bits, full_scale_peak):
    quantized_signal, quantized_int, bit_strings = quantize(
        signal, bits, full_scale_peak
    )
    error = signal - quantized_signal

    return (
        quantized_signal,
        quantized_int,
        bit_strings,
        error
    )


def calculate_lsb(bit_depth, full_scale_peak):
    # FS is the positive peak magnitude: range is -FS to +FS.
    return (2 * full_scale_peak) / (2**bit_depth)


def min_bit_depth_for_snr(snr_db):
    # Ideal full-scale sine-wave quantization SNR:
    # SNR ~= 6.02*N + 1.76 dB.
    return max(1, math.ceil((snr_db - 1.76) / 6.02))


quantized_signal, quantized_int, bit_strings, quant_error = (
    compute_quantization(signal_noisy, num_bits, FS)
)

# SNR of the clean sine relative to the added analog noise.
# For a sine wave, signal RMS = amplitude / sqrt(2).
signal_rms = amplitude / math.sqrt(2)
snr_analog_db = 20 * math.log10(
    signal_rms / noise_std_dev
)

lsb = calculate_lsb(num_bits, FS)
min_bits_ideal = min_bit_depth_for_snr(snr_analog_db)

# Detect samples outside the selected input range.
overload_count = int(
    np.count_nonzero(
        (signal_noisy < -FS) | (signal_noisy > FS)
    )
)
overload_percent = 100 * overload_count / signal_noisy.size

# Empirical quantization error and quantization SNR.
quant_error_rms = float(
    np.sqrt(np.mean(quant_error**2))
)

signal_quant_snr_db = (
    20 * math.log10(signal_rms / quant_error_rms)
    if quant_error_rms > 0
    else float("inf")
)


st.subheader("Schritt 1: Quantisierung und Kennwerte")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "SNR Signal / analoges Rauschen",
        f"{snr_analog_db:.1f} dB"
    )

with col2:
    st.metric("LSB-Schrittweite", f"{lsb:.4f} V")

with col3:
    st.metric(
        "Ideal abgeschätzte Mindestbittiefe",
        f"{min_bits_ideal} Bits"
    )

st.write(
    f"Das saubere Sinussignal hat einen Effektivwert von "
    f"{signal_rms:.4f} V. Das Verhältnis dieses Effektivwerts "
    f"zur Rauschstandardabweichung ergibt ein Signal-Rausch-"
    f"Verhältnis von {snr_analog_db:.1f} dB."
)

st.write(
    f"Bei dieser Implementierung bezeichnet FS den positiven "
    f"Spitzenwert: Der nominale Eingangsbereich ist "
    f"−{FS:.2f} V bis +{FS:.2f} V. Die Schrittweite beträgt "
    f"2 · FS / 2ᴺ = {lsb:.4f} V."
)

st.caption(
    "Die Mindestbittiefe ist nur eine grobe theoretische "
    "Orientierung: Die bekannte Formel SNR ≈ 6,02·N + 1,76 dB "
    "gilt für einen idealen Quantisierer und ein Sinussignal, "
    "das den verfügbaren Bereich nahezu voll ausnutzt. Sie "
    "berücksichtigt weder ADC-Nichtidealitäten noch Übersteuerung "
    "oder die konkrete analoge Rauschquelle."
)


st.subheader("Schritt 2: Originalsignal und quantisiertes Signal")

fig, axes = plt.subplots(
    2, 1, figsize=(10, 6), sharex=True
)

axes[0].plot(
    t,
    signal_noisy,
    label="Signal + analoges Rauschen",
    alpha=0.7,
    color="tab:blue"
)

axes[0].plot(
    t,
    quantized_signal,
    label=f"{num_bits}-Bit-Ausgang",
    color="tab:orange"
)

axes[0].axhline(
    FS, linestyle="--", linewidth=1,
    color="gray", label="±FS-Grenzen"
)
axes[0].axhline(
    -FS, linestyle="--", linewidth=1, color="gray"
)

axes[0].set_title(
    f"Signal mit {num_bits}-Bit-Quantisierung"
)
axes[0].set_ylabel("Amplitude (V)")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(
    t, quant_error, color="tab:red"
)
axes[1].axhline(
    0, linewidth=0.8, color="gray"
)
axes[1].set_title(
    "Quantisierungsfehler: Eingang − quantisierter Ausgang"
)
axes[1].set_xlabel("Zeit (s)")
axes[1].set_ylabel("Fehler (V)")
axes[1].grid(True)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


st.subheader("Schritt 3: Interpretation")

if overload_count:
    st.warning(
        f"{overload_count} von {signal_noisy.size} Samples "
        f"({overload_percent:.2f} %) liegen außerhalb von ±FS. "
        "Diese Samples werden begrenzt (Clipping). Ein größerer "
        "Full-Scale-Bereich oder eine kleinere Signalamplitude "
        "kann Übersteuerung vermeiden."
    )
else:
    st.success(
        "Kein Sample des verrauschten Signals überschreitet "
        "den Bereich ±FS."
    )


if lsb > noise_std_dev:
    st.info(
        f"Die LSB-Schrittweite ({lsb:.4f} V) ist größer als "
        f"die Rauschstandardabweichung ({noise_std_dev:.4f} V). "
        "Das analoge Rauschen ist relativ zur Quantisierungsstufe "
        "klein. Kleine Änderungen am Eingang können deshalb im "
        "selben ADC-Code bleiben."
    )

else:
    st.info(
        f"Die LSB-Schrittweite ({lsb:.4f} V) ist kleiner oder "
        f"gleich der Rauschstandardabweichung "
        f"({noise_std_dev:.4f} V). Das analoge Rauschen kann "
        "benachbarte ADC-Codes anregen. Daraus folgt aber nicht "
        "automatisch, dass kleinere Signaländerungen zuverlässig "
        "erkennbar sind. Unter geeigneten Bedingungen können "
        "Oversampling und Mittelung die Schätzung kleiner "
        "Änderungen verbessern; die intrinsische ADC-Schrittweite "
        "bleibt gleich."
    )


st.write(
    f"Der aus den simulierten Daten berechnete RMS-"
    f"Quantisierungsfehler beträgt {quant_error_rms:.4f} V. "
    f"Das Verhältnis des RMS-Werts des sauberen Signals zum "
    f"RMS-Quantisierungsfehler beträgt in dieser Simulation "
    f"{signal_quant_snr_db:.1f} dB. Dieser empirische Wert hängt "
    "vom Signal, der Quantisierer-Auslastung, dem Rauschen und "
    "möglichem Clipping ab."
)

st.write(
    "Mehr Bits verkleinern die Quantisierungsstufe, beseitigen "
    "aber kein analoges Rauschen. Wenn das analoge Rauschen "
    "bereits die relevante Signaländerung überdeckt, bringt eine "
    "höhere nominelle ADC-Auflösung allein nicht unbedingt einen "
    "entsprechenden Gewinn an nutzbarer Messinformation."
)


st.subheader("Schritt 4: Vollskalen-Effekt")

st.write(
    "Bei gleicher Bitzahl führt ein größerer Full-Scale-"
    "Spitzenwert zu einer größeren LSB-Schrittweite und damit "
    "zu gröberer Quantisierung. Ein kleinerer Full-Scale-Bereich "
    "macht die Stufen feiner, erhöht aber das Risiko von Clipping, "
    "wenn Signal plus Rauschen den Eingangsbereich überschreiten."
)

st.write(
    f"Aktueller Bereich: −{FS:.2f} V bis +{FS:.2f} V. "
    f"Die Signalamplitude beträgt {amplitude:.2f} V "
    f"(Spitzenwert), die Rauschstandardabweichung "
    f"{noise_std_dev:.3f} V."
)