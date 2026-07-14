import streamlit as st
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
 
st.set_page_config(page_title="Abtastung & Rekonstruktion", layout="wide")
 
st.title("Abtastung und Signalrekonstruktion")

st.markdown('''
## Abtasttheorem
Das Abtasttheorem sagt aus, dass ein kontinuierliches Signal nur dann korrekt abgetastet werden kann, wenn es keine Frequenzkomponenten über der Hälfte der Abtastfrequenz beinhaltet
oder

𝑓𝑠>2𝑓𝑚𝑎𝑥

fs: Abtastfrequenz, 

fmax: Maximale Frequenz in einem Signal 

Ein 2-Hz-Sinus-Signal ist gegeben.''')

# --- Feste Signalparameter ---
f1 = 2          # Signalfrequenz in Hz
t_min = 0
t_max = 2       # Sekunden
 
@st.cache_data
def get_original_signal(t_min, t_max, f1, n_points=1000):
    """Fein aufgelöstes 'Originalsignal' - wird nur einmal pro Parametersatz berechnet."""
    t1 = np.linspace(t_min, t_max, n_points)
    xt = np.sin(2 * np.pi * f1 * t1)
    return t1, xt
 
t1, xt = get_original_signal(t_min, t_max, f1)
 
# --- SelectSlider für die Abtastfrequenz ---
fs_options = [2, 3, 4, 6, 8, 10, 16, 20, 32, 50]
fs = st.select_slider(
    "Abtastfrequenz fs (Hz)",
    options=fs_options,
    value=8
)
Ts = 1 / fs
 
st.caption(f"Ts = 1/fs = {Ts:.4f} s")
 
@st.cache_data
def get_sampled_signal(t_min, t_max, Ts, f1):
    """Abtastung des Signals - Ergebnis wird für jede (fs)-Kombination zwischengespeichert."""
    impulse_train = np.arange(t_min, t_max + 0.1, Ts)  # 0.1 addiert, um t_max einzuschließen
    xt_sampled = np.sin(2 * np.pi * f1 * impulse_train)
    return impulse_train, xt_sampled
 
@st.cache_data
def get_reconstructed_signal(xt_sampled, impulse_train, n_points=1000):
    """Rekonstruktion durch Resampling - ebenfalls gecacht."""
    x_rs, t_rs = signal.resample(xt_sampled, n_points, impulse_train)
    return x_rs, t_rs
 
impulse_train, xt_sampled = get_sampled_signal(t_min, t_max, Ts, f1)
x_rs, t_rs = get_reconstructed_signal(xt_sampled, impulse_train)
 
# --- Drei Spalten für die drei Plots ---
col1, col2, col3 = st.columns(3)
 
with col1:
    fig1, ax1 = plt.subplots()
    ax1.stem(impulse_train, np.ones(len(impulse_train)))
    ax1.set_xlabel('Zeit (s)')
    ax1.set_ylabel('Amplitude (v)')
    ax1.set_title('Impulse train')
    st.pyplot(fig1)
 
with col2:
    fig2, ax2 = plt.subplots()
    ax2.stem(impulse_train, xt_sampled, linefmt='b-', markerfmt='bo', basefmt=' ')
    ax2.plot(t1, xt, 'r')
    ax2.set_xlabel('Zeit (s)')
    ax2.set_ylabel('Amplitude (v)')
    ax2.set_title(f'Abgetastetes Signal ({f1} Hz, fs={fs} Hz)')
    st.pyplot(fig2)
 
with col3:
    fig3, ax3 = plt.subplots()
    ax3.plot(t_rs, x_rs, label='Rekonstruiert')
    ax3.plot(t1, xt, 'r', label='Original')
    ax3.set_xlabel('Zeit (s)')
    ax3.set_ylabel('Amplitude (v)')
    ax3.set_title('Signalrekonstruktion')
    ax3.legend()
    st.pyplot(fig3)
 
# Hinweis auf das Abtasttheorem
nyquist = 2 * f1
if fs < nyquist:
    st.warning(f"fs = {fs} Hz liegt unter der Nyquist-Rate ({nyquist} Hz) — Aliasing tritt auf!")
else:
    st.success(f"fs = {fs} Hz erfüllt das Abtasttheorem (Nyquist-Rate = {nyquist} Hz).")
 
