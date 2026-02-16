import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop
import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns

df = pd.read_csv("eeg_numeric_3.csv")

eeg_df = df.drop(columns=["lsl_timestamp", "Right AUX (microvolts)"])

sfreq = 256  # Muse is 256 Hz
data = eeg_df.to_numpy().T * 1e-6  # µV -> V
ch_names = eeg_df.columns.tolist()

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
raw = mne.io.RawArray(data, info)
print(raw.info)

rename = {
    "TP9 (microvolts)": "TP9",
    "TP10 (microvolts)": "TP10",
    "AF7 (microvolts)": "AF7",
    "AF8 (microvolts)": "AF8",
}
raw.rename_channels(rename)
raw.filter(1, 40, fir_design='firwin', verbose=False)

raw.info['bads'] = ['TP10']

montage_std= mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage_std,match_case=False, on_missing='ignore')

epochs = mne.make_fixed_length_epochs(
    raw, duration=4.0, overlap=3.5, preload=True
)

# Reject bad windows
epochs_clean = epochs.copy().drop_bad(reject=dict(eeg=150e-6))
print("Dropped:", len(epochs) - len(epochs_clean))

bands = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta":  (13, 30),
    "Gamma": (30, 40),
}

# --- 4) PSD (V^2/Hz) ---
spec = epochs_clean.compute_psd(method="welch", fmin=1, fmax=40, n_fft=1024, exclude='bads')
psds, freqs = spec.get_data(return_freqs=True)

band_power = {}
for band, (fmin, fmax) in bands.items():
    mask = (freqs >= fmin) & (freqs <= fmax)
    band_power[band] = psds[:, :, mask].mean(axis=-1)  # (epochs, channels)




bands_list = list(band_power.keys())
n_epochs, n_channels = next(iter(band_power.values())).shape

df_power = pd.DataFrame({
    band: bp.mean(axis=1)   # average over channels -> shape (n_epochs,)
    for band, bp in band_power.items()
})

# Example: Plot average power across all channels for each band
avg_band_power = df_power.mean() # Average across channels
print("Average Band Power across all EEG channels:")
print(avg_band_power)

# Plotting the average band power
plt.figure(figsize=(8, 5))
avg_band_power.plot(kind='bar', color='skyblue')
plt.title('Average EEG Frequency Band Power')
plt.xlabel('EEG Band')
plt.ylabel('Power Spectral Density (µV²/Hz)')
plt.show()

# Visualize topography for a specific band (e.g., Alpha)
# alpha_power_values = df_power['Alpha'].values
# mne.viz.plot_topomap(alpha_power_values, epochs.info, show_names=True, cmap='viridis', sensors=True, res=32)
# plt.title('Alpha Band Power Topography')
# plt.show()





# raw.plot_sensors(kind='topomap', show_names=True)
# plt.show()

# raw.plot(start=10, block=True)
# #plt.show(True)


# print(raw.info)

# spectrum= raw.compute_psd(fmax=40)
# spectrum.plot(average=True, picks='data', exclude='bads', amplitude=False)
spec.plot_topomap()
plt.show(block=True)
