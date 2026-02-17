import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt

'''This module computes the band powers for each kind of brain wave (alpha, beta, delta, gamma, theta) and provides visualizations showing topological maps of  
wave localization and average band power per wave.'''

class Compute_Band_Powers:

    def __init__(self, csv_file, sfreq=256, drop_channels=None):
        df = pd.read_csv(csv_file)

        if drop_channels:
            self.eeg_df = df.drop(columns=drop_channels, errors="ignore")
        else:
            self.eeg_df = df.copy()


        self.sfreq=sfreq
        self.data = self.eeg_df.to_numpy().T 
        self.ch_names = self.eeg_df.columns.tolist()

        self.info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types="eeg")
        self.raw = mne.io.RawArray(self.data, self.info)

        self.epochs_clean = None
        self.band_power = None
        self.rel_power = None

        self.bands =  {
            "Delta": (1, 4),
            "Theta": (4, 8),
            "Alpha": (8, 13),
            "Beta":  (13, 30),
            "Gamma": (30, 40),
        }

        montage = mne.channels.make_standard_montage("standard_1020")
        self.raw.set_montage(montage, match_case=False, on_missing="ignore")

    
    def preprocess(self):
        # preprocessing for band powers
        self.raw = self.raw.copy().set_eeg_reference("average")
        self.raw = self.raw.filter(1.5, 40, fir_design="firwin")
        self.raw = self.raw.notch_filter([60])  # use 50 if you're in a 50Hz mains region

    def create_epochs(self):
        #Splits the stream into epochs so that a stable average of band powers over different intervals can be obtained 
        epochs = mne.make_fixed_length_epochs(
            self.raw, duration=4.0, overlap=3.5, preload=True
        )

        # Reject bad windows
        self.raw.info['bads'] = ['TP10']  

        # create epochs
        self.epochs_clean = epochs.copy()

        #drops bad/noisy epochs
        reject_criteria = dict(eeg=300e-6)  # 100 μV
        epochs.drop_bad(reject=reject_criteria)

        # epochs.plot(n_epochs=15, scalings=dict(eeg=100e-6), block=True)        # before
        # epochs_clean.plot(n_epochs=15, scalings=dict(eeg=100e-6), block=True) # after

    def compute_powers(self):
       

        #compute power spectral density to separate power of signals by frequency
        spec = self.epochs_clean.compute_psd(method="welch", fmin=1, fmax=40, n_fft=1024)
        psds, freqs = spec.get_data(return_freqs=True)

        psds = psds.mean(axis=0)  # average over epochs -> (n_ch, n_freq)
        dfreq = freqs[1] - freqs[0]

        #compute total and relative band powers from psds
        self.band_power = {}
        for band, (fmin, fmax) in self.bands.items():
            idx = (freqs >= fmin) & (freqs < fmax)
            self.band_power[band] = psds[:, idx].sum(axis=1) * dfreq

        total_idx = (freqs >= 0) & (freqs < 40)  #obtain relevant frequency indicies
        total_power = psds[:, total_idx].sum(axis=1) * dfreq

        self.rel_power = {b: self.band_power[b] / (total_power + 1e-12) for b in self.band_power}

    def print_channel(self):
       
        # Print per-channel relative band power
        for band in self.bands:
            vals = self.band_power[band]
            print(band, {ch: float(v) for ch, v in zip(self.ch_names, vals)})

        

    def plot_channel(self, channel="AF7"):
        #Plot per channel relative band power

        total = (
            self.band_power["Delta"] + self.band_power["Theta"] + self.band_power["Alpha"] +
            self.band_power["Beta"]  + self.band_power["Gamma"]
        )
        rel_power = {b: self.band_power[b] / (total + 1e-12) for b in self.bands}

        ch_ix = self.epochs_clean.ch_names.index(channel)
        vals = [rel_power[b][ch_ix] for b in self.bands]
         
        plt.figure(figsize=(7,3.5))
        plt.bar(list(self.bands.keys()), vals)
        plt.ylabel("Relative power (fraction of 1–40 Hz)")
        plt.title(f"{channel} band power (relative)")
        plt.tight_layout()
        plt.show()

        print("Sum:", sum(vals))  # should be ~1

        

    def plot_psd_topomap(self, fmin=1, fmax=40):
        #Plots PSD topomap averaged over a frequency range. Allows you to see where each band power is highest
    
        spec = self.epochs_clean.compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax,
            n_fft=1024
        )

        spec.plot_topomap(
            ch_type="eeg",
            agg_fun=np.mean,
            normalize=False
        )
    
   
    def plot_sensors(self):
        self.raw.plot_sensors(kind="topomap", show_names=True)
        plt.show()
    
    def plot_band_summary(self, relative=True):
        #Plots relative band powers as a bar chart over ALL channels
        
        if relative:
            total = (
                self.band_power["Delta"] + self.band_power["Theta"] +
                self.band_power["Alpha"] + self.band_power["Beta"] +
                self.band_power["Gamma"]
            )
            avg = {
                band: (self.band_power[band] / (total + 1e-12)).mean()
                for band in self.bands
            }
            ylabel = "Relative power"
        else:
            avg = {band: self.band_power[band].mean() for band in self.bands}
            ylabel = "Power (V**2)"

        plt.figure(figsize=(7, 4))
        plt.bar(avg.keys(), avg.values(), color="skyblue")
        plt.ylabel(ylabel)
        plt.title("Average EEG Band Power (All Channels)")
        plt.tight_layout()
        plt.show()

def main():
    cbp = Compute_Band_Powers(
        csv_file="dataset/test/eeg_numeric_3.csv",
        drop_channels=["timestamps", "Right AUX"]
    )
    cbp.preprocess()
    cbp.create_epochs()
    cbp.compute_powers()

    cbp.plot_sensors()
    cbp.plot_band_summary()

    cbp.plot_psd_topomap(1, 40)

if __name__ == "__main__":
    main()

