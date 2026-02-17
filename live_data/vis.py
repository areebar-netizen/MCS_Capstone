##With AI Assistance

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph.exporters
from Stream import Stream

'''This module creates an initial data visualization for the streamed EEG MUSE LSL data. 
This uses the pyqtgraph python library to display the raw EEG signals live. Plots of this graph can also be saved by pressing the S key.
The channels shown for this visualization include the AUX channel which just records ouside noise.'''
##Params

N_CH = 5
CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
BUFFER_SEC = 5
SFREQ = 256
Y_RANGE = (-200, 200)
BUFFER_SIZE = SFREQ* BUFFER_SEC



class EEG_Plot:
    def __init__(self, inlet, labels, buffer_size=BUFFER_SIZE, y_range=Y_RANGE):
        self.inlet=inlet
        self.labels=labels
        self.n_ch=len(labels)
        self.buffer_size = buffer_size
        self.y_range = y_range


        # Create initial Qt  window
       
        self.win = pg.GraphicsLayoutWidget(title="Muse EEG Plots")
        self.win.resize(1100, 180 * self.n_ch)
        self.win.show()

         # Qt will save current plot with press of S key
        def keyPressEvent(self, event):
            if event.key() == QtCore.Qt.Key_S:
                self.save_plot("muse_eeg.png")
        self.win.keyPressEvent = keyPressEvent
        

        # Create plots + curves
        self.curves = []
        for i, lab in enumerate(labels):
            p = self.win.addPlot(row=i, col=0)          # row=i (not 1)
            p.setLabel("left", f"{lab}")
            p.setYRange(*self.y_range)                  # unpack tuple
            p.showGrid(x=True, y=True)
            self.curves.append(p.plot(pen="c"))

        # Data buffer
        self.data = np.zeros((self.n_ch, self.buffer_size), dtype=np.float32)

    def update(self):
        '''Pulls a sample from the stream, updates the data buffer and redraws the plot'''
        
        sample, _ = self.inlet.pull_sample(timeout=0.0)
        if sample is None:
            return

        # slide left, append newest at end
        self.data[:, :-1] = self.data[:, 1:]
        self.data[:, -1] = sample[:self.n_ch]

        for i in range(self.n_ch):
            self.curves[i].setData(self.data[i])

    def save_plot(self, filename="muse_eeg.png"):
        '''Exports current plot to a png'''

        exporter = pg.exporters.ImageExporter(self.win.scene())
        exporter.parameters()["width"] = 1200
        exporter.export(filename)
        print(f"Saved plot to {filename}")

    
    

def main():
    app = QtWidgets.QApplication([])

    stream = Stream()
    inlet = stream.connect_to_eeg_stream()
    labels = stream.get_meta(inlet)

    
   
    plot = EEG_Plot(inlet, labels)

    timer = QtCore.QTimer()
    timer.timeout.connect(plot.update)  # update takes NO args
    timer.start(16)                # ~60 FPS

    pg.exec()

    

   

if __name__ == "__main__":
    main()