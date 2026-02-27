##With AI Assistance

import time
import csv
from Stream import Stream
import os

'''This module allows a user to collected MUSE EEG data from an lsl stream and save it to a file.'''

class StreamTable:
        
    def save_table(self, channels, inlet, base='eeg_table', t=60*3):
        print("Saving EEG stream to CSV...")


        # timestamp is based on stream sampling frequency 256 Hz
        header = ['timestamps'] + [f'{c}' for c in channels]

        counter=1
        while(os.path.isfile(f'{base}_{counter}.csv')):
            counter+= 1
        
        fname = f'{base}_{counter}.csv'

        #Create a new file
        with open(fname, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            print(f"Created {fname}")

            #Start timer
            end_time = time.time() + t
            while time.time() < end_time:
                #pulls up to 512 samples at a time, waits up to 1 second to obtain samples if none are currently available by the stream
                samples, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=512)

                #Writes currently available samples to file
                if timestamps:
                    for ts, sample in zip(timestamps, samples):
                        writer.writerow([ts] + list(sample))

                    # helps ensure data is written even if you stop abruptly
                    f.flush()

def main():
    TIME = 60*1 # 1 min

    #Connect to EEG Device
    stream = Stream()
    inlet = stream.connect_to_eeg_stream()
    channels = stream.get_meta(inlet)

    #Create Table
    table = StreamTable()
    table.save_table(channels, inlet, 'areeba_tests/eeg_table', TIME)

if __name__ == '__main__':
    main()

   