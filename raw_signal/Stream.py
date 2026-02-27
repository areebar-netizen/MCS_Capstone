from pylsl import StreamInlet, resolve_byprop


'''
This module probes for a MUSE EEG device connected via bluetooth to your local device using the pylsl library functions 
and starts an lsl stream. It also allows a user to retreive channel information from that stream.
'''
class Stream:

    def connect_to_eeg_stream(self):
        #Connects to EEG stream from MUSE
        print("Looking for EEG")
        stream = resolve_byprop('type', 'EEG', timeout=5)
        if not stream:
            raise RuntimeError("No EEG stream found")
        
        inlet = StreamInlet(stream[0])
        print("connected to EEG")
        return inlet

    def get_meta(self, inlet):
        #returns streamed channels from EEG device
        info = inlet.info()
        channels = []
        ch = info.desc().child('channels').child('channel')
        for i in range(info.channel_count()):
            chan = ch.child_value('label')
            channels.append(chan)
            ch = ch.next_sibling()
        
        return channels