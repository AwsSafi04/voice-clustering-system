#AUDIO RECORDING FILE 

import sounddevice as sd
import soundfile as sf

def record_speaker(speaker_name , duration=60) :
    print(f'Recording {speaker_name} for {duration} seconds')
    print('press ENTER to start recording')
    input()

    fs = 16000
    audio = sd.rec(int(duration*fs) , samplerate=fs , channels=1)
    print('Recording... speak now!')
    sd.wait()

    filename = f'{speaker_name}.wav'
    sf.write(filename , audio , fs)
    print(f'Saved {filename}')


record_speaker('person1',60)
record_speaker('person2',60)
record_speaker('person3',60)