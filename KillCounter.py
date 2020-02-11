import numpy as np
import pyautogui
import cv2
#from record import trigger
import threading
import time
import argparse
import pyaudio
import wave
from panotti.datautils import *
from predict_class import *
import queue


# Hacky fix for some weird error
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

# Where we will push and pop from when threading
my_queue = queue.Queue()
my_queue_2 = queue.Queue()

#Preload model
model, class_names = load_model_ext('weights.hdf5')
expected_melgram_shape = model.layers[0].input_shape[1:]

# Long blocking func eg(audio detection)
def audio_classification():
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True

    signal, sr = load_audio('output.wav', mono=True, sr=44100)
    y_proba = predict_one(signal, sr, model, expected_melgram_shape) # class_names, model, weights_file=args.weights)
    
    nb_classes = len(class_names)
    for i in range(nb_classes):
        print( class_names[i],": ",y_proba[i],", ",end="",sep="")
    answer = class_names[ np.argmax(y_proba)]
    #print("--> ANSWER:", answer)

    # Put data into queue so main thread knows what we did (Category from Audio Detection)
    my_queue_2.put(answer)


# Launches a long blocking func in another thread
def trigger_audio_classification():
    thread = threading.Thread(target=audio_classification)
    thread.start()
    print("Spun off thread")


# Launches a long blocking func in another thread
def start_audio_recording():
    thread = threading.Thread(target=trigger_recording)
    thread.start()
    print("Spun off thread")

def trigger_all():
    start_audio_recording()

def trigger_recording(RECORD_SECONDS=2):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    #RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    print('frames:',len(frames))

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    trigger_audio_classification()  

    #while True:
    #    try:
    #        answer = my_queue.get_nowait()  # non-blocking queue check
    #        break # got result
    #    except:
    #        pass  # nothing in queue
    

    #print("Exited record.trigger()")
    #return (answer == "Death")




show = False
template_source = 'x.png'
template = cv2.imread(template_source, 0)


frames_since_last_detection = 0
detection_count = 0
kill_count = 0
while True:
    frames_since_last_detection = frames_since_last_detection+1
    frame = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_cropped = img_gray[625:675,940:980]

    res = cv2.matchTemplate(img_cropped, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    found = len(loc[0])
    if found > 0 and frames_since_last_detection >= 30:
        detection_count = detection_count+1
        print(f"New Detection! {frames_since_last_detection}")
        frames_since_last_detection = 0

        print("running audio classification")
        trigger_all()
        
    #print('Found:', found)

    try:
        answer = my_queue_2.get_nowait()  # non-blocking queue check
        print(f"Got result: {answer}")
        if answer == "Death":
            kill_count = kill_count+1
            print(f'Kill count: {kill_count}')
    except:
        pass  # nothing in queue

    if show:
        position = (50, 100)
        cv2.putText(frame, str(detection_count), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 3) 

        position = (50, 200)
        cv2.putText(frame, str(kill_count), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 3) 

        position = (850, 700)
        cv2.putText(frame, str(found), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 3) 

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
