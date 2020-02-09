import cv2
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer


def run(source, template_source):
    vid = cv2.VideoCapture(source)
    template = cv2.imread(template_source, 0)
    while vid.isOpened():
        ret, frame = vid.read()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_cropped = img_gray[625:675,940:980]

        res = cv2.matchTemplate(img_cropped, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        print('Found:', len(loc[0]))

        position = (850, 700)
        cv2.putText(frame, str(len(loc[0])), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 3) 

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        run('vid2.mp4', 'x.png')
    except:
        pass