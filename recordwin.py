# import the necessary packages
import numpy as np
import pyautogui
import cv2

show = False
template_source = 'x.png'
template = cv2.imread(template_source, 0)

frames_since_last_detection = 0
detection_count = 0
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
    #print('Found:', found)

    if show:
        position = (50, 100)
        cv2.putText(frame, str(detection_count), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 3) 

        position = (850, 700)
        cv2.putText(frame, str(found), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 3) 

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()