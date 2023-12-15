import cv2
import time
import math
import numpy as np

ideal_golden_ratio = 1.618

def removeOutliers(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    low = mean - std * 2
    high = mean + std * 2
    return [x for x in arr if low <= x <= high]

def rate_face(user_golden_ratio):
    # Calculate the difference between actual and ideal ratios
    ratio_difference = abs(user_golden_ratio - ideal_golden_ratio)

    # Calculate the face rating
    face_rating = 10 - min(ratio_difference, 1) * 10

    return face_rating

def main():
    top2pupil = []
    pupil2lip = []
    noseWidth = []
    nose2lips = []

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade_nose.xml')
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    startTime = time.time()

    while True:
        ret, img = cap.read()
        height, width, channels = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 2.5, 5)
            smiles = smile_cascade.detectMultiScale(roi_gray, 3.4, 5)
            noses = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 1)
            for (nx, ny, nw, nh) in noses:
                cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (255, 0, 255), 1)

            if time.time() > (startTime + 2) and len(eyes) == 2 and len(smiles) == 1 and len(faces) == 1 and len(noses) == 1:
                cv2.putText(img, 'Scanning Face...', (math.floor(width / 3), math.floor(height / 12)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(roi_color, (int(math.floor(w / 2)), int(sy + math.floor(sh / 2.5))), 2, (255, 255, 255), 2)  # lips
                cv2.circle(roi_color, (int(math.floor(w / 3)), int(ey + math.floor(eh / 2))), 2, (255, 255, 255), 2)  # eye
                cv2.circle(roi_color, (int(nx + math.floor(nw / 2)), int(ny + math.floor(nh / 2))), 2, (255, 255, 255), 2)  # nose
                top2pupil.append(ey + (eh / 2))
                pupil2lip.append((sy + (sh / 2.5)) - (ey + (eh / 2)))
                noseWidth.append(0.75 * nw)
                nose2lips.append((sy + (sh / 3)) - (ny + (nh / 2)))

        cv2.imshow("Face Detector", img)
        k = cv2.waitKey(30) & 0xff
        if k == 27 or len(top2pupil) > 40:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("The ideal golden ratio is ", ideal_golden_ratio)

    if len(top2pupil) < 2 or len(pupil2lip) < 2 or len(noseWidth) < 1 or len(nose2lips) < 1:
        print("Insufficient data for calculation.")
    else:
        top2pupil = removeOutliers(top2pupil)
        pupil2lip = removeOutliers(pupil2lip)
        noseWidth = removeOutliers(noseWidth)
        nose2lips = removeOutliers(nose2lips)

        if not top2pupil or not pupil2lip or not noseWidth or not nose2lips:
            print("Insufficient data after removing outliers.")
        else:
            avg = (np.mean(top2pupil) / np.mean(pupil2lip) + np.mean(noseWidth) / np.mean(nose2lips)) / 2
            print("Your calculated ratio is: " + str(avg))
            rating = rate_face(avg)
            print(f"Your face rating is: {rating:.2f}/10")

if __name__ == "__main__":
    main()
