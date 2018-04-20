import cv2
import numpy as np
import train_face_recognition as train

def get_face_predicition(image):
    try:
        face, rect = train.detect_face(image)
        label, confidence_distance = face_recognizer.predict(face)
        if (label == 1):
            return "ME: " + str(confidence_distance)
        else:
            return "NOT ME: " + str(confidence_distance)
    except:
        return "UNKNOWN"

def main():
    cascade_classifier = cv2.CascadeClassifier(train.HAAR_CASCADE_PATH)

    video_capture = cv2.VideoCapture(0)
    video_capture.set(3,800)
    video_capture.set(4,480)

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) #,flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            #print(len((x, y, w, h)))
            #print("Found {0} faces!".format(len(faces)))
            #print((x, y, w, h))
            #print(frame)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            train.display_name_over_frame(frame, get_face_predicition(frame), x, y) # need to change param in get_face_predicition in order to detect multiple faces
            # right now only works for single face

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("./trained_face_recognizer.xml")
    main()
