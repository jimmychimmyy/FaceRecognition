import cv2
import os
import numpy as np

HAAR_CASCADE_PATH = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def detect_face(snapshot):
    gray = cv2.cvtColor(snapshot, cv2.COLOR_BGR2GRAY)
    haar_face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    face = haar_face_cascade.detectMultiScale(snapshot, scaleFactor=1.1, minNeighbors=5)
    if (len(face) == 0):
        return None, None
    x, y, w, h = face[0]
    return gray[y:y+w, x:x+h], face[0]

def prepare_my_faces():
    faces = []
    labels = []

    my_faces = os.listdir("./my_faces")
    for snapshot in my_faces:
        if snapshot.startswith("."):
            continue
        image = cv2.imread("./my_faces/"+snapshot)
        face, rect = detect_face(image)
        faces.append(face)
        labels.append(1)

    return faces, labels

def prepare_other_faces():
    faces = []
    labels = []

    orl_faces = os.listdir("./orl_faces")
    for snapshot in orl_faces:
        if snapshot.startswith("."):
            continue
        if snapshot.startswith("README"):
            continue
        image = cv2.imread("./orl_faces/"+snapshot)
        face, rect = detect_face(image)
        if (face is not None and rect is not None):
            faces.append(face)
            labels.append(0)

    return faces, labels

def display_name_over_frame(frame, name, x, y):
    cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)

def split_data_into_training_and_testing(examples, labels):
    # 80 training, 20 testing
    examples_training = examples[0: int(len(examples) * .8)]
    examples_testing = examples[int(len(examples) * .8): -1]

    labels_training = labels[0: int(len(examples) * .8)]
    labels_testing = labels[int(len(examples) * .8): -1]

    return examples_training, labels_training, examples_testing, labels_testing

def main():
    my_faces, my_labels = prepare_my_faces()
    other_faces, other_labels = prepare_other_faces()

    (positive_training_data, positive_training_labels,
    positive_testing_data, positive_testing_labels) = split_data_into_training_and_testing(my_faces, my_labels)

    (negative_training_data, negative_training_labels,
    negative_testing_data, negative_testing_labels) = split_data_into_training_and_testing(other_faces, other_labels)

    training_data = positive_training_data + negative_training_data
    training_labels = positive_training_labels + negative_training_labels

    testing_data = positive_testing_data + negative_testing_data
    testing_labels = positive_testing_labels + negative_testing_labels

    face_recognizer.train(training_data, np.array(training_labels))
    face_recognizer.save("trained_face_recognizer.xml")

    mean_confidence_distance_for_me = 0
    num_me = 0
    for data in testing_data:
        label, confidence_distance = face_recognizer.predict(data)
        if (label == 1):
            mean_confidence_distance_for_me += confidence_distance
            num_me += 1
    mean_confidence_distance_for_me = mean_confidence_distance_for_me / num_me
    print("The mean confidence score for my face: " + str(mean_confidence_distance_for_me))


if __name__ == "__main__":
    main()
