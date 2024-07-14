import numpy as np
from PIL import  Image
import os, cv2


def train_classifer(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

   
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("C:\\Users\\naiks\\OneDrive\\Desktop\\HaarCascade\\classifier.yml")


train_classifer("C:\\Users\\naiks\\OneDrive\\Desktop\\HaarCascade\\data")