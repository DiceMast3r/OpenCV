import cv2.cv2 as cv2
import os
from imutils import paths
import numpy as np

path_out = "out"
cas = cv2.CascadeClassifier('cascade\lbpcascade_animeface.xml')
scale = 50
imagePath = list(paths.list_images('Img'))
for (i, imagePath) in enumerate(imagePath):
    img = cv2.imread(imagePath)
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    faces = cas.detectMultiScale(gray)
    for x, y, w, h in faces:
        cv2.rectangle(resized, (x, y), (x + w, y + h), (51, 87, 255), 2)
    cv2.imwrite(os.path.join(path_out, "{0}.jpg".format(i)), resized)
    # cv2.imshow('face', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

