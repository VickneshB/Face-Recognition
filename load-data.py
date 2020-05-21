import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_faces(path, num_person, num_face):
    faces = np.zeros((num_person*num_face, 160*160*3), dtype=np.uint8)

    for i in range(num_person):
        for j in range(num_face):
            img = cv2.imread(path + str(i + 1) + '/' + str(j + 1) + '.jpg')
            reimg = cv2.resize(img[:,80:559,:], (160,160,3))
            faces[i * num_face + j, :] = reimg.ravel()

    return faces


if __name__ == "__main__":
    path = './face_dataset_1/s'
    person_count = 200
    face_count = 14
    print("There are", person_count, "subjects in this dataset, each has", face_count, "photos")
    faces = read_faces(path,person_count,face_count)
    np.save('facedata2',faces)
    # plot several images
#    fig = plt.figure()
    for i in range(28):
        cv2.imshow('',faces[i].reshape(160, 160, 3))
        cv2.waitKey(0)
#        ax = fig.add_subplot(4, 7, i + 1, xticks=[], yticks=[])
#        ax.imshow(faces[i].reshape(160, 160, 3))
#    plt.show()
