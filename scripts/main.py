import numpy as np
import os
import cv2
from sklearn.externals import joblib
import pre_proc as ic
import facenet
import post_proc as pp

MAX_TRAIN_IMGS = 10
img_wpath = '/home/prathamesh/undergrad/btech_proj/misc/openface/pc_demo/images/'
dict_path = '/home/prathamesh/undergrad/btech_proj/misc/openface/pc_demo/face_dict.sav'

def find_face(model, get_crop=False):
    imgg = ic.img_capt(bness=250)
    if imgg.any() == None:
        print("Unable to capture image1. Try again")
        return None
    cv2.imwrite(img_wpath + 'captured/c1.jpg', imgg)

    imgg = ic.face_crop(imgg)
    if imgg == None:
        print("No face found. Try again")
        return None

    if get_crop:
        return imgg

    embd = facenet.forward(model, imgg).reshape(1,128)
    if embd.any() == None:
        print("Error in embeddings")
        return None
    print("Embeddings: {0}".format(embd))

    face_dict = joblib.load(dict_path)
    mind = 100
    index = -1
    for i in range(len(face_dict[0])):
        dis = pp.relative_dist(face_dict[1], embd)
        if dis < mind:
            mind, _index = dis, i

    if mind <= 1:
        print("Person identified: " + face_dict[0][_index])
        cv2.imwrite(img_wpath + 'test/' +  face_dict[0][_index] + '.jpg', imgg)
    else:
        print("No person found. Minimum distance = {0}. Embedding is {1}".format(mind, embd))
        cv2.imwrite(img_wpath + 'test/none.jpg', imgg)


if __name__ == '__main__':
    while True:

        cc = int(input("remember cache 0\nForget cache 1"))
        if cc == 1:
            ll = [[], []]
            joblib.dump(ll, dict_path)
            del ll

        print("----------------Face recognition using Facenet CNN-----------------------")
        print("--------------Legend---------------")
        print("Press 1 for entering recognizing a face")
        print("Press 2 for adding new face to the directory")
        print("Press 0 to exit program")

        ch = int(input())

        model = facenet.load_model()

        if ch == 0:
            print("----------Exiting-----------");
            break
        elif ch == 1:
            while True:

                print("------------Face recognition mode----------")
                print("Press 0 to exit")
                print("Press 5 to capture image. Then wait for output......")

                itt = int(input())

                if itt == 0:
                    print("Exiting Face recognition mode")
                    break
                elif itt == 5:
                    find_face(model)
                else:
                    print("Invalid input. Try again")

        elif ch == 2:
            print("---------API for adding new face to dictionary---------")
            pname = str(input("Enter name of new person: "))
            f_dict = joblib.load(dict_path)
            f_dict[0].append(pname)
            print("Enter 10 images of the same person. Click 1 to capture image.")
            embd_l = []
            for i in range(MAX_TRAIN_IMGS):
                print("Press 1 to capture {0}th image".format(i+1))
                if int(input()) == 1:
                    imag = find_face(model, get_crop=True)
                    if type(imag) == type(None):
                        print("Try again")
                        continue
                    cv2.imwrite(img_wpath + 'train/' + pname + '.jpg', imag)
                    embd_l.append(facenet.forward(model, imag.reshape(96,96,3)).reshape(1,128))

            f_dict[1].append(pp.dis_avg(embd_l))
            joblib.dump(f_dict, dict_path)

            print("New face added for {0}".format(pname))
            print("average embedding {0}".format(f_dict[1][-1]))

        else:
            print("Invalid option")
