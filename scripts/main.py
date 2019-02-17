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

def find_face(model, get_crop=False, imgg=None, path=None, aff_en=False, skip_prep=False):

    if skip_prep == False:
        while True:
            '''Infinite loop to capture and get largest bounding box image'''
            if path == None:
                _, imgg = ic.img_capt(cam_id=0, bness=50)
                if not _:
                    print("Unable to capture image. Trying again...")
                    continue
                cv2.imwrite(img_wpath + 'captured/c1.jpg', imgg)
            else:
                imgg = cv2.imread(path)

            _, imgg = ic.face_crop(imgg, affine=aff_en)
            if not _:
                if not get_crop:
                    print("No face found in image.")
                    return
                print("No face found in frame. Trying again...")
                continue
            break

        if get_crop:
            return imgg

    embd = facenet.forward(model, imgg).reshape(1,128)
    if embd.any() == None:
        print("Error in embeddings")
        return None
    #print("Embeddings: {0}".format(embd))

    face_dict = joblib.load(dict_path)
    mind = 100
    index = -1
    print("Searching face in codebook of {0} faces".format(len(face_dict[0])))
    for i in range(len(face_dict[0])):
        dis = pp.relative_dist(face_dict[1][i], embd)
        if dis < mind:
            mind, _index = dis, i

    #if mind <= 1:
        print("Person: {0}  Distance: {1}".format(face_dict[0][i], str(dis)))
        cv2.imwrite(img_wpath + 'test/' +  face_dict[0][_index] + '.jpg', imgg)
    #else:
        #print("No person found. Minimum distance = {0}".format(mind))
        #cv2.imwrite(img_wpath + 'test/none.jpg', imgg)


if __name__ == '__main__':
    while True:

        cc = int(input("Remember codebook 0\nForget codebook 1  \n"))
        if cc == 1:
            ll = [[], []]
            joblib.dump(ll, dict_path)
            del ll

        print("----------------Face recognition using Facenet CNN-----------------------")
        print("Press 0 to disable affine alignment\nPress 1 to enable affine based aligning\n")
        aff_al = int(input('Enter 0 or 1: '))
        aff_en = bool()
        if aff_al == 1:
            aff_en = True
        else:
            aff_al = False
        print("--------------------------Legend-----------------------------------------")
        print("Press 1 for recognizing a face")
        print("Press 2 for adding new face to the codebook")
        print("Press 0 to exit program")

        ch = int(input())

        model = facenet.load_model()

        if ch == 0:
            print("----------------------------Exiting---------------------------------");
            break
        elif ch == 1:
            while True:

                print("------------Face recognition mode----------")

                print("Press 1 to provide stored unprocessed image as input")
                print("Press 2 to enter path of folder having images(uncropped)")
                print("Press 3 for stored cropped image for forward pass")
                print("Press 4 to enter path to folder for forward pass")
                print("Press 5 to capture image. Then wait for output......")
                print("Press 0 to exit face recognition mode")

                itt = int(input())

                if itt == 0:
                    print("Exiting Face recognition mode")
                    break
                elif itt == 1:
                    pth = str(input('Enter full path to image:  '))
                    find_face(model, path=pth, aff_en=aff_en)
                elif itt == 2:
                    pth = str(input("Enter full path to folder: "))
                    for _ in os.listdir(pth):
                        find_face(model, path=pth + '/' + _, aff_en=aff_en)
                elif itt == 3:
                    pth = str(input('Enter full path to image:  '))
                    find_face(model, imgg=cv2.imread(pth), aff_en=aff_en, skip_prep=True)
                elif itt == 5:
                    find_face(model)
                else:
                    print("Invalid input. Try again")

        elif ch == 2:
            print("---------API for adding new face to dictionary---------")
            pname = str(input("Enter name of new person:  "))
            f_dict = joblib.load(dict_path)
            f_dict[0].append(pname)
            print("Press 1 for capturing real time images for training")
            print("Press 2 for providing path for training images")

            in1 = int(input())
            if in1 == 2:
                pth = str(input('Enter path to folder: '))
                embd_l = []
                for fl1 in os.listdir(pth):
                    im1 = cv2.imread(pth + fl1).reshape(96, 96, 3)
                    embd_l.append(facenet.forward(model, im1).reshape(1,128))
            elif in1 == 1:
                print("Enter 10 images of the same person. Click 1 to capture image.")
                embd_l = []
                for i in range(MAX_TRAIN_IMGS):
                    print("Press 1 to capture {0}th image".format(i+1))
                    if int(input()) == 1:
                        imag = find_face(model, get_crop=True, aff_en=aff_en)
                        cv2.imwrite(img_wpath + 'train/' + pname + '.jpg', imag)
                        embd_l.append(facenet.forward(model, imag.reshape(96,96,3)).reshape(1,128))

            f_dict[1].append(pp.dis_avg(embd_l))
            joblib.dump(f_dict, dict_path)

            print("New face added for {0}".format(pname))
            print("Average embedding {0}".format(f_dict[1][-1]))

        else:
            print("Invalid option")
