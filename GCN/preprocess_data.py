from skimage.transform import resize
from os import listdir
import numpy as np


def Img_filter(path, des):
    files = listdir(path)
    counter = 0
    print(path)
    for name in files:
        fpath = path + "/" + name
        print(name)
        try:
            img = np.asarray(np.loadtxt(fpath))
            s = img.shape
            print(s)
            if s[0] == s[1]:
                name = name.replace(".txt", "")
                newpath = des + "/" + name
                np.save(newpath,img)
            else: 
                counter += 1

        except: 
            counter += 1
    print(str(counter) + " images not meeting the selection criteria/failed to load")


# resize the images to (28,28)
def resize_28_28(image, ideal_shape = (28, 28)):
    img_new2 = resize(image, ideal_shape, anti_aliasing=True)
    return img_new2


# create a preprocessing procedure
def final_preprocessing(img):
    img_std = resize_28_28(img)
    return img_std


# assign labels to the images in a folder
def label_assign(path, ref):
    files = listdir(path)
    label_assign = []

    for names in files:
        sub_id = "".join(names)
        if sub_id.endswith("_pat_processed.npy"):
            new_lab = 1
        else:
            new_lab = 0
        label_assign.append(new_lab)
    print(str(len(label_assign)) + " labels have been assigned to the data")
    return label_assign


