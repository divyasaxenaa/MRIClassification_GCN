import pandas as pd
from os import walk
from sklearn.model_selection import train_test_split
from preprocess_data import *
import numpy as np
import os

path = os.getcwd()


def preprocessing_testing():
    rawimages = path+'/CSE6389_project2/CSE6389_project2/Training/Health_raw'
    rawimages_pat =  path+'/CSE6389_project2/CSE6389_project2/Training/Patient_raw'
    rawimages_test =  path+'/CSE6389_project2/CSE6389_project2/Testing/Health_raw'
    rawimages_pat_test =  path+'/CSE6389_project2/CSE6389_project2/Testing/Patient_raw'
    filtered =  path+"/CSE6389_project2/CSE6389_project2/Training/Health_filtered"
    filtered_pat =  path+"/CSE6389_project2/CSE6389_project2/Training/Patient_filtered"
    filtered_test =  path+"/CSE6389_project2/CSE6389_project2/Testing/Health_filtered"
    filtered_pat_test =  path+"/CSE6389_project2/CSE6389_project2/Testing/Patient_filtered"
    Img_filter(rawimages, filtered)
    Img_filter(rawimages_pat, filtered_pat)
    Img_filter(rawimages_test, filtered_test)
    Img_filter(rawimages_pat_test, filtered_pat_test)
    processed =  path+"/CSE6389_project2/CSE6389_project2/Training/processed"
    processed_test =  path+"/CSE6389_project2/CSE6389_project2/Testing/processed"
    counter = 0
    for root, dirs, files in walk(filtered):
        for name in files:
            try:
                print(name)
                file_path = root + "/" + name
                img = np.load(file_path)
                processed_img = final_preprocessing(img)
                counter += 1
                new_name = name.replace(".npy", "")
                print("1")
                np.save(processed + '/' + new_name + "_processed", processed_img)
                print("2")
            except ValueError:
                print("Oops!  That was no valid number.  Try again...",name)

    for root, dirs, files in walk(filtered_pat):
        print(filtered_pat)
        for name in files:
            try:
                print(name)
                file_path = root + "/" + name
                img = np.load(file_path)
                processed_img = final_preprocessing(img)
                counter += 1
                new_name = name.replace(".npy", "")
                np.save(processed + '/' + new_name + "_pat_processed", processed_img)
            except ValueError:
                print("Oops!  That was no valid number.  Try again...",name)

    for root, dirs, files in walk(filtered_test):
        for name in files:
            try:
                file_path = root + "/" + name
                img = np.load(file_path)
                processed_img = final_preprocessing(img)
                counter += 1
                new_name = name.replace(".npy", "")
                np.save(processed_test + '/' + new_name + "_processed", processed_img)
            except ValueError:
                print("Oops!  That was no valid number.  Try again...",name)

    for root, dirs, files in walk(filtered_pat_test):
        for name in files:
            try:
                file_path = root + "/" + name
                img = np.load(file_path)
                processed_img = final_preprocessing(img)
                counter += 1
                new_name = name.replace(".npy", "")
                np.save(processed_test + '/' + new_name + "_pat_processed", processed_img)
            except ValueError:
                print("Oops!  That was no valid number.  Try again...",name)

    labels = pd.read_csv( path+'/CSE6389_project2/CSE6389_project2/Training/sub_labels.csv')
    labels_test = pd.read_csv( path+'/CSE6389_project2/CSE6389_project2/Testing/sub_labels.csv')
    uq_ids = set(labels['Subject'])
    uq_ids_test = set(labels_test['Subject'])
    # define a dictionary to store subject_id(keys) and class labels(values)
    sub_labels = dict()
    sub_labels_test = dict()
    for id in uq_ids:
        if id not in sub_labels.keys():
            label = ''.join(np.unique(labels['Group'][labels['Subject'] == id]))
            sub_labels[id] = label

    for id in uq_ids_test:
        if id not in sub_labels_test.keys():
            label_test = ''.join(np.unique(labels_test['Group'][labels_test['Subject'] == id]))
            sub_labels_test[id] = label_test
    # check the ID-label is correct
    print("sub_labels", sub_labels)

    labels_img = label_assign(processed, sub_labels)
    labels_img_test = label_assign(processed_test, sub_labels_test)

    # save the processed images in a list, data
    data = []
    for root, dirs, files in walk(processed):
        for name in files:
            file_path = root + "/" + name
            img = np.load(file_path)
            data.append(img)

    data_test = []
    for root, dirs, files in walk(processed_test):
        for name in files:
            file_path = root + "/" + name
            img_test = np.load(file_path)
            data_test.append(img_test)

    # split all images randomly into training/validation
    train_rs, val_rs, train_y_rs, val_y_rs = train_test_split(data, labels_img, stratify = labels_img, test_size = 0.16, random_state = 87)
    create_npy =  path+"/CSE6389_project2/CSE6389_project2/input/"
    np.save(create_npy + "train_data", np.asarray(train_rs))
    np.save(create_npy + "train_label", np.asarray(train_y_rs))
    np.save(create_npy + "val_data", np.asarray(val_rs))
    np.save(create_npy + "val_label", np.asarray(val_y_rs))
    np.save(create_npy + "test_data", np.asarray(data_test))
    np.save(create_npy + "test_label", np.asarray(labels_img_test))


if __name__ == "__main__":
    preprocessing_testing()







