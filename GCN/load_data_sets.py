import numpy as np
import os

path = os.getcwd()

def load_data():
  x_train = np.load( path+"/CSE6389_project2/CSE6389_project2/input/train_data.npy",
                       allow_pickle=True)
  y_train = np.load(
       path+"/CSE6389_project2/CSE6389_project2/input/train_label.npy",
      allow_pickle=True)
  x_test = np.load( path+"/CSE6389_project2/CSE6389_project2/input/test_data.npy",
                      allow_pickle=True)
  y_test = np.load( path+"/CSE6389_project2/CSE6389_project2/input/test_label.npy",
                           allow_pickle=True)
  x_val= np.load( path+"/CSE6389_project2/CSE6389_project2/input/val_data.npy",
                   allow_pickle=True)
  y_val = np.load( path+"/CSE6389_project2/CSE6389_project2/input/val_label.npy",
                   allow_pickle=True)

  return (x_train, y_train), (x_test, y_test), (x_val, y_val)


