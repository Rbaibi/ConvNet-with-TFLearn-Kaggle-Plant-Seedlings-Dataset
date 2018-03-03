#import modules
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, max_pool_2d
import os
import cv2
from random import shuffle
import numpy as np


#put the location of train and test data here
train_image_loc= r'C:\Users\Gebruiker\Documents\Github\3 CNN with TFLearn on Kaggle Plant Seedlings Dataset\train'# + mapname
test_image_loc = r'C:\Users\Gebruiker\Documents\Github\3 CNN with TFLearn on Kaggle Plant Seedlings Dataset\test'

#parameters
image_size = 50
LR = 0.001

#model name
MODEL_NAME = 'Plant_Seedlings_Classifier'

#number of channels
NUM_Channels = 3




#Data Preprocessing
#Turn data into a one-hot array
def one_hot(label):
    if label == 'Black-grass':
        return [1,0,0,0,0,0,0,0,0,0,0,0]
    elif label == 'Charlock':
        return [0,1,0,0,0,0,0,0,0,0,0,0]
    elif label == 'Cleavers':
        return [0,0,1,0,0,0,0,0,0,0,0,0]
    elif label == 'Common Chickweed':
        return [0,0,0,1,0,0,0,0,0,0,0,0]
    elif label == 'Common wheat':
        return [0,0,0,0,1,0,0,0,0,0,0,0]
    elif label == 'Fat Hen':
        return [0,0,0,0,0,1,0,0,0,0,0,0]
    elif label == 'Loose Silky-bent':
        return [0,0,0,0,0,0,1,0,0,0,0,0]
    elif label == 'Maize':
        return [0,0,0,0,0,0,0,1,0,0,0,0]
    elif label == 'Scentless Mayweed':
        return [0,0,0,0,0,0,0,0,1,0,0,0]
    elif label == 'Shepherds Purse':
        return [0,0,0,0,0,0,0,0,0,1,0,0]
    elif label == 'Small-flowered Cranesbill':
        return [0,0,0,0,0,0,0,0,0,0,1,0]
    elif label == 'Sugar beet':
        return [0,0,0,0,0,0,0,0,0,0,0,1]

#prepare the training data
def prep_train():
    train_data = []
    nr = 1
    folders = ['Black-grass', 'Charlock', 'Cleavers'
               ,'Common Chickweed','Common wheat','Fat Hen'
               ,'Loose Silky-bent','Maize','Scentless Mayweed'
               ,'Shepherds Purse','Small-flowered Cranesbill','Sugar beet']
    for folder in folders:
        specific_loc= train_image_loc + '\\' + folder
        for image in os.listdir(specific_loc):
            label = one_hot(folder) # fix this, needs to be 1 hot encoded
            path = os.path.join(specific_loc,image)
            img = cv2.imread(path,cv2.IMREAD_COLOR)
            img = cv2.resize(img, (image_size,image_size))
            train_data.append([np.array(img),np.array(label)])
            print('Preparing training image ',nr)
            nr += 1
        
    shuffle(train_data)
    np.save('train_data.npy', train_data)
    return train_data

#prepare the testing data
def prep_test():
    test_data = []
    img_num = 0
    nr = 1
    for image in os.listdir(test_image_loc):
        path = os.path.join(test_image_loc,image)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (image_size,image_size))
        test_data.append([np.array(img), img_num])
        print('Preparing testing image ',nr)
        img_num += 1
        nr += 1
        
    shuffle(test_data)
    np.save('test_data.npy', test_data)
    return test_data





#prepare the dataset for training
train_data = prep_train()
test_data = prep_test()



#Convolutional Neural Network
cnn = input_data(shape=[None, image_size, image_size, NUM_Channels], name='input')
cnn = conv_2d(cnn, 40, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = conv_2d(cnn, 80, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = conv_2d(cnn, 160, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = conv_2d(cnn, 80, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = conv_2d(cnn, 40, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = fully_connected(cnn, 1024, activation='relu')
cnn = dropout(cnn, 0.9)
cnn = fully_connected(cnn, 12, activation='softmax')
cnn = regression(cnn, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(cnn, tensorboard_dir='log')


train = train_data[:-600]
test = train_data[-600:]


X = []
for image in train:
    X.append(image[0])

Y = []
for image in train:
    Y.append(image[1])

test_x = []
for image in test:
    test_x.append(image[0])

test_y = []
for image in test:
    test_y.append(image[1])

X      = np.array(X).reshape(-1,image_size,image_size,NUM_Channels)
test_x = np.array(test_x).reshape(-1,image_size,image_size,NUM_Channels)


#Fit the model
model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

#save the model
model.save(MODEL_NAME)



















