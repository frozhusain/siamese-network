# -*- coding: utf-8 -*-
"""real siamese.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b5JMWtOKlrqIaxiyp74lT7_GIT_LjRmJ
"""

#import tensorflow.contrib.eager as tfe
import tensorflow as tf
from keras.preprocessing.image import load_img ,img_to_array
from keras.applications.resnet50 import preprocess_input, decode_predictions
import keras.preprocessing.image
import PIL
from PIL import Image
from keras.models import Model
from keras.layers import Dense, Input, subtract, concatenate, Lambda, add, maximum
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras.models import load_model, model_from_json
import numpy as np

def triplet_loss(inputs, dist='euclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, 2 + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
        
    returned_loss = K.mean(loss)
    return returned_loss
  
def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

from google.colab import drive
drive.mount('/content/drive')

import cv2
import os
def get_len_folder(dir):
  c = 0
  for i in os.listdir(dir):
    c = c+1
  return c

import cv2
from os import listdir
from random import randrange

dir = "drive/My Drive/images"
len_of_p = {}

for i in listdir(dir):
  len_of_p[i] = get_len_folder(dir +"/" + i)
  

train = []
progress=0
curr_person = -1
for p in listdir(dir):
  curr_person = curr_person+1
  p_path = dir + "/" + p
  curr_len = get_len_folder(p_path)
  for i in range(30):
    an = randrange(0, curr_len, 1)
    pos = -1
    while(pos==-1 or pos==an):
      pos = randrange(0, curr_len, 1)
    an_name = listdir(p_path)[an]
    pos_name = listdir(p_path)[pos]
    an_dir = p_path +"/" + an_name
    pos_dir = p_path + "/" + pos_name
    
    n_person = -1
    dir_len=get_len_folder(dir)
    while(n_person == -1 or n_person == curr_person):
      n_person = randrange(0, dir_len, 1)
    n_per_dir = dir + "/" + listdir(dir)[n_person]
    n_len = len_of_p[ listdir(dir)[n_person] ]
    n_p_name = randrange(0, n_len, 1)
    neg_dir = n_per_dir + "/" + listdir(n_per_dir)[n_p_name]
   # print(an_dir,'\n',pos_dir,'\n',neg_dir,'\n\n')
    #an_img=cv2.imread(an_dir)
    an_img=load_img(an_dir,target_size=(224,224))
    an_img=img_to_array(an_img)
    pos_img=load_img(pos_dir,target_size=(224,224))
    pos_img=img_to_array(pos_img)
    #pos_img=cv2.resize(cv2.imread(pos_dir),(224,224))
    neg_img=load_img(neg_dir,target_size=(224,224))
    neg_img=img_to_array(neg_img)
   # neg_img=cv2.resize(cv2.imread(neg_dir),(224,224))
     
    temp=[]
    temp.append(an_img)
    temp.append(pos_img)
    temp.append(neg_img)
    temp=np.array(temp, dtype="float32")
    train.append(temp)
    print(progress, )
    progress = progress+1

train=np.array(train, dtype="float32")

#np.random.shuffle(train)

train_label = np.zeros((180,2048))

#np.save("drive/My Drive/siamese_data/traindata.npy",train)

train.shape

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Conv2D,BatchNormalization,Dense,Dropout,Flatten,MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.models import Model

base = ResNet50(weights='imagenet')
for layer in base.layers:
    layer.trainable = True


lay=base.layers[-6].output
lay=GlobalAveragePooling2D()(lay)

for layer in base.layers[:-16]:
    layer.trainable = False

'''''new_model = Model(inputs=base.input, outputs=x)

anchor_input = Input(shape=( 224, 224, 3), name='anchor_input')
pos_input = Input(shape=( 224, 224, 3), name='pos_input')
neg_input = Input(shape=( 224, 224, 3), name='neg_input')
encoding_anchor   = new_model(anchor_input)
encoding_pos      = new_model(pos_input)
encoding_neg      = new_model(neg_input)
loss = Lambda(triplet_loss2)([encoding_anchor, encoding_pos, encoding_neg])
siamese_network = Model(inputs  = [anchor_input, pos_input, neg_input],
                       outputs = loss) # Note that the output of the model is the
                                       # return value from the triplet_loss function above
siamese_network.compile(optimizer=Adam(lr=.00004, clipnorm=1.), loss=identity_loss)

'''def triplet_loss2(inputs, dist='euclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, 2 + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
        
    returned_loss = K.mean(loss)
    return returned_loss
  
def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

new_model = Model(inputs=base.input, outputs=lay)

new_model.summary()

anchor_input = Input(shape=(224, 224, 3), name='anchor_input')
pos_input = Input(shape=(224, 224, 3), name='pos_input')
neg_input = Input(shape=(224, 224, 3), name='neg_input')
encoding_anchor   = new_model(anchor_input)
encoding_pos      = new_model(pos_input)
encoding_neg      = new_model(neg_input)
loss = Lambda(triplet_loss)([encoding_anchor, encoding_pos, encoding_neg])
siamese_network = Model(inputs  = [anchor_input, pos_input, neg_input],
                       outputs = loss) # Note that the output of the model is the
                                       # return value from the triplet_loss function above
siamese_network.compile(optimizer=Adam(lr=.00004, clipnorm=1.), loss=identity_loss)

for i in range(10):
  for j in range(180):
    an_final = train[j][0]
    an_final=np.resize(an_final, (1,224, 224, 3))
    an_final=preprocess_input(an_final)
    pos_final = train[j][1]
    pos_final=np.resize(pos_final, (1,224, 224, 3))
    pos_final=preprocess_input(pos_final)
    neg_final = train[j][2]
    neg_final=np.resize(neg_final, (1,224, 224, 3))
    neg_final=preprocess_input(neg_final)
    label_final =  train_label[0]
    label_final = np.resize(label_final, (1, 2048))
    
    siamese_network.fit(x = [an_final,pos_final,neg_final ] ,y =label_final,epochs = 1)

new_model.save_weights("drive/My Drive/abcd_n.h5")

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input



new_model.load_weights("drive/My Drive/abcd.h5", by_name=False)

dir = "drive/My Drive/siame_test/im"
import cv2

dic_img={}
a=[]
for x in listdir(dir):
  path=dir+'/'+x
  img=load_img(path,target_size=(224,224))
  img=img_to_array(img)
  img=img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
  img=preprocess_input(img)
#img = K.variable(img)
  #img=cv2.resize(cv2.imread(path),(224,224))
  enco_img = new_model.predict(img)
  print(enco_img)
  print("\n\n")
  a.append(enco_img)

new_model.summary()

check=a[11]
for i in range(12):
  temp=a[i]
  su=np.sum(np.abs(check-temp))
  print(su,"\n")

a=np.array(a)
for x in a:
  positive_distance = K.square(x - a[6])
  positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
  print(positive_distance)

img=load_img('/content/sandey.JPG',target_size=(224,224))
img=img_to_array(img)
img=img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
img=preprocess_input(img)
#img = K.variable(img)
  #img=cv2.resize(cv2.imread(path),(224,224))
enco_img1 = new_model.predict(img)

img=load_img('/content/FROZ.jpg',target_size=(224,224))
img=img_to_array(img)
img=img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
img=preprocess_input(img)
#img = K.variable(img)
  #img=cv2.resize(cv2.imread(path),(224,224))
enco_img2 = new_model.predict(img)

img=load_img('/content/FROZ1.jpg',target_size=(224,224))
img=img_to_array(img)
img=img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
img=preprocess_input(img)
#img = K.variable(img)
  #img=cv2.resize(cv2.imread(path),(224,224))
enco_img3 = new_model.predict(img)

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

diff=np.sum(np.abs(enco_img1-enco_img2))
if(diff<100):
  print("same")
else:
  print("different") 
img=load_img('/content/FROZ.jpg',target_size=(224,224))
plt.imshow(img)
print("\n")

img=load_img('/content/sandey.JPG',target_size=(224,224))
plt.imshow(img)

diff=np.sum(np.abs(enco_img2-enco_img3))
if(diff<100):
  print("same")
else:
  print("different")  
img=load_img('/content/FROZ.jpg',target_size=(224,224))
plt.imshow(img)

img=load_img('/content/FROZ1.jpg',target_size=(224,224))
plt.imshow(img)

