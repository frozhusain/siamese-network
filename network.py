
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
import os

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
    
    
new_model = Model(inputs=base.input, outputs=lay)

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



from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
new_model.load_weights("abcd_n.h5", by_name=False)

dir = r"test"
import cv2

dic_img={}
a=[]
b=[]
for x in os.listdir(dir):
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
  b.append(x)  

import numpy as np
import cv2
import math

# Open Camera
capture = cv2.VideoCapture(0)

while capture.isOpened():


    ret, frame = capture.read(-1)

    cv2.rectangle(frame, (100, 100), (324, 324), (0, 255,0), 0)
    crop_image = frame[100:324, 100:324]
    #img_gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gesture", frame)
  #  cv2.imshow("gray crop",img_gray)
    cv2.imshow("crop image ", crop_image)
    #img=load_img(path,target_size=(224,224))
    img=img_to_array(crop_image)
    img=img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
    img=preprocess_input(img)
    check=new_model.predict(img)
    
    for i in range(7):
        temp=a[i]
        su=np.sum(np.abs(check-temp))
        if(su<100):
            print(b[i],"\n")
    if cv2.waitKey(1) == ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()


