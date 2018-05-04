
# coding: utf-8

# In[1]:


import numpy as np
import h5py
from keras.models import load_model


# In[2]:


import scipy.io
x = scipy.io.loadmat('train_images.mat')
x.keys()


# In[3]:


x_hr=x[x.keys()[1]]
x_lr=x[x.keys()[3]]


# In[4]:


import matplotlib.pyplot as plt

img=x_hr[0]
plt.imshow(img)
plt.show()
img=x_lr[0]
plt.imshow(img)
plt.show()


# In[5]:


y = scipy.io.loadmat('test_images.mat')
y.keys()


# In[6]:


y_hr=y[y.keys()[2]]
y_lr=y[y.keys()[0]]


# In[7]:


img=y_hr[0]
plt.imshow(img)
plt.show()
img=y_lr[0]
plt.imshow(img)
plt.show()


# In[8]:


z = scipy.io.loadmat('val_images.mat')
z.keys()


# In[9]:


z_hr=z[z.keys()[3]]
z_lr=z[z.keys()[0]]


# In[10]:


img=z_hr[0]
plt.imshow(img)
plt.show()
img=z_lr[0]
plt.imshow(img)
plt.show()


# In[11]:


hr_images=np.vstack([x_hr,y_hr,z_hr])


# In[12]:


lr_images=np.vstack([x_lr,y_lr,z_lr])


# In[13]:


hr_images=hr_images/255.0
lr_images=lr_images/255.0


# In[14]:


# hr_images=np.transpose(hr_images, [0,3,1,2])
# lr_images=np.transpose(lr_images, [0,3,1,2])


# In[15]:


img=hr_images[0]
plt.imshow(img)
plt.show()
img=lr_images[0]
plt.imshow(img)
plt.show()


# In[16]:


from __future__ import print_function
import numpy as np
from keras.layers import Input, Convolution2D, merge
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import h5py
import math


# In[17]:


# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# set_session(tf.Session(config=config))


# In[18]:


K.image_dim_ordering='th'


# In[19]:


def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return 10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)


# In[20]:


def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.1
	epochs_drop = 30
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


# In[21]:


batch_size = 64
nb_epoch = 50
#input imaage dimensions
img_rows, img_cols = 224, 224
out_rows, out_cols = 224, 224
#filter number
n1 = 64
n2 = 32
n3 = 1
#filter size
f1 = 9
f2 = 1
f3 = 5


# In[22]:


input_shape = (img_rows,img_cols,3)


# In[23]:


# print('in_train shape:', in_train.shape)
# print(in_train.shape[0], 'train samples')
# print(in_test.shape[0], 'test samples')
# #SR Model
#input tensor for a 1_channel image region
x = Input(shape = input_shape)
c1 = Convolution2D(n1, f1,f1, activation = 'relu', init = 'he_normal', border_mode='same')(x)
c2 = Convolution2D(n2, f2, f2, activation = 'relu', init = 'he_normal', border_mode='same')(c1)
c3 = Convolution2D(n2, f2, f2, activation = 'relu', init = 'he_normal', border_mode='same')(c2)
c4 = Convolution2D(3, f3, f3, init = 'he_normal', border_mode='same')(c3)
model = Model(input = x, output = c4)

model.summary()
##compile
# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
adam = Adam(lr=0.001) 
model.compile(loss='mse', metrics=[PSNRLoss], optimizer=adam)     
# learning schedule callback
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
history = model.fit(hr_images, lr_images, batch_size=batch_size, nb_epoch=nb_epoch, callbacks = [lrate],
          verbose=1, validation_split=.1, shuffle=True)            
print(history.history.keys())
#save model and weights
json_string = model.to_json()  
open('srcnn_model.json','w').write(json_string)  
model.save_weights('srcnn_model_weights.h5') 
# summarize history for loss




# In[24]:


model.save('srcnn_model.h5') 
plt.plot(history.history['PSNRLoss'])
plt.plot(history.history['val_PSNRLoss'])
plt.title('model loss')
plt.ylabel('PSNR/dB')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()


# In[25]:


# x_lr[1]=x_lr[1].resize((1,224,224,3) )
y = x_lr[2][np.newaxis,:,:,:]
y.shape


# In[26]:


img=model.predict(y)


# In[27]:


# img[0]=img[0].astype('uint8')


# In[28]:


img[0]=img[0]/255


# In[29]:


from PIL import Image
# img1=Image.fromarray(img[0])



plt.imshow(x_hr[2])
plt.show()


plt.imshow(img[0])
plt.show()


plt.imshow(x_lr[2])
plt.show()


# In[6]:


model2 = load_model('srcnn_model.h5')

