
# coding: utf-8

# In[1]:


# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os


# In[2]:


json_file = open('srcnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
srcnn_model = model_from_json(loaded_model_json)
srcnn_model = model_from_json(loaded_model_json)
# load weights into new model
srcnn_model.load_weights("srcnn_model_weights.h5")
print("Loaded model from disk")


# In[3]:


srcnn_model.summary()


# In[4]:


json_file = open('custom_vgg_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
fcnn_model = model_from_json(loaded_model_json)
fcnn_model = model_from_json(loaded_model_json)
# load weights into new model
fcnn_model.load_weights("custom_vgg_model_weights.h5")
print("Loaded model from disk")


# In[5]:


fcnn_model.summary()


# In[6]:


x=fcnn_model.get_layer('fc8').output


# In[7]:


from keras.models import Model
model_base=fcnn_model


# In[8]:


from keras.layers import Input
input_shape = (224,224,3)
input1 = Input(shape=input_shape)
pre1=srcnn_model.output
base_out = model_base(pre1)


# In[9]:


model = Model(srcnn_model.input, base_out)


# In[10]:


model.summary()


# In[11]:


from keras.preprocessing.image import ImageDataGenerator

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        './lr_3/train',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        './lr_3/val',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = val_datagen.flow_from_directory(
        './lr_3/test',
        target_size=(224, 224),
        batch_size=1,
        shuffle=False,
        class_mode='categorical')


# In[12]:


from keras.optimizers import Adam
adam=Adam(lr=.000001)
model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
# a=model.evaluate_generator( test_generator, len(test_generator.filenames))


# In[13]:


model.fit_generator(
        train_generator,
        steps_per_epoch=6400// batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=6400 // batch_size)


# In[ ]:


test_generator = val_datagen.flow_from_directory(
        './lr_3/test',
        target_size=(224, 224),
        batch_size=1,
        shuffle=False,
        class_mode='categorical')


# In[22]:


a=model.evaluate_generator( test_generator, len(test_generator.filenames))


# In[23]:


a


# In[24]:


json_string = model.to_json()  
open('final_model.json','w').write(json_string)  
model.save_weights('final_model_weights.h5') 

