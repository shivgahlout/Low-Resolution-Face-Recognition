
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os


# In[2]:


json_file = open('final_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("final_model_weights.h5")
print("Loaded model from disk")


# In[12]:


from keras.preprocessing.image import ImageDataGenerator

batch_size = 16


val_datagen = ImageDataGenerator(rescale=1./255)

test_generator = val_datagen.flow_from_directory(
        './lr_3/test',
        target_size=(224, 224),
        batch_size=1,
        shuffle=False,
        class_mode='categorical')


# In[13]:


from keras.optimizers import Adam
adam=Adam(lr=.000001)
model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])

a=model.evaluate_generator( test_generator, len(test_generator.filenames))


# In[14]:


a

