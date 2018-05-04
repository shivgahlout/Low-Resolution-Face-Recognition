
# coding: utf-8

# In[1]:


from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam


# In[2]:


# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# set_session(tf.Session(config=config))


# In[3]:


nb_class = 159
hidden_dim = 512

vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))

vgg_model.summary()


# In[4]:


# last_layer = vgg_model.get_layer('pool5').output
# x = Flatten(name='flatten')(last_layer)
# # x = Dense(hidden_dim, activation='relu', name='fc6')(x)
# x = Dense(hidden_dim, activation='relu', name='fc7')(x)
# out = Dense(nb_class, activation='softmax', name='fc8')(x)
# custom_vgg_model = Model(vgg_model.input, out)


# In[5]:


last_layer = vgg_model.get_layer('fc7').output
# x = Flatten(name='flatten')(last_layer)
# x = Dense(hidden_dim, activation='relu', name='fc6')(x)
# x = Dense(nb_class, activation='relu', name='fc7')(x)
out = Dense(nb_class, activation='softmax', name='fc8')(last_layer)
custom_vgg_model = Model(vgg_model.input, out)


# In[6]:


custom_vgg_model.summary()


# In[7]:


datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
        horizontal_flip=True)
#         fill_mode='nearest')


# In[8]:


batch_size = 32

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
        'train',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'val',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = val_datagen.flow_from_directory(
        'test',
        target_size=(224, 224),
        batch_size=1,
        shuffle=False,
        class_mode='categorical')


# In[9]:


adam=Adam(lr=.0001)
custom_vgg_model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])


# In[10]:


custom_vgg_model.fit_generator(
        train_generator,
        steps_per_epoch=6400// batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=6400 // batch_size)


# In[11]:


json_string = custom_vgg_model.to_json()  
open('custom_vgg_model.json','w').write(json_string)  
custom_vgg_model.save_weights('custom_vgg_model_weights.h5') 
a=custom_vgg_model.evaluate_generator( test_generator, len(test_generator.filenames))


# In[12]:


a


# In[13]:


custom_vgg_model.save('fcnn_model.h5') 


# In[140]:


image=(next(test_generator)[0])
label=np.argmax((next(test_generator)[1]))


# In[141]:


a=custom_vgg_model.predict( image, 1)


# In[142]:


np.argmax(a)


# In[153]:


a=test_generator.class_indices


# In[158]:


a['Abdullah_Gul']

