#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.compat.v1 import disable_eager_execution
import numpy as np
import asyncio
from telethon import TelegramClient, events
import time


# In[2]:


api_id = 'your_api_id'
api_hash = 'your_api'


# In[3]:


disable_eager_execution()


# In[4]:


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)


# In[5]:


#loading the saved model
model = load_model('AliceModel.h5')#, compile = False)


# In[6]:


def predict_class(model):  
    #taking photo with cv2
    camera = cv2.VideoCapture(0)
    return_value, image = camera.read()
    cv2.imwrite('opencv.png', image)
    del(camera)
    
    image = load_img('opencv.png', target_size=(480,640))
    #transforming to array and then to batch
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])
    input_arr = input_arr.astype('float32') / 255
    
    #predicting and returning the class
    prediction = model.predict(input_arr, verbose=0)
    #taking the maximum argument from array
    predicted_class = np.argmax(prediction, axis=-1)
    
    #return '0' if Nobody, '1' if Irina, '2' if Nikita
    return predicted_class[0]


# In[7]:

async def Alice():
    #time.sleep(5)
    n = 0
    async with TelegramClient('session_name', api_id, api_hash) as client:
        while n<10:
            if predict_class(model) == 2:
                n = 11
                await client.send_message('alice_speaker_bot', '/say Привет, Никита')
                
            elif predict_class(model) == 1:
                n = 11
                await client.send_message('alice_speaker_bot', '/say Здравствуй, Ирина')
            else:
                time.sleep(10)
                n += 1

def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(Alice())
    loop.close()

if __name__ == '__main__':
    main()
# In[ ]:




