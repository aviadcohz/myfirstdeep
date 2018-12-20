from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import keras
import pandas as pd
import pylab as plt
from tqdm import tqdm
from glob import glob


def image_to_keras_model(np_image):
    import numpy as np
    from skimage import transform
    np_image = np.array(np_image).astype('float32')
    np_image = transform.resize(np_image, (128, 128, 3))
    np_image = np.expand_dims(np_image, axis=0)
    np_image/= 255.
    return np_image


cnn_saved_model=r"C:\Users\lisrael1\Desktop\jupyter\computer_vision\cnn_saved_history/cnn_saved_model.h5"
cnn_saved_history=r"C:\Users\lisrael1\Desktop\jupyter\computer_vision\cnn_saved_history/cnn_saved_history.csv"

checkpoint=keras.callbacks.ModelCheckpoint(cnn_saved_model, monitor='val_loss',
                                           verbose=0, save_best_only=False,
                                           save_weights_only=False, mode='auto', period=1)
hist=keras.callbacks.CSVLogger(cnn_saved_history, separator=',', append=True)

full_path=r'C:/Users/lisrael1/Desktop/jupyter/computer_vision/'
pictures_input=full_path+'only_10_pictures_not_identical/'
pictures_input=full_path+'only_10_pictures/'
pictures_input=full_path+'pictures/'

train_datagen = ImageDataGenerator(rescale = None, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = None)
print('training db:')
training_set = train_datagen.flow_from_directory(pictures_input+'training', target_size = (128, 128), color_mode='rgb', batch_size = 32, class_mode = 'categorical')
print('test db:')
test_set = test_datagen.flow_from_directory(pictures_input+'test', target_size = (128, 128), batch_size = 32, class_mode = 'categorical')

if len(glob(cnn_saved_model)):
    print('trained model is saved, loading it')
    from keras.models import load_model
    conv_model = load_model(cnn_saved_model)

else:
    input_shape=[128,128,3]
    conv_model = Sequential()

    conv_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,strides=1))
    conv_model.add(Conv2D(32, (3, 3), activation='relu'))
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    conv_model.add(Conv2D(64, (3, 3), activation='relu',strides=1))
    conv_model.add(Conv2D(64, (3, 3), activation='relu'))
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    conv_model.add(Conv2D(64, (3, 3), activation='relu'))
    conv_model.add(MaxPooling2D(pool_size=(1, 3)))
    # model.add(Dropout(0.25))

    conv_model.add(Conv2D(64, (1, 2), activation='relu'))
    conv_model.add(MaxPooling2D(pool_size=(2, 1)))
    # model.add(Dropout(0.25))

    conv_model.add(Flatten())
    conv_model.add(Dense(256, activation='relu'))
    conv_model.add(Dropout(0.5))
    conv_model.add(Dense(2, activation='softmax'))

    opt = keras.optimizers.Adam(lr=.0001)
    conv_model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])


conv_model.fit_generator(training_set, nb_epoch=50, validation_data=test_set, callbacks=[checkpoint, hist])
# conv_model.predict_proba(image_to_keras_model(imread(r"C:\Users\lisrael1\Desktop\jupyter\computer_vision\pictures\training\bottle_caps\187464394_5f1495594b_b.jpg")))
# conv_model.predict_generator(test_set).round(2)
# int(conv_model.predict_proba(image_to_keras_model(cropped)).tolist()[0][0]*100)

df=pd.read_csv(cnn_saved_history, index_col=None)
df[['acc', 'val_acc']].plot()
plt.show()
print('hi')
