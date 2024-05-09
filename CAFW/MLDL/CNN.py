import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

def CNN_model_train(data_train, data_test, data_train_label, data_test_label,
              input_shape = (35, 35, 3, 1), 
              num_classes = 7,
              epochs = 10,
              batch_size = 16):
    model = Sequential([
        # First Convolutional Block
        Conv3D(32, (3, 3, 3), padding='same', input_shape=input_shape),
        Activation('relu'),
        MaxPooling3D(pool_size=(2, 2, 1), padding='same'),

        # Second Convolutional Block
        Conv3D(64, (3, 3, 3), padding='same'),
        Activation('relu'),

        # Third Convolutional Block
        Conv3D(32, (3, 3, 3), padding='same'),
        Activation('relu'),

        # Flatten and Fully Connected Layer
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    rms = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

    model.fit(data_train,data_train_label,epochs=epochs, batch_size=batch_size)
    loss,accuracy = model.evaluate(data_test,data_test_label)
    return model

def CNN_model_predict(model,sample_2010_norm,im_data_land2010,im_data_land2015,im_height,im_width,number,output_path):
    label_predict = model.predict(sample_2010_norm)
    np.savetxt(output_path, label_predict, fmt='%s', newline='\n')