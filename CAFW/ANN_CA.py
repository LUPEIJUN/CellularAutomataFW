from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import RMSprop
import numpy as np

def ANN_model_train(data_train, data_test, data_train_label, data_test_label,
              input_dim = 23, 
              num_classes = 7,
              epochs = 10,
              batch_size = 16):
    # set neural networks
    model = Sequential([
        Dense(32, input_dim=input_dim),
        Activation('relu'),
        Dense(64),
        Activation('relu'),
        Dropout(0.02),
        Dense(num_classes),
        Activation('softmax'),
    ])

    # select optimizer
    rmsprop = RMSprop(learning_rate=0.02, rho=0.8, epsilon=1e-08, decay=0.0)
    
    # The loss function and the precision evaluation index are selected
    model.compile(optimizer=rmsprop,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    model.fit(data_train, data_train_label, epochs=epochs, batch_size=batch_size)
    loss, accuracy = model.evaluate(data_test, data_test_label)
    return model


def ANN_model_predict(model,sample_2010_norm,im_data_land2010,im_data_land2015,im_height,im_width,number,output_path):

    threshold = 0.8
    predictions = (model.predict(sample_2010_norm, batch_size=32, verbose=0) >= threshold).astype(int)
    change_sum = predictions.sum()

    print("The number of predictions > threshold:", change_sum)
    print("The rate of predictions > threshold:", change_sum / number)

    # Map predictions back to pixel positions
    label_predict = np.zeros((number, 1))
    valid_pixels = [
        (row, col)
        for row in range(12, im_height - 12)
        for col in range(12, im_width - 12)
        if im_data_land2010[row][col] != 0
    ]
    
    for index, (row, col) in enumerate(valid_pixels):
        # Assign the predicted class
        if predictions[index].sum() > 0:
            label_predict[index][0] = np.argmax(predictions[index])
        else:
            label_predict[index][0] = im_data_land2010[row][col]
    # Create the simulated data array
    data_new = np.zeros((im_height, im_width))
    for index, (row, col) in enumerate(valid_pixels):
        data_new[row][col] = label_predict[index][0]
    
    # Compare original and simulated data
    same_label_origin = sum(
        1
        for row, col in valid_pixels
        if im_data_land2010[row][col] == im_data_land2015[row][col]
    )
    same_label = sum(
        1
        for row, col in valid_pixels
        if im_data_land2010[row][col] == data_new[row][col]
    )

    same = sum(
        1
        for row, col in valid_pixels
        if im_data_land2015[row][col] == data_new[row][col]
    )

    print("The same label between im_data_land2010 and im_data_land2015 = ", same_label_origin)
    print("The same label between im_data_land2010 and data_new = ", same_label)
    print("The same label between im_data_land2015 and data_new = ", same)
    print("The accuracy of prediction is:", same / number)    
    np.savetxt(output_path, data_new, fmt='%s', newline='\n')
    return data_new