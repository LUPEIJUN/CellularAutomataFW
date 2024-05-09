from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from sklearn.linear_model import LogisticRegression

def LR_model_train(data_train, data_test, data_train_label, data_test_label):
    # set Logistic Regression model
    model = LogisticRegression()
    model.fit(data_train, data_train_label)
    return model


def LR_model_predict(model,sample_2010_norm,im_data_land2010,im_data_land2015,im_height,im_width,number,output_path):

    label_predict = model.predict(sample_2010_norm)
    data_new = np.zeros((im_height, im_width))

    valid_pixels = [
        (row, col)
        for row in range(12, im_height - 12)
        for col in range(12, im_width - 12)
        if im_data_land2010[row][col] != 0
    ]

    for index, (row, col) in enumerate(valid_pixels):
        data_new[row][col] = label_predict[index]

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