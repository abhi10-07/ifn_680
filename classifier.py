import os
from tabnanny import verbose
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

data_dir = './small_flower_dataset'

categories = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

data = []

class Utils():

    def make_data():
        for category in categories:
            path = os.path.join(data_dir, category)
            label = categories.index(category)

            for img_name in os.listdir(path):
                image_path = os.path.join(path, img_name)
                image = cv2.imread(image_path)

                # cv2.imshow("Random image", image)

                try: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (224, 224))

                    image = np.array(image, dtype=np.float32)

                    data.append([image, label])
                
                except Exception as e:
                    pass

            print(len(data))

            pic = open('data.pickle', 'wb')
            pickle.dump(data, pic)
            pic.close()


            #     break
            # break
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    make_data()

    def load_data():
        pick = open('data.pickle', 'rb')
        data = pickle.load(pick)
        pick.close()

        np.random.shuffle(data)

        features = []
        labels = []

        for img, label in data:
            features.append(img)
            labels.append(label)

        features = np.array(features, dtype=np.float32)
        labels = np.array(labels)

        return [features, labels]
        

(features, labels) = Utils.load_data()

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.1)

def models(): 
    

    input_layer = tf.keras.layers.Input([224, 224, 3])

    conv1  = tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu')(input_layer)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)) (conv2)

    conv3 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), padding='Same', activation='relu')(pool2)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)) (conv3)

    conv4 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), padding='Same', activation='relu')(pool3)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)) (conv4)

    flat1 = tf.keras.layers.Flatten()(pool4)
    dns1 = tf.keras.layers.Dense(512, activation='relu')(flat1)
    out = tf.keras.layers.Dense(5, activation='softmax')(dns1)

    #Model
    model = tf.keras.Model(input_layer, out)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=100, epochs=10)

    model.save('mymodel.h5')


def detect():
    model = tf.keras.models.load_model('mymodel.h5')

    # model.evaluate(x_test, y_test, verbose = 1)

    prediction = model.predict(x_test)
    plt.figure(figsize=(9,9))
    for i in range(9):
        plt.subplot(3, 3,i+1)
        plt.imshow(x_test[i])
        plt.xlabel('Actual: ' + categories[y_test[i]] + '\n Predicted:' + categories[np.argmax(prediction[i])])
        plt.xticks([])

    plt.show()  


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [(11369574, 'Cedric', 'Oliveira da Silva Costa'), (10831908, 'Abhishek', 'Sapkal')]

# ----------------------------------------

if __name__ == "__main__":
    my_team()
    # models()
    detect()


# ----------------------------------------
