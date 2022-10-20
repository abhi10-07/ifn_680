from re import A
#import matplotlib.pyplot as plt
#import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
directory = "small_flower_dataset/"


def task2():
    print("Task 2 - Use MobileNetV2 network")
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    return base_model


def task3(base_model):
    print("Task 3 - Replace last layer of downloaded NN")
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tfl.Dropout(.2)(x)  # prevent overfitting
    # Classifying into 5 categories
    prediction_layer = tf.keras.layers.Dense(5, activation='softmax')
    outputs = prediction_layer(x)
    # Create new model
    new_model = tf.keras.Model(inputs, outputs)
    new_model.save('task3_model.h5')
    return new_model


def task4():
    print("Task 4 - Prepare train & validation data")
    train_dataset = image_dataset_from_directory(directory,
                                                 shuffle=True,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE,
                                                 validation_split=0.2,
                                                 subset='training',  # different subsets
                                                 seed=42)  # seed match to prevent overlapping
    validation_dataset = image_dataset_from_directory(directory,
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE,
                                                      image_size=IMG_SIZE,
                                                      validation_split=0.2,
                                                      subset='validation',  # different subsets
                                                      seed=42)  # seed match to prevent overlapping

    # Test dataset TODO

    categories = train_dataset.class_names
    return [train_dataset, validation_dataset, categories]


def task5(v2_model, train_dataset, validation_dataset):
    print("Task 5 - Compile and train with SGD")
    model = tf.keras.models.clone_model(v2_model)

    opt = tf.keras.optimizers.SGD(
        learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD"
    )

    # Train the model on new data
    model.compile(optimizer=opt,
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics="accuracy")

    initial_epochs = 3
    hist = model.fit(
        train_dataset, validation_data=validation_dataset, epochs=initial_epochs)

    # model.save('task5_model.h5')
    print(hist.history)


def task7(v2_model, train_dataset, validation_dataset):
    print("Task 7 - Try different learning rates, plot and conclude")

    learning_rates = [0.02, 0.005, 0.03]
    hist_list = []

    for lr in learning_rates:
        model = tf.keras.models.clone_model(v2_model)
        opt = tf.keras.optimizers.SGD(
            learning_rate=lr, momentum=0.0, nesterov=False, name="SGD")
        # Train the model on new data with LR 0.02
        model.compile(optimizer=opt,
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics="accuracy")
        initial_epochs = 3
        hist = model.fit(
            train_dataset, validation_data=validation_dataset, epochs=initial_epochs)
        hist_list.append(hist.history)
        # model.save('task7_model_lr_{}.h5'.format(lr))

    best_model = hist_list[0]
    best_lr = learning_rates[0]

    for i in range(len(learning_rates)):
        if(max(hist_list[i]["accuracy"]) > max(best_model["accuracy"])):
            best_model = hist_list[i]
            best_lr = learning_rates[i]

    return best_lr


def task8(v2_model, train_dataset, validation_dataset, learning_rate):
    print("Task 8 - Try different momentum rates, plot and conclude")

    momentum_rates = [0.01, 0.02, 0.03]
    hist_list = []

    for mr in momentum_rates:
        model = tf.keras.models.clone_model(v2_model)
        opt = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=mr, nesterov=False, name="SGD")
        # Train the model on new data with LR 0.02
        model.compile(optimizer=opt,
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics="accuracy")
        initial_epochs = 3
        hist = model.fit(
            train_dataset, validation_data=validation_dataset, epochs=initial_epochs)
        hist_list.append(hist.history)

    best_model = hist_list[0]
    best_mr = momentum_rates[0]

    for i in range(len(momentum_rates)):
        if(max(hist_list[i]["accuracy"]) > max(best_model["accuracy"])):
            best_model = hist_list[i]
            best_mr = momentum_rates[i]

    return best_mr


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    '''
    return [(11369574, 'Cedric', 'Oliveira da Silva Costa'), (10831908, 'Abhishek', 'Sapkal')]


if __name__ == "__main__":
    my_team()
    # task1()
    print("Task 2")
    base_model = task2()
    v2_model = task3(base_model)
    [train_dataset, validation_dataset, categories] = task4()  # TODO: test dataset
    task5(v2_model, train_dataset, validation_dataset)
    # TODO: task6 - Plot the training and validation errors vs time as well as the training and validation accuracies
    best_learning_rate = task7(
        v2_model, train_dataset, validation_dataset)  # TODO: plotting
    best_momentum = task8(v2_model, train_dataset,
                          validation_dataset, best_learning_rate)  # TODO: plotting

    # task9 - Prepare your training, validation and test sets
    # task10 - Do 8 and 9 on new dataset
