from re import A
import matplotlib.pyplot as plt
#import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
EPOCHS = 4
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
directory = "small_flower_dataset/"

AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.get_logger().setLevel('ERROR')

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])


def task2():
    print("Task 2 - Use MobileNetV2 network")
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    # print(base_model.summary())
    return base_model


def create_base_model(base_model, useAugmenter=False):
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    if(useAugmenter):
        # print("Model using augmenter")
        x = data_augmentation(inputs)
        x = preprocess_input(x)
    else:
        # print("Model not using augmenter")
        x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(.2)(x)  # prevent overfitting
    # Classifying into 5 categories
    prediction_layer = tf.keras.layers.Dense(5, activation="softmax")
    outputs = prediction_layer(x)
    # Create new model
    new_model = tf.keras.Model(inputs, outputs)
    # print(new_model.summary())
    return new_model


def task3(base_model):
    print("Task 3 - Replace last layer of downloaded NN")
    new_model = create_base_model(base_model)
    # print(new_model.summary())
    return new_model


def task4():
    print("Task 4 - Prepare train, validation and test datasets")
    train_dataset = image_dataset_from_directory(directory,
                                                 shuffle=True,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE,
                                                 validation_split=0.2,
                                                 label_mode='categorical',
                                                 subset='training',  # different subsets
                                                 seed=42)  # seed match to prevent overlapping
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = image_dataset_from_directory(directory,
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE,
                                                      image_size=IMG_SIZE,
                                                      validation_split=0.2,
                                                      label_mode='categorical',
                                                      subset='validation',  # different subsets
                                                      seed=42)  # seed match to prevent overlapping
    test_dataset = validation_dataset.take(1)
    validation_dataset = validation_dataset.skip(1)
    print("Train dataset batches: {}".format(train_dataset.cardinality()))
    print("Validation dataset batches: {}".format(
        validation_dataset.cardinality()))
    print("Test dataset batches: {}".format(test_dataset.cardinality()))
    return [train_dataset, validation_dataset, test_dataset]


def task5(model, train_dataset, validation_dataset, test_dataset):
    print("Task 5 - Compile and train with SGD")
    current_model = tf.keras.models.clone_model(model)
    opt = tf.keras.optimizers.SGD(
        learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD"
    )
    # Train the model on new data
    current_model.compile(optimizer=opt,
                          loss=keras.losses.CategoricalCrossentropy(),
                          metrics="accuracy")
    history = current_model.fit(
        train_dataset, validation_data=validation_dataset, epochs=EPOCHS)
    return history

def task6(hist, task_no, acc_filename, loss_filename):
    # print(hist.history)
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1,len(acc)+1)
    # Training and validation accuracy
    plt.figure()
    plt.plot(epochs, acc, 'b', label = 'Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Task-'+format(task_no)+'A: Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(acc_filename)
    plt.show()
    # Training and validation loss
    plt.figure()
    plt.plot(epochs, loss, 'b', label = 'Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Task-'+format(task_no)+'B: Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_filename)
    plt.show()


def get_best_lr(model, train_dataset, validation_dataset):
    learning_rates = [0.01, 0.02, 0.005, 0.03]
    hist_list = []
    history_arr = []

    for lr in learning_rates:
        print("Testing learning rate: {}".format(lr))
        current_model = tf.keras.models.clone_model(model)
        opt = tf.keras.optimizers.SGD(
            learning_rate=lr, momentum=0.0, nesterov=False, name="SGD")
        # Train the model on new data with LR 0.02
        current_model.compile(optimizer=opt,
                              loss=keras.losses.CategoricalCrossentropy(),
                              metrics="accuracy")

        hist = current_model.fit(
            train_dataset, validation_data=validation_dataset, epochs=EPOCHS)
        hist_list.append(hist.history)
        history_arr.append(hist)

    best_model = hist_list[0]
    best_lr = learning_rates[0]

    for i in range(len(learning_rates)):
        if(max(hist_list[i]["accuracy"]) > max(best_model["accuracy"])):
            best_model = hist_list[i]
            best_lr = learning_rates[i]

    return [best_lr,learning_rates,history_arr]


def get_best_mr(model, train_dataset, validation_dataset, learning_rate):
    momentum_rates = [0.0, 0.01, 0.02, 0.03]
    hist_list = []
    history_arr = []

    for mr in momentum_rates:
        print("Testing momentum rate: {}".format(mr))
        current_model = tf.keras.models.clone_model(model)
        opt = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=mr, nesterov=False, name="SGD")
        # Train the model on new data with LR 0.02
        current_model.compile(optimizer=opt,
                              loss=keras.losses.CategoricalCrossentropy(),
                              metrics="accuracy")

        hist = current_model.fit(
            train_dataset, validation_data=validation_dataset, epochs=EPOCHS)
        hist_list.append(hist.history)
        history_arr.append(hist)

    best_model = hist_list[0]
    best_mr = momentum_rates[0]

    for i in range(len(momentum_rates)):
        if(max(hist_list[i]["accuracy"]) > max(best_model["accuracy"])):
            best_model = hist_list[i]
            best_mr = momentum_rates[i]

    return [best_mr, momentum_rates, history_arr]


def task7(model, train_dataset, validation_dataset):
    print("Task 7 - Try different learning rates, plot and conclude")
    [best_lr,learning_rates, history_arr] = get_best_lr(model, train_dataset, validation_dataset)
    print("Best found learning rate: {}".format(best_lr))
    for i in range(len(history_arr)):
        task_no = '7-learning-rate:'+ format(learning_rates[i])
        acc_filename = 'Accuracy-task-'+task_no+'.jpg'
        loss_filename = 'Loss-task-'+task_no+'.jpg'
        task6(history_arr[i], task_no, acc_filename, loss_filename)
    return best_lr


def task8(model, train_dataset, validation_dataset, learning_rate):
    print("Task 8 - Try different momentum rates, plot and conclude")
    [best_mr, momentum_rates, history_arr] = get_best_mr(model, train_dataset,
                          validation_dataset, learning_rate)
    print("Best found momentum rate: {}".format(best_mr))
    for i in range(len(history_arr)):
        task_no = '8-learning-rate:'+ format(momentum_rates[i])
        acc_filename = 'Accuracy-task-'+task_no+'.jpg'
        loss_filename = 'Loss-task-'+task_no+'.jpg'
        task6(history_arr[i], task_no, acc_filename, loss_filename)
    return best_mr


def task9(base_model):
    print("Task 9 - Use F(x) function to augument datasets")
    return create_base_model(base_model, useAugmenter=True)


def task10(model, train_dataset, validation_dataset, learning_rate):
    print("Task 10 - Try different momentum rates with new dataset, plot and conclude")
    [best_mr, momentum_rates, history_arr] =  get_best_mr(model, train_dataset, validation_dataset, learning_rate)
    for i in range(len(history_arr)):
        task_no = '10-learning-rate:'+ format(momentum_rates[i])
        acc_filename = 'Accuracy-task-'+task_no+'.jpg'
        loss_filename = 'Loss-task-'+task_no+'.jpg'
        task6(history_arr[i], task_no, acc_filename, loss_filename)
    return best_mr

def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    '''
    return [(11369574, 'Cedric', 'Oliveira da Silva Costa'), (10831908, 'Abhishek', 'Sapkal')]


if __name__ == "__main__":
    my_team()
    base_model = task2()
    v2_non_accelerated_model = task3(base_model)
    [train_dataset, validation_dataset, test_dataset] = task4()
    history = task5(v2_non_accelerated_model, train_dataset,
          validation_dataset, test_dataset)
    task6(history, 6, 'Accuracy-task-6.jpg', 'Loss-task-6.jpg')
    best_learning_rate = task7(
        v2_non_accelerated_model, train_dataset, validation_dataset)  # TODO: plotting
    best_momentum_non_accelerated = task8(v2_non_accelerated_model, train_dataset,
                                          validation_dataset, best_learning_rate)  # TODO: plotting

    v2_accelerated_model = task9(base_model)
    best_momentum_accelerated = task10(
        v2_accelerated_model, train_dataset, validation_dataset, best_learning_rate)  # TODO: plotting
