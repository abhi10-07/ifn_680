import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Constants and vars
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
EPOCHS = 7
AUTOTUNE = tf.data.AUTOTUNE
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
directory = "small_flower_dataset/"

# Ignore Warnings due to data augmentation giving while_loop warnings while converting. Seems to be a keras bug in certain versions.
# https://stackoverflow.com/questions/73304934/tensorflow-data-augmentation-gives-a-warning-using-a-while-loop-for-converting
tf.get_logger().setLevel('ERROR')

# Data preprocess layers
resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.Resizing(IMG_SIZE[0], IMG_SIZE[1]),
    tf.keras.layers.Rescaling(1./127.5, offset=-1)
])

# Data augmentation layers
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])


def create_new_model(base_model):
    '''
    Adds a preprocess input layer and output layers to a MobileNetV2 model.

    @param base_model: Takes a MobileNetV2 base model WITHOUT the output layers as an input.

    @return
        Returns a model of the given input model with new input and output layers.
    '''

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(.2)(x)  # prevent overfitting

    # Classifying into 5 categories
    prediction_layer = tf.keras.layers.Dense(5, activation="softmax")
    outputs = prediction_layer(x)

    # Create new model
    new_model = tf.keras.Model(inputs, outputs)
    return new_model


def prepare(ds, shuffle=False, augment=False):
    '''
    Prepares given datasets for the model and augments the data.

    @param ds: Dataset of type 'tf.keras.datasets'

    @param shuffle: Boolean to determine if dataset should be shuffled

    @param augment: Bollean to determine if dataset should be augmented

    @return
        Returns a resized, rescaled, shuffeld and augmented dataset.
    '''
    # Resize and rescale all datasets.
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
                num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)


def get_best_lr(model, train_dataset, validation_dataset, test_dataset):
    '''
    Tests multiple learning rates and picks the best one for the given input.

    @param model: Model of type 'tf.keras.Model()'

    @param train_dataset: Train dataset of type 'tf.keras.datasets'

    @param validation_dataset: Valdiation dataset of type 'tf.keras.datasets'

    @param test_dataset: Test dataset of type 'tf.keras.datasets'

    @return
        Returns the best found learning rate.
    '''
    learning_rates = [0.000005, 0.00001, 0.1, 0.01]
    loss_list = []
    acc_list = []

    for lr in learning_rates:
        print("Testing learning rate: {}".format(lr))
        current_model = tf.keras.models.clone_model(model)
        opt = tf.keras.optimizers.SGD(
            learning_rate=lr, momentum=0.0, nesterov=False, name="SGD")
        current_model.compile(optimizer=opt,
                              loss=tf.keras.losses.CategoricalCrossentropy(),
                              metrics="accuracy")

        hist = current_model.fit(
            train_dataset, validation_data=validation_dataset, epochs=EPOCHS)
        loss, acc = current_model.evaluate(test_dataset)
        acc_list.append(acc)
        loss_list.append(loss)

    best_lr = learning_rates[0]
    best_acc = acc_list[0]

    for i in range(len(learning_rates)):
        if(acc_list[i] > best_acc):
            best_acc = acc_list[i]
            best_lr = learning_rates[i]

    return [best_lr, learning_rates, acc_list, loss_list]


def get_best_mr(model, train_dataset, validation_dataset, test_dataset, learning_rate, task_no):
    '''
    Tests multiple momentum rates and picks the best one for the given input.

    @param model: Model of type 'tf.keras.Model()'

    @param train_dataset: Train dataset of type 'tf.keras.datasets'

    @param validation_dataset: Valdiation dataset of type 'tf.keras.datasets'

    @param test_dataset: Test dataset of type 'tf.keras.datasets'

    @return
        Returns the best found momentum rate.
    '''
    momentum_rates = [0.0, 0.001, 0.01, 0.1]
    loss_list = []
    acc_list = []

    for index, mr in enumerate(momentum_rates):
        print("Testing momentum rate: {}".format(mr))
        current_model = tf.keras.models.clone_model(model)
        opt = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=mr, nesterov=False, name="SGD")
        current_model.compile(optimizer=opt,
                              loss=tf.keras.losses.CategoricalCrossentropy(),
                              metrics="accuracy")
        hist = current_model.fit(
            train_dataset, validation_data=validation_dataset, epochs=EPOCHS)
        loss, acc = current_model.evaluate(test_dataset)
        acc_list.append(acc)
        loss_list.append(loss)
        current_model.save('task' + format(task_no) +
                           '-momentum-' + format(index+1) + '.h5')

    best_mr = momentum_rates[0]
    best_acc = acc_list[0]

    for i in range(len(momentum_rates)):
        if(acc_list[i] > best_acc):
            best_acc = acc_list[i]
            best_mr = momentum_rates[i]

    return [best_mr, momentum_rates, acc_list, loss_list]


def bar_plot(labels, acc_arr, loss_arr, title, filename):
    '''
    Uses pyplot to plot provided data to the function and saves it as an image. 

    @param labels: Labels for x-axis as array of strings

    @param acc_arr: Accuarcy values as array

    @param loss_arr: Loss values as array

    @param title: Title as string

    @param filename: Filename as string
    '''
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, acc_arr, width, label='Accuracy')
    rects2 = ax.bar(x + width/2, loss_arr, width, label='Loss')

    ax.set_ylabel('Accuracy/Loss')
    ax.set_title(title)
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.savefig(filename)
    plt.show()


def prediction_plot(model_file, data, class_names, filename):
    '''
    Load the saved model and prediction on the image dataset is executed

    @param model: Model of type 'tf.keras.Model()'

    @param model_file: Name of the model to be loaded

    @param data: Test dataset of type 'tf.keras.datasets'

    @param class_names: Classification labels such as daisy, tulip etc.

    @param filename: Name of the image file to be saved.

    @return
        Returns prediction plot image 
    '''
    model = tf.keras.models.load_model(model_file)
    image_batch, label_batch = next(iter(data))
    # turn the original labels into human-readable text
    label_batch = [class_names[np.argmax(label_batch[i])]
                   for i in range(BATCH_SIZE)]
    # predict the images on the model
    predicted_class_names = model.predict(image_batch)
    predicted_ids = [np.argmax(predicted_class_names[i])
                     for i in range(BATCH_SIZE)]
    # turn the predicted vectors to human readable labels
    predicted_class_names = np.array([class_names[id] for id in predicted_ids])
    # plotting
    plt.figure(figsize=(10, 9))
    for n in range(24
                   ):
        plt.subplot(6, 4, n+1)
        plt.subplots_adjust(hspace=0.3)
        plt.imshow(image_batch[n])
        if predicted_class_names[n] == label_batch[n]:
            color = "blue"
            title = predicted_class_names[n].title()
        else:
            color = "red"
            title = f"{predicted_class_names[n].title()}, correct:{label_batch[n]}"
        plt.title(title, color=color)
        plt.axis('off')
    _ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
    plt.savefig(filename)
    plt.show()

def task2():
    '''
    Using the tf.keras.applications module download a pretrained MobileNetV2 network.

    @return
        Returns a model of a pretrained MobileNetV2 network WITHOUT the output layer.
    '''
    print("Task 2 - Use MobileNetV2 network")
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    return base_model


def task3(base_model):
    '''
    Replace the last layer of the downloaded neural network with a Dense layer of the appropriate shape for the 5 classes of the small flower dataset {(x1,t1), (x2,t2),..., (xm,tm)}     

    @param base_model: Base MobileNetV2 model of type 'tf.keras.Model()'

    @return
        Returns model with a Dense layer shape of 5 classes.
    '''
    print("Task 3 - Replace last layer of downloaded NN")
    new_model = create_new_model(base_model)
    return new_model


def task4():
    '''
    Prepare your training, validation and test sets for the non-accelerated version of transfer learning. 

    @return    
        Returns new training, validation and test datasets in an array of form [train,validation,test].
    '''
    print("Task 4 - Prepare train, validation and test datasets")
    train_dataset = tf.keras.utils.image_dataset_from_directory(directory,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE,
                                                                validation_split=0.2,
                                                                label_mode='categorical',
                                                                subset='training',  # different subsets
                                                                seed=42)  # seed match to prevent overlapping

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = tf.keras.utils.image_dataset_from_directory(directory,
                                                                     shuffle=True,
                                                                     batch_size=BATCH_SIZE,
                                                                     image_size=IMG_SIZE,
                                                                     validation_split=0.2,
                                                                     label_mode='categorical',
                                                                     subset='validation',  # different subsets
                                                                     seed=42)  # seed match to prevent overlapping
    class_names = validation_dataset.class_names
    test_dataset = validation_dataset.take(1)
    validation_dataset = validation_dataset.skip(1)
    print("Train dataset batches: {}".format(train_dataset.cardinality()))
    print("Validation dataset batches: {}".format(
        validation_dataset.cardinality()))
    print("Test dataset batches: {}".format(test_dataset.cardinality()))
    return [train_dataset, validation_dataset, test_dataset, class_names]


def task5(model, train_dataset, validation_dataset, test_dataset):
    '''
    Compile and train your model with an SGD3 optimizer using the following parameters learning_rate=0.01, momentum=0.0, nesterov=False.

    @param model: Model of type 'tf.keras.Model()'

    @param train_dataset: Train dataset of type 'tf.keras.datasets'

    @param validation_dataset: Valdiation dataset of type 'tf.keras.datasets'

    @param test_dataset: Test dataset of type 'tf.keras.datasets'

    @return
        Returns training history as an object. 
    '''
    print("Task 5 - Compile and train with SGD")
    current_model = tf.keras.models.clone_model(model)

    opt = tf.keras.optimizers.SGD(
        learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD"
    )
    # Train the model on new data
    current_model.compile(optimizer=opt,
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics="accuracy")
    history = current_model.fit(
        train_dataset, validation_data=validation_dataset, epochs=EPOCHS)

    model.save('task5.h5')

    loss, acc = current_model.evaluate(test_dataset)
    print("Task5: Accuracy for test_dataset: {}".format(acc))

    return history


def task6(hist, task_no, acc_filename, loss_filename):
    '''
    Plot the training and validation errors vs time as well as the training and validation accuracies.

    @param hist: Keras history object

    @param task_no: Integer to identify the task

    @param acc_filename: Filename as string

    @param loss_filename: Filename as string
    '''
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(acc)+1)
    # Training and validation accuracy
    plt.figure()
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Task-'+format(task_no)+'A: Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(acc_filename)
    plt.show()
    # Training and validation loss
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Task-'+format(task_no)+'B: Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_filename)
    plt.show()


def task7(model, train_dataset, validation_dataset, test_dataset):
    '''
    Experiment with 3 different orders of magnitude for the learning rate. Plot the results, draw conclusions.

    @param model: Model of type 'tf.keras.Model()'

    @param train_dataset: Train dataset of type 'tf.keras.datasets'

    @param validation_dataset: Valdiation dataset of type 'tf.keras.datasets'

    @param test_dataset: Test dataset of type 'tf.keras.datasets'

    @return
        Returns the best calculated learning rate.
    '''
    print("Task 7 - Try different learning rates, plot and conclude")
    [best_lr, learning_rates, acc_list, loss_list] = get_best_lr(
        model, train_dataset, validation_dataset, test_dataset)
    print("Best found learning rate: {}".format(best_lr))

    filename = 'Task7-best-learning-rate.jpg'
    title = "Get highest Accuracy/Loss for Learning rate for task 7"

    bar_plot(learning_rates, acc_list, loss_list, title, filename)

    return best_lr


def task8(model, train_dataset, validation_dataset, test_dataset, learning_rate):
    '''
    With the best learning rate that you found in task7, add a non zero momentum to the training with the SGD optimizer (consider 3 values for the momentum). Report how your results change.

    @param model: Model of type 'tf.keras.Model()'

    @param train_dataset: Train dataset of type 'tf.keras.datasets'

    @param validation_dataset: Valdiation dataset of type 'tf.keras.datasets'

    @param test_dataset: Test dataset of type 'tf.keras.datasets'

    @param learning_rate: Defined learning rate for task8

    @return
        Returns the best calculated momentum rate.
    '''
    print("Task 8 - Try different momentum rates, plot and conclude")
    [best_mr, momentum_rates, acc_list, loss_list] = get_best_mr(model, train_dataset,
                                                                 validation_dataset, test_dataset, learning_rate, 8)
    print("Best found momentum rate: {}".format(best_mr))
    filename = 'Task8-best-momentum-rate.jpg'
    title = "Get highest Accuracy/Loss for Momentum rate for Task 8"

    bar_plot(momentum_rates, acc_list, loss_list, title, filename)

    return best_mr


def task9(train_dataset, validation_dataset, test_dataset):
    '''
    Prepare your training, validation and test sets. Those are based on  {(F(x1).t1),(F(x2),t2),...,(F(xm),tm)}

    @param train_dataset: Train dataset of type 'tf.keras.datasets'

    @param validation_dataset: Valdiation dataset of type 'tf.keras.datasets'

    @param test_dataset: Test dataset of type 'tf.keras.datasets'

    @return
        Returns new training, validation and test datasets in an array of form [train,validation,test].
    '''
    print("Task 9 - Prepare train, validation and datasets based on F(x)")
    train_ds = prepare(train_dataset, shuffle=True, augment=True)
    val_ds = prepare(validation_dataset)
    test_ds = prepare(test_dataset)

    return [train_ds, val_ds, test_ds]


def task10(model, fx_train_dataset, fx_validation_dataset, fx_test_dataset):
    '''
    Perform Task 8 on the new dataset created in Task 9.

    @param model: Model of type 'tf.keras.Model()'

    @param fx_train_dataset: Train dataset of type 'tf.keras.datasets'

    @param fx_validation_dataset: Valdiation dataset of type 'tf.keras.datasets'

    @param fx_test_dataset: Test dataset of type 'tf.keras.datasets'

    @return
        Returns the best calculated learning & momentum rate in an array of form [learning_rate, momentum_rate]
    '''
    print("Task 10 - Try different learning/momentum rates with new dataset, plot and conclude")
    [best_lr, learning_rates, acc_list, loss_list] = get_best_lr(
        model, fx_train_dataset, fx_validation_dataset, fx_test_dataset)
    filename = 'Task10-best-learning-rate.jpg'
    title = "Get highest Accuracy/Loss for Learning rate with new dataset"

    bar_plot(learning_rates, acc_list, loss_list, title, filename)
    [best_mr, momentum_rates, acc_list, loss_list] = get_best_mr(
        model, fx_train_dataset, fx_validation_dataset, fx_test_dataset, best_lr, 10)
    filename = 'Task10-best-momentum-rate.jpg'
    title = "Get highest Accuracy/Loss for Momentum rate with new dataset"

    bar_plot(momentum_rates, acc_list, loss_list, title, filename)

    return [best_lr, best_mr]


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    '''
    return [(11369574, 'Cedric', 'Oliveira da Silva Costa'), (10831908, 'Abhishek', 'Sapkal')]


if __name__ == "__main__":
    my_team()
    # Task2
    base_model = task2()

    # Task3
    model = task3(base_model)

    # Task4
    [train_dataset, validation_dataset, test_dataset, class_names] = task4()

    # Task5
    history = task5(model, train_dataset,
                    validation_dataset, test_dataset)
    prediction_plot('task5.h5', test_dataset,
                    class_names, 'prediction-task-5.jpg')
    # Task6
    task6(history, 6, 'Accuracy-task-6.jpg', 'Loss-task-6.jpg')

    # Task7
    best_learning_rate_non_accelerated = task7(
        model, train_dataset, validation_dataset, test_dataset)

    # Task8
    best_momentum_non_accelerated = task8(
        model, train_dataset, validation_dataset, test_dataset, best_learning_rate_non_accelerated)
    prediction_plot('task8-momentum-1.h5', test_dataset,
                    class_names, 'prediction-task-8.1.jpg')
    prediction_plot('task8-momentum-2.h5', test_dataset,
                    class_names, 'prediction-task-8.2.jpg')
    prediction_plot('task8-momentum-3.h5', test_dataset,
                    class_names, 'prediction-task-8.3.jpg')
    prediction_plot('task8-momentum-4.h5', test_dataset,
                    class_names, 'prediction-task-8.4.jpg')

    # Task9
    [fx_train_dataset, fx_validation_dataset, fx_test_dataset] = task9(
        train_dataset, validation_dataset, test_dataset)

    # Task10
    [best_learning_rate_accelerated, best_momentum_accelerated] = task10(
        model, fx_train_dataset, fx_validation_dataset, fx_test_dataset)

    prediction_plot('task10-momentum-1.h5', test_dataset,
                    class_names, 'prediction-task-10.1.jpg')
    prediction_plot('task10-momentum-2.h5', test_dataset,
                    class_names, 'prediction-task-10.2.jpg')
    prediction_plot('task10-momentum-3.h5', test_dataset,
                    class_names, 'prediction-task-10.3.jpg')
    prediction_plot('task10-momentum-4.h5', test_dataset,
                    class_names, 'prediction-task-10.4.jpg')
