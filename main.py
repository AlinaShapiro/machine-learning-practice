from tensorflow.keras import layers
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import mobilenet, mobilenet_v2
import numpy as np
import os


if __name__ == '__main__':
    dir = './brain/'

    yes_dir = os.path.join(dir, 'yes')
    fnames = [os.path.join(yes_dir, fname) for fname in os.listdir(yes_dir)][:]
    #print(len(fnames))
    no_dir = os.path.join(dir, 'no')
    fnames.extend([os.path.join(no_dir, fname) for fname in os.listdir(no_dir)][:])
    print(fnames)

    X = []
    INPUT_SHAPE = 224
    net_model = mobilenet_v2
    net_model_class = net_model.MobileNetV2
    c = 0
    # Features
    model = net_model_class(weights='imagenet', input_shape=(INPUT_SHAPE, INPUT_SHAPE, 3), include_top=False,
                            pooling='avg')
    features_dict = {}
    for img_path in fnames:
        c += 1
        img = image.load_img(img_path, target_size=(INPUT_SHAPE, INPUT_SHAPE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = net_model.preprocess_input(x)

        preds = model.predict(x)
        features_dict[str(os.path.basename(img_path) + "_" + str(c))] = preds
        # decode the results into a list of tuples (class, description, probability)
        # (one such list for each sample in the batch)
        X.append(np.array(preds))
        print('Predicted:', preds.shape, preds)
    print(c)

    print(features_dict.items())

    y = []
    count_pos = 0
    count_neg = 0
    for img_path in fnames:
        if 'Y' in os.path.basename(img_path):
            y.append(1)
            count_pos += 1
        else:
            y.append(0)
            count_neg += 1
    print(y)
    Y = np.array(y)
    print('count_pos: ', count_pos, '\ncount_neg: ', count_neg)

    X = np.reshape(np.array(X), (253, 1280))
    print(X)
    print(X.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    print(X_train[:5], "\n", Y_train[:5])
    print(len(X_train))
    print(len(X_test))

    model = keras.Sequential([
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["accuracy"]
                 )
    x_val = X_train[:53]
    partial_x_train = X_train[53:]
    y_val = Y_train[:53]
    partial_y_train = Y_train[53:]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))

    print(model.predict(X_test))

