import numpy as np
from keras.layers import Input, Dense, Activation, ZeroPadding2D, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.models import load_model
from keras import metrics

from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from PIL import Image
import keras.backend as K
import tensorflow as tf
import keras
from keras.wrappers.scikit_learn import KerasClassifier

K.set_image_data_format('channels_last')
from matplotlib.pyplot import imshow
import os

modelSavePath = 'tumour_model.h5'
numOfTestPoints = 2
batchSize = 16
numOfEpoches = 100

classes = []


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)



def getCropImgs(img, needRotations=False):
    # img = img.convert('L')
    z = np.asarray(img, dtype=np.int8)
    c = []
    for i in range(3):
        for j in range(4):
            crop = z[512 * i:512 * (i + 1), 512 * j:512 * (j + 1), :]

            c.append(crop)
            if needRotations:
                c.append(np.rot90(np.rot90(crop)))

    return c

def getAsSoftmax(fname):
    if (fname == 'b'):
        return [1, 0, 0, 0]
    elif (fname == 'is'):
        return [0, 1, 0, 0]
    elif (fname == 'iv'):
        return [0, 0, 1, 0]
    else:
        return [0, 0, 0, 1]

def get_imgs_frm_folder(path):

    x = []
    y = []

    cnt = 0
    for foldname in os.listdir(path):
        for filename in os.listdir(os.path.join(path, foldname)):
            img = Image.open(os.path.join(os.path.join(path, foldname), filename))
            crpImgs = getCropImgs(img)
            cnt += 1
            if cnt % 10 == 0:
                print(str(cnt) + " Images loaded")
            for im in crpImgs:
                x.append(np.divide(np.asarray(im, np.float16), 255.))
                y.append(getAsSoftmax(foldname))
              

    print("Images cropped")
    print("Loading as array")

    return x, y, cnt

# Load the dataset
def load_dataset(testNum=numOfTestPoints):
    print("Loading images..")

    train_set_x_orig, train_set_y_orig, cnt = get_imgs_frm_folder(dataTrainPath)

    testNum = numOfTestPoints * 12
    trainNum = (cnt * 12) - testNum

    print(testNum, trainNum)

    train_set_x_orig = np.array(train_set_x_orig, np.float16)
    train_set_y_orig = np.array(train_set_y_orig, np.int8)

    nshapeX = train_set_x_orig.shape
    nshapeY = train_set_y_orig.shape

    print("folder trainX" + str(nshapeX))
    print("folder trainY" + str(nshapeY))

    print("Images loaded")

    print("Loading all data")

    test_set_x_orig = train_set_x_orig[trainNum:, :, :, :]
    train_set_x_orig = train_set_x_orig[0:trainNum, :, :, :]

    test_set_y_orig = train_set_y_orig[trainNum:]
    train_set_y_orig = train_set_y_orig[0:trainNum]

    classes = np.array(os.listdir(dataTrainPath))  # the list of classes

    print(train_set_y_orig[0:50, :])
    print(train_set_x_orig[1])
    print("Data load complete")

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def defModel(input_shape):
    X_input = Input(input_shape)

    # The max pooling layers use a stride equal to the pooling size

    X = Conv2D(16, (3, 3), strides=(1, 1))(X_input)  # 'Conv.Layer(1)'

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=3)(X)  # MP Layer(2)

    X = Conv2D(32, (3, 3), strides=(1, 1))(X)  # Conv.Layer(3)

    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), strides=2)(X)  # MP Layer(4)

    X = Conv2D(64, (2, 2), strides=(1, 1))(X)  # Conv.Layer(5)

    X = Activation('relu')(X)

    X = ZeroPadding2D(padding=(2, 2))(X)  # Output of convlayer(5) will be 82x82, we want 84x84

    X = MaxPooling2D((2, 2), strides=2)(X)  # MP Layer(6)

    X = Conv2D(64, (2, 2), strides=(1, 1))(X)  # Conv.Layer(7)

    X = Activation('relu')(X)

    X = ZeroPadding2D(padding=(2, 2))(X)  # Output of convlayer(7) will be 40x40, we want 42x42

    X = MaxPooling2D((3, 3), strides=3)(X)  # MP Layer(8)

    X = Conv2D(32, (3, 3), strides=(1, 1))(X)  # Con.Layer(9)

    X = Activation('relu')(X)

    X = Flatten()(X)  # Convert it to FC

    X = Dense(256, activation='relu')(X)  # F.C. layer(10)

    X = Dense(128, activation='relu')(X)  # F.C. layer(11)

    X = Dense(4, activation='softmax')(X)

    # ------------------------------------------------------------------------------

    model = Model(inputs=X_input, outputs=X, name='Model')

    return model


def train(batch_size, epochs):
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    model = defModel(X_train.shape[1:])

    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
  
    while True:
        try:
            model = load_model(modelSavePath)
        except:
            print("Training a new model")

        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size) # <>

        model.save(modelSavePath)

        preds = model.evaluate(X_test, Y_test_orig, batch_size=1, verbose=1, sample_weight=None)
        print(preds)

        print()
        print("Loss = " + str(preds[0]))
        print("Test Accuracy = " + str(preds[1]) + "\n\n\n\n\n")
        ch = input("Do you wish to continue training? (y/n) ")
        if ch == 'y':
            epochs = int(input("How many epochs this time? : "))
            continue
        else:
            break

    return model


def predict(img, savedModelPath, showImg=True):
    model = load_model(savedModelPath)

    x = img
    if showImg:
        Image.fromarray(np.array(img, np.float16), 'RGB').show()
    x = np.expand_dims(x, axis=0)

    softMaxPred = model.predict(x)
    print("prediction from CNN: " + str(softMaxPred) + "\n")
    probs = softmaxToProbs(softMaxPred)

    maxprob = 0
    maxI = 0
    for j in range(len(probs)):
        if probs[j] > maxprob:
            maxprob = probs[j]
            maxI = j
    # print(softMaxPred)
    print("prediction index: " + str(maxI))
    return maxI, probs


def softmaxToProbs(soft):
    z_exp = [np.math.exp(i) for i in soft[0]]
    sum_z_exp = sum(z_exp)
    return [(i / sum_z_exp) * 100 for i in z_exp]


def predictImage(img_path='my_image.jpg', arrayImg=None, printData=True):
    crops = []
    if arrayImg == None:
        img = image.load_img(img_path)
        crops = np.array(getCropImgs(img, needRotations=False), np.float16)
        crops = np.divide(crops, 255.)
    Image.fromarray(np.array(crops[0]), "RGB").show()

    classes = []
    classes.append("Benign")
    classes.append("InSitu")
    classes.append("Invasive")
    classes.append("Normal")

    compProbs = []
    compProbs.append(0)
    compProbs.append(0)
    compProbs.append(0)
    compProbs.append(0)

    for i in range(len(crops)):
        if printData:
            print("\n\nCrop " + str(i + 1) + " prediction:\n")

        ___, probs = predict(crops[i], modelSavePath, showImg=False)

        for j in range(len(classes)):
            if printData:
                print(str(classes[j]) + " : " + str(round(probs[j], 4)) + "%")
            compProbs[j] += probs[j]

    if printData:
        print("\n\nAverage from all crops\n")

    for j in range(len(classes)):
        if printData:
            print(str(classes[j]) + " : " + str(round(compProbs[j] / 12, 4)) + "%")


print("1. Do you want to train the network\n"
      "2. Test the model\n(Enter 1 or 2)?\n")
ch = int(input())
if ch == 1:

    try:
        classes = np.load('classes.npy')
        print("Loading")
        X_train = np.load('X_train.npy')
        Y_train = np.load('Y_train.npy')
        X_test = np.load('X_test.npy')
        Y_test_orig = np.load('Y_test_orig.npy')
    except:
        X_train, Y_train, X_test, Y_test_orig, classes = load_dataset()
        print("Saving...")
        np.save('X_train', X_train)
        np.save('Y_train', Y_train)
        np.save('X_test', X_test)
        np.save('Y_test_orig', Y_test_orig)
        np.save('classes', classes)
    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test_orig.shape))
    model = train(batch_size=batchSize, epochs=numOfEpoches)

elif ch == 2:

    c = int(input("1. Test from random images\n2. Test your own custom image\n(Enter 1 or 2)\n"))
    if c == 1:

        try:
            classes = np.load('classes.npy')
            print("Loading")
            X_train = np.load('X_train.npy')
            Y_train = np.load('Y_train.npy')
            X_test = np.load('X_test.npy')
            Y_test_orig = np.load('Y_test_orig.npy')
        except:
            X_train, Y_train, _, __, classes = load_dataset()
            print("Saving...")
            np.save('X_train', X_train)
            np.save('Y_train', Y_train)
            np.save('X_test', _)
            np.save('Y_test_orig', __)
            np.save('classes', classes)

        _ = None
        __ = None
        testImgsX = []
        testImgsY = []
        ran = []
        print("X_train shape: " + str(X_train.shape))
        print("Y_train shape: " + str(Y_train.shape))
        # print(X_train[1])
        for i in range(10):
            ran.append(np.random.randint(0, X_train.shape[0] - 1))
        for ranNum in ran:
            testImgsX.append(X_train[ranNum])
            testImgsY.append(Y_train[ranNum])

        X_train = None
        Y_train = None

        print("testImgsX shape: " + str(len(testImgsX)))
        print("testImgsY shape: " + str(len(testImgsY)))
      

        cnt = 0.0

        classes = []
        classes.append("Benign")
        classes.append("InSitu")
        classes.append("Invasive")
        classes.append("Normal")

        compProbs = []
        compProbs.append(0)
        compProbs.append(0)
        compProbs.append(0)
        compProbs.append(0)

        for i in range(len(testImgsX)):
            print("\n\nTest image " + str(i + 1) + " prediction:\n")

            predi, probs = predict(testImgsX[i], modelSavePath, showImg=False)

            for j in range(len(classes)):
                print(str(classes[j]) + " : " + str(round(probs[j], 4)) + "%")
                compProbs[j] += probs[j]

            maxi = 0
            for j in range(len(testImgsY[0])):
                if testImgsY[i][j] == 1:  # The right class
                    maxi = j
                    break
            if predi == maxi:
                cnt += 1

        print("% of images that are correct: " + str((cnt / len(testImgsX)) * 100))

    elif c == 2:
        predictImage()

else:
    print("Please enter only 1 or 2")
