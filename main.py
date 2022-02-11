import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential

# Load the digits dataset
digits = datasets.load_digits()
x = digits.data
y = digits.target
x, y = shuffle(x, y)

# normalize the data
num_class = len(digits.target_names)
x = StandardScaler().fit_transform(x)

# shuffle and split the train set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)

# The training data for CNN
x_train_cnn = np.reshape(x_train, (x_train.shape[0], 8, 8))  # Convert to a 8*8 matrix
x_train_cnn = np.expand_dims(x_train_cnn, axis=3)  # add 1 dimension
x_test_cnn = np.expand_dims(np.reshape(x_test, (x_test.shape[0], 8, 8)), axis=3)

# k value for KNN
k = 5

# The parameter for nn and cnn
# optimizer
batch = 10
epoch = 15
# dnn
hidden = 128  # hidden layer size
# cnn
k_size = (3, 3)  # kernel
num_filter = 64  # convolutional filter num
max_pool = (2, 2)  # max_pool size
# drop_out rate
dropout_rate = 0.3

# Cross validation fold
fold = 5


# Calculate the score through compare prediction and target
def own_accu_score(y_true, y_predict):
    diff = 0
    length = len(y_true)
    for index in range(length):
        if y_true[index] != y_predict[index]:
            diff += 1
    return 1 - float(diff / length)


# Own confusion matrix for check the performance of ML algorithm
def own_confusion_matrix(y_true1, y_predict1):
    unique1 = np.unique(y_true1)
    matrix = np.zeros([len(unique1), len(unique1)], dtype=int)
    for i in range(len(y_true1)):
        matrix[y_true1[i]][y_predict1[i]] += 1
    return matrix


# Draw the confusion matrix by pyplot
def draw_cm(matrix, title):
    import warnings
    warnings.filterwarnings('ignore')
    plt.figure()
    # Draw the matrix
    plt.matshow(matrix)
    for xis in range(len(matrix)):
        for yis in range(len(matrix)):
            # Draw the text
            if xis == yis:
                plt.annotate(matrix[xis, yis], xy=(xis, yis), color='k', horizontalalignment='center',
                             verticalalignment='center')
            else:
                plt.annotate(matrix[xis, yis], xy=(xis, yis), color='w', horizontalalignment='center',
                             verticalalignment='center')
    plt.title('confusion matrix of: ' + title)
    plt.show()


# My Own Knn_classifier algorithm
class OwnKnnClassifier:
    # initialize with k = 4
    def __init__(self, key=4):
        self.key = key
        self.xtr, self.ytr = None, None

    def fit(self, xtr, ytr):
        self.xtr = xtr
        self.ytr = ytr

    # to predict the digits
    def predict(self, x_test1, p=1):
        num = x_test1.shape[0]
        y_predict = np.zeros(num, dtype=self.ytr.dtype)  # Use to store the prediction result
        for i in range(num):
            # Calculate the distance based on p, p = 1 Manhattan, p = 2 euclidean distance
            distances = np.sum(np.abs(x_test1[i] - self.xtr) ** p, axis=1) ** (1 / p)
            # Sort and get the index of first key distance
            sorted_distances = np.argsort(distances)[:self.key]
            # Accumulate the count for each possible prediction label
            accumulate = np.zeros(len(np.unique(self.ytr)))
            for z in sorted_distances:
                # ytr[z] = the prediction result for one result corresponding k_value
                accumulate[self.ytr[z]] += 1
            # Check the index of the most possible prediction, index = the prediction result
            y_predict[i] = np.argmax(accumulate)
        return y_predict

    def proba(self, x_test1):  # calculate the confidence
        proba = []
        num = x_test1.shape[0]
        class_num = len(np.unique(self.ytr))
        y_predict = self.predict(x_test1)
        for i in range(num):
            res = np.zeros(class_num)
            res[y_predict[i]] = 1
            proba.append(res)
        return proba


# Convert the result form output layer to a specific result
def convert_target(pred):
    num = pred.shape[0]  # number of result
    target = []
    for i in range(num):
        a = np.argmax(pred[i])  # the layer with biggest weight
        target.append(a)
    return target


# DNN model, return a not trained model with input parameters
def dnn_model(hidden_size, drop_rate):
    network = Sequential()
    network.add(Dense(hidden_size, activation='relu'))
    network.add(Dropout(rate=drop_rate))
    network.add(Dense(hidden_size, activation='relu'))
    network.add(Dropout(rate=drop_rate))
    network.add(Dense(num_class, activation='softmax'))  # Classify function
    return network


k_size = (3, 3)  # kernel
num_filter = 64  # convolutional filter num
max_pool = (2, 2)  # max_pool size


# DNN model, return a not trained model with input parameters
def cnn_model(filter_num, kernel_size, max_pool_size, drop_rate):
    network = Sequential()
    # First convolutional layer with a input data type
    network.add(
        Conv2D(filters=filter_num, kernel_size=kernel_size, activation='relu', padding='SAME', input_shape=(8, 8, 1)))
    network.add(MaxPool2D(max_pool_size))
    network.add(Dropout(drop_rate))  # To prevent over-fitting
    network.add(Conv2D(filters=filter_num, kernel_size=kernel_size, activation='relu', padding='SAME'))
    network.add(Dropout(drop_rate))  # To prevent over-fitting
    network.add(MaxPool2D(max_pool_size))
    network.add(Flatten())  # Flat the output
    network.add(Dense(num_class, activation='softmax'))
    return network


# The function to train and save a DNN model
def train_dnn():
    print('###DNN Model###')
    model = dnn_model(hidden_size=hidden, drop_rate=dropout_rate)
    model.compile(optimizer='adam',  # Use adm optimizer
                  loss='sparse_categorical_crossentropy',  # loss function
                  metrics=['acc'])
    model.fit(x_train, y_train, epochs=epoch, batch_size=batch)  # train
    model.predict(x_test)
    with open('DNN_model.pickle', 'wb') as fw:
        pickle.dump(model, fw)


# The function to train and save a CNN model
def train_cnn():
    print('###CNN Model###')
    network = cnn_model(drop_rate=dropout_rate, kernel_size=k_size, filter_num=num_filter, max_pool_size=max_pool)
    network.compile(optimizer='adam',  # Use adm optimizer
                    loss='sparse_categorical_crossentropy',  # loss function
                    metrics=['acc'])
    network.fit(x_train_cnn, y_train, epochs=epoch, batch_size=batch)
    with open('CNN_model.pickle', 'wb') as fw:
        pickle.dump(network, fw)


def f3_nn_cm():
    # load the saved model
    with open('DNN_model.pickle', 'rb') as d:
        dnn = pickle.load(d)
    with open('CNN_model.pickle', 'rb') as c:
        cnn = pickle.load(c)
    y_dnn = convert_target(dnn.predict(x_test))
    y_cnn = convert_target(cnn.predict(x_test_cnn))
    dmatrix = own_confusion_matrix(y_test, y_dnn)
    cmatrix = own_confusion_matrix(y_test, y_cnn)
    print('Testing accuracy of DNN model:', own_accu_score(y_test, y_dnn))
    print('Confusion matrix of DNN model: \n', dmatrix, '\n')
    draw_cm(dmatrix, 'DNN model')
    print('Testing accuracy of CNN model:', own_accu_score(y_test, y_cnn))
    print('Confusion matrix of CNN model: \n', cmatrix)
    draw_cm(cmatrix, 'CNN model')
    print('\n')


def knn_confusion_matrix():
    # Draw knn confusion matrixdef knn_confusion_matrix():
    scikit_knn, own_knn = train_and_save_knn()
    y_skl = scikit_knn.predict(x_test)
    skl_cm = own_confusion_matrix(y_test, y_skl)
    print('Testing accuracy of Scikit-learn KNN model:', own_accu_score(y_test, y_skl))
    print('Confusion matrix of Scikit-learn KNN model: \n', skl_cm, '\n')
    draw_cm(skl_cm, 'Scikit-learn KNN model')
    y_oknn = own_knn.predict(x_test)
    own_cm = own_confusion_matrix(y_test, y_oknn)
    print('Testing accuracy of Own KNN model:', own_accu_score(y_test, y_oknn))
    print('Confusion matrix of Own KNN model: \n', own_cm)
    draw_cm(own_cm, 'Own KNN model')
    print('\n')


def f1_train_and_save_nn():
    train_dnn()
    train_cnn()


# an iteration for data
def k_fold(fold_num, _y):
    length = len(_y)
    result = np.arange(length)
    index = 0
    fold_size = (length // fold_num) * np.ones(fold_num, dtype=int)  # The remainder based on fold
    fold_size[:length % fold_num] += 1  # Calculate the size
    for i in fold_size:
        first, end = index, i  # The range of test data
        train_dex = list(np.concatenate((result[:first], result[end:])))
        test_dex = list(result[first:end])
        yield train_dex, test_dex
        first = end  # move to next iteration


def cross_validation_dnn():
    print('Own DNN cross validation\n')
    index = 1
    kf = k_fold(fold, y)
    model = dnn_model(hidden_size=hidden, drop_rate=dropout_rate)
    pre = []
    for train_dex, test_dex in kf:  # k-fold
        train_x, test_x = x[train_dex], x[test_dex]
        train_y, test_y = y[train_dex], y[test_dex]
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        model.fit(train_x, train_y, epochs=epoch, batch_size=batch)  # train
        result = model.predict(test_x)
        result = convert_target(result)
        print('Training count:', index, 'Testing accuracy:', own_accu_score(test_y, result))
        index += 1
        pre.append(own_accu_score(test_y, result))
    pre = np.array(pre)
    with open('DNN_model.pickle', 'wb') as fw:
        pickle.dump(model, fw)
    print('The average prediction accuracy is', pre.mean())
    print('\n')


def cross_validation_cnn():
    print('Own CNN cross validation\n')
    index = 1
    pre = []
    kf = k_fold(fold, y)
    model = cnn_model(filter_num=num_filter, max_pool_size=max_pool, drop_rate=dropout_rate, kernel_size=k_size)
    for train_dex, test_dex in kf:
        train_x, test_x = x[train_dex], x[test_dex]
        train_y, test_y = y[train_dex], y[test_dex]
        train_x = np.expand_dims(np.reshape(train_x, (train_x.shape[0], 8, 8)), axis=3)
        test_x = np.expand_dims(np.reshape(test_x, (test_x.shape[0], 8, 8)), axis=3)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        model.fit(train_x, train_y, epochs=epoch, batch_size=batch)  # train
        result = model.predict(test_x)
        result = convert_target(result)
        print('Training count:', index, 'Testing accuracy:', own_accu_score(test_y, result))
        index += 1
        pre.append(own_accu_score(test_y, result))
    pre = np.array(pre)
    with open('CNN_model.pickle', 'wb') as fw:
        pickle.dump(model, fw)
    print('The average prediction accuracy is', pre.mean())
    print('\n')


# Train and test the algorithm from scikit-learn and save model
def train_and_save_knn():
    print('###Scikit-learn Classifier Model###')
    knn_classifier = KNeighborsClassifier(n_neighbors=k)  # k=3
    knn_classifier.fit(x_train, y_train)  # to fit the KNN-classifier
    y_predict_skl = knn_classifier.predict_proba(x_test)
    y_predict_skl = convert_target(y_predict_skl)
    print('Learning accuracy:', own_accu_score(y_test, y_predict_skl))
    print('###Own Knn Classifier Model###')
    own_knn_classifier = OwnKnnClassifier(k)  # k=3
    own_knn_classifier.fit(x_train, y_train)
    y_predict_own = own_knn_classifier.predict(x_test)  # predict the digits using knn_classifier
    score2 = own_accu_score(y_test, y_predict_own)
    print('Learning accuracy:', score2)
    return knn_classifier, own_knn_classifier


def cross_validation_skl_knn():
    index = 1
    pre = []
    kf = k_fold(fold, y)
    print('Skl-knn cross validation\n')
    knn_classifier = KNeighborsClassifier(n_neighbors=k)  # k=3
    for train_dex, test_dex in kf:  # k-fold
        train_x, test_x = x[train_dex], x[test_dex]
        train_y, test_y = y[train_dex], y[test_dex]
        knn_classifier.fit(train_x, train_y)  # to fit the KNN-classifier
        result = knn_classifier.predict(test_x)
        print('Training count:', index, 'Testing accuracy:', own_accu_score(test_y, result), '\n')
        index += 1
        pre.append(own_accu_score(test_y, result))
    pre = np.array(pre)
    print('The average prediction accuracy is', pre.mean())
    print('\n')


def cross_validation_own_knn():
    pre = []
    index = 1
    kf = k_fold(fold, y)
    print('Own-knn cross validation\n')
    own_knn = OwnKnnClassifier(key=3)
    for train_dex, test_dex in kf:  # k-fold
        train_x, test_x = x[train_dex], x[test_dex]
        train_y, test_y = y[train_dex], y[test_dex]
        own_knn.fit(x_train, y_train)
        result = own_knn.predict(test_x)
        print('Training count:', index, 'Testing accuracy:', own_accu_score(test_y, result), '\n')
        index += 1
        pre.append(own_accu_score(test_y, result))
    pre = np.array(pre)
    print('The average prediction accuracy is', pre.mean())


def roc_curve(y_true, y_predict, name):
    tpr_list = []
    fpr_list = []
    for i in range(num_class):
        ac_neg, ac_pos = 0, 0  # actual negative and positive
        confidence = np.zeros(len(y_true))  # possibility
        label = np.zeros(len(y_true))  # is true or not
        for j in range(len(y_true)):
            if y_true[j] == i:
                ac_pos += 1
                label[j] = 1  # If is true, change to one
            else:
                ac_neg += 1
            confidence[j] = y_predict[j][i]  # The probability it is label i
        index = np.argsort(confidence)[::-1][:len(y_true)]  # the index of sort in desc
        tpr = []
        fpr = []
        for j in range(len(y_true)):
            threshold = confidence[index[j]]
            tn, fp, fn, tp = 0, 0, 0, 0
            for idx in range(len(y_true)):  # Confusion matrix
                if label[idx] == 0 and confidence[idx] < threshold:
                    tn += 1  # TN value
                if label[idx] == 0 and confidence[idx] >= threshold:
                    fp += 1  # FP
                if label[idx] == 1 and confidence[idx] < threshold:
                    fn += 1  # FN value
                if label[idx] == 1 and confidence[idx] >= threshold:
                    tp += 1  # TP'''
            recall = (tp / ac_pos)  # TPR for point
            facall = (fp / ac_neg)  # FPR for point
            tpr.append(recall)
            fpr.append(facall)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    # Draw
    plt.figure()
    for i in range(num_class):
        curve_name = 'Class ' + str(i) + ' vs other class'
        plt.plot(fpr_list[i], tpr_list[i],
                 label=curve_name,
                 linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.legend(loc="lower right")
    plt.show()


def roc_for_tow_nn_model():
    with open('DNN_model.pickle', 'rb') as d:  # load
        dnn = pickle.load(d)
    y_predict_d = dnn.predict(x_test)
    roc_curve(y_test, y_predict_d, 'ROC Curve for DNN_model')
    with open('CNN_model.pickle', 'rb') as d:
        cnn = pickle.load(d)
    y_predict_c = cnn.predict(x_test_cnn)
    roc_curve(y_test, y_predict_c, 'ROC Curve for CNN_model')


def roc_for_tow_knn():
    scikit_knn, own_knn = train_and_save_knn()
    ysk = scikit_knn.predict_proba(x_test)
    roc_curve(y_test, ysk, 'ROC Curve for skl-KNN_model')
    y_own = own_knn.proba(x_test)
    roc_curve(y_test, y_own, 'ROC Curve for Own-KNN_model')


def load_and_test():
    with open('DNN_model.pickle', 'rb') as d:  # load
        dnn = pickle.load(d)
    y_predict_d = convert_target(dnn.predict(x_test))
    with open('CNN_model.pickle', 'rb') as d:
        cnn = pickle.load(d)
    y_predict_c = convert_target(cnn.predict(x_test_cnn))
    print('Testing accuracy for dnn model is: ', own_accu_score(y_test, y_predict_d))
    print('Testing accuracy for cnn model is: ', own_accu_score(y_test, y_predict_c))


def switch_mode():
    mode = True
    while mode:
        print("Enter a command: \n"
              "1 = Train two neural network and two knn and save the model\n"
              "2 = Cross Validation for DNN\n"
              "3 = Cross Validation for CNN\n"
              "4 = Cross Validation for 2 KNN\n"
              "5 = Draw confusion matrix for two neural network model\n"
              "6 = Draw confusion matrix for two KNN algorithms\n"
              "7 - Draw roc-curve for two NN model\n"
              "8 = Draw roc-curve for two KNN model\n"
              "9 = Load and test two models\n"
              "E = Exit from the program")
        str1 = input().upper()  # Lower or upper case is not important
        if str1 == '1':
            f1_train_and_save_nn()
        elif str1 == '2':
            cross_validation_dnn()
        elif str1 == '3':
            cross_validation_cnn()
        elif str1 == '4':
            cross_validation_skl_knn()
            cross_validation_own_knn()
        elif str1 == '5':
            print('Please Wait...')
            f3_nn_cm()
        elif str1 == '6':
            knn_confusion_matrix()
        elif str1 == '7':
            print('Please Wait...')
            roc_for_tow_nn_model()
        elif str1 == '8':
            roc_for_tow_knn()
        elif str1 == '9':
            load_and_test()
        elif str1 == 'E':
            break
        else:
            print('Please re-entre the correct command')


switch_mode()
