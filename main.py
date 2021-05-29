import numpy as np
import matplotlib.pyplot as plt1

# Creating data set

# A
a = [0, 0, 1, 1, 0, 0,
     0, 1, 0, 0, 1, 0,
     1, 1, 1, 1, 1, 1,
     1, 0, 0, 0, 0, 1,
     1, 0, 0, 0, 0, 1]
# B
b = [0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 1, 0]
# C
c = [0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 1, 1, 1, 0]

# Creating labels
y_small = [[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]]

letters_vector = [np.array(a).reshape(1, 30), np.array(b).reshape(1, 30),
                  np.array(c).reshape(1, 30)]

labels_vector = np.array(y_small)


# print(letters_vector, "\n\n", labels_vector)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 1 Input layer(1, 30)
# 1 hidden layer (1, 5)
# 1 output layer(3, 3)

def f_forward(data, w1, w2):
    # hidden layer
    input_l2 = data.dot(w1)  # input from layer 1
    output_l2 = sigmoid(input_l2)  # out put of layer 2

    # Output layer
    input_out_layer = output_l2.dot(w2)  # input of out layer
    output = sigmoid(input_out_layer)  # output of out layer
    return output


# initializing the weights randomly
def generate_wt(columns, rows):
    weights_mat = []
    for i in range(columns * rows):
        weights_mat.append(np.random.randn())
    return np.array(weights_mat).reshape(columns, rows)


# Using mean square error(MSE)
def loss_function(out, Y_caps):
    mse = (np.square(out - Y_caps))
    mse = np.sum(mse) / len(labels_vector)
    return mse


# Back propagation of error
def back_prop(data_cont, label_array, w1, w2, learning_rate):
    # hidden layer
    input_l2 = data_cont.dot(w1)
    output_l2 = sigmoid(input_l2)

    # Output layer
    input_out_layer = output_l2.dot(w2)
    output = sigmoid(input_out_layer)

    # error in output layer
    d2 = (output - label_array)
    d1 = np.multiply((w2.dot((d2.transpose()))).transpose(),
                     (np.multiply(output_l2, 1 - output_l2)))

    # Gradient for w1 and w2
    w1_adj = data_cont.transpose().dot(d1)
    w2_adj = output_l2.transpose().dot(d2)

    # Updating parameters
    w1 = w1 - (learning_rate * w1_adj)
    w2 = w2 - (learning_rate * w2_adj)

    return w1, w2


def train(letters, labels, w1, w2, learning_rate=0.01, epoch=10):
    accuracy_list = []
    loss = []
    for j in range(epoch):
        l = []
        for i in range(len(letters)):
            out = f_forward(letters[i], w1, w2)
            l.append((loss_function(out, labels[i])))
            w1, w2 = back_prop(letters[i], labels[i], w1, w2, learning_rate)
        print("epochs:", j + 1, "======== acc:", (1 - (sum(l) / len(letters))) * 100)
        accuracy_list.append((1 - (sum(l) / len(letters))) * 100)
        loss.append(sum(l) / len(letters))
    return accuracy_list, loss, w1, w2


def predict(letter, w1, w2):
    output = f_forward(letter, w1, w2)
    maximum = 0
    k = 0
    for i in range(len(output[0])):
        if maximum < output[0][i]:
            maximum = output[0][i]
            k = i
    if k == 0:
        print("Image is of letter A.")
    elif k == 1:
        print("Image is of letter B.")
    else:
        print("Image is of letter C.")
    print(output)
    plt1.imshow(letter.reshape(5, 6))
    plt1.show()


weight_mat1 = generate_wt(30, 5)
weight_mat2 = generate_wt(5, 3)
# print(w1, "\n\n", w2)

accuracy, loss, trained_weight1, trained_weight2 = train(letters_vector, labels_vector, weight_mat1, weight_mat2, 0.1, 100)

# # plotting accuracy
# plt1.plot(accuracy)
# plt1.ylabel('Accuracy')
# plt1.xlabel("Epochs:")
# plt1.show()
#
# # plotting Loss
# plt1.plot(loss)
# plt1.ylabel('Loss')
# plt1.xlabel("Epochs:")
# plt1.show()

predict(letters_vector[2], trained_weight1, trained_weight2)
