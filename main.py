import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import matplotlib.pyplot as plt

filename = "card_transdata.csv"

D = np.matrix(pd.read_csv(filename, header=None).values)

max_entries = 10000

#If using AdamOptimizer
learning_rate = 0.2; num_epochs = 1000

#If using GradientDescentOptimizer
learning_rate = 0.00000002; num_epochs = 20000


# Enter Input row(s) and output row
in0 = np.asarray(D[1:max_entries, 0])
in1 = np.asarray(D[1:max_entries, 1])
in2 = np.asarray(D[1:max_entries, 2])
in3 = np.asarray(D[1:max_entries, 3])
in4 = np.asarray(D[1:max_entries, 4])
in5 = np.asarray(D[1:max_entries, 5])
in6 = np.asarray(D[1:max_entries, 6])
out = np.asarray(D[1:max_entries, 7])

# Convert table columns into numby matrix
data_columns = np.column_stack((in0, in1, in2, in3, in4, in5, in6, out))

# shuffle entries randomly
np.random.shuffle(data_columns)

# Pick 70% of data for train & 30% for test
data_train, data_test = np.split(data_columns, [int(0.7 * len(data_columns))])

#split data into X & y
X_train, y_train = np.split(data_train, [-1], axis=1)
X_test, y_test = np.split(data_test, [-1], axis=1)

#Transpose data
X_train = np.asarray(X_train).transpose()
X_test = np.asarray(X_test).transpose()
y_train = np.asarray(y_train).transpose()
y_test = np.asarray(y_test).transpose()

# number of features and samples
n_features = X_train.shape[0]
n_samples_train = y_train.size
n_samples_test = y_test.size

# Define data placeholders
x = tf.placeholder(tf.float32, shape=(n_features, None))
y = tf.placeholder(tf.float32, shape=(1, None))

# Define trainable variables
A = tf.get_variable("A", shape=(1, n_features))
b = tf.get_variable("b", shape=())

# Define model output
y_predicted = tf.matmul(A, x) + b

# Define the loss function
L = tf.reduce_sum((y_predicted - y)**2)

# Define optimizer object
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(L)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(L)

# Create a session and initialize variables
session = tf.Session()
session.run(tf.global_variables_initializer())

# Main optimization loop
for t in range(num_epochs):
    _, current_loss, current_A, current_b = session.run([optimizer, L, A, b], feed_dict={
        x: X_train,
        y: y_train
    })
    print("t = %g, loss = %g, A = %s, b = %g" % (t, current_loss, str(current_A), current_b))


# Run trained model on test data and determine accuracy
theta = current_A[0]
bias = current_b
false_fraud = 0
missed_fraud = 0
fraud = 0
no_fraud = 0
prediction_threshold = 0.25

for i in range(n_samples_test):
    temp = []
    for j in range(n_features):
        temp.append(float(X_test.item(j,i)))

    predict = np.dot(temp,theta) + bias
    if(predict > prediction_threshold): 
        predict = 1
    else: predict = 0

    actual = float(y_test.item(i))
    if predict != actual: error = 100
    else: error = 0
    if predict == 1 and actual == 0: false_fraud +=1
    elif predict == 0 and actual == 1: missed_fraud +=1
    elif predict == 1 and actual == 1: fraud +=1
    else: no_fraud +=1
    print(f"actual: {actual} , prediction: {predict}, error: = {error}%")

error_percentage = (missed_fraud + false_fraud)/n_samples_test*100
print("avg error : {:.4f}%".format(error_percentage))
print(f"fraud: {fraud} , no_fraud: {no_fraud}, missed_fraud: = {missed_fraud}, false_fraud: = {false_fraud}")