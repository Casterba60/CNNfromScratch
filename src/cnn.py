from tensorflow.keras.datasets import mnist
from conv import Conv3x3

#mnist dataset has all handwritten digits
(x_train, y_train),(x_test, y_test) = mnist.load_data()

conv = Conv3x3(8)
output = conv.forward(x_train[0])
print(output.shape)
