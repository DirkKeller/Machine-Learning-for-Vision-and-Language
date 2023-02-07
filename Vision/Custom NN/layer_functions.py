import numpy as np

from skimage.util.shape import view_as_windows
import math

def conv_op_3d(input, kernels):
    featureMaps = []
    inp = np.moveaxis(input, -1, 0)

    for z, kernel in enumerate(kernels):
        xKernShape, yKernShape, _ = kernel.shape
        xImgShape, yImgShape, cImgShape = input.shape
        xOutput = int((xImgShape - xKernShape) + 1)
        yOutput = int((yImgShape - yKernShape) + 1)

        kern = np.moveaxis(kernel, -1, 0)
        colorMaps = []
        for c in range(cImgShape):
            vectorized_Input = MemStr_Vectorize(inp[c], kern[c])
            featureMap = np.dot(kern[c].flatten(), vectorized_Input.T)
            featureMap = featureMap.reshape(xOutput, yOutput)
            colorMaps.append(featureMap)

        featureMaps.append(np.array(colorMaps).sum(axis=0))

    return np.moveaxis(np.array(featureMaps), 0, 2)


def MemStr_Vectorize(Input, Kernel):
    if Input.shape[0] < Kernel.shape[0] or Input.shape[1] < Kernel.shape[1]:
        raise Exception("Kernel dimensions exceed image dimensions")

    xOutShape, yOutShape = ((Input.shape[0] - Kernel.shape[0] + 1) * (Input.shape[1] - Kernel.shape[1] + 1),
                            Kernel.shape[0] * Kernel.shape[1])

    return view_as_windows(Input, Kernel.shape).reshape(xOutShape, yOutShape)


# EXERCISE 3: ReLu operation
def relu(output):
    output[output < 0] = 0
    return output

def relu_3d(input):
    for i in range(input.shape[2]):
        input[:, :, i] = relu(input[:, :, i])
    return input


def maxPool(Input, pool_size):
    pooled_map = np.full((int((Input.shape[0] / pool_size[0])), int((Input.shape[1] / pool_size[1])),
                          Input.shape[2]),
                         fill_value=None)
    for f in range(Input.shape[2]):
        for w1 in range(pooled_map.shape[0]):
            for h1 in range(pooled_map.shape[1]):
                pooled_map[w1, h1, f] = Input[w1 * pool_size[0]:w1 * pool_size[0] + pool_size[0],
                                        h1 * pool_size[1]:h1 * pool_size[1] + pool_size[1], f].max()
    return pooled_map

def normalize_3d(input):
    for i in range(input.shape[2]):
        input[:, :, i] = normalize(input[:, :, i])
    return input

def normalize(Input):
    x = np.array(Input, dtype=np.float64)
    x -= np.mean(x)
    x /= np.std(x)
    return x


def fullyconnected(input, weights):
    """
    Creates a fully connected layer
    The purpose is to conntect all inputs from the previous layer to the next layer
    input ~ input stack of feature maps from previous layer
    output_size ~ the number of output nodes
    """
    # Begin by flattening stack of feature maps into a 1-dimensional matrix.
    flattened_input = input.flatten()
    # Return the matrix multiplication of the flattened input and the weight matrix
    return np.dot(flattened_input, weights)


def softmax(output_layer):
    exp_values = [math.exp(x) for x in output_layer]
    sum_of_ev = sum(exp_values)
    probabilities = [math.exp(x) / sum_of_ev for x in output_layer]
    return probabilities
