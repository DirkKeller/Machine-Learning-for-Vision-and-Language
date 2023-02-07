import numpy as np
import math
import random
import statistics as stat
from skimage.util.shape import view_as_windows

# EXERCISE 3: Convolution operation
def convolve2D(InputStack, KernelStack, padding=0, strides=1):

    if isinstance(InputStack, list) and isinstance(KernelStack, list):
        pass
    else:    
        raise Exception("Input or kernel arguments are not list objects!")
     
    #empty feature map list
    FeatureMapStack = []
            
    # loop over each image, Kernel and color channel
    for IdxI, Input in enumerate(InputStack):
        for IdxK, Kernel in enumerate(KernelStack):
            
            # Gather Shapes of Kernel + Image + Padding
            cKernShape, xKernShape, yKernShape = (Kernel.shape[0], Kernel.shape[1], Kernel.shape[2])
            cInputShape, xInputShape, yInputShape = (Input.shape[0], Input.shape[1], Input.shape[2])

            # Shape of Output Convolution
            xOutput = int(((xInputShape - xKernShape + 2 * padding) / strides) + 1)
            yOutput = int(((yInputShape - yKernShape + 2 * padding) / strides) + 1)
            """
            try:
                ColorInput, ColorKernel = (Input.shape[0], Kernel.shape[0])
                
                if ColorInput != ColorKernel:
                    raise Exception("Image and Kernel color channels do not match!")
            except: 
                ColorInput = 1
            """
           
            for c in range(cInputShape):
                        
                # Apply Equal Padding to All Sides
                if padding != 0:
                    InputPadded = np.zeros((xInputShape + padding*2, yInputShape + padding*2))
                    InputPadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = Input
                    print(InputPadded)
                else:
                    InputPadded = Input
           
                vectorized_Input = MemStr_Vectorize(InputPadded[c], Kernel[c])
                FeatureMap = np.dot(Kernel[c].flatten(), vectorized_Input.T)
                FeatureMap = FeatureMap.reshape(xOutput, yOutput)               

                FeatureMapStack.append(FeatureMap)

    return FeatureMapStack

# V2 padding = 1, stride = 1
def MemStr_Vectorize(Input, Kernel):
    
    if Input.shape[0]<Kernel.shape[0] or Input.shape[1] < Kernel.shape[1]:
        raise Exception("Kernel dimensions exceed image dimensions")
        
    xOutShape, yOutShape = ((Input.shape[0] - Kernel.shape[0] + 1) * (Input.shape[1] - Kernel.shape[1] + 1), 
                            Kernel.shape[0] * Kernel.shape[1])
    
    return view_as_windows(Input, Kernel.shape).reshape(xOutShape, yOutShape)

# EXERCISE 3: ReLu operation
def activationReLu (FeatureMapStack, Maximal):
    
    for idx, FeatureMap in enumerate(FeatureMapStack):
        FeatureMap[FeatureMap > Maximal] = Maximal
        FeatureMap[FeatureMap < 0] = 0

    return (FeatureMapStack)
"""
def maxPool (Input, pool_size):
    num_feature_maps = Input.shape[0]
    map_w = 1 + Input.shape[1] - pool_size[0]
    map_h = 1 + Input.shape[2] - pool_size[1]

    Input_dims = len(Input[:].shape) - 1
    
    if Input_dims == 3:
        num_color = Input.shape[3]
        
        pooled_map = np.full((num_feature_maps, 
                              map_w, map_h, 
                              num_color), 
                             fill_value = None)
                
        # loop over each filter
        for f in range(num_feature_maps):
            for c in range(num_color):            
                for w in range(map_w):
                    for h in range(map_h):
                        #print("h", h)
                        pool_area = Input[f, 
                                          w:w+pool_size[0], 
                                          h:h+pool_size[1], 
                                          c]
                        #print("pool_area", pool_area)
                        #print(pool_area.max())
                        pooled_map[f, w, h, c] = pool_area.max()
        return (pooled_map)
    
    elif Input_dims == 2:
        pooled_map = np.full((num_feature_maps, 
                              map_w, 
                              map_h), 
                             fill_value = None)
    
        # loop over each filter
        for f in range(len(Input[:])):
                    
            for w in range(map_w):
                for h in range(map_h):
                    #print("h", h)
                    pool_area = Input[f, 
                                      w:w+pool_size[0], 
                                      h:h+pool_size[1]]
                    pooled_map[f, w, h] = pool_area.max()
        return (pooled_map)
                
    else:
        print("Input dimensions are wrong!\n",
              "The imput has: ", Input_dims, " dimensions.\n",
              "But input should have 2 or 3 dimensions.\n")

"""
# EXERCISE 3: MaxPool
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

# EXERCISE 3: Normalization
def normalize (FeatureMapStack):
    for idx, FeatureMap in enumerate(FeatureMapStack):
        FeatureMap = (FeatureMap - stat.mean(FeatureMap.flatten())) / stat.stdev(FeatureMap.flatten())

    return (FeatureMapStack)

# EXERCISE 3:SoftMax
def softmax(output_layer):
    exp_values = [math.exp(x) for x in output_layer]
    sum_of_ev = sum(exp_values)
    probabilities = [math.exp(x)/sum_of_ev for x in output_layer]
    return probabilities

random.seed(0)

# 3 dimensional/RGB
img_rgb = np.random.rand(3,5,5)
kernel_rgb = np.random.rand(3,3,3)

# 2 dimensional/greyscale
img_gs = np.random.rand(1,5,5)
kernel_gs = np.random.rand(1,3,3)

ImageStack = [img_rgb]
KernelStack = [kernel_rgb]

from time import time
begin = time()                           
output = convolve2D(ImageStack, KernelStack, 0, 1) 
#print("convolve2D", output)
end = time()
# total time taken
print(f"Total runtime of the program is {end - begin}")

output = activationReLu(output, 3)
print("ReLu", output)
output = normalize(output)
print("Normalize", output)
#output = MaxPool(output, (2,2))
#print("MaxPool", output)

maximum = 8 # 8 is default
pool_size = (3,3) # (3,3) is default