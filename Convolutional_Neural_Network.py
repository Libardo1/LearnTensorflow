# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 19:33:05 2016

@author: me(at)liusida.com
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle


# How many training samples used one step
batch_size = 16
# What's the weight and height in pixel of each training sample
image_size = 28
# RGB image using 4 channels, and grey image just use one.
channels = 1
# How many different Letters (Classes) should classified. (A-J)
num_labels = 10
# What's the weight and height of the filter, for every convolution
filter_size = 5
# How deep is the convolution result
depth = 16
# What's the H of the 3rd layer
num_3rd_H = 16
# the Step Size for Gradient Descent Optimizer
step_size = 0.1
# Total Training Steps
max_train_step = 200000 / 16
train_step = 500
#train_step = max_train_step 

# Step 1. Read Data into Memory
print('Opening Data File...')
with open('notMNIST.pickle', 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels_original = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels_original = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels_original = save['test_labels']
    del save

    # Reshape the data into a more proper format
    # For dataset is simply adding the channel dimension
    train_dataset = train_dataset.reshape(-1, image_size, image_size, channels).astype('float32')
    valid_dataset = valid_dataset.reshape(-1, image_size, image_size, channels).astype('float32')
    test_dataset = test_dataset.reshape(-1, image_size, image_size, channels).astype('float32')
    # For label, we need to map the result into a vector
    #  in the pickle file, 0 = A, 1 = B ... 
    #  we need to map to [1,0,0,...,0] = A, [0,1,0,...,0] = B ...
    def remap_labels( data ):
        a = np.arange(num_labels)
        b = a==data[:,None]
        c = b.astype('float32')
        return c
    train_labels = remap_labels(train_labels_original)
    valid_labels = remap_labels(valid_labels_original)
    test_labels = remap_labels(test_labels_original)


# Step 2. Build a Convolutional Neural Network Model
g = tf.Graph()
with g.as_default():
#
#    batch * 
#       28 * 28 * 1 
#        ==1stLayerConv==> 14 * 14 * 16 
#         ==2ndLayerConv==> 7 * 7 * 16 
#          ==reshape==> 1-d
#           ==3rdLayerMultiply==> num_3rd_H
#            ==4thLayerMultiply==> 10
#
    # Input Data    
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, channels), name='train_dataset')
    tf_train_labels  = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='train_labels')
    tf_valid_dataset = tf.constant(valid_dataset, name='valid_dataset')
    tf_test_dataset  = tf.constant(test_dataset, name='test_dataset')
    
    # Variables
    layer1_weights = tf.Variable( tf.truncated_normal([filter_size, filter_size, channels, depth],stddev=0.1), name='W1' )
    layer1_biases = tf.Variable( tf.zeros([depth]), name='b1' )
    
    layer2_weights = tf.Variable( tf.truncated_normal([filter_size, filter_size, depth, depth],stddev=0.1), name='W2' )
    layer2_biases = tf.Variable( tf.constant(1.0, shape=[depth]), name='b2' )
    
    layer3_weights = tf.Variable( tf.truncated_normal([image_size*image_size*depth/16, num_3rd_H],stddev=0.1), name='W3' )
    layer3_biases = tf.Variable( tf.constant(1.0, shape=[num_3rd_H]), name='b3' )
    
    layer4_weights = tf.Variable( tf.truncated_normal([num_3rd_H, num_labels],stddev=0.1), name='W4' )
    layer4_biases = tf.Variable( tf.constant(1.0, shape=[num_labels]), name='b4' )

    # Model
    def Model(data, name=None):
        if name==None:
            prefix = ""
        else:
            prefix = name+"_"
        #1st Layer: conv
        conv = tf.nn.conv2d(data, layer1_weights, [1,2,2,1], padding='SAME', name=prefix+'conv1')
        layer_output = tf.nn.relu(conv+layer1_biases, name=prefix+'relu1')
        #2nd Layer: conv
        conv = tf.nn.conv2d(layer_output, layer2_weights, [1,2,2,1], padding='SAME', name=prefix+'conv2')
        layer_output = tf.nn.relu(conv+layer2_biases, name=prefix+'relu2')
        # Because Conv use strides of 2,2
        #   the output image is now 1/4 of original image_size after conv twice.
        # We need to reuse this function to deal with all valid data and test data,
        #   So we calculate how many images coming in first using shape[0]
        #3rd Layer: multiply
        shape = tf.shape(layer_output)
        reshape = tf.reshape(layer_output, [shape[0], image_size*image_size*depth/16])
        matmul = tf.matmul(reshape, layer3_weights, name=prefix+'matmul3')
        layer_output = tf.nn.relu(matmul+layer3_biases, name=prefix+'relu3')
        #4th Layer: multiply
        matmul = tf.matmul(layer_output, layer4_weights, name=prefix+'matmul4')
        layer_output = matmul + layer4_biases
        return layer_output
        
        
    logits = Model(tf_train_dataset, name='Train')
    loss = tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels, name='Softmax_CrossEntropy')
    loss = tf.reduce_mean(loss)
    
    # Optimizing Operation    
    optimizer = tf.train.GradientDescentOptimizer(step_size).minimize(loss)
    
    # Predicting
    tf_train_predict = tf.nn.softmax(logits)
    tf_valid_predict = tf.nn.softmax( Model(tf_valid_dataset) )
    tf_test_predict = tf.nn.softmax( Model(tf_test_dataset) )
    
    
# Step 3. Run Training Process
with tf.Session(graph=g) as ss:
    # calculate the accuracy for the result
    def accuracy( prediction, labels_original ):
        indices = np.argmax(prediction,1)
        matches = indices==labels_original
        return 100.0 * sum(matches) / indices.shape[0]

    print("Initializing Variables...")
    tf.initialize_all_variables().run()
    for step in range(train_step):
        offset = (step*batch_size)
        shape = train_dataset.shape
        if offset>=shape[0]:
            print("No Enough Data! Please reduce your train_step or batch_size.")
            break
        feed_dataset = train_dataset[offset:(offset+batch_size),:,:,:]
        feed_labels = train_labels[offset:(offset+batch_size),:]
        feed_labels_original = train_labels_original[offset:(offset+batch_size)]
        feed_dict = {tf_train_dataset: feed_dataset, tf_train_labels: feed_labels}
        
        _,l,p = ss.run([optimizer, loss, tf_train_predict], feed_dict=feed_dict)
        if (step%500==0):
            #print("Offset = %d, step = %d, shape[0] = %d" % (offset, step, shape[0]))
            print("Step %d:" % step)
            print("  Batch Loss = %.1f" % l)
            print("  Train Accuracy = %.1f%%" % accuracy(p,feed_labels_original))
            print("  Valid Accuracy = %.1f%%" % accuracy(tf_valid_predict.eval(),valid_labels_original))
    

# Step 4. Predicting 

    # Let draw out some images which the program make mistakes.
    # To see if we human will make same mistakes too. :)
    test_prediction = tf_test_predict.eval()
    test_prediction_original = np.argmax(test_prediction,1)
    WrongIndices = np.where(test_prediction_original != test_labels_original)
    Letter = 'ABCDEFGHIJ'
    # Randomly pick some images
    
    np.random.shuffle(WrongIndices[0])
    for i in range(10):
        wrongPicture = WrongIndices[0][i]
        img = test_dataset[wrongPicture,:,:,:]
        img = img.reshape(28,28)
        print("Mistake Image Index = %d:\n Label = %s, but Predict = %s" % (wrongPicture,Letter[test_labels_original[wrongPicture]],Letter[test_prediction_original[wrongPicture]]))
        plt.imshow(img, cmap='gray')
        plt.show()
        
    # Report the Final Score
    print("\n\nFinal Test Accuracy = %.1f%%" % accuracy(test_prediction, test_labels_original))        

