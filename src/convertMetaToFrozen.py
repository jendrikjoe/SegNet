'''
Created on Aug 21, 2017

@author: jendrik
'''

import tensorflow as tf
from tensorflow.python.framework import graph_util
from Network.Network import SegNetInference
from Data.Datasets import CityScapes

if __name__ == '__main__':
    sess = tf.Session()
    dataset = CityScapes('../data/CityScapes', numberOfClasses=0, samplesPerBatch=1)
    image = dataset.generatorAllClasses(1, False).__next__()[0]['inputImg']
    globalStep = tf.Variable(0, name='globalStep')
    inputShape = image.shape
    learningRate = tf.train.exponential_decay(
                    1e-4,                # Base learning rate.
                    globalStep,  # Current index into the dataset.
                    1024,          # Decay step.
                    (1.-0.0005),                # Decay rate.
                    staircase=True) 
    segNet = SegNetInference(inputShape, dataset.numberOfClasses, 
                             learningRate, 0)
    previous_variables = [
      var_name for var_name, _
      in tf.contrib.framework.list_variables('./ckpt/segNet.ckpt')]
    print(previous_variables)
    restore_map = {variable.op.name:variable for variable in tf.global_variables()
                   if variable.op.name in previous_variables}
    print(restore_map)
    tf.contrib.framework.init_from_checkpoint(
        './ckpt/segNet.ckpt', restore_map)
    init = tf.global_variables_initializer()
    sess.run(init)
    #tf.train.write_graph(sess.graph, '../ckpt', 'toFreeze.pb', as_text=False)

    
    
    # get graph definition
    gd = sess.graph.as_graph_def()
    
    # fix batch norm nodes
    for node in gd.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
    
    # generate protobuf
    converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, ["output"])
    tf.train.write_graph(converted_graph_def, 'ckpt', 'frozenSegNet.pb', as_text=False)
    
    
    
    
    
    
    
    
    