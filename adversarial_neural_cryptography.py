# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 09:52:44 2016
Modified on Fri Mar 24
@author: liam schoneveld
@author: Andy Pitcher
Implementation of model described in 'Learning to Protect Communications with 
Adversarial Neural Cryptography' (Martín Abadi & David G. Andersen, 2016, 
https://arxiv.org/abs/1610.06918)
"""
from termcolor import colored
import theano
import texttable as tt
import re
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from layers import ConvLayer, HiddenLayer, get_all_params
from lasagne.updates import adam

#Banner

Banner= """


|_ _| \ | / ___|| ____/ /_ / / |/ _ \
 | ||  \| \___ \|  _|| '_ \| | | | | |
 | || |\  |___) | |__| (_) | | | |_| |
|___|_| \_|____/|_____\___/|_|_|\___/

		|  _ \ _ __ ___ (_) ___  ___| |_
		| |_) | '__/ _ \| |/ _ \/ __| __|
		|  __/| | | (_) | |  __/ (__| |_
		|_|   |_|  \___// |\___|\___|\__|
              		       |__/
"""


#Parameters
batch_size = 1
msg_len = 8
key_len = 8
comm_len = 8

# Set this flag to exclude convolutional layers from the networks
skip_conv = False

#stores all the parameters and tensors relavent to all of the convolutional layers in the model
class StandardConvSetup():
    '''
    Standard convolutional layers setup used by Alice, Bob and Eve.
    Input should be 4d tensor of shape (batch_size, 1, msg_len + key_len, 1)
    Output is 4d tensor of shape (batch_size, 1, msg_len, 1)
    '''
    def __init__(self, reshaped_input, name='unnamed'):
        self.name = name
        self.conv_layer1 = ConvLayer(reshaped_input,
                                     filter_shape=(2, 1, 4, 1), #num outs, num ins, size
                                     image_shape=(None, 1, None, 1),
                                     stride=(1,1),
                                     name=self.name + '_conv1',
                                     border_mode=(2,0),
                                     act_fn='relu')
        
        self.conv_layer2 = ConvLayer(self.conv_layer1, 
                                     filter_shape=(4, 2, 2, 1),
                                     image_shape=(None, 2, None, 1),
                                     stride=(2,1),
                                     name=self.name + '_conv2',
                                     border_mode=(0,0),
                                     act_fn='relu')
        
        self.conv_layer3 = ConvLayer(self.conv_layer2, 
                                     filter_shape=(4, 4, 1, 1),
                                     image_shape=(None, 4, None, 1),
                                     stride=(1,1),
                                     name=self.name + '_conv3',
                                     border_mode=(0,0),
                                     act_fn='relu')
        
        self.conv_layer4 = ConvLayer(self.conv_layer3, 
                                     filter_shape=(1, 4, 1, 1),
                                     image_shape=(None, 4, None, 1),
                                     stride=(1,1),
                                     name=self.name + '_conv4',
                                     border_mode=(0,0),
                                     act_fn='tanh')
        
        self.output = self.conv_layer4.output
        self.layers = [self.conv_layer1, self.conv_layer2, 
                       self.conv_layer3, self.conv_layer4]
        self.params = []
        for l in self.layers:
            self.params += l.params
            
# Tensor variables for the message and key
msg_in = T.matrix('msg_in')
key = T.matrix('key')

# Alice's input is the concatenation of the message and the key
alice_in = T.concatenate([msg_in, key], axis=1)

# Alice's hidden layer
alice_hid = HiddenLayer(alice_in,
                        input_size=msg_len + key_len,
                        hidden_size=msg_len + key_len,
                        name='alice_to_hid',
                        act_fn='relu')
if skip_conv:
    alice_conv = HiddenLayer(alice_hid,
                             input_size=msg_len + key_len,
                             hidden_size=msg_len,
                             name='alice_hid_to_comm',
                             act_fn='tanh')
    alice_comm = alice_conv.output
else:
    # Reshape the output of Alice's hidden layer for convolution
    alice_conv_in = alice_hid.output.reshape((batch_size, 1, msg_len + key_len, 1))
    # Alice's convolutional layers
    alice_conv = StandardConvSetup(alice_conv_in, 'alice')
    # Get the output communication
    alice_comm = alice_conv.output.reshape((batch_size, msg_len))
# Bob's input is the concatenation of Alice's communication and the key
bob_in = T.concatenate([alice_comm, key], axis=1)
# He decrypts using a hidden layer and a conv net as per Alice
bob_hid = HiddenLayer(bob_in, 
                      input_size=comm_len + key_len,
                      hidden_size=comm_len + key_len,
                      name='bob_to_hid',
                      act_fn='relu')
if skip_conv:
    bob_conv = HiddenLayer(bob_hid,
                           input_size=comm_len + key_len,
                           hidden_size=msg_len,
                           name='bob_hid_to_msg',
                           act_fn='tanh')
    bob_msg = bob_conv.output
else:
    bob_conv_in = bob_hid.output.reshape((batch_size, 1, comm_len + key_len, 1))
    bob_conv = StandardConvSetup(bob_conv_in, 'bob')
    bob_msg = bob_conv.output.reshape((batch_size, msg_len))

# Eve see's Alice's communication to Bob, but not the key
# She gets an extra hidden layer to try and learn to decrypt the message
eve_hid1 = HiddenLayer(alice_comm, 
                       input_size=comm_len,
                       hidden_size=comm_len + key_len,
                       name='eve_to_hid1',
                       act_fn='relu')
                          
eve_hid2 = HiddenLayer(eve_hid1, 
                       input_size=comm_len + key_len,
                       hidden_size=comm_len + key_len,
                       name='eve_to_hid2',
                       act_fn='relu')

if skip_conv:
    eve_conv = HiddenLayer(eve_hid2,
                           input_size=comm_len + key_len,
                           hidden_size=msg_len,
                           name='eve_hid_to_msg',
                           act_fn='tanh')
    eve_msg = eve_conv.output
else:
    eve_conv_in = eve_hid2.output.reshape((batch_size, 1, comm_len + key_len, 1))
    eve_conv = StandardConvSetup(eve_conv_in, 'eve')
    eve_msg = eve_conv.output.reshape((batch_size, msg_len))

# Eve's loss function is the L1 norm between true and recovered msg
decrypt_err_eve = T.mean(T.abs_(msg_in - eve_msg))

# Bob's loss function is the L1 norm between true and recovered
decrypt_err_bob = T.mean(T.abs_(msg_in - bob_msg))
# plus (N/2 - decrypt_err_eve) ** 2 / (N / 2) ** 2
# --> Bob wants Eve to do only as good as random guessing
loss_bob = decrypt_err_bob + (1. - decrypt_err_eve) ** 2.

##Variable 'cipher' returns the value of Alice output: 'msg_in:key'->'cipher'
cipher = alice_comm

# Get all the parameters for Bob and Alice, make updates, train and pred funcs
params   = {'bob' : get_all_params([bob_conv, bob_hid, 
                                    alice_conv, alice_hid])}
updates  = {'bob' : adam(loss_bob, params['bob'])}
err_fn   = {'bob' : theano.function(inputs=[msg_in, key],
                                    outputs=decrypt_err_bob)}
##This fn takes as input the 'msg_val' and 'key_val' created in each iterations from gen_st_data()
cipher_fn   = {'bob' : theano.function(inputs=[msg_in, key],
                                    outputs=cipher)}

train_fn = {'bob' : theano.function(inputs=[msg_in, key],
                                    outputs=loss_bob,
                                    updates=updates['bob'])}
pred_fn  = {'bob' : theano.function(inputs=[msg_in, key], outputs=bob_msg)}

# Get all the parameters for Eve, make updates, train and pred funcs
params['eve']   = get_all_params([eve_hid1, eve_hid2, eve_conv])
updates['eve']  = adam(decrypt_err_eve, params['eve'])
err_fn['eve']   = theano.function(inputs=[msg_in, key], 
                                  outputs=decrypt_err_eve)
train_fn['eve'] = theano.function(inputs=[msg_in, key], 
                                  outputs=decrypt_err_eve,
                                  updates=updates['eve'])
pred_fn['eve']  = theano.function(inputs=[msg_in, key], outputs=eve_msg)

# Generate static matrix: integer -> binary -> replace 0 with -1 -> create the matrix 
def gen_st_data(integer,n=msg_len):
        #Convert 'integer' to bin
	d=bin(integer)[2:].zfill(n)
	#Format the binary 'd' and replace 0's with -1's to match the layers matrix format
        a = (" ".join(d))
        b = a.replace('0', '-1') 
        #Generate the matrix out ot the binary
	gen_matrix= np.matrix(b)
        return(gen_matrix)


# Function for training either Bob+Alice or Eve for some time
def train(integer,key,bob_or_eve,results, max_iters, print_every, es=0., es_limit=10):
    count = 0
    for i in range(max_iters):
        	# Generate some data
        	#Feed 'key_val' and 'msg_in_val' from gen_st_data()
		key_val = gen_st_data(key)
        	msg_in_val = gen_st_data(integer)
		# Train on this batch and get loss
        	loss = train_fn[bob_or_eve](msg_in_val, key_val)
		#Retrieve the 'cipher' value from given 'msg_in_val:key_val'
		cipher_val = cipher_fn[bob_or_eve](msg_in_val, key_val)
		#Setting entries > 0 to 1 and entries <=0 to 0
		cipher_val[cipher_val <= 0] = '0'
		cipher_val[cipher_val > 0] = '1'
		#Create 'result_ciph' containing a valid binary format eg. 0001001
		cipher_val=(re.sub('[\[\]]', '', np.array_str(cipher_val))).replace(".", "")
		result_ciph=cipher_val.replace(" ", "")
        	# Store absolute decryption error of the model on this batch
		results = np.hstack((results, 
                             err_fn[bob_or_eve](msg_in_val, key_val).sum()))
        	# Print loss now and then
        	if i % print_every == 0:
			count +=1
			count -=1
        	# Early stopping if we see a low-enough decryption error enough times
        	if es and loss < es:
            		count += 1
            	if count > es_limit:
                	break
    return (result_ciph, np.hstack((results, np.repeat(results[-1], max_iters - i - 1))))

###############################################################################################################################
# Initialise the values for iterations, message from 0 to 'last_message', static 'key' value with its binary version 'key_bin'#
###############################################################################################################################

adversarial_iterations = 1
last_message= 256
key=119
key_bin=bin(key)[2:].zfill(msg_len)


#Initialise simple table for storing the final lookup table
tab = tt.Texttable()
x = [[]] 

print "\t\t\n\n",Banner
print "\t\t\n\n\n\nLookup table for static KEY -> [",key_bin,"] (",key,")"

#Loop from 0 to 'last_message'

for message in range(last_message):
	#Initialise a new 'results_bob' array for each 'message'
	results_bob=[]
      	message_bin=bin(message)[2:].zfill(msg_len)
	# Perform adversarial training
	for i in range(adversarial_iterations):
		#Set 'n' for the number of 'max_iters'
    		n = 1
    		print_every = 100
    		cipher,results_bob = train(message,key, 'bob', results_bob, n, print_every, es=0.3)
		x.append([message,message_bin,cipher])

#Print the table with values
tab.add_rows(x)
tab.set_cols_align(['r','r','r'])
tab.header(['Message_int', 'Message(8bits)', 'Cipher(8bits)'])
print tab.draw()


