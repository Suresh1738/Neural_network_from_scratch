#required libraries
import numpy as np
import scipy.special
from matplotlib import pyplot as plt

class neural_network:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.lr=learningrate
    
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        #activaion
        self.activation_function=lambda x:scipy.special.expit(x)
        pass

    def train(self,input_list,target_list):
        inputs=np.array(input_list,ndmin=2).T
        targets=np.array(target_list,ndmin=2).T
        hidden_inputs=np.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        #preparing final inputs
        final_inputs=np.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        #maths for finding output layer error
        output_errors=targets-final_outputs
        hidden_errors=np.dot(self.who.T,output_errors)
        #updating weights of hidden layer and output layer
        self.who +=self.lr*np.dot((output_errors*final_outputs)*(1-final_outputs),np.transpose(hidden_outputs))
        #updating weights of input layer and hidden layer
        self.wih +=self.lr*np.dot((hidden_errors*hidden_outputs)*(1-hidden_outputs),np.transpose(inputs))

        pass
    def query(self,input_list):
        #converting input list to 2d array
        inputs=np.array(input_list,ndmin=2).T
        hidden_inputs=np.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        #for final inputs
        final_inputs=np.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        return final_outputs 



input_nodes=784
output_nodes=10
hidden_nodes=100
learning_rate=0.3
#creating a neural network object 
n=neural_network(input_nodes,hidden_nodes,output_nodes,learning_rate)


training_data_file=open("mnist_dataset/mnist_train.csv",'r')
training_data_list=training_data_file.readlines()
training_data_file.close()

epochs = 5

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass



# load the mnist test data CSV file into a list
test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()


scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    
    pass


# calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)

