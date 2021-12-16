import numpy as np
from scipy.stats import truncnorm

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)




def softmax_activation(x):
    e_x = np.exp(x - np.max(x))
    #print(e_x/np.sum(e_x))
    return e_x/np.sum(e_x, axis = 0)

def mini_softmax(x):
    y = np.empty(x.shape[1])
    for i in range(x.shape[1]):
        exps = np.exp(x[i] - x[i].max())
        y[i] = exps / np.sum(exps)
    return y

@np.vectorize
def sigmoid_activation(x):
    return 1.0/(1.0+(np.exp(-x)))

def relu(Z):
    return np.maximum(0,Z)

def loss_function(y, y_hat):
    L_sum =  np.sum(np.multiply(y, (np.log(y_hat)).T))
    m = y.shape[1]
    L = -(1./m) * L_sum

    return L


class Neural_Network:
    def __init__(self, network_structure, learning_rate ,bias = None):
        self.net_structure = network_structure
        self.learning_rate = learning_rate;
        self.bias = bias
        self.initialize_weights()

    def initialize_weights(self):
        X = truncated_normal(mean=2, sd=1, low=-0.5, upp=0.5)
        
        bias_node = 1 if self.bias else 0

        self.weight_matrices = []
        no_of_layers = len(self.net_structure)
        layer_id = 1
        while layer_id < no_of_layers:
            input_nodes = self.net_structure[layer_id - 1]
            output_nodes = self.net_structure[layer_id]
            n = (input_nodes + bias_node) * output_nodes
            rad = 1/np.sqrt(input_nodes)
            X = truncated_normal(mean = 2, sd = 1, low = -rad, upp = rad)
            weighted_mean = X.rvs(n).reshape((output_nodes, input_nodes + bias_node))
            self.weight_matrices.append(weighted_mean)
            layer_id += 1

    def train_single_epoch(self, input_layer, target_layer):
        no_of_layers = len(self.net_structure)
        input_layer = np.array(input_layer, ndmin = 2).T

        layer_index = 0

        response_vector = [input_layer]

        
        #forward propogation
        while layer_index < no_of_layers-1:
            in_vector = response_vector[-1]

            if self.bias:
                in_vector = np.concatenate((in_vector, [[self.bias]]))
                response_vector[-1] = in_vector
            x = np.dot(self.weight_matrices[layer_index], in_vector)
            #x = self.weight_matrices[layer_index] @ in_vector
            #Using Softmax for the last layer
            if layer_index == no_of_layers-2:
                out_vector = softmax_activation(x)
                #print("out vector = ", softmax_activation(x))

            else:
                out_vector = sigmoid_activation(x)      #Try changing this to relu later
                #print("out vector = ", out_vector)
                
            #out_vector = sigmoid_activation(x)
            response_vector.append(out_vector)
            
            
            layer_index += 1
        
        layer_index = no_of_layers-1
        target_layer = np.array(target_layer, ndmin=2).T
        output_errors = target_layer-out_vector
        #output_errors = loss_function(target_layer,out_vector)
        #print("output errors= ", output_errors)
        #print("Size of response vector = ", len(response_vector[layer_index]))
        #print("output vector= ", out_vector)
        

        #Backpropogation
        while layer_index>0:
            out_vector = response_vector[layer_index]
            in_vector = response_vector[layer_index-1]

            if self.bias and not layer_index == (no_of_layers-1):
                out_vector = out_vector[:-1,:].copy()
            
            if(layer_index == (no_of_layers-1)):
                #print("out vector: ", out_vector)
                ovn = out_vector.reshape(out_vector.size,)
                #print("ovn: ", ovn)
                si_sj = - ovn * ovn.reshape(self.net_structure[-1], 1)
                s_der = np.diag(ovn) + si_sj
                #print("Output errors = ", output_errors)
                tmp =  (s_der @ output_errors)
                #print("in_vector: ", in_vector.T)
                #print("s_der: ", s_der)
                #print("value: ", tmp @ in_vector.T)
                #Weight Update
                self.weight_matrices[layer_index-1]  += self.learning_rate  * (tmp @ in_vector.T)
              
            else:
                temp = output_errors * out_vector * (1.0 - out_vector)
                #print("Sigmoid Gradient = ", temp)
                temp = np.dot(temp, in_vector.T)
                
                #Weight Update
                self.weight_matrices[layer_index-1] += self.learning_rate * temp

            output_errors = np.dot(self.weight_matrices[layer_index-1].T, output_errors)

            if self.bias:
                output_errors = output_errors[:-1,:]

            layer_index -= 1

    def train(self, data_array, label_array, epochs = 1, intermediate_result = False):
        intermediate_wts = []
        no_of_input_elements = len(data_array)

        for epoch in range(epochs):
            for i in range(len(data_array)):
                self.train_single_epoch(data_array[i], label_array[i])
                #if intermediate_result:
                    #   intermediate_wts.append(self.wih.copy(), self.woh.copy())
        
        return intermediate_wts

    def minibatch_feedforward(self, input_layer):
        no_of_layers = len(self.net_structure)
        input_layer = np.array(input_layer, ndmin = 2).T
        print("Shape of Input layer = ", input_layer.shape)
        layer_index = 0

        response_vector = [input_layer]
        
        #forward propogation
        while layer_index < no_of_layers-1:
            in_vector = response_vector[-1]

            if self.bias:
                in_vector = np.concatenate((in_vector, np.full((1,1000),self.bias)))
                response_vector[-1] = in_vector
            x = np.dot(self.weight_matrices[layer_index], in_vector)
            #x = self.weight_matrices[layer_index] @ in_vector
            #Using Softmax for the last layer
            if layer_index == no_of_layers-2:
                out_vector = softmax_activation(x)
                #print("softmax: ", out_vector)
                #print("softmax out vector shape = ", out_vector.shape)

            else:
                out_vector = sigmoid_activation(x)      #Try changing this to relu later
                #print("sigmoid: ", out_vector)
                #print("sigmoid out vector shape = ", out_vector.shape)
                
            #out_vector = sigmoid_activation(x)
            response_vector.append(out_vector)
            
            
            layer_index += 1
        
        return response_vector


    def minibatch_backprop(self, input_layer, target_layer, cache):
        no_of_layers = len(self.net_structure)
        layer_index = no_of_layers-1
        target_layer = np.array(target_layer, ndmin=2).T
        #print("Target Layer: ", target_layer)
        output_errors = target_layer-cache[layer_index]
        grads = []
        batch_size = 1000
        print("Size of erros in backprop", output_errors.shape)
        print("Size of cache in backprop", cache[0].shape, cache[1].shape, cache[2].shape, cache[3].shape)
        #output_errors = loss_function(target_layer,out_vector)
        #print("output errors= ", output_errors)
        #print("Size of response vector = ", len(response_vector[layer_index]))
        #print("output vector= ", out_vector)
        #grads[len(cache)-1] = (1./batch_size)

        #Backpropogation
        while layer_index>0:
            out_vector = cache[layer_index]
            in_vector = cache[layer_index-1]

            if self.bias and not layer_index == (no_of_layers-1):
                out_vector = out_vector[:-1,:].copy()
            
            if(layer_index == (no_of_layers-1)):
                temp_grads = []
                for i in range(out_vector.shape[1]):
                    #print("out_vector: ", out_vector[:,i])
                    ovn = out_vector[:,i].reshape(out_vector[:,i].size,)
                    #print("ovn: ", ovn)
                    #print("ovn size = ", ovn.shape)
                    si_sj = - ovn * ovn.reshape(self.net_structure[-1], 1)
                    #print("Size of si_sj = ", si_sj.shape)
                    s_der = np.diag(ovn) + si_sj
                    #print("s_der = ", s_der)
                    #print("Output errors = ", output_errors[:,i].reshape(output_errors[:,i].shape[0],1))
                    tmp =  (s_der @ output_errors[:,i].reshape(output_errors[:,i].shape[0],1))
                    #print("Temp", tmp)
                    #print("Temp shape", )
                    temp_grads.append(tmp @ in_vector[:,i].reshape(in_vector[:,i].shape[0],1).T)
                    print(tmp @in_vector[:,i].reshape(in_vector[:,i].shape[0],1).T)
                    #Weight Update
                    #self.weight_matrices[layer_index-1]  += self.learning_rate  * (tmp @ in_vector.T)
                print("temp grads: ", temp_grads)
                grads.append(temp_grads)
              
            else:
                temp = output_errors * out_vector * (1.0 - out_vector)
                #print("Sigmoid Gradient = ", temp)
                temp = np.dot(temp, in_vector.T)
                grads.append(temp)
                
                #Weight Update
                #self.weight_matrices[layer_index-1] += self.learning_rate * temp

            output_errors = np.dot(self.weight_matrices[layer_index-1].T, output_errors)

            if self.bias:
                output_errors = output_errors[:-1,:]

            layer_index -= 1
        #print(grads)

        return grads


    def train_mini_batch(self, data_array, label_array, epochs = 1, batch_size = 1, intermediate_results = False):
        intermediate_wts = []
        no_of_input_elements = len(data_array)
        mean_gradients = []
        
        for epoch in range(epochs):
            for batch_id in range(batch_size):
                cache = []
                grads = []
                begin = batch_id*batch_size
                end = min(begin + batch_size, no_of_input_elements)
                cache = self.minibatch_feedforward(data_array[0:1000,:])
                grads = self.minibatch_backprop(data_array[0:1000,:], label_array[0:1000,:],cache)
                layer_index = len(self.net_structure)-1
                
                while layer_index>0:
                    #print("grad length = ", grads[layer_index-1])
                    self.weight_matrices[layer_index-1] += self.learning_rate * (1/len(grads[layer_index-1]) * np.sum(grads[layer_index-1]))
                    layer_index-=1


    
    def predict(self, input_vector):
        no_of_layers = len(self.net_structure)

        if self.bias:
            input_vector = np.concatenate((input_vector, [self.bias]))

        in_vector = np.array(input_vector, ndmin = 2).T

        layer_index = 1

        while layer_index < no_of_layers:
            #x = np.dot(self.weight_matrices[layer_index-1], in_vector)
            x = self.weight_matrices[layer_index-1] @ in_vector
            
            if layer_index == no_of_layers-1:
                out_vector = softmax_activation(x)
            else:
                out_vector = sigmoid_activation(x)

            #out_vector = sigmoid_activation(x)
            in_vector = out_vector

            if self.bias:
                in_vector = np.concatenate((in_vector, [[self.bias]]))

            layer_index += 1


        return out_vector

    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        
        for i in range(len(data)):
            res = self.predict(data[i])
            res_max = res.argmax()
            #print("Predicted Value: ", res_max)
            #print("Actual Value: ", labels[i])
            #print(" ")
            if res_max == labels[i]:
                corrects+=1
            else:
                wrongs += 1


        return corrects, wrongs

    def get_predictions(self, data):
        predictions = []
        for i in range(len(data)):
            res = self.predict(data[i])
            res_max = res.argmax()
            predictions.append(res_max)

        return np.array(predictions)


    def confusion_matrix(self, data_array, labels):
        cm = {}
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            if (target, res_max) in cm:
                cm[(target, res_max)] += 1
            else:
                cm[(target, res_max)] = 1
        return cm


                
