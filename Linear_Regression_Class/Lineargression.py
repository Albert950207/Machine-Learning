import numpy as np

class lineargression:
    #initialize class
    def __init__(self, train_x,train_y,weights,bias,loss_fucntion,learning_rate,iterations):
        self.train_x = train_x
        self.train_y = train_y
        self.bias = bias
        self.weights = weights
        self.loss_fucntion = loss_fucntion
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = 0


    #training method
    def train(self):
        # initialize variables
        estimation = 0
        #initialize bias and weights before training
        gradient_dct=dict(bias_gradient=0,weights_gradient=0)
        print("Training......")
        # tranin
        for i in range(self.iterations):
            #covert 1 D array to N D array
            self.train_y = np.reshape(self.train_y, (self.train_y.shape[0],1))
            # estimated value of Y
            estimation = np.dot(self.train_x, self.weights.T) + self.bias
            # caculate the diff between estimated value and Y from traning data
            self.loss = np.dot((self.train_y - estimation).T,(self.train_y - estimation))/self.train_y.shape[0]
            # calculate gradients for both weights and bias
            gradient_dct["weights_gradient"] = -2*(1/self.train_x.shape[0])*np.dot(self.train_x.T,(self.train_y-estimation)).T
            gradient_dct["bias_gradient"] = -2*np.dot((self.train_y - estimation).T,np.ones(shape=[self.train_x.shape[0],1]))/self.train_x.shape[0]
            #covert 'o' Type from pandas to float 64
            gradient_dct["weights_gradient"] = np.array(gradient_dct["weights_gradient"], dtype = 'float64')
            gradient_dct["bias_gradient"] = np.array(gradient_dct["bias_gradient"], dtype = 'float64')
            # update weights gradient and bias gradient
            self.weights -= self.learning_rate*gradient_dct["weights_gradient"]
            self.bias -= self.learning_rate*gradient_dct["bias_gradient"]
            if self.loss[0][0] < 0.01:
                break


            print("iterations:", i,"loss:", self.loss)


    #prediction method
    def prediction(self,test_x,test_y):

        k = (np.max(test_y)-np.min(test_y))+np.min(test_y)
        y_prediction = np.dot(test_x,self.weights.T)+self.bias
        Error_p = abs((test_y-y_prediction)/test_y)

        return y_prediction, Error_p
