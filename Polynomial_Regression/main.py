import numpy as np
from numpy import mat
from Lineargression import lineargression
from predata.predata import preprocessor
from PolynomialRegression import Polynomialregression
#load data
data_path = "/Users/bohanli/Desktop/Linear_Regression_Class/abalone.txt"
#data preprecessing
pr = preprocessor(data_path)
#pr.clear(-1, 'row')
pr.Transpose()
#pr.Normalization()
pr.Transpose()
Data = np.array(pr.data)
print(Data)
#Seprete data
Y = pr.data[:,-1]
X = pr.data[:,0:-1]
N = round(0.8*Y.shape[0])
print(N)
train_x = X[0:N,:]
train_y =Y[0:N]
test_x = X[N:Y.shape[0],:]
test_y = Y[N:Y.shape[0]]
print(X)
print(Y)
print(train_x.shape)
print(train_y.shape)

#set Hyperparameters for model
Dimension = len(X[0])
Hyper_parameters=dict(
                    weights = np.array(np.random.randn(1,Dimension), dtype = 'float64'),
                    bias =  np.array([[1.0]]),
                    loss_fucntion = "MSE" ,
                    learning_rate = 0.01,
                    Iterations = 10000,
                    )
#create a lineargression model
#model = lineargression(train_x,train_y,Hyper_parameters["weights"],Hyper_parameters["bias"],
#Hyper_parameters["loss_fucntion"], Hyper_parameters["learning_rate"], Hyper_parameters["Iterations"])
model = Polynomialregression(train_x,train_y,Hyper_parameters["weights"],Hyper_parameters["bias"],
Hyper_parameters["loss_fucntion"], Hyper_parameters["learning_rate"], Hyper_parameters["Iterations"])
#training
model.train()
#print(model.weights)
results = model.prediction(test_x,test_y)

print("prediction is:", results[0], "error is:",results[1])
for i in range(test_y.shape[0]):
    print("prediction is:", results[0][i], "real is:", test_y[i])
#print(pr.data)

#train_data = standarization(train_matrix)
#test_data = standarization(test_matrix)

#lineargression_model = lineargression()
