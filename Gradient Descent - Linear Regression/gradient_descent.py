import numpy as np 
import csv
import matplotlib.pyplot as plt

"""
The dataset represents distance cycled vs calories burned.
We'll create the line of best fit (linear regression) via gradient descent to predict the mapping.
"""

#Get Dataset
def get_data(file_name):
	"""
	This method gets the data points from the csv file 
	"""
	data = []
	with open(file_name, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for line in reader:
			row = map(float,[line[0], line[1]])
			data.append(row)
	return data



def loss(m, b, data):
	"""
	This method computes the loss for the given value of paramters
	Error = 1/N * sum(Yi - (mXi + b))^2
	"""
	N = len(data)
	error = 0
	for i in range(N):
		x = data[i][0]
		y = data[i][1]
		t = m * x + b
		error += (y - t)**2
	error = error/N

	return error

def step_gradient(m,b, data, eta):
	"""
	This function calculates the gradient of paramters wrt Error and then
	compute the new values of the paramters
	"""
	dE_dm = 0
	dE_db = 0
	N = len(data)
	for i in range(N):
		x = data[i][0]
		y = data[i][1]
		t = m * x + b
		dE_dm += -1 * x * (y - t)
		dE_db += -1 * (y - t)
	dE_dm = (2 * dE_dm) / N
	dE_db = (2 * dE_db) / N

	new_m = m - eta * dE_dm
	new_b = b - eta * dE_db

	return [new_m, new_b]

def gradient_descent(m,b, data, num_iterations, eta):
	"""
	This method performs the gradient descent for a given number of iterations
	"""
	for i in range(num_iterations):
		#Compute the Error
		error = loss(m, b, data)
		#Compute the gradients
		[m, b] = step_gradient(m, b, data, eta)

		print "Epoch No : {0} -----> Error : {1} -----> m : {2} , b : {3}".format(i,error, m, b)

	return [m,b,error]








#Compute Gradient

#Update Parameters


def main():
	#Get the dataset from csv file
	data = get_data('data.csv')
	

	#Intialize the Parameters for line equation y = mx + c
	m = 0
	b = 0

	#Initialize hyperparamters
	eta = 0.0001
	num_iterations = 1000

	#Run Gradient Descent
	print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(0,0, loss(m, b, data))
	print "Running....."
	[m,b, error] = gradient_descent(m, b, data, num_iterations, eta)
	print "Completed!"
	print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, error)

	#Plot the Best Fit line
	points = np.asarray(data)
	X_axis = points[:,0]
	Y_axis = points[:,1]
	plt.plot(X_axis, Y_axis, 'bo')
	plt.plot(X_axis, m * X_axis + b, 'r-')
	plt.axis([0,1.5* max(X_axis), 0, 1.3 * max(Y_axis)])
	plt.title("Best fit : Linear Regression")
	plt.text(10, 130, "m="+str(round(m,4))+"  b="+str(round(b,4)))
	plt.show()
		


if __name__ == "__main__":
	main()
