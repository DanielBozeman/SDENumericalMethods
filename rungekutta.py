import numpy as np
import  numpy as np
import matplotlib.pyplot as mp
from Week3EulerMaruyama import SDE_Model
import math

def y_function(t, y):
    return( math.tan(y) + 1)

def dW(dt : float):
    return np.random.normal(loc = 0.0, scale=np.sqrt(dt))

def stochastic_RK_simulation():

    model = SDE_Model(drift=-2, variance=1)

    #Time interval that we are summing over
    time_interval = [1.0, 1.1]
    time_discretization = 10

    #The time step of the time-discretization
    #dt = (time_interval[1] - time_interval[0]) / (pow(2,time_discretization))
    dt = 0.025

    #Matrix for each time step
    time_steps = np.arange(time_interval[0], time_interval[1] + dt, dt)

    #The butcher matrix of our particular runge kutta
    c_butcher = [[0],[2/3,2/3],[1/4,3/4]]
    a_butcher= []
    b_butcher = []
    gamma_butcher = []
    q_butcher = []

    #initial y value
    initial_y = 1

    #Matrix of zeroes for the y values, we fill in as we go
    y_values = np.zeros(time_steps.size)

    #setting the initial y
    y_values[0] = initial_y

    #Iterating over each time step and setting the assosciated y value
    for i in range(1, time_steps.size):

        #The previous y value and time step
        cur_y = y_values[i-1]
        cur_time = time_steps[i-1]

        #Matrix for all of the k values in our runge kutta
        k_matrix = []

        k_matrix = np.zeros(len(butcher) - 1)

        k_matrix[0] = (y_function(cur_time, cur_y))

        for j in range(1,len(butcher) - 1 ):
            time_input = cur_time + dt * butcher[j][0]
            #print(time_input)

            y_input = 0

            #print("Current j is " + str(j))

            for k in range(j):
                #print("Cur k:" + str(k))
                y_input += (butcher[j][k] * k_matrix[k])

            #print("\n")
            
            y_input *= dt

            y_input += cur_y

            k_matrix[j] = (y_function(time_input, y_input))

        y_value = 0

        for j in range(len(k_matrix)):
            print(k_matrix[j])
            y_value += (k_matrix[j] * butcher[-1][j] * dt)

        y_value += cur_y

        y_values[i] = y_value

        print("Current y: " + str(y_values[i]))

        print("\n")

    print(y_values[-1])

    print(time_steps)
    return

def deterministic_RK_simulation():

    #Time interval that we are summing over
    time_interval = [1.0, 1.1]
    time_discretization = 10

    #The time step of the time-discretization
    #dt = (time_interval[1] - time_interval[0]) / (pow(2,time_discretization))
    dt = 0.025

    #Matrix for each time step
    time_steps = np.arange(time_interval[0], time_interval[1] + dt, dt)

    #The butcher matrix of our particular runge kutta
    butcher = [[0],[2/3,2/3],[1/4,3/4]]

    #initial y value
    initial_y = 1

    #Matrix of zeroes for the y values, we fill in as we go
    y_values = np.zeros(time_steps.size)

    #setting the initial y
    y_values[0] = initial_y

    #Iterating over each time step and setting the assosciated y value
    for i in range(1, time_steps.size):

        #The previous y value and time step
        cur_y = y_values[i-1]
        cur_time = time_steps[i-1]

        #Matrix for all of the k values in our runge kutta
        k_matrix = []

        k_matrix = np.zeros(len(butcher) - 1)

        k_matrix[0] = (y_function(cur_time, cur_y))

        for j in range(1,len(butcher) - 1 ):
            time_input = cur_time + dt * butcher[j][0]
            #print(time_input)

            y_input = 0

            #print("Current j is " + str(j))

            for k in range(j):
                #print("Cur k:" + str(k))
                y_input += (butcher[j][k] * k_matrix[k])

            #print("\n")
            
            y_input *= dt

            y_input += cur_y

            k_matrix[j] = (y_function(time_input, y_input))

        y_value = 0

        for j in range(len(k_matrix)):
            print(k_matrix[j])
            y_value += (k_matrix[j] * butcher[-1][j] * dt)

        y_value += cur_y

        y_values[i] = y_value

        print("Current y: " + str(y_values[i]))

        print("\n")

    print(y_values[-1])

    print(time_steps)
    return

if __name__ == "__main__":
    deterministic_RK_simulation()

    print(0.5 * y_function(0,0.5))