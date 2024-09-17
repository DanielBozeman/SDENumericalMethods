import numpy as np
import matplotlib.pyplot as mp

#Model class to store the drift and variance as variables
class SDE_Model:
    drift = 0
    variance = 0

    def __init__ (self, drift, variance):
        self.drift = drift
        self.variance = variance

    def drift_function(self):
        return self.drift

    def variance_function(self):
        return self.drift


def dW(dt : float):
    return np.random.normal(loc = 0.0, scale=np.sqrt(dt))

def run_simulation():
    model = SDE_Model(drift=0.75, variance=0.3)

    time_interval = [0.0, 1.0]
    time_discretization = 10

    dt = (time_interval[1] - time_interval[0]) / (pow(2,time_discretization))

    time_steps = np.arange(time_interval[0], time_interval[1] + dt, dt)

    initial_stock_price = 307.65

    stock_prices = np.zeros(time_steps.size)

    stock_prices[0] = initial_stock_price

    for i in range(1, time_steps.size):
        stock_price = stock_prices[i-1]
        #print( ( dW(dt) * stock_price))
        stock_prices[i] = stock_price + model.drift_function() * stock_price * dt + stock_price * model.variance_function() * dW(dt) + 0.5 * model.variance_function() * model.variance_prime() * (pow(dW(dt),2) - dt)

        #stock_prices[i] = max(stock_prices[i], 0)

    return time_steps, stock_prices

def run_EM(brownian_path, time_steps, drift_function, variance_function, initial_value):
    dt = time_steps[1] - time_steps[0]

    y_values = np.zeros(time_steps.size)

    y_values[0] = initial_value

    for i in range(1, time_steps.size):
        dW = brownian_path[i]-brownian_path[i-1]
        stock_price = y_values[i-1]
        cur_time = time_steps[i]
        y_values[i] = stock_price + drift_function(cur_time, stock_price) * dt + variance_function(cur_time, stock_price) * dW

        #stock_prices[i] = max(stock_prices[i], 0)

    return time_steps, y_values

def run_Milstein(brownian_path, time_steps, drift_function, variance_function, variance_prime, initial_value):
    dt = time_steps[1] - time_steps[0]

    y_values = np.zeros(time_steps.size)

    y_values[0] = initial_value

    for i in range(1, time_steps.size):
        dW = brownian_path[i]-brownian_path[i-1]
        stock_price = y_values[i-1]
        cur_time = time_steps[i]
        y_value = stock_price + drift_function(cur_time, stock_price) * dt + variance_function(cur_time, stock_price) * dW
        y_value += 0.5 * variance_function(cur_time, stock_price) * variance_prime(cur_time, stock_price) * ( dW * dW - dt)
        
        y_values[i] = y_value
        #stock_prices[i] = max(stock_prices[i], 0)

    return time_steps, y_values

if __name__ == "__main__":

    num_sims = 100

    total = 0;

    for i in range(num_sims):
        simulation_times, simulation_prices = run_simulation()
        total += simulation_prices[-2]
        #print(simulation_prices[-1])
        mp.plot(simulation_times, simulation_prices)

    average = total/num_sims

    value = average - 30

    print(average)
    print(value)

    mp.xlabel("Time")
    mp.ylabel("x")
    mp.show()