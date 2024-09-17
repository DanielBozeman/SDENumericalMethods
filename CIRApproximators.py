import numpy as np
import matplotlib.pyplot as mp
import math

class CIR_Model():
    s = 0
    m = 0
    v = 0

    def __init__(self,s,m,v) -> None:
        self.s = s
        self.m = m
        self.v = v

    def drift_function(self,t,y):
        return self.s * (self.m - y)
    
    def abs_variance_function(self,t,y):
        return self.v * math.sqrt(abs(y))
    
    def max_variance_function(self,t,y):
        return self.v * math.sqrt(max(y,0))
    
    def variance_function(self,t,y):
        return self.v * math.sqrt(y)
    
    def milstein_term(self,t,y):
        term = self.v * self.v * (1/4)
        return term


def run_abs_EM(brownian_path, time_steps, model : CIR_Model, initial_value):
    dt = time_steps[1] - time_steps[0]

    y_values = np.zeros(time_steps.size)

    y_values[0] = initial_value

    for i in range(1, time_steps.size):
        dW = brownian_path[i]-brownian_path[i-1]
        stock_price = y_values[i-1]
        cur_time = time_steps[i]
        y_values[i] = stock_price + model.drift_function(cur_time, stock_price) * dt + model.abs_variance_function(cur_time, stock_price) * dW

        #stock_prices[i] = max(stock_prices[i], 0)

    return time_steps, y_values

def run_max_EM(brownian_path, time_steps, model : CIR_Model, initial_value):
    dt = time_steps[1] - time_steps[0]

    y_values = np.zeros(time_steps.size)

    y_values[0] = initial_value

    for i in range(1, time_steps.size):
        dW = brownian_path[i]-brownian_path[i-1]
        stock_price = y_values[i-1]
        cur_time = time_steps[i]
        y_values[i] = stock_price + model.drift_function(cur_time, stock_price) * dt + model.max_variance_function(cur_time, stock_price) * dW

        #stock_prices[i] = max(stock_prices[i], 0)

    return time_steps, y_values

def run_total_max_EM(brownian_path, time_steps, model : CIR_Model, initial_value):
    dt = time_steps[1] - time_steps[0]

    y_values = np.zeros(time_steps.size)

    y_values[0] = initial_value

    for i in range(1, time_steps.size):
        dW = brownian_path[i]-brownian_path[i-1]
        stock_price = y_values[i-1]
        cur_time = time_steps[i]
        y_values[i] = max(stock_price + model.drift_function(cur_time, stock_price) * dt + model.variance_function(cur_time, stock_price) * dW, 0)

        #stock_prices[i] = max(stock_prices[i], 0)

    return time_steps, y_values

def run_abs_Milstein(brownian_path, time_steps, model : CIR_Model, initial_value):
    dt = time_steps[1] - time_steps[0]

    y_values = np.zeros(time_steps.size)

    y_values[0] = initial_value

    for i in range(1, time_steps.size):
        dW = brownian_path[i]-brownian_path[i-1]
        stock_price = y_values[i-1]
        cur_time = time_steps[i]
        y_value = stock_price + model.drift_function(cur_time, stock_price) * dt + model.abs_variance_function(cur_time, stock_price) * dW
        y_value += model.milstein_term(cur_time, stock_price) * (dW * dW - dt)
        y_values[i] = y_value
        #stock_prices[i] = max(stock_prices[i], 0)

    return time_steps, y_values

def run_max_Milstein(brownian_path, time_steps, model : CIR_Model, initial_value):
    dt = time_steps[1] - time_steps[0]

    y_values = np.zeros(time_steps.size)

    y_values[0] = initial_value

    for i in range(1, time_steps.size):
        dW = brownian_path[i]-brownian_path[i-1]
        stock_price = y_values[i-1]
        cur_time = time_steps[i]
        y_value = stock_price + model.drift_function(cur_time, stock_price) * dt + model.max_variance_function(cur_time, stock_price) * dW
        y_value += model.milstein_term(cur_time, stock_price) * (dW * dW - dt)
        y_values[i] = y_value
        #stock_prices[i] = max(stock_prices[i], 0)

    return time_steps, y_values

def run_total_max_Milstein(brownian_path, time_steps, model : CIR_Model, initial_value):
    dt = time_steps[1] - time_steps[0]

    y_values = np.zeros(time_steps.size)

    y_values[0] = initial_value

    for i in range(1, time_steps.size):
        dW = brownian_path[i]-brownian_path[i-1]
        stock_price = y_values[i-1]
        cur_time = time_steps[i]
        y_value = stock_price + model.drift_function(cur_time, stock_price) * dt + model.variance_function(cur_time, stock_price) * dW
        y_value += model.milstein_term(cur_time, stock_price) * (dW * dW - dt)
        y_value = max(y_value, 0)
        y_values[i] = y_value
        #stock_prices[i] = max(stock_prices[i], 0)

    return time_steps, y_values

def run_exact(brownian_path, time_steps, model : CIR_Model, initial_value):
    dt = time_steps[1] - time_steps[0]

    y_values = np.zeros(time_steps.size)

    y_values[0] = initial_value
    y_values[1] = initial_value

    for i in range(1, time_steps.size):
        dW = brownian_path[i]-brownian_path[i-1]
        dWPrev = brownian_path[i-1]-brownian_path[i-2]
        if i == 1:
            dWPrev = dW
        stock_price = y_values[i-1]
        cur_time = time_steps[i]
        y_value = stock_price + model.s * model.m * dt + model.v * math.sqrt(stock_price) * dt * dWPrev + 0.25 * model.v * model.v * (dWPrev * dWPrev - dt)
        y_value = y_value/ (1 + model.s * dt)
        y_values[i] = y_value
        #stock_prices[i] = max(stock_prices[i], 0)

    return time_steps, y_values
