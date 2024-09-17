import numpy as np
import matplotlib.pyplot as mp
import Week3EulerMaruyama
import math
import CIRApproximators

def dW(dt : float):
    return np.random.normal(loc = 0.0, scale=np.sqrt(dt))


def make_path(time_discretization, interval_start, interval_end):
    time_interval = [interval_start, interval_end]

    #dt = (time_interval[1] - time_interval[0]) / (pow(2,time_discretization))

    dt = math.pow(2, time_discretization)

    time_steps = np.arange(time_interval[0], time_interval[1] + dt, dt)

    brownian_path = np.zeros(time_steps.size)

    brownian_path[0] = 0

    for i in range(1, time_steps.size):
        brownian_path[i] = brownian_path[i - 1] + dW(dt)

    return time_steps, brownian_path

def drift_function(t,y):
    #rate = (0.08446 + 0.2 * math.log(y))
    rate = 3 * y
    return rate

def variance_function(t,y):
    return 1 * y

def variance_prime(t,y):
    return 1

def BS_exact_solution(brownian_path, time_steps, initial_value):

    drift_coefficient = 3
    variance_coefficient = 1

    dt = time_steps[1] - time_steps[0]

    y_values = np.zeros(time_steps.size)

    y_values[0] = initial_value

    for i in range(1, time_steps.size):
        dW = brownian_path[i]-brownian_path[i-1]
        stock_price = y_values[i-1]
        cur_time = time_steps[i]
        y_values[i] = initial_value * math.exp((drift_coefficient - 0.5 * variance_coefficient * variance_coefficient) * cur_time + variance_coefficient * brownian_path[i])

        #stock_prices[i] = max(stock_prices[i], 0)

    return time_steps, y_values

def average_Milstein_simulations(initial_value, num_simulations, time_discretization):
    brownian_times, brownian_path = make_path(time_discretization)

    average_path = np.zeros(brownian_path.size)

    for i in range(num_simulations):
        brownian_times, brownian_path = make_path(time_discretization)
        
        simulation_times, simulation_path = Week3EulerMaruyama.run_Milstein(brownian_path, brownian_times, drift_function, variance_function, variance_prime, initial_value)
        
        for j in range(simulation_path.size):
            average_path[j] = average_path[j] + simulation_path[j]
        #mp.plot(simulation_times, simulation_path)
    
    for j in range(simulation_path.size):
            average_path[j] = average_path[j]/num_simulations
        
    return simulation_times, average_path


def average_EM_simulations(initial_value, num_simulations, time_discretization):
    brownian_times, brownian_path = make_path(time_discretization)

    average_path = np.zeros(brownian_path.size)

    for i in range(num_simulations):
        brownian_times, brownian_path = make_path(time_discretization)
        
        simulation_times, simulation_path = Week3EulerMaruyama.run_EM(brownian_path, brownian_times, drift_function, variance_function, initial_value)
        
        for j in range(simulation_path.size):
            average_path[j] = average_path[j] + simulation_path[j]
        #mp.plot(simulation_times, simulation_path)
    
    for j in range(simulation_path.size):
            average_path[j] = average_path[j]/num_simulations
        
    return simulation_times, average_path

def average_CIR_simulations(simulation_function, time_discretization, num_simulations, model, initial_value):
    brownian_times, brownian_path = make_path(time_discretization)

    average_path = np.zeros(brownian_path.size)

    for i in range(num_simulations):
        brownian_times, brownian_path = make_path(time_discretization)
        
        simulation_times, simulation_path = simulation_function(brownian_path, brownian_times, model, initial_value)

        for j in range(simulation_path.size):
            average_path[j] = average_path[j] + simulation_path[j]
        #mp.plot(simulation_times, simulation_path)
    
    for j in range(simulation_path.size):
            average_path[j] = average_path[j]/num_simulations
        
    return simulation_times, average_path

def average_CIR_Milstein_simulations(time_discretization, num_simulations, model, initial_value):
    time_start = 0.0
    time_end = 5.0

    brownian_times, brownian_path = make_path(time_discretization, time_start, time_end)

    average_path_1 = np.zeros(brownian_path.size)
    average_path_2 = np.zeros(brownian_path.size)
    average_path_3 = np.zeros(brownian_path.size)

    for i in range(num_simulations):
        brownian_times, brownian_path = make_path(time_discretization, time_start, time_end)
        
        simulation_times_1, simulation_path_1 = CIRApproximators.run_abs_Milstein(brownian_path, brownian_times, model, initial_value)
        simulation_times_2, simulation_path_2 = CIRApproximators.run_max_Milstein(brownian_path, brownian_times, model, initial_value)
        simulation_times_3, simulation_path_3 = CIRApproximators.run_total_max_Milstein(brownian_path, brownian_times, model, initial_value)

        for j in range(simulation_path_1.size):
            average_path_1[j] = average_path_1[j] + simulation_path_1[j]
            average_path_2[j] = average_path_2[j] + simulation_path_2[j]
            average_path_3[j] = average_path_3[j] + simulation_path_3[j]
        #mp.plot(simulation_times, simulation_path)
    
    for j in range(simulation_path_1.size):
            average_path_1[j] = average_path_1[j]/num_simulations
            average_path_2[j] = average_path_2[j]/num_simulations
            average_path_3[j] = average_path_3[j]/num_simulations
        
    return simulation_times_1, average_path_1, average_path_2, average_path_3

def average_CIR_EM_simulations(time_discretization, num_simulations, model, initial_value):
    time_start = 0.0
    time_end = 5.0

    brownian_times, brownian_path = make_path(time_discretization, time_start, time_end)

    average_path_1 = np.zeros(brownian_path.size)
    average_path_2 = np.zeros(brownian_path.size)
    average_path_3 = np.zeros(brownian_path.size)

    for i in range(num_simulations):
        brownian_times, brownian_path = make_path(time_discretization, time_start, time_end)
        
        simulation_times_1, simulation_path_1 = CIRApproximators.run_abs_EM(brownian_path, brownian_times, model, initial_value)
        simulation_times_2, simulation_path_2 = CIRApproximators.run_max_EM(brownian_path, brownian_times, model, initial_value)
        simulation_times_3, simulation_path_3 = CIRApproximators.run_total_max_EM(brownian_path, brownian_times, model, initial_value)

        for j in range(simulation_path_1.size):
            average_path_1[j] = average_path_1[j] + simulation_path_1[j]
            average_path_2[j] = average_path_2[j] + simulation_path_2[j]
            average_path_3[j] = average_path_3[j] + simulation_path_3[j]
        #mp.plot(simulation_times, simulation_path)
    
    for j in range(simulation_path_1.size):
            average_path_1[j] = average_path_1[j]/num_simulations
            average_path_2[j] = average_path_2[j]/num_simulations
            average_path_3[j] = average_path_3[j]/num_simulations
        
    return simulation_times_1, average_path_1, average_path_2, average_path_3

def average_CIR_2EM_simulations(time_discretization, num_simulations, model, initial_value):
    time_start = 0.0
    time_end = 5.0

    brownian_times, brownian_path = make_path(time_discretization, time_start, time_end)

    average_path_1 = np.zeros(brownian_path.size)
    average_path_2 = np.zeros(brownian_path.size)

    for i in range(num_simulations):
        brownian_times, brownian_path = make_path(time_discretization, time_start, time_end)
        
        simulation_times_1, simulation_path_1 = CIRApproximators.run_abs_EM(brownian_path, brownian_times, model, initial_value)
        simulation_times_2, simulation_path_2 = CIRApproximators.run_total_max_EM(brownian_path, brownian_times, model, initial_value)

        for j in range(simulation_path_1.size):
            average_path_1[j] = average_path_1[j] + simulation_path_1[j]
            average_path_2[j] = average_path_2[j] + simulation_path_2[j]
        #mp.plot(simulation_times, simulation_path)
    
    for j in range(simulation_path_1.size):
            average_path_1[j] = average_path_1[j]/num_simulations
            average_path_2[j] = average_path_2[j]/num_simulations
        
    return simulation_times_1, average_path_1, average_path_2

def average_CIR_all_simulations(time_discretization, num_simulations, model, initial_value):
    time_start = 0.0
    time_end = 10.0

    brownian_times, brownian_path = make_path(time_discretization, time_start, time_end)

    average_path_1 = np.zeros(brownian_path.size)
    average_path_2 = np.zeros(brownian_path.size)
    average_path_3 = np.zeros(brownian_path.size)
    average_path_4 = np.zeros(brownian_path.size)
    average_path_5 = np.zeros(brownian_path.size)
    average_path_6 = np.zeros(brownian_path.size)
    average_path_7 = np.zeros(brownian_path.size)

    for i in range(num_simulations):
        brownian_times, brownian_path = make_path(time_discretization, time_start, time_end)
        
        simulation_times_1, simulation_path_1 = CIRApproximators.run_abs_EM(brownian_path, brownian_times, model, initial_value)
        simulation_times_2, simulation_path_2 = CIRApproximators.run_max_EM(brownian_path, brownian_times, model, initial_value)
        simulation_times_3, simulation_path_3 = CIRApproximators.run_total_max_EM(brownian_path, brownian_times, model, initial_value)
        simulation_times_4, simulation_path_4 = CIRApproximators.run_abs_Milstein(brownian_path, brownian_times, model, initial_value)
        simulation_times_5, simulation_path_5 = CIRApproximators.run_max_Milstein(brownian_path, brownian_times, model, initial_value)
        simulation_times_6, simulation_path_6 = CIRApproximators.run_total_max_Milstein(brownian_path, brownian_times, model, initial_value)
        simulation_times_7, simulation_path_7 = CIRApproximators.run_exact(brownian_path, brownian_times, model, initial_value)

        for j in range(simulation_path_1.size):
            average_path_1[j] = average_path_1[j] + simulation_path_1[j]
            average_path_2[j] = average_path_2[j] + simulation_path_2[j]
            average_path_3[j] = average_path_3[j] + simulation_path_3[j]
            average_path_4[j] = average_path_4[j] + simulation_path_4[j]
            average_path_5[j] = average_path_5[j] + simulation_path_5[j]
            average_path_6[j] = average_path_6[j] + simulation_path_6[j]
            average_path_7[j] = average_path_7[j] + simulation_path_7[j]
        #mp.plot(simulation_times, simulation_path)
    
    for j in range(simulation_path_1.size):
            average_path_1[j] = average_path_1[j]/num_simulations
            average_path_2[j] = average_path_2[j]/num_simulations
            average_path_3[j] = average_path_3[j]/num_simulations
            average_path_4[j] = average_path_4[j]/num_simulations
            average_path_5[j] = average_path_5[j]/num_simulations
            average_path_6[j] = average_path_6[j]/num_simulations
            average_path_7[j] = average_path_7[j]/num_simulations
        
    return simulation_times_1, average_path_1, average_path_2, average_path_3, average_path_4, average_path_5, average_path_6, average_path_7

def main():
    brownian_times, brownian_path = make_path(10)
    
    initial_value = 43

    for i in range(1):
        #brownian_times, brownian_path = make_path()
        #simulation_times, simulation_path = Week3EulerMaruyama.run_EM(brownian_path, brownian_times, drift_function, variance_function, initial_value)
        
        simulation_times, simulation_path = BS_exact_solution(brownian_path, brownian_times, initial_value)
        mp.plot(simulation_times, simulation_path)
        simulation_times, simulation_path = average_EM_simulations(initial_value, 500, 10)
        mp.plot(simulation_times, simulation_path)
        simulation_times, simulation_path = average_Milstein_simulations(initial_value, 500, 10)
        mp.plot(simulation_times, simulation_path)

    mp.xlabel("Time")
    mp.ylabel("x")
    mp.show()

    return

def compare_cir_all2():
    model = CIRApproximators.CIR_Model(s = 0.44, m=0.06, v=0.32)
    
    initial_value = 0.06

    discretization = -5

    num_simulations = 10000

    simulation_times, simulation_path1, simulation_path2, simulation_path3, simulation_path4, simulation_path5, simulation_path6, simulation_path7 = average_CIR_all_simulations(discretization, num_simulations, model, initial_value)
    mp.plot(simulation_times, simulation_path1, label = "Absolute Value Euler")
    mp.plot(simulation_times, simulation_path2, label = "Square Root Truncation Euler")
    mp.plot(simulation_times, simulation_path3, label = "Full Truncation Euler")
    mp.plot(simulation_times, simulation_path4, label = "Absolute Value Milstein")
    mp.plot(simulation_times, simulation_path5, label = "Square Root Truncation Milstein")
    mp.plot(simulation_times, simulation_path6, label = "Full Truncation Milstein")
    mp.plot(simulation_times, simulation_path7, label = "Exact")

    mp.legend(loc="best")
    mp.xlabel("Time")
    mp.ylabel("x")
    mp.show()

    return

def compare_cir_all():
    model = CIRApproximators.CIR_Model(s = 0.44, m=0.2, v=0.5)
    
    initial_value = 0.24

    discretization = -4

    num_simulations = 10000

    simulation_times, simulation_path1, simulation_path2, simulation_path3, simulation_path4, simulation_path5, simulation_path6, simulation_path7 = average_CIR_all_simulations(discretization, num_simulations, model, initial_value)
    mp.plot(simulation_times, simulation_path1, label = "Absolute Value Euler")
    mp.plot(simulation_times, simulation_path2, label = "Square Root Truncation Euler")
    mp.plot(simulation_times, simulation_path3, label = "Full Truncation Euler")
    mp.plot(simulation_times, simulation_path4, label = "Absolute Value Milstein")
    mp.plot(simulation_times, simulation_path5, label = "Square Root Truncation Milstein")
    mp.plot(simulation_times, simulation_path6, label = "Full Truncation Milstein")
    mp.plot(simulation_times, simulation_path7, label = "Exact")

    mp.legend(loc="upper right")
    mp.xlabel("Time")
    mp.ylabel("x")
    mp.show()

    return

def compare_cir_milstein():
    model = CIRApproximators.CIR_Model(s = 0.44, m=0.06, v=0.32)
    
    initial_value = 0.06

    discretization = -5

    num_simulations = 10000
  
    simulation_times, simulation_path1, simulation_path2, simulation_path3 = average_CIR_Milstein_simulations(discretization, num_simulations, model, initial_value)
    mp.plot(simulation_times, simulation_path1, label = "Absolute Value Milstein")
    mp.plot(simulation_times, simulation_path2, label = "Square Root Truncation Milstein")
    mp.plot(simulation_times, simulation_path3, label = "Full Truncation Milstein")

    mp.legend(loc="best")
    mp.xlabel("Time")
    mp.ylabel("x")
    mp.show()

    return

def compare_cir_em():
    model = CIRApproximators.CIR_Model(s = 0.44, m=0.06, v=0.32)
    
    initial_value = 0.06

    discretization = -5

    num_simulations = 10000
  
    simulation_times, simulation_path1, simulation_path2, simulation_path3 = average_CIR_EM_simulations(discretization, num_simulations, model, initial_value)
    mp.plot(simulation_times, simulation_path1, label = "Absolute Value Euler")
    mp.plot(simulation_times, simulation_path2, label = "Square Root Truncation Euler")
    mp.plot(simulation_times, simulation_path3, label = "Full Truncation Euler")

    mp.legend(loc="best")
    mp.xlabel("Time")
    mp.ylabel("x")
    mp.show()

    return

def compare_cir_2em():
    model = CIRApproximators.CIR_Model(s = 0.44, m=0.06, v=0.32)
    
    initial_value = 0.06

    discretization = -5

    num_simulations = 1
  
    simulation_times, simulation_path1, simulation_path2= average_CIR_2EM_simulations(discretization, num_simulations, model, initial_value)
    mp.plot(simulation_times, simulation_path1, label = "Absolute Value Euler")
    mp.plot(simulation_times, simulation_path2, label = "Full Truncation Euler")

    mp.legend(loc="best")
    mp.axhline(y=0, c="k", linewidth = 0.5)
    mp.xlabel("Time")
    mp.ylabel("x")
    mp.show()

    return

def test():
    model = CIRApproximators.CIR_Model(s = 0.44, m=0.06, v=0.32)

    initial_value = 0.06

    brownian_times, brownian_path = make_path(10, 0.00, 5.0)

    average_path = np.zeros(brownian_path.size)

    for i in range(10000):
        simulation_times_4, simulation_path_4 = CIRApproximators.run_max_Milstein(brownian_path, brownian_times, model, initial_value)

        for j in range(simulation_path_4.size):
             average_path[j] = average_path[j] + simulation_path_4[j]

    for i in range(10000):
        average_path[j] = average_path[j]/1000

    mp.plot(simulation_times_4, average_path)

    mp.xlabel("Time")
    mp.ylabel("x")
    mp.show()

if __name__ == "__main__":

    compare_cir_all()
    #compare_cir_2em()