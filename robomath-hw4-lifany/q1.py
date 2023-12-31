import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# the true function
def y(x):
    return np.cbrt(x**2)

# y'(x)
def y_prime(y):
    return 2/(3*y**0.5)

# create a table with x, true y, numerical y and error
def create_table(x_vals, y_true, y_numerical, my_title=""):
    table = np.stack([x_vals, y_true, y_numerical, y_true - y_numerical], axis = 1)[::-1]
    df = pd.DataFrame(data = table)
    df.columns = ["x_i", "True y(x_i)", "Numerical y_i", "Err y(x_i) - y_i"]
    err = np.absolute(y_true - y_numerical)
    max_err, mean_err = np.max(err), np.mean(err)
    print("\nTable for " + my_title + " :\n", df)
    print("\nError metric: mean absolute error\n", "Maximum error", max_err, "\n", "Mean error", mean_err)
    return

# plot the results
def plotting(x_vals, y_true, y_numerical, my_title=""):
    plt.plot(x_vals, y_true, color ='limegreen', linewidth = 2, label ='True y')
    plt.plot(x_vals, y_numerical, color ='darkgreen', linestyle ='dashed', linewidth = 2, label ='Numerical y')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(my_title)
    plt.legend()
    plt.savefig("results/" + "_".join(my_title.split()) + ".png")
    plt.show()

def plot_multiple(x_vals, y_true, y1, y2, y3, my_title=""):
    plt.plot(x_vals, y_true, color ='limegreen', linewidth = 5, label ='True y')
    plt.plot(x_vals, y1, color ='purple', linestyle ='dashed', linewidth = 2, label ='Numerical y Euler')
    plt.plot(x_vals, y2, color ='red', linestyle ='dashed', linewidth = 2, label ='Numerical y Runge-Kutta')
    plt.plot(x_vals, y3, color ='blue', linestyle ='dashed', linewidth = 2, label ='Numerical y Adams-Bashforth')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(my_title)
    plt.legend()
    plt.savefig("results/" + "_".join(my_title.split()) + ".png")
    plt.show()
    
# question (b)
def Euler(x_vals, x0, y0):
    y_numerical = [y0]
    sign = 1 if x0 == interval[0] else -1
    yn = y0
    for i in range(len(x_vals)-1):
        yn1 = yn + sign * h * y_prime(yn)
        y_numerical.append(yn1)
        yn = yn1
    return np.array(y_numerical[::sign])
  
# question (c)
def RungeKutta(x_vals, x0, y0):
    y_numerical = [y0]
    sign = 1 if x0 == interval[0] else -1
    yn = y0
    for i in range(len(x_vals)-1):
        k1 = h * sign * y_prime(yn)
        k2 = h * sign * y_prime(yn + k1/2)
        k3 = h * sign * y_prime(yn + k2/2)
        k4 = h * sign * y_prime(yn + k3)
        yn1 = yn + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        y_numerical.append(yn1)
        yn = yn1
    return np.array(y_numerical[::sign])

# question (d)
def AdamsBashforth(x_vals, x0, y0):
    y_numerical = [y0]
    sign = 1 if x0 == interval[0] else -1
    yn, yn_1, yn_2, yn_3 = y0, 1.03306155414651, 1.06560223676661, 1.09765339982501
    for i in range(len(x_vals)-1):
        fn, fn_1, fn_2, fn_3 = y_prime(yn), y_prime(yn_1), y_prime(yn_2), y_prime(yn_3)
        yn1 = yn + (h/24) * sign * (55 * fn - 59 * fn_1 + 37 * fn_2 - 9 * fn_3)
        y_numerical.append(yn1)
        yn, yn_1, yn_2, yn_3 = yn1, yn, yn_1, yn_2
    return np.array(y_numerical[::sign])

if __name__ == "__main__":
    # Known data
    h = 0.05
    interval = [0, 1]
    x_vals = np.arange(interval[0], interval[1] + h / 2, h)
    x0, y0 = 1, 1
    y_true = y(x_vals)

    # Question (b)
    y_numerical1 = Euler(x_vals, x0, y0)
    create_table(x_vals, y_true, y_numerical1, "Euler's method")
    plotting(x_vals, y_true, y_numerical1, "Euler's method")

    # Question (c)
    y_numerical2 = RungeKutta(x_vals, x0, y0)
    create_table(x_vals, y_true, y_numerical2, "Runge-Kutta method")
    plotting(x_vals, y_true, y_numerical2, "Runge-Kutta method")

    # Question (d)
    y_numerical3 = AdamsBashforth(x_vals, x0, y0)
    create_table(x_vals, y_true, y_numerical3, "Adams-Bashforth method")
    plotting(x_vals, y_true, y_numerical3, "Adams-Bashforth method")

    # Visualize all results
    plot_multiple(x_vals, y_true, y_numerical1, y_numerical2, y_numerical3, my_title="All results")