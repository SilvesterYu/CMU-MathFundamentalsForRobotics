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
    table = np.hstack([x_vals, y_true, y_numerical, y_true - y_numerical])
    print(table)
    err = np.absolute(y_true - y_numerical)
    max_err, mean_err = np.max(err), np.mean(err)
    print("Maximum error", max_err, "Mean error", mean_err)
    return

# plot the results
def plotting(x_vals, y_true, y_numerical, my_title=""):
    plt.plot(x_vals, y_true, color ='limegreen', linewidth = 2, label ='True y')
    plt.plot(x_vals, y_numerical, color ='darkgreen', linestyle ='dashed', linewidth = 2, label ='Numerical y')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(my_title)
    plt.legend()
    plt.savefig("_".join(my_title.split()) + ".png")
    plt.show()
    
# question (a)
def Euler(x_vals, x0, y0):
    y_numerical = [y0]
    sign = 1 if x0 == interval[0] else -1
    yn = y0

    for i in range(len(x_vals)-1):
        yn1 = yn + sign * h * y_prime(yn)
        y_numerical.append(yn1)
        yn = yn1
    
    return np.array(y_numerical[::sign])


if __name__ == "__main__":
    # Known data
    h = 0.05
    interval = [0, 1]
    x_vals = np.arange(interval[0], interval[1] + h / 2, h)
    x0, y0 = 1, 1
    y_true = y(x_vals)

    # Question a
    y_numerical1 = Euler(x_vals, x0, y0)
    print("(a) Euler's method y true ", y_true)
    print("(a) Euler's method y numerical ", y_numerical1)
    create_table(x_vals, y_true, y_numerical1, my_title="")
    plotting(x_vals, y_true, y_numerical1, "Euler's method")
