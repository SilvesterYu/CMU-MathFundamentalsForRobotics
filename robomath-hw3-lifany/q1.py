import numpy as np
import matplotlib.pyplot as plt

#(b)
low, high = 0, 2

def f(x):
  return np.cos((np.pi/2)*(x))

x_plot = np.arange(low-0.01, high+0.02, .01)
y_plot = [f(item) for item in x_plot]
plt.figure(figsize = (8, 5))
plt.plot(x_plot, y_plot, "limegreen", label = "f(x)")
plt.xlabel("x")
leg = plt.legend(loc='lower left')
#plt.show()
plt.savefig("results/q1-b.png")

# (c)
def p(x):
  return -1.13821685*x + 1.13821685

x_plot = np.arange(low-0.01, high+0.02, .01)
y_plot = [f(item) for item in x_plot]
plt.figure(figsize = (8, 5))
plt.plot(x_plot, y_plot, "limegreen", label = "f(x)")
y_line = [p(x) for x in x_plot]
plt.plot(x_plot, y_line, "darkgreen", linestyle='--', label="approximating polynomial")
xi = [0, 0.515961406, 1.484038594, 2]
yi1 = [f(xi[0]), p(xi[1]), f(xi[2]), p(xi[3])]
yi2 = [p(xi[0]), f(xi[1]), p(xi[2]), f(xi[3])]
plt.vlines(x=xi, ymin=yi1, ymax=yi2, colors='teal', ls='-', lw=2, label='e(xi)')
plt.xlabel("x")
leg = plt.legend(loc='lower left')
#plt.show()
plt.savefig("results/q1-c.png")

# (D)
def p2(x):
  return -1.215854*x + 1.215854

x_plot = np.arange(low-0.01, high+0.02, .01)
y_plot = [f(item) for item in x_plot]
plt.figure(figsize = (8, 5))
plt.plot(x_plot, y_plot, "cyan", label = "f(x)")
y_line = [p(x) for x in x_plot]
plt.plot(x_plot, y_line, "green", linestyle='--', label="best uniform approximation")
y_line2 = [p2(x) for x in x_plot]
plt.plot(x_plot, y_line2, "darkblue", linestyle='--', label="least squares approximation")
plt.xlabel("x")
leg = plt.legend(loc='lower left')
#plt.show()
plt.savefig("results/q1-d.png")

