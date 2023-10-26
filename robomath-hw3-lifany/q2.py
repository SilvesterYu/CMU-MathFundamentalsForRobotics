import numpy as np
import matplotlib.pyplot as plt

f = open("data/problem2.txt", "r" )
str_data = ""
for line in f:
  str_data += line
f.close()
data = np.array([float(item) for item in str_data.split()])
x = np.arange(0, 1.0001, 0.01)

# visualize f(x)
plt.figure(figsize = (8, 5))
plt.plot(x, data, "limegreen", label = "f(x)")
plt.xlabel("x")
leg = plt.legend(loc='lower left')
plt.savefig("results/q2-data.png")

# basis functions
phi0 = np.array([1 for i in range(len(x))])
phi1 = x
phi2 = np.sin(2*np.pi * x)

# matrices of basis functions and errors
A = np.asmatrix((phi0, phi1, phi2)).T
fs = np.asmatrix((data)).T
P = np.matmul(A.T, A)
q = np.matmul(A.T, fs)

# the coefficients
c = np.linalg.solve(P, q)
approx_f = np.ravel(np.matmul(A, c))
print("functions: 1, x, sin(2pi x)")
print("coefficients: ", c, sep = "\n")

# visualize the polynomials
plt.figure(figsize = (8, 5))
plt.plot(x, data, "palegreen", linewidth=5, label = "f(x)")
plt.plot(x, approx_f, "darkgreen", linestyle='--', label = "linear combination")
plt.xlabel("x")
leg = plt.legend(loc='lower left')
plt.savefig("results/q2-polynomial.png")














