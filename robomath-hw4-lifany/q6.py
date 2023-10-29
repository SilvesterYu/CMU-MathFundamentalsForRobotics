import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Ellipse
from mpl_toolkits.mplot3d import Axes3D, art3d


def main():
    obstacle_cost = generate_cost()
    gx, gy = np.gradient(obstacle_cost)

    start_point = np.array([10, 10])
    end_point = np.array([90, 90])
    vector = end_point - start_point

    num_pts = 300
    initial_path = start_point + \
        np.outer(np.linspace(0, 1, num_pts), vector)

    plot_scene(initial_path, obstacle_cost)

    # FURTHER CODE HERE


def generate_cost():
    n = 101
    obstacles = np.array([[20, 30], [60, 40], [70, 85]])
    epsilon = np.array([[25], [20], [30]])
    obstacle_cost = np.zeros((n, n))
    for i in range(obstacles.shape[0]):
        t = np.ones((n, n))
        t[obstacles[i, 0], obstacles[i, 1]] = 0
        t_cost = distance_transform_edt(t)
        t_cost[t_cost > epsilon[i]] = epsilon[i]
        t_cost = (1 / (2 * epsilon[i])) * (t_cost - epsilon[i])**2
        obstacle_cost += + t_cost
    return obstacle_cost


def get_values(path, cost):
    x, y = path.astype(int).T
    return cost[x, y].reshape((path.shape[0], 1))


def plot_scene(path, cost):
    values = get_values(path, cost)

    # Plot 2D
    plt.imshow(cost.T)
    plt.plot(path[:, 0], path[:, 1], "ro")

    # Plot 3D
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection="3d")
    xx, yy = np.meshgrid(range(cost.shape[1]), range(cost.shape[0]))
    ax3d.plot_surface(xx, yy, cost.T, cmap=plt.get_cmap("coolwarm"))
    ax3d.scatter(path[:, 0], path[:, 1], values, s=20, c="r")
    plt.show()

if __name__ == "__main__":
    main()