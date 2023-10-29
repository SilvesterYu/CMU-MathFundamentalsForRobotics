import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Ellipse
from mpl_toolkits.mplot3d import Axes3D, art3d
from scipy.interpolate import RectBivariateSpline
import copy


def main():
    obstacle_cost = generate_cost()
    gx, gy = np.gradient(obstacle_cost)

    start_point = np.array([10, 10])
    end_point = np.array([90, 90])
    vector = end_point - start_point

    num_pts = 300
    initial_path = start_point + \
        np.outer(np.linspace(0, 1, num_pts), vector)

    # plot_scene(initial_path, obstacle_cost)
    #plot_path(initial_path, obstacle_cost)

    # FURTHER CODE HERE
    print(obstacle_cost.shape)

    # (a)
    #new_path = gradient_optimization(initial_path, gx, gy)
    #plot_path(new_path, obstacle_cost)

    # (b)
    # new_path2 = gradient_optimization2(initial_path, gx, gy)
    # plot_path(new_path2, obstacle_cost)

    # (c)
    new_path3 = gradient_optimization3(initial_path, gx, gy, 5000)
    plot_path(new_path3, obstacle_cost)
    new_path3 = gradient_optimization3(initial_path, gx, gy, 100)
    plot_path(new_path3, obstacle_cost)

# --#
# for question (a)
def gradient_optimization(initial_path, gx, gy, l2_thresh = 0.02):

    N = len(initial_path)
    l2_diff = np.Inf
    new_path, curr_path = copy.deepcopy(initial_path)[1:-1], copy.deepcopy(initial_path)[1:-1]
    xarr, yarr = np.arange(0, gx.shape[1]), np.arange(0, gx.shape[0])
    gx_interp, gy_interp = RectBivariateSpline(xarr, yarr, gx), RectBivariateSpline(xarr, yarr, gy)
    iter = 0

    while l2_diff > l2_thresh:
        xs, ys = curr_path[:, 0], curr_path[:, 1]
        us = [gx_interp.ev(xs, ys), gy_interp.ev(xs, ys)]
        xs_new, ys_new = xs - 0.1*us[0], ys - 0.1*us[1]
        new_path[:, 0], new_path[:, 1] = xs_new, ys_new 
        l2_diff = np.linalg.norm(new_path - curr_path)
        curr_path = copy.deepcopy(new_path)
        print("iter", iter, "diff", l2_diff)
        iter += 1
        #break

    new_path = np.vstack((np.vstack((initial_path[0].reshape(1, 2), new_path)), initial_path[-1].reshape(1, 2)))
    print(new_path)
    return new_path

# for question (b)
def gradient_optimization2(initial_path, gx, gy, l2_thresh = 0.02):

    N = len(initial_path)
    l2_diff = np.Inf
    new_path, curr_path = copy.deepcopy(initial_path)[1:-1], copy.deepcopy(initial_path)[1:-1]
    xarr, yarr = np.arange(0, gx.shape[1]), np.arange(0, gx.shape[0])
    gx_interp, gy_interp = RectBivariateSpline(xarr, yarr, gx), RectBivariateSpline(xarr, yarr, gy)
    iter = 0

    while l2_diff > l2_thresh:
        xs, ys = curr_path[:, 0], curr_path[:, 1]
        us = [gx_interp.ev(xs, ys), gy_interp.ev(xs, ys)]
        # --modification
        prev_path= np.vstack((initial_path[0].reshape(1, 2), curr_path[:-1]))
        xs_prev, ys_prev = prev_path[:, 0], prev_path[:, 1]
        smoothness_cost_x, smoothness_cost_y = (xs - xs_prev), (ys - ys_prev)
        x_step = - 0.8*us[0] - 4*smoothness_cost_x
        y_step = - 0.8*us[1] - 4*smoothness_cost_y
        xs_new, ys_new = xs + 0.1*x_step, ys + 0.1*y_step
        # -- end of modification
        new_path[:, 0], new_path[:, 1] = xs_new, ys_new 
        l2_diff = np.linalg.norm(new_path - curr_path)
        curr_path = copy.deepcopy(new_path)
        print("iter", iter, "diff", l2_diff)
        if iter == 100:
            break
        iter += 1

    new_path = np.vstack((np.vstack((initial_path[0].reshape(1, 2), new_path)), initial_path[-1].reshape(1, 2)))
    return new_path

# for question (c)
def gradient_optimization3(initial_path, gx, gy, iter_thresh):

    N = len(initial_path)
    l2_diff = np.Inf
    new_path, curr_path = copy.deepcopy(initial_path)[1:-1], copy.deepcopy(initial_path)[1:-1]
    xarr, yarr = np.arange(0, gx.shape[1]), np.arange(0, gx.shape[0])
    gx_interp, gy_interp = RectBivariateSpline(xarr, yarr, gx), RectBivariateSpline(xarr, yarr, gy)
    iter = 0

    while iter < iter_thresh:
        xs, ys = curr_path[:, 0], curr_path[:, 1]
        us = [gx_interp.ev(xs, ys), gy_interp.ev(xs, ys)]
        # --modification
        prev_path= np.vstack((initial_path[0].reshape(1, 2), curr_path[:-1]))
        next_path = np.vstack((curr_path[1:], initial_path[-1].reshape(1, 2)))
        xs_prev, ys_prev = prev_path[:, 0], prev_path[:, 1]
        xs_next, ys_next = next_path[:, 0], next_path[:, 1]
        smoothness_cost_x, smoothness_cost_y = -xs_prev + 2*xs - xs_next, -ys_prev + 2*ys - ys_next
        # -- end of modification
        x_step = - 0.8*us[0] - 4*smoothness_cost_x
        y_step = - 0.8*us[1] - 4*smoothness_cost_y
        xs_new, ys_new = xs + 0.1*x_step, ys + 0.1*y_step
        new_path[:, 0], new_path[:, 1] = xs_new, ys_new 
        l2_diff = np.linalg.norm(new_path - curr_path)
        curr_path = copy.deepcopy(new_path)
        print("iter", iter, "diff", l2_diff)
        iter += 1

    new_path = np.vstack((np.vstack((initial_path[0].reshape(1, 2), new_path)), initial_path[-1].reshape(1, 2)))
    return new_path



# --

# for i in range(1, N-1):
        #     x, y = curr_path[i][0], curr_path[i][1]
        #     print(x, y)
        #     u = [float(gx_interp.ev(x, y)), float(gy_interp.ev(x, y))]
        #     print("ev", u)
        #     if u[0] != 0 and u[1] != 0:
        #         x_new, y_new = x - 0.1*u[0], y - 0.1*u[1]
        #         # print("new path i", new_path[i], curr_path[i])
        #         new_path[i][:] = np.array([x_new, y_new])
        #         # print("new path i again", new_path[i], curr_path[i])
# --

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

### for 3D plotting ###

def add_point(ax, x, y, z, fc = None, ec = None, radius = 0.005):
    # https://stackoverflow.com/a/65115447/5487412
       xy_len, z_len = ax.get_figure().get_size_inches()
       axis_length = [x[1] - x[0] for x in [ax.get_xbound(), ax.get_ybound(), ax.get_zbound()]]
       axis_rotation =  {'z': ((x, y, z), axis_length[1]/axis_length[0]),
                         'y': ((x, z, y), axis_length[2]/axis_length[0]*xy_len/z_len),
                         'x': ((y, z, x), axis_length[2]/axis_length[1]*xy_len/z_len)}
       for a, ((x0, y0, z0), ratio) in axis_rotation.items():
           p = Ellipse((x0, y0), width = radius, height = radius*ratio, fc=fc, ec=ec)
           ax.add_patch(p)
           art3d.pathpatch_2d_to_3d(p, z=z0, zdir=a)

def plot_path(path, obs_cost, figsize=(7,7)):
    N = obs_cost.shape[0]
    tt = path.shape[0]
    path_values = np.zeros((tt, 1))
    for i in range(tt):
        path_values[i] = obs_cost[int(np.floor(path[i, 0])), int(np.floor(path[i, 1]))]

    # Plot 2D
    plt.figure(figsize=figsize)
    plt.imshow(obs_cost.T)
    plt.plot(path[:, 0], path[:, 1], 'ro')

    # Plot 3D
    fig3d = plt.figure(figsize=figsize)
    ax3d = fig3d.add_subplot(111, projection='3d')
    xx, yy = np.meshgrid(range(N), range(N))
    ax3d.plot_surface(xx, yy, obs_cost.T, cmap=plt.get_cmap('coolwarm'))
    ax3d.scatter(path[:, 0], path[:, 1], path_values, s=20, c='r', alpha=1)
    for i,(x,y) in enumerate(path):
        z = path_values[i][0]
        add_point(ax3d, x, y, z, fc="r", radius=1)
    ax3d.view_init(elev=47, azim=27)
    plt.show()

### end of 3D plotting ###

if __name__ == "__main__":
    main()