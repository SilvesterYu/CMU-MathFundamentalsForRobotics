import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def read_points(fname):
  f = open(fname, "r" )
  x = []
  y = []
  z = []
  for line in f:
    x.append(float(line.split()[0]))
    y.append(float(line.split()[1]))
    z.append(float(line.split()[2]))
  f.close()
  x = np.array(x)
  y = np.array(y)
  z = np.array(z)
  
  return x, y, z

def fit_plane(x, y, z, d):
  # use linear regression to fit a plane ax + by + cz + d = 0
  # z = (-d-ax-by)/c
  # calculate the z's for the plane
  A = np.asmatrix((x, y, z)).T
  b = np.asmatrix(np.array([-d for i in range(len(x))])).T
  U, Sigma, VT = np.linalg.svd(A, full_matrices = False)
  Sigma = 1/Sigma
  cs = np.ravel(np.matmul(VT.T, np.matmul(np.diag(Sigma), np.matmul(U.T, b))))
  c1, c2, c3 = cs[0], cs[1], cs[2]
  xr = np.arange(np.min(x), np.max(x)+0.01, 0.01)
  yr = np.arange(np.min(y), np.max(y)+0.01, 0.01)
  xx, yy = np.meshgrid(xr, yr)
  z_plane = (-d-c1*xx-c2*yy)/c3
  return z_plane, xx, yy, cs

def plotting(fname, savename, x, y, z, xx, yy, z_plane, best_in_idx=None):
  # plot the fitted plane
  fig = plt.figure(figsize=(12,10))
  ax = fig.add_subplot(projection='3d')
  if best_in_idx != None:
    all_idx = np.arange(0, len(x))
    ax.scatter(x[best_in_idx], y[best_in_idx], z[best_in_idx], s=15, c='darkgreen', marker='o')
    out_idx = np.setdiff1d(all_idx, best_in_idx)
    ax.scatter(x[out_idx], y[out_idx], z[out_idx], s=3, c='limegreen', marker='o')
  else:
    ax.scatter(x, y, z, s=3, c='limegreen', marker='o')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.savefig("results/q4-" + savename + "-data.png")
  ax.plot_surface(xx, yy, z_plane, color="lawngreen", alpha=0.3)
  # plt.show()
  plt.savefig("results/q4-" + savename + "-plane.png")

# calculates the distance of a point to the fitted plane
def calc_dist(x, y, z, d, cs):
  A, B, C = cs[0], cs[1], cs[2]
  const = 1/(np.sqrt(A**2 + B**2 + C**2))
  M = np.asmatrix((x, y, z, np.array([1 for i in range(len(x))]))).T
  abcd = np.asmatrix((np.array([A, B, C, d]))).T
  dists = np.abs(np.ravel(const * np.matmul(M, abcd)))
  return dists

# questions (a) and (b)
def find_plane(fname, savename, d=1):
  # get points and fit the plane
  x, y, z = read_points(fname)
  z_plane, xx, yy, cs = fit_plane(x, y, z, d)
  mean_dist = np.mean(calc_dist(x, y, z, d, cs))
  print("Average distance from point to plane: ", mean_dist)
  plotting(fname, savename, x, y, z, xx, yy, z_plane)

# questions (c)
def ransac_plane(fname, savename, n=5, dist_thresh=0.005, max_iter=500, d=1):
  # dist_thresh: thrshold to be considered as inlier
  X, Y, Z = read_points(fname)
  it, best_in_idx, best_ratio = 0, 0, 0
  while it < max_iter:
    idx = random.sample(range(0, len(X)), n)
    x, y, z = X[idx], Y[idx], Z[idx]
    z_plane, xx, yy, cs = fit_plane(x, y, z, d)
    dists = calc_dist(X, Y, Z, d, cs)
    # finding the dominant plane
    in_idx = np.where(dists < dist_thresh)
    ratio = (np.count_nonzero(in_idx)) / len(X)
    if ratio > best_ratio:
      best_in_idx = in_idx
      best_ratio = ratio
    #print(ratio)
    it += 1
  inx, iny, inz = X[best_in_idx], Y[best_in_idx], Z[best_in_idx]
  best_z_plane, best_xx, best_yy, _ = fit_plane(inx, iny, inz, d)
  plotting(fname, savename, X, Y, Z, best_xx, best_yy, best_z_plane, best_in_idx)

# question (d)
def ransac_planes(fname, savename, ratio_thresh, n=5, dist_thresh=0.05, d=1, num_planes=4):
  X, Y, Z = read_points(fname)
  num_all = len(X)

  fig = plt.figure(figsize=(12,10))
  ax = fig.add_subplot(projection='3d')
  #ax.scatter(X, Y, Z, s=3, c='limegreen', marker='o')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  for i in range(num_planes):
    print(len(X))
    all_idx = np.arange(0, len(X))
    best_in_idx, best_ratio = 0, 0
    while best_ratio < ratio_thresh:
      idx = random.sample(range(0, len(X)), n)
      x, y, z = X[idx], Y[idx], Z[idx]
      z_plane, xx, yy, cs = fit_plane(x, y, z, d)
      dists = calc_dist(X, Y, Z, d, cs)
      # finding the dominant plane
      in_idx = np.where(dists < dist_thresh)
      ratio = (np.count_nonzero(in_idx)) / num_all
      if ratio > best_ratio:
        best_in_idx = in_idx
        best_ratio = ratio
    #print(best_ratio)
    inx, iny, inz = X[best_in_idx], Y[best_in_idx], Z[best_in_idx]
    best_z_plane, best_xx, best_yy, _ = fit_plane(inx, iny, inz, d)
    ax.plot_surface(best_xx, best_yy, best_z_plane, color=(0, i/num_planes, 0), alpha=0.3)
    ax.scatter(inx, iny, inz, s=5, c='darkgreen', marker='o')
    out_idx = np.setdiff1d(all_idx, in_idx)
    X, Y, Z = X[out_idx], Y[out_idx], Z[out_idx]
    plt.savefig("results/q4-" + savename + str(i) + "-plane.png")
  ax.scatter(X, Y, Z, s=5, c='limegreen', marker='o')
  # plt.show()
  plt.savefig("results/q4-" + savename + "-plane.png")

# question (e)
def ransac_planes_clutter(fname, savename, max_iter=500, n=5, ratio_threshes = [0.18, 0.18, 0.045, 0.05], dist_thresh=0.012, mean_dist_thresh = 0.05, d=1, num_planes=4):
  X, Y, Z = read_points(fname)
  print("original X", X, X.shape)
  num_all = len(X)

  fig = plt.figure(figsize=(12,10))
  ax = fig.add_subplot(projection='3d')
  #ax.scatter(X, Y, Z, s=3, c='limegreen', marker='o')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  for i in range(num_planes):
    ratio_thresh = ratio_threshes[i]
    print("here", len(X))
    print("mean_dist_thresh", mean_dist_thresh, "ratio_thresh", ratio_thresh)
    all_idx = np.arange(0, len(X))
    it, best_in_idx, best_ratio, best_new_in_idx= 0, 0, 0, 0
    min_dist = np.Inf
    while min_dist > mean_dist_thresh or best_ratio < ratio_thresh:  
      if i <= 1:
        idx = random.sample(range(0, len(X)), n)
        x, y, z = X[idx], Y[idx], Z[idx]
      else:
        #print("mean X", mean_X)
        X_small_idx = [i for i in range(X.shape[0]) if X[i] < 0]
        X_large_idx = [i for i in range(X.shape[0]) if X[i] >= 0]
        
        if i%2 != 0:
          idx0 = random.sample(range(0, len(X_small_idx)), n)
          idx = [X_small_idx[j] for j in idx0]
        else:
          idx0 = random.sample(range(0, len(X_large_idx)), n)
          idx = [X_large_idx[j] for j in idx0]
        
        x, y, z = X[idx], Y[idx], Z[idx]
        #print("x---", x)

      z_plane, xx, yy, cs = fit_plane(x, y, z, d)
      dists = calc_dist(X, Y, Z, d, cs)
      # finding the dominant plane
      if i <= 1:
        in_idx = np.where(dists < dist_thresh)
      else:
        in_idx = np.where(dists < dist_thresh*1.2)
      ratio = (np.count_nonzero(in_idx)) / num_all
      this_inx, this_iny, this_inz = X[in_idx], Y[in_idx], Z[in_idx]
      _, _, _, this_cs = fit_plane(this_inx, this_iny, this_inz, d)
      dists = calc_dist(this_inx, this_iny, this_inz, d, this_cs)
      mean_dist = np.mean(dists)
      if (i<= 1 and ratio > best_ratio and mean_dist < min_dist) or (i > 1  and ratio > best_ratio):
        print("min_dist", mean_dist, "ratio", ratio)
        #best_new_in_idx = np.where(dists < dist_thresh*1.5)
        best_in_idx = in_idx
        best_ratio = ratio
        min_dist = mean_dist
    print("fitted plane ", i+1, " inliers ratio", best_ratio)
    print("fitted plane ", i+1, " inliers mean distance", min_dist)
    inx, iny, inz = X[best_in_idx], Y[best_in_idx], Z[best_in_idx]
    best_z_plane, best_xx, best_yy, best_cs = fit_plane(inx, iny, inz, d)
    ax.plot_surface(best_xx, best_yy, best_z_plane, color=(0, 1/num_planes, 0), alpha=0.3)
    ax.scatter(inx, iny, inz, s=5, c='darkgreen', marker='o')
    out_idx = np.setdiff1d(all_idx, best_in_idx)
    X, Y, Z = X[out_idx], Y[out_idx], Z[out_idx]
    #ax.scatter(X, Y, Z, s=5, c='limegreen', marker='o')
    plt.savefig("results/q4-"  + savename + str(i) + "-plane.png")
    print("best cs", best_cs)
    

  ax.scatter(X, Y, Z, s=5, c='limegreen', marker='o')
  #plt.show()
  plt.savefig("results/q4-" + savename + "-plane.png")

if __name__ == "__main__":
  
  # (a)
  print("running question (a)", "-"*40)
  print()
  find_plane("data/clean_table.txt", "a-clean-table")

  # (b)
  print("running question (b)", "-"*40)
  print()
  find_plane("data/cluttered_table.txt", "b-cluttered-table")
  
  # (c)
  print("running question (c)", "-"*40)
  print()
  ransac_plane("data/cluttered_table.txt", "c-cluttered-table", 15)
  
  
  # (d)
  print("running question (d)", "-"*40)
  print()
  ransac_planes("data/clean_hallway.txt", "d-clean-hallway", 0.18)
  
  # (e)
  print("running question (e)", "-"*40)
  print()
  #ransac_planes_clutter("data/cluttered_hallway.txt", "e-cluttered-hallway")















