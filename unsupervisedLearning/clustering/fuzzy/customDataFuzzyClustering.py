from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

colors = ['b', 'orange', 'g', 'burlywood', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

xpts = np.array([1, 1.5, 8.2, 6.7, 7, 2, 8, 1, 1.8, 9, 3.6, 4.2, 4.7, 2.8, 3.1])
ypts = np.array([2, 1.8, 8.3, 9, 10, 3, 8, 0.6, 0.9, 11, 5, 4.7, 5.3, 6.3, 5.1])
labels = np.array([0, 0, 2, 2, 2, 0, 2, 0, 0, 2, 1, 1, 1, 1, 1])

# Visualize the data
fig0, ax0 = plt.subplots()
for label in range(3):
    ax0.plot(xpts[labels == label], ypts[labels == label], '.',
             color=colors[label])
ax0.set_title('Test data: x3 clusters.')

plt.show()

# Set up the loop and plot
fig1, axes1 = plt.subplots(2, 2, figsize=(8, 8))
alldata = np.vstack((xpts, ypts))
fpcs = []

for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

    # Store fpc values for later
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(xpts[cluster_membership == j],
                ypts[cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1], '-rs', linewidth=4)

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))

fig1.tight_layout()

plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(np.r_[2:6], fpcs)
ax2.set_xlabel("Number of centers")
ax2.set_ylabel("Fuzzy partition coefficient")

plt.show()

# Regenerate fuzzy model with 3 cluster centers - note that center ordering
# is random in this clustering algorithm, so the centers may change places
cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
    alldata, 3, 2, error=0.005, maxiter=1000)

# Show 3-cluster model
fig2, ax2 = plt.subplots()
ax2.set_title('Trained model for 3 clusters')
for j in range(3):
    ax2.plot(alldata[0, u_orig.argmax(axis=0) == j],
             alldata[1, u_orig.argmax(axis=0) == j], 'o',
             label='series ' + str(j))
for pt in cntr:
    ax2.plot(pt[0], pt[1], '-rs', linewidth=4)
ax2.legend()

plt.show()

# Generate uniformly sampled data spread across the range [0, 10] in x and y
newdata = np.random.uniform(0, 1, (1100, 2)) * 10

# Predict new cluster membership with `cmeans_predict` as well as
# `cntr` from the 3-cluster model
u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    newdata.T, cntr, 2, error=0.005, maxiter=1000)

# Plot the classified uniform data. Note for visualization the maximum
# membership value has been taken at each point (i.e. these are hardened,
# not fuzzy results visualized) but the full fuzzy result is the output
# from cmeans_predict.
cluster_membership = np.argmax(u, axis=0)  # Hardening for visualization

fig3, ax3 = plt.subplots()
ax3.set_title('Random points classifed according to trained centers')
for j in range(3):
    ax3.plot(newdata[cluster_membership == j, 0],
             newdata[cluster_membership == j, 1], 'o',
             label='series ' + str(j))
for pt in cntr:
    ax3.plot(pt[0], pt[1], '-rs', linewidth=4)
ax3.legend()

plt.show()
