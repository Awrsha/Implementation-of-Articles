from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA as sklearnPCA

# importing the image / size = 768*1024
filename = "C:\\Users\\nilfam\\Desktop\\Coding\Principal Component Analysis\desert.jpg"
image = Image.open(filename)

# image.size returns a 2-tuple
width, height = image.size

# transforming the "image" list into ndarray object "npimage"
npimage = np.array(image)

# define an array in a desired form (section 7.5 of the book) / [X1 X2 ... Xn] that Xi represents a [x1, x2, x3] for [R, G, B]
arr = []

# copying the im array into arr in the desired form
for y in range(height):
    for x in range(width):
        arr.append(npimage[y, x])

# transforming the "arr" list into ndarray object "nparr"
nparr = np.array(arr)

def show_data(nparr, ax):
    ax.scatter(nparr[:, 0], nparr[:, 1], nparr[:, 2], c='b', marker='.')
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')

fig = plt.figure(figsize=plt.figaspect(1.))

# original data plot
ax = fig.add_subplot(2, 2, 1, projection='3d')
show_data(nparr, ax)
ax.set_title('Original Data')

# scikit-learn PCA plot
ax = fig.add_subplot(2, 2, 2)
sklearn_pca = sklearnPCA(n_components=2)
new_arr_sklearn = sklearn_pca.fit_transform(nparr)
ax.scatter(new_arr_sklearn[:, 0], new_arr_sklearn[:, 1], c='b', marker='.')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('Scikit-learn PCA')

# custom PCA plot
nparr_mean = nparr.mean(axis=0)
nparr_mean_deviation = nparr - nparr_mean
covariance_matrix = np.cov(nparr_mean_deviation.T)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)
matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
new_arr = nparr.dot(matrix_w)
ax = fig.add_subplot(2, 2, 3)
ax.scatter(new_arr[:, 0], new_arr[:, 1], c='b', marker='.')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('Custom PCA')

# new image plot
new_arr_with_zeros = np.hstack([new_arr, np.zeros([height * width, 1])])
new_image = Image.fromarray(np.reshape(new_arr_with_zeros, (height, width, 3)).astype('uint8'))
new_image.save("new_desert.jpg")
ax = fig.add_subplot(2, 2, 4)
ax.imshow(new_image)
ax.set_title('New Image')

plt.tight_layout()
plt.show()