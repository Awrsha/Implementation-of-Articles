## PCA (Image Processing Project)

One of the conventional techniques for dimensionality reduction is **PCA** or **Principal Component Analysis**. Usually, it is not possible to work with large data with many dimensions, and we need to remove some dimensions that are not very important, if possible, before processing. In this project, as an example, we have tried to examine the data in the matrix of an image and at the end we have reduced one of the dimensions of the image. The initial image is the following photo:

</div>
<div align="center">
<img align="center" height="250" width="375" alt="first" src="https://s2.uupload.ir/files/نمودار7-667x500_1j0l.jpg"/></div>
<br />

Massive data in today's world is not only a big challenge for computing hardware, but also an obstacle for machine learning algorithms. One of the methods used to reduce the volume and complexity of data is Principal Component Analysis or PCA. In PCA analysis, our goal is to find different patterns in the desired data; More precisely, in this analysis, we try to obtain the correlation between the data.

If there is a strong correlation between our data in two or more specific dimensions, those dimensions can be converted into one dimension; In this way and by reducing the dimension, the complexity and volume of data is greatly reduced. If we want to express the ultimate goal of PCA analysis, we should say: the goal is to find the direction (dimension) with the most variance of the data and reduce the dimension; So that the least amount of important data is lost.

In the upcoming image processing project, we want to receive an image from the user and after performing various processes on the matrix of this image, if possible, remove one of its dimensions; More clearly, we want to remove one of the colors in the image in such a way that the least amount of data is lost.

## Languages  
<code>
<img align="center" src="https://github.com/devicons/devicon/blob/v2.15.1/icons/python/python-original-wordmark.svg" width="50" height="50" /> <img align="center" src="https://github.com/devicons/devicon/blob/v2.15.1/icons/vscode/vscode-original-wordmark.svg" width="50" height="50"/><img align="center" src="https://github.com/devicons/devicon/blob/v2.15.1/icons/numpy/numpy-original.svg" width="50" height="50"/><img align="center" src="https://github.com/devicons/devicon/blob/v2.15.1/icons/opencv/opencv-original-wordmark.svg" width="50" height="50" /> <img align="center" src="https://github.com/devicons/devicon/blob/v2.15.1/icons/pycharm/pycharm-original.svg" width="50" height="50"/>
</code>

## Requirements

To run this program, you will need:
- Python 3.x installed on your machine
- NumPy library for Python
- Matplotlib library for Python

## Installation

1. Install Python 3.x from the official website: [https://www.python.org/downloads/ ↗](https://www.python.org/downloads/)
2. Install NumPy library using pip:
   ```
   pip install numpy
   ```
3. Install Matplotlib library using pip:
   ```
   pip install matplotlib
   ```
   
To carry out this project, the following tools have been used, and we will explain each one at the time of use:

  
- Python programming language
- Pillow library in Python
- Numpy library in Python
- Matplotlib library in Python

  
## Import the image into the program using the pillow library

To import the image into the program, we use the pillow library. This library is used for image manipulation, such as image resizing, image rotation, etc. Here, our only use of the PIL library or pillow is to import the image into the program.

```python:
from PIL import Image

#importing the image / size = 768*1024

filename  =  "desert.jpg"

image  =  Image.open(filename)

#image.size returns a 2-tuple

width,  height  =  image.size
```
In the first line, we import the Image object from the pillow library. Then, using the Image.open method and entering the address of the image on the computer, we save the image in the form of an object named image. Please note, because the image is located in the main directory of the program, there is no need to enter its exact address. In the last two lines, we store these two parameters using the image.size method that returns a tuple containing the length and width of the image.

## Save the image as a matrix Image preprocessing


The information in an image can be shown in different ways. One of the common ways to show the data in the image is to use the RGB color system. These three letters are the keywords of the colors Red, Green and Blue. As you know, images are made up of tiny units called pixels. Each pixel can have all or some of these three colors with different intensities. The intensity of each color is usually indicated by a number between 0 and 255. These numbers represent the spectrum that can be made with 8-bit binary memory. For example, a pixel can have the following values:

```sh
           [R, G, B] = [25, 150, 231]
```

An image has different number of pixels according to its size. To store an image numerically, we must store the RGB values ​​of all its pixels; More precisely, we ultimately need a two-dimensional matrix, similar to the following matrix:

```sh
    [[3 2 241],[231 150 25],...,[250 30 12]]
```
The above two-dimensional matrix contains a number of one-dimensional matrices. This number is actually the same number of pixels; For example, in an image with the size of 768 x 1024 pixels, we have 786,432 pixels; In other words, to represent this image numerically, we need a two-dimensional matrix in which there are 786,432 one-dimensional matrices. Each of these one-dimensional matrices has three scalar numbers that represent the RGB of a pixel.
```python:
import numpy as  np

#transforming the "image" list into ndarray object "npimage"

npimage  =  np.array(image)

#define an array in a desired form (section 7.5 of the book) / [X1 X2 ... Xn] that Xi represents a [x1, x2, x3] for [R, G, B]

arr  =  []

#copying the im array into arr in the desired form

for  y  in  range(height  -  1):

for  x  in  range(width  -  1):

arr.append(npimage[y,  x])

#transforming the "arr" list into ndarray object "nparr"

nparr  =  np.array(arr)
```
To work with matrices in Python, we use the numpy library, which was created for this purpose. We have added this library in the first line of the above code with the abbreviation np. The nampay library contains an important object named ndarray. An ndarray object can be created using the np.array method. The input of this method is a list, tuple, or any other array-like object in Python.  
We save the image we received as input in the form of an ndarray named npimage. The resulting matrix does not conform to our desired form; This matrix is ​​actually a three-dimensional matrix; Inside this three-dimensional matrix, there is a two-dimensional array as many vertical pixels as the image; In each of these two-dimensional arrays, there is a one-dimensional array as many horizontal pixels as the image; These one-dimensional arrays contain RGB scalar numbers.  To clarify the issue, imagine we have an image with 3 vertical pixels and 2 horizontal pixels; The resulting matrix will be as follows:
```sh
    [[[25, 120, 251],[34, 12, 78]],[25, 150, 231]][[12, 30, 250],[241, 2, 3]]]
```
As mentioned in the previous section, our desired form is not such a form; Therefore, we need to make the matrix form as desired. Our desired form is as follows:
```sh
    [[12 30 250],[25 150 231],...,[241 2 3]]
```
For this, we have created a new list named arr and fill it using for loops. Finally, we convert the arr list into an ndarray object named nparr using the np.array method. The reason for this is to perform subsequent calculations optimally and at a suitable speed using the numpy library.

## Plot the data using matplotlib

In this step, we want to depict the data in nparr using a suitable diagram. This matrix contains 786432 one-dimensional matrices that hold RGB values. These one-dimensional matrices can be viewed as coordinates in the R3 plane. In this way, by using the function that we have written below called show_data, the data can be drawn on the three-dimensional coordinate page.

```python:
from matplotlib import pyplot as  plt

def show_data(nparr):

fig  =  plt.figure()

ax  =  fig.add_subplot(111,  projection  =  '3d')

ax.scatter(nparr[:,  ],  nparr[:,  1],  nparr[:,  2],  c  =  'b',  marker  =  '.'  )

ax.set_xlabel('Red')

ax.set_ylabel('Green')

ax.set_zlabel('Blue')

plt.show()

show_data(nparr)
```
We use Python's matplotlib library to plot the data. This library is the richest Python library for drawing all kinds of diagrams.

Using the plt.figure method, we have created a frame and coordinate plane, and then using the fig.add_subplot method, we tell the program to draw our diagram in three dimensions and on the whole page. The ax.scatter method is also responsible for receiving the data we want. Here, the c parameter indicates the color of the graph and the marker parameter indicates the sign of each point in the graph. Finally, after naming the axes, we draw the diagram. By calling the function for the nparr matrix, the following graph is obtained:

</div>
<div align="center">
<img align="center" height="250" width="375" alt="graph1" src="https://s2.uupload.ir/files/graph1_wmdh.png"/></div>
<br />

This graph shows that the color of most of the pixels in the image is close to red and some of the pixels are blue. This is completely in line with our expectations. (corresponds to the existing image)

## Find the mean of the data The first step is to perform PCA

At this stage, it is time to implement the PCA method step by step. To perform PCA analysis, we perform the following main steps:

1. Averaging all data
Obtaining the mean deviation of the data
2. Find the covariance matrix
Finding eigenvalues ​​and eigenvectors of the covariance matrix
3. Constructing matrix w to perform linear transformation from R3 space to R2 space
4. Applying the w matrix to the data and obtaining new data in R2 space
5. Outputting the new image obtained

In the first step, using the following function, we obtain the average of the data:

```python:
#it receives a 2D ndarray and shows the mean of all 1D arrays in it

def data_mean(nparr):

result  =  nparr.sum(axis  =  )

columns  =  nparr.shape[]

return  result/columns
```
The function of this function is quite clear. At first, we keep the number of data, which is the same as the number of columns of the nparr matrix, in the columns variable, and after that, we divide the sum of all the columns of the matrix by the number of data to get the average. We print the average data and get the following matrix:
```sh
    [101.27734595 99.45147271 119.58644272]
```
As it was clear from the diagram drawn in the previous part, the intensity of red color is higher than the other two colors.

## Construction of the covariance matrix and its Python code
At this stage, we must first go to the summary center so that our work will be easier in the next steps. To understand this issue, we pay attention to the following two images:

</div>
<div align="center">
<img align="center" height="250" width="375" alt="graph2" src="https://s2.uupload.ir/files/graph2_fgkt.png"/></div>
<br />

</div>
<div align="center">
<img align="center" height="250" width="375" alt="graph3" src="https://s2.uupload.ir/files/graph3_brn2.png"/></div>
<br />

To move the data from the desired location on the coordinate plane to the center of the coordinates (according to what is clear in the image above), it is enough to subtract the average of the data from each data. The resulting matrix after performing this operation is called mean deviation.


```python:
def mean_deviation_func(nparr,  mean):

mean_deviation  =  np.array(nparr  -  mean)

return  mean_deviation

nparr_mean  =  data_mean(nparr)

nparr_mean_deviation  =  mean_deviation_func(nparr,  nparr_mean)
```
Now we can find the covariance matrix. First, we provide explanations about this matrix and then we state the method of obtaining it.

If our data is n-dimensional, the covariance matrix is ​​an n*n matrix with the following properties:

- The elements on the main diameter represent the variance of the data of each dime
- The main non-diagonal elements represent the two-by-two covariance of the data of the dimensions relative to each other.nsion.
- The covariance matrix is ​​a symmetric square matrix; Therefore, its special vectors are perpendicular to each other.
If we call the mean deviation matrix B, the covariance matrix can be obtained by the following formula:
```sh
           S=(1/N-1)*BB^T
```
```python:
def covariance_matrix(nparr_mean_deviation):

transpose  =  nparr_mean_deviation.transpose()

columns  =  nparr_mean_deviation.shape[]

return  (1  /  (columns  -  1))  *  np.dot(transpose,  nparr_mean_deviation)

covariance_matrix  =  covariance_matrix(nparr_mean_deviation)
```
## Finding the value of variance and covariance of the data Application of PCA in image processing

Using the code from the previous part, the covariance matrix was obtained as follows (the decimal part is removed):

```sh
       [[3906 3750 3461],
        [3705 6590 8208],
        [3461 8208 11124]]
```


```python:
print("variance of Red is:")

print(covariance_matrix[,  ])

print("variance of Green is:")

print(covariance_matrix[1,  1])

print("variance of Blue is:")

print(covariance_matrix[2,  2])

print("covariance between Red and Green is:")

print(covariance_matrix[,  1])

print("covariance between Red and Blue is:")

print(covariance_matrix[,  2])

print("covariance between Blue and Green is:")

print(covariance_matrix[1,  2])
```
In the code snippet above, we print the variance and covariance of all data.

## Dimension reduction with PCA method

So far we have made the covariance matrix. Now we need to find a linear transformation that transforms our R3 space into R2 space and reduce the dimension. To make the matrix corresponding to this linear transformation (w matrix), we need to find the eigenvalues ​​and eigenpairs of the covariance matrix. If some eigenvalues ​​are significantly larger than other values, the dimension reduction can be done.In this way, the eigenvectors corresponding to small eigenvalues ​​are discarded and a matrix (the same w matrix) is made with the remaining eigenvectors. This matrix provides the required linear transformation.

The following function provides us with eigenvalues ​​and eigenvectors:

```python:
#returns a tuple of (eigenvalues, eigenvectors)

def eigen(covariance_matrix):

return  np.linalg.eig(covariance_matrix)

eigenvalues,  eigenvectors  =  eigen(covariance_matrix)

print("eigenvalues are:")

print(eigenvalues)

print("eigenvectors are:")

print(eigenvectors)
```
By executing the above code, the eigenvalues ​​and eigenvectors are obtained as follows (the decimal point is omitted):
```sh
       [19030 2549 41]
```
```sh
       [[-0.31 -0.89 +0.33],
        [-0.59 -0.91 -0.80],
        [-0.75 +0.45 +0.49]]
```
As can be seen, there is a huge difference between the size of the eigenvalues. To clarify this difference, using the following code, we obtain a bar graph of the percentage of all eigenvalues ​​out of their total sum:
```python:
def explained_variance(eigenvalues):

total  =  sum(eigenvalues)

exp_variance  =  [(i  /  total)*100  for  i  in  eigenvalues]

objects  =  ('Red',  'Green',  'Blue')

y_pos  =  np.arange(len(objects))

plt.bar(y_pos,  exp_variance,  align='center',  alpha=0.5)

plt.xticks(y_pos,  objects)

plt.ylabel('Percentage')

plt.title('Explained Variance')

plt.show()

#explained_variance(eigenvalues)

explained_variance(eigenvalues)
```
The resulting graph is as follows:

</div>
<div align="center">
<img align="center" height="250" width="375" alt="graph5" src="https://s2.uupload.ir/files/graph5_hhmx.png"/></div>
<br />


It is clear that by removing the blue dimension data, not much data is lost and this dimension reduction can be done.

Now, it is enough to first sort the eigenvalues ​​and eigenvectors from the largest to the smallest and extract the first two eigenvectors as columns of the w matrix and build this matrix. In the following code, we have done this process:

```python:
#sorting the eigen pairs in descending order

eig_pairs  =  [(np.abs(eigenvalues[i]),  eigenvectors[:,i])  for  i  in  range(len(eigenvalues))]

eig_pairs.sort(key=lambda  x:  x[],  reverse=True)

#constructing matrix w

matrix_w  =  np.hstack((eig_pairs[][1].reshape(3,1),  eig_pairs[1][1].reshape(3,1)))

print('Matrix W:\n',  matrix_w)
```
Only one step left! It is enough to apply the matrix w on the dataset to convert our 3D data into a 2D data. We note that w is a 2x3 matrix and it converts a 3D matrix into a 2D matrix. Using the following piece of code, w can be multiplied from the right in the previous data to obtain a new dataset with the name new_arr. Finally, we draw this two-dimensional dataset in the R2 coordinate plane:

```python:
def show_data_2d(nparr):

# x-axis values

x  =  nparr[:,  ]

# y-axis values

y  =  nparr[:,  1]

# plotting points as a scatter plot

plt.scatter(x,  y,  label=  "pixels",  color=  "green",  marker=  ".",  s=30)

# x-axis label

plt.xlabel('principle component 1')

# frequency label

plt.ylabel('principle component 2')

# plot title

plt.title('New Space')

# showing legend

plt.legend()

# function to show the plot

plt.show()

#mapping the dataset to new space and printing it

new_arr  =  nparr.dot(matrix_w)

#show_data_2d(new_arr)

show_data_2d(new_arr)
```
The resulting graph is as follows:

</div>
<div align="center">
<img align="center" height="250" width="375" alt="graph6" src="https://s2.uupload.ir/files/graph6_r95e.png"/></div>
<br />

Now everything is ready to create a new photo and get an output from it. First, instead of the reduced third dimension, we put zero and save the new image using the pillow library and the save method.

Note: we use astype to convert datatype to unit8; Because the PIL library accepts such data.

```python:
#constructing new image array

new_arr_with_zeros  =  np.hstack([new_arr,  np.zeros([height *  width,  1])])

new_image  =  Image.fromarray(np.reshape(new_arr_with_zeros,  (height,  width,  3)).astype('uint8'))

#saving the new image

new_image.save("new_deset.jpg")
```
At the end, we compare the new image with the original image:

</div>
<div align="center">
<img align="center" height="250" width="375" alt="graph7" src="https://s2.uupload.ir/files/نمودار7-667x500_(1)_0qg1.jpg"/></div>
<br />
</div>
<div align="center">
<img align="center" height="250" width="375" alt="graph8" src="https://s2.uupload.ir/files/نمودار8-667x500_mhhy.jpg"/></div>
<br />

## Conclusion
In this project, we learned about one of the dimension reduction techniques called PCA, and it is possible to reduce the dimension of different datasets by using it. We coded in detail in this project for educational purposes; Instead of all these steps, PCA analysis can be performed using the scikit learn library in Python using the following code:

```python:
from sklearn.decomposition import PCA as  sklearnPCA

sklearn_pca  =  sklearnPCA(n_components=2)

new_arr  =  sklearn_pca.fit_transform(nparr)
```
In the use of many machine learning algorithms that deal with many features, it is better to first remove related features using dimension reduction techniques such as PCA so that the performance of the machine learning algorithm is optimized.

## References:
 -   Linear Algebra and its Applications by David C. Lay – section 7.5
 -  [https://en.wikipedia.org/wiki/RGB_color_model](https://en.wikipedia.org/wiki/RGB_color_model)
 -  [https://sebastianraschka.com/](https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html#:~:text=Obtain%20the%20Eigenvectors%20and%20Eigenvalues,subspace%20(k%E2%89%A4d).)
 
 ##  System & Hardware 🛠
<br>
  <summary><b>⚙️ Things I use to get stuff done</b></summary>
  	<ul>
  	    <li><b>OS:</b> Ubuntu 22.04 LTI</li>
	    <li><b>Laptop: </b>TUF Gaming</li>
	    <li><b>Code Editor:</b> VSCode - The best editor out there.</li>
	    <li><b>To Stay Updated:</b> Medium, Linkedin and Twitter.</li>
	    <br />
	⚛️ Checkout Our VSCode Configrations <a href="">Here</a>.
	</ul>	
<p align="center">💙 If you like my projects, Give them ⭐ and Share it with friends!</p>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/mayhemantt/mayhemantt/Update/svg/Bottom.svg" alt="Github Stats" />
</p>
