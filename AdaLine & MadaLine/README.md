## 1. AdaLine

### a) Define and Plot Two Sets of Data:

**First set:**

- 100 data points

-  `x`: Mean = 0, Std Dev = 0.1

-  `y`: Mean = 0, Std Dev = 0.4

**Second set:**

- 100 data points

-  `x`: Mean = 1, Std Dev = 0.2

-  `y`: Mean = 1, Std Dev = 0.2

### b) Train an AdaLine Network:

**Steps:**

1. Initialize weights and bias.

2. Train the AdaLine network using the learning rule.

3. Plot the changes in error over time.

**Justification:**

The data separation can be evaluated based on the final plot of the cost function. If the cost decreases steadily and stabilizes at a low value, the data is separated well. If not, the separation might be poor.

### c) Change Data Set and Repeat:

**New Data Sets:**

**First set:**

- 100 data points

-  `x`: Mean = 0, Std Dev = 0.4

-  `y`: Mean = 0, Std Dev = 0.4

**Second set:**

- 100 data points

-  `x`: Mean = 1, Std Dev = 0.3

-  `y`: Mean = 1, Std Dev = 0.3

### d) Compare Results:

Compare the plots of error change over epochs and discuss the differences in convergence and error rates for the two sets of data.

## 2. MadaLine

### a) Explanation of an Algorithm:

**MRII (Modified Resilient Backpropagation with Inertial Impact):**

MRII is an enhanced version of the standard resilient backpropagation (Rprop) algorithm. It incorporates an inertial impact term to accelerate convergence by considering the direction of previous weight updates. This helps in avoiding oscillations and improves the learning speed, especially in deeper networks.

### b) Load and Plot Data:
  

**Code to load and plot data:**

### c) Train a Network:

Train a network based on the MRII algorithm with different numbers of neurons.

### d) Compare and Analyze:

Compare the three resulting plots, accuracies, and the number of epochs for each case, discussing how the number of neurons impacts the network's performance and convergence.