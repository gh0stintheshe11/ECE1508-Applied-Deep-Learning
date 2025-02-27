# Midterm 1

## Question 1: Short Answers (33 Points)

Answer the following items briefly.

1. You are hired by a bank to develop a solution for their problem. The problem is as follows:  
   The bank wants to estimate the amount it can loan safely to its customers. For this, it needs to know their ability for monthly payment. However, this information cannot be collected due to privacy concerns. The only information available is customers’ income and bank records in the last 3 years.  
   Through volunteer customers, the bank makes a database of 10,000 samples. Each sample contains:  
   - The income and bank records of a customer in the last 3 years, and  
   - Their ability for monthly payment.  

   Propose a machine learning model that uses this database and finds what the bank is looking for. Explain how you train this model. Suggest a choice for the loss function. Explain how you confirm if your trained model generalizes.

2. Your friend suggests that you use the following activation instead of ReLU in your FNN:

   \[
   f(x) =
   \begin{cases}
   \max{(x-1, 0.9(x-1))}, & x > 1 \\
   \max{(x-1, 1.1(x-1))}, & x \leq 1
   \end{cases}
   \]

   Do you listen to your friend’s suggestion? Explain your answer by giving a reason.

3. The output layer of a binary classifier is activated with a sigmoid function. Let \( y \) denote the output after activation, i.e., the output of the sigmoid. To compute the loss between \( y \) and the true label \( v \in \{0,1\} \), the following loss function has been suggested:

   \[
   L(y, v) =
   \begin{cases}
   1 - v, & y > 0.75 \\
   0.5, & 0.25 < y \leq 0.75 \\
   v, & y \leq 0.25
   \end{cases}
   \]

   Do you think that this is a good choice of loss function? Explain your answer by giving a reason.

4. You have trained a deep fully-connected feedforward neural network (FNN) for MNIST classification (10-class classification). You have used all 60,000 training images and have observed a very promising test result (close to 95% accuracy). Given your experience, your friend decides to train the same FNN for MNIST classification. However, instead of 60,000 they use only 3,000 samples for training.  
   What observation might your friend have after testing their trained model? How do you explain their observation?

---

## Question 2: Multiclass Classification (25 Points)

You are given a dataset of \(32 \times 32\) gray 8-bit images. Each image belongs to one of the following classes:  

\[
\{ \text{cat, dog, car, airplane, house} \}
\]

The goal is to train a deep fully-connected feedforward neural network (FNN) that classifies a sample \(32 \times 32\) gray image from these classes. The network has a depth of 4, with the second hidden layer having 100 neurons and the other hidden layers having 50 neurons. At the output layer, a softmax activation is used.  

Training is done using the cross-entropy loss:

\[
CE(y, v) = -\sum_{i=1}^{C} v_i \log y_i
\]

where \( y, v \in [0,1]^C \) and log is in the natural base \( e \).  

Answer the following items:

1. Specify the width of each layer and compute the total number of learnable parameters.
2. Suggest an integer labeling for the data points in the dataset.
3. Let \( y \) be the output of the softmax when an image of “car” is passed through the FNN. Explain how you compute the loss between \( y \) and the true label “car” using the integer labeling you have proposed in the previous part.
4. Someone claims that they have trained this network for a large number of epochs. To validate their claim, you decide to pass a few samples forward and observe the output. You can pick samples from (i) training dataset or (ii) test dataset. Which one do you choose? Explain your answer.  
   *Hint: Remember that you are validating this claim that the FNN has been trained.*
5. For the given dataset, you decide to solve the following easier problem: given an input image, you want to find out whether the input image is the image of a “cat” or not. Modify the FNN and your integer labeling to solve this new problem.
6. Assume that you have trained both the original FNN and the modified FNN. You now pick two images from the test dataset: a “car” image and an “airplane” image and mix them by averaging their pixel values. You give this mixed image as an input to both FNNs.  
   What do you expect to see at the output of each of these FNNs? Explain your answer.

---

## Question 3: Forward and Backward Pass (22 Points)

Consider a simple FNN with a single hidden layer used for binary classification. The FNN takes a real-valued scalar \( x \) as input. The hidden layer has 2 neurons activated by a ReLU function. The output layer has a single neuron activated by a sigmoid function. All weights are initially set to 1 and all biases are set to 0. The loss function is:

\[
R̂ = CE(y, v) = - v \log y - (1 - v) \log (1 - y)
\]

where log is in natural base \( e \).  

Answer the following:

1. Sketch a computation graph representing the flow of computations from input data \( x \) to the loss value \( R̂ \).
2. Pass the sample \( x = 0.5 \) with true label \( v = 1 \) forward and specify the value of all nodes in the computation graph.  
   *Hint: You can give your final results in terms of \( \sigma(\cdot) \). No further simplification is needed.*
3. Use the backpropagation algorithm and compute the gradient of the loss \( R̂ \) with respect to the output of the hidden layer before activation.  
   *Hint: You do not need to complete the full backward pass. Only solutions that use backpropagation are marked.*
4. Compute the gradient of the loss \( R̂ \) with respect to the weights and bias of the output layer.  
   *Hint: You can use the result of your computation in Part 3. You do not need to complete the backward pass.*

---

## Question 4: SGD and Optimizers (20 Points)

You are given a dataset of 50,000 samples. You decide to use 80% of this data for training with mini-batch stochastic gradient descent (SGD).

Answer the following:

1. Let the batch size be 100. How many iterations does the SGD do after 20 epochs of training?
2. Are the samples within the first mini-batch of the first and second epochs the same? Explain your answer.
3. What happens if you change the batch size to 200? Explain the benefits and costs of this change.
4. Compare mini-batch SGD with (i) Resilient propagation (Rprop) and (ii) Adam optimizers.
5. Between Rprop and Adam, which one do you choose? Explain your answer.
6. What are the key parameters of your chosen optimizer? What are your suggested values for these parameters?