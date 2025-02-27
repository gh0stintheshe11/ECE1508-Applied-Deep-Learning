# Midterm 2

## Question 1: Short Answers (33 Points)

Answer the following items briefly.

1. You are hired by a health organization to develop a solution for their problem. The problem is as follows:  
   The organization wants to compute the chance of developing heart disease for the people registered in their basic health plan. For this, it needs to know their key blood test results, such as blood pressure and sugar level. However, such information has not been collected in the basic health plan. The only information available is basic factors, such as weight, height, age, gender, etc.  
   A group of 10,000 volunteer members have announced that they could share their blood test results with you so that you could provide a data-driven solution for this problem.  

   - Explain how you can make a dataset from these volunteer members.  
   - Propose a machine learning model that uses this dataset and finds what the organization looks for.  
   - Suggest a choice for the loss function.  
   - Explain how you train this model and how you confirm if the trained model generalizes.

2. Your friend suggests that you replace the ReLU activation in an FNN with the following activation:

   \[
   f(x) =
   \begin{cases}
   \max(\text{ReLU}(x - 1), (x - 1)), & x \leq 1 \\
   \max(x - 1, 2), & 1 < x \leq 2 \\
   \min(\text{ReLU}(x), 0), & x > 2
   \end{cases}
   \]

   Do you listen to your friend’s suggestion? Explain your answer by giving a reason.

3. The output layer of a 4-class classifier is activated by the softmax function. Let \( y \) denote the output of this classifier, i.e., the output of the softmax function. To compute the loss between \( y \) and the true label \( v_{label} \in \{1,2,3,4\} \), we intend to use the cross-entropy loss function:

   \[
   CE(y,v) = -\sum_{i=1}^{C} v_i \log y_i
   \]

   where \( v \) is the one-hot representation of \( v_{label} \) and log is in natural base \( e \).  

   In the implementation, we mistakenly exchange the arguments of the loss function, computing \( CE(v,y) \) instead of \( CE(y,v) \). Assume that in our code, we have defined \( \log 0 = -100 \) (though it does not exist).  
   - Do you expect that our mistaken implementation still trains the model? Explain your answer.  
   *Hint: Open up the mistaken loss and discuss what happens when you minimize it.*

4. Two neural networks have the same (very large) number of neurons. However, one of these networks is shallow, and the other one is very deep.  
   - Which of these networks would you choose for a given learning task? Explain your answer.

---

## Question 2: Multiclass Classification (25 Points)

You are given a dataset of \(64 \times 64\) gray 8-bit images. Each image belongs to one of the following classes:

\[
\{ \text{book, pen, mug, phone} \}
\]

The goal is to train a deep fully-connected feedforward neural network (FNN) that classifies a sample \(64 \times 64\) gray image from these classes. The network has a depth of 5, where the second and third hidden layers have 100 neurons, and the other hidden layers have 50 neurons. At the output layer, a softmax activation is used.  

Training is done using the cross-entropy loss:

\[
CE(y, v) = -\sum_{i=1}^{C} v_i \log y_i
\]

where \( y, v \in [0,1]^C \) and log is in the natural base \( e \).  

Answer the following items:

1. For activation of the hidden neurons, we have three options:  
   (i) ReLU function, (ii) tanh function, and (iii) step function.  
   - Which one do you choose?  
   - Explain why you do not choose the other options. You may suggest more than one.

2. Compute the total number of learnable parameters.

3. Someone claims that they have trained this network for a large number of epochs. To validate their claim, you decide to pass a few samples forward and observe the output. You can pick samples from:  
   (i) training dataset, or (ii) test dataset.  
   - Which one do you choose? Explain your answer.  
   *Hint: Remember that you are validating this claim that the FNN has been trained.*

4. For the given dataset, you decide to solve an easier problem: given an input image, determine whether the image is a “book” or not.  
   - Modify the FNN and your integer labeling to solve this new problem.

5. Assume that you have trained both the original FNN (for the original multiclass classification) and the modified FNN in Part 4. You now give an image of a transparent mug with a pen in it as an input to both of these FNNs.  
   - What do you expect to see at the output of each of these FNNs? Explain your answer.

---

## Question 3: Forward and Backward Pass (22 Points)

Consider a simple FNN with a single hidden layer used for binary classification. The FNN takes a real-valued scalar \( x \) as input. The hidden layer has 2 neurons activated by a ReLU function. The output layer has a single neuron activated by a linear function, \( f(z) = z \). All weights are initially set to 1 and all biases are initially set to 0. The true label \( v \) is a real number, i.e., \( v \in \mathbb{R} \). The loss function is the squared error:

\[
R̂ = L(y, v) = (y - v)^2
\]

Answer the following:

1. Sketch a computation graph representing the flow of computations from input \( x \) to the loss value \( R̂ \).
2. Pass the sample \( x = 1 \) with true label \( v = 0.5 \) forward and specify the value of all nodes in the computation graph.
3. Use the backpropagation algorithm and compute the gradient of the loss \( R̂ \) with respect to the output of the hidden layer before activation.  
   *Hint: You do not need to complete the full backward pass. Only solutions using backpropagation are marked.*
4. Compute the gradient of the loss \( R̂ \) with respect to the weights and bias of the output layer.  
   *Hint: You can use the result of Part 3.*

---

## Question 4: SGD and Optimizers (20 Points)

You are given a dataset of 80,000 samples. You decide to use 80% of this data for training with mini-batch stochastic gradient descent (SGD).

Answer the following:

1. Let the batch size be 64. Assume that each iteration of SGD takes \( 10^{-3} \) seconds. How long does it take to train the network for 20 epochs?
2. Are the samples within the first mini-batch of the first and second epochs the same? Explain your answer.
3. If we increase the batch size to 128, how does the training time change? What are the benefits of this change?
4. Compare mini-batch SGD with (i) RMSprop and (ii) Adam optimizers.
5. Between RMSprop and Adam, which one do you choose? Explain your answer.
6. What are the key parameters of your chosen optimizer? What values do you suggest for these parameters?