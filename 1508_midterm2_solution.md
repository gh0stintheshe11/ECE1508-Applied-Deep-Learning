# Q1.1

## **Step 1: Creating the Dataset**
We need to estimate the **risk of heart disease** using **basic health factors** (weight, height, age, gender, etc.) and **key blood test results** (blood pressure, sugar level) from **volunteer members**.

### **Dataset Structure**
Each **data sample** (patient record) should contain:
- **Features (Input Variables \( x \))**:
  - **Basic Health Factors**: \( x_1 = \) Weight, \( x_2 = \) Height, \( x_3 = \) Age, \( x_4 = \) Gender, etc.
  - **Key Blood Test Results**: \( x_5 = \) Blood Pressure, \( x_6 = \) Sugar Level

- **Label (Output \( y \))**:
  - **Binary Classification:** \( y = 1 \) (High risk of heart disease), \( y = 0 \) (Low risk of heart disease).

Thus, the dataset consists of **\( 10,000 \) samples** with feature-label pairs:

\[
(x_1, x_2, \dots, x_6) \rightarrow y
\]

---

## **Step 2: Choosing a Machine Learning Model**
Since the task is **binary classification** (predicting heart disease risk), we can use a **Feedforward Neural Network (FNN) with Sigmoid Activation**.

### **Model Architecture**
- **Input Layer:** 6 neurons (one per feature).
- **Hidden Layers:** Fully connected layers to capture complex relationships.
- **Output Layer:** 1 neuron with **Sigmoid Activation**:
  \[
  y = \sigma(Wx + b) = \frac{1}{1 + e^{-(Wx + b)}}
  \]

where:
- \( W \) = weight matrix,
- \( b \) = bias vector.

✅ **Final Model:** **Feedforward Neural Network (FNN) with Sigmoid Activation.**

---

## **Step 3: Choosing the Loss Function**
For binary classification, we use **Binary Cross-Entropy Loss**:

\[
\hat{R} = - \left( v \log y + (1 - v) \log (1 - y) \right)
\]

where:
- \( v \) is the **true label**,
- \( y \) is the **model’s predicted probability**.

✅ **Final Choice:** **Binary Cross-Entropy Loss.**

---

## **Step 4: Training the Model**
We train using **Stochastic Gradient Descent (SGD)** with mini-batches.

1. **Forward Pass:** Compute predictions using:
   \[
   y = \sigma(Wx + b)
   \]
2. **Compute Loss:** Use **Binary Cross-Entropy Loss**.
3. **Backward Pass:** Compute gradients using **Backpropagation**:
   \[
   \frac{\partial \hat{R}}{\partial W} = - (v - y) x
   \]
   \[
   \frac{\partial \hat{R}}{\partial b} = - (v - y)
   \]
4. **Update Weights Using SGD**:
   \[
   W = W - \eta \frac{\partial \hat{R}}{\partial W}
   \]
   \[
   b = b - \eta \frac{\partial \hat{R}}{\partial b}
   \]
   where \( \eta \) is the **learning rate**.

✅ **Final Answer:** **Train the model using mini-batch SGD and backpropagation.**

---

## **Step 5: Confirming Generalization**
To ensure the model **generalizes well**, we:
1. **Split Data into Train & Test Sets**:
   - **Train** on \( 80\% \) of data.
   - **Test** on \( 20\% \) of data.
   
2. **Measure Performance on Test Data**:
   - Compute **Accuracy, Precision, Recall**.
   - Evaluate using **ROC Curve & AUC Score**.

3. **Check for Overfitting**:
   - If **Training Accuracy >> Test Accuracy**, the model is overfitting.
   - Use **Regularization (Dropout, L2 Norm)** if needed.

✅ **Final Answer:** **Generalization is checked using a test set and evaluating accuracy & ROC-AUC.**

---

## **Final Summary**
| **Step** | **Final Answer** |
|----------|------------------|
| **Dataset Creation** | **10,000 samples with (basic health factors + blood test results) → binary label (disease risk: Yes/No).** |
| **Model Choice** | **Feedforward Neural Network (FNN) with Sigmoid Activation.** |
| **Loss Function** | **Binary Cross-Entropy Loss.** |
| **Training Process** | **Mini-Batch SGD + Backpropagation.** |
| **Generalization Check** | **Split data, evaluate test accuracy, and check ROC-AUC.** |

---

# Q1.2

## **Step 1: Understanding the Proposed Activation Function**
Your friend suggests replacing the **ReLU** activation with a **piecewise-defined function**:

\[
f(x) =
\begin{cases}
\max\{\text{ReLU}(x-1), (x-1)\} & x \leq 1 \\
\max\{(x-1), 2\} & 1 < x \leq 2 \\
\min\{\text{ReLU}(x), 0\} & x > 2
\end{cases}
\]

### **Breaking Down Each Case:**
1. **For \( x \leq 1 \):**  
   - **ReLU(x-1) = max(0, x-1)**  
   - **So, \( f(x) = \max(0, x-1, x-1) \Rightarrow f(x) = \max(0, x-1) \)**
   - **This is just ReLU(x-1), which behaves like a shifted ReLU.**

2. **For \( 1 < x \leq 2 \):**  
   - **\( f(x) = \max(x-1, 2) \)**
   - **If \( x-1 \leq 2 \), then \( f(x) = 2 \), meaning the function saturates at 2.**
   - **This is a problem because saturation leads to vanishing gradients, slowing down learning.**

3. **For \( x > 2 \):**  
   - **ReLU(x) = max(0, x)**
   - **\( \min(\text{ReLU}(x), 0) \) forces the output to be at most 0.**
   - **This prevents positive values, which is an issue for training.**

✅ **Key Issues:**
- **Gradient Vanishing for \( x > 2 \)**: If \( f(x) \leq 0 \), the neuron will not contribute to learning.
- **Saturation at \( x > 1 \) (ReLU(x-1))**: Limits learning capacity.
- **Unnecessary Complexity**: A good activation function should be simple and differentiable.

---

## **Step 2: Comparison with ReLU**
The standard **ReLU function**:
\[
\text{ReLU}(x) = \max(0, x)
\]
- **Advantages:**
  - **Non-linearity**: Helps learn complex functions.
  - **Sparse Activation**: Only positive inputs activate neurons.
  - **Does not saturate**: Unlike Sigmoid, ReLU avoids vanishing gradients.

- **Why is the proposed function worse?**
  - **Adds unnecessary constraints**.
  - **Saturation at \( x > 1 \)** and **clipping at \( x > 2 \)** limit the model’s capacity.

✅ **Final Answer: No, do not use this function. Stick with ReLU for efficient training.**

---

## **Final Summary**
| **Aspect** | **Proposed Function** | **ReLU** |
|------------|----------------------|----------|
| **Gradient Flow** | Vanishes for \( x > 2 \) | No vanishing gradients |
| **Saturation** | \( f(x) = 2 \) for \( x > 1 \) | No saturation |
| **Complexity** | Over-complicated piecewise function | Simple definition \( \max(0, x) \) |
| **Training Efficiency** | Poor due to non-continuous gradient | Efficient learning |

### **Conclusion**
**Do not use the proposed function** because:
- It introduces **vanishing gradients**.
- It **limits neuron activation** beyond \( x > 2 \).
- **ReLU is already optimal** for FNNs due to its simplicity and effectiveness.

---

# Q1.3

## **Step 1: Understanding Correct Cross-Entropy Loss**
For a **4-class classification problem**, the correct **Cross-Entropy (CE) Loss** is:

\[
CE(\mathbf{y}, \mathbf{v}) = - \sum_{i=1}^{C} v_i \log y_i
\]

where:
- \( \mathbf{y} \) = softmax output vector \( (y_1, y_2, y_3, y_4) \).
- \( \mathbf{v} \) = **one-hot encoded true label** (e.g., for class 2: \( (0,1,0,0) \)).
- \( C = 4 \) (number of classes).
- The summation ensures that **only one term is non-zero** (corresponding to the true class).

✅ **Correctly implemented loss:**  
\[
CE(\mathbf{y}, \mathbf{v}) = - \log y_{\text{true class}}
\]

---

## **Step 2: Mistaken Implementation**
The **incorrect loss** swaps arguments:

\[
CE(\mathbf{v}, \mathbf{y}) = - \sum_{i=1}^{C} y_i \log v_i
\]

Since **\( \mathbf{v} \) is one-hot encoded**, we analyze:
- **For the correct class**: \( v_{\text{true class}} = 1 \Rightarrow \log v_{\text{true class}} = 0 \).
- **For incorrect classes**: \( v_i = 0 \Rightarrow \log v_i = \log 0 \).

**Problem:**
- **Log(0) is undefined**, but in the question, it is set to \( -100 \).
- This means \( CE(\mathbf{v}, \mathbf{y}) \) becomes:

\[
CE(\mathbf{v}, \mathbf{y}) = - \sum_{i=1}^{C} y_i \cdot (-100)
\]

\[
= 100 \sum_{i=1}^{C} y_i
\]

Since \( \sum_{i=1}^{C} y_i = 1 \) (softmax property), we get:

\[
CE(\mathbf{v}, \mathbf{y}) = 100
\]

✅ **Key Issue:**
- The mistaken loss **is constant** (always **100**, regardless of predictions).
- **No gradient updates occur** → **Model does not learn.**

---

## **Step 3: Effect of Minimization**
Minimizing the **correct loss**:
- Encourages **higher probability for the true class**.
- Improves **classification accuracy**.

Minimizing the **mistaken loss**:
- Since it is **always 100**, its gradient is **zero**.
- **No updates happen** → **Model does not improve.**

---

## **Final Answer**
❌ **No, the mistaken implementation does not train the model.**  
- **Loss remains constant at 100**.
- **No gradient updates** → **Weights do not change**.
- **The model never learns.**

✅ **Final Recommendation**: **Fix the implementation to use \( CE(\mathbf{y}, \mathbf{v}) \).**  

---

# Q1.4

## **Step 1: Understanding the Difference**
We are given **two neural networks** with the **same number of neurons**, but:
- **One is shallow** (fewer layers, more neurons per layer).
- **One is deep** (more layers, fewer neurons per layer).

✅ **Key Question:** Which one should we choose for a learning task?

---

## **Step 2: Theoretical Comparison**
| **Aspect** | **Shallow Network** | **Deep Network** |
|------------|------------------|---------------|
| **Representation Power** | **Limited** to simple patterns | **Can learn hierarchical features** |
| **Feature Learning** | **Harder** to capture complex structures | **Better** for deep feature extraction |
| **Computational Cost** | **Less costly** (fewer layers) | **More costly** (many layers) |
| **Vanishing Gradient Problem** | **Less affected** | **More affected** (but solved by ReLU, Batch Norm) |
| **Overfitting Risk** | **High** (Too many neurons per layer) | **Lower** with regularization |

---

## **Step 3: Choosing the Best Network**
1. **For Simple Tasks (e.g., Linear Classification, Basic Patterns)**
   - A **shallow network** may work fine.
   - **No need for deep feature extraction**.

2. **For Complex Tasks (e.g., Image Recognition, NLP, Speech Processing)**
   - A **deep network** is better.
   - It learns **hierarchical features** (edges → textures → objects).

3. **If Computational Resources Are Limited**
   - A **shallow network** is faster.
   - Deep networks require **more training time and data**.

---

## **Final Answer**
✅ **A deep network is generally preferred for complex learning tasks.**  
- **Shallow networks lack hierarchical feature learning**.
- **Deep networks capture abstract representations, improving accuracy**.
- **However, for simple tasks, a shallow network may suffice.**

---

# Q2

## **1. Choosing an Activation Function for Hidden Neurons**
We have three activation function options for the hidden layers:
- **(i) ReLU**
- **(ii) tanh**
- **(iii) Step Function**

### **Analysis of Each Option**
| Activation | Pros | Cons |
|------------|------|------|
| **ReLU** \( \max(0, x) \) | Efficient, avoids vanishing gradient, non-linearity enables deep learning | Can have "dead neurons" (if \( x < 0 \)) |
| **tanh** \( \frac{e^x - e^{-x}}{e^x + e^{-x}} \) | Zero-centered, useful for small networks | Can suffer from vanishing gradients |
| **Step Function** \( f(x) = 1 \) if \( x > 0 \) else 0 | Clear decision boundary | **Not differentiable**, prevents backpropagation |

### **Best Choice**
✅ **ReLU is the best choice** because:
- It **avoids the vanishing gradient problem** (which tanh suffers from).
- It is **computationally efficient** (faster than tanh).
- It enables **deep learning** by allowing better feature learning.

### **Rejected Options**
❌ **Step Function is a bad choice** because:
- It is **not differentiable** at \( x = 0 \), making backpropagation impossible.

❌ **tanh is not ideal** because:
- While it is zero-centered, it suffers from **vanishing gradients**, slowing deep learning.

---

## **2. Computing the Total Number of Learnable Parameters**
### **Network Structure**
- Input: **\( 64 \times 64 = 4096 \) neurons**
- Hidden Layers:
  - **First hidden layer**: 100 neurons
  - **Second hidden layer**: 100 neurons
  - **Remaining 3 hidden layers**: 50 neurons each
- Output layer: **4 neurons (one per class)**

### **Parameter Calculation**
#### **Layer 1 (Input → First Hidden Layer)**
- **Weights:** \( 4096 \times 100 = 409600 \)
- **Biases:** \( 100 \)
- **Total:** **409700**

#### **Layer 2 (First Hidden → Second Hidden)**
- **Weights:** \( 100 \times 100 = 10000 \)
- **Biases:** \( 100 \)
- **Total:** **10100**

#### **Layer 3 (Second Hidden → Third Hidden)**
- **Weights:** \( 100 \times 50 = 5000 \)
- **Biases:** \( 50 \)
- **Total:** **5050**

#### **Layer 4 (Third Hidden → Fourth Hidden)**
- **Weights:** \( 50 \times 50 = 2500 \)
- **Biases:** \( 50 \)
- **Total:** **2550**

#### **Layer 5 (Fourth Hidden → Output Layer)**
- **Weights:** \( 50 \times 4 = 200 \)
- **Biases:** \( 4 \)
- **Total:** **204**

### **Final Total Parameters**
\[
409700 + 10100 + 5050 + 2550 + 204 = 422604
\]

✅ **Final Answer: 422,604 learnable parameters.**

---

## **3. Validating a Training Claim**
**Question:** Should we validate training using the training set or test set?

✅ **Answer:** **Use the training set.**
- We are validating if the network **has been trained**.
- If trained correctly, **training accuracy should be high**.
- **Test set is used for generalization, not validation of training**.

---

## **4. Modifying FNN for a Binary Classification Task**
**Question:** Modify the FNN to classify only "book" vs. "not book".

✅ **Changes to the FNN:**
1. **Output layer:**  
   - Change from **4 neurons → 1 neuron**.
   - Use **sigmoid activation** \( \sigma(x) = \frac{1}{1 + e^{-x}} \).
   
2. **Loss Function:**  
   - Use **binary cross-entropy** instead of multiclass cross-entropy:

   \[
   CE(y, v) = - v \log y - (1 - v) \log (1 - y)
   \]

3. **Label Encoding:**  
   - Instead of **one-hot encoding**, assign:
     - **1 → "book"**
     - **0 → "not book"** (pen, mug, phone)

✅ **Effect:**  
- Model outputs **a probability** of an image being a book.
- **Threshold at 0.5**: \( y \geq 0.5 \) → **book**, otherwise **not book**.

---

## **5. Prediction for an Ambiguous Image (Mug with a Pen)**
**Question:** What happens if the model sees an ambiguous object (e.g., a "mug with a pen")?

✅ **Original Multiclass FNN (4-class classification)**
- The model will output **softmax probabilities** over the four classes.
- If the image **contains features of both mug and pen**, softmax will distribute probability between **mug and pen**.

✅ **Binary Classifier (Book vs. Not-Book)**
- Since the object is **not a book**, it should output a probability **close to 0**.
- However, **if the network is uncertain**, the output may be **around 0.5**.

---

## **Final Answer Summary**
1. **Best Activation:** ReLU (avoids vanishing gradient, efficient).
2. **Total Learnable Parameters:** **422,604**.
3. **Validate Training:** **Use the training set** (to check if the model has actually learned).
4. **Modifying FNN for "Book vs. Not-Book":**
   - Change **output layer to 1 neuron** with **sigmoid activation**.
   - Use **binary cross-entropy** instead of multiclass cross-entropy.
5. **Prediction for a "Mug with a Pen":**
   - **Multiclass FNN:** Distributes probability over "mug" and "pen".
   - **Binary FNN:** Should classify as "not book" (close to 0), but might be uncertain (around 0.5).

---

# Q3

### **Solution to Question 3: Forward and Backward Pass**

#### **Problem Recap:**
We have a simple **fully-connected feedforward neural network (FNN)** with:
- A **single hidden layer** with **2 neurons**.
- **ReLU activation** in the hidden layer.
- A **linear activation** at the output layer: \( f(z) = z \).
- **All weights initialized to 1**, and **all biases initialized to 0**.
- **Squared error loss**:  
  \[
  \hat{R} = \mathcal{L}(y, v) = (y - v)^2
  \]
- Given **input**: \( x = 1 \) and **true label**: \( v = 0.5 \).

---

### **1. Sketch a Computation Graph**
The computation graph consists of:
1. **Forward Pass:**
   - Compute **hidden layer activations**.
   - Compute **output neuron activation**.
   - Compute **loss function**.

2. **Backward Pass:**
   - Compute **gradient of loss** w.r.t. the output.
   - Compute **gradient of the loss w.r.t. hidden neurons before activation**.
   - Compute **gradient w.r.t. weights and biases**.

#### **Graph Representation**
A computation graph illustrating the dependencies in calculations would follow:
\[
x \rightarrow \text{Hidden Layer (ReLU)} \rightarrow \text{Output Layer (Linear)} \rightarrow \text{Loss Function}
\]

---

### **2. Forward Pass with \( x = 1 \)**
#### **Step 1: Compute Hidden Layer Outputs**
Each neuron \( h_i \) in the hidden layer follows:
\[
z_i = w_i x + b_i
\]
Since all weights \( w_i = 1 \) and biases \( b_i = 0 \):

\[
z_1 = 1(1) + 0 = 1
\]
\[
z_2 = 1(1) + 0 = 1
\]

Since we apply **ReLU activation**:
\[
h_i = \max(0, z_i)
\]

Thus:
\[
h_1 = \max(0,1) = 1, \quad h_2 = \max(0,1) = 1
\]

#### **Step 2: Compute Output Neuron Activation**
The output neuron follows a **linear activation**:
\[
y = w_o h_1 + w_o h_2 + b_o
\]

Since all **weights and biases are 1 and 0** respectively:
\[
y = 1(1) + 1(1) + 0 = 2
\]

---

### **3. Compute the Loss**
Using **squared error loss**:
\[
\hat{R} = (y - v)^2 = (2 - 0.5)^2 = (1.5)^2 = 2.25
\]

---

### **4. Backpropagation: Compute Gradients**
We now compute the **gradient of the loss w.r.t. all parameters**.

#### **Step 1: Compute Gradient of Loss w.r.t Output \( y \)**
\[
\frac{\partial \hat{R}}{\partial y} = 2 (y - v) = 2(2 - 0.5) = 2(1.5) = 3
\]

#### **Step 2: Compute Gradient w.r.t Hidden Layer Activations**
Since the output layer applies a **linear function**:
\[
\frac{\partial y}{\partial h_1} = w_o = 1, \quad \frac{\partial y}{\partial h_2} = w_o = 1
\]

Using the chain rule:
\[
\frac{\partial \hat{R}}{\partial h_1} = \frac{\partial \hat{R}}{\partial y} \cdot \frac{\partial y}{\partial h_1} = 3 \cdot 1 = 3
\]

\[
\frac{\partial \hat{R}}{\partial h_2} = \frac{\partial \hat{R}}{\partial y} \cdot \frac{\partial y}{\partial h_2} = 3 \cdot 1 = 3
\]

#### **Step 3: Compute Gradient w.r.t Hidden Layer Before Activation**
ReLU function derivative:
\[
\frac{d}{dz} \max(0, z) =
\begin{cases}
1 & z > 0 \\
0 & z \leq 0
\end{cases}
\]

Since \( z_1 = z_2 = 1 > 0 \), **ReLU is active**, so its derivative is 1:
\[
\frac{\partial h_1}{\partial z_1} = 1, \quad \frac{\partial h_2}{\partial z_2} = 1
\]

Applying the chain rule:
\[
\frac{\partial \hat{R}}{\partial z_1} = \frac{\partial \hat{R}}{\partial h_1} \cdot \frac{\partial h_1}{\partial z_1} = 3 \cdot 1 = 3
\]

\[
\frac{\partial \hat{R}}{\partial z_2} = \frac{\partial \hat{R}}{\partial h_2} \cdot \frac{\partial h_2}{\partial z_2} = 3 \cdot 1 = 3
\]

---

### **5. Compute Gradients w.r.t. Weights and Biases**
#### **Step 1: Compute Gradient w.r.t. Hidden-to-Output Weights**
Since:
\[
y = w_o h_1 + w_o h_2 + b_o
\]

The derivative is:
\[
\frac{\partial y}{\partial w_o} = h_i
\]

Applying the chain rule:
\[
\frac{\partial \hat{R}}{\partial w_o} = \frac{\partial \hat{R}}{\partial y} \cdot \frac{\partial y}{\partial w_o}
\]

\[
\frac{\partial \hat{R}}{\partial w_o} = 3 \cdot 1 = 3
\]

#### **Step 2: Compute Gradient w.r.t. Bias of Output Layer**
\[
\frac{\partial y}{\partial b_o} = 1
\]

\[
\frac{\partial \hat{R}}{\partial b_o} = \frac{\partial \hat{R}}{\partial y} \cdot \frac{\partial y}{\partial b_o} = 3 \cdot 1 = 3
\]

#### **Step 3: Compute Gradient w.r.t. Input-to-Hidden Weights**
Since:
\[
z_i = w_i x + b_i
\]

\[
\frac{\partial z_i}{\partial w_i} = x
\]

Using the chain rule:
\[
\frac{\partial \hat{R}}{\partial w_i} = \frac{\partial \hat{R}}{\partial z_i} \cdot \frac{\partial z_i}{\partial w_i} = 3 \cdot 1 = 3
\]

#### **Step 4: Compute Gradient w.r.t. Bias of Hidden Layer**
\[
\frac{\partial z_i}{\partial b_i} = 1
\]

\[
\frac{\partial \hat{R}}{\partial b_i} = \frac{\partial \hat{R}}{\partial z_i} \cdot \frac{\partial z_i}{\partial b_i} = 3 \cdot 1 = 3
\]

---

### **Final Answer Summary**
1. **Forward pass result**:
   - Hidden Layer: \( h_1 = h_2 = 1 \)
   - Output: \( y = 2 \)
   - Loss: \( \hat{R} = 2.25 \)

2. **Backward pass results**:
   - \( \frac{\partial \hat{R}}{\partial y} = 3 \)
   - \( \frac{\partial \hat{R}}{\partial w_o} = 3 \), \( \frac{\partial \hat{R}}{\partial b_o} = 3 \)
   - \( \frac{\partial \hat{R}}{\partial w_i} = 3 \), \( \frac{\partial \hat{R}}{\partial b_i} = 3 \)


---

# Q4

### **Solution to Question 4: SGD and Optimizers**

#### **1. Let the batch size be 64. Assume that each iteration of SGD takes \(10^{-3}\) seconds. How long does it take to train the network for 20 epochs?**
   
We are given:
- **Total dataset size**: \( 80,000 \) samples
- **Training dataset size**: \( 80\% \) of 80,000 = \( 64,000 \) samples
- **Batch size**: 64
- **Each iteration time**: \(10^{-3}\) seconds
- **Total epochs**: 20

##### **Step 1: Compute number of iterations per epoch**
Each iteration processes **one mini-batch**, so the number of iterations per epoch is:

\[
\frac{\text{Training samples}}{\text{Batch size}} = \frac{64,000}{64} = 1,000
\]

##### **Step 2: Compute total iterations**
Since we train for **20 epochs**, the total number of iterations is:

\[
\text{Total iterations} = 1,000 \times 20 = 20,000
\]

##### **Step 3: Compute total time**
Each iteration takes \(10^{-3}\) seconds, so the total training time is:

\[
\text{Total time} = 20,000 \times 10^{-3} = 20 \text{ seconds}
\]

Thus, the network takes **20 seconds** to train for 20 epochs.

---

#### **2. Are the samples within the first mini-batch of the first and second epochs the same? Explain your answer.**

**No, the samples in the first mini-batch of the first and second epochs are not necessarily the same.**

- **Mini-batch stochastic gradient descent (SGD)** involves **shuffling** the dataset before each epoch. 
- **Shuffling ensures that** the data is seen in a different order in every epoch, preventing the model from learning a fixed pattern from the sequence of training data.

Thus, **due to dataset shuffling, the first mini-batch of each epoch is different from the previous epoch**.

---

#### **3. Assume that we increase the batch size to 128. How does the training time change as compared to the former choice of batch size? Is there any benefit in doing that?**

##### **Step 1: Compute new number of iterations per epoch**
If the batch size is **doubled** to 128, the number of iterations per epoch becomes:

\[
\frac{\text{Training samples}}{\text{New batch size}} = \frac{64,000}{128} = 500
\]

##### **Step 2: Compute new total iterations**
With 20 epochs:

\[
\text{Total iterations} = 500 \times 20 = 10,000
\]

##### **Step 3: Compute new total training time**
Since each iteration still takes \(10^{-3}\) seconds, the new total training time is:

\[
\text{Total time} = 10,000 \times 10^{-3} = 10 \text{ seconds}
\]

Thus, **doubling the batch size reduces the total training time by half, from 20 seconds to 10 seconds**.

##### **Step 4: Trade-offs of increasing batch size**
**Pros:**
- **Faster training**: Fewer iterations per epoch reduce computation time.
- **More stable updates**: Larger batches lead to **smoother gradient updates**.

**Cons:**
- **Worse generalization**: Larger batches can lead to models converging to sharp minima, which may generalize poorly to unseen data.
- **Higher memory consumption**: Larger batches require more GPU/CPU memory.

Thus, while increasing the batch size improves speed, it may **reduce the model's generalization performance**.

---

#### **4. Compare RMSprop and Adam to vanilla mini-batch SGD.**

##### **(i) RMSprop (Root Mean Squared Propagation)**

- **Key idea**: RMSprop **adjusts the learning rate for each parameter individually** by normalizing the updates using the moving average of past squared gradients.
- **Formula for weight update**:

  \[
  v_t = \beta v_{t-1} + (1 - \beta) g_t^2
  \]

  \[
  \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t + \epsilon}} g_t
  \]

  where:
  - \( g_t \) is the gradient,
  - \( v_t \) is the moving average of squared gradients,
  - \( \beta \) is the decay factor (e.g., 0.9),
  - \( \epsilon \) is a small constant for numerical stability.

- **Advantage**: Helps prevent oscillations in training and adaptively adjusts learning rates for different parameters.

##### **(ii) Adam (Adaptive Moment Estimation)**

- **Key idea**: Adam combines **momentum** and **adaptive learning rate scaling** from RMSprop.
- **Formula for weight update**:

  \[
  m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
  \]

  \[
  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
  \]

  \[
  \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
  \]

  \[
  \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
  \]

- **Advantages**:
  - **Combines benefits of both momentum and adaptive learning rates**.
  - **Works well in practice** for deep learning problems.

##### **Comparison to Vanilla Mini-Batch SGD**
| Optimizer  | Adaptive Learning Rate? | Momentum? | Works Well for? |
|------------|----------------------|----------|----------------|
| **SGD** | No | No | Simple problems, small datasets |
| **RMSprop** | Yes | No | Recurrent networks, non-stationary problems |
| **Adam** | Yes | Yes | General deep learning, CNNs, NLP |

**Conclusion**: **Adam is generally the best choice** as it **converges faster and is more robust to different types of problems**.

---

#### **5. Between RMSprop and Adam, which one do you choose?**

I choose **Adam**, because:
- It has both **adaptive learning rates** (like RMSprop) and **momentum** (like SGD with momentum).
- It is **less sensitive to hyperparameters**.
- It **works well for most deep learning applications**, including CNNs and NLP.

However, **RMSprop is better suited for RNNs**, where gradients tend to explode or vanish.

---

#### **6. What are the key parameters of your chosen optimizer? What are your suggestions for their values?**

For **Adam**, the key parameters are:

| Parameter | Description | Suggested Value |
|-----------|-------------|----------------|
| \( \eta \) (learning rate) | Step size for updates | **0.001** (default) |
| \( \beta_1 \) | Decay rate for first moment (momentum) | **0.9** |
| \( \beta_2 \) | Decay rate for second moment (RMS) | **0.999** |
| \( \epsilon \) | Small constant to prevent division by zero | **\(10^{-8}\)** |

**Why these values?**
- The default values (0.001, 0.9, 0.999) **work well across many deep learning tasks**.
- \( \beta_1 \) helps **smooth gradients**, while \( \beta_2 \) helps with **adaptive learning rates**.
- \( \epsilon \) ensures numerical stability.

---

### **Final Summary**
- **SGD requires careful tuning** but is simple.
- **RMSprop helps with adaptive learning** but lacks momentum.
- **Adam is the best choice** due to its combination of momentum and adaptive learning.
- **Increasing batch size improves training time but can hurt generalization**.