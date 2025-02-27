# Q1.1

## **Solution to Question 1: Loan Payment Prediction Model**

### **1. Machine Learning Model Selection**
The problem requires predicting a customer’s **ability for monthly payment** based on their **income and bank records from the last 3 years**. This is a **regression problem**, as we are estimating a continuous value.

A suitable model for this task is a **Feedforward Neural Network (FNN)**, as covered in Chapter 2. This model can capture complex patterns in financial data.

**Model Architecture:**
- **Input Layer**: Takes in numerical features related to income and bank transactions.
- **Hidden Layers**: Multiple layers with ReLU activation for non-linearity.
- **Output Layer**: A single neuron with a **Sigmoid activation** function if we are predicting a probability (e.g., likelihood of being able to pay) or a **Linear activation** if predicting a continuous amount.

\[
\hat{y} = F(x; W, b)
\]

where \( x \) is the input feature vector, and \( W, b \) are learnable parameters.

---

### **2. Training Procedure**
We train the model using **Stochastic Gradient Descent (SGD) with mini-batches**, as covered in **Chapter 3**. The training steps are:

1. **Data Preprocessing**:
   - Normalize income and bank record features using **mean normalization**.
   - Handle missing values appropriately.
   - Split data into **training (80%) and validation (20%) sets**.

2. **Forward Pass**:
   - Compute the network’s output \( \hat{y} \) using forward propagation.

3. **Loss Calculation**:
   - Compute the error using an appropriate **loss function** (discussed in the next section).

4. **Backward Pass (Backpropagation)**:
   - Compute gradients of the loss w.r.t. model parameters.
   - Update weights using **SGD or Adam optimizer**.

5. **Evaluation**:
   - Track the model’s performance using metrics such as **Mean Squared Error (MSE) or Mean Absolute Error (MAE)**.

---

### **3. Loss Function Selection**
Since we are predicting a **continuous value (ability for monthly payment)**, we use **Mean Squared Error (MSE)** as the loss function:

\[
L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
\]

where:
- \( y_i \) is the true payment ability,
- \( \hat{y}_i \) is the predicted payment ability.

**Why MSE?**
- MSE **penalizes larger errors** more heavily, making it suitable for financial predictions.
- It is **differentiable**, allowing for efficient gradient-based optimization.

---

### **4. Evaluating Generalization**
To ensure the model generalizes well, we check:
1. **Validation Performance**:  
   - Train on 80% of data, validate on 20%.  
   - Monitor **MSE on validation data**.
   
2. **Overfitting Prevention**:  
   - Use **L2 regularization (Weight Decay)** to prevent overfitting.  
   - Apply **Dropout** in hidden layers to reduce reliance on specific neurons.

3. **Test on Unseen Data**:  
   - If possible, test the model on a **new dataset** to evaluate performance.

4. **Cross-Validation**:  
   - Use **K-fold cross-validation** to check if the model performs well across different data splits.

---

## **Final Answer**
| **Step** | **Solution** |
|----------|-------------|
| **Model Choice** | **Feedforward Neural Network (FNN)** for regression |
| **Training Process** | **Mini-batch SGD with backpropagation** |
| **Loss Function** | **Mean Squared Error (MSE)** |
| **Ensuring Generalization** | **Validation set, L2 regularization, Dropout, and Cross-validation** |

This solution ensures that the bank can **accurately estimate customers' ability for monthly payments** while preventing **overfitting**.

---

# Q1.2

### **Question 2: Evaluating an Alternative Activation Function**
We need to determine whether the suggested activation function is a good replacement for **ReLU** in a **Feedforward Neural Network (FNN)**.

---

### **1. Understanding the Given Activation Function**
The function is defined as:

\[
f(x) =
\begin{cases}
\max\{(x - 1), 0.9(x - 1)\}, & x > 1 \\
\max\{(x - 1), 1.1(x - 1)\}, & x \leq 1
\end{cases}
\]

- For **\( x > 1 \)**:  
  \[
  f(x) = \max(x - 1, 0.9(x - 1))
  \]
  - Since \( x - 1 \) is always **larger** than \( 0.9(x - 1) \), we get:
    \[
    f(x) = x - 1
    \]

- For **\( x \leq 1 \)**:  
  \[
  f(x) = \max(x - 1, 1.1(x - 1))
  \]
  - Since **\( 1.1(x - 1) \) grows faster**, we get:
    \[
    f(x) = 1.1(x - 1)
    \]

#### **Comparison to ReLU**
- **ReLU (Rectified Linear Unit) is defined as**:
  \[
  \text{ReLU}(x) = \max(0, x)
  \]
- Unlike ReLU, the proposed function **can take negative values** when \( x \leq 1 \), meaning it does not have the zeroing-out effect of ReLU for negative inputs.
- The function scales negative values by **1.1** rather than **clamping them to zero** like ReLU.
- This might cause **exploding gradients** for negative values since they are **amplified instead of suppressed**.

---

### **2. Should You Use This Activation Function?**
❌ **No, I would not use this activation function in my FNN.**  

✅ **Reasons**:
1. **No Zero Clamping for Negative Values**  
   - ReLU **suppresses negative values**, preventing unnecessary activation.  
   - This function **amplifies negative values**, which can lead to instability in training.

2. **Leads to Exploding Gradients**  
   - The scaling factor **1.1** for \( x \leq 1 \) **increases gradients**, which can cause **divergence** in training.  
   - In deep networks, this can lead to **unstable weight updates**.

3. **Vanishing Gradient Issue is Not Solved**  
   - Unlike **Leaky ReLU** (which has a small positive slope for negative inputs), this function does not guarantee **gradient flow in all scenarios**.

4. **ReLU is Simpler and More Efficient**  
   - ReLU is **computationally efficient** since it only requires a max operation.
   - The proposed function **requires multiple comparisons**, making it slower.

---

### **Final Answer**
**No, I would not use this activation function.**  
- It **amplifies negative values**, leading to **exploding gradients**.  
- It **does not improve upon ReLU**, which is already effective and widely used.  
- Instead, I would consider using **Leaky ReLU** or **Parametric ReLU** if I need a non-zero gradient for negative inputs.

---

# Q1.3

### **Question 3: Evaluating a Proposed Loss Function for Binary Classification**

We need to assess whether the given loss function is a good choice for a **binary classifier** with a **sigmoid-activated output layer**.

---

## **1. Understanding the Given Loss Function**
The proposed loss function is defined as:

\[
\mathcal{L}(y, v) =
\begin{cases}
1 - v, & y > 0.75 \\
0.5, & 0.25 < y \leq 0.75 \\
v, & y \leq 0.25
\end{cases}
\]

where:
- \( y \) is the model's predicted probability after the **sigmoid activation**.
- \( v \in \{0,1\} \) is the **true label**.

This function assigns different loss values based on **confidence levels** of predictions.

---

## **2. Evaluating the Loss Function**
A good loss function for binary classification should:
1. **Penalize incorrect predictions more than correct ones.**
2. **Provide a smooth gradient for optimization.**
3. **Encourage confidence in correct predictions.**
4. **Be consistent with probabilistic outputs from the sigmoid activation.**

---

### **Problems with the Given Loss Function**
❌ **1. Loss is Constant for Middle Predictions (0.25 < y ≤ 0.75)**
- For predictions in the range **0.25 to 0.75**, the loss is **fixed at 0.5**, **regardless of correctness**.
- This means the model **receives no feedback (gradient = 0)** in this range, making it difficult to update weights.
- **SGD cannot optimize well if there is no gradient in this range.**

❌ **2. Loss is Not Symmetric**
- Ideally, a loss function should penalize incorrect predictions symmetrically.
- However:
  - If \( v = 1 \) and \( y > 0.75 \), loss = **1 - 1 = 0** ✅
  - If \( v = 0 \) and \( y > 0.75 \), loss = **1 - 0 = 1** ❌ (too high)
  - If \( v = 0 \) and \( y \leq 0.25 \), loss = **0** ✅
  - If \( v = 1 \) and \( y \leq 0.25 \), loss = **1** ❌ (wrongly penalizing correct answers)

- This **penalizes confident incorrect predictions too much while ignoring medium-confidence errors**.

❌ **3. Not a Proper Logarithmic Loss**
- The standard loss function for binary classification is **Binary Cross-Entropy (BCE)**:

  \[
  \mathcal{L}_{BCE}(y, v) = -[v \log(y) + (1 - v) \log(1 - y)]
  \]

- This loss **ensures smooth gradients** and proper feedback across all values of \( y \).
- Unlike BCE, the proposed function **ignores** errors in the range \( 0.25 < y \leq 0.75 \), which can **slow down convergence**.

---

## **3. Final Answer**
❌ **No, this is NOT a good choice of loss function for binary classification.**  

✅ **Reason:**
- The **constant loss** for **mid-confidence predictions (0.25 < y ≤ 0.75)** prevents proper gradient updates.
- **Asymmetric penalties** make it difficult for the model to learn correct decision boundaries.
- The **standard loss function for binary classification** is **Binary Cross-Entropy (BCE)**, which provides proper optimization.

---

### **Alternative (Correct) Loss Function: Binary Cross-Entropy**
\[
\mathcal{L}(y, v) = -[v \log(y) + (1 - v) \log(1 - y)]
\]

This function:
- Provides **a smooth gradient** for all values of \( y \).
- Penalizes incorrect predictions **logarithmically**, making training more stable.
- Encourages **confidence in correct predictions**.

---

### **Final Summary**
| **Criteria** | **Proposed Loss Function** | **Binary Cross-Entropy (BCE)** |
|-------------|------------------|----------------|
| **Gradient Flow** | No gradient for \( 0.25 < y \leq 0.75 \) | Smooth gradients for all \( y \) |
| **Penalty for Incorrect Predictions** | Asymmetric | Properly scaled |
| **Encourages Confident Correct Predictions?** | No | Yes |
| **Used in Deep Learning?** | No | Yes |

✅ **Conclusion**: **The proposed loss function is NOT suitable. Binary Cross-Entropy should be used instead.**  

---

# Q1.4

### **Question 4: Effect of Training Data Size on Model Performance**

This question focuses on the impact of **reducing the training dataset size** in a **Fully Connected Neural Network (FNN) trained on MNIST**.

---

## **1. Expected Observation After Testing**
Your friend **only uses 3,000 training samples** instead of the full **60,000 MNIST training set**.  
After testing, their model will likely show **one or both of the following issues**:

✅ **1. High Training Accuracy but Low Test Accuracy (Overfitting)**
- The model memorizes the small dataset but **fails to generalize** to unseen images.
- Training accuracy might be **90%+**, but test accuracy could drop **below 70%**.

✅ **2. Poor Training and Test Accuracy (Underfitting)**
- If the model is too **complex** (too many parameters), it may struggle to learn effectively from **only 3,000 images**.
- Training accuracy might remain **low (~70%)** due to **insufficient data** for meaningful feature learning.

---

## **2. Why Does This Happen?**
The performance drop is due to **Data Size and Generalization**:

### **1. Small Dataset Causes Overfitting**
- With only **3,000 samples**, the model has **too few examples per class** (only ~300 per digit in MNIST).
- The network learns to memorize instead of generalizing.
- **More data helps smooth decision boundaries** in classification.

### **2. Deep Networks Require Large Datasets**
- FNNs have **millions of parameters** that need large datasets to train effectively.
- Training on **3,000 samples is not enough** to learn good representations.

### **3. Limited Variation in Data**
- The full 60,000 images capture **many variations** of handwritten digits.
- With only 3,000 images, the model may not **see enough examples** of different handwriting styles, rotations, noise, etc.

---

## **3. Explanation of the Observation**
Your friend will likely see:
1. **Poor Generalization**: The model performs **well on training data** but fails on test data.
2. **High Variance**: Small dataset training makes the model **sensitive to small changes**, leading to inconsistent results.
3. **Increased Risk of Memorization**: The model learns **specific training examples instead of general patterns**.

✅ **Final Explanation:**  
Reducing the training dataset from **60,000 to 3,000** leads to **overfitting** or **underfitting**, causing **poor generalization and lower test accuracy**.

---

## **4. How to Improve Performance with Small Data?**
If the dataset size is fixed at **3,000 samples**, we can improve generalization by:
1. **Data Augmentation**: Introduce **rotations, shifts, and noise** to create artificial samples.
2. **Regularization**:
   - **L2 weight decay** to prevent large weight values.
   - **Dropout layers** to prevent over-reliance on specific neurons.
3. **Transfer Learning**: Pre-train on a large dataset and fine-tune on the small MNIST subset.

---

### **Final Summary**
| **Observation** | **Reason** |
|----------------|------------|
| **Lower test accuracy** | Too few training samples lead to **poor generalization**. |
| **Overfitting** | Model memorizes the small dataset instead of learning patterns. |
| **High variance** | Model is **unstable** due to insufficient data. |

✅ **Conclusion**: A deep FNN **needs a large dataset** to generalize well. Training on only 3,000 samples leads to **overfitting or poor accuracy**.

---

# Q2

## **1. Specify the Width of Each Layer and Compute the Total Number of Learnable Parameters**

### **Neural Network Architecture**
Each **32 × 32 grayscale image** is flattened into a **vector of 1024 features** (since \( 32 \times 32 = 1024 \)).

The network structure consists of:
- **Input Layer:** \( 1024 \) neurons (one per pixel).
- **Hidden Layer 1:** \( 100 \) neurons.
- **Hidden Layer 2:** \( 50 \) neurons.
- **Hidden Layer 3:** \( 50 \) neurons.
- **Hidden Layer 4:** \( 50 \) neurons.
- **Output Layer:** \( 5 \) neurons (for **5 classes**: cat, dog, car, airplane, house).

Each layer \( L \) has **weights \( W \) and biases \( b \)**:

\[
W^{(L)} x^{(L)} + b^{(L)}
\]

where:
- \( W^{(L)} \) is the weight matrix,
- \( x^{(L)} \) is the input to layer \( L \),
- \( b^{(L)} \) is the bias vector.

The **number of parameters** in each layer is given by:

\[
\text{Parameters} = (\text{Input size} + 1) \times \text{Output size}
\]

---

### **Computing Parameters for Each Layer**
#### **1. Input to First Hidden Layer**  
- **Weights**: \( (1024 + 1) \times 100 = 102,500 \)  
- **(1024 inputs + bias for each neuron)**  

#### **2. First to Second Hidden Layer**  
- **Weights**: \( (100 + 1) \times 50 = 5,050 \)  

#### **3. Second to Third Hidden Layer**  
- **Weights**: \( (50 + 1) \times 50 = 2,550 \)  

#### **4. Third to Fourth Hidden Layer**  
- **Weights**: \( (50 + 1) \times 50 = 2,550 \)  

#### **5. Fourth Hidden Layer to Output Layer**  
- **Weights**: \( (50 + 1) \times 5 = 255 \)  

---

### **Total Learnable Parameters**
Summing all the above:

\[
102,500 + 5,050 + 2,550 + 2,550 + 255 = 112,905
\]

✅ **Final Answer**: **Total Parameters = 112,905**

---

## **2. Integer Labeling for Classes**
We assign each class an integer:

\[
\{\text{cat, dog, car, airplane, house}\} \rightarrow \{0, 1, 2, 3, 4\}
\]

✅ **Final Answer**:
- **Cat → 0**
- **Dog → 1**
- **Car → 2**
- **Airplane → 3**
- **House → 4**

---

## **3. Compute the Cross-Entropy Loss for an Image of "Car"**
### **Softmax Activation Function**
The output layer applies the **softmax function** to the logits \( z \):

\[
y_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
\]

where:
- \( y_i \) is the probability for class \( i \),
- \( z_i \) is the pre-activation output of neuron \( i \),
- \( C = 5 \) (number of classes).

### **Cross-Entropy Loss Formula**
Given **true label vector** \( v \) (one-hot encoded):

\[
v = [0, 0, 1, 0, 0] \quad \text{(for "Car", which is class 2)}
\]

\[
CE(y, v) = -\sum_{i=1}^{C} v_i \log y_i
\]

Since \( v \) is one-hot encoded, only \( y_2 \) contributes:

\[
CE(y, v) = -\log y_2
\]

If **softmax output is**:

\[
y = [0.1, 0.2, 0.5, 0.1, 0.1]
\]

Then:

\[
CE = -\log(0.5) = 0.693
\]

✅ **Final Answer**: **\( -\log y_2 \)**

---

## **4. Train vs. Test Dataset for Model Validation**
The claim is that the model **has been trained**.  
To validate, we check **generalization**.

### **Choosing Between Training and Test Dataset**
✅ **Use the Test Dataset**
- If we use **training data**, the model may have **memorized it** (overfitting).
- **Test data checks generalization** (real-world performance).

### **Mathematical Justification**
Overfitting is detected by:

\[
\Delta A = A_{\text{train}} - A_{\text{test}}
\]

where:
- \( A_{\text{train}} \) = Training Accuracy
- \( A_{\text{test}} \) = Test Accuracy

If \( \Delta A \) is **large** (e.g., \( A_{\text{train}} = 98\% \), \( A_{\text{test}} = 75\% \)), then **overfitting is present**.

✅ **Final Answer**: **Use the test dataset to validate training.**

---

## **5. Modify the FNN for Binary Classification (Cat vs. Non-Cat)**
We modify the FNN to **classify only "Cat" vs. "Non-Cat"**.

### **Changes**
- **Change Output Layer**:
  - Instead of **5 neurons (for 5 classes)**, use **1 neuron** with **Sigmoid Activation**.
  - The output represents **\( P(\text{Cat}) \)**.

- **New Integer Labeling**:
  - **Cat → 1**
  - **Non-Cat (Dog, Car, Airplane, House) → 0**

- **New Loss Function**: Binary Cross-Entropy (BCE)

\[
L = - \left[ v \log y + (1 - v) \log (1 - y) \right]
\]

✅ **Final Answer**:  
- **Modify output to 1 neuron (Sigmoid activation)**.
- **Labels: {1 (Cat), 0 (Not Cat)}**.

---

## **6. Predicting Output for a Mixed Image**
Given a mixed image:

\[
\bar{x} = 0.5 (x_{\text{car}} + x_{\text{airplane}})
\]

Each FNN sees an **unclear input**, leading to:

### **1. Original Multiclass Model**
Since softmax is non-linear:

\[
\bar{y}_i = \frac{e^{\bar{z}_i}}{\sum_{j} e^{\bar{z}_j}}
\]

The resulting **softmax output will split probability** between Car and Airplane.

### **2. Binary Model (Cat vs. Non-Cat)**
Since the image is **not a cat**, the model should predict **"Not Cat" (Output close to 0)**.

The **Sigmoid activation** gives:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

where \( z \) is the final activation.

✅ **Final Answer**:  
- **Multiclass Model**: Softmax probabilities will be **split between "Car" and "Airplane"**.
- **Binary Model (Cat/Not-Cat)**: Sigmoid output will be **close to 0 (Not a Cat)**.

---

## **Final Summary**
| **Question** | **Final Answer** |
|-------------|------------------|
| **Width & Parameters** | **Total: 112,905 parameters** |
| **Integer Labels** | **{Cat: 0, Dog: 1, Car: 2, Airplane: 3, House: 4}** |
| **Cross-Entropy Loss for Car** | **\( -\log y_2 \)** |
| **Validate Training** | **Use Test Data** |
| **Modify for Binary Classification** | **1 neuron (Sigmoid), Labels: {1 (Cat), 0 (Not Cat)}** |
| **Output for Mixed Image** | **Multiclass: Softmax split between Car/Airplane, Binary: Output near 0 (Not a Cat)** |


# Q3

### **Full Solution to Question 3: Forward and Backward Pass**
This question involves **forward propagation**, **computing values at each node**, and **deriving gradients using backpropagation**.

---

## **1. Computation Graph for Forward Pass**
The given FNN consists of:
- **Input**: Scalar \( x \)
- **Hidden Layer**: 2 neurons with **ReLU activation**
- **Output Layer**: Single neuron with **Sigmoid activation**
- **Loss Function**: Binary Cross-Entropy (BCE)

### **Computation Graph Structure**
1. **Input:** \( x \)
2. **Hidden Layer (Before Activation):**  
   \[
   z_1 = w_{1} x + b_1, \quad z_2 = w_{2} x + b_2
   \]
3. **Hidden Layer (After Activation, using ReLU):**  
   \[
   a_1 = \max(0, z_1), \quad a_2 = \max(0, z_2)
   \]
4. **Output Layer (Before Activation):**  
   \[
   z = w_o a_1 + w_o a_2 + b_o
   \]
5. **Output Layer (After Activation, using Sigmoid):**  
   \[
   y = \sigma(z) = \frac{1}{1 + e^{-z}}
   \]
6. **Binary Cross-Entropy Loss:**  
   \[
   \hat{R} = - v \log y - (1 - v) \log (1 - y)
   \]

---

## **2. Forward Pass Calculation for \( x = 0.5 \), \( v = 1 \)**
Given:
- **All weights \( w \) are 1**.
- **All biases \( b \) are 0**.

### **Step 1: Compute Hidden Layer Values**
Using the weight and bias settings:

\[
z_1 = (1 \times 0.5) + 0 = 0.5, \quad z_2 = (1 \times 0.5) + 0 = 0.5
\]

Applying **ReLU Activation**:

\[
a_1 = \max(0, 0.5) = 0.5, \quad a_2 = \max(0, 0.5) = 0.5
\]

### **Step 2: Compute Output Layer Values**
\[
z = (1 \times 0.5) + (1 \times 0.5) + 0 = 1.0
\]

Applying **Sigmoid Activation**:

\[
y = \sigma(1.0) = \frac{1}{1 + e^{-1}}
\]

### **Step 3: Compute Loss**
Since \( v = 1 \), the **Binary Cross-Entropy loss** simplifies to:

\[
\hat{R} = - \log y = - \log \sigma(1)
\]

✅ **Final Answer (Forward Pass Values)**
- **Hidden Layer Outputs:** \( a_1 = 0.5, a_2 = 0.5 \)
- **Output Before Sigmoid:** \( z = 1.0 \)
- **Output After Sigmoid:** \( y = \sigma(1) \)
- **Loss:** \( -\log \sigma(1) \)

---

## **3. Compute Gradient of \( \hat{R} \) w.r.t. Hidden Layer Output (Backpropagation)**
We apply the chain rule:

\[
\frac{\partial \hat{R}}{\partial a_i} = \frac{\partial \hat{R}}{\partial y} \times \frac{\partial y}{\partial z} \times \frac{\partial z}{\partial a_i}
\]

### **Step 1: Compute \( \frac{\partial \hat{R}}{\partial y} \)**
From the **BCE loss function**:

\[
\frac{\partial \hat{R}}{\partial y} = - \frac{v}{y} + \frac{1 - v}{1 - y}
\]

Since \( v = 1 \):

\[
\frac{\partial \hat{R}}{\partial y} = -\frac{1}{y}
\]

Substituting \( y = \sigma(1) \):

\[
\frac{\partial \hat{R}}{\partial y} = -\frac{1}{\sigma(1)}
\]

### **Step 2: Compute \( \frac{\partial y}{\partial z} \)**
Using **Sigmoid Derivative**:

\[
\frac{\partial y}{\partial z} = y (1 - y) = \sigma(1) (1 - \sigma(1))
\]

### **Step 3: Compute \( \frac{\partial z}{\partial a_i} \)**
Since:

\[
z = w_o a_1 + w_o a_2 + b_o
\]

\[
\frac{\partial z}{\partial a_1} = w_o = 1, \quad \frac{\partial z}{\partial a_2} = w_o = 1
\]

### **Final Gradient Computation**
\[
\frac{\partial \hat{R}}{\partial a_i} = \left(-\frac{1}{\sigma(1)} \right) \times \left( \sigma(1) (1 - \sigma(1)) \right) \times 1
\]

\[
= - (1 - \sigma(1))
\]

✅ **Final Answer (Gradient w.r.t. Hidden Layer Output)**
\[
\frac{\partial \hat{R}}{\partial a_1} = - (1 - \sigma(1))
\]

\[
\frac{\partial \hat{R}}{\partial a_2} = - (1 - \sigma(1))
\]

---

## **4. Compute Gradient of \( \hat{R} \) w.r.t. Output Layer Weights and Bias**
We use the chain rule:

\[
\frac{\partial \hat{R}}{\partial w_o} = \frac{\partial \hat{R}}{\partial y} \times \frac{\partial y}{\partial z} \times \frac{\partial z}{\partial w_o}
\]

\[
\frac{\partial \hat{R}}{\partial b_o} = \frac{\partial \hat{R}}{\partial y} \times \frac{\partial y}{\partial z} \times \frac{\partial z}{\partial b_o}
\]

Using:

\[
\frac{\partial z}{\partial w_o} = a_i, \quad \frac{\partial z}{\partial b_o} = 1
\]

We substitute:

\[
\frac{\partial \hat{R}}{\partial w_o} = - (1 - \sigma(1)) a_i
\]

\[
\frac{\partial \hat{R}}{\partial b_o} = - (1 - \sigma(1))
\]

✅ **Final Answer (Gradients w.r.t. Weights and Bias)**
\[
\frac{\partial \hat{R}}{\partial w_o} = - (1 - \sigma(1)) a_i
\]

\[
\frac{\partial \hat{R}}{\partial b_o} = - (1 - \sigma(1))
\]

---

## **Final Summary**
| **Step** | **Final Answer** |
|----------|------------------|
| **Forward Pass Values** | \( a_1 = 0.5, a_2 = 0.5, z = 1, y = \sigma(1), \hat{R} = -\log \sigma(1) \) |
| **Gradient w.r.t. Hidden Layer Outputs** | \( \frac{\partial \hat{R}}{\partial a_1} = - (1 - \sigma(1)) \), \( \frac{\partial \hat{R}}{\partial a_2} = - (1 - \sigma(1)) \) |
| **Gradient w.r.t. Output Layer Weights** | \( \frac{\partial \hat{R}}{\partial w_o} = - (1 - \sigma(1)) a_i \) |
| **Gradient w.r.t. Output Layer Bias** | \( \frac{\partial \hat{R}}{\partial b_o} = - (1 - \sigma(1)) \) |

✅ **Conclusion:**  
- The **computation graph** flows from \( x \) → \( z_1, z_2 \) → \( a_1, a_2 \) → \( z \) → \( y \) → \( \hat{R} \).  
- The **gradients flow backward**, following **chain rule differentiation**.


# Q4

### **Full Solution to Question 4: SGD and Optimizers**
This question focuses on **Stochastic Gradient Descent (SGD)**, **batch size effects**, and **comparing optimizers**.

---

## **1. Number of SGD Iterations after 20 Epochs**
### **Understanding Iterations in Mini-Batch SGD**
- We have **50,000 samples** in total.
- **80% of data** is used for training → **Training set size =**  
  \[
  0.8 \times 50,000 = 40,000
  \]
- **Batch size = 100**, meaning each iteration processes **100 samples**.
- **Number of iterations per epoch**:
  \[
  \frac{\text{Training samples}}{\text{Batch size}} = \frac{40,000}{100} = 400
  \]
- **Total iterations after 20 epochs**:
  \[
  400 \times 20 = 8,000
  \]

✅ **Final Answer**: **8,000 iterations after 20 epochs**

---

## **2. Are Mini-Batches the Same Across Epochs?**
No, because **SGD shuffles the dataset at the beginning of each epoch**.

### **Why Do Mini-Batches Change?**
- **Shuffling ensures different batches across epochs**.
- **If data is not shuffled**, SGD might overfit or fail to generalize.

✅ **Final Answer**: **No, the mini-batches change due to random shuffling.**

---

## **3. Effect of Changing Batch Size to 200**
If the batch size is doubled from **100 to 200**, we analyze:

### **Effect on Iterations per Epoch**
- **New iterations per epoch**:
  \[
  \frac{40,000}{200} = 200
  \]
- **Total iterations after 20 epochs**:
  \[
  200 \times 20 = 4,000
  \]
  (Half the previous number of iterations.)

### **Benefits**
✅ **More stable updates**: Larger batches give a better estimate of the gradient.  
✅ **Less frequent updates**: Reduces computational overhead.

### **Costs**
❌ **Slower convergence**: Larger batch sizes reduce gradient variance, which might slow learning.  
❌ **More memory usage**: Larger batches require more GPU memory.

✅ **Final Answer**:
- **Iterations decrease to 4,000.**
- **More stable updates but slower convergence.**
- **Higher memory requirement.**

---

## **4. Comparing Rprop and Adam to Mini-Batch SGD**
### **(i) Resilient Propagation (Rprop)**
✅ **Key Feature:** Uses only the **sign** of the gradient, not the magnitude.

- **No step size decay** → Learning rate is **adaptive**.
- **Fast convergence** in batch learning but **not suitable for mini-batch SGD**.

### **(ii) Adam (Adaptive Moment Estimation)**
✅ **Key Feature:** Uses **momentum** and **adaptive learning rates**.

- **Momentum (Exponential moving average of gradients):**
  \[
  m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
  \]
- **Adaptive scaling (RMSprop component):**
  \[
  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
  \]
- **Parameter update step:**
  \[
  \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t
  \]

✅ **Final Answer**:
- **Rprop**: Not suitable for mini-batch SGD (works best in batch settings).
- **Adam**: **Better than SGD**, adapts learning rates.

---

## **5. Choosing Between Rprop and Adam**
### **Why Choose Adam?**
✅ **Handles mini-batch SGD better** due to:
- Adaptive learning rate.
- Momentum term helps avoid local minima.

✅ **Why Not Rprop?**
- **Does not work with mini-batches**.
- **Not suitable for deep learning tasks**.

✅ **Final Answer**: **Adam is better for mini-batch training.**

---

## **6. Key Parameters of Adam**
### **Adam’s Hyperparameters**
| Parameter | Symbol | Recommended Value |
|-----------|--------|------------------|
| Learning rate | \( \eta \) | \( 0.001 \) |
| Momentum decay | \( \beta_1 \) | \( 0.9 \) |
| RMSprop decay | \( \beta_2 \) | \( 0.999 \) |
| Stability term | \( \epsilon \) | \( 10^{-8} \) |

### **Suggested Values**
- Use **\( \eta = 0.001 \)** for stability.
- Keep **\( \beta_1 = 0.9, \beta_2 = 0.999 \)** for effective momentum.

✅ **Final Answer**:
- **Key Parameters**: \( \eta, \beta_1, \beta_2, \epsilon \).
- **Suggested Values**: \( 0.001, 0.9, 0.999, 10^{-8} \).

---

## **Final Summary**
| **Question** | **Final Answer** |
|-------------|------------------|
| **Iterations in 20 Epochs** | **8,000 iterations** |
| **Same Mini-Batches Across Epochs?** | **No, due to shuffling** |
| **Effect of Batch Size 200** | **Iterations decrease, updates are stable but slower** |
| **Rprop vs. Adam** | **Adam is better for mini-batch training** |
| **Adam Hyperparameters** | **\( \eta = 0.001, \beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8} \)** |








