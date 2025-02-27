# Q9

This problem is about **multiclass classification** using a neural network for **MNIST** (digit classification with 10 classes: 0-9). Instead of **softmax**, this problem uses a different activation function.

### **1. Given Activation Function**
The activation function for the output layer is defined as:

\[
y_i = \frac{z_i^2}{\sum_{j=1}^{10} z_j^2}
\]

- This function **normalizes** the squared values of \( z \) across all 10 output neurons.
- The output \( y \) represents the **probability distribution** over 10 classes.

### **2. Given Input to the Output Layer**
The vector \( \mathbf{z} \) at the **output layer before activation** is:

\[
\mathbf{z} = [2, 2, 2, 2, 1, 1, 1, 1, 0.5, 1]^T
\]

This means that before applying the activation function, each class has these scores.

### **3. True Label of the Image**
- The image is a **number 3** (MNIST digit "3").
- Since MNIST labels are **zero-indexed**, the correct label **corresponds to index 3**.
- The **true label vector** (one-hot encoding) is:

  \[
  v = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]^T
  \]

---

## **Step-by-Step Breakdown of the Problem**

### **1. Compute the Activated Vector**
The activation function given is:

\[
y_i = \frac{z_i^2}{\sum_{j=1}^{10} z_j^2}
\]

- Compute the denominator:

  \[
  S = \sum_{j=1}^{10} z_j^2 = 2^2 + 2^2 + 2^2 + 2^2 + 1^2 + 1^2 + 1^2 + 1^2 + 0.5^2 + 1^2
  \]

  \[
  S = 4 + 4 + 4 + 4 + 1 + 1 + 1 + 1 + 0.25 + 1 = 21.25
  \]

- Compute each probability \( y_i \):

  \[
  y_1 = \frac{2^2}{21.25} = \frac{4}{21.25} \approx 0.188
  \]
  \[
  y_2 = y_3 = y_4 = \frac{4}{21.25} \approx 0.188
  \]
  \[
  y_5 = y_6 = y_7 = y_8 = \frac{1}{21.25} \approx 0.047
  \]
  \[
  y_9 = \frac{0.25}{21.25} \approx 0.012
  \]
  \[
  y_{10} = \frac{1}{21.25} \approx 0.047
  \]

So, the **activated vector \( y \)** is:

\[
y = [0.188, 0.188, 0.188, 0.188, 0.047, 0.047, 0.047, 0.047, 0.012, 0.047]^T
\]

---

### **2. Compute the Cross-Entropy Loss**
The cross-entropy loss is:

\[
L = -\sum_{i=1}^{10} v_i \log(y_i)
\]

Since the true label **is digit 3** (which corresponds to index 3 in zero-based indexing):

\[
L = -\log(y_4)
\]

\[
= -\log(0.188) \approx -(-1.67) = 1.67
\]

So, the **cross-entropy loss** is **1.67**.

---

### **3. Compute the Gradient of the Loss w.r.t. Activated Vector**
The **derivative of the cross-entropy loss** w.r.t. the activated vector \( y \) is:

\[
\frac{\partial L}{\partial y_i} = y_i - v_i
\]

Since **only index 3 is the true label (with \( v_4 = 1 \))**, the gradients for each output are:

\[
\frac{\partial L}{\partial y_i} =
\begin{cases}
y_i, & \text{if } i \neq 4 \\
y_4 - 1, & \text{if } i = 4
\end{cases}
\]

Substituting values:

\[
\frac{\partial L}{\partial y} = [0.188, 0.188, 0.188, -0.812, 0.047, 0.047, 0.047, 0.047, 0.012, 0.047]^T
\]

---

### **4. Backpropagation from Activated Vector to \( z \)**
To **propagate gradients backward**, we compute:

\[
\frac{\partial L}{\partial z_i} = \sum_j \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial z_i}
\]

Since \( y \) depends on \( z \) through the squared normalization:

\[
\frac{\partial y_i}{\partial z_i} = \frac{2 z_i}{S} - 2 y_i \frac{z_i}{S}
\]

This means the gradients are **backpropagated through the Jacobian matrix**. The key steps are:
1. **Compute \(\frac{\partial y}{\partial z}\) using the given squared activation function.**
2. **Multiply the loss gradient by this Jacobian matrix.**
3. **Update weights using gradient descent.**

---

## **Final Answers**
1. **Activated vector:**
   \[
   y = [0.188, 0.188, 0.188, 0.188, 0.047, 0.047, 0.047, 0.047, 0.012, 0.047]^T
   \]
2. **Cross-entropy loss:**
   \[
   L = 1.67
   \]
3. **Gradient of loss w.r.t. \( y \):**
   \[
   \frac{\partial L}{\partial y} = [0.188, 0.188, 0.188, -0.812, 0.047, 0.047, 0.047, 0.047, 0.012, 0.047]^T
   \]
4. **Backpropagation to \( z \):**
   - Use the **Jacobian matrix** to transform the gradient of \( y \) into the gradient of \( z \).

---

### **Conclusion**
- **Step 1:** Compute the **activated vector \( y \)** using the given squared normalization.  
- **Step 2:** Compute **cross-entropy loss** using the one-hot encoded label.  
- **Step 3:** Compute **gradients** of the loss w.r.t. activated vector \( y \).  
- **Step 4:** **Backpropagate the gradient to \( z \) using the Jacobian matrix transformation**.  

---

# Q10

### **Question 10: Mini-Batch Training Explanation**
This problem deals with **mini-batch training using Stochastic Gradient Descent (SGD)** on the **CIFAR-10 dataset**. Let's go step by step.

---

## **1. Compute the Total Number of Iterations**
We are given:
- **Training dataset size** = **50,000 images**.
- **Mini-batch size** = **64**.
- **Number of epochs** = **40**.

Each **epoch** processes all **50,000 images**, but in **mini-batches of size 64**.  
The number of **mini-batches per epoch**:

\[
\frac{50,000}{64} = 781.25 \approx 782 \text{ (rounding up)}
\]

Since training runs for **40 epochs**, the total number of **iterations (mini-batch updates)**:

\[
782 \times 40 = 31,280
\]

Thus, the **total number of iterations is 31,280**.

---

## **2. Compute the Number of Forward Passes for Testing**
We are also testing the model **every 2 epochs**.

- The training runs for **40 epochs**.
- Testing happens **every 2 epochs**.
- The total number of testing sessions:

\[
\frac{40}{2} = 20
\]

Each test runs on the **entire test set of 10,000 images**.  
Thus, the **total number of forward passes during testing**:

\[
20 \times 10,000 = 200,000
\]

---

## **3. What Does "Stochastic" Refer To?**
- In **Stochastic Gradient Descent (SGD)**, we **do not** use the entire dataset at once.  
- Instead, we take **small random batches** from the dataset in each iteration.
- The term **"stochastic"** means **random selection**, specifically:
  - The training data is **shuffled** at the beginning of each epoch.
  - Each mini-batch contains **a random subset** of the dataset.
  - This randomness helps avoid **overfitting** and improves **generalization**.

Thus, **stochastic** refers to **random shuffling at the beginning of each epoch**.

---

# Q11

### **Question 11: Optimizers - Explanation**

This question covers **optimization methods** used in deep learning, particularly focusing on **convergence speed, momentum, Rprop vs. RMSprop, and Adam optimizer**.

---

## **1. Convergence Order for Linear Rate Optimization**
We are given that the optimizer converges **linearly** and need to determine the **order of time** it takes to reach an \( \epsilon \)-neighborhood of the minimum.

- **Linear convergence** means the error decreases proportionally in each step:
  
  \[
  f(x_{t+1}) - f^* \leq c (f(x_t) - f^*)
  \]

  where \( c \) is a constant between **0 and 1**.

- The time complexity for **linear convergence** is:

  \[
  \mathcal{O}(1/\epsilon)
  \]

  meaning it takes **\( \mathcal{O}(1/\epsilon) \) iterations** to get within **\( \epsilon \)** of the optimal solution.

✅ **Final Answer: \( \mathcal{O}(1/\epsilon) \)**

---

## **2. Difference Between Momentum Optimizer and Standard Gradient Descent**
### **Standard Gradient Descent (SGD)**
- SGD updates weights using:

  \[
  w_{t+1} = w_t - \eta \nabla f(w_t)
  \]

  where:
  - \( \eta \) is the learning rate,
  - \( \nabla f(w_t) \) is the gradient.

- **Issues with SGD**:
  - If gradients oscillate a lot, it takes longer to converge.
  - It slows down in ravines (steep and narrow valleys in loss function).

---

### **Momentum-Based Gradient Descent**
Momentum helps **smooth out oscillations** and **accelerates convergence** by keeping track of past gradients:

\[
v_t = \beta v_{t-1} + (1 - \beta) \nabla f(w_t)
\]

\[
w_{t+1} = w_t - \eta v_t
\]

where:
- \( v_t \) is the accumulated gradient (velocity),
- \( \beta \) (typically **0.9**) determines how much past gradients influence updates.

✅ **Key Difference**:  
Momentum **stores past gradients** to **accelerate learning and reduce oscillations**, whereas **SGD only relies on the current gradient**.

---

## **3. Difference Between Rprop and RMSprop - Which is Better for Mini-Batch Training?**
### **Resilient Propagation (Rprop)**
- Only **uses the sign** of the gradient (\(+\) or \(-\)), ignoring magnitude.
- **Independent of gradient size**, so it adapts well.
- Works well for **batch gradient descent** (full dataset).

✅ **Limitation**:  
**Not suitable for mini-batch training** because it does **not scale gradient updates properly**.

---

### **Root Mean Square Propagation (RMSprop)**
- Uses the **moving average of squared gradients** to normalize updates:

  \[
  v_t = \beta v_{t-1} + (1 - \beta) \nabla f(w_t)^2
  \]

  \[
  w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \nabla f(w_t)
  \]

- This prevents **large updates for frequent gradients** and **scales small updates for rare gradients**.
- **Well-suited for mini-batch training** because it **adapts to different gradient magnitudes dynamically**.

✅ **Which One for Mini-Batch Training?**  
Use **RMSprop** because it **adjusts learning rates per parameter** based on gradient history, making it **suitable for mini-batch updates**.

---

## **4. Two Key Ideas in Adam Optimizer**
Adam (**Adaptive Moment Estimation**) combines the best parts of **Momentum** and **RMSprop**.

- **First key idea (Momentum-like updates):**
  
  \[
  m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla f(w_t)
  \]

  where \( m_t \) is the **moving average of gradients** (like momentum).

- **Second key idea (Scaling using RMSprop-like adaptive learning rate):**
  
  \[
  v_t = \beta_2 v_{t-1} + (1 - \beta_2) \nabla f(w_t)^2
  \]

  \[
  w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t
  \]

✅ **Summary**:  
- **Momentum effect**: Uses **gradient moving average** to smooth updates.
- **RMSprop effect**: **Scales gradients** based on past values.

---

# Q12

### **Question 12: Overfitting - Explanation**

This problem is about **overfitting in neural networks** and how to prevent it. Let's analyze the problem and solution step by step.

---

## **1. Do You Expect the Trained Model to be Overfitted?**
✅ **Answer: Yes. The model is too complex for this simple task.**

### **Why?**
- **Dataset size is small**: The dataset has **only 1000 training samples**, which is too small for a large neural network.
- **Model is too complex**: The given **Fully Connected Neural Network (FNN)** has:
  - **6 hidden layers**.
  - **128 neurons per layer**.
  - **High number of parameters** compared to only 1000 training samples.
- **Training for 300 epochs**: Long training without proper regularization increases overfitting risk.

🔹 **Overfitting happens when the model memorizes the training data instead of generalizing to new samples.**  

---

## **2. How to Reduce Overfitting Without Changing the Model?**
✅ **Solution 1: Regularization (L1/L2 Weight Decay)**
- **L1 Regularization** (Lasso): Encourages sparsity by penalizing absolute weight values.
- **L2 Regularization** (Ridge): Adds a penalty to large weights, preventing overfitting.
- The new **loss function** with L2 regularization is:

  \[
  L = L_{\text{original}} + \lambda \sum W^2
  \]

  where \( \lambda \) is the regularization strength.

✅ **Solution 2: Data Augmentation & Dropout**
- **Data Augmentation** (if applicable): Generates new variations of training data.
- **Dropout**: Randomly disables neurons during training, preventing dependency on specific features.
  - Dropout rate (e.g., 0.5) means **50% of neurons are randomly deactivated per batch**.
  - Helps prevent over-reliance on small patterns.

---

## **3. How to Prevent Overfitting by Modifying the Model?**
✅ **Solution: Reduce Model Complexity**
- **Reduce the number of hidden layers**: Instead of **6 hidden layers**, try **2-3 layers**.
- **Reduce neurons per layer**: Instead of **128 neurons per layer**, try **32 or 64**.
- **Use validation to find the best model**: Train models with different configurations and choose the one that performs best on validation data.

---

# Q13

### **Question 13: Data Cleaning - Explanation**
This problem is about **identifying duplicates and detecting outliers** in a dataset containing **heights (cm) and weights (kg) of individuals**.

---

## **1. Detecting Duplicates**
✅ **Answer: Yes, there is a duplicate.**

### **Why?**
- The data includes **(183.6, 88.1) twice**.
- It is **unlikely** that two different individuals have **exactly** the same height and weight up to one decimal place.
- Duplicates should be removed in data cleaning to avoid bias in analysis.

✅ **Final Answer: (183.6, 88.1) is a duplicate and should be removed.**

---

## **2. Detecting Outliers**
✅ **Answer: Yes, there are outliers.**

Outliers are data points that **do not follow the general trend of the dataset**. Two suspicious points:
1. **(178.9, 23.1)**:  
   - **Weight (23.1 kg) is too low** for someone of 178.9 cm height.
   - **Does not fit normal human body proportions**.
   
2. **(243.6, 80.1)**:  
   - **Height (243.6 cm) is extremely tall** (around 8 feet).
   - **Rare and significantly different from others in the dataset**.

---

## **3. Type of Outliers: Univariate vs. Multivariate**
### **Multivariate Outlier**:
- **(178.9, 23.1)** is **multivariate** because:
  - **Individually**, 178.9 cm is a normal height and 23.1 kg is a valid weight.
  - **Together**, the combination is unrealistic (very tall but extremely underweight).
  - It does not match typical height-weight correlations.

### **Univariate Outlier**:
- **(243.6, 80.1)** is **univariate** because:
  - **Height (243.6 cm) alone is extreme**, while 80.1 kg is normal.
  - This height is highly unusual, making it an outlier **on a single variable**.

✅ **Final Answer:**
| **Outlier** | **Type** | **Reason** |
|------------|---------|------------|
| **(178.9, 23.1)** | **Multivariate** | Height and weight **do not match typical proportions**. |
| **(243.6, 80.1)** | **Univariate** | Height **alone is extremely high**, while weight is normal. |

---

### **Summary**
1. **Duplicate detected**: (183.6, 88.1).
2. **Outliers found**: (178.9, 23.1) (multivariate) and (243.6, 80.1) (univariate).
3. **Explanation**: Multivariate outliers occur when two variables don’t match, while univariate outliers occur when a single value is extreme.

