# CS229 Lecture Notes

# Andrew Ng and Tengyu Ma

# June 11, 2023



# Contents

# I  Supervised learning

5

# 1  Linear regression

8

# 1.1     LMS algorithm

9

# 1.2     The normal equations

13

# 1.2.1       Matrix derivatives

13

# 1.2.2     Least squares revisited

14

# 1.3     Probabilistic interpretation

15

# 1.4     Locally weighted linear regression (optional reading)

17

# 2  Classification and logistic regression

20

# 2.1     Logistic regression

20

# 2.2     Digression: the perceptron learning algorithm

23

# 2.3     Multi-class classification

24

# 2.4     Another algorithm for maximizing `(θ)`

27

# 3  Generalized linear models

29

# 3.1     The exponential family

29

# 3.2     Constructing GLMs

31

# 3.2.1     Ordinary least squares

32

# 3.2.2     Logistic regression

33

# 4  Generative learning algorithms

34

# 4.1     Gaussian discriminant analysis

35

# 4.1.1     The multivariate normal distribution

35

# 4.1.2     The Gaussian discriminant analysis model

38

# 4.1.3     Discussion: GDA and logistic regression

40

# 4.2     Naive bayes (Option Reading)

41

# 4.2.1     Laplace smoothing

44

# 4.2.2     Event models for text classification

46

1




CS229 Spring 20223

# 5   Kernel methods

5.1     Feature maps     . . . . . . . . . . . . . . . . . . . . . . . . . . .    48

5.2     LMS (least mean squares) with features . . . . . . . . . . . . .          49

5.3     LMS with the kernel trick      . . . . . . . . . . . . . . . . . . . .    49

5.4     Properties of kernels    . . . . . . . . . . . . . . . . . . . . . . .    53

# 6   Support vector machines

6.1     Margins: intuition . . . . . . . . . . . . . . . . . . . . . . . . .      59

6.2     Notation (option reading)      . . . . . . . . . . . . . . . . . . . .    61

6.3     Functional and geometric margins (option reading)          . . . . . .    61

6.4     The optimal margin classifier (option reading)       . . . . . . . . .    63

6.5     Lagrange duality (optional reading) . . . . . . . . . . . . . . .         65

6.6     Optimal margin classifiers: the dual form (option reading)         . .    68

6.7     Regularization and the non-separable case (optional reading) .            72

6.8     The SMO algorithm (optional reading)         . . . . . . . . . . . . .    73

6.8.1      Coordinate ascent . . . . . . . . . . . . . . . . . . . . .    74

6.8.2      SMO . . . . . . . . . . . . . . . . . . . . . . . . . . . .    75

# II   Deep learning

# 7   Deep learning

7.1     Supervised learning with non-linear models . . . . . . . . . . .          80

7.2     Neural networks . . . . . . . . . . . . . . . . . . . . . . . . . .       84

7.3     Modules in Modern Neural Networks          . . . . . . . . . . . . . .    92

7.4     Backpropagation      . . . . . . . . . . . . . . . . . . . . . . . . .    98

7.4.1     Preliminaries on partial derivatives   . . . . . . . . . . .    99

7.4.2     General strategy of backpropagation        . . . . . . . . . . 102

7.4.3           Backward functions for basic modules . . . . . . . . . . 105

7.4.4     Back-propagation for MLPs        . . . . . . . . . . . . . . . 107

7.5     Vectorization over training examples         . . . . . . . . . . . . . . 109

# III        Generalization and regularization

# 8   Generalization

8.1     Bias-variance tradeoff . . . . . . . . . . . . . . . . . . . . . . . 115

8.1.1            A mathematical decomposition (for regression) . . . . . 120

8.2     The double descent phenomenon . . . . . . . . . . . . . . . . . 121

8.3     Sample complexity bounds (optional readings) . . . . . . . . . 126




CS229 Spring 20223

# 8.3.1 Preliminaries

126

# 8.3.2 The case of finite H

128

# 8.3.3 The case of infinite H

131

# 9 Regularization and model selection

135

# 9.1 Regularization

135

# 9.2 Implicit regularization effect (optional reading)

137

# 9.3 Model selection via cross validation

139

# 9.4 Bayesian statistics and regularization

142

# IV Unsupervised learning

144

# 10 Clustering and the k-means algorithm

145

# 11 EM algorithms

148

# 11.1 EM for mixture of Gaussians

148

# 11.2 Jensen’s inequality

151

# 11.3 General EM algorithms

152

# 11.3.1 Other interpretation of ELBO

158

# 11.4 Mixture of Gaussians revisited

158

# 11.5 Variational inference and variational auto-encoder (optional reading)

160

# 12 Principal components analysis

165

# 13 Independent components analysis

171

# 13.1 ICA ambiguities

172

# 13.2 Densities and linear transformations

173

# 13.3 ICA algorithm

174

# 14 Self-supervised learning and foundation models

177

# 14.1 Pretraining and adaptation

177

# 14.2 Pretraining methods in computer vision

179

# 14.3 Pretrained large language models

181

# 14.3.1 Open up the blackbox of Transformers

183

# 14.3.2 Zero-shot learning and in-context learning

186




CS229 Spring 20223

# 4 Reinforcement Learning and Control

# 15 Reinforcement learning

189

# 15.1 Markov decision processes

190

# 15.2 Value iteration and policy iteration

192

# 15.3 Learning a model for an MDP

194

# 15.4 Continuous state MDPs

196

# 15.4.1 Discretization

196

# 15.4.2 Value function approximation

199

# 15.5 Connections between Policy and Value Iteration (Optional)

203

# 16 LQR, DDP and LQG

206

# 16.1 Finite-horizon MDPs

206

# 16.2 Linear Quadratic Regulation (LQR)

210

# 16.3 From non-linear dynamics to LQR

213

# 16.3.1 Linearization of dynamics

214

# 16.3.2 Differential Dynamic Programming (DDP)

214

# 16.4 Linear Quadratic Gaussian (LQG)

216

# 17 Policy Gradient (REINFORCE)

220




# Part I

# Supervised learning

5



# 6

Let’s start by talking about a few examples of supervised learning problems. Suppose we have a dataset giving the living areas and prices of 47 houses from Portland, Oregon:

| Living area (feet²) | Price (1000$s) |
| ------------------- | -------------- |
| 2104                | 400            |
| 1600                | 330            |
| 2400                | 369            |
| 1416                | 232            |
| 3000                | 540            |
| .                   | .              |
| .                   | .              |
| .                   | .              |

We can plot this data:

housing prices

Given data like this, how can we learn to predict the prices of other houses in Portland, as a function of the size of their living areas?

To establish notation for future use, we’ll use *x(i) to denote the “input” variables (living area in this example), also called input features, and y(i) to denote the “output” or target variable that we are trying to predict (price). A pair (x(i), y(i)) is called a training example, and the dataset that we’ll be using to learn—a list of n training examples {(x(i), y(i)); i = 1, . . . , n}—is called a training set. Note that the superscript “(i)” in the notation is simply an index into the training set, and has nothing to do with exponentiation. We will also use X to denote the space of input values, and Y the space of output values. In this example, X = Y = R*.

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function *h : X → Y so that h(x) is a “good” predictor for the corresponding value of y*. For historical reasons, this




function h is called a hypothesis. Seen pictorially, the process is therefore like this:

| Training set              |                     |                      |
| ------------------------- | ------------------- | -------------------- |
| Learning algorithm        |                     |                      |
| x (living area of house.) | h (predicted price) | predicted y of house |

When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.



# Chapter 1

# Linear regression

To make our housing example more interesting, let’s consider a slightly richer dataset in which we also know the number of bedrooms in each house:

| Living area (feet²) | #bedrooms | Price (1000$s) |
| ------------------- | --------- | -------------- |
| 2104                | 3         | 400            |
| 1600                | 3         | 330            |
| 2400                | 3         | 369            |
| 1416                | 2         | 232            |
| 3000                | 4         | 540            |
| .                   | .         | .              |
| .                   | .         | .              |
| .                   | .         | .              |

Here, the x’s are two-dimensional vectors in R². For instance, x(i) is the living area of the i-th house in the training set, and x(i) is its number of bedrooms. (In general, when designing a learning problem, it will be up to you to decide what features to choose, so if you are out in Portland gathering housing data, you might also decide to include other features such as whether each house has a fireplace, the number of bathrooms, and so on. We’ll say more about feature selection later, but for now let’s take the features as given.)

To perform supervised learning, we must decide how we’re going to represent functions/hypotheses h in a computer. As an initial choice, let’s say we decide to approximate y as a linear function of x:

hθ(x) = θ0 + θ1x1 + θ2x2

Here, the θi’s are the parameters (also called weights) parameterizing the space of linear functions mapping from X to Y. When there is no risk of





confusion, we will drop the θ subscript in hθ(x), and write it more simply as h(x). To simplify our notation, we also introduce the convention of letting x₀ = 1 (this is the intercept term), so that

∑

d

h(x) = θixi = θᵀ x,

i=0

where on the right-hand side above we are viewing θ and x both as vectors, and here d is the number of input variables (not counting x₀).

Now, given a training set, how do we pick, or learn, the parameters θ? One reasonable method seems to be to make h(x) close to y, at least for the training examples we have. To formalize this, we will define a function that measures, for each value of the θ’s, how close the h(x(i))’s are to the corresponding y(i)’s. We define the cost function:

∑

J (θ) = 1/n (hθ(x(i)) − y(i))².

2 i=1

If you’ve seen linear regression before, you may recognize this as the familiar least-squares cost function that gives rise to the ordinary least squares regression model. Whether or not you have seen it previously, let’s keep going, and we’ll eventually show this to be a special case of a much broader family of algorithms.

# 1.1 LMS algorithm

We want to choose θ so as to minimize J (θ). To do so, let’s use a search algorithm that starts with some “initial guess” for θ, and that repeatedly changes θ to make J (θ) smaller, until hopefully we converge to a value of θ that minimizes J (θ). Specifically, let’s consider the gradient descent algorithm, which starts with some initial θ, and repeatedly performs the update:

θj := θj − α ∂ J (θ).

∂θj

(This update is simultaneously performed for all values of j = 0, . . . , d.) Here, α is called the learning rate. This is a very natural algorithm that repeatedly takes a step in the direction of steepest decrease of J.

In order to implement this algorithm, we have to work out what is the partial derivative term on the right hand side. Let’s first work it out for the





case of if we have only one training example (x, y), so that we can neglect the sum in the definition of J. We have:

∂ J (θ) = ∂ 1 (hθ(x) − y)2

∂θj          ∂θj 2

= 2 · 1 (hθ(x) − y) · ∂ (hθ(x) − y)

2          ∂θj

= (hθ(x) − y) · ∂          d θixi − y

∂θj          i=0

= (hθ(x) − y) xj

For a single training example, this gives the update rule:¹

θj := θj + α (y(i) − hθ(x(i))) x(i).

j

The rule is called the LMS update rule (LMS stands for “least mean squares”), and is also known as the Widrow-Hoff learning rule. This rule has several properties that seem natural and intuitive. For instance, the magnitude of the update is proportional to the error term (y(i) − hθ(x(i))); thus, for instance, if we are encountering a training example on which our prediction nearly matches the actual value of y(i), then we find that there is little need to change the parameters; in contrast, a larger change to the parameters will be made if our prediction hθ(x(i)) has a large error (i.e., if it is very far from y(i)).

We’d derived the LMS rule for when there was only a single training example. There are two ways to modify this method for a training set of more than one example. The first is to replace it with the following algorithm:

Repeat until convergence {
θj := θj + α ∑i=1n (y(i) − hθ(x(i))) x(i), (for every j) (1.1)
}

¹We use the notation “a := b” to denote an operation (in a computer program) in which we set the value of a variable a to be equal to the value of b. In other words, this operation overwrites a with the value of b. In contrast, we will write “a = b” when we are asserting a statement of fact, that the value of a is equal to the value of b.





By grouping the updates of the coordinates into an update of the vector θ, we can rewrite update (1.1) in a slightly more succinct way:

θ := θ + αn ∑i=1 (y(i) − hθ(x(i))) x(i)

The reader can easily verify that the quantity in the summation in the update rule above is just ∂J (θ)/∂θj (for the original definition of J). So, this is simply gradient descent on the original cost function J. This method looks at every example in the entire training set on every step, and is called batch gradient descent. Note that, while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus gradient descent always converges (assuming the learning rate α is not too large) to the global minimum. Indeed, J is a convex quadratic function.

Here is an example of gradient descent as it is run to minimize a quadratic function.

50
45
40
35
30
25
20
15
10
5

5  10  15   20  25  30  35   40  45   50

The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at (48,30). The x’s in the figure (joined by straight lines) mark the successive values of θ that gradient descent went through.

When we run batch gradient descent to fit θ on our previous dataset, to learn to predict housing price as a function of living area, we obtain θ0 = 71.27, θ1 = 0.1345. If we plot hθ(x) as a function of x (area), along with the training data, we obtain the following figure:





# housing prices

1000
900
800
700
600
price (in $1000)
500
400
300
200
100
0
| 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 |
| --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |

If the number of bedrooms were included as one of the input features as well, we get θ₀ = 89.60, θ₁ = 0.1392, θ₂ = −8.738. The above results were obtained with batch gradient descent. There is an alternative to batch gradient descent that also works very well. Consider the following algorithm:

Loop {
for i = 1 to n, {
θj := θj + α (y(i) − hθ(x(i))) x(i), (for every j)  (1.2)
}
}

By grouping the updates of the coordinates into an update of the vector θ, we can rewrite update (1.2) in a slightly more succinct way:

θ := θ + α (y(i) − hθ(x(i))) x(i)

In this algorithm, we repeatedly run through the training set, and each time we encounter a training example, we update the parameters according to the gradient of the error with respect to that single training example only. This algorithm is called stochastic gradient descent (also incremental gradient descent). Whereas batch gradient descent has to scan through the entire training set before taking a single step—a costly operation if n is large—stochastic gradient descent can start making progress right away, and






continues to make progress with each example it looks at. Often, stochastic gradient descent gets θ “close” to the minimum much faster than batch gradient descent. (Note however that it may never “converge” to the minimum, and the parameters θ will keep oscillating around the minimum of J (θ); but in practice most of the values near the minimum will be reasonably good approximations to the true minimum.²) For these reasons, particularly when the training set is large, stochastic gradient descent is often preferred over batch gradient descent.

# 1.2 The normal equations

Gradient descent gives one way of minimizing J. Let’s discuss a second way of doing so, this time performing the minimization explicitly and without resorting to an iterative algorithm. In this method, we will minimize J by explicitly taking its derivatives with respect to the θj ’s, and setting them to zero. To enable us to do this without having to write reams of algebra and pages full of matrices of derivatives, let’s introduce some notation for doing calculus with matrices.

# 1.2.1 Matrix derivatives

For a function f :  Rⁿ×d → R mapping from n-by-d matrices to the real numbers, we define the derivative of f with respect to A to be:

∇ f (A) =

| ∂f   | · · · | ∂f   |
| ---- | ----- | ---- |
| ∂A¹¹ |       | ∂A1d |
| .    | .     | .    |
| .    | .     | .    |
| .    | .     | .    |
| ∂f   | · · · | ∂f   |
| ∂An1 |       | ∂And |

Thus, the gradient ∇ f (A) is itself an n-by-d matrix, whose (i, j)-element is ∂f /∂A . For example, suppose A = [ A11 A12 ]
[ A21 A22 ] is a 2-by-2 matrix, and the function f : R²×2 → R is given by

f (A) = 3 A₁₁ + 5A² + A₂₁A₂₂.

2By slowly letting the learning rate α decrease to zero as the algorithm runs, it is also possible to ensure that the parameters will converge to the global minimum rather than merely oscillate around the minimum.





# 14

Here, Aij denotes the (i, j) entry of the matrix A. We then have

∇ f (A) = [ 3 10A₁₂ ] .

A 2

A₂₂ A₂₁

# 1.2.2 Least squares revisited

Armed with the tools of matrix derivatives, let us now proceed to find in closed-form the value of θ that minimizes J (θ). We begin by re-writing J in matrix-vectorial notation.

Given a training set, define the design matrix X to be the n-by-d matrix (actually n-by-d + 1, if we include the intercept term) that contains the training examples’ input values in its rows:

X =



# 15

Finally, to minimize J, let’s find its derivatives with respect to θ. Hence,

∇ J (θ) = ∇1 (Xθ − ~Ty) (Xθ − ~y)

= 1/2 (∇θ (Xθ) Xθ − (Xθ) ~y − ~y (Xθ) + ~y ~y)

= 1/2 ∇θ (θT (XT X)θ − ~yT (Xθ))

= 1/2 (XT Xθ − XT ~y)

In the third step, we used the fact that aT b = bT a, and in the fifth step used the facts ∇x bT x = b and ∇x xT Ax = 2Ax for symmetric matrix A (for more details, see Section 4.3 of “Linear Algebra Review and Reference”). To minimize J, we set its derivatives to zero, and obtain the normal equations:

XT Xθ = XT ~y

Thus, the value of θ that minimizes J(θ) is given in closed form by the equation

θ = (XT X)−1XT ~y.

# 1.3 Probabilistic interpretation

When faced with a regression problem, why might linear regression, and specifically why might the least-squares cost function J, be a reasonable choice? In this section, we will give a set of probabilistic assumptions, under which least-squares regression is derived as a very natural algorithm.

Let us assume that the target variables and the inputs are related via the equation

y(i) = θT x(i) + (i),

Note that in the above step, we are implicitly assuming that XT X is an invertible matrix. This can be checked before calculating the inverse. If either the number of linearly independent examples is fewer than the number of features, or if the features are not linearly independent, then XT X will not be invertible. Even in such cases, it is possible to “fix” the situation with additional techniques, which we skip here for the sake of simplicity.





# 16

where (i) is an error term that captures either unmodeled effects (such as if there are some features very pertinent to predicting housing price, but that we’d left out of the regression), or random noise. Let us further assume that the (i) are distributed IID (independently and identically distributed) according to a Gaussian distribution (also called a Normal distribution) with mean zero and some variance σ². We can write this assumption as “(i) ∼ N(0, σ²).” I.e., the density of (i) is given by

p((i)) = ~~√~~ 1 exp(−((i))²).

2πσ 2σ²

This implies that

p(y(i)|x(i); θ) = ~~√~~ 1 exp(−(y(i) − θᵀ x(i))²).

2πσ 2σ²

The notation “p(y(i)|x(i); θ)” indicates that this is the distribution of y(i) given x(i) and parameterized by θ. Note that we should not condition on θ (“p(y(i)|x(i), θ)”), since θ is not a random variable. We can also write the distribution of y(i) as y(i) | x(i); θ ∼ N(θᵀ x(i), σ²).

Given X (the design matrix, which contains all the x(i)’s) and θ, what is the distribution of the y(i)’s? The probability of the data is given by p(~y|X; θ). This quantity is typically viewed as a function of ~y (and perhaps X), for a fixed value of θ. When we wish to explicitly view this as a function of θ, we will instead call it the likelihood function:

L(θ) = L(θ; X, ~y) = p(~y|X; θ).

Note that by the independence assumption on the (i)’s (and hence also the y(i)’s given the x(i)’s), this can also be written

∏n L(θ) = p(y(i) | x(i); θ)

i=1

∏ = n ~~√~~ 1 exp(−(y(i) − θᵀ x(i))²).

i=1 2πσ 2σ²

Now, given this probabilistic model relating the y(i)’s and the x(i)’s, what is a reasonable way of choosing our best guess of the parameters θ? The principle of maximum likelihood says that we should choose θ so as to make the data as high probability as possible. I.e., we should choose θ to maximize L(θ).





Instead of maximizing L(θ), we can also maximize any strictly increasing function of L(θ). In particular, the derivations will be a bit simpler if we instead maximize the log likelihood `(θ):

`(θ)  =  log L(θ)
∏
=  log   n √1  exp        − (y(i) − θᵀ x(i))²
∑i=1           2πσ      (           2σ²           )
=  n    log       √1  exp − (y(i) − θᵀ x(i))²
i=1              2πσ                2σ²
∑
=  n log √1           − 1 · 1 n (y(i) − θᵀ x(i))².
2πσ         σ²  2 i=1

Hence, maximizing `(θ) gives the same answer as minimizing

∑
1        n (y(i) − θᵀ x(i))²,
2 i=1

which we recognize to be J (θ), our original least-squares cost function. To summarize: Under the previous probabilistic assumptions on the data, least-squares regression corresponds to finding the maximum likelihood estimate of θ. This is thus one set of assumptions under which least-squares regression can be justified as a very natural method that’s just doing maximum likelihood estimation. (Note however that the probabilistic assumptions are by no means necessary for least-squares to be a perfectly good and rational procedure, and there may—and indeed there are—other natural assumptions that can also be used to justify it.)

Note also that, in our previous discussion, our final choice of θ did not depend on what was σ², and indeed we’d have arrived at the same result even if σ² were unknown. We will use this fact again later, when we talk about the exponential family and generalized linear models.

# 1.4 Locally weighted linear regression (optional reading)

Consider the problem of predicting y from x ∈ R. The leftmost figure below shows the result of fitting a y = θ₀ + θ₁x to a dataset. We see that the data doesn’t really lie on straight line, and so the fit is not very good.





# 4.5

| 4.5 | 4.5 | 4.5 |
| --- | --- | --- |
| 4   | 4   | 4   |
| 3.5 | 3.5 | 3.5 |
| 3   | 3   | 3   |
| 2.5 | 2.5 | 2.5 |
| y   | y   | y   |
| 2   | 2   | 2   |
| 1.5 | 1.5 | 1.5 |
| 1   | 1   | 1   |
| 0.5 | 0.5 | 0.5 |
| 0   | 0   | 0   |

0    1    2    3    4    5    6    7        0    1    2    3    4    5    6    7    0    1    2    3    4    5    6    7

x                                           x                                       x

Instead, if we had added an extra feature x², and fit y = θ₀ + θ₁x + θ₂x², then we obtain a slightly better fit to the data. (See middle figure) Naively, it might seem that the more features we add, the better. However, there is also a danger in adding too many features: The rightmost figure is the result of fitting a 5-th order polynomial y = ∑⁵ θj xj. We see that even though the fitted curve passes through the data perfectly, we would not expect this to be a very good predictor of, say, housing prices (y) for different living areas (x). Without formally defining what these terms mean, we’ll say the figure on the left shows an instance of underfitting—in which the data clearly shows structure not captured by the model—and the figure on the right is an example of overfitting. (Later in this class, when we talk about learning theory we’ll formalize some of these notions, and also define more carefully just what it means for a hypothesis to be good or bad.)

As discussed previously, and as shown in the example above, the choice of features is important to ensuring good performance of a learning algorithm. (When we talk about model selection, we’ll also see algorithms for automatically choosing a good set of features.) In this section, let us briefly talk about the locally weighted linear regression (LWR) algorithm which, assuming there is sufficient training data, makes the choice of features less critical. This treatment will be brief, since you’ll get a chance to explore some of the properties of the LWR algorithm yourself in the homework.

In the original linear regression algorithm, to make a prediction at a query point x (i.e., to evaluate h(x)), we would:

1. Fit θ to minimize ∑i(y(i) − θᵀ x(i))².
2. Output θᵀ x.

In contrast, the locally weighted linear regression algorithm does the following:

1. Fit θ to minimize ∑i w(i)(y(i) − θᵀ x(i))².
2. Output θᵀ x.





# 19

Here, the w(i)’s are non-negative valued weights. Intuitively, if w(i) is large for a particular value of i, then in picking θ, we’ll try hard to make (y(i) − θᵀ x(i))² small. If w(i) is small, then the (y(i) − θᵀ x(i))² error term will be pretty much ignored in the fit.

A fairly standard choice for the weights is4

w(i) = exp (− (x(i) − x)² )

2τ²

Note that the weights depend on the particular point x at which we’re trying to evaluate x. Moreover, if |x(i) − x| is small, then w(i) is close to 1; and if |x(i) − x| is large, then w(i) is small. Hence, θ is chosen giving a much higher “weight” to the (errors on) training examples close to the query point x. (Note also that while the formula for the weights takes a form that is cosmetically similar to the density of a Gaussian distribution, the w(i)’s do not directly have anything to do with Gaussians, and in particular the w(i) are not random variables, normally distributed or otherwise.) The parameter τ controls how quickly the weight of a training example falls off with distance of its x(i) from the query point x; τ is called the bandwidth parameter, and is also something that you’ll get to experiment with in your homework.

Locally weighted linear regression is the first example we’re seeing of a non-parametric algorithm. The (unweighted) linear regression algorithm that we saw earlier is known as a parametric learning algorithm, because it has a fixed, finite number of parameters (the θi’s), which are fit to the data. Once we’ve fit the θi’s and stored them away, we no longer need to keep the training data around to make future predictions. In contrast, to make predictions using locally weighted linear regression, we need to keep the entire training set around. The term “non-parametric” (roughly) refers to the fact that the amount of stuff we need to keep in order to represent the hypothesis h grows linearly with the size of the training set.

4If x is vector-valued, this is generalized to be w(i) = exp(−(x(i) −x)T (x(i) −x)/(2τ²)), or w(i) = exp(−(x(i) − x)T Σ−1(x(i) − x)/(2τ²)), for an appropriate choice of τ or Σ.





# Chapter 2

# Classification and logistic regression

Let’s now talk about the classification problem. This is just like the regression problem, except that the values y we now want to predict take on only a small number of discrete values. For now, we will focus on the binary classification problem in which y can take on only two values, 0 and 1. (Most of what we say here will also generalize to the multiple-class case.) For instance, if we are trying to build a spam classifier for email, then x(i) may be some features of a piece of email, and y may be 1 if it is a piece of spam mail, and 0 otherwise. 0 is also called the negative class, and 1 the positive class, and they are sometimes also denoted by the symbols “-” and “+.” Given x(i), the corresponding y(i) is also called the label for the training example.

# 2.1 Logistic regression

We could approach the classification problem ignoring the fact that y is discrete-valued, and use our old linear regression algorithm to try to predict y given x. However, it is easy to construct examples where this method performs very poorly. Intuitively, it also doesn’t make sense for hθ(x) to take values larger than 1 or smaller than 0 when we know that y ∈ {0, 1}. To fix this, let’s change the form for our hypotheses hθ(x). We will choose

hθ(x) = g(θᵀ x) = 1 / (1 + e-θᵀ x)

g(z) = 1 / (1 + e-z)





# 21

is called the logistic function or the sigmoid function. Here is a plot showing g(z):

|                            | 1 | 0.9 | 0.8 | 0.7 | 0.6 | g(z) | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | 0 |
| -------------------------- | - | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | - |
| −5 −4 −3 −2 −1 0 1 2 3 4 5 |   |     |     |     |     |      |     |     |     |     |     |   |

Notice that g(z) tends towards 1 as z → ∞, and g(z) tends towards 0 as z → −∞. Moreover, g(z), and hence also h(x), is always bounded between 0 and 1. As before, we are keeping the convention of letting x = 1, so that θᵀ x = θ₀ + ∑ᵈ θj xj.

For now, let’s take the choice of g as given. Other functions that smoothly increase from 0 to 1 can also be used, but for a couple of reasons that we’ll see later (when we talk about GLMs, and when we talk about generative learning algorithms), the choice of the logistic function is a fairly natural one. Before moving on, here’s a useful property of the derivative of the sigmoid function, which we write as g′:

g′(z) = d/dz (1 + e−z) = (1 + 1/e−z) = (1/e−z)²· (1 − 1/(1 + e−z)) = g(z)(1 − g(z)).

So, given the logistic regression model, how do we fit θ for it? Following how we saw least squares regression could be derived as the maximum likelihood estimator under a set of assumptions, let’s endow our classification model with a set of probabilistic assumptions, and then fit the parameters via maximum likelihood.





# 22

Let us assume that

P (y = 1 | x; θ) = hθ(x)

P (y = 0 | x; θ) = 1 − hθ(x)

Note that this can be written more compactly as

p(y | x; θ) = (hθ(x))y (1 − hθ(x))1−y

Assuming that the n training examples were generated independently, we can then write down the likelihood of the parameters as

L(θ) = p(~y | X; θ) = ∏n p(y(i) | x(i); θ) = ∏i=1n (hθ(x(i)))y(i) (1 − hθ(x(i)))1−y(i)

As before, it will be easier to maximize the log likelihood:

` (θ) = log L(θ) = ∑i=1n [y(i) log h(x(i)) + (1 − y(i)) log(1 − h(x(i))]  (2.1)

How do we maximize the likelihood? Similar to our derivation in the case of linear regression, we can use gradient ascent. Written in vectorial notation, our updates will therefore be given by θ := θ + α∇θ`(θ). (Note the positive rather than negative sign in the update formula, since we’re maximizing, rather than minimizing, a function now.)

Let’s start by working with just one training example (x, y), and take derivatives to derive the stochastic gradient ascent rule:

∂`(θ) = (y 1 − (1 − y) 1) ∂g(θTx)

∂θj = (g(θTx)(1 − g(θTx))) ∂θTx

= y g(θTx) − (1 − y)1 − g(θTx)(1 − g(θTx)) ∂θTx

= y(1 − g(θTx)) − (1 − y)g(θTx) xj

= (y − hθ(x)) xj                                                      (2.2)





Above, we used the fact that g′(z) = g(z)(1 − g(z)). This therefore gives us the stochastic gradient ascent rule

θj := θj + α (y(i) − hθ(x(i))) xj(i)

If we compare this to the LMS update rule, we see that it looks identical; but this is not the same algorithm, because hθ(x(i)) is now defined as a non-linear function of θᵀ x(i). Nonetheless, it’s a little surprising that we end up with the same update rule for a rather different algorithm and learning problem. Is this coincidence, or is there a deeper reason behind this? We’ll answer this when we get to GLM models.

Remark 2.1.1: An alternative notational viewpoint of the same loss function is also useful, especially for Section 7.1 where we study nonlinear models. Let `logistic : R × {0, 1} → R≥0 be the logistic loss defined as

`logistic(t, y) , y log(1 + exp(−t)) + (1 − y) log(1 + exp(t)).

One can verify by plugging in hθ(x) = 1/(1 + e−θ>ˣ) that the negative log-likelihood (the negation of `(θ) in equation (2.1)) can be re-written as

−`(θ) = `logistic(θ>x, y).

Oftentimes θ>x or t is called the logit. Basic calculus gives us that

∂`logistic(t, y)  = y − exp(−t) + (1 − y) 1/(1 + exp(−t))

Then, using the chain rule, we have that

∂`(θ) = − ∂`logistic(t, y) · ∂t

∂θj = (y − 1/(1 + exp(−t))) · xj = (y − hθ(x))xj,

which is consistent with the derivation in equation (2.2). We will see this viewpoint can be extended to nonlinear models in Section 7.1.

# 2.2 Digression: the perceptron learning algorithm

We now digress to talk briefly about an algorithm that’s of some historical interest, and that we will also return to later when we talk about learning





theory. Consider modifying the logistic regression method to “force” it to output values that are either 0 or 1 or exactly. To do so, it seems natural to change the definition of g to be the threshold function:

g(z) = { 1  if  z ≥ 0

0  if  z &#x3C; 0

If we then let hθ(x) = g(θᵀ x) as before but using this modified definition of g, and if we use the update rule

θj := θj + α (y(i) − hθ(x(i))) x(i).

j

then we have the perceptron learning algorithm. In the 1960s, this “perceptron” was argued to be a rough model for how individual neurons in the brain work. Given how simple the algorithm is, it will also provide a starting point for our analysis when we talk about learning theory later in this class. Note however that even though the perceptron may be cosmetically similar to the other algorithms we talked about, it is actually a very different type of algorithm than logistic regression and least squares linear regression; in particular, it is difficult to endow the perceptron’s predictions with meaningful probabilistic interpretations, or derive the perceptron as a maximum likelihood estimation algorithm.

# 2.3 Multi-class classification

Consider a classification problem in which the response variable y can take on any one of k values, so y ∈ {1, 2, . . . , k}. For example, rather than classifying emails into the two classes spam or not-spam—which would have been a binary classification problem—we might want to classify them into three classes, such as spam, personal mails, and work-related mails. The label / response variable is still discrete, but can now take on more than two values. We will thus model it as distributed according to a multinomial distribution.

In this case, p(y | x; θ) is a distribution over k possible discrete outcomes and is thus a multinomial distribution. Recall that a multinomial distribution involves k numbers φ1, . . . , φk specifying the probability of each of the outcomes. Note that these numbers must satisfy ∑i=1k φi = 1. We will define a parameterized model that outputs φ1, . . . , φk satisfying this constraint given the input x.

We introduce k groups of parameters θ1, . . . , θk, each of them being a vector in Rd. Intuitively, we would like to use θ1ᵀx, . . . , θkᵀx to represent



φ₁, . . . , φₖ , the probabilities P (y = 1 | x; θ), . . . , P (y = k | x; θ). However, there are two issues with such a direct approach. First, θ>x is not necessarily within [0, 1]. Second, the summation of θ>x’s is not necessarily 1. Thus, instead, we will use the softmax function to turn (θ>x, · · · , θ>x) into a probability vector with nonnegative entries that sum up to 1.

Define the softmax function softmax : Rᵏ → Rᵏ as

softmax(t₁, . . . , tₖ) =

∑ₖ exp(tj)

∑ₖ exp(tj)

The inputs to the softmax function, the vector t here, are often called logits. Note that by definition, the output of the softmax function is always a probability vector whose entries are nonnegative and sum up to 1.

Let (t₁, . . . , tₖ) = (θ>x, · · · , θ>x). We apply the softmax function to (t₁, . . . , tₖ), and use the output as the probabilities P (y = 1 | x; θ), . . . , P (y = k | x; θ). We obtain the following probabilistic model:

P (y = 1 | x; θ) = softmax(t₁, · · · , tₖ) =

∑ₖ exp(θjx)

For notational convenience, we will let φi = ∑ₖ exp(ti). More succinctly, the equation above can be written as:

P (y = i | x; θ) = φi =

∑ₖ exp(ti) =
∑ₖ exp(θ>x)

Next, we compute the negative log-likelihood of a single example (x, y).

− log p(y | x, θ) = − log ∑ₖ exp(tj) = − log ∑ₖ exp(θ>x)

Thus, the loss function, the negative log-likelihood of the training data, is given as

`(θ) = n − log ∑ exp(θy(i)x(i))





# 26

It’s convenient to define the cross-entropy loss ce : Rᵏ × {1, . . . , k} → R≥0, which modularizes in the complex equation above:¹

`ce((t₁, . . . , tₖ ), y) = − log ~~∑~~exp(ty) .  (2.14)

With this notation, we can simply rewrite equation (2.13) as

∑i=1n `(θ) = ~~∑~~ `ce((θ>x(i), . . . , θ>x(i)), y(i)) .  (2.15)

Moreover, conveniently, the cross-entropy loss also has a simple gradient. Let t = (t₁, . . . , tₖ ), and recall φi = ~~∑~~j=1kexp(ti). By basic calculus, we can derive

∂`ce(t, y) = φi − 1{y = i} , (2.16)

where 1{·} is the indicator function, that is, 1{y = i} = 1 if y = i, and 1{y = i} = 0 if y ≠ i. Alternatively, in vectorized notations, we have the following form which will be useful for Chapter 7:

∂`ce(t, y) = φ − ey , (2.17)

where es ∈ Rᵏ is the s-th natural basis vector (where the s-th entry is 1 and all other entries are zeros.) Using Chain rule, we have that

∂`ce((θ>x, . . . , θ>x), y) = ∂`(t, y) ∂ti 1 ∂θi k = ∂ti · ∂θi = (φi − 1{y = i}) · x . (2.18)

Therefore, the gradient of the loss with respect to the part of parameter θi is

∂`(θ) = ~~∑~~j=1n (φ(j) − 1{y(j) = i}) · x(j) , (2.19)

where φ(j) = exp(θ>x(j)) is the probability that the model predicts item i for example x(j). With the gradients above, one can implement (stochastic) gradient descent to minimize the loss function `(θ).

¹There are some ambiguity in the naming here. Some people call the cross-entropy loss the function that maps the probability vector (the φ in our language) and label y to the final real number, and call our version of cross-entropy loss softmax-cross-entropy loss. We choose our current naming convention because it’s consistent with the naming of most modern deep learning library such as PyTorch and Jax.





# 2.4 Another algorithm for maximizing `(θ)

Returning to logistic regression with g(z) being the sigmoid function, let’s now talk about a different algorithm for maximizing `(θ).

To get us started, let’s consider Newton’s method for finding a zero of a function. Specifically, suppose we have some function f : R → R, and we wish to find a value of θ so that f (θ) = 0. Here, θ ∈ R is a real number.

Newton’s method performs the following update:

θ := θ − f (θ) / f ′(θ)

This method has a natural interpretation in which we can think of it as approximating the function f via a linear function that is tangent to f at the current guess θ, solving for where that linear function equals to zero, and letting the next guess for θ be where that linear function is zero.

Here’s a picture of the Newton’s method in action:

In the leftmost figure, we see the function f plotted along with the line y = 0. We’re trying to find θ so that f (θ) = 0; the value of θ that achieves this is about 1.3. Suppose we initialized the algorithm with θ = 4.5. Newton’s method then fits a straight line tangent to f at θ = 4.5, and solves for the where that line evaluates to 0. (Middle figure.) This gives us the next guess for θ, which is about 2.8. The rightmost figure shows the result of running one more iteration, which updates θ to about 1.8. After a few more iterations, we rapidly approach θ = 1.3.

Newton’s method gives a way of getting to f (θ) = 0. What if we want to use it to maximize some function `? The maxima of ` correspond to points where its first derivative `′(θ) is zero. So, by letting f (θ) = `′(θ), we can use the same algorithm to maximize `, and we obtain update rule:

θ := θ − `′(θ) / `′′(θ)

(Something to think about: How would this change if we wanted to use Newton’s method to minimize rather than maximize a function?)


Lastly, in our logistic regression setting, θ is vector-valued, so we need to generalize Newton’s method to this setting. The generalization of Newton’s method to this multidimensional setting (also called the Newton-Raphson method) is given by

θ := θ − H−1∇θ`(θ).

Here, ∇θ`(θ) is, as usual, the vector of partial derivatives of `(θ) with respect to the θi’s; and H is an d-by-d matrix (actually, d+1−by−d+1, assuming that we include the intercept term) called the Hessian, whose entries are given by

Hij = ∂²`(θ) / ∂θi∂θj

Newton’s method typically enjoys faster convergence than (batch) gradient descent, and requires many fewer iterations to get very close to the minimum. One iteration of Newton’s can, however, be more expensive than one iteration of gradient descent, since it requires finding and inverting an d-by-d Hessian; but so long as d is not too large, it is usually much faster overall. When Newton’s method is applied to maximize the logistic regression log likelihood function `(θ), the resulting method is also called Fisher scoring.


Chapter 3

# Generalized linear models

So far, we’ve seen a regression example, and a classification example. In the regression example, we had y|x; θ ∼ N (μ, σ²), and in the classification one, y|x; θ ∼ Bernoulli(φ), for some appropriate definitions of μ and φ as functions of x and θ. In this section, we will show that both of these methods are special cases of a broader family of models, called Generalized Linear Models (GLMs).¹ We will also show how other models in the GLM family can be derived and applied to other classification and regression problems.

# 3.1 The exponential family

To work our way up to GLMs, we will begin by defining exponential family distributions. We say that a class of distributions is in the exponential family if it can be written in the form

p(y; η) = b(y) exp(ηᵀ T (y) − a(η))                     (3.1)

Here, η is called the natural parameter (also called the canonical parameter) of the distribution; T (y) is the sufficient statistic (for the distributions we consider, it will often be the case that T (y) = y); and a(η) is the log partition function. The quantity e−a(η) essentially plays the role of a normalization constant, that makes sure the distribution p(y; η) sums/integrates over y to 1.

A fixed choice of T, a and b defines a family (or set) of distributions that is parameterized by η; as we vary η, we then get different distributions within this family.

¹The presentation of the material in this section takes inspiration from Michael I. Jordan, Learning in graphical models (unpublished book draft), and also McCullagh and Nelder, Generalized Linear Models (2nd ed.).

29



We now show that the Bernoulli and the Gaussian distributions are examples of exponential family distributions. The Bernoulli distribution with mean φ, written Bernoulli(φ), specifies a distribution over y ∈ {0, 1}, so that

p(y = 1; φ) = φ; p(y = 0; φ) = 1 − φ. As we vary φ, we obtain Bernoulli distributions with different means. We now show that this class of Bernoulli distributions, ones obtained by varying φ, is in the exponential family; i.e., that there is a choice of T, a and b so that Equation (3.1) becomes exactly the class of Bernoulli distributions.

We write the Bernoulli distribution as:

p(y; φ)      =  φʸ(1 − φ)¹−y
=  exp(y log φ + (1 − y) log(1 − φ))
=  exp ((log (1 φ     )) y + log(1 − φ)).
− φ

Thus, the natural parameter is given by η = log(φ/(1 − φ)). Interestingly, if we invert this definition for η by solving for φ in terms of η, we obtain φ = 1/(1 + e−η). This is the familiar sigmoid function! This will come up again when we derive logistic regression as a GLM. To complete the formulation of the Bernoulli distribution as an exponential family distribution, we also have

T (y)  =   y
a(η)  =   − log(1 − φ)
=   log(1 + eη)
b(y)  =   1

This shows that the Bernoulli distribution can be written in the form of Equation (3.1), using an appropriate choice of T, a and b.

Let’s now move on to consider the Gaussian distribution. Recall that, when deriving linear regression, the value of σ² had no effect on our final choice of θ and hθ(x). Thus, we can choose an arbitrary value for σ² without changing anything. To simplify the derivation below, let’s set σ² = 1.2

2 If we leave σ2 as a variable, the Gaussian distribution can also be shown to be in the exponential family, where η ∈ R2 is now a 2-dimension vector that depends on both μ and σ. For the purposes of GLMs, however, the σ2 parameter can also be treated by considering a more general definition of the exponential family: p(y; η, τ) = b(a, τ) exp((ηT T (y) − a(η))/c(τ)). Here, τ is called the dispersion parameter, and for the Gaussian, c(τ) = σ2; but given our simplification above, we won’t need the more general definition for the examples we will consider here.





then have: *p(y; μ) = √(1 / (2π)) exp(−(1/2)(y − μ)²)*

= *√(1 / (2π)) exp(−(1/2)y²) · exp(μy − (1/2)μ²)*

Thus, we see that the Gaussian is in the exponential family, with

- η = μ
- T(y) = y
- a(η) = μ²/2 = η²/2
- b(y) = (1/√(2π)) exp(−y²/2).

There’re many other distributions that are members of the exponential family: The multinomial (which we’ll see later), the Poisson (for modelling count-data; also see the problem set); the gamma and the exponential (for modelling continuous, non-negative random variables, such as time-intervals); the beta and the Dirichlet (for distributions over probabilities); and many more. In the next section, we will describe a general “recipe” for constructing models in which y (given x and θ) comes from any of these distributions.

# 3.2 Constructing GLMs

Suppose you would like to build a model to estimate the number y of customers arriving in your store (or number of page-views on your website) in any given hour, based on certain features x such as store promotions, recent advertising, weather, day-of-week, etc. We know that the Poisson distribution usually gives a good model for numbers of visitors. Knowing this, how can we come up with a model for our problem? Fortunately, the Poisson is an exponential family distribution, so we can apply a Generalized Linear Model (GLM). In this section, we will describe a method for constructing GLM models for problems such as these.

More generally, consider a classification or regression problem where we would like to predict the value of some random variable y as a function of x. To derive a GLM for this problem, we will make the following three assumptions about the conditional distribution of y given x and about our model:





# 1. y | x; θ ∼ ExponentialFamily(η)

I.e., given x and θ, the distribution of y follows some exponential family distribution, with parameter η.

# 2. Given x, our goal is to predict the expected value of T (y) given x.

In most of our examples, we will have T (y) = y, so this means we would like the prediction h(x) output by our learned hypothesis h to satisfy h(x) = E[y|x]. (Note that this assumption is satisfied in the choices for hθ(x) for both logistic regression and linear regression. For instance, in logistic regression, we had hθ(x) = p(y = 1|x; θ) = 0 · p(y = 0|x; θ) + 1 · p(y = 1|x; θ) = E[y|x; θ].)

# 3. The natural parameter η and the inputs x are related linearly:

η = θᵀ x. (Or, if η is vector-valued, then ηi = θᵀ x.)

The third of these assumptions might seem the least well justified of the above, and it might be better thought of as a “design choice” in our recipe for designing GLMs, rather than as an assumption per se. These three assumptions/design choices will allow us to derive a very elegant class of learning algorithms, namely GLMs, that have many desirable properties such as ease of learning. Furthermore, the resulting models are often very effective for modelling different types of distributions over y; for example, we will shortly show that both logistic regression and ordinary least squares can both be derived as GLMs.

# 3.2.1 Ordinary least squares

To show that ordinary least squares is a special case of the GLM family of models, consider the setting where the target variable y (also called the response variable in GLM terminology) is continuous, and we model the conditional distribution of y given x as a Gaussian N (μ, σ²). (Here, μ may depend on x.) So, we let the ExponentialFamily(η) distribution above be the Gaussian distribution. As we saw previously, in the formulation of the Gaussian as an exponential family distribution, we had μ = η. So, we have

hθ(x) = E[y|x; θ] = μ = η = θᵀ x.

The first equality follows from Assumption 2, above; the second equality follows from the fact that y|x; θ ∼ N (μ, σ²), and so its expected value is given




# 3.2.2 Logistic regression

We now consider logistic regression. Here we are interested in binary classification, so y ∈ {0, 1}. Given that y is binary-valued, it therefore seems natural to choose the Bernoulli family of distributions to model the conditional distribution of y given x. In our formulation of the Bernoulli distribution as an exponential family distribution, we had φ = 1/(1 + e−η). Furthermore, note that if y|x; θ ∼ Bernoulli(φ), then E[y|x; θ] = φ. So, following a similar derivation as the one for ordinary least squares, we get:

hθ(x) = E[y|x; θ] = φ = 1/(1 + e−η) = 1/(1 + e−θᵀ x)

So, this gives us hypothesis functions of the form hθ(x) = 1/(1 + e−θᵀ x). If you are previously wondering how we came up with the form of the logistic function 1/(1 + e−z), this gives one answer: Once we assume that y conditioned on x is Bernoulli, it arises as a consequence of the definition of GLMs and exponential family distributions.

To introduce a little more terminology, the function g giving the distribution’s mean as a function of the natural parameter (g(η) = E[T (y); η]) is called the canonical response function. Its inverse, g−1, is called the canonical link function. Thus, the canonical response function for the Gaussian family is just the identity function; and the canonical response function for the Bernoulli is the logistic function.3

3 Many texts use g to denote the link function, and g−1 to denote the response function; but the notation we’re using here, inherited from the early machine learning literature, will be more consistent with the notation used in the rest of the class.


# Chapter 4

# Generative learning algorithms

So far, we’ve mainly been talking about learning algorithms that model p(y|x; θ), the conditional distribution of y given x. For instance, logistic regression modeled p(y|x; θ) as hθ(x) = g(θᵀ x) where g is the sigmoid function. In these notes, we’ll talk about a different type of learning algorithm.

Consider a classification problem in which we want to learn to distinguish between elephants (y = 1) and dogs (y = 0), based on some features of an animal. Given a training set, an algorithm like logistic regression or the perceptron algorithm (basically) tries to find a straight line—that is, a decision boundary—that separates the elephants and dogs. Then, to classify a new animal as either an elephant or a dog, it checks on which side of the decision boundary it falls, and makes its prediction accordingly.

Here’s a different approach. First, looking at elephants, we can build a model of what elephants look like. Then, looking at dogs, we can build a separate model of what dogs look like. Finally, to classify a new animal, we can match the new animal against the elephant model, and match it against the dog model, to see whether the new animal looks more like the elephants or more like the dogs we had seen in the training set.

Algorithms that try to learn p(y|x) directly (such as logistic regression), or algorithms that try to learn mappings directly from the space of inputs X to the labels {0, 1}, (such as the perceptron algorithm) are called discriminative learning algorithms. Here, we’ll talk about algorithms that instead try to model p(x|y) (and p(y)). These algorithms are called generative learning algorithms. For instance, if y indicates whether an example is a dog (0) or an elephant (1), then p(x|y = 0) models the distribution of dogs’ features, and p(x|y = 1) models the distribution of elephants’ features.

After modeling p(y) (called the class priors) and p(x|y), our algorithm



can then use Bayes rule to derive the posterior distribution on y given x:

p(y|x) = p(x|y)p(y).
p(x)

Here, the denominator is given by p(x) = p(x|y = 1)p(y = 1) + p(x|y = 0)p(y = 0) (you should be able to verify that this is true from the standard properties of probabilities), and thus can also be expressed in terms of the quantities p(x|y) and p(y) that we’ve learned. Actually, if were calculating p(y|x) in order to make a prediction, then we don’t actually need to calculate the denominator, since

arg max p(y|x) = arg max p(x|y)p(y)
y y p(x)
= arg max p(x|y)p(y).

# 4.1 Gaussian discriminant analysis

The first generative learning algorithm that we’ll look at is Gaussian discriminant analysis (GDA). In this model, we’ll assume that p(x|y) is distributed according to a multivariate normal distribution. Let’s talk briefly about the properties of multivariate normal distributions before moving on to the GDA model itself.

# 4.1.1 The multivariate normal distribution

The multivariate normal distribution in d-dimensions, also called the multivariate Gaussian distribution, is parameterized by a mean vector μ ∈ Rd and a covariance matrix Σ ∈ Rd×d, where Σ ≥ 0 is symmetric and positive semi-definite. Also written “N (μ, Σ)”, its density is given by:

p(x; μ, Σ) = (2π)d/2 |Σ|−1/2 exp (− 1/2 (x − μ)ᵀ Σ−1(x − μ)).

In the equation above, “|Σ|” denotes the determinant of the matrix Σ. For a random variable X distributed N (μ, Σ), the mean is (unsurprisingly) given by μ:

E[X] = ∫ x p(x; μ, Σ)dx = μ.

The covariance of a vector-valued random variable Z is defined as Cov(Z) = E[(Z − E[Z])(Z − E[Z])ᵀ]. This generalizes the notion of the variance of a





# 36

real-valued random variable. The covariance can also be defined as Cov(Z) = E[ZZᵀ ] − (E[Z])(E[Z])ᵀ . (You should be able to prove to yourself that these two definitions are equivalent.) If X ∼ N (μ, Σ), then

Cov(X) = Σ.

Here are some examples of what the density of a Gaussian distribution looks like:

| 0.25 | 0.25 | 0.25 |
| ---- | ---- | ---- |
| 0.2  | 0.2  | 0.2  |
| 0.15 | 0.15 | 0.15 |
| 0.1  | 0.1  | 0.1  |
| 0.05 | 0.05 | 0.05 |
| 3    | 3    | 3    |
| 2    | 1    | 0    |
| −1   | −2   | −3   |
| −3   | −2   | −1   |
| 0    | 1    | 2    |
| 3    |      |      |

The left-most figure shows a Gaussian with mean zero (that is, the 2x1 zero-vector) and covariance matrix Σ = I (the 2x2 identity matrix). A Gaussian with zero mean and identity covariance is also called the standard normal distribution. The middle figure shows the density of a Gaussian with zero mean and Σ = 0.6I; and in the rightmost figure shows one with Σ = 2I. We see that as Σ becomes larger, the Gaussian becomes more “spread-out,” and as it becomes smaller, the distribution becomes more “compressed.”

Let’s look at some more examples.

| 0.25 | 0.25 | 0.25 |
| ---- | ---- | ---- |
| 0.2  | 0.2  | 0.2  |
| 0.15 | 0.15 | 0.15 |
| 0.1  | 0.1  | 0.1  |
| 0.05 | 0.05 | 0.05 |
| 3    | 3    | 3    |
| 2    | 2    | 2    |
| 1    | 1    | 1    |
| 0    | 0    | 0    |
| −1   | −2   | −3   |
| −3   | −2   | −1   |
| 0    | 1    | 2    |
| 3    |      |      |

The figures above show Gaussians with mean 0, and with covariance matrices respectively

| Σ = \[ 1 0 ] ; | Σ = \[ 1 0.5 ] ; | Σ = \[ 1 0.8 ] . |
| -------------- | ---------------- | ---------------- |
| 0 1            | 0.5 1            | 0.8 1            |

The leftmost figure shows the familiar standard normal distribution, and we see that as we increase the off-diagonal entry in Σ, the density becomes more






“compressed” towards the 45◦ line (given by x₁ = x₂). We can see this more clearly when we look at the contours of the same three densities:

| 3  | 3  | 3  |
| -- | -- | -- |
| 2  | 2  | 2  |
| 1  | 1  | 1  |
| 0  | 0  | 0  |
| −1 | −1 | −1 |
| −2 | −2 | −2 |
| −3 | −3 | −3 |

| −3 | −2 | −1 | 0 | 1 | 2 | 3 |
| -- | -- | -- | - | - | - | - |
| −3 | −2 | −1 | 0 | 1 | 2 | 3 |
| −3 | −2 | −1 | 0 | 1 | 2 | 3 |

Here’s one last set of examples generated by varying Σ:

| 3  | 3  | 3  |
| -- | -- | -- |
| 2  | 2  | 2  |
| 1  | 1  | 1  |
| 0  | 0  | 0  |
| −1 | −1 | −1 |
| −2 | −2 | −2 |
| −3 | −3 | −3 |

| −3 | −2 | −1 | 0 | 1 | 2 | 3 |
| -- | -- | -- | - | - | - | - |
| −3 | −2 | −1 | 0 | 1 | 2 | 3 |
| −3 | −2 | −1 | 0 | 1 | 2 | 3 |

The plots above used, respectively,

Σ = [  1  -0.5  ] ;  Σ = [  1  -0.8 ] ;  Σ = [  3  0.8  ] .

−0.5  1  −0.8  1  0.8  1

From the leftmost and middle figures, we see that by decreasing the off-diagonal elements of the covariance matrix, the density now becomes “compressed” again, but in the opposite direction. Lastly, as we vary the parameters, more generally the contours will form ellipses (the rightmost figure showing an example).

As our last set of examples, fixing Σ = I , by varying μ, we can also move the mean of the density around.

| 0.25 | 0.25 | 0.25 |
| ---- | ---- | ---- |
| 0.2  | 0.2  | 0.2  |
| 0.15 | 0.15 | 0.15 |
| 0.1  | 0.1  | 0.1  |
| 0.05 | 0.05 | 0.05 |

| 3  | 3  | 3  |    |    |    |   |
| -- | -- | -- | -- | -- | -- | - |
| 2  | 1  | 0  | −1 | −2 | −3 |   |
| −3 | −2 | −1 | 0  | 1  | 2  | 3 |
| 2  | 1  | 0  | −1 | −2 | −3 |   |
| −3 | −2 | −1 | 0  | 1  | 2  | 3 |
| 2  | 1  | 0  | −1 | −2 | −3 |   |
| −3 | −2 | −1 | 0  | 1  | 2  | 3 |





# 4.1.2 The Gaussian discriminant analysis model

When we have a classification problem in which the input features x are continuous-valued random variables, we can then use the Gaussian Discriminant Analysis (GDA) model, which models p(x|y) using a multivariate normal distribution. The model is:

y ∼ Bernoulli(φ)

x|y = 0 ∼ N(μ₀, Σ)

x|y = 1 ∼ N(μ₁, Σ)

Writing out the distributions, this is:

p(y) = φy(1 − φ)1−y (1)

p(x|y = 0) = (2π)d/2 |Σ|−1/2 exp(−1/2(x − μ₀)ᵀ Σ−1(x − μ₀))

p(x|y = 1) = (2π)d/2 |Σ|−1/2 exp(−1/2(x − μ₁)ᵀ Σ−1(x − μ₁))

Here, the parameters of our model are φ, Σ, μ₀ and μ₁. (Note that while there’re two different mean vectors μ₀ and μ₁, this model is usually applied using only one covariance matrix Σ.) The log-likelihood of the data is given by

∏i=1n `(φ, μ₀, μ₁, Σ) = log p(x(i), y(i); φ, μ₀, μ₁, Σ

∏i=1n = log p(x(i)|y(i); μ₀, μ₁, Σ)p(y(i); φ.





# 39

By maximizing  `  with respect to the parameters, we find the maximum likelihood estimate of the parameters (see problem set 1) to be:

|   |   |          | ∑                  | φ = 1 n 1{y(i) = 1}                     |
| - | - | -------- | ------------------ | --------------------------------------- |
|   |   |          | n i=1              |                                         |
|   |   |          | ∑ₙ 1{y(i) = 0}x(i) |                                         |
|   |   |          | ∑                  | μ₀ = i=1                                |
|   |   |          | n 1{y(i) = 0}      |                                         |
|   |   |          | ∑ₙ i=1 (i) (i)     |                                         |
|   |   |          |                    | ∑ 1{y = 1}x                             |
|   |   | μ₁ = i=1 |                    |                                         |
|   |   |          | n 1{y(i) = 1}      |                                         |
|   |   |          | i=1                |                                         |
|   |   |          | ∑                  | Σ = 1 n (x(i) − μy(i))(x(i) − μy(i))ᵀ . |
|   |   |          | n i=1              |                                         |

Pictorially, what the algorithm is doing can be seen in as follows:

1

0

−1

−2

−3

−4

−5

−6

−7
−2    −1    0  1     2      3    4  5  6         7

Shown in the figure are the training set, as well as the contours of the two Gaussian distributions that have been fit to the data in each of the two classes. Note that the two Gaussians have contours that are the same shape and orientation, since they share a covariance matrix Σ, but they have different means μ₀   and μ₁. Also shown in the figure is the straight line giving the decision boundary at which p(y = 1|x) = 0.5. On one side of the boundary, we’ll predict y = 1 to be the most likely outcome, and on the other side, we’ll predict y = 0.




# 4.1.3 Discussion: GDA and logistic regression

The GDA model has an interesting relationship to logistic regression. If we view the quantity p(y = 1|x; φ, μ₀, μ₁, Σ) as a function of x, we’ll find that it can be expressed in the form

p(y = 1|x; φ, Σ, μ₀, μ₁) = 1 / (1 + exp(−θᵀ x))

where θ is some appropriate function of φ, Σ, μ₀, μ₁.1 This is exactly the form that logistic regression—a discriminative algorithm—used to model p(y = 1|x).

When would we prefer one model over another? GDA and logistic regression will, in general, give different decision boundaries when trained on the same dataset. Which is better?

We just argued that if p(x|y) is multivariate gaussian (with shared Σ), then p(y|x) necessarily follows a logistic function. The converse, however, is not true; i.e., p(y|x) being a logistic function does not imply p(x|y) is multivariate gaussian. This shows that GDA makes stronger modeling assumptions about the data than does logistic regression. It turns out that when these modeling assumptions are correct, then GDA will find better fits to the data, and is a better model. Specifically, when p(x|y) is indeed gaussian (with shared Σ), then GDA is asymptotically efficient. Informally, this means that in the limit of very large training sets (large n), there is no algorithm that is strictly better than GDA (in terms of, say, how accurately they estimate p(y|x)). In particular, it can be shown that in this setting, GDA will be a better algorithm than logistic regression; and more generally, even for small training set sizes, we would generally expect GDA to be better.

In contrast, by making significantly weaker assumptions, logistic regression is also more robust and less sensitive to incorrect modeling assumptions. There are many different sets of assumptions that would lead to p(y|x) taking the form of a logistic function. For example, if x|y = 0 ∼ Poisson(λ₀), and x|y = 1 ∼ Poisson(λ₁), then p(y|x) will be logistic. Logistic regression will also work well on Poisson data like this. But if we were to use GDA on such data—and fit Gaussian distributions to such non-Gaussian data—then the results will be less predictable, and GDA may (or may not) do well.

To summarize: GDA makes stronger modeling assumptions, and is more data efficient (i.e., requires less training data to learn “well”) when the modeling assumptions are correct or at least approximately correct.

Logistic1

1 This uses the convention of redefining the x(i)’s on the right-hand-side to be (d + 1)-dimensional vectors by adding the extra coordinate x(i) = 1; see problem set 1.




regression makes weaker assumptions, and is significantly more robust to deviations from modeling assumptions. Specifically, when the data is indeed non-Gaussian, then in the limit of large datasets, logistic regression will almost always do better than GDA. For this reason, in practice logistic regression is used more often than GDA. (Some related considerations about discriminative vs. generative models also apply for the Naive Bayes algorithm that we discuss next, but the Naive Bayes algorithm is still considered a very good, and is certainly also a very popular, classification algorithm.)

# 4.2 Naive bayes (Option Reading)

In GDA, the feature vectors x were continuous, real-valued vectors. Let’s now talk about a different learning algorithm in which the xj’s are discrete-valued.

For our motivating example, consider building an email spam filter using machine learning. Here, we wish to classify messages according to whether they are unsolicited commercial (spam) email, or non-spam email. After learning to do this, we can then have our mail reader automatically filter out the spam messages and perhaps place them in a separate mail folder. Classifying emails is one example of a broader set of problems called text classification.

Let’s say we have a training set (a set of emails labeled as spam or non-spam). We’ll begin our construction of our spam filter by specifying the features xj used to represent an email.

We will represent an email via a feature vector whose length is equal to the number of words in the dictionary. Specifically, if an email contains the j-th word of the dictionary, then we will set xj = 1; otherwise, we let xj = 0.

For instance, the vector

 1      a
 0      aardvark
          
 0      aardwolf
 .             .
 .             .
x =  .         .
 1      buy
          
 .             .
 .             .
.          .
0          zygmurgy

is used to represent an email that contains the words “a” and “buy,” but not





“aardvark,” “aardwolf” or “zygmurgy.”² The set of words encoded into the feature vector is called the vocabulary, so the dimension of x is equal to the size of the vocabulary.

Having chosen our feature vector, we now want to build a generative model. So, we have to model p(x|y). But if we have, say, a vocabulary of 50000 words, then x ∈ {0, 1}50000 (x is a 50000-dimensional vector of 0’s and 1’s), and if we were to model x explicitly with a multinomial distribution over the 250000 possible outcomes, then we’d end up with a (250000 − 1)-dimensional parameter vector. This is clearly too many parameters.

To model p(x|y), we will therefore make a very strong assumption. We will assume that the xi’s are conditionally independent given y. This assumption is called the Naive Bayes (NB) assumption, and the resulting algorithm is called the Naive Bayes classifier. For instance, if y = 1 means spam email; “buy” is word 2087 and “price” is word 39831; then we are assuming that if I tell you y = 1 (that a particular piece of email is spam), then knowledge of x2087 (knowledge of whether “buy” appears in the message) will have no effect on your beliefs about the value of x39831 (whether “price” appears).

More formally, this can be written p(x2087|y) = p(x2087|y, x39831). (Note that this is not the same as saying that x2087 and x39831 are independent, which would have been written “p(x2087) = p(x2087|x39831); rather, we are only assuming that x2087 and x39831 are conditionally independent given y.)

We now have:

p(x1, . . . , x50000|y) = p(x1|y)p(x2|y, x1)p(x3|y, x1, x2) · · · p(x50000|y, x1, . . . , x49999) = p(x1|y)p(x2|y)p(x3|y) · · · p(x50000|y) ∏d = p(xj|y)

The first equality simply follows from the usual properties of probabilities, and the second equality used the NB assumption. We note that even though

²Actually, rather than looking through an English dictionary for the list of all English words, in practice it is more common to look through our training set and encode in our feature vector only the words that occur at least once there. Apart from reducing the number of words modeled and hence reducing our computational and space requirements, this also has the advantage of allowing us to model/include as a feature many words that may appear in your email (such as “cs229”) but that you won’t find in a dictionary. Sometimes (as in the homework), we also exclude the very high frequency words (which will be words like “the,” “of,” “and”; these high frequency, “content free” words are called stop words) since they occur in so many documents and do little to indicate whether an email is spam or non-spam.





# Naive Bayes Algorithm

The Naive Bayes assumption is an extremely strong assumption; the resulting algorithm works well on many problems.

Our model is parameterized by:

- φj|y=1 = p(xj = 1|y = 1)
- φj|y=0 = p(xj = 1|y = 0)
- φy = p(y = 1)

As usual, given a training set {(x(i), y(i)); i = 1, . . . , n}, we can write down the joint likelihood of the data:

L(φy, φj|y=0, φj|y=1) = ∏i=1n p(x(i), y(i)).

Maximizing this with respect to φy, φj|y=0, and φj|y=1 gives the maximum likelihood estimates:

φj|y=1 = ∑i=1n 1{x(i) = 1 ∧ y(i) = 1} / ∑i=1n 1{y(i) = 1}

φj|y=0 = ∑i=1n 1{x(i) = 1 ∧ y(i) = 0} / ∑i=1n 1{y(i) = 0}

φy = ∑i=1n 1{y(i) = 1} / n

In the equations above, the “∧” symbol means “and.” The parameters have a very natural interpretation. For instance, φj|y=1 is just the fraction of the spam (y = 1) emails in which word j does appear.

Having fit all these parameters, to make a prediction on a new example with features x, we then simply calculate:

p(y = 1|x) = p(x|y = 1)p(y = 1) / p(x)

= (∏j=1d p(xj|y = 1)) p(y = 1) + (∏j=1d p(xj|y = 0)) p(y = 0)

and pick whichever class has the higher posterior probability.

Lastly, we note that while we have developed the Naive Bayes algorithm mainly for the case of problems where the features xj are binary-valued, the generalization to where xj can take values in {1, 2, . . . , kj} is straightforward. Here, we would simply model p(xj|y) as multinomial rather than as Bernoulli.

Indeed, even if some original input attribute (say, the living area of a house, as in our earlier example) were continuous valued, it is quite common to discretize it—that is, turn it into a small set of discrete values—and apply Naive Bayes. For instance, if we use some feature xj to represent living area, we might discretize the continuous values as follows:





Living area (sq. feet)



“neurips” is spam, it calculates the class posterior probabilities, and obtains

p(y = 1 | x) =
∏d p(xj | y = 1)p(y = 1)

p(y = 1 | x) =
∏d p(xj | y = 1)p(y = 1) + ∏d p(xj | y = 0)p(y = 0)

= 0.

This is because each of the terms “∏d p(xj | y)” includes a term p(x350000 | y) = 0 that is multiplied into it. Hence, our algorithm obtains 0/0, and doesn’t know how to make a prediction.

Stating the problem more broadly, it is statistically a bad idea to estimate the probability of some event to be zero just because you haven’t seen it before in your finite training set. Take the problem of estimating the mean of a multinomial random variable z taking values in {1, . . . , k}. We can parameterize our multinomial with φj = p(z = j). Given a set of n independent observations {z(1), . . . , z(n)}, the maximum likelihood estimates are given by

φj = (∑n 1{z(i) = j}) / n.

As we saw previously, if we were to use these maximum likelihood estimates, then some of the φj’s might end up as zero, which was a problem. To avoid this, we can use Laplace smoothing, which replaces the above estimate with

φj = (1 + ∑n 1{z(i) = j}) / (k + n).

Here, we’ve added 1 to the numerator, and k to the denominator. Note that ∑k φj = 1 still holds (check this yourself!), which is a desirable property since the φj’s are estimates for probabilities that we know must sum to 1. Also, φj = 0 for all values of j, solving our problem of probabilities being estimated as zero. Under certain (arguably quite strong) conditions, it can be shown that the Laplace smoothing actually gives the optimal estimator of the φj’s.

Returning to our Naive Bayes classifier, with Laplace smoothing, we therefore obtain the following estimates of the parameters:

φj | y=1 = (1 + ∑n 1{x(i) = 1 ∧ y(i) = 1}) / (2 + ∑n 1{y(i) = 1})

φj | y=0 = (1 + ∑n 1{x(i) = 1 ∧ y(i) = 0}) / (2 + ∑n 1{y(i) = 0})





(In practice, it usually doesn’t matter much whether we apply Laplace smoothing to φy or not, since we will typically have a fair fraction each of spam and non-spam messages, so φy will be a reasonable estimate of p(y = 1) and will be quite far from 0 anyway.)

# 4.2.2 Event models for text classification

To close off our discussion of generative learning algorithms, let’s talk about one more model that is specifically for text classification. While Naive Bayes as we’ve presented it will work well for many classification problems, for text classification, there is a related model that does even better.

In the specific context of text classification, Naive Bayes as presented uses the what’s called the Bernoulli event model (or sometimes multi-variate Bernoulli event model). In this model, we assumed that the way an email is generated is that first it is randomly determined (according to the class priors p(y)) whether a spammer or non-spammer will send you your next message. Then, the person sending the email runs through the dictionary, deciding whether to include each word j in that email independently and according to the probabilities p(xj = 1 | y) = φj | y. Thus, the probability of a message was given by p(y) ∏j=1d p(xj | y).

Here’s a different model, called the Multinomial event model. To describe this model, we will use a different notation and set of features for representing emails. We let xj denote the identity of the j-th word in the email. Thus, xj is now an integer taking values in {1, . . . , V}, where V is the size of our vocabulary (dictionary). An email of d words is now represented by a vector (x1, x2, . . . , xd) of length d; note that d can vary for different documents. For instance, if an email starts with “A NeurIPS . . . ,” then x1 = 1 (“a” is the first word in the dictionary), and x2 = 35000 (if “neurips” is the 35000th word in the dictionary).

In the multinomial event model, we assume that the way an email is generated is via a random process in which spam/non-spam is first determined (according to p(y)) as before. Then, the sender of the email writes the email by first generating x1 from some multinomial distribution over words (p(x1 | y)). Next, the second word x2 is chosen independently of x1 but from the same multinomial distribution, and similarly for x3, x4, and so on, until all d words of the email have been generated. Thus, the overall probability of a message is given by p(y) ∏j=1d p(xj | y). Note that this formula looks like the one we had earlier for the probability of a message under the Bernoulli event model, but that the terms in the formula now mean very different things. In particular, xj | y is now a multinomial, rather than a Bernoulli distribution.





The parameters for our new model are φy = p(y) as before, φk|y=1 = p(xj = k|y = 1) (for any j) and φk|y=0 = p(xj = k|y = 0). Note that we have assumed that p(xj|y) is the same for all values of j (i.e., that the distribution according to which a word is generated does not depend on its position j within the email).

If we are given a training set {(x(i), y(i)); i = 1, . . . , n} where x(i) = (x1(i), x2(i), . . . , xdi(i)) (here, di is the number of words in the i-training example), the likelihood of the data is given by

L(φy, φk|y=0, φk|y=1) = p(x(i), y(i))

= ∏i=1n ( ∏j p(x(i)|y; φk|y=0, φk|y=1) p(y(i); φy).

Maximizing this yields the maximum likelihood estimates of the parameters:

φk|y=1 = ∑i=1n ∑j=1di 1{x(i) = k ∧ y(i) = 1} / ∑i=1n 1{y(i) = 1}di

φk|y=0 = ∑i=1n ∑j=1di 1{x(i) = k ∧ y(i) = 0} / ∑i=1n 1{y(i) = 0}di

φy = ∑i=1n 1{y(i) = 1} / n

If we were to apply Laplace smoothing (which is needed in practice for good performance) when estimating φk|y=0 and φk|y=1, we add 1 to the numerators and V to the denominators, and obtain:

φk|y=1 = 1 + ∑i=1n ∑j=1di 1{x(i) = k ∧ y(i) = 1} / V + ∑i=1n 1{y(i) = 1}di

φk|y=0 = 1 + ∑i=1n ∑j=1di 1{x(i) = k ∧ y(i) = 0} / V + ∑i=1n 1{y(i) = 0}di

While not necessarily the very best classification algorithm, the Naive Bayes classifier often works surprisingly well. It is often also a very good “first thing to try,” given its simplicity and ease of implementation.





# Chapter 5

# Kernel methods

# 5.1  Feature maps

Recall that in our discussion about linear regression, we considered the problem of predicting the price of a house (denoted by y) from the living area of the house (denoted by x), and we fit a linear function of x to the training data. What if the price y can be more accurately represented as a non-linear function of x? In this case, we need a more expressive family of models than linear models.

We start by considering fitting cubic functions y = θ₃x³ + θ₂x² + θ₁x + θ₀. It turns out that we can view the cubic function as a linear function over a different set of feature variables (defined below). Concretely, let the function φ : R → R⁴ be defined as

 1 
 x 
φ(x) =                   ∈ R⁴.                          (5.1)
 x² 
x³

Let θ ∈ R⁴ be the vector containing θ₀, θ₁, θ₂, θ₃ as entries. Then we can rewrite the cubic function in x as:

θ₃x³ + θ₂x² + θ₁x + θ₀ = θᵀ φ(x)

Thus, a cubic function of the variable x can be viewed as a linear function over the variables φ(x). To distinguish between these two sets of variables, in the context of kernel methods, we will call the “original” input value the input attributes of a problem (in this case, x, the living area). When the





original input is mapped to some new set of quantities φ(x), we will call those new quantities the features variables. (Unfortunately, different authors use different terms to describe these two things in different contexts.) We will call φ a feature map, which maps the attributes to the features.

# 5.2   LMS (least mean squares) with features

We will derive the gradient descent algorithm for fitting the model θᵀ φ(x). First recall that for ordinary least square problem where we were to fit θᵀ x, the batch gradient descent update is (see the first lecture note for its derivation):

∑
θ := θ + α n  (y(i) − hθ(x(i))) x(i)
i=1
∑
:= θ + α n  (y(i) − θT x(i)) x(i).                            (5.2)
i=1

Let φ : Rᵈ → Rᵖ be a feature map that maps attribute x (in Rᵈ) to the features φ(x) in Rᵖ. (In the motivating example in the previous subsection, we have d = 1 and p = 4.) Now our goal is to fit the function θᵀ φ(x), with θ being a vector in Rᵖ instead of Rᵈ. We can replace all the occurrences of x(i) in the algorithm above by φ(x(i)) to obtain the new update:

∑
θ := θ + α  n (y(i) − θT φ(x(i))) φ(x(i))                        (5.3)
i=1

Similarly, the corresponding stochastic gradient descent update rule is

θ := θ + α (y(i) − θᵀ φ(x(i))) φ(x(i))                           (5.4)

# 5.3   LMS with the kernel trick

The gradient descent update, or stochastic gradient update above becomes computationally expensive when the features φ(x) is high-dimensional. For example, consider the direct extension of the feature map in equation (5.1) to high-dimensional input x: suppose x ∈ Rᵈ, and let φ(x) be the vector that





# 50

contains all the monomials of x with degree ≤ 3

|        | 1    |   |
| ------ | ---- | - |
| x      |      |   |
|        | 1    |   |
| x      |      |   |
|        | 2    |   |
|        | .    |   |
|        | .    |   |
|        | .    |   |
|        | x²   |   |
|        | 1    |   |
| x₁x₂   |      |   |
|        |      |   |
| φ(x) = | x₁x₃ |   |
|        | 1    | . |
|        | .    |   |
| x₂x₁   |      |   |
|        | 2    |   |
|        | .    |   |
|        | .    |   |
|        | .    |   |
|        | x³   |   |
|        | 1    |   |
| x₂x    |      |   |
|        | 1    | 2 |

The dimension of the features φ(x) is on the order of d³.¹ This is a prohibitively long vector for computational purpose — when d = 1000, each update requires at least computing and storing a 1000³ = 10⁹ dimensional vector, which is 10⁶ times slower than the update rule for ordinary least squares updates (5.2).

It may appear at first that such d³ runtime per update and memory usage are inevitable, because the vector θ itself is of dimension p ≈ d³, and we may need to update every entry of θ and store it. However, we will introduce the kernel trick with which we will not need to store θ explicitly, and the runtime can be significantly improved.

For simplicity, we assume the initialize the value θ = 0, and we focus on the iterative update (5.3). The main observation is that at any time, θ can be represented as a linear combination of the vectors φ(x(1)), . . . , φ(x(ⁿ)). Indeed, we can show this inductively as follows. At initialization, θ = 0 = ∑ₙ 0 · φ(x(i)). Assume at some point, θ can be represented as

θ = ∑i=1n βiφ(x(i)) (5.6)

¹ Here, for simplicity, we include all the monomials with repetitions (so that, e.g., x1x2x3 and x2x3x1 both appear in φ(x)). Therefore, there are totally 1 + d + d² + d³ entries in φ(x).





for some β₁, . . . , βₙ ∈ R. Then we claim that in the next round, θ is still a linear combination of φ(x(1)), . . . , φ(x(ⁿ)) because

θ := θ + α n (y(i) − θT φ(x(i))) φ(x(i))

= n βiφ(x(i)) + α n (y(i) − θT φ(x(i))) φ(x(i))

= n (βi + α (y(i) − θᵀ φ(x(i)))) φ(x(i))         (5.7)

You may realize that our general strategy is to implicitly represent the p-dimensional vector θ by a set of coefficients β₁, . . . , βₙ. Towards doing this, we derive the update rule of the coefficients β₁, . . . , βₙ. Using the equation above, we see that the new βi depends on the old one via

βi := βi + α (y(i) − θᵀ φ(x(i)))             (5.8)

Here we still have the old θ on the RHS of the equation. Replacing θ by θ = ∑ⁿ βj φ(x(ʲ)) gives

∀i ∈ {1, . . . , n}, βi := βi + α (y(i) − n βj φ(x(ʲ))ᵀ φ(x(i)))

We often rewrite φ(x(ʲ))ᵀ φ(x(i)) as 〈φ(x(ʲ)), φ(x(i))〉 to emphasize that it’s the inner product of the two feature vectors. Viewing βi’s as the new representation of θ, we have successfully translated the batch gradient descent algorithm into an algorithm that updates the value of β iteratively. It may appear that at every iteration, we still need to compute the values of 〈φ(x(ʲ)), φ(x(i))〉 for all pairs of i, j, each of which may take roughly O(p) operation. However, two important properties come to rescue:

1. We can pre-compute the pairwise inner products 〈φ(x(ʲ)), φ(x(i))〉 for all pairs of i, j before the loop starts.
2. For the feature map φ defined in (5.5) (or many other interesting feature maps), computing 〈φ(x(ʲ)), φ(x(i))〉 can be efficient and does not





necessarily require computing φ(x(i)) explicitly. This is because:



the knowledge of the representation β suffices to compute the prediction θᵀ φ(x). Indeed, we have

∑
θᵀ φ(x) = n βiφ(x(i))ᵀ φ(x) = n βiK(x(i), x) (5.12)
∑

You may realize that fundamentally all we need to know about the feature map φ(·) is encapsulated in the corresponding kernel function K(·, ·). We will expand on this in the next section.

# 5.4 Properties of kernels

In the last subsection, we started with an explicitly defined feature map φ, which induces the kernel function K(x, z), 〈φ(x), φ(z)〉. Then we saw that the kernel function is so intrinsic so that as long as the kernel function is defined, the whole training algorithm can be written entirely in the language of the kernel without referring to the feature map φ, so can the prediction of a test example x (equation (5.12).)

Therefore, it would be tempted to define other kernel function K(·, ·) and run the algorithm (5.11). Note that the algorithm (5.11) does not need to explicitly access the feature map φ, and therefore we only need to ensure the existence of the feature map φ, but do not necessarily need to be able to explicitly write φ down.

What kinds of functions K(·, ·) can correspond to some feature map φ? In other words, can we tell if there is some feature mapping φ so that K(x, z) = φ(x)ᵀ φ(z) for all x, z?

If we can answer this question by giving a precise characterization of valid kernel functions, then we can completely change the interface of selecting feature maps φ to the interface of selecting kernel function K. Concretely, we can pick a function K, verify that it satisfies the characterization (so that there exists a feature map φ that K corresponds to), and then we can run update rule (5.11). The benefit here is that we don’t have to be able to compute φ or write it down analytically, and we only need to know its existence. We will answer this question at the end of this subsection after we go through several concrete examples of kernels.

Suppose x, z ∈ Rᵈ, and let’s first consider the function K(·, ·) defined as:

K(x, z) = (xᵀ z)².




We can also write this as

(∑)
(∑)

K(x, z) = xizi xjzj

∑ ∑

= xixjzizj

∑

= (xixj)(zizj)

i,j=1

Thus, we see that K(x, z) = ⟨φ(x), φ(z)⟩ is the kernel function that corresponds to the feature mapping φ given (shown here for the case of d = 3) by

x1x2
x1x3
x2x1
x2x2
x2x3
x3x1
x3x2
x3x3

| φ(x) = | x1x1 |
| ------ | ---- |

Revisiting the computational efficiency perspective of kernel, note that whereas calculating the high-dimensional φ(x) requires O(d²) time, finding K(x, z) takes only O(d) time—linear in the dimension of the input attributes.

For another related example, also consider K(·, ·) defined by

K(x, z) = (xᵀz + c)²

= ∑i,j=1d (xixj)(zizj) + ∑i=1d (√2c xi)(√2c zi) + c².

(Check this yourself.) This function K is a kernel function that corresponds



to the feature mapping (again shown for d = 3)

|      | x₁x₁ |   |
| ---- | ---- | - |
|      | x₁x₂ |   |
|      |      |   |
|      | x₁x₃ |   |
|      |      |   |
| x₂x₁ |      |   |
| x₂x₂ |      |   |
| x₂x₃ |      |   |
| x₃x₁ |      |   |
| x₃x₂ |      |   |
| x₃x₃ |      |   |
| √3   |      |   |
| 2cx₁ |      |   |
|      | √    | 1 |
| 2cx₂ |      |   |
|      | √    | 2 |
|      | 2cx₃ |   |
|      | c    |   |

and the parameter c controls the relative weighting between the xi (first order) and the xixj (second order) terms.

More broadly, the kernel K(x, z) = (xᵀ z + c)ᵏ corresponds to a feature mapping to an (d+k) feature space, corresponding of all monomials of the form xi1 xi2 ... xik that are up to order k. However, despite working in this dk-dimensional space, computing K(x, z) still takes only O(d) time, and hence we never need to explicitly represent feature vectors in this very high dimensional feature space.

# Kernels as similarity metrics.

Now, let’s talk about a slightly different view of kernels. Intuitively, (and there are things wrong with this intuition, but nevermind), if φ(x) and φ(z) are close together, then we might expect K(x, z) = φ(x)ᵀ φ(z) to be large. Conversely, if φ(x) and φ(z) are far apart—say nearly orthogonal to each other—then K(x, z) = φ(x)ᵀ φ(z) will be small. So, we can think of K(x, z) as some measurement of how similar are φ(x) and φ(z), or of how similar are x and z.

Given this intuition, suppose that for some learning problem that you’re working on, you’ve come up with some function K(x, z) that you think might be a reasonable measure of how similar x and z are. For instance, perhaps you chose K(x, z) = exp (−||x − z||² / 2σ²).





# 56

a feature map φ such that the kernel K defined above satisfies K(x, z) = φ(x)ᵀ φ(z)? In this particular example, the answer is yes. This kernel is called the Gaussian kernel, and corresponds to an infinite dimensional feature mapping φ. We will give a precise characterization about what properties a function K needs to satisfy so that it can be a valid kernel function that corresponds to some feature map φ.

# Necessary conditions for valid kernels.

Suppose for now that K is indeed a valid kernel corresponding to some feature mapping φ, and we will first see what properties it satisfies. Now, consider some finite set of n points (not necessarily the training set) {x(1), . . . , x(ⁿ)}, and let a square, n-by-n matrix K be defined so that its (i, j)-entry is given by Kij = K(x(i), x(ʲ)). This matrix is called the kernel matrix. Note that we’ve overloaded the notation and used K to denote both the kernel function K(x, z) and the kernel matrix K, due to their obvious close relationship.

Now, if K is a valid kernel, then Kij = K(x(i), x(ʲ)) = φ(x(i))ᵀ φ(x(ʲ)) = φ(x(ʲ))ᵀ φ(x(i)) = K(x(ʲ), x(i)) = Kji, and hence K must be symmetric. Moreover, letting φk(x) denote the k-th coordinate of the vector φ(x), we find that for any vector z, we have

zᵀ Kz = ∑ ∑ ziKij zj

= ∑ ∑ ziφ(x(i))T φ(x(j))zj

= ∑ ∑ zi ∑ φk(x(i))φk(x(j))zj

= ∑ ∑ ∑ ziφk(x(i))φk(x(j))zj

= ∑ (∑ ziφk(x(i)))2

≥ 0.

The second-to-last step uses the fact that ∑i,j aiaj = (∑i ai)2 for ai = ziφk(x(i)). Since z was arbitrary, this shows that K is positive semi-definite (K ≥ 0).

Hence, we’ve shown that if K is a valid kernel (i.e., if it corresponds to some feature mapping φ), then the corresponding kernel matrix K ∈ Rn×n is symmetric positive semidefinite.





# 57

# Sufficient conditions for valid kernels.

More generally, the condition above turns out to be not only a necessary, but also a sufficient, condition for K to be a valid kernel (also called a Mercer kernel). The following result is due to Mercer.³

# Theorem (Mercer).

Let K : Rᵈ × Rᵈ → R be given. Then for K to be a valid (Mercer) kernel, it is necessary and sufficient that for any {x(1), . . . , x(ⁿ)}, (n &#x3C; ∞), the corresponding kernel matrix is symmetric positive semi-definite.

Given a function K, apart from trying to find a feature mapping φ that corresponds to it, this theorem therefore gives another way of testing if it is a valid kernel. You’ll also have a chance to play with these ideas more in problem set 2.

In class, we also briefly talked about a couple of other examples of kernels. For instance, consider the digit recognition problem, in which given an image (16x16 pixels) of a handwritten digit (0-9), we have to figure out which digit it was. Using either a simple polynomial kernel K(x, z) = (xᵀ z)ᵏ or the Gaussian kernel, SVMs were able to obtain extremely good performance on this problem. This was particularly surprising since the input attributes x were just 256-dimensional vectors of the image pixel intensity values, and the system had no prior knowledge about vision, or even about which pixels are adjacent to which other ones.

Another example that we briefly talked about in lecture was that if the objects x that we are trying to classify are strings (say, x is a list of amino acids, which strung together form a protein), then it seems hard to construct a reasonable, “small” set of features for most learning algorithms, especially if different strings have different lengths. However, consider letting φ(x) be a feature vector that counts the number of occurrences of each length-k substring in x. If we’re considering strings of English letters, then there are 26ᵏ such strings. Hence, φ(x) is a 26ᵏ dimensional vector; even for moderate values of k, this is probably too big for us to efficiently work with. (e.g., 26⁴ ≈ 460000.) However, using (dynamic programming-ish) string matching algorithms, it is possible to efficiently compute K(x, z) = φ(x)ᵀ φ(z), so that we can now implicitly work in this 26ᵏ-dimensional feature space, but without ever explicitly computing feature vectors in this space.

³Many texts present Mercer’s theorem in a slightly more complicated form involving L2 functions, but when the input attributes take values in Rd, the version given here is equivalent.




Application of kernel methods: We’ve seen the application of kernels to linear regression. In the next part, we will introduce the support vector machines to which kernels can be directly applied. dwell too much longer on it here. In fact, the idea of kernels has significantly broader applicability than linear regression and SVMs. Specifically, if you have any learning algorithm that you can write in terms of only inner products 〈x, z〉 between input attribute vectors, then by replacing this with K(x, z) where K is a kernel, you can “magically” allow your algorithm to work efficiently in the high dimensional feature space corresponding to K. For instance, this kernel trick can be applied with the perceptron to derive a kernel perceptron algorithm. Many of the algorithms that we’ll see later in this class will also be amenable to this method, which has come to be known as the “kernel trick.”



Chapter 6

# Support vector machines

This set of notes presents the Support Vector Machine (SVM) learning algorithm. SVMs are among the best (and many believe are indeed the best) “off-the-shelf” supervised learning algorithms. To tell the SVM story, we’ll need to first talk about margins and the idea of separating data with a large “gap.” Next, we’ll talk about the optimal margin classifier, which will lead us into a digression on Lagrange duality. We’ll also see kernels, which give a way to apply SVMs efficiently in very high dimensional (such as infinite-dimensional) feature spaces, and finally, we’ll close off the story with the SMO algorithm, which gives an efficient implementation of SVMs.

# 6.1 Margins: intuition

We’ll start our story on SVMs by talking about margins. This section will give the intuitions about margins and about the “confidence” of our predictions; these ideas will be made formal in Section 6.3.

Consider logistic regression, where the probability p(y = 1|x; θ) is modeled by hθ(x) = g(θᵀ x). We then predict “1” on an input x if and only if hθ(x) ≥ 0.5, or equivalently, if and only if θᵀ x ≥ 0. Consider a positive training example (y = 1). The larger θᵀ x is, the larger also is hθ(x) = p(y = 1|x; θ), and thus also the higher our degree of “confidence” that the label is 1. Thus, informally we can think of our prediction as being very confident that y = 1 if θᵀ x > 0. Similarly, we think of logistic regression as confidently predicting y = 0, if θᵀ x &#x3C; 0. Given a training set, again informally it seems that we’d have found a good fit to the training data if we can find θ so that θᵀ x(i) > 0 whenever y(i) = 1, and θᵀ x(i) &#x3C; 0 whenever y(i) = 0, since this would reflect a very confident (and correct) set of classifications for all the



training examples. This seems to be a nice goal to aim for, and we’ll soon formalize this idea using the notion of functional margins.

For a different type of intuition, consider the following figure, in which x’s represent positive training examples, o’s denote negative training examples, a decision boundary (this is the line given by the equation θᵀ x = 0, and is also called the separating hyperplane) is also shown, and three points have also been labeled A, B and C.

A
C
B

Notice that the point A is very far from the decision boundary. If we are asked to make a prediction for the value of y at A, it seems we should be quite confident that y = 1 there. Conversely, the point C is very close to the decision boundary, and while it’s on the side of the decision boundary on which we would predict y = 1, it seems likely that just a small change to the decision boundary could easily have caused our prediction to be y = 0. Hence, we’re much more confident about our prediction at A than at C. The point B lies in-between these two cases, and more broadly, we see that if a point is far from the separating hyperplane, then we may be significantly more confident in our predictions. Again, informally we think it would be nice if, given a training set, we manage to find a decision boundary that allows us to make all correct and confident (meaning far from the decision boundary) predictions on the training examples. We’ll formalize this later using the notion of geometric margins.





# 6.2 Notation (option reading)

To make our discussion of SVMs easier, we’ll first need to introduce a new notation for talking about classification. We will be considering a linear classifier for a binary classification problem with labels y and features x. From now, we’ll use y ∈ {−1, 1} (instead of {0, 1}) to denote the class labels. Also, rather than parameterizing our linear classifier with the vector θ, we will use parameters w, b, and write our classifier as

hw,b(x) = g(wT x + b).
Here, g(z) = 1 if z ≥ 0, and g(z) = −1 otherwise. This “w, b” notation allows us to explicitly treat the intercept term b separately from the other parameters. (We also drop the convention we had previously of letting x0 = 1 be an extra coordinate in the input feature vector.) Thus, b takes the role of what was previously θ0, and w takes the role of [θ1 . . . θd]T. Note also that, from our definition of g above, our classifier will directly predict either 1 or −1 (cf. the perceptron algorithm), without first going through the intermediate step of estimating p(y = 1) (which is what logistic regression does).

# 6.3 Functional and geometric margins (option reading)

Let’s formalize the notions of the functional and geometric margins. Given a training example (x(i), y(i)), we define the functional margin of (w, b) with respect to the training example as

γ(i) = y(i) (wT x(i) + b).
Note that if y(i) = 1, then for the functional margin to be large (i.e., for our prediction to be confident and correct), we need wT x(i) + b to be a large positive number. Conversely, if y(i) = −1, then for the functional margin to be large, we need wT x(i) + b to be a large negative number. Moreover, if y(i)(wT x(i) + b) > 0, then our prediction on this example is correct. (Check this yourself.) Hence, a large functional margin represents a confident and a correct prediction.

For a linear classifier with the choice of g given above (taking values in {−1, 1}), there’s one property of the functional margin that makes it not a very good measure of confidence, however. Given our choice of g, we note that





If we replace w with 2w and b with 2b, then since g(wᵀ x + b) = g(2wᵀ x + 2b), this would not change hw,b(x) at all. I.e., g, and hence also hw,b(x), depends only on the sign, but not on the magnitude, of wᵀ x + b. However, replacing (w, b) with (2w, 2b) also results in multiplying our functional margin by a factor of 2. Thus, it seems that by exploiting our freedom to scale w and b, we can make the functional margin arbitrarily large without really changing anything meaningful. Intuitively, it might therefore make sense to impose some sort of normalization condition such as that ||w||2 = 1; i.e., we might replace (w, b) with (w/||w||2, b/||w||2), and instead consider the functional margin of (w/||w||2, b/||w||2). We’ll come back to this later.

Given a training set S = {(x(i), y(i)); i = 1, . . . , n}, we also define the function margin of (w, b) with respect to S as the smallest of the functional margins of the individual training examples. Denoted by γˆ, this can therefore be written:

γˆ = mini=1,...,n γiˆ.

Next, let’s talk about geometric margins. Consider the picture below:

A  w

γ(i)

B

The decision boundary corresponding to (w, b) is shown, along with the vector w. Note that w is orthogonal (at 90°) to the separating hyperplane. (You should convince yourself that this must be the case.) Consider the point at A, which represents the input x(i) of some training example with label y(i) = 1. Its distance to the decision boundary, γ(i), is given by the line segment AB.

How can we find the value of γ(i)? Well, w/||w|| is a unit-length vector pointing in the same direction as w. Since A represents x(i), we therefore





find that the point B is given by x(i) − γ(i) · w/||w||. But this point lies on the decision boundary, and all points x on the decision boundary satisfy the equation wᵀ x + b = 0. Hence,

wᵀ (x(i) − γ(i) w ) + b = 0.

Solving for γ(i) yields

γ(i) = wᵀ x(i) + b = ( w )T x(i) + b.

This was worked out for the case of a positive training example at A in the figure, where being on the “positive” side of the decision boundary is good. More generally, we define the geometric margin of (w, b) with respect to a training example (x(i), y(i)) to be

γ(i) = y(i) (( w )T x(i) + b.

Note that if ||w|| = 1, then the functional margin equals the geometric margin—this thus gives us a way of relating these two different notions of margin. Also, the geometric margin is invariant to rescaling of the parameters; i.e., if we replace w with 2w and b with 2b, then the geometric margin does not change. This will in fact come in handy later. Specifically, because of this invariance to the scaling of the parameters, when trying to fit w and b to training data, we can impose an arbitrary scaling constraint on w without changing anything important; for instance, we can demand that ||w|| = 1, or |w₁| = 5, or |w₁ + b| + |w₂| = 2, and any of these can be satisfied simply by rescaling w and b.

Finally, given a training set S = {(x(i), y(i)); i = 1, . . . , n}, we also define the geometric margin of (w, b) with respect to S to be the smallest of the geometric margins on the individual training examples:

γ = min γ(i).

# 6.4 The optimal margin classifier (option reading)

Given a training set, it seems from our previous discussion that a natural desideratum is to try to find a decision boundary that maximizes the (geometric) margin, since this would reflect a very confident set of predictions.





on the training set and a good “fit” to the training data. Specifically, this will result in a classifier that separates the positive and the negative training examples with a “gap” (geometric margin).

For now, we will assume that we are given a training set that is linearly separable; i.e., that it is possible to separate the positive and negative examples using some separating hyperplane. How will we find the one that achieves the maximum geometric margin? We can pose the following optimization problem:

max γ,w,b     γ
s.t.     y(i)(wᵀ x(i) + b) ≥ γ, i = 1, . . . , n
||w|| = 1.

I.e., we want to maximize γ, subject to each training example having functional margin at least γ. The ||w|| = 1 constraint moreover ensures that the functional margin equals to the geometric margin, so we are also guaranteed that all the geometric margins are at least γ. Thus, solving this problem will result in (w, b) with the largest possible geometric margin with respect to the training set.

If we could solve the optimization problem above, we’d be done. But the “||w|| = 1” constraint is a nasty (non-convex) one, and this problem certainly isn’t in any format that we can plug into standard optimization software to solve. So, let’s try transforming the problem into a nicer one. Consider:

γ
max γ,w,b       ˆ
ˆ           ||w||
s.t.     y(i)(wᵀ x(i) + b) ≥ γ,
ˆ     i = 1, . . . , n

Here, we’re going to maximize γ/||w||, subject to the functional margins all being at least γ̂. Since the geometric and functional margins are related by γ = γ̂ ||w||, this will give us the answer we want. Moreover, we’ve gotten rid of the constraint ||w|| = 1 that we didn’t like. The downside is that we now have a nasty (again, non-convex) objective function; and, we still don’t have any off-the-shelf software that can solve this form of an optimization problem.

Let’s keep going. Recall our earlier discussion that we can add an arbitrary scaling constraint on w and b without changing anything. This is the key idea we’ll use now. We will introduce the scaling constraint that the functional margin of w, b with respect to the training set must be 1:

γ
ˆ = 1.





Since multiplying w and b by some constant results in the functional margin being multiplied by that same constant, this is indeed a scaling constraint, and can be satisfied by rescaling w, b. Plugging this into our problem above, and noting that maximizing γ/ˆ ||w|| = 1/||w|| is the same thing as minimizing ||w||², we now have the following optimization problem:

minw,b 1/2 ||w||₂

s.t.  y(i)(wᵀx(i) + b) ≥ 1,    i = 1, . . . , n

We’ve now transformed the problem into a form that can be efficiently solved. The above is an optimization problem with a convex quadratic objective and only linear constraints. Its solution gives us the optimal margin classifier. This optimization problem can be solved using commercial quadratic programming (QP) code.1

While we could call the problem solved here, what we will instead do is make a digression to talk about Lagrange duality. This will lead us to our optimization problem’s dual form, which will play a key role in allowing us to use kernels to get optimal margin classifiers to work efficiently in very high dimensional spaces. The dual form will also allow us to derive an efficient algorithm for solving the above optimization problem that will typically do much better than generic QP software.

# 6.5  Lagrange duality (optional reading)

Let’s temporarily put aside SVMs and maximum margin classifiers, and talk about solving constrained optimization problems. Consider a problem of the following form:

minw f(w)

s.t. hi(w) = 0,       i = 1, . . . , l.

Some of you may recall how the method of Lagrange multipliers can be used to solve it. (Don’t worry if you haven’t seen it before.) In this method, we define the Lagrangian to be

L(w, β) = f(w) + ∑i=1l βihi(w)

1You may be familiar with linear programming, which solves optimization problems that have linear objectives and linear constraints. QP software is also widely available, which allows convex quadratic objectives and linear constraints.





# 6.1 Generalized Lagrangian

Here, the βi’s are called the Lagrange multipliers. We would then find and set L’s partial derivatives to zero:

∂L = 0; ∂L = 0,
∂wi          ∂βi

and solve for w and β.

In this section, we will generalize this to constrained optimization problems in which we may have inequality as well as equality constraints. Due to time constraints, we won’t really be able to do the theory of Lagrange duality justice in this class,2 but we will give the main ideas and results, which we will then apply to our optimal margin classifier’s optimization problem.

Consider the following, which we’ll call the primal optimization problem:

minw f(w)
s.t. gi(w) ≤ 0, i = 1, . . . , k
hi(w) = 0, i = 1, . . . , l.

To solve it, we start by defining the generalized Lagrangian

L(w, α, β) = f(w) + ∑i=1k αigi(w) + ∑i=1l βihi(w).

Here, the αi’s and βi’s are the Lagrange multipliers. Consider the quantity

θP(w) = maxα,β : αi≥0 L(w, α, β).

Here, the “P” subscript stands for “primal.” Let some w be given. If w violates any of the primal constraints (i.e., if either gi(w) > 0 or hi(w) = 0 for some i), then you should be able to verify that

θP(w) = maxα,β : αi≥0 f(w) + ∑i=1k αigi(w) + ∑i=1l βihi(w) (6.1)
= ∞. (6.2)

Conversely, if the constraints are indeed satisfied for a particular value of w, then θP(w) = f(w). Hence,

θP(w) = { f(w) if w satisfies primal constraints
∞ otherwise.

2Readers interested in learning more about this topic are encouraged to read, e.g., R. T. Rockarfeller (1970), Convex Analysis, Princeton University Press.





Thus, θP takes the same value as the objective in our problem for all values of w that satisfies the primal constraints, and is positive infinity if the constraints are violated. Hence, if we consider the minimization problem

min θP(w) = minw maxα,β : αi≥0 L(w, α, β),

we see that it is the same problem (i.e., and has the same solutions as) our original, primal problem. For later use, we also define the optimal value of the objective to be p* = minw θP(w); we call this the value of the primal problem.

Now, let’s look at a slightly different problem. We define

θD(α, β) = minw L(w, α, β).

Here, the “D” subscript stands for “dual.” Note also that whereas in the definition of θP we were optimizing (maximizing) with respect to α, β, here we are minimizing with respect to w.

We can now pose the dual optimization problem:

maxα,β : αi≥0 θD(α, β) = maxα,β : αi≥0 minw L(w, α, β).

This is exactly the same as our primal problem shown above, except that the order of the “max” and the “min” are now exchanged. We also define the optimal value of the dual problem’s objective to be d* = maxα,β : αi≥0 θD(w).

How are the primal and the dual problems related? It can easily be shown that

d* = maxα,β : αi≥0 minw L(w, α, β) ≤ minw maxα,β : αi≥0 L(w, α, β) = p*.

(You should convince yourself of this; this follows from the “max min” of a function always being less than or equal to the “min max.”) However, under certain conditions, we will have

d* = p*,

so that we can solve the dual problem in lieu of the primal problem. Let’s see what these conditions are.

Suppose f and the gi’s are convex,3 and the hi’s are affine.4 Suppose further that the constraints gi are (strictly) feasible; this means that there exists some w so that gi(w) &#x3C; 0 for all i.

3 When f has a Hessian, then it is convex if and only if the Hessian is positive semi-definite. For instance, f(w) = wT w is convex; similarly, all linear (and affine) functions are also convex. (A function f can also be convex without being differentiable, but we won’t need those more general definitions of convexity here.)

4 I.e., there exists ai, bi, so that hi(w) = aiTw + bi. “Affine” means the same thing as linear, except that we also allow the extra intercept term bi.





Under our above assumptions, there must exist *w*, α*, β* so that w* is the solution to the primal problem, α*, β* are the solution to the dual problem, and moreover p* = d* = L(w*, α*, β*). Moreover, w*, α* and β** satisfy the Karush-Kuhn-Tucker (KKT) conditions, which are as follows:

- ∂*L(w*, α*, β*) / ∂wi = 0,      i = 1, . . . , d* (6.3)
- ∂*L(w*, α*, β*) / ∂βi = 0,      i = 1, . . . , l* (6.4)
- *α*gi(w*) = 0,      i = 1, . . . , k* (6.5)
- *gi(w*) ≤ 0,      i = 1, . . . , k* (6.6)
- *α* ≥ 0,      i = 1, . . . , k* (6.7)

Moreover, if some *w*, α*, β** satisfy the KKT conditions, then it is also a solution to the primal and dual problems.

We draw attention to Equation (6.5), which is called the KKT dual complementarity condition. Specifically, it implies that if *α* > 0, then gi(w*) = 0. (I.e., the “gi(w)* ≤ 0” constraint is active, meaning it holds with equality rather than with inequality.) Later on, this will be key for showing that the SVM has only a small number of “support vectors”; the KKT dual complementarity condition will also give us our convergence test when we talk about the SMO algorithm.

# 6.6 Optimal margin classifiers: the dual form

(option reading)

Note: The equivalence of optimization problem (6.8) and the optimization problem (6.12), and the relationship between the primary and dual variables in equation (6.10) are the most important take home messages of this section.

Previously, we posed the following (primal) optimization problem for finding the optimal margin classifier:

minw,b 1/2 ||*w*||2 (6.8)

s.t. *y(i)(wT x(i) + b) ≥ 1,      i = 1, . . . , n*

We can write the constraints as

*gi(w) = -y(i)(wT x(i) + b) + 1 ≤ 0.*





# 69

We have one such constraint for each training example. Note that from the KKT dual complementarity condition, we will have αi > 0 only for the training examples that have functional margin exactly equal to one (i.e., the ones corresponding to constraints that hold with equality, gi(w) = 0). Consider the figure below, in which a maximum margin separating hyperplane is shown by the solid line.

The points with the smallest margins are exactly the ones closest to the decision boundary; here, these are the three points (one negative and two positive examples) that lie on the dashed lines parallel to the decision boundary. Thus, only three of the αi’s—namely, the ones corresponding to these three training examples—will be non-zero at the optimal solution to our optimization problem. These three points are called the support vectors in this problem. The fact that the number of support vectors can be much smaller than the size of the training set will be useful later.

Let’s move on. Looking ahead, as we develop the dual form of the problem, one key idea to watch out for is that we’ll try to write our algorithm in terms of only the inner product ⟨x(i), x(j)⟩ (think of this as (x(i))T x(j)) between points in the input feature space. The fact that we can express our algorithm in terms of these inner products will be key when we apply the kernel trick.

When we construct the Lagrangian for our optimization problem we have:

L(w, b, α) = 1/2 ||w||² - ∑i=1n αi [y(i)(wT x(i) + b) - 1]. (6.9)

Note that there’re only “αi” but no “βi” Lagrange multipliers, since the problem has only inequality constraints.





# Let’s find the dual form of the problem.

To do so, we need to first minimize L(w, b, α) with respect to w and b (for fixed α), to get θD, which we’ll do by setting the derivatives of L with respect to w and b to zero. We have:

∑n
∇wL(w, b, α) = w − ∑i=1n αiy(i)x(i) = 0

This implies that

∑n
w = ∑i=1n αiy(i)x(i). (6.10)

As for the derivative with respect to b, we obtain

∂ L(w, b, α) = n ∑i=1n αiy(i) = 0. (6.11)

If we take the definition of w in Equation (6.10) and plug that back into the Lagrangian (Equation 6.9), and simplify, we get

∑i=1n αi − 1/2 ∑i,j=1n y(i)y(j)αiαj (x(i))T x(j) − b ∑i=1n αiy(i).

But from Equation (6.11), the last term must be zero, so we obtain

∑i=1n αi − 1/2 ∑i,j=1n y(i)y(j)αiαj (x(i))T x(j).

Recall that we got to the equation above by minimizing L with respect to w and b. Putting this together with the constraints αi ≥ 0 (that we always had) and the constraint (6.11), we obtain the following dual optimization problem:

maxα W(α) = ∑i=1n αi − 1/2 ∑i,j=1n y(i)y(j)αiαj ⟨x(i), x(j)⟩. (6.12)

s.t. αi ≥ 0, i = 1, . . . , n

∑i=1n αiy(i) = 0,

You should also be able to verify that the conditions required for p* = d* and the KKT conditions (Equations 6.3–6.7) to hold are indeed satisfied in





our optimization problem. Hence, we can solve the dual in lieu of solving the primal problem. Specifically, in the dual problem above, we have a maximization problem in which the parameters are the αi’s. We’ll talk later about the specific algorithm that we’re going to use to solve the dual problem, but if we are indeed able to solve it (i.e., find the α’s that maximize W (α) subject to the constraints), then we can use Equation (6.10) to go back and find the optimal w’s as a function of the α’s. Having found w*, by considering the primal problem, it is also straightforward to find the optimal value for the intercept term b as

b* = − maxi: y(i)=−1 w*ᵀ x(i) + mini: y(i)=1 w*ᵀ x(i) . (6.13)

(Check for yourself that this is correct.)

Before moving on, let’s also take a more careful look at Equation (6.10), which gives the optimal value of w in terms of (the optimal value of) α. Suppose we’ve fit our model’s parameters to a training set, and now wish to make a prediction at a new point input x. We would then calculate wᵀ x + b, and predict y = 1 if and only if this quantity is bigger than zero. But using (6.10), this quantity can also be written:

wᵀ x + b = ∑i=1n αiy(i)x(i) x + b (6.14)

= ∑i=1n αiy(i)〈x(i, x〉 + b. (6.15)

Hence, if we’ve found the αi’s, in order to make a prediction, we have to calculate a quantity that depends only on the inner product between x and the points in the training set. Moreover, we saw earlier that the αi’s will all be zero except for the support vectors. Thus, many of the terms in the sum above will be zero, and we really need to find only the inner products between x and the support vectors (of which there is often only a small number) in order calculate (6.15) and make our prediction.

By examining the dual form of the optimization problem, we gained significant insight into the structure of the problem, and were also able to write the entire algorithm in terms of only inner products between input feature vectors. In the next section, we will exploit this property to apply the kernels to our classification problem. The resulting algorithm, support vector machines, will be able to efficiently learn in very high dimensional spaces.





# 6.7 Regularization and the non-separable case

(optional reading)

The derivation of the SVM as presented so far assumed that the data is linearly separable. While mapping data to a high dimensional feature space via φ does generally increase the likelihood that the data is separable, we can’t guarantee that it always will be so. Also, in some cases it is not clear that finding a separating hyperplane is exactly what we’d want to do, since that might be susceptible to outliers. For instance, the left figure below shows an optimal margin classifier, and when a single outlier is added in the upper-left region (right figure), it causes the decision boundary to make a dramatic swing, and the resulting classifier has a much smaller margin.

To make the algorithm work for non-linearly separable datasets as well as be less sensitive to outliers, we reformulate our optimization (using `₁ regularization) as follows:

∑
minγ,w,b  1 ||w||₂ + C n   ξi
2           i=1
s.t.      y(i)(wᵀ x(i) + b) ≥ 1 − ξi,           i = 1, . . . , n
ξi ≥ 0,          i = 1, . . . , n.

Thus, examples are now permitted to have (functional) margin less than 1, and if an example has functional margin 1 − ξi (with ξ > 0), we would pay a cost of the objective function being increased by Cξi. The parameter C controls the relative weighting between the twin goals of making the ||w||² small (which we saw earlier makes the margin large) and of ensuring that most examples have functional margin at least 1.





As before, we can form the Lagrangian:

L(w, b, ξ, α, r) = ∑ ∑ ∑
1 wᵀ w + C n ξi − n αi [y(i)(xᵀ w + b) − 1 + ξi] − n riξi.

Here, the αi’s and ri’s are our Lagrange multipliers (constrained to be ≥ 0). We won’t go through the derivation of the dual again in detail, but after setting the derivatives with respect to w and b to zero as before, substituting them back in, and simplifying, we obtain the following dual form of the problem:

maxα W (α) = ∑ n αi − 1 n y(i)y(ʲ)αiαj 〈x(i), x(ʲ)〉

s.t. 0 ≤ αi ≤ C, i = 1, . . . , n

∑ n αiy(i) = 0,

As before, we also have that w can be expressed in terms of the αi’s as given in Equation (6.10), so that after solving the dual problem, we can continue to use Equation (6.15) to make our predictions. Note that, somewhat surprisingly, in adding `₁ regularization, the only change to the dual problem is that what was originally a constraint that 0 ≤ αi has now become 0 ≤ αi ≤ C. The calculation for b∗ also has to be modified (Equation 6.13 is no longer valid); see the comments in the next section/Platt’s paper.

Also, the KKT dual-complementarity conditions (which in the next section will be useful for testing for the convergence of the SMO algorithm) are:

αi = 0 ⇒ y(i)(wᵀ x(i) + b) ≥ 1 (6.16)

αi = C ⇒ y(i)(wᵀ x(i) + b) ≤ 1 (6.17)

0 &#x3C; αi &#x3C; C ⇒ y(i)(wᵀ x(i) + b) = 1. (6.18)

Now, all that remains is to give an algorithm for actually solving the dual problem, which we will do in the next section.

# 6.8 The SMO algorithm (optional reading)

The SMO (sequential minimal optimization) algorithm, due to John Platt, gives an efficient way of solving the dual problem arising from the derivation.





# 6.8.1 Coordinate ascent

Consider trying to solve the unconstrained optimization problem

max W (α₁, α₂, . . . , αₙ).

α

Here, we think of W as just some function of the parameters αi’s, and for now ignore any relationship between this problem and SVMs. We’ve already seen two optimization algorithms, gradient ascent and Newton’s method. The new algorithm we’re going to consider here is called coordinate ascent:

Loop until convergence: {
For i = 1, . . . , n, {
α := arg max          W (α , . . . , α  , α
i                  α                  ˆ , α   , . . . , α  ).
ˆi   1    i−1         i  i+1            n
}
}

Thus, in the innermost loop of this algorithm, we will hold all the variables except for some αi fixed, and reoptimize W with respect to just the parameter αi. In the version of this method presented here, the inner-loop reoptimizes the variables in order α₁, α₂, . . . , αₙ, α₁, α₂, . . .. (A more sophisticated version might choose other orderings; for instance, we may choose the next variable to update according to which one we expect to allow us to make the largest increase in W (α).)

When the function W happens to be of such a form that the “arg max” in the inner loop can be performed efficiently, then coordinate ascent can be a fairly efficient algorithm. Here’s a picture of coordinate ascent in action:





The ellipses in the figure are the contours of a quadratic function that we want to optimize. Coordinate ascent was initialized at (2, −2), and also plotted in the figure is the path that it took on its way to the global maximum. Notice that on each step, coordinate ascent takes a step that’s parallel to one of the axes, since only one variable is being optimized at a time.

# 6.8.2 SMO

We close off the discussion of SVMs by sketching the derivation of the SMO algorithm. Here’s the (dual) optimization problem that we want to solve:

maxα W (α) = n ∑ αi − 1/2 ∑ n y(i)y(j)αiαj 〈x(i), x(j)〉. (6.19)

s.t. 0 ≤ αi ≤ C, i = 1, . . . , n (6.20)

∑ n αiy(i) = 0. (6.21)

Let’s say we have set of αi’s that satisfy the constraints (6.20-6.21). Now, suppose we want to hold α₂, . . . , αₙ fixed, and take a coordinate ascent step and reoptimize the objective with respect to α₁. Can we make any progress? The answer is no, because the constraint (6.21) ensures that

∑ n α₁y(1) = − ∑ αiy(i).

i=2





Or, by multiplying both sides by y(1), we equivalently have

∑n
α₁ = −y(1) αiy(i).
i=2

(This step used the fact that y(1) ∈ {−1, 1}, and hence (y(1))² = 1.) Hence, α₁ is exactly determined by the other αi’s, and if we were to hold α₂, . . . , αₙ fixed, then we can’t make any change to α₁ without violating the constraint (6.21) in the optimization problem.

Thus, if we want to update some subject of the αi’s, we must update at least two of them simultaneously in order to keep satisfying the constraints. This motivates the SMO algorithm, which simply does the following:

1. Repeat till convergence {
2. Select some pair αi and αj to update next (using a heuristic that tries to pick the two that will allow us to make the biggest progress towards the global maximum).
3. Reoptimize W (α) with respect to αi and αj, while holding all the other αk’s (k = i, j) fixed.

}

To test for convergence of this algorithm, we can check whether the KKT conditions (Equations 6.16-6.18) are satisfied to within some tol. Here, tol is the convergence tolerance parameter, and is typically set to around 0.01 to 0.001. (See the paper and pseudocode for details.)

The key reason that SMO is an efficient algorithm is that the update to αi, αj can be computed very efficiently. Let’s now briefly sketch the main ideas for deriving the efficient update.

Let’s say we currently have some setting of the αi’s that satisfy the constraints (6.20-6.21), and suppose we’ve decided to hold α₃, . . . , αₙ fixed, and want to reoptimize W (α₁, α₂, . . . , αₙ) with respect to α₁ and α₂ (subject to the constraints). From (6.21), we require that

∑n
α₁y(1) + α₂y(2) = − αiy(i).
i=3

Since the right hand side is fixed (as we’ve fixed α₃, . . . αₙ), we can just let it be denoted by some constant ζ :

α₁y(1) + α₂y(2) = ζ. (6.22)

We can thus picture the constraints on α₁ and α₂ as follows:





77

C

H       α₁y(1)+α₂y(2)=ζ

α₂

L

α₁       C

From the constraints (6.20), we know that α₁ and α₂ must lie within the box [0, C ] × [0, C ] shown. Also plotted is the line α₁y(1) + α₂y(2) = ζ , on which we know α₁ and α₂ must lie. Note also that, from these constraints, we know L ≤ α₂ ≤ H; otherwise, (α₁, α₂) can’t simultaneously satisfy both the box and the straight line constraint. In this example, L = 0. But depending on what the line α₁y(1) + α₂y(2) = ζ looks like, this won’t always necessarily be the case; but more generally, there will be some lower-bound L and some upper-bound H on the permissible values for α₂ that will ensure that α₁, α₂ lie within the box [0, C ] × [0, C ].

Using Equation (6.22), we can also write α₁ as a function of α₂:

α₁ = (ζ − α₂y(2))y(1).

(Check this derivation yourself; we again used the fact that y(1) ∈ {−1, 1} so that (y(1))² = 1.) Hence, the objective W (α) can be written

W (α₁, α₂, . . . , αₙ) = W ((ζ − α₂y(2))y(1), α₂, . . . , αₙ).

Treating α₃, . . . , αₙ as constants, you should be able to verify that this is just some quadratic function in α₂. I.e., this can also be expressed in the form aα² + bα₂ + c for some appropriate a, b, and c. If we ignore the “box” constraints (6.20) (or, equivalently, that L ≤ α₂ ≤ H), then we can easily maximize this quadratic function by setting its derivative to zero and solving. We’ll let αⁿew,unclipped denote the resulting value of α₂. You should also be able to convince yourself that if we had instead wanted to maximize W with respect to α₂ but subject to the box constraint, then we can find the resulting value optimal simply by taking αⁿew,unclipped and “clipping” it to lie in the




Finally, having found the αnew, we can use Equation (6.22) to go back and find the optimal value of α1ew.

There’re a couple more details that are quite easy but that we’ll leave you to read about yourself in Platt’s paper: One is the choice of the heuristics used to select the next αi, αj to update; the other is how to update b as the SMO algorithm is run.

| \[L, H] interval, to get |                |                            |
| ------------------------ | -------------- | -------------------------- |
| αnew =                   | H              | if αnew,unclipped > H      |
|                          | αnew,unclipped | if L2 ≤ αnew,unclipped ≤ H |
| 2                        |                | 2                          |
| αnew,unclipped           | L              | if αnew,unclipped < L      |
|                          | 2              |                            |



# Part II

# Deep learning

79



# Chapter 7

# Deep learning

We now begin our study of deep learning. In this set of notes, we give an overview of neural networks, discuss vectorization and discuss training neural networks with backpropagation.

# 7.1 Supervised learning with non-linear models

In the supervised learning setting (predicting y from the input x), suppose our model/hypothesis is hθ(x). In the past lectures, we have considered the cases when hθ(x) = θTx (in linear regression) or hθ(x) = θTφ(x) (where φ(x) is the feature map). A commonality of these two models is that they are linear in the parameters θ. Next we will consider learning general family of models that are non-linear in both the parameters θ and the inputs x. The most common non-linear models are neural networks, which we will define starting from the next section. For this section, it suffices to think hθ(x) as an abstract non-linear model.1

Suppose {(x(i), y(i))}i=1n are the training examples. We will define the nonlinear model and the loss/cost function for learning it.

# Regression problems.

For simplicity, we start with the case where the output is a real number, that is, y(i) ∈ R, and thus the model hθ also outputs a real number hθ(x) ∈ R. We define the least square cost function for the

1If a concrete example is helpful, perhaps think about the model hθ(x) = θ2x2 + θ2x2 + · · · + θ2x2 in this subsection, even though it’s not a neural network.





# 7. Cost Functions

i-th example (x(i), y(i)) as

J (i)(θ) = 1 (hθ(x(i)) − y(i))² , (7.1)

and define the mean-square cost function for the dataset as

J (θ) = 1/n ∑ J (i)(θ) , (7.2)

which is same as in linear regression except that we introduce a constant 1/n in front of the cost function to be consistent with the convention. Note that multiplying the cost function with a scalar will not change the local minima or global minima of the cost function. Also note that the underlying parameterization for hθ(x) is different from the case of linear regression, even though the form of the cost function is the same mean-squared loss. Throughout the notes, we use the words “loss” and “cost” interchangeably.

# Binary classification

Next we define the model and loss function for binary classification. Suppose the inputs x ∈ Rᵈ.

Let hθ : R → R be a parameterized model (the analog of θ>x in logistic linear regression). We call the output hθ(x) ∈ R the logit. Analogous to Section 2.1, we use the logistic function g(·) to turn the logit hθ(x) to a probability hθ(x) ∈ [0, 1]:

h (x) = g(hθ(x)) = 1/(1 + exp(−hθ(x)) . (7.3)

We model the conditional distribution of y given x and θ by

P (y = 1 | x; θ) = hθ(x)

P (y = 0 | x; θ) = 1 − hθ(x)

Following the same derivation in Section 2.1 and using the derivation in Remark 2.1.1, the negative likelihood loss function is equal to:

J (i)(θ) = − log p(y(i) | x(i); θ) = logistic(hθ(x(i)), y(i)) (7.4)

As done in equation (7.2), the total loss function is also defined as the average of the loss function over individual training examples, J (θ) = 1/n ∑ J (i)(θ).




# Multi-class classification.

Following Section 2.3, we consider a classification problem where the response variable y can take on any one of k values, i.e. y ∈ {1, 2, . . . , k}. Let hθ : R → R be a parameterized model. We call the outputs hθ(x) ∈ R the logits. Each logit corresponds to the prediction for one of the k classes. Analogous to Section 2.3, we use the softmax function to turn the logits hθ(x) into a probability vector with non-negative entries that sum up to 1:

P(y = j | x; θ) = \frac{exp(hθ(x)j)}{\sum_{s=1}^{k} exp(hθ(x)s)}, (7.5)

where hθ(x)s denotes the s-th coordinate of hθ(x).

Similarly to Section 2.3, the loss function for a single training example (x(i), y(i)) is its negative log-likelihood:

J(i)(θ) = -log P(y(i) | x(i); θ) = -log \frac{exp(hθ(x(i))y(i)}}{\sum_{s=1}^{k} exp(hθ(x(i))s)}, (7.6)

Using the notations of Section 2.3, we can simply write in an abstract way:

J(i)(θ) = ` ce(hθ(x(i)), y(i)). (7.7)

The loss function is also defined as the average of the loss function of individual training examples, J(θ) = \frac{1}{n} \sum_{i=1}^{n} J(i)(θ).

We also note that the approach above can also be generated to any conditional probabilistic model where we have an exponential distribution for y, Exponential-family(y; η), where η = hθ(x) is a parameterized nonlinear function of x. However, the most widely used situations are the three cases discussed above.

# Optimizers (SGD).

Commonly, people use gradient descent (GD), stochastic gradient (SGD), or their variants to optimize the loss function J(θ). GD’s update rule can be written as:

θ := θ - α∇θJ(θ) (7.8)

where α > 0 is often referred to as the learning rate or step size. Next, we introduce a version of the SGD (Algorithm 1), which is lightly different from that in the first lecture notes.

Recall that, as defined in the previous lecture notes, we use the notation “a := b” to denote an operation (in a computer program) in which we set the value of a variable a to be equal to the value of b. In other words, this operation overwrites a with the value of b. In contrast, we will write “a = b” when we are asserting a statement of fact, that the value of a is equal to the value of b.



# 83

# Algorithm 1 Stochastic Gradient Descent

1. Hyperparameter: learning rate α, number of total iteration niter.
2. Initialize θ randomly.
3. for i = 1 to niter do
4. Sample j uniformly from {1, . . . , n}, and update θ by

θ := θ − α∇θJ (ʲ)(θ) (7.9)

Oftentimes computing the gradient of B examples simultaneously for the parameter θ can be faster than computing B gradients separately due to hardware parallelization. Therefore, a mini-batch version of SGD is most commonly used in deep learning, as shown in Algorithm 2. There are also other variants of the SGD or mini-batch SGD with slightly different sampling schemes.

# Algorithm 2 Mini-batch Stochastic Gradient Descent

1. Hyperparameters: learning rate α, batch size B, # iterations niter.
2. Initialize θ randomly
3. for i = 1 to niter do
4. Sample B examples j₁, . . . , jB (without replacement) uniformly from {1, . . . , n}, and update θ by

θ := θ − α ∑k=1B ∇θJ (ʲₖ)(θ) (7.10)

With these generic algorithms, a typical deep learning model is learned with the following steps. 1. Define a neural network parametrization hθ(x), which we will introduce in Section 7.2, and 2. write the backpropagation algorithm to compute the gradient of the loss function J (ʲ)(θ) efficiently, which will be covered in Section 7.4, and 3. run SGD or mini-batch SGD (or other gradient-based optimizers) with the loss function J (θ).




# 7.2 Neural networks

Neural networks refer to a broad type of non-linear models/parametrizations hθ(x) that involve combinations of matrix multiplications and other entry-wise non-linear operations. To have a unified treatment for regression problem and classification problem, here we consider hθ(x) as the output of the neural network. For regression problem, the final prediction hθ(x) and for classification problem, hθ(x) is the logits and the predicted probability will be hθ(x) = 1/(1+exp(−hθ(x))) (see equation 7.3) for binary classification or hθ(x) = softmax(hθ(x)) for multi-class classification (see equation 7.5). We will start small and slowly build up a neural network, step by step.

# A Neural Network with a Single Neuron

Recall the housing price prediction problem from before: given the size of the house, we want to predict the price. We will use it as a running example in this subsection. Previously, we fit a straight line to the graph of size vs. housing price. Now, instead of fitting a straight line, we wish to prevent negative housing prices by setting the absolute minimum price as zero. This produces a “kink” in the graph as shown in Figure 7.1. How do we represent such a function with a single kink as hθ(x) with unknown parameter? (After doing so, we can invoke the machinery in Section 7.1.)

We define a parameterized function hθ(x) with input x, parameterized by θ, which outputs the price of the house y. Formally, hθ : x → y. Perhaps one of the simplest parametrization would be hθ(x) = max(wx + b, 0), where θ = (w, b) ∈ R2 (7.11).

Here hθ(x) returns a single value: (wx+b) or zero, whichever is greater. In the context of neural networks, the function max{t, 0} is called a ReLU (pronounced “ray-lu”), or rectified linear unit, and often denoted by ReLU(t) = max{t, 0}.

Generally, a one-dimensional non-linear function that maps R to R such as ReLU is often referred to as an activation function. The model hθ(x) is said to have a single neuron partly because it has a single non-linear activation function. (We will discuss more about why a non-linear activation is called neuron.)

When the input x ∈ Rd has multiple dimensions, a neural network with a single neuron can be written as hθ(x) = ReLU(wTx + b), where w ∈ Rd, b ∈ R, and θ = (w, b) (7.12).



# 7.1 Housing Prices

1000
900
800
700
price (in $1000)
600
500
400
300
200
100
0

500     1000   1500  2000     2500  3000  3500  4000  4500  5000

square feet

Figure 7.1: Housing prices with a “kink” in the graph.

The term b is often referred to as the “bias”, and the vector w is referred to as the weight vector. Such a neural network has 1 layer. (We will define what multiple layers mean in the sequel.)

# Stacking Neurons

A more complex neural network may take the single neuron described above and “stack” them together such that one neuron passes its output as input into the next neuron, resulting in a more complex function.

Let us now deepen the housing prediction example. In addition to the size of the house, suppose that you know the number of bedrooms, the zip code and the wealth of the neighborhood. Building neural networks is analogous to Lego bricks: you take individual bricks and stack them together to build complex structures. The same applies to neural networks: we take individual neurons and stack them together to create complex neural networks.

Given these features (size, number of bedrooms, zip code, and wealth), we might then decide that the price of the house depends on the maximum family size it can accommodate. Suppose the family size is a function of the size of the house and number of bedrooms (see Figure 7.2). The zip code may provide additional information such as how walkable the neighborhood is (i.e., can you walk to the grocery store or do you need to drive everywhere). Combining the zip code with the wealth of the neighborhood may predict the quality of the local elementary school. Given these three derived features (family size, walkable, school quality), we may conclude that the price of the house depends on these factors.





# 7.2 Neural Networks for Predicting Housing Prices

home ultimately depends on these three features.

| Size     | # Bedrooms | Family Size    | Price |
| -------- | ---------- | -------------- | ----- |
| Walkable |            | y              |       |
| Zip Code |            | School Quality |       |
| Wealth   |            |                |       |

Figure 7.2: Diagram of a small neural network for predicting housing prices.

Formally, the input to a neural network is a set of input features x₁, x₂, x₃, x₄. We denote the intermediate variables for “family size”, “walkable”, and “school quality” by a₁, a₂, a₃ (these ai’s are often referred to as “hidden units” or “hidden neurons”). We represent each of the ai’s as a neural network with a single neuron with a subset of x₁, . . . , x₄ as inputs. Then as in Figure 7.1, we will have the parameterization:

a₁ = ReLU(θ₁x₁ + θ₂x₂ + θ₃)

a₂ = ReLU(θ₄x₃ + θ₅)

a₃ = ReLU(θ₆x₃ + θ₇x₄ + θ₈)

where (θ₁, · · · , θ₈) are parameters. Now we represent the final output hθ(x) as another linear function with a₁, a₂, a₃ as inputs, and we get

hθ(x) = θ₉a₁ + θ₁₀a₂ + θ₁₁a₃ + θ₁₂ (7.13)

where θ contains all the parameters (θ₁, · · · , θ₁₂).

Now we represent the output as a quite complex function of x with parameters θ. Then you can use this parametrization hθ with the machinery of Section 7.1 to learn the parameters θ.

# Inspiration from Biological Neural Networks

As the name suggests, artificial neural networks were inspired by biological neural networks. The hidden units a₁, . . . , aₘ correspond to the neurons in a biological neural network, and the parameters θi’s correspond to the synapses. However, it’s unclear how similar the modern deep artificial neural networks are to the biological ones. For example, perhaps not many neuroscientists think biological

3Typically, for multi-layer neural network, at the end, near the output, we don’t apply ReLU, especially when the output is not necessarily a positive number.





neural networks could have 1000 layers, while some modern artificial neural networks do (we will elaborate more on the notion of layers.) Moreover, it’s an open question whether human brains update their neural networks in a way similar to the way that computer scientists learn artificial neural networks (using backpropagation, which we will introduce in the next section).

# Two-layer Fully-Connected Neural Networks.

We constructed the neural network in equation (7.13) using a significant amount of prior knowledge/belief about how the “family size”, “walkable”, and “school quality” are determined by the inputs. We implicitly assumed that we know the family size is an important quantity to look at and that it can be determined by only the “size” and “# bedrooms”. Such a prior knowledge might not be available for other applications. It would be more flexible and general to have a generic parameterization. A simple way would be to write the intermediate variable a₁ as a function of all x₁, . . . , x₄:

a₁ = ReLU(w>x + b₁),  where w₁ ∈ R⁴ and b₁ ∈ R               (7.14)
a₂ = ReLU(w>x + b₂),  where w₂ ∈ R⁴ and b₂ ∈ R
a₃ = ReLU(w>x + b₃),  where w₃ ∈ R⁴ and b₃ ∈ R

We still define ¯hθ(x) using equation (7.13) with a₁, a₂, a₃ being defined as above. Thus we have a so-called fully-connected neural network because all the intermediate variables ai’s depend on all the inputs xi’s.

For full generality, a two-layer fully-connected neural network with m hidden units and d dimensional input x ∈ Rᵈ is defined as

∀j ∈ [1, ..., m],  zj  = w[1]>x + b[1] where w[1] ∈ Rᵈ, b[1] ∈ R     (7.15)
j            j       j    j
aj  = ReLU(zj ),
a = [a₁, . . . , aₘ]> ∈ Rᵐ
¯            [2]>  [2]            [2]  m [2]
hθ(x) = w          a + b       where w  ∈ R , b  ∈ R,     (7.16)

Note that by default the vectors in Rᵈ are viewed as column vectors, and in particular a is a column vector with components a₁, a₂, ..., aₘ. The indices [1] and [2] are used to distinguish two sets of parameters: the w[1]’s (each of which is a vector in Rᵈ) and w[2] (which is a vector in Rᵐ). We will have more of these later.

# Vectorization.

Before we introduce neural networks with more layers and more complex structures, we will simplify the expressions for neural networks.





with more matrix and vector notations. Another important motivation of vectorization is the speed perspective in the implementation. In order to implement a neural network efficiently, one must be careful when using for loops. The most natural way to implement equation (7.15) in code is perhaps to use a for loop. In practice, the dimensionalities of the inputs and hidden units are high. As a result, code will run very slowly if you use for loops. Leveraging the parallelism in GPUs is/was crucial for the progress of deep learning.

This gave rise to vectorization. Instead of using for loops, vectorization takes advantage of matrix algebra and highly optimized numerical linear algebra packages (e.g., BLAS) to make neural network computations run quickly. Before the deep learning era, a for loop may have been sufficient on smaller datasets, but modern deep networks and state-of-the-art datasets will be infeasible to run with for loops.

We vectorize the two-layer fully-connected neural network as below. We define a weight matrix W[1] in Rm×d as the concatenation of all the vectors w[1]’s in the following way:

W[1] =

— w[1]1 —

— w[1]2 —

.

.

.

— w[1]m —

Now by the definition of matrix vector multiplication, we can write z = [z1, . . . , zm]T ∈ Rm as

z =

z1

.

.

.

.

zm

=

— w[1]1 —

— w[1]2 —

.

.

.

— w[1]m —

Or succinctly,

z = W[1]x + b[1] (7.19)

We remark again that a vector in Rd in this notes, following the conventions previously established, is automatically viewed as a column vector, and can





also be viewed as a d × 1 dimensional matrix. (Note that this is different from numpy where a vector is viewed as a row vector in broadcasting.)

Computing the activations a ∈ Rm from z ∈ Rm involves an element-wise non-linear application of the ReLU function, which can be computed in parallel efficiently. Overloading ReLU for element-wise application of ReLU (meaning, for a vector t ∈ Rd, ReLU(t) is a vector such that ReLU(t)i = ReLU(ti)), we have

a = ReLU(z) (7.20)

Define W[2] = [w[2]]T ∈ R1×m similarly. Then, the model in equation (7.16) can be summarized as

a = ReLU(W[1]x + b[1])

hθ(x) = W[2]a + b[2] (7.21)

Here θ consists of W[1], W[2] (often referred to as the weight matrices) and b[1], b[2] (referred to as the biases). The collection of W[1], b[1] is referred to as the first layer, and W[2], b[2] the second layer. The activation a is referred to as the hidden layer. A two-layer neural network is also called one-hidden-layer neural network.

# Multi-layer fully-connected neural networks.

With this succinct notations, we can stack more layers to get a deeper fully-connected neural network. Let r be the number of layers (weight matrices). Let W[1], . . . , W[r], b[1], . . . , b[r] be the weight matrices and biases of all the layers. Then a multi-layer neural network can be written as

a[1] = ReLU(W[1]x + b[1])

a[2] = ReLU(W[2]a[1] + b[2])

· · ·

a[r−1] = ReLU(W[r−1]a[r−2] + b[r−1])

hθ(x) = W[r]a[r−1] + b[r] (7.22)

We note that the weight matrices and biases need to have compatible dimensions for the equations above to make sense. If a[k] has dimension mk, then the weight matrix W[k] should be of dimension mk × mk−1, and the bias b[k] ∈ Rmk. Moreover, W[1] ∈ Rm1×d and W[r] ∈ R1×mr−1.




The total number of neurons in the network is m₁ + · · · + mᵣ, and the total number of parameters in this network is (d + 1)m₁ + (m₁ + 1)m₂ + · · · + (mᵣ−1 + 1)mᵣ. Sometimes for notational consistency we also write a[0] = x, and a[r] = hθ(x). Then we have simple recursion that

a[ᵏ] = ReLU(W [ᵏ]a[ᵏ−1] + b[ᵏ]), ∀k = 1, . . . , r − 1 (7.23)

Note that this would have be true for k = r if there were an additional ReLU in equation (7.22), but often people like to make the last layer linear (aka without a ReLU) so that negative outputs are possible and it’s easier to interpret the last layer as a linear model. (More on the interpretability at the “connection to kernel method” paragraph of this section.)

# Other activation functions.

The activation function ReLU can be replaced by many other non-linear function σ(·) that maps R to R such as

σ(z) = 1 (sigmoid) (7.24)

1 + e−z

σ(z) = ez − e−z (tanh) (7.25)

ez + e−z

σ(z) = max{z, γz}, γ ∈ (0, 1) (leaky ReLU) (7.26)

z [ z ]

σ(z) = 21 + erf(√2) (GELU) (7.27)

σ(z) = 1 log(1 + exp(βz)), β > 0 (Softplus) (7.28)

β

The activation functions are plotted in Figure 7.3. Sigmoid and tanh are less and less used these days partly because their are bounded from both sides and the gradient of them vanishes as z goes to both positive and negative infinity (whereas all the other activation functions still have gradients as the input goes to positive infinity.) Softplus is not used very often either in practice and can be viewed as a smoothing of the ReLU so that it has a proper second order derivative. GELU and leaky ReLU are both variants of ReLU but they have some non-zero gradient even when the input is negative. GELU (or its slight variant) is used in NLP models such as BERT and GPT (which we will discuss in Chapter 14.)

# Why do we not use the identity function for σ(z)?

That is, why not use σ(z) = z? Assume for sake of argument that b[1] and b[2] are zeros.



# 4 ReLU

| 4  | ReLU                |
| -- | ------------------- |
| 3  | sigmoid             |
| 2  | tanh                |
| 1  | leaky ReLU, y = 0.3 |
| 0  | GELU                |
| -1 | Softplus, β = 1     |

Suppose σ(z) = z, then for two-layer neural network, we have that



92

suits the particular applications. The process of choosing the feature maps is often referred to as feature engineering.

We can view deep learning as a way to automatically learn the right feature map (sometimes also referred to as “the representation”) as follows. Suppose we denote by β the collection of the parameters in a fully-connected neural networks (equation (7.22)) except those in the last layer. Then we can abstract right ar−1 as a function of the input x and the parameters in β: ar−1 = φβ(x). Now we can write the model as

hθ(x) = W[r] φβ(x) + b[r] (7.34)

When β is fixed, then φ(·) can viewed as a feature map, and therefore hθ(x) is just a linear model over the features φβ(x). However, we will train the neural networks, both the parameters in β and the parameters W[r], b[r] are optimized, and therefore we are not learning a linear model in the feature space, but also learning a good feature map φβ(·) itself so that it’s possible to predict accurately with a linear model on top of the feature map. Therefore, deep learning tends to depend less on the domain knowledge of the particular applications and requires often less feature engineering. The penultimate layer ar is often (informally) referred to as the learned features or representations in the context of deep learning.

In the example of house price prediction, a fully-connected neural network does not need us to specify the intermediate quantity such “family size”, and may automatically discover some useful features in the last penultimate layer (the activation ar−1), and use them to linearly predict the housing price. Often the feature map / representation obtained from one datasets (that is, the function φβ(·) can be also useful for other datasets, which indicates they contain essential information about the data. However, oftentimes, the neural network will discover complex features which are very useful for predicting the output but may be difficult for a human to understand or interpret. This is why some people refer to neural networks as a black box, as it can be difficult to understand the features it has discovered.

# 7.3  Modules in Modern Neural Networks

The multi-layer neural network introduced in equation (7.22) of Section 7.2 is often called multi-layer perceptron (MLP) these days. Modern neural networks used in practice are often much more complex and consist of multiple building blocks or multiple layers of building blocks. In this section, we will




introduce some of the other building blocks and discuss possible ways to combine them.

First, each matrix multiplication can be viewed as a building block. Consider a matrix multiplication operation with parameters (W, b) where W is the weight matrix and b is the bias vector, operating on an input z,

MMW,b(z) = W z + b .

Note that we implicitly assume all the dimensions are chosen to be compatible. We will also drop the subscripts under MM when they are clear in the context or just for convenience when they are not essential to the discussion.

Then, the MLP can be written as a composition of multiple matrix multiplication modules and nonlinear activation modules (which can also be viewed as a building block):

MLP(x) = MMW [ᵣ],b[ᵣ] (σ(MMW [ᵣ−1],bᵣ−1))).

Alternatively, when we drop the subscripts that indicate the parameters for convenience, we can write

MLP(x) = MM(σ(MMσ(· · · MM(x))).

Note that in this lecture notes, by default, all the modules have different sets of parameters, and the dimensions of the parameters are chosen such that the composition is meaningful.

Larger modules can be defined via smaller modules as well, e.g., one activation layer σ and a matrix multiplication layer MM are often combined and called a “layer” in many papers. People often draw the architecture with the basic modules in a figure by indicating the dependency between these modules. E.g., see an illustration of an MLP in Figure 7.4, Left.

# Residual connections.

One of the very influential neural network architecture for vision application is ResNet, which uses the residual connections that are essentially used in almost all large-scale deep learning architectures these days. Using our notation above, a very much simplified residual block can be defined as

Res(z) = z + σ(MM(σ(MM(z))).

A much simplified ResNet is a composition of many residual blocks followed by a matrix multiplication,

ResNet-S(x) = MM(Res(Res(· · · Res(x)))).



# Figure 7.4: Illustrative Figures for Architecture.

Left: An MLP with r layers. Right: A residual network.

We also draw the dependency of these modules in Figure 7.4, Right. We note that the ResNet-S is still not the same as the ResNet architecture introduced in the seminal paper [He et al., 2016] because ResNet uses convolution layers instead of vanilla matrix multiplication, and adds batch normalization between convolutions and activations. We will introduce convolutional layers and some variants of batch normalization below. ResNet-S and layer normalization are part of the Transformer architecture that are widely used in modern large language models.

# Layer normalization.

Layer normalization, denoted by LN in this text, is a module that maps a vector z ∈ Rm to a more normalized vector LN(z) ∈ Rm. It is oftentimes used after the nonlinear activations.

We first define a sub-module of the layer normalization, denoted by LN-S.

LN-S(z) =

| z - μ | 1         | ˆ               |
| ----- | --------- | --------------- |
| σ     | z ˆ       | - μ             |
| 2σ ˆ  | LN-S(z) = | z - μ           |
| m ˆ   | σ         | ˆ               |
| ∑m z  | √         | ~~∑~~m (z - μ2) |

where μˆ = ∑i=1m zi / m is the empirical mean of the vector z and σˆ = √(∑i=1m (zi - μˆ)2 / m) is the empirical standard deviation of the entries of z.4 Intuitively, LN-S(z) is a vector that is normalized to having empirical mean zero and empirical standard deviation 1.

Note that we divide by m instead of m - 1 in the empirical standard deviation here because we are interested in making the output of LN-S(z) have sum of squares equal to 1 (as opposed to estimating the standard deviation in statistics.)





Oftentimes zero mean and standard deviation 1 is not the most desired normalization scheme, and thus layernorm introduces to parameters learnable scalars β and γ as the desired mean and standard deviation, and use an affine transformation to turn the output of LN-S(z) into a vector with mean β and standard deviation γ.

LN(z) = β + γ · LN-S(z) =
           ( z  −μ ) 
β + γ (     1σ ˆ )
                z ˆ      
−μ
 β + γ          2σ ˆ     
                         .    ˆ        
           .             
.            )
( z    −μ
β + γ       m      ˆ
σ
ˆ

Here the first occurrence of β should be technically interpreted as a vector with all the entries being β. We also note that μ ˆ and σ ˆ are also functions of z and shouldn’t be treated as constants when computing the derivatives of layernorm. Moreover, β and γ are learnable parameters and thus layernorm is a parameterized module (as opposed to the activation layer which doesn’t have any parameters.)

# Scaling-invariant property.

One important property of layer normalization is that it will make the model invariant to scaling of the parameters in the following sense. Suppose we consider composing LN with MMW,b and get a subnetwork LN(MMW,b(z)). Then, we have that the output of this sub-network does not change when the parameter in MMW,b is scaled:

LN(MMαW,αb(z)) = LN(MMW,b(z)), ∀α > 0.

To see this, we first know that LN-S(·) is scale-invariant

LN-S(αz) =
                      
αz      −αμ
1         ˆ
ασ
ˆ
 αz −αμ 
     ˆ     
       2              
 2     ˆ 
LN-S(αz) =          .                 
=     ˆ      = LN-S(z).
    .                 
.                          .
     .     

Then we have

LN(MMαW,αb(z)) = β + γLN-S(MMαW,αb(z))

= β + γLN-S(αMMW,b(z))

= β + γLN-S(MMW,b(z))

= LN(MMW,b(z)).

Due to this property, most of the modern DL architectures for large-scale computer vision and language applications have the following scale-invariant





property w.r.t all the weights that are not at the last layer. Suppose the network f has last layer’s weights Wlast, and all the rest of the weights are denote by W. Then, we have fWlast,αW(x) = fWlast,W(x) for all α > 0. Here, the last layer's weights are special because there are typically no layernorm or batchnorm after the last layer’s weights.

# Other normalization layers.

There are several other normalization layers that aim to normalize the intermediate layers of the neural networks to a more fixed and controllable scaling, such as batch-normalization [?], and group normalization [?]. Batch normalization and group normalization are more often used in computer vision applications whereas layer norm is used more often in language applications.

# Convolutional Layers.

Convolutional Neural Networks are neural networks that consist of convolution layers (and many other modules), and are particularly useful for computer vision applications. For the simplicity of exposition, we focus on 1-D convolution in this text and only briefly mention 2-D convolution informally at the end of this subsection. (2-D convolution is more suitable for images which have two dimensions. 1-D convolution is also used in natural language processing.)

We start by introducing a simplified version of the 1-D convolution layer, denoted by Conv1D-S(·) which is a type of matrix multiplication layer with a special structure. The parameters of Conv1D-S are a filter vector w ∈ Rk where k is called the filter size (oftentimes k &#x3C; m), and a bias scalar b. Oftentimes the filter is also called a kernel (but it does not have much to do with the kernel in kernel method.) For simplicity, we assume k = 2` + 1 is an odd number. We first pad zeros to the input vector z in the sense that we let z1−` = z1−`+1 = .. = z0 = 0 and zm+1 = zm+2 = .. = zm+` = 0, and treat z as an (m + 2`)-dimension vector. Conv1D-S outputs a vector of dimension Rm where each output dimension is a linear combination of subsets of zj’s with coefficients from w,

∑j=12`+1 Conv1D-S(z)i = w1zi−` + w2zi−`+1 + · · · + w2`+1zi+` = wjzi−`+(j−1).

(7.48)

Therefore, one can view Conv1D-S as a matrix multiplication with shared





97

parameters: Conv1D-S(z) = Qz, where
 w`+1           · · ·    w2`+1     0          0        · · ·     · · ·     · · ·     · · ·     · · ·    · · ·     0      
     w`         · · ·    w2`       w2`+1      0        · · ·     · · ·     · · ·     · · ·     · · ·    · · ·     0      
                                                                                                                         
     .                                                                                                                   
     .                                                                                                                   
     .                                                                                                                   
     w          · · ·    w         · · ·     · · ·     · · ·    w             0      · · ·     · · ·    · · ·     0      
     1                      `+1                                    2`+1                                                  
     0          w         · · ·    · · ·     · · ·     · · ·    w          w         0         · · ·    · · ·     0      
                1                                                  2`       2`+1                                         
     .                                                                                                                   
Q =   .                                                                                                                    .  (7.49)
     .                                                                                                                   
                                                                                                                         
     .                                                                                                                   
     .                                                                                                                   
     .                                                                                                                   
                                                                                                                         
     0          · · ·     · · ·    · · ·     · · ·     · · ·       0        w1       · · ·              · · ·     w2`+1  
                                                                                                                         
     .                                                                                                                   
     .                                                                                                                   
.
0          · · ·     · · ·    · · ·     · · ·     · · ·     · · ·     · · ·     0         w1       · · ·      w`+1
Note that Qi,j = Qi−1,j−1 for all i, j ∈ {2, . . . , m}, and thus convoluation is a
matrix multiplication with parameter sharing. We also note that computing
the convolution only takes                O(km) times but computing a generic matrix
multiplication takes O(m²) time. Convolution has k parameters but generic
matrix multiplication will have m² parameters. Thus convolution is supposed
to be much more efficient than a generic matrix multiplication (as long as
the additional structure imposed does not hurt the flexibility of the model
to fit the data).
We also note that in practice there are many variants of the convolutional
layers that we define here, e.g., there are other ways to pad zeros or sometimes
the dimension of the output of the convolutional layers could be different from
the input. We omit some of this subtleties here for simplicity.
The convolutional layers used in practice have also many “channels” and
the simplified version above corresponds to the 1-channel version. Formally,
Conv1D takes in C                    vectors z₁, . . . , zC        ∈ Rᵐ as inputs, where C                               is referred
to as the number of channels.                           In other words, the more general version,
denoted by Conv1D, takes in a matrix as input, which is the concatenation
of z₁, . . . , zC and has dimension m × C . It can output C ′ vectors of dimension
m, denoted by Conv1D(z)₁, . . . , Conv1D(z)C′ , where C ′ is referred to as the
output channel, or equivalently a matrix of dimension m × C ′. Each of the
output is a sum of the simplified convolutions applied on various channels.
∑
C
∀i ∈ [C ′], Conv1D(z)i =                          Conv1D-Si,j(zj ).                             (7.50)
j=1
Note that each Conv1D-Si,j                                                             are modules with different parameters, and
thus the total number of parameters is k                                                           (the number of parameters in a
Conv1D-S) ×CC′ (the number of Conv1D-Si.j’s) = kCC′.                                                               In contrast, a
generic linear mapping from Rᵐ×C and Rᵐ×C′ has m²CC′ parameters. The





parameters can also be represented as a three-dimensional tensor of dimension k × C × C ′.

# 2-D convolution (brief)

A 2-D convolution with one channel, denoted by Conv2D-S, is analogous to the Conv1D-S, but takes a 2-dimensional input z ∈ Rm×m and applies a filter of size k × k, and outputs Conv2D-S(z) ∈ Rm×m. The full 2-D convolutional layer, denoted by Conv2D, takes in a sequence of matrices z1, . . . , zC ∈ Rm×m, or equivalently a 3-D tensor z = (z1, . . . , zC) ∈ Rm×m×C and outputs a sequence of matrices, Conv2D(z)1, . . . , Conv2D(z)C′ ∈ Rm×m, which can also be viewed as a 3D tensor in Rm×m×C′. Each channel of the output is sum of the outcomes of applying Conv2D-S layers on all the input channels.

∑C ∀i ∈ [C ′], Conv2D(z)i = Conv2D-Si,j(zj). (7.51)

Because there are CC′ number of Conv2D-S modules and each of the Conv2D-S module has k² parameters, the total number of parameters is CC′k². The parameters can also be viewed as a 4D tensor of dimension C × C ′ × k × k.

# 7.4 Backpropagation

In this section, we introduce backpropagation or auto-differentiation, which computes the gradient of the loss ∇J (θ) efficiently. We will start with an informal theorem that states that as long as a real-valued function f can be efficiently computed/evaluated by a differentiable network or circuit, then its gradient can be efficiently computed in a similar time. We will then show how to do this concretely for neural networks.

Because the formality of the general theorem is not the main focus here, we will introduce the terms with informal definitions. By a differentiable circuit or a differentiable network, we mean a composition of a sequence of differentiable arithmetic operations (additions, subtraction, multiplication, divisions, etc) and elementary differentiable functions (ReLU, exp, log, sin, cos, etc.). Let the size of the circuit be the total number of such operations and elementary functions. We assume that each of the operations and functions, and their derivatives or partial derivatives can be computed in O(1) time.

Theorem 7.4.1: [backpropagation or auto-differentiation, informally stated] Suppose a differentiable circuit of size N computes a real-valued function





f : R` → R. Then, the gradient ∇f can be computed in time O(N), by a circuit of size O(N).⁵

We note that the loss function J (ʲ)(θ) for j-th example can be indeed computed by a sequence of operations and functions involving additions, subtraction, multiplications, and non-linear activations. Thus the theorem suggests that we should be able to compute the ∇J (ʲ)(θ) in a similar time to that for computing J (ʲ)(θ) itself. This does not only apply to the fully-connected neural network introduced in the Section 7.2, but also many other types of neural networks that uses more advance modules.

We remark that auto-differentiation or backpropagation is already implemented in all the deep learning packages such as tensorflow and pytorch, and thus in practice, in most of cases a researcher does not need to write their backpropagation algorithms. However, understanding it is very helpful for gaining insights into the working of deep learning.

Organization of the rest of the section. In Section 7.4.1, we will start reviewing the basic Chain rule with a new perspective that is particularly useful for understanding backpropagation. Section 7.4.2 will introduce the general strategy for backpropagation. Section 7.4.2 will discuss how to compute the so-called backward function for basic modules used in neural networks, and Section 7.4.4 will put everything together to get a concrete backprop algorithm for MLPs.

# 7.4.1 Preliminaries on partial derivatives

Suppose a scalar variable J depend on some variables z (which could be a scalar, matrix, or high-order tensor), we write ∂J as the partial derivatives of J w.r.t to the variable z. ∂z

We stress that the convention here is that ∂J has exactly the same dimension as z itself. For example, if z ∈ Rᵐ×n, then ∂J ∈ Rm×n, and the (i, j)-entry of ∂J is equal to ∂J.

∂z

Remark 7.4.2: When both J and z are not scalars, the partial derivatives of J w.r.t z becomes either a matrix or tensor and the notation becomes somewhat tricky. Besides the mathematical or notational challenges in dealing

5We note if the output of the function f does not depend on some of the input coordinates, then we set by default the gradient w.r.t that coordinate to zero. Setting to zero does not count towards the total runtime here in our accounting scheme. This is why when N ≤ `, we can compute the gradient in O(N) time, which might be potentially even less than `.





with these partial derivatives of multi-variate functions, they are also expensive to compute and store, and thus rarely explicitly constructed empirically. The experience of authors of this note is that it’s generally more productive to think only about derivatives of scalar function w.r.t to vector, matrices, or tensors. For example, in this note, we will not deal with derivatives of multi-variate functions.

# Chain rule.

We review the chain rule in calculus but with a perspective and notions that are more relevant for auto-differentiation. Consider a scalar variable J which is obtained by the composition of f and g on some variable z,

z ∈ Rm

u = g(z) ∈ Rn

J = f(u) ∈ R. (7.52)

The same derivations below can be easily extend to the cases when z and u are matrices or tensors; but we insist that the final variable J is a scalar. (See also Remark 7.4.2.) Let u = (u1, . . . , un) and let g(z) = (g1(z), · · · , gn(z)). Then, the standard chain rule gives us that

∀i ∈ {1, . . . , m},  ∂J = ∑j=1n ∂J · ∂gj (7.53)

∂zi        ∂uj ∂zi

Alternatively, when z and u are both vectors, in a vectorized notation:

∂J

=
| ∂g1 | · · · | ∂gn |
| --- | ----- | --- |
| ∂z1 | ∂z1   | ∂J  |
| .   | .     | .   |
| .   | .     | .   |
| ∂g1 | · · · | ∂gn |
| ∂zm | ∂zm   | ∂u  |

In other words, the backward function is always a linear map from ∂J to ∂J, though note that the mapping itself can depend on z in complex ways. The matrix on the RHS of (7.54) is actually the transpose of the Jacobian matrix of the function g. However, we do not discuss in-depth about Jacobian matrices to avoid complications. Part of the reason is that when z is a matrix (or tensor), to write an analog of equation (7.54), one has to either flatten z into a vector or introduce additional notations on tensor-matrix product. In this sense, equation (7.53) is more convenient and effective to use in all cases.

For example, when z ∈ Rr×s is a matrix, we can easily rewrite equation (7.53)





# 101

to

∑

∀i, k,   ∂J    =      n  ∂J       · ∂gj  .                (7.55)

∂zik         j=1 ∂uj       ∂zik

which will indeed be used in some of the derivations in Section 7.4.3.

# Key interpretation of the chain rule.

We can view the formula above (equation (7.53) or (7.54)) as a way to compute ∂J from ∂J . Consider the following abstract problem. Suppose J depends on z ∂z via u as defined in equation (7.52). However, suppose the function f is not given or the function f is complex, but we are given the value of ∂J . Then, the formula in equation (7.54) gives us a way to compute ∂z from ∂J .

∂J = chain rule, formula (7.54) ∂J

=

=

=

=

=

=

=

=

=

=

=

=

=

=

=

=

=

∂u only requires info about g(·) and ⇒ ∂z .        (7.56)

Moreover, this formula only involves knowledge about g (more precisely ∂gʲ ). We will repeatedly use this fact in situations where g is a building block of a complex network f.

Empirically, it’s often useful to modularize the mapping in (7.53) or (7.54) into a black-box, and mathematically it’s also convenient to define a notation for it.⁶ We use B[g, z] to define the function that maps ∂J to ∂J, and write ∂J = B[g, z] (∂J ).                            ∂u    ∂z

∂z                 ∂u                               (7.57)

We call B[g, z] the backward function for the module g. Note that when z is fixed, B[g, z] is merely a linear map from Rⁿ to Rᵐ. Using equation (7.53), we have

∑

(Bg, z)i =      m ∂gj      · vj .                (7.58)

j=1 ∂zi

Or in vectorized notation, using (7.54), we have

 ∂g₁       · · ·     ∂gn 

 ∂z¹                 ∂z1 

.          .         .

Bg, z =    .          . .       .   · v .             (7.59)

.                    .

∂g1         · · ·    ∂gn

∂zm                  ∂zm

6e.g., the function is the .backward() method of the module in pytorch.





and therefore B[g, z] can be viewed as a matrix. However, in reality, z will be changing and thus the backward mapping has to be recomputed for different z’s while g is often fixed. Thus, empirically, the backward function Bg, z is often viewed as a function which takes in z (=the input to g) and v (=a vector that is supposed to be the gradient of some variable J w.r.t to the output of g) as the inputs, and outputs a vector that is supposed to be the gradient of J w.r.t to z.

# 7.4.2 General strategy of backpropagation

We discuss the general strategy of auto-differentiation in this section to build a high-level understanding. Then, we will instantiate the approach to concrete neural networks. We take the viewpoint that neural networks are complex compositions of small building blocks such as MM, σ, Conv2D, LN, etc., defined in Section 7.3. Note that the losses (e.g., mean-squared loss, or the cross-entropy loss) can also be abstractly viewed as additional modules. Thus, we can abstractly write the loss function J (on a single example (x, y)) as a composition of many modules:⁷

J = Mₖ (Mₖ−1(· · · M₁(x))) . (7.60)

For example, for a binary classification problem with a MLP ¯ hθ(x) (defined in equation (7.36) and (7.37)), the loss function has been written in the form of equation (7.60) with M₁ = MMW [1],b[1], M₂ = σ, M₃ = MMW [2],b[2], . . . , and Mₖ−1 = MMW [ᵣ],b[ᵣ] and Mₖ = `logistic. We can see from this example that some modules involve parameters, and other modules might only involve a fixed set of operations. For generality, we assume that each Mi involves a set of parameters θ[i], though θ[i] could possibly be an empty set when Mi is a fixed operation such as the nonlinear activations. We will discuss more on the granularity of the modularization, but so far we assume all the modules Mi’s are simple enough.

We introduce the intermediate variables for the computation in (7.60). 7Technically, we should write J = Mk(Mk−1(· · · M1(x)), y). However, y is treated as a constant for the purpose of computing the derivatives w.r.t to the parameters, and thus we can view it as part of Mk for the sake of simplicity of notations.





# 103

Let

u[0] = x
u[1] = M₁(u[0])
u[2] = M₂(u[1])
.
.
.
J = u[ᵏ] = Mₖ (u[ᵏ−1]) .                               (F)

Backpropagation consists of two passes, the forward pass and backward pass. In the forward pass, the algorithm simply computes u[1], . . . , u[ᵏ] from i = 1, . . . , k, sequentially using the definition in (F), and save all the intermediate variables u[i]’s in the memory.

In the backward pass, we first compute the derivatives w.r.t to the intermediate variables, that is, ∂J/∂u[k], . . . , ∂J/∂u[1] sequentially in this backward order, and then compute the derivatives of the parameters ∂θ[i]/∂u[i]. These two types of computations can also be interleaved with each other because ∂J/∂u[i] only depends on ∂J/∂u[i−1] and u[i−1] but not any ∂J/∂u[k] with k &#x3C; i.

We first see why ∂J/∂u[i−1] can be computed efficiently from ∂J/∂u[i] and u[i−1] by invoking the discussion in Section 7.4.1 on the chain rule. We instantiate the discussion by setting u = u[i] and z = u[i−1], and f (u) = Mₖ (Mₖ−1(· · · Mi+1(u[i]))), and g(·) = Mi(·). Note that f is very complex but we don’t need any concrete information about f. Then, the conclusive equation (7.56) corresponds to

∂J           =                    chain rule                          ∂J
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
∂u[i]                                                             [  ⇒  ∂u[i−1] .      (7.61)
only requires info about Mi(·) and u[i−1]

More precisely, we can write, following equation (7.57)

∂J        = B[Mi, u[i−1]] ( ∂J             ) .              (B1)
∂u[i−1]                                  ∂u[i]

Instantiating the chain rule with z = θ[i] and u = u[i], we also have

∂J                                = B[Mi, θ[i]] ( ∂J               ) .              (B2)
∂θ[i]                                                       ∂u[i]

See Figure 7.5 for an illustration of the algorithm.





# Figure 7.5: Back-propagation.

Remark 7.4.3: [Computational efficiency and granularity of the modules]

The main underlying purpose of treating a complex network as compositions of small modules is that small modules tend to have efficiently implementable backward function. In fact, the backward functions of all the atomic modules such as addition, multiplication and ReLU can be computed as efficiently as the evaluation of these modules (up to multiplicative constant factor). Using this fact, we can prove Theorem 7.4.1 by viewing neural networks as compositions of many atomic operations, and invoking the backpropagation discussed above. However, in practice, it’s oftentimes more convenient to modularize the networks using modules on the level of matrix multiplication, layernorm, etc. As we will see, naive implementation of these operations’ backward functions also have the same runtime as the evaluation of these functions.





# 7.4.3 Backward functions for basic modules

Using the general strategy in Section 7.4.2, it suffices to compute the backward function for all modules Mi’s used in the networks. We compute the backward function for the basic module MM, activations σ, and loss functions in this section.

# Backward function for MM

Suppose MMM,b(z) = Wz + b is a matrix multiplication module where z ∈ Rm and W ∈ Rn×m. Then, using equation (7.59), we have for v ∈ Rn

BMM, z =
| ∂(Wz+b)1 | · · · | ∂(Wz+b)n |
| -------- | ----- | -------- |
| ∂z1      | ∂z1   |          |
| .        | .     | .        |
| .        | .     | .        |
| ∂(Wz+b)1 | · · · | ∂(Wz+b)n |
| ∂zm      | ∂zm   |          |

v. (7.62)

Using the fact that ∀i ∈ [m], j ∈ [n],
∂(Wz+b)j = ∂bj + ∑k=1m Wjkzk = Wji ∂zi, we have

BMM, z = WTv ∈ Rm. (7.63)

In the derivation above, we have treated MM as a function of z. If we treat MM as a function of W and b, then we can also compute the backward function for the parameter variables W and b. It’s less convenient to use equation (7.59) because the variable W is a matrix and the matrix in (7.59) will be a 4-th order tensor that is challenging for us to mathematically write down. We use (7.58) instead:

BMM, Wij =

∑k=1m ∂(Wz + b)k ∂Wij · vk = ∑s=1m Wkszs · vk = vizj.

In vectorized notation, we have

BMM, W = vzT ∈ Rn×m. (7.65)

Using equation (7.59) for the variable b, we have,

BMM, b =
| ∂(Wz+b)1 | · · · | ∂(Wz+b)n |
| -------- | ----- | -------- |
| ∂b1      | ∂b1   |          |
| .        | .     | .        |
| .        | .     | .        |
| ∂(Wz+b)1 | · · · | ∂(Wz+b)n |
| ∂bn      | ∂bn   |          |

v = v. (7.66)





# 106

Here we used that ∂(W z+b)j = 0 if i = j and ∂(W ᶻ+b)j = 1 if i = j. The ∂bi computational efficiency for computing the backward function is O(mn), the same as evaluating the result of matrix multiplication up to constant factor.

# Backward function for the activations.

Suppose M (z) = σ(z) where σ is an element-wise activation function and z ∈ Rm. Then, using equation (7.59), we have

| ∂σ(z1) | · · · | ∂σ(zm) |
| ------ | ----- | ------ |
| ∂z1    |       | ∂z1    |
| .      | .     | .      |
| .      | .     | .      |
| ∂σ(z1) | · · · | ∂σ(zm) |
| ∂zm    |       | ∂zm    |

Bσ, z = diag(σ′(z1), · · · , σ′(zm))v (7.68)

= σ′(z) v ∈ Rm. (7.69)

Here, we used the fact that ∂σ(zj) = 0 when j = i, diag(λ1, . . . , λm) denotes the diagonal matrix with λ1, . . . , λm on the diagonal, and denotes the element-wise product of two vectors with the same dimension, and σ′(·) is the element-wise application of the derivative of the activation function σ.

Regarding computation efficiency, we note that at the first sight, equation (7.67) appears to indicate the backward function takes O(m²) time, but equation (7.69) shows that it’s implementable in O(m) time (which is the same as the time for evaluating of the function.) We are not supposed to be surprised by that the possibility of simplifying equation (7.67) to (7.69)—if we use smaller modules, that is, treating the vector-to-vector nonlinear activation as m scalar-to-scalar non-linear activation, then it’s more obvious that the backward pass should have similar time to the forward pass.

# Backward function for loss functions.

When a module M takes in a vector z and outputs a scalar, by equation (7.59), the backward function takes in a scalar v and outputs a vector with entries (BM, z)i = ∂M/∂zi v. Therefore, in vectorized notation, BM, z = ∂M · v.

Recall that squared loss ` (z, ∂z) = 1/2 (z−y)2 · v = (z − y) · v. MSE = 1/2 (z − y)2. Thus, B`MSE, z = 2 ∂z.

For logistics loss, by equation (2.6), we have

B`logistic, t = ∂`logistic(t, y) · v = (1/(1 + exp(−t)) − y) · v. (7.70)





For cross-entropy loss, by equation (2.17), we have

Bce, t = ∂ce(t, y) · v = (φ − ey) · v , (7.71)

where φ = softmax(t).

# 7.4.4 Back-propagation for MLPs

Given the backward functions for every module needed in evaluating the loss of an MLP, we follow the strategy in Section 7.4.2 to compute the gradient of the loss w.r.t to the hidden activations and the parameters.

We consider the an r-layer MLP with a logistic loss. The loss function can be computed via a sequence of operations (that is, the forward pass),

z[1]     = MMW [1],b1,
a[1] = σ(z[1])
z[2]     = MMW [2],b2
a[2] = σ(z[2])
.
.
.
z[r]     = MMW [ᵣ],b[ᵣ] (a[r−1])
J = logistic(z[r], y) . (7.72)

We apply the backward function sequentially in a backward order. First, we have that ∂J = B[logistic, z[r]] (∂J ) = B[logistic, z[r]](1) . (7.73)

Then, we iteratively compute ∂J and ∂J ’s by repeatedly invoking the chain rule (equation (7.58)),

∂J                             = B[MM, a[r−1]] ( ∂J   )
∂a[r−1]                                           (  ∂z[r]
∂J                             = B[σ, z[r−1]]     ∂J  )
∂z[r−1]                                             ∂a[r−1]
.
.
.
∂J . = B[σ, z[1]] (      ∂J ) . (7.74)
∂z[1]                    ∂a[1]





# 108

Numerically, we compute these quantities by repeatedly invoking equations (7.69) and (7.63) with different choices of variables. We note that the intermediate values of a[i] and z[i] are used in the back-propagation (equation (7.74)), and therefore these values need to be stored in the memory after the forward pass.

Next, we compute the gradient of the parameters by invoking equations (7.65) and (7.66),

| ∂J | = B\[MM, W \[r]] ( ∂J ) | ∂W \[r] |
| -- | ----------------------- | ------- |
| ∂J | = B\[MM, b\[r]] ∂J      | ∂b\[r]  |
| ∂J | = B\[MM, W \[1]] ( ∂J ) | ∂W \[1] |
| ∂J | = B\[MM, b\[1]] ∂J      | ∂b\[1]  |

We also note that the block of computations in equations (7.75) can be interleaved with the block of computation in equations (7.74) because the ∂J[i] and ∂J can be computed as soon as ∂J is computed.

Putting all of these together, and explicitly invoking the equations (7.72), (7.74) and (7.75), we have the following algorithm (Algorithm 3).





# Algorithm 3 Back-propagation for multi-layer neural networks.

1. Forward pass. Compute and store the values of *a[ᵏ]’s, z[ᵏ]’s, and J* using the equations (7.72).
2. Backward pass. Compute the gradient of loss *J with respect to z[r]*:

*∂J = B[`logistic, z[r]](1) = (1/(1 + exp(−z[r])) − y)* . (7.76)

*∂z[r]*
3. for *k = r − 1 to 0* do
4. Compute the gradient with respect to parameters *W [ᵏ+1] and b[ᵏ+1]*.

*∂J = B[MM, W [ᵏ+1]] (∂J / ∂z[ᵏ+1])*

= *∂J a[ᵏ]>* . (7.77)

*∂J = ∂z[ᵏ+1] [k+1] (∂J / ∂z[ᵏ+1])*

= *∂J* . (7.78)

*∂z[ᵏ+1]*
5. When *k ≥ 1, compute the gradient with respect to z[ᵏ] and a[ᵏ]*.

*∂J = B[σ, a[ᵏ]] (∂J / ∂z[ᵏ+1])*

= *W [ᵏ+1]> ∂J / ∂z[ᵏ+1]* . (7.79)

*∂J = B[σ, z[ᵏ]] (∂J / ∂a[ᵏ])*

= *σ′(z[ᵏ]) ∂J / ∂a[ᵏ]* . (7.80)

# 7.5 Vectorization over training examples

As we discussed in Section 7.1, in the implementation of neural networks, we will leverage the parallelism across the multiple examples. This means that we will need to write the forward pass (the evaluation of the outputs) of the neural network and the backward pass (backpropagation) for multiple





# Training Examples in Matrix Notation

The basic idea is simple. Suppose you have a training set with three examples x(1), x(2), x(3). The first-layer activations for each example are as follows:

z1 = W[1]x(1) + b[1]

z1 = W[1]x(2) + b[1]

z1 = W[1]x(3) + b[1]

Note the difference between square brackets [·], which refer to the layer number, and parenthesis (·), which refer to the training example number. Intuitively, one would implement this using a for loop. It turns out, we can vectorize these operations as well. First, define:

X =

| x(1) | x(2) | x(3) |
| ---- | ---- | ---- |

∈ Rd×3

Note that we are stacking training examples in columns and not rows. We can then combine this into a single unified formulation:

Z[1] =

| z\[1]\(1) | z\[1]\(2) | z\[1]\(3) |
| --------- | --------- | --------- |

= W[1]X + b[1]

You may notice that we are attempting to add b[1] ∈ R4×1 to W[1]X ∈ R4×3. Strictly following the rules of linear algebra, this is not allowed. In practice however, this addition is performed using broadcasting. We create an intermediate b˜[1] ∈ R4×3:

b˜ =

| b\[1] | b\[1] | b\[1] |
| ----- | ----- | ----- |

We can then perform the computation: Z[1] = W[1]X + b˜[1]. It is not necessary to explicitly construct b˜[1]. By inspecting the dimensions in (7.82), you can assume b[1] ∈ R4×1 is correctly broadcast to W[1]X ∈ R4×3. The matricization approach as above can easily generalize to multiple layers, with one subtlety though, as discussed below.




Complications/Subtlety in the Implementation. All the deep learning packages or implementations put the data points in the rows of a data matrix. (If the data point itself is a matrix or tensor, then the data are concentrated along the zero-th dimension.) However, most of the deep learning papers use a similar notation to these notes where the data points are treated as column vectors.8 There is a simple conversion to deal with the mismatch: in the implementation, all the columns become row vectors, row vectors become column vectors, all the matrices are transposed, and the orders of the matrix multiplications are flipped. In the example above, using the row major convention, the data matrix is X ∈ R3×d, the first layer weight matrix has dimensionality d × m (instead of m × d as in the two layer neural net section), and the bias vector b[1] ∈ R1×m. The computation for the hidden activation becomes

Z[1] = XW[1] + b[1] ∈ R3×m (7.84)

8The instructor suspects that this is mostly because in mathematics we naturally multiply a matrix to a vector on the left hand side.


# Part III

# Generalization and regularization

112


# Chapter 8

# Generalization

This chapter discusses tools to analyze and understand the generalization of machine learning models, i.e, their performances on unseen test examples. Recall that for supervised learning problems, given a training dataset {(x(i), y(i))}ⁿ, we typically learn a model hθ by minimizing a loss/cost function J(θ), which encourages hθ to fit the data. E.g., when the loss function is the least square loss (aka mean squared error), we have:

J(θ) = 1/n ∑i=1n (y(i) − hθ(x(i)))². This loss function for training purposes is oftentimes referred to as the training loss/error/cost.

However, minimizing the training loss is not our ultimate goal—it is merely our approach towards the goal of learning a predictive model. The most important evaluation metric of a model is the loss on unseen test examples, which is oftentimes referred to as the test error. Formally, we sample a test example (x, y) from the so-called test distribution D, and measure the model’s error on it, by, e.g., the mean squared error, (hθ(x) − y)². The expected loss/error over the randomness of the test example is called the test loss/error,¹

L(θ) = E(x,y)∼D[(y − hθ(x))²] (8.1)

Note that the measurement of the error involves computing the expectation, and in practice, it can be approximated by the average error on many sampled test examples, which are referred to as the test dataset. Note that the key difference here between training and test datasets is that the test examples.

¹In theoretical and statistical literature, we oftentimes call the uniform distribution over the training set {(x(i), y(i))}ⁿ, denoted by D, an empirical distribution, and call D the population distribution. Partly because of this, the training loss is also referred to as the empirical loss/risk/error, and the test loss is also referred to as the population loss/risk/error.


are unseen, in the sense that the training procedure has not used the test examples. In classical statistical learning settings, the training examples are also drawn from the same distribution as the test distribution D, but still the test examples are unseen by the learning procedure whereas the training examples are seen.2

Because of this key difference between training and test datasets, even if they are both drawn from the same distribution D, the test error is not necessarily always close to the training error.3 As a result, successfully minimizing the training error may not always lead to a small test error. We typically say the model overfits the data if the model predicts accurately on the training dataset but doesn’t generalize well to other test examples, that is, if the training error is small but the test error is large. We say the model underfits the data if the training error is relatively large4 (and in this case, typically the test error is also relatively large.)

This chapter studies how the test error is influenced by the learning procedure, especially the choice of model parameterizations. We will decompose the test error into “bias” and “variance” terms and study how each of them is affected by the choice of model parameterizations and their tradeoffs. Using the bias-variance tradeoff, we will discuss when overfitting and underfitting will occur and be avoided. We will also discuss the double descent phenomenon in Section 8.2 and some classical theoretical results in Section 8.3.

These days, researchers have increasingly been more interested in the setting with “domain shift”, that is, the training distribution and test distribution are different.

the difference between test error and training error is often referred to as the generalization gap. The term generalization error in some literature means the test error, and in some other literature means the generalization gap.

e.g., larger than the intrinsic noise level of the data in regression problems.



# 8.1 Bias-variance tradeoff

| training dataset        |               | test dataset            |           |   |   |
| ----------------------- | ------------- | ----------------------- | --------- | - | - |
|                         | training data |                         | test data |   |   |
| ground truth h\*        |               | ground truth h\*        |           |   |   |
| 1.0                     | 1.0           |                         |           |   |   |
| 0.5                     | 0.5           |                         |           |   |   |
| 0.0                     | 0.0           |                         |           |   |   |
| 0.0 0.2 0.4 0.6 0.8 1.0 |               | 0.0 0.2 0.4 0.6 0.8 1.0 |           |   |   |

Figure 8.1: A running example of training and test dataset for this section.

As an illustrating example, we consider the following training dataset and test dataset, which are also shown in Figure 8.1. The training inputs x(i)’s are randomly chosen and the outputs y(i) are generated by y(i) = h?(x(i)) + ξ(i) where the function h?(·) is a quadratic function and is shown in Figure 8.1 as the solid line, and ξ(i) is the observation noise assumed to be generated from ∼ N (0, σ²). A test example (x, y) also has the same input-output relationship y = h?(x) + ξ where ξ ∼ N (0, σ²). It’s impossible to predict the noise ξ, and therefore essentially our goal is to recover the function h?(·).

We will consider the test error of learning various types of models. When talking about linear regression, we discussed the problem of whether to fit a “simple” model such as the linear “y = θ₀ + θ₁x,” or a more “complex” model such as the polynomial “y = θ₀ + θ₁x + · · · θ₅x⁵.”

We start with fitting a linear model, as shown in Figure 8.2. The best fitted linear model cannot predict y from x accurately even on the training dataset, let alone on the test dataset. This is because the true relationship between y and x is not linear—any linear model is far away from the true function h?(·). As a result, the training error is large and this is a typical situation of underfitting.






# 1.5 training data

# 1.5 test data

|     | best fit linear model |     |     |     |     |
| --- | --------------------- | --- | --- | --- | --- |
| 1.0 |                       |     |     |     |     |
| y   | 0.5                   |     |     |     |     |
| 0.0 |                       |     |     |     |     |
| 0.0 | 0.2                   | 0.4 | 0.6 | 0.8 | 1.0 |
|     |                       | x   |     |     |     |

Figure 8.2: The best fit linear model has large training and test errors.

The issue cannot be mitigated with more training examples—even with a very large amount of, or even infinite training examples, the best fitted linear model is still inaccurate and fails to capture the structure of the data (Figure 8.3). Even if the noise is not present in the training data, the issue still occurs (Figure 8.4). Therefore, the fundamental bottleneck here is the linear model family’s inability to capture the structure in the data—linear models cannot represent the true quadratic function h?—, but not the lack of the data. Informally, we define the bias of a model to be the test error even if we were to fit it to a very (say, infinitely) large training dataset. Thus, in this case, the linear model suffers from large bias, and underfits (i.e., fails to capture structure exhibited by) the data.

# 1.5 fitting linear models on a large dataset

# fitting linear models on a noiseless dataset

| training data | ground truth h \*     |     |     |     |     |
| ------------- | --------------------- | --- | --- | --- | --- |
| 1.0           | best fit linear model |     |     |     |     |
| y             | 0.5                   |     |     |     |     |
| 0.0           |                       |     |     |     |     |
| 0.0           | 0.2                   | 0.4 | 0.6 | 0.8 | 1.0 |
|               | x                     |     |     |     |     |

Figure 8.3: The best fit linear model on a much larger dataset still has a large training error.

Figure 8.4: The best fit linear model on a noiseless dataset also has a large training/test error.

Next, we fit a 5th-degree polynomial to the data. Figure 8.5 shows that it fails to learn a good model either. However, the failure pattern is different from the linear model case. Specifically, even though the learnt 5th-degree





polynomial did a very good job predicting y(i)’s from x(i)’s for training examples, it does not work well on test examples (Figure 8.5). In other words, the model learnt from the training set does not generalize well to other test examples—the test error is high. Contrary to the behavior of linear models, the bias of the 5-th degree polynomials is small—if we were to fit a 5-th degree polynomial to an extremely large dataset, the resulting model would be close to a quadratic function and be accurate (Figure 8.6). This is because the family of 5-th degree polynomials contains all the quadratic functions (setting θ₅ = θ₄ = θ₃ = 0 results in a quadratic function), and, therefore, 5-th degree polynomials are in principle capable of capturing the structure of the data.

# 1.5

training data

# 1.5

test data

best fit 5-th degree model

ground truth h *

# 1.0

1.0

y

# 0.5

y

# 0.0

0.0

0.2 0.4 0.6 0.8 1.0

x

Figure 8.5: Best fit 5-th degree polynomial has zero training error, but still has a large test error and does not recover the ground truth. This is a classic situation of overfitting.

fitting 5-th degree model on large dataset

# 1.5

training data

best fit 5-th degree model

# 1.0

ground truth h *

y

# 0.5

0.0

0.0 0.2 0.4 0.6 0.8 1.0

x

Figure 8.6: The best fit 5-th degree polynomial on a huge dataset nearly recovers the ground-truth—suggesting that the culprit in Figure 8.5 is the variance (or lack of data) but not bias.

The failure of fitting 5-th degree polynomials can be captured by another




component of the test error, called variance of a model fitting procedure. Specifically, when fitting a 5-th degree polynomial as in Figure 8.7, there is a large risk that we’re fitting patterns in the data that happened to be present in our small, finite training set, but that do not reflect the wider pattern of the relationship between x and y. These “spurious” patterns in the training set are (mostly) due to the observation noise ξ(i), and fitting these spurious patterns results in a model with large test error. In this case, we say the model has a large variance.

# fitting 5-th degree model on different datasets

|     | training data              | training data              | training data              |
| --- | -------------------------- | -------------------------- | -------------------------- |
| 1.5 | best fit 5-th degree model | best fit 5-th degree model | best fit 5-th degree model |
| 1.0 |                            |                            |                            |
| y   | 0.5                        | 0.5                        | 0.5                        |
| 0.0 |                            |                            |                            |
| 0.0 | 0.2                        | 0.2                        | 0.2                        |
| 0.4 | 0.4                        | 0.4                        | 0.4                        |
| 0.6 | 0.6                        | 0.6                        | 0.6                        |
| 0.8 | 0.8                        | 0.8                        | 0.8                        |
| 1.0 | 1.0                        | 1.0                        | 1.0                        |

Figure 8.7: The best fit 5-th degree models on three different datasets generated from the same distribution behave quite differently, suggesting the existence of a large variance.

The variance can be intuitively (and mathematically, as shown in Section 8.1.1) characterized by the amount of variations across models learnt on multiple different training datasets (drawn from the same underlying distribution). The “spurious patterns” are specific to the randomness of the noise (and inputs) in a particular dataset, and thus are different across multiple training datasets. Therefore, overfitting to the “spurious patterns” of multiple datasets should result in very different models. Indeed, as shown in Figure 8.7, the models learned on the three different training datasets are quite different, overfitting to the “spurious patterns” of each datasets.

Often, there is a tradeoff between bias and variance. If our model is too “simple” and has very few parameters, then it may have large bias (but small variance), and it typically may suffer from underfitting. If it is too “complex” and has very many parameters, then it may suffer from large variance (but have smaller bias), and thus overfitting. See Figure 8.8 for a typical tradeoff between bias and variance.



# Optimal Tradeoff

Test Error (= Bias² + Variance)

Variance

Error

Bias²

Model Complexity

Figure 8.8: An illustration of the typical bias-variance tradeoff.

As we will see formally in Section 8.1.1, the test error can be decomposed as a summation of bias and variance. This means that the test error will have a convex curve as the model complexity increases, and in practice we should tune the model complexity to achieve the best tradeoff. For instance, in the example above, fitting a quadratic function does better than either of the extremes of a first or a 5-th degree polynomial, as shown in Figure 8.9.

1.5

training data

best fit quadratic model

1.0

ground truth h *

y

0.5

0.0

0.0    0.2  0.4  0.6         0.8        1.0

x

Figure 8.9: Best fit quadratic model has small training and test error because quadratic model achieves a better tradeoff.

Interestingly, the bias-variance tradeoff curves or the test error curves do not universally follow the shape in Figure 8.8, at least not universally when the model complexity is simply measured by the number of parameters. (We will discuss the so-called double descent phenomenon in Section 8.2.) Nevertheless, the principle of bias-variance tradeoff is perhaps still the first resort when analyzing and predicting the behavior of test errors.





# 8.1.1 A mathematical decomposition (for regression)

To formally state the bias-variance tradeoff for regression problems, we consider the following setup (which is an extension of the beginning paragraph of Section 8.1).

- Draw a training dataset S = {x(i), y(i)}ⁿ such that y(i) = h?(x(i)) + ξ(i) where ξ(i) ∈ N (0, σ²). i=1
- Train a model on the dataset S, denoted by ˆhS.
- Take a test example (x, y) such that y = h?(x) + ξ where ξ ∼ N (0, σ²), and measure the expected test error (averaged over the random draw of the training set S and the randomness of ξ)⁵⁶

MSE(x) = ES,ξ [(y − hS (x))²] (8.2)

We will decompose the MSE into a bias and variance term. We start by stating a following simple mathematical tool that will be used twice below.

# Claim 8.1.1:

Suppose A and B are two independent real random variables and E[A] = 0. Then, E[(A + B)²] = E[A²] + E[B²].

As a corollary, because a random variable A is independent with a constant c, when E[A] = 0, we have E[(A + c)²] = E[A²] + c².

The proof of the claim follows from expanding the square: E[(A + B)²] = E[A²] + E[B²] + 2E[AB] = E[A²] + E[B²]. Here we used the independence to show that E[AB] = E[A]E[B] = 0.

Using Claim 8.1.1 with A = ξ and B = h(x) − hS(x), we have

MSE(x) = E[(y − hS (x))²] = E[(ξ + (h?(x) − hS (x)))²] (8.3)

= E[ξ²] + E[(h?(x) − hS (x))²] (by Claim 8.1.1)

= σ² + E[(h?(x) − hS (x))²] (8.4)

Then, let’s define havg(x) = ES [hS (x)] as the “average model”—the model obtained by drawing an infinite number of datasets, training on them, and averaging their predictions on x. Note that havg is a hypothetical model for analytical purposes that can not be obtained in reality (because we don’t

5For simplicity, the test input x is considered to be fixed here, but the same conceptual message holds when we average over the choice of x’s.

6The subscript under the expectation symbol is to emphasize the variables that are considered as random by the expectation operation.





have infinite number of datasets). It turns out that for many cases, havg is (approximately) equal to the model obtained by training on a single dataset with infinite samples. Thus, we can also intuitively interpret havg this way, which is consistent with our intuitive definition of bias in the previous subsection.

We can further decompose MSE(x) by letting c = h?(x)−havg(x) (which is a constant that does not depend on the choice of S!) and A = havg(x)− hS (x) in the corollary part of Claim 8.1.1:

MSE(x) = σ² + E[(h?(x) − hS (x))²] (8.5)

= σ² + (h?(x) − havg(x))² + E[(havg − hS (x))²] (8.6)

= σ² + (h?(x) − havg(x))² + var(hS (x)) (8.7)

We call the second term the bias (square) and the third term the variance. As discussed before, the bias captures the part of the error that are introduced due to the lack of expressivity of the model. Recall that havg can be thought of as the best possible model learned even with infinite data. Thus, the bias is not due to the lack of data, but is rather caused by that the family of models fundamentally cannot approximate the h? For example, in the illustrating example in Figure 8.2, because any linear model cannot approximate the true quadratic function h?, neither can havg, and thus the bias term has to be large.

The variance term captures how the random nature of the finite dataset introduces errors in the learned model. It measures the sensitivity of the learned model to the randomness in the dataset. It often decreases as the size of the dataset increases.

There is nothing we can do about the first term σ² as we cannot predict the noise ξ by definition.

Finally, we note that the bias-variance decomposition for classification is much less clear than for regression problems. There have been several proposals, but there is as yet no agreement on what is the “right” and/or the most useful formalism.

# 8.2 The double descent phenomenon

Model-wise double descent. Recent works have demonstrated that the test error can present a “double descent” phenomenon in a range of machine





learning models including linear models and deep neural networks.7 The conventional wisdom, as discussed in Section 8.1, is that as we increase the model complexity, the test error first decreases and then increases, as illustrated in Figure 8.8. However, in many cases, we empirically observe that the test error can have a second descent—it first decreases, then increases to a peak around when the model size is large enough to fit all the training data very well, and then decreases again in the so-called overparameterized regime, where the number of parameters is larger than the number of data points. See Figure 8.10 for an illustration of the typical curves of test errors against model complexity (measured by the number of parameters). To some extent, the overparameterized regime with the second descent is considered as new to the machine learning community—partly because lightly-regularized, overparameterized models are only extensively used in the deep learning era. A practical implication of the phenomenon is that one should not hold back from scaling into and experimenting with over-parametrized models because the test error may well decrease again to a level even smaller than the previous lowest point. Actually, in many cases, larger overparameterized models always lead to a better test performance (meaning there won’t be a second ascent after the second descent).

| classical regime:      | modern regime:                                            |
| ---------------------- | --------------------------------------------------------- |
| bias-variance tradeoff | over-parameterization                                     |
|                        | typically when # parameters is sufficient to fit the data |

error
test

# parameters

Figure 8.10: A typical model-wise double descent phenomenon. As the number of parameters increases, the test error first decreases when the number of parameters is smaller than the training data. Then in the overparameterized regime, the test error decreases again.

7The discovery of the phenomenon perhaps dates back to Opper [1995, 2001], and has been recently popularized by Belkin et al. [2020], Hastie et al. [2019], etc.





# Sample-wise double descent

A priori, we would expect that more training examples always lead to smaller test errors—more samples give strictly more information for the algorithm to learn from. However, recent work [Nakkiran, 2019] observes that the test error is not monotonically decreasing as we increase the sample size. Instead, as shown in Figure 8.11, the test error decreases, and then increases and peaks around when the number of examples (denoted by n) is similar to the number of parameters (denoted by d), and then decreases again. We refer to this as the sample-wise double descent phenomenon. To some extent, sample-wise double descent and model-wise double descent are essentially describing similar phenomena—the test error is peaked when n ≈ d.

# Explanation and mitigation strategy

The sample-wise double descent, or, in particular, the peak of test error at n ≈ d, suggests that the existing training algorithms evaluated in these experiments are far from optimal when n ≈ d. We will be better off by tossing away some examples and run the algorithms with a smaller sample size to steer clear of the peak. In other words, in principle, there are other algorithms that can achieve smaller test error when n ≈ d, but the algorithms evaluated in these experiments fail to do so. The sub-optimality of the learning procedure appears to be the culprit of the peak in both sample-wise and model-wise double descent.

Indeed, with an optimally-tuned regularization (which will be discussed more in Section 9), the test error in the n ≈ d regime can be dramatically improved, and the model-wise and sample-wise double descent are both mitigated. See Figure 8.11.

The intuition above only explains the peak in the model-wise and sample-wise double descent, but does not explain the second descent in the model-wise double descent—why overparameterized models are able to generalize so well. The theoretical understanding of overparameterized models is an active research area with many recent advances. A typical explanation is that the commonly-used optimizers such as gradient descent provide an implicit regularization effect (which will be discussed in more detail in Section 9.2). In other words, even in the overparameterized regime and with an unregularized loss function, the model is still implicitly regularized, and thus exhibits a better test performance than an arbitrary solution that fits the data. For example, for linear models, when n &#x3C; d, the gradient descent optimizer with zero initialization finds the minimum norm solution that fits the data (instead of an arbitrary solution that fits the data), and the minimum norm regularizer turns out to be a sufficiently good for the overparameterized regime (but it’s not a good regularizer when n ≈ d, resulting in the peak of test error).





# 1.25 Test Risk for Regularized Regression

Figure 8.11: Left: The sample-wise double descent phenomenon for linear models. Right: The sample-wise double descent with different regularization λr strength for linear models. Using the optimal regularization parameter λ (optimally tuned for each n, shown in green solid curve) mitigates double descent. Setup: The data distribution of (x, y) is x ∼ N (0, Id) and y ∼ xTβ + N (0, σ²) where d = 500, σ = 0.5 and ‖β‖2 = 1.

Finally, we also remark that the double descent phenomenon has been mostly observed when the model complexity is measured by the number of parameters. It is unclear if and when the number of parameters is the best complexity measure of a model. For example, in many situations, the norm of the models is used as a complexity measure. As shown in Figure 8.12 right, for a particular linear case, if we plot the test error against the norm of the learnt model, the double descent phenomenon no longer occurs. This is partly because the norm of the learned model is also peaked around n ≈ d (See Figure 8.12 (middle) or Belkin et al. [2019], Mei and Montanari [2022], and discussions in Section 10.8 of James et al. [2021]). For deep neural networks, the correct complexity measure is even more elusive. The study of double descent phenomenon is an active research topic.

8 The figure is reproduced from Figure 1 of Nakkiran et al. [2020]. Similar phenomenon are also observed in Hastie et al. [2022], Mei and Montanari [2022]

|             | Regularization Strength |             |             |             |             |
| ----------- | ----------------------- | ----------- | ----------- | ----------- | ----------- |
| λ = 2-8λopt | λ = 2-7λopt             | λ = 2-6λopt | λ = 2-5λopt | λ = 2-4λopt | λ = 2-3λopt |
| 1.50        | 1.25                    | 1.00        | 0.75        | 0.50        | 0.25        |

0.00  0  200  400  600  800  1000

Num Samples





# Figure 8.12

Left: The double descent phenomenon, where the number of parameters is used as the model complexity. Middle: The norm of the learned model is peaked around n ≈ d. Right: The test error against the norm of the learnt model. The color bar indicates the number of parameters and the arrows indicate the direction of increasing model size. Their relationships are closer to the conventional wisdom than to a double descent. Setup: We consider a linear regression with a fixed dataset of size n = 500. The input x is a random ReLU feature on Fashion-MNIST, and output y ∈ R¹⁰ is the one-hot label. This is the same setting as in Section 5.2 of Nakkiran et al. [2020].





# 8.3 Sample complexity bounds (optional readings)

# 8.3.1 Preliminaries

In this set of notes, we begin our foray into learning theory. Apart from being interesting and enlightening in its own right, this discussion will also help us hone our intuitions and derive rules of thumb about how to best apply learning algorithms in different settings. We will also seek to answer a few questions: First, can we make formal the bias/variance tradeoff that was just discussed? This will also eventually lead us to talk about model selection methods, which can, for instance, automatically decide what order polynomial to fit to a training set. Second, in machine learning it’s really generalization error that we care about, but most learning algorithms fit their models to the training set. Why should doing well on the training set tell us anything about generalization error? Specifically, can we relate error on the training set to generalization error? Third and finally, are there conditions under which we can actually prove that learning algorithms will work well?

We start with two simple but very useful lemmas.

Lemma. (The union bound). Let A₁, A₂, . . . , Aₖ be k different events (that may not be independent). Then

P (A₁ ∪ · · · ∪ Aₖ ) ≤ P (A₁) + . . . + P (Aₖ ).

In probability theory, the union bound is usually stated as an axiom (and thus we won’t try to prove it), but it also makes intuitive sense: The probability of any one of k events happening is at most the sum of the probabilities of the k different events.

Lemma. (Hoeffding inequality) Let Z₁, . . . , Zₙ be n independent and identically distributed (iid) random variables drawn from a Bernoulli(φ) distribution. I.e., P (Z = 1) = φ, and P (Z = 0) = 1 − φ. Let ˆ ∑ₙ be the mean of these random variables, and let any γ > 0 be fixed. Then

P (|φ − ˆφ| > γ) ≤ 2 exp(−2γ²n)

This lemma (which in learning theory is also called the Chernoff bound) says that if we take ˆφ—the average of n Bernoulli(φ) random variables—to be our estimate of φ, then the probability of our being far from the true value is small, so long as n is large. Another way of saying this is that if you have a biased coin whose chance of landing on heads is φ, then if you toss it n times...





times and calculate the fraction of times that it came up heads, that will be a good estimate of φ with high probability (if n is large).

Using just these two lemmas, we will be able to prove some of the deepest and most important results in learning theory.

To simplify our exposition, let’s restrict our attention to binary classification in which the labels are y ∈ {0, 1}. Everything we’ll say here generalizes to other problems, including regression and multi-class classification.

We assume we are given a training set S = {(x(i), y(i)); i = 1, . . . , n} of size n, where the training examples (x(i), y(i)) are drawn iid from some probability distribution D. For a hypothesis h, we define the training error (also called the empirical risk or empirical error in learning theory) to be

ε̂(h) = 1/n ∑i=1n 1{h(x(i)) = y(i)}.

This is just the fraction of training examples that h misclassifies. When we want to make explicit the dependence of ε̂(h) on the training set S, we may also write this as ε̂S(h). We also define the generalization error to be

ε(h) = P(x,y)∼D(h(x) = y).

I.e. this is the probability that, if we now draw a new example (x, y) from the distribution D, h will misclassify it.

Note that we have assumed that the training data was drawn from the same distribution D with which we’re going to evaluate our hypotheses (in the definition of generalization error). This is sometimes also referred to as one of the PAC assumptions.9

Consider the setting of linear classification, and let hθ(x) = 1{θᵀ x ≥ 0}. What’s a reasonable way of fitting the parameters θ? One approach is to try to minimize the training error, and pick

θ̂ = arg minθ ε̂(hθ).

We call this process empirical risk minimization (ERM), and the resulting hypothesis output by the learning algorithm is h = ĥθ. We think of ERM as the most “basic” learning algorithm, and it will be this algorithm that we

9PAC stands for “probably approximately correct,” which is a framework and set of assumptions under which numerous results on learning theory were proved. Of these, the assumption of training and testing on the same distribution, and the assumption of the independently drawn training examples, were the most important.





focus on in these notes. (Algorithms such as logistic regression can also be viewed as approximations to empirical risk minimization.)

In our study of learning theory, it will be useful to abstract away from the specific parameterization of hypotheses and from issues such as whether we’re using a linear classifier. We define the hypothesis class H used by a learning algorithm to be the set of all classifiers considered by it. For linear classification, H = {hθ : hθ(x) = 1{θᵀ x ≥ 0}, θ ∈ Rd+1} is thus the set of all classifiers over X (the domain of the inputs) where the decision boundary is linear. More broadly, if we were studying, say, neural networks, then we could let H be the set of all classifiers representable by some neural network architecture.

Empirical risk minimization can now be thought of as a minimization over the class of functions H, in which the learning algorithm picks the hypothesis:



Hoeffding inequality, and obtain

P (|ε(hi) − ˆε(hi)| > γ) ≤ 2 exp(−2γ² n).

This shows that, for our particular hi, training error will be close to generalization error with high probability, assuming n is large. But we don’t just want to guarantee that ε(hi) will be close to ˆε(hi) (with high probability) for just only one particular hi. We want to prove that this will be true simultaneously for all h ∈ H. To do so, let Ai denote the event that |ε(hi) − ˆε(hi)| > γ. We’ve already shown that, for any particular Ai, it holds true that P (Ai) ≤ 2 exp(−2γ²n). Thus, using the union bound, we have that

P (∃ h ∈ H.|ε(h) − ˆε(hi)| > γ) = P (A1 ∪ · · · ∪ Ak) ≤ ∑i=1k P (Ai) ≤ 2k exp(−2γ²n)

If we subtract both sides from 1, we find that

P (¬∃ h ∈ H.|ε(h) − ˆε(h)| > γ) = P (∀h ∈ H.|ε(h) − ˆε(hi)| ≤ γ) ≥ 1 − 2k exp(−2γ²n)

(The “¬” symbol means “not.”) So, with probability at least 1 − 2k exp(−2γ²n), we have that ε(h) will be within γ of ˆε(h) for all h ∈ H. This is called a uniform convergence result, because this is a bound that holds simultaneously for all (as opposed to just one) h ∈ H.

In the discussion above, what we did was, for particular values of n and γ, give a bound on the probability that for some h ∈ H, |ε(h) − ˆε(h)| > γ. There are three quantities of interest here: n, γ, and the probability of error; we can bound either one in terms of the other two.

For instance, we can ask the following question: Given γ and some δ > 0, how large must n be before we can guarantee that with probability at least 1 − δ, training error will be within γ of generalization error? By setting δ = 2k exp(−2γ²n) and solving for n, [you should convince yourself this is the right thing to do!], we find that if

n ≥ 1 / (2γ²) log(2k / δ),



then with probability at least 1 − δ, we have that |ε(h) − ˆε(h)| ≤ γ for all h ∈ H. (Equivalently, this shows that the probability that |ε(h) − ˆε(h)| > γ for some h ∈ H is at most δ.) This bound tells us how many training examples we need in order to make a guarantee. The training set size n that a certain method or algorithm requires in order to achieve a certain level of performance is also called the algorithm’s sample complexity.

The key property of the bound above is that the number of training examples needed to make this guarantee is only logarithmic in k, the number of hypotheses in H. This will be important later.

Similarly, we can also hold n and δ fixed and solve for γ in the previous equation, and show [again, convince yourself that this is right!] that with probability 1 − δ, we have that for all h ∈ H,

|ε(h) − ˆε(h)| ≤ √(1/(2n)) log(2k/δ)

Now, let’s assume that uniform convergence holds, i.e., that |ε(h) − ˆε(h)| ≤ γ for all h ∈ H. What can we prove about the generalization of our learning algorithm that ˆh = arg minh∈H εˆ(h)?

Define h* = arg minh∈H ε(h) to be the best possible hypothesis in H. Note that h* is the best that we could possibly do given that we are using H, so it makes sense to compare our performance to that of h*. We have:

εˆ(h) ≤ εˆ(h) + γ ≤ εˆ(h*) + γ ≤ ε(h*) + 2γ

The first line used the fact that |εˆ(h) − εˆ(h)| ≤ γ (by our uniform convergence assumption). The second used the fact that ˆh was chosen to minimize εˆ(h) and hence εˆ(h) ≤ εˆ(h) for all h, and in particular εˆ(h) ≤ εˆ(h*). The third line used the uniform convergence assumption again, to show that εˆ(h*) ≤ ε(h*) + γ. So, what we’ve shown is the following: If uniform convergence occurs, then the generalization error of h is at most 2γ worse than the best possible hypothesis in H!

Let’s put all this together into a theorem.

# Theorem.

Let |H| = k, and let any n, δ be fixed. Then with probability at least 1 − δ, we have that

εˆ(h) ≤ minh∈H ε(h) + √(2k/(2n log(1/δ))).





This is proved by letting γ equal the √· term, using our previous argument that uniform convergence occurs with probability at least 1 − δ, and then noting that uniform convergence implies ε(h) is at most 2γ higher than ε(h∗) = minh∈H ε(h) (as we showed previously).

This also quantifies what we were saying previously about the bias/variance tradeoff in model selection. Specifically, suppose we have some hypothesis class H, and are considering switching to some much larger hypothesis class H′ ⊇ H. If we switch to H′, then the first term minₕ ε(h) can only decrease (since we’d then be taking a min over a larger set of functions). Hence, by learning using a larger hypothesis class, our “bias” can only decrease. However, if k increases, then the second 2√· term would also increase. This increase corresponds to our “variance” increasing when we use a larger hypothesis class.

By holding γ and δ fixed and solving for n like we did before, we can also obtain the following sample complexity bound:

# Corollary.

Let |H| = k, and let any δ, γ be fixed. Then for ε ˆ (h) ≤ minh∈H ε(h) + 2γ to hold with probability at least 1 − δ, it suffices that

n ≥ 1 log 2k

2γ² δ = O ( 1 log k ),

γ² δ

# 8.3.3 The case of infinite H

We have proved some useful theorems for the case of finite hypothesis classes. But many hypothesis classes, including any parameterized by real numbers (as in linear classification) actually contain an infinite number of functions. Can we prove similar results for this setting?

Let’s start by going through something that is not the “right” argument. Better and more general arguments exist, but this will be useful for honing our intuitions about the domain.

Suppose we have an H that is parameterized by d real numbers. Since we are using a computer to represent real numbers, and IEEE double-precision floating point (double’s in C) uses 64 bits to represent a floating point number, this means that our learning algorithm, assuming we’re using double-precision floating point, is parameterized by 64d bits. Thus, our hypothesis class really consists of at most k = 2⁶⁴ᵈ different hypotheses. From the Corollary at the end of the previous section, we therefore find that, to guarantee





ε ˆ ∗ (h) ≤ ε(h ) + 2γ, with to hold with probability at least 1 − δ, it suffices that n ≥ O ( γ¹₂ log 2⁶⁴ᵈ ) = O ( d₂ log 1 ) = Oγ,δ(d). (The γ, δ subscripts indicate that the last big-O is hiding constants that may depend on γ and δ.) Thus, the number of training examples needed is at most linear in the parameters of the model.

The fact that we relied on 64-bit floating point makes this argument not entirely satisfying, but the conclusion is nonetheless roughly correct: If what we try to do is minimize training error, then in order to learn “well” using a hypothesis class that has d parameters, generally we’re going to need on the order of a linear number of training examples in d.

(At this point, it’s worth noting that these results were proved for an algorithm that uses empirical risk minimization. Thus, while the linear dependence of sample complexity on d does generally hold for most discriminative learning algorithms that try to minimize training error or some approximation to training error, these conclusions do not always apply as readily to discriminative learning algorithms. Giving good theoretical guarantees on many non-ERM learning algorithms is still an area of active research.)

The other part of our previous argument that’s slightly unsatisfying is that it relies on the parameterization of H. Intuitively, this doesn’t seem like it should matter: We had written the class of linear classifiers as hθ(x) = 1{θ₀ + θ₁x₁ + · · · θdxd ≥ 0}, with n + 1 parameters θ₀, . . . , θd. But it could also be written hu,v (x) = 1{(u² − v²) + (u² − v²)x₁ + · · · (u² − v²)xd ≥ 0} with 2d + 2 parameters u₀, v₀, u₁, v₁, . . . , u₁, v₁, d.

Yet, both of these are just defining the same H: The set of linear classifiers in d dimensions.

To derive a more satisfying argument, let’s define a few more things. Given a set S = {x(i), . . . , x(ᴰ)} (no relation to the training set) of points x(i) ∈ X, we say that H shatters S if H can realize any labeling on S. I.e., if for any set of labels {y(1), . . . , y(ᴰ)}, there exists some h ∈ H so that h(x(i)) = y(i) for all i = 1, . . . D.

Given a hypothesis class H, we then define its Vapnik-Chervonenkis dimension, written VC(H), to be the size of the largest set that is shattered by H. (If H can shatter arbitrarily large sets, then VC(H) = ∞.)

For instance, consider the following set of three points:





# 133

Can the set H of linear classifiers in two dimensions (h(x) = 1{θ₀ + θ₁x₁ + θ₂x₂ ≥ 0}) can shatter the set above? The answer is yes. Specifically, we see that, for any of the eight possible labelings of these points, we can find a linear classifier that obtains “zero training error” on them:

| x₂ | x₂ |
| -- | -- |
| x₁ | x₁ |
| x₂ | x₂ |
| x₁ | x₁ |

Moreover, it is possible to show that there is no set of 4 points that this hypothesis class can shatter. Thus, the largest set that H can shatter is of size 3, and hence VC(H) = 3.

Note that the VC dimension of H here is 3 even though there may be sets of size 3 that it cannot shatter. For instance, if we had a set of three points lying in a straight line (left figure), then there is no way to find a linear separator for the labeling of the three points shown below (right figure):





In order words, under the definition of the VC dimension, in order to prove that VC(H) is at least D, we need to show only that there’s at least one set of size D that H can shatter.

The following theorem, due to Vapnik, can then be shown. (This is, many would argue, the most important theorem in all of learning theory.)

# Theorem

Let H be given, and let D = VC(H). Then with probability at least 1 − δ, we have that for all h ∈ H,

(√)       ε(h) − ε̂(h) ≤ O(√(D log n + 1) log(1/δ))

Thus, with probability at least 1 − δ, we also have that:

(√)       ε̂(h*) ≤ ε(h) + O(√(D/n) log(D) + n log(1/δ))

In other words, if a hypothesis class has finite VC dimension, then uniform convergence occurs as n becomes large. As before, this allows us to give a bound on ε(h) in terms of ε(h∗). We also have the following corollary:

# Corollary

For |ε(h) − ε̂(h)| ≤ γ to hold for all h ∈ H (and hence ε(h) ≤ ε(h∗) + 2γ) with probability at least 1 − δ, it suffices that n = O(γ, δ(D)).

In other words, the number of training examples needed to learn “well” using H is linear in the VC dimension of H. It turns out that, for “most” hypothesis classes, the VC dimension (assuming a “reasonable” parameterization) is also roughly linear in the number of parameters. Putting these together, we conclude that for a given hypothesis class H (and for an algorithm that tries to minimize training error), the number of training examples needed to achieve generalization error close to that of the optimal classifier is usually roughly linear in the number of parameters of H.




# Chapter 9

# Regularization and model selection

# 9.1 Regularization

Recall that as discussed in Section 8.1, overfitting is typically a result of using too complex models, and we need to choose a proper model complexity to achieve the optimal bias-variance tradeoff. When the model complexity is measured by the number of parameters, we can vary the size of the model (e.g., the width of a neural net). However, the correct, informative complexity measure of the models can be a function of the parameters (e.g., L2 norm of the parameters), which may not necessarily depend on the number of parameters. In such cases, we will use regularization, an important technique in machine learning, to control the model complexity and prevent overfitting.

Regularization typically involves adding an additional term, called a regularizer and denoted by R(θ) here, to the training loss/cost function:

Jλ(θ) = J(θ) + λR(θ)
(9.1)

Here Jλ is often called the regularized loss, and λ ≥ 0 is called the regularization parameter. The regularizer R(θ) is a nonnegative function (in almost all cases). In classical methods, R(θ) is purely a function of the parameter θ, but some modern approaches allow R(θ) to depend on the training dataset.1

The regularizer R(θ) is typically chosen to be some measure of the complexity of the model θ. Thus, when using the regularized loss, we aim to find a model that both fits the data (a small loss J(θ)) and has a small.

1 Here our notations generally omit the dependency on the training dataset for simplicity—we write J(θ) even though it obviously needs to depend on the training dataset.



model complexity (a small R(θ)). The balance between the two objectives is controlled by the regularization parameter λ. When λ = 0, the regularized loss is equivalent to the original loss. When λ is a sufficiently small positive number, minimizing the regularized loss is effectively minimizing the original loss with the regularizer as the tie-breaker. When the regularizer is extremely large, then the original loss is not effective (and likely the model will have a large bias.)

The most commonly used regularization is perhaps `₂ regularization, where R(θ) = 1 ‖θ‖2. It encourages the optimizer to find a model with small `2 norm.

In deep learning, it’s oftentimes referred to as weight decay, because gradient descent with learning rate η on the regularized loss Rλ(θ) is equivalent to shrinking/decaying θ by a scalar factor of 1 − ηλ and then applying the standard gradient

θ ← θ − η∇Jλ(θ) = θ − ηλθ − η∇J (θ)
= (1 − λη)θ − η∇J (θ)                 (9.2)

Besides encouraging simpler models, regularization can also impose inductive biases or structures on the model parameters. For example, suppose we had a prior belief that the number of non-zeros in the ground-truth model parameters is small,2—which is oftentimes called sparsity of the model—, we can impose a regularization on the number of non-zeros in θ, denoted by ‖θ‖0, to leverage such a prior belief. Imposing additional structure of the parameters narrows our search space and makes the complexity of the model family smaller,—e.g., the family of sparse models can be thought of as having lower complexity than the family of all models—, and thus tends to lead to a better generalization. On the other hand, imposing additional structure may risk increasing the bias. For example, if we regularize the sparsity strongly but no sparse models can predict the label accurately, we will suffer from large bias (analogously to the situation when we use linear models to learn data than can only be represented by quadratic functions in Section 8.1.)

The sparsity of the parameters is not a continuous function of the parameters, and thus we cannot optimize it with (stochastic) gradient descent. A common relaxation is to use R(θ) = ‖θ‖1 as a continuous surrogate.3

2For linear models, this means the model just uses a few coordinates of the inputs to make an accurate prediction.

3There has been a rich line of theoretical work that explains why ‖θ‖1 is a good surrogate for encouraging sparsity, but it’s beyond the scope of this course. An intuition is: assuming the parameter is on the unit sphere, the parameter with smallest `1 norm also





The R(θ) = ‖θ‖₁ (also called LASSO) and R(θ) = 1 ‖θ‖2 are perhaps among the most commonly used regularizers for linear models. Other norm and powers of norms are sometimes also used. The `₂ norm regularization is much more commonly used with kernel methods because `₁ regularization is typically not compatible with the kernel trick (the optimal solution cannot be written as functions of inner products of features.)

In deep learning, the most commonly used regularizer is `₂ regularization or weight decay. Other common ones include dropout, data augmentation, regularizing the spectral norm of the weight matrices, and regularizing the Lipschitzness of the model, etc. Regularization in deep learning is an active research area, and it’s known that there is another implicit source of regularization, as discussed in the next section.

# 9.2 Implicit regularization effect (optional reading)

The implicit regularization effect of optimizers, or implicit bias or algorithmic regularization, is a new concept/phenomenon observed in the deep learning era. It largely refers to that the optimizers can implicitly impose structures on parameters beyond what has been imposed by the regularized loss.

In most classical settings, the loss or regularized loss has a unique global minimum, and thus any reasonable optimizer should converge to that global minimum and cannot impose any additional preferences. However, in deep learning, oftentimes the loss or regularized loss has more than one (approximate) global minima, and difference optimizers may converge to different global minima. Though these global minima have the same or similar training losses, they may be of different nature and have dramatically different generalization performance. See Figures 9.1 and 9.2 and its caption for an illustration and some experiment results. For example, it’s possible that one global minimum gives a much more Lipschitz or sparse model than others and thus has a better test error. It turns out that many commonly-used optimizers (or their components) prefer or bias towards finding global minima of certain properties, leading to a better test performance.

happen to be the sparsest parameter with only 1 non-zero coordinate. Thus, sparsity and `1 norm gives the same extremal points to some extent.






# 9.1

loss

θ

Figure 9.1: An Illustration that different global minima of the training loss can have different test performance.

| 5   | 30       | 2         | CIFAR-10                    | Ir = 0.1 → 0.01       | 2.00 | Quadratically Parameterized Model |
| --- | -------- | --------- | --------------------------- | --------------------- | ---- | --------------------------------- |
| 25  |          | Ir = 0.01 | 175                         | test error, init.=0.1 |      |                                   |
| 20  |          | 150       | test error, init.=0.001     |                       |      |                                   |
| 0₁₅ |          | 20¹²⁵     | training error, init.=0.001 |                       |      |                                   |
| 10  |          | 0.75      |                             |                       |      |                                   |
| 5   |          | 0.50      |                             |                       |      |                                   |
| 0   | training | 0.25      |                             |                       |      |                                   |

0  25  50  75  100  125  150   175    200    O.DO    0  2ID  4ID  E4D  aID                  1000

Figure 9.2: Left: Performance of neural networks trained by two different learning rates schedules on the CIFAR-10 dataset. Although both experiments used exactly the same regularized losses and the optimizers fit the training data perfectly, the models’ generalization performance differ much. Right: On a different synthetic dataset, optimizers with different initializations have the same training error but different generalization performance.

In summary, the takehome message here is that the choice of optimizer does not only affect minimizing the training loss, but also imposes implicit regularization and affects the generalization of the model. Even if your current optimizer already converges to a small training error perfectly, you may still need to tune your optimizer for a better generalization.

4The setting is the same as in Woodworth et al. [2020], HaoChen et al. [2020]

# 2

test

# 1

training

0  0.5  1.0  1.5  2.0  2.5  3.0

0.0     Good      Not so good

global min    global min





One may wonder which components of the optimizers bias towards what type of global minima and what type of global minima may generalize better. These are open questions that researchers are actively investigating. Empirical and theoretical research have offered some clues and heuristics. In many (but definitely far from all) situations, among those setting where optimization can succeed in minimizing the training loss, the use of larger initial learning rate, smaller initialization, smaller batch size, and momentum appears to help with biasing towards more generalizable solutions. A conjecture (that can be proven in certain simplified case) is that stochasticity in the optimization process help the optimizer to find flatter global minima (global minima where the curvature of the loss is small), and flat global minima tend to give more Lipschitz models and better generalization. Characterizing the implicit regularization effect formally is still a challenging open research question.

# 9.3 Model selection via cross validation

Suppose we are trying select among several different models for a learning problem. For instance, we might be using a polynomial regression model hθ(x) = g(θ0 + θ1x + θ2x² + · · · + θkxk), and wish to decide if k should be 0, 1, . . . , or 10. How can we automatically select a model that represents a good tradeoff between the twin evils of bias and variance5? Alternatively, suppose we want to automatically choose the bandwidth parameter τ for locally weighted regression, or the parameter C for our ℓ1-regularized SVM. How can we do that?

For the sake of concreteness, in these notes we assume we have some finite set of models M = {M1, . . . , Md} that we’re trying to select among. For instance, in our first example above, the model Mi would be an i-th degree polynomial regression model. (The generalization to infinite M is not hard.6) Alternatively, if we are trying to decide between using an SVM, a neural network or logistic regression, then M may contain these models.

5 Given that we said in the previous set of notes that bias and variance are two very different beasts, some readers may be wondering if we should be calling them “twin” evils here. Perhaps it’d be better to think of them as non-identical twins. The phrase “the fraternal twin evils of bias and variance” doesn’t have the same ring to it, though.

6 If we are trying to choose from an infinite set of models, say corresponding to the possible values of the bandwidth τ ∈ R+, we may discretize τ and consider only a finite number of possible values for it. More generally, most of the algorithms described here can all be viewed as performing optimization search in the space of models, and we can perform this search over infinite model classes as well.




Cross validation. Lets suppose we are, as usual, given a training set S. Given what we know about empirical risk minimization, here’s what might initially seem like a algorithm, resulting from using empirical risk minimization for model selection:

1. Train each model Mi on S, to get some hypothesis hi.
2. Pick the hypotheses with the smallest training error.

This algorithm does not work. Consider choosing the degree of a polynomial. The higher the degree of the polynomial, the better it will fit the training set S, and thus the lower the training error. Hence, this method will always select a high-variance, high-degree polynomial model, which we saw previously is often poor choice.

Here’s an algorithm that works better. In hold-out cross validation (also called simple cross validation), we do the following:

1. Randomly split S into Strain (say, 70% of the data) and Scv (the remaining 30%). Here, Scv is called the hold-out cross validation set.
2. Train each model Mi on Strain only, to get some hypothesis hi.
3. Select and output the hypothesis h that had the smallest error εi(h) on the hold out cross validation set. (Here εi(h) denotes the average error of h on the set of examples in Scv.) The error on the hold out validation set is also referred to as the validation error.

By testing/validating on a set of examples Scv that the models were not trained on, we obtain a better estimate of each hypothesis hi’s true generalization/test error. Thus, this approach is essentially picking the model with the smallest estimated generalization/test error. The size of the validation set depends on the total number of available examples. Usually, somewhere between 1/4 − 1/3 of the data is used in the hold out cross validation set, and 30% is a typical choice. However, when the total dataset is huge, the validation set can be a smaller fraction of the total examples as long as the absolute number of validation examples is decent. For example, for the ImageNet dataset that has about 1M training images, the validation set is sometimes set to be 50K images, which is only about 5% of the total examples.

Optionally, step 3 in the algorithm may also be replaced with selecting the model M according to arg min εi(hi), and then retraining Mi on the training set S. (This is often a good idea, with one exception being learning algorithms that are be very sensitive to perturbations of the initial.)



conditions and/or data. For these methods, Mi doing well on Strain does not necessarily mean it will also do well on Scv, and it might be better to forgo this retraining step.

The disadvantage of using hold out cross validation is that it “wastes” about 30% of the data. Even if we were to take the optional step of retraining the model on the entire training set, it’s still as if we’re trying to find a good model for a learning problem in which we had 0.7n training examples, rather than n training examples, since we’re testing models that were trained on only 0.7n examples each time. While this is fine if data is abundant and/or cheap, in learning problems in which data is scarce (consider a problem with n = 20, say), we’d like to do something better.

# Here is a method, called k-fold cross validation, that holds out less data each time:

1. Randomly split S into k disjoint subsets of m/k training examples each. Lets call these subsets S₁, . . . , Sₖ.
2. For each model Mi, we evaluate it as follows:
1. For j = 1, . . . , k
Train the model Mi on S₁ ∪ · · · ∪ Sj−1 ∪ Sj₊₁ ∪ · · · Sₖ (i.e., train on all the data except Sj ) to get some hypothesis hij.
Test the hypothesis h on S, to get εj(hij).

The estimated generalization error of model Mi is then calculated as the average of the εj(hij)’s (averaged over j).
3. Pick the model Mi with the lowest estimated generalization error, and retrain that model on the entire training set S. The resulting hypothesis is then output as our final answer.

A typical choice for the number of folds to use here would be k = 10. While the fraction of data held out each time is now 1/k—much smaller than before—this procedure may also be more computationally expensive than hold-out cross validation, since we now need to train each model k times.

While k = 10 is a commonly used choice, in problems in which data is really scarce, sometimes we will use the extreme choice of k = m in order to leave out as little data as possible each time. In this setting, we would repeatedly train on all but one of the training examples in S, and test on that held-out example. The resulting m = k errors are then averaged together to obtain our estimate of the generalization error of a model. This method has





its own name; since we’re holding out one training example at a time, this method is called leave-one-out cross validation.

Finally, even though we have described the different versions of cross validation as methods for selecting a model, they can also be used more simply to evaluate a single model or algorithm. For example, if you have implemented some learning algorithm and want to estimate how well it performs for your application (or if you have invented a novel learning algorithm and want to report in a technical paper how well it performs on various test sets), cross validation would give a reasonable way of doing so.

# 9.4 Bayesian statistics and regularization

In this section, we will talk about one more tool in our arsenal for our battle against overfitting.

At the beginning of the quarter, we talked about parameter fitting using maximum likelihood estimation (MLE), and chose our parameters according to

θMLE = arg maxθ ∏n p(y(i) | x(i); θ).

Throughout our subsequent discussions, we viewed θ as an unknown parameter of the world. This view of the θ as being constant-valued but unknown is taken in frequentist statistics. In the frequentist this view of the world, θ is not random—it just happens to be unknown—and it’s our job to come up with statistical procedures (such as maximum likelihood) to try to estimate this parameter.

An alternative way to approach our parameter estimation problems is to take the Bayesian view of the world, and think of θ as being a random variable whose value is unknown. In this approach, we would specify a prior distribution p(θ) on θ that expresses our “prior beliefs” about the parameters. Given a training set S = {(x(i), y(i))n, when we are asked to make a prediction on a new value of x, we can then compute the posterior distribution on the parameters

p(θ | S) = p(S | θ)p(θ) / p(S)

= (∏i=1n p(y(i) | x(i), θ)) p(θ)

= ∫ (∏i=1n p(y(i) | x(i), θ)p(θ)) dθ

(9.3)

In the equation above, p(y(i) | x(i), θ) comes from whatever model you’re using





# 9. Bayesian Logistic Regression

For your learning problem. For example, if you are using Bayesian logistic regression, then you might choose *p(y(i) | x(i), θ) = hθ(x(i))y(i)(1 − hθ(x(i)))(1−y(i)), where hθ(x(i)) = 1/(1 + exp(−θᵀ x(i)))*.7

When we are given a new test example *x and asked to make a prediction on it, we can compute our posterior distribution on the class label using the posterior distribution on θ*:

*p(y | x, S) = ∫ p(y | x, θ)p(θ | S)dθ* (9.4)

In the equation above, *p(θ | S) comes from Equation (9.3). Thus, for example, if the goal is to predict the expected value of *y* given x*, then we would output8

*E[y | x, S] = ∫ yp(y | x, S)dy*

*y*

The procedure that we’ve outlined here can be thought of as doing “fully Bayesian” prediction, where our prediction is computed by taking an average with respect to the posterior *p(θ | S) over θ. Unfortunately, in general it is computationally very difficult to compute this posterior distribution. This is because it requires taking integrals over the (usually high-dimensional) θ* as in Equation (9.3), and this typically cannot be done in closed-form.

Thus, in practice we will instead approximate the posterior distribution for *θ. One common approximation is to replace our posterior distribution for θ (as in Equation 9.4) with a single point estimate. The MAP (maximum a posteriori) estimate for θ* is given by

*θMAP = arg maxθ ∏i=1n p(y(i) | x(i), θ)p(θ).* (9.5)

Note that this is the same formula as for the MLE (maximum likelihood) estimate for *θ, except for the prior p(θ)* term at the end.

In practical applications, a common choice for the prior *p(θ) is to assume that θ ∼ N (0, τ2I). Using this choice of prior, the fitted parameters θMAP will have smaller norm than that selected by maximum likelihood. In practice, this causes the Bayesian MAP estimate to be less susceptible to overfitting than the ML estimate of the parameters. For example, Bayesian logistic regression turns out to be an effective algorithm for text classification, even though in text classification we usually have d ≪ n*.

7 Since we are now viewing *θ as a random variable, it is okay to condition on its value, and write “p(y | x, θ)” instead of “p(y | x; θ)*.”

8 The integral below would be replaced by a summation if y is discrete-valued.




# Part IV

# Unsupervised learning

144


Chapter 10

# Clustering and the k-means algorithm

In the clustering problem, we are given a training set {x(1), . . . , x(ⁿ)}, and want to group the data into a few cohesive “clusters.” Here, x(i) ∈ Rᵈ as usual; but no labels y(i) are given. So, this is an unsupervised learning problem.

The k-means clustering algorithm is as follows:

1. Initialize cluster centroids μ₁, μ₂, . . . , μₖ ∈ Rᵈ randomly.
2. Repeat until convergence: {
- For every i, set c(i) := arg min ||x(i) − μj ||².
- For each j, set μj := ∑ₙ 1{c(i) = j}x(i) / ∑ₙ 1{c(i) = j}.

}

In the algorithm above, k (a parameter of the algorithm) is the number of clusters we want to find; and the cluster centroids μj represent our current guesses for the positions of the centers of the clusters. To initialize the cluster centroids (in step 1 of the algorithm above), we could choose k training examples randomly, and set the cluster centroids to be equal to the values of these k examples. (Other initialization methods are also possible.)

The inner-loop of the algorithm repeatedly carries out two steps: (i) “Assigning” each training example x(i) to the closest cluster centroid μj, and

145



# 10.1 K-means Algorithm

Training examples are shown as dots, and cluster centroids are shown as crosses. (a) Original dataset. (b) Random initial cluster centroids (in this instance, not chosen to be equal to two training examples). (c-f) Illustration of running two iterations of k-means. In each iteration, we assign each training example to the closest cluster centroid (shown by “painting” the training examples the same color as the cluster centroid to which it is assigned); then we move each cluster centroid to the mean of the points assigned to it. (Best viewed in color.) Images courtesy Michael Jordan.

(ii) Moving each cluster centroid μj to the mean of the points assigned to it. Figure 10.1 shows an illustration of running k-means.

Is the k-means algorithm guaranteed to converge? Yes it is, in a certain sense. In particular, let us define the distortion function to be:

∑i=1n J(c, μ) = ∑i=1n ||x(i) − μc(i)||²

Thus, J measures the sum of squared distances between each training example x(i) and the cluster centroid μc(i) to which it has been assigned. It can be shown that k-means is exactly coordinate descent on J. Specifically, the inner-loop of k-means repeatedly minimizes J with respect to c while holding μ fixed, and then minimizes J with respect to μ while holding c fixed. Thus,




J must monotonically decrease, and the value of J must converge. (Usually, this implies that c and μ will converge too. In theory, it is possible for k-means to oscillate between a few different clusterings—i.e., a few different values for c and/or μ—that have exactly the same value of J, but this almost never happens in practice.)

The distortion function J is a non-convex function, and so coordinate descent on J is not guaranteed to converge to the global minimum. In other words, k-means can be susceptible to local optima. Very often k-means will work fine and come up with very good clusterings despite this. But if you are worried about getting stuck in bad local minima, one common thing to do is run k-means many times (using different random initial values for the cluster centroids μj). Then, out of all the different clusterings found, pick the one that gives the lowest distortion J (c, μ).



# Chapter 11

# EM algorithms

In this set of notes, we discuss the EM (Expectation-Maximization) algorithm for density estimation.

# 11.1 EM for mixture of Gaussians

Suppose that we are given a training set {x(1), . . . , x(ⁿ)} as usual. Since we are in the unsupervised learning setting, these points do not come with any labels.

We wish to model the data by specifying a joint distribution p(x(i), z(i)) = p(x(i) | z(i))p(z(i)). Here, z(i) ∼ Multinomial(φ) (where φj ≥ 0, ∑j=1k φj = 1, and the parameter φj gives p(z(i) = j)), and x(i) | z(i) = j ∼ N (μj, Σj). We let k denote the number of values that the z(i)’s can take on. Thus, our model posits that each x(i) was generated by randomly choosing z(i) from {1, . . . , k}, and then x(i) was drawn from one of k Gaussians depending on z(i). This is called the mixture of Gaussians model. Also, note that the z(i)’s are latent random variables, meaning that they’re hidden/unobserved. This is what will make our estimation problem difficult.

The parameters of our model are thus φ, μ and Σ. To estimate them, we can write down the likelihood of our data:

∑i=1n `(φ, μ, Σ) = log p(x(i); φ, μ, Σ)

However, if we set to zero the derivatives of this formula with respect to





the parameters and try to solve, we’ll find that it is not possible to find the maximum likelihood estimates of the parameters in closed form.  (Try this yourself at home.)

The random variables z(i) indicate which of the k Gaussians each x(i) had come from. Note that if we knew what the z(i)’s were, the maximum likelihood problem would have been easy. Specifically, we could then write down the likelihood as

∑

n

`(φ, μ, Σ) =                       log p(x(i)|z(i); μ, Σ) + log p(z(i); φ).

i=1

Maximizing this with respect to φ, μ and Σ gives the parameters:

∑

φj     =  1 n      1{z(i) = j},

n i=1

∑ₙ       1{z(i) = j}x(i)

∑

μj     =    i=1                   ,

n     1{z(i) = j}

∑ₙ i=1   (i)         (i)    (i)    T

1{z    = j}(x      − μj )(x  − μj )

Σj     =    i=1         ∑ₙ 1{z(i) = j}                .

i=1

Indeed, we see that if the z(i)’s were known, then maximum likelihood estimation becomes nearly identical to what we had when estimating the parameters of the Gaussian discriminant analysis model, except that here the z(i)’s playing the role of the class labels.1

However, in our density estimation problem, the z(i)’s are not known. What can we do?

The EM algorithm is an iterative algorithm that has two main steps. Applied to our problem, in the E-step, it tries to “guess” the values of the z(i)’s. In the M-step, it updates the parameters of our model based on our guesses. Since in the M-step we are pretending that the guesses in the first part were correct, the maximization becomes easy. Here’s the algorithm:

Repeat until convergence: {

(E-step) For each i, j, set

w(i) := p(z(i) = j|x(i); φ, μ, Σ)

1There are other minor differences in the formulas here from what we’d obtained in PS1 with Gaussian discriminant analysis, first because we’ve generalized the z(i)’s to be multinomial rather than Bernoulli, and second because here we are using a different Σj for each Gaussian.





# 150

# (M-step) Update the parameters:

∑

φj := 1 n w(i),

n ∑ w(i)x(i)

μj := ∑ n w(i) x(i),

∑ n w(i)

Σj := ∑ wj (x(i) − μj)(x(i) − μj)T

n ∑ w(i)

i=1 j

In the E-step, we calculate the posterior probability of our parameters the z(i)’s, given the x(i) and using the current setting of our parameters. I.e., using Bayes rule, we obtain:

p(z(i) = j | x(i); φ, μ, Σ) = ∑ p(x(i) | z(i) = j; μ, Σ)p(z(i) = j; φ)



# 11.2 Jensen’s inequality

We begin our discussion with a very useful result called Jensen’s inequality. Let f be a function whose domain is the set of real numbers. Recall that f is a convex function if f ′′(x) ≥ 0 (for all x ∈ R). In the case of f taking vector-valued inputs, this is generalized to the condition that its hessian H is positive semi-definite (H ≥ 0). If f ′′(x) > 0 for all x, then we say f is strictly convex (in the vector-valued case, the corresponding statement is that H must be positive definite, written H > 0). Jensen’s inequality can then be stated as follows:

Theorem. Let f be a convex function, and let X be a random variable. Then:

E[f (X)] ≥ f (EX).

Moreover, if f is strictly convex, then E[f (X)] = f (EX) holds true if and only if X = E[X] with probability 1 (i.e., if X is a constant). Recall our convention of occasionally dropping the parentheses when writing expectations, so in the theorem above, f (EX) = f (E[X]).

For an interpretation of the theorem, consider the figure below.

| f(a) | f     | E\[f(X)] |
| ---- | ----- | -------- |
| f(b) | f(EX) |          |
| a    | E\[X] | b        |

Here, f is a convex function shown by the solid line. Also, X is a random variable that has a 0.5 chance of taking the value a, and a 0.5 chance of





taking the value b (indicated on the x-axis). Thus, the expected value of X is given by the midpoint between a and b. We also see the values f (a), f (b) and f (E[X]) indicated on the y-axis. Moreover, the value E[f (X)] is now the midpoint on the y-axis between f (a) and f (b). From our example, we see that because f is convex, it must be the case that E[f (X)] ≥ f (EX). Incidentally, quite a lot of people have trouble remembering which way the inequality goes, and remembering a picture like this is a good way to quickly figure out the answer.

Remark. Recall that f is [strictly] concave if and only if −f is [strictly] convex (i.e., f ′′(x) ≤ 0 or H ≤ 0). Jensen’s inequality also holds for concave functions f, but with the direction of all the inequalities reversed (E[f (X)] ≤ f (EX), etc.).

# 11.3 General EM algorithms

Suppose we have an estimation problem in which we have a training set {x(1), . . . , x(ⁿ)} consisting of n independent examples. We have a latent variable model p(x, z; θ) with z being the latent variable (which for simplicity is assumed to take finite number of values). The density for x can be obtained by marginalized over the latent variable z:

p(x; θ) = ∑ p(x, z; θ)
z

We wish to fit the parameters θ by maximizing the log-likelihood of the data, defined by

∑
n
`(θ) = log p(x(i); θ)
i=1

We can rewrite the objective in terms of the joint density p(x, z; θ) by

∑
n
`(θ) = log p(x(i); θ)
i=1

∑
n
= log p(x(i), z(i); θ).
i=1
z(i)

But, explicitly finding the maximum likelihood estimates of the parameters θ may be hard since it will result in difficult non-convex optimization prob-





lems.³ Here, the z(i)’s are the latent random variables; and it is often the case that if the z(i)’s were observed, then maximum likelihood estimation would be easy.

In such a setting, the EM algorithm gives an efficient method for maximum likelihood estimation. Maximizing `(θ) explicitly might be difficult, and our strategy will be to instead repeatedly construct a lower-bound on `(E-step), and then optimize that lower-bound (M-step).⁴

It turns out that the summation ∑i=1n is not essential here, and towards a simpler exposition of the EM algorithm, we will first consider optimizing the likelihood log p(x) for a single example x. After we derive the algorithm for optimizing log p(x), we will convert it to an algorithm that works for n examples by adding back the sum to each of the relevant equations. Thus, now we aim to optimize log p(x; θ) which can be rewritten as

log p(x; θ) = log ∑z p(x, z; θ) (11.5)

Let Q be a distribution over the possible values of z. That is, ∑z Q(z) = 1, Q(z) ≥ 0).

Consider the following:⁵

log p(x; θ) = log ∑z p(x, z; θ)

= log ∑z Q(z)p(x, z; θ) (11.6)

∑z Q(z) ≥ ∑z Q(z) log p(x, z; θ) (11.7)

The last step of this derivation used Jensen’s inequality. Specifically, f (x) = log x is a concave function, since f ′′(x) = −1/x² &#x3C; 0 over its domain.

³ It’s mostly an empirical observation that the optimization problem is difficult to optimize.

⁴ Empirically, the E-step and M-step can often be computed more efficiently than optimizing the function `(·) directly. However, it doesn’t necessarily mean that alternating the two steps can always converge to the global optimum of `(·). Even for mixture of Gaussians, the EM algorithm can either converge to a global optimum or get stuck, depending on the properties of the training data. Empirically, for real-world data, often EM can converge to a solution with relatively high likelihood (if not the optimum), and the theory behind it is still largely not understood.

⁵ If z were continuous, then Q would be a density, and the summations over z in our discussion are replaced with integrals over z.





# 154

x ∈ R⁺. Also, the term

∑ Q(z) [ p(x, z; θ)]

in the summation is just an expectation of the quantity [p(x, z; θ)/Q(z)] with respect to z drawn according to the distribution given by Q.⁶ By Jensen’s inequality, we have

f (Ez∼Q [ p(x, z; θ)]) ≥ Ez∼Q [f (p(x, z; θ))],

where the “z ∼ Q” subscripts above indicate that the expectations are with respect to z drawn from Q. This allowed us to go from Equation (11.6) to Equation (11.7.

Now, for any distribution Q, the formula (11.7) gives a lower-bound on log p(x; θ). There are many possible choices for the Q’s. Which should we choose? Well, if we have some current guess θ of the parameters, it seems natural to try to make the lower-bound tight at that value of θ. I.e., we will make the inequality above hold with equality at our particular value of θ.

To make the bound tight for a particular value of θ, we need for the step involving Jensen’s inequality in our derivation above to hold with equality. For this to be true, we know it is sufficient that the expectation be taken over a “constant”-valued random variable. I.e., we require that

p(x, z; θ) = c

Q(z) for some constant c that does not depend on z. This is easily accomplished by choosing

Actually, since we know ∑ Q(z) ∝ p(x, z; θ). further tells us that z Q(z) = 1 (because it is a distribution), this

p(x, z; θ)

Q(z) = ∑z p(x, z; θ)

= p(x, z; θ)

p(x; θ)

= p(z|x; θ) (11.8)

6We note that the notion p(x,z;θ) only makes sense if Q(z) = 0 whenever p(x, z; θ) = 0. Here we implicitly assume that we only consider those Q with such a property.





Thus, we simply set the Q’s to be the posterior distribution of the z’s given x and the setting of the parameters θ. Indeed, we can directly verify that when Q(z) = p(z|x; θ), then equation (11.7) is an equality because

∑ Q(z) log p(x, z; θ) = ∑ p(z|x; θ) log p(x, z; θ)

z Q(z) z p(z|x; θ)

= ∑ p(z|x; θ) log p(z|x; θ)p(x; θ)

z p(z|x; θ)

= ∑ p(z|x; θ) log p(x; θ)

= z ∑ log p(x; θ) p(z|x; θ)

= log p(x; θ) z (because ∑z p(z|x; θ) = 1)

For convenience, we call the expression in Equation (11.7) the evidence lower bound (ELBO) and we denote it by

ELBO(x; Q, θ) = ∑ Q(z) log p(x, z; θ) (11.9)

z Q(z)

With this equation, we can re-write equation (11.7) as

∀Q, θ, x, log p(x; θ) ≥ ELBO(x; Q, θ) (11.10)

Intuitively, the EM algorithm alternatively updates Q and θ by a) setting Q(z) = p(z|x; θ) following Equation (11.8) so that ELBO(x; Q, θ) = log p(x; θ) for x and the current θ, and b) maximizing ELBO(x; Q, θ) w.r.t θ while fixing the choice of Q.

Recall that all the discussion above was under the assumption that we aim to optimize the log-likelihood log p(x; θ) for a single example x. It turns out that with multiple training examples, the basic idea is the same and we only needs to take a sum over examples at relevant places. Next, we will build the evidence lower bound for multiple training examples and make the EM algorithm formal.

Recall we have a training set {x(1), . . . , x(ⁿ)}. Note that the optimal choice of Q is p(z|x; θ), and it depends on the particular example x. Therefore here we will introduce n distributions Q₁, . . . , Qₙ, one for each example x(i). For each example x(i), we can build the evidence lower bound

log p(x(i); θ) ≥ ELBO(x(i); Qi, θ) = ∑ Qi(z(i)) log p(x(i), z(i); θ)

z(i) Qi(z(i))





Taking sum over all the examples, we obtain a lower bound for the log-likelihood

`(θ) ≥ ∑ ELBO(x(i); Qi, θ)                           (11.11)

i

= ∑ ∑ Qi(z(i)) log p(x(i), z(i); θ)

i  z(i)                       Qi(z(i))

For any set of distributions Q₁, . . . , Qₙ, the formula (11.11) gives a lower-bound on `(θ), and analogous to the argument around equation (11.8), the Qi that attains equality satisfies

Qi(z(i))       = p(z(i)|x(i); θ)

Thus, we simply set the Qi’s to be the posterior distribution of the z(i)’s given x(i) with the current setting of the parameters θ.

Now, for this choice of the Qi’s, Equation (11.11) gives a lower-bound on the loglikelihood ` that we’re trying to maximize. This is the E-step. In the M-step of the algorithm, we then maximize our formula in Equation (11.11) with respect to the parameters to obtain a new setting of the θ’s. Repeatedly carrying out these two steps gives us the EM algorithm, which is as follows:

Repeat until convergence {
(E-step) For each i, set
Qi(z(i)) := p(z(i)|x(i); θ).
(M-step) Set
∑
n
θ := arg max           ELBO(x(i); Qi, θ)
θ  i=1
= arg max ∑ ∑ Qi(z(i)) log p(x(i), z(i); θ).  (11.12)
θ     i        z(i)             Qi(z(i))
}
How do we know if this algorithm will converge? Well, suppose θ(t) and θ(t+1) are the parameters from two successive iterations of EM. We will now prove that `(θ(t))         ≤ `(θ(t+1)), which shows EM always monotonically improves the log-likelihood. The key to showing this result lies in our choice of





the Qi’s. Specifically, on the iteration of EM in which the parameters had started out as θ(t), we would have chosen Q(t)(z(i)) := p(z(i)|x(i); θ(t)). We saw earlier that this choice ensures that Jensen’s inequality, as applied to get Equation (11.11), holds with equality, and hence

∑i=1n `(θ(t)) = ELBO(x(i); Q(t), θ(t)) (11.13)

The parameters θ(t+1) are then obtained by maximizing the right hand side of the equation above. Thus,

∑i=1n `(θ(t+1)) ≥ ELBO(x(i); Q(t), θ(t+1))

(because inequality (11.11) holds for all Q and θ)

∑i=1n ≥ ELBO(x(i); Q(t), θ(t)) (see reason below)

∑i=1n = `(θ(t)) (by equation (11.13))

where the last inequality follows from that θ(t+1) is chosen explicitly to be

∑i=1n arg maxθ ELBO(x(i); Q(t), θ)

Hence, EM causes the likelihood to converge monotonically. In our description of the EM algorithm, we said we’d run it until convergence. Given the result that we just showed, one reasonable convergence test would be to check if the increase in `(θ) between successive iterations is smaller than some tolerance parameter, and to declare convergence if EM is improving `(θ) too slowly.

Remark. If we define (by overloading ELBO(·))

∑i=1n ELBO(Q, θ) = ∑i=1n ELBO(x(i); Qi, θ) = ∑i=1n ∑z(i) Qi(z(i)) log p(x(i), z(i); θ)

then we know `(θ) ≥ ELBO(Q, θ) from our previous derivation. The EM can also be viewed as an alternating maximization algorithm on ELBO(Q, θ), in which the E-step maximizes it with respect to Q (check this yourself), and the M-step maximizes it with respect to θ.





# 11.3.1 Other interpretation of ELBO

Let ELBO(x; Q, θ) = ∑z Q(z) log p(x,z;θ) be defined as in equation (11.9). There are several other forms of ELBO. First, we can rewrite

ELBO(x; Q, θ) = Ez∼Q[log p(x, z; θ)] − Ez∼Q[log Q(z)]

= Ez∼Q[log p(x|z; θ)] − DKL(Q‖pz)  (11.15)

where we use pz to denote the marginal distribution of z (under the distribution p(x, z; θ)), and DKL() denotes the KL divergence

DKL(Q‖pz) = ∑ Q(z) log Q(z)                              (11.16)

z             p(z)

In many cases, the marginal distribution of z does not depend on the parameter θ. In this case, we can see that maximizing ELBO over θ is equivalent to maximizing the first term in (11.15). This corresponds to maximizing the conditional likelihood of x conditioned on z, which is often a simpler question than the original question.

Another form of ELBO(·) is (please verify yourself)

ELBO(x; Q, θ) = log p(x) − DKL(Q‖pz|ₓ)                     (11.17)

where pz|ₓ is the conditional distribution of z given x under the parameter θ. This forms shows that the maximizer of ELBO(Q, θ) over Q is obtained when Q = pz|ₓ, which was shown in equation (11.8) before.

# 11.4 Mixture of Gaussians revisited

Armed with our general definition of the EM algorithm, let’s go back to our old example of fitting the parameters φ, μ and Σ in a mixture of Gaussians. For the sake of brevity, we carry out the derivations for the M-step updates only for φ and μj , and leave the updates for Σj as an exercise for the reader.

The E-step is easy. Following our algorithm derivation above, we simply calculate

w(i) = Qi(z(i) = j) = P (z(i) = j|x(i); φ, μ, Σ).

Here, “Qi(z(i) = j)” denotes the probability of z(i) taking the value j under the distribution Qi.





Next, in the M-step, we need to maximize, with respect to our parameters φ, μ, Σ, the quantity

∑ ∑

n Qi(z(i)) log p(x(i), z(i); φ, μ, Σ)

i=1 z(i)

∑ ∑

= n k Qi(z(i) = j) log p(x(i) | z(i) = j; μ, Σ)p(z(i) = j; φ)

i=1 j=1 1 ( Qi(z(i) = j) − 1 )

∑ ∑ exp − 1 (x(i) − μ )ᵀ Σ (x(i) − μ ) · φ

= n k w(i) log (2π)d/2Σj1/2

2 j j j j

w(i)

i=1 j=1

Let’s maximize this with respect to μl. If we take the derivative with respect to μl, we find

∑ ∑ 1 exp (− 1 (x(i) − μ )ᵀ Σ−1(x(i) − μ )) · φ

∇μ n k w(i) log (2π)d/2Σj1/2

2 j j j j

l j

w(i)

i=1 j=1

∑ ∑

= −∇μ n k w(i) 1 (x(i) − μj)ᵀ Σ−1(x(i) − μj)

l i=1 j=1 j 2 j

∑

= 1 n w(i)∇μ 2μᵀ Σ−1x(i) − μᵀ Σ−1μl

2 i=1 l l l l l

∑

= n w(i) (Σ−1x(i) − Σ−1μl)

i=1 l l l

Setting this to zero and solving for μl therefore yields the update rule

∑n w(i)x(i)

μ := i=1 l ,

l ~~∑~~ n w(i)

i=1 l

which was what we had in the previous set of notes.

Let’s do one more example, and derive the M-step update for the parameters φj. Grouping together only the terms that depend on φj, we find that we need to maximize

∑ ∑

n k w(i) log φj.

i=1 j=1

However, there is an additional constraint that the φj’s sum to 1, since they represent the probabilities φj = p(z(i) = j; φ). To deal with the constraint





that ∑k φj = 1, we construct the Lagrangian

L(φ) = ∑i=1ⁿ ∑j=1ⁿ w(i) log φj + β(∑j=1ⁿ φj − 1),

where β is the Lagrange multiplier.7 Taking derivatives, we find

∂ L(φ) / ∂φj = ∑i=1ⁿ wj + β

Setting this to zero and solving, we get

φj = ∑i=1ⁿ w(i) / (−β)

I.e., φj ∝ ∑i=1ⁿ w(i). Using the constraint that ∑j φj = 1, we easily find that −β = ∑i=1ⁿ ∑j=1 w(i) = ∑i=1ⁿ 1 = n. (This used the fact that w(i) = ∑j=1ⁿ wj = 1.)

Qi(z = j), and since probabilities sum to 1, we therefore have our M-step updates for the parameters φj:

φj := (1/n) ∑i=1ⁿ w(i).

The derivation for the M-step updates to Σj are also entirely straightforward.

# 11.5 Variational inference and variational auto-encoder (optional reading)

Loosely speaking, variational auto-encoder Kingma and Welling [2013] generally refers to a family of algorithms that extend the EM algorithms to more complex models parameterized by neural networks. It extends the technique of variational inference with the additional “re-parametrization trick” which will be introduced below. Variational auto-encoder may not give the best performance for many datasets, but it contains several central ideas about how to extend EM algorithms to high-dimensional continuous latent variables.

7We don’t need to worry about the constraint that φj ≥ 0, because as we’ll shortly see, the solution we’ll find from this derivation will automatically satisfy that anyway.





with non-linear models. Understanding it will likely give you the language and backgrounds to understand various recent papers related to it.

As a running example, we will consider the following parameterization of p(x, z; θ) by a neural network. Let θ be the collection of the weights of a neural network g(z; θ) that maps z ∈ Rk to Rd. Let

z ∼ N (0, Ik×k)                                         (11.18)
x|z ∼ N (g(z; θ), σ²Id×d)                                (11.19)

Here Ik×k denotes identity matrix of dimension k by k, and σ is a scalar that we assume to be known for simplicity.

For the Gaussian mixture models in Section 11.4, the optimal choice of Q(z) = p(z|x; θ) for each fixed θ, that is the posterior distribution of z, can be analytically computed. In many more complex models such as the model (11.19), it’s intractable to compute the exact the posterior distribution p(z|x; θ).

Recall that from equation (11.10), ELBO is always a lower bound for any choice of Q, and therefore, we can also aim for finding an approximation of the true posterior distribution. Often, one has to use some particular form to approximate the true posterior distribution. Let Q be a family of Q’s that we are considering, and we will aim to find a Q within the family of Q that is closest to the true posterior distribution. To formalize, recall the definition of the ELBO lower bound as a function of Q and θ defined in equation (11.14)

∑                     ∑ ∑
ELBO(Q, θ) =    n ELBO(x(i); Qi, θ) =    Qi(z(i)) log p(x(i), z(i); θ)
i=1                     i  z(i)           Qi(z(i))

Recall that EM can be viewed as alternating maximization of ELBO(Q, θ). Here instead, we optimize the ELBO over Q ∈ Q

max max ELBO(Q, θ)                                  (11.20)
Q∈Q      θ

Now the next question is what form of Q (or what structural assumptions to make about Q) allows us to efficiently maximize the objective above. When the latent variable z are high-dimensional discrete variables, one popular assumption is the mean field assumption, which assumes that Qi(z) gives a distribution with independent coordinates, or in other words, Qi can be decomposed into Qi(z) = Q1(z1) · · · Qk(zk). There are tremendous applications of mean field assumptions to learning generative models with discrete latent variables, and we refer to Blei et al. [2017] for a survey of these models and





their impact to a wide range of applications including computational biology, computational neuroscience, social sciences. We will not get into the details about the discrete latent variable cases, and our main focus is to deal with continuous latent variables, which requires not only mean field assumptions, but additional techniques.

When z ∈ Rk is a continuous latent variable, there are several decisions to make towards successfully optimizing (11.20). First we need to give a succinct representation of the distribution Qi because it is over an infinite number of points. A natural choice is to assume Qi is a Gaussian distribution with some mean and variance. We would also like to have more succinct representation of the means of Qi of all the examples. Note that Qi(z(i)) is supposed to approximate p(z(i)|x(i); θ). It would make sense let all the means of the Qi’s be some function of x(i). Concretely, let q(·; φ), v(·; φ) be two functions that map from dimension d to k, which are parameterized by φ and ψ, we assume that

Qi = N (q(x(i); φ), diag(v(x(i); ψ))2) (11.21)

Here diag(w) means the k × k matrix with the entries of w ∈ Rk on the diagonal. In other words, the distribution Qi is assumed to be a Gaussian distribution with independent coordinates, and the mean and standard deviations are governed by q and v. Often in variational auto-encoder, q and v are chosen to be neural networks.8 In recent deep learning literature, often q, v are called encoder (in the sense of encoding the data into latent code), whereas g(z; θ) if often referred to as the decoder.

We remark that Qi of such form in many cases are very far from a good approximation of the true posterior distribution. However, some approximation is necessary for feasible optimization. In fact, the form of Qi needs to satisfy other requirements (which happened to be satisfied by the form (11.21)).

Before optimizing the ELBO, let’s first verify whether we can efficiently evaluate the value of the ELBO for fixed Q of the form (11.21) and θ. We rewrite the ELBO as a function of φ, ψ, θ by

ELBO(φ, ψ, θ) = ∑i=1n Ez(i)∼Q log p(x(i), z(i); θ), (11.22)

where Qi = N (q(x(i); φ), diag(v(x(i); ψ))2)

Note that to evaluate Qi(z(i)) inside the expectation, we should be able to compute the density of Qi. To estimate the expectation Ez(i)∼Qi, we

8q and v can also share parameters. We sweep this level of details under the rug in this note.





# 163

should be able to sample from distribution *Qi so that we can build an empirical estimator with samples. It happens that for Gaussian distribution Qi = N (q(x(i); φ), diag(v(x(i); ψ))²)*, we are able to be both efficiently. Now let’s optimize the ELBO. It turns out that we can run gradient ascent over φ, ψ, θ instead of alternating maximization. There is no strong need to compute the maximum over each variable at a much greater cost. (For Gaussian mixture model in Section 11.4, computing the maximum is analytically feasible and relatively cheap, and therefore we did alternating maximization.)

Mathematically, let η be the learning rate, the gradient ascent step is

θ := θ + η∇θELBO(φ, ψ, θ)
φ := φ + η∇φELBO(φ, ψ, θ)
ψ := ψ + η∇ψELBO(φ, ψ, θ)

Computing the gradient over θ is simple because

∇θELBO(φ, ψ, θ) = ∇θ n ∑i=1 Ez(i)∼*Qi* [log p(x(i), z(i); θ]

= ∇θ n ∑i=1 Ez(i)∼*Qi* [log p(x(i), z(i); θ]

= n Ez(i)∼*Qi* [∇θ log p(x(i), z(i); θ)] ,      (11.23)

But computing the gradient over φ and ψ is tricky because the sampling distribution *Qi depends on φ and ψ. (Abstractly speaking, the issue we face can be simplified as the problem of computing the gradient Ez∼Qφ [f (φ)] with respect to variable φ. We know that in general, ∇Ez∼Qφ [f (φ)] = Ez∼Qφ [∇f (φ)] because the dependency of Qφ* on φ has to be taken into account as well.)

The idea that comes to rescue is the so-called re-parameterization trick: we rewrite z(i) ∼ *Qi* = N (q(x(i); φ), diag(v(x(i); ψ))²) in an equivalent way:

z(i) = q(x(i); φ) + v(x(i); ψ) ξ(i) where ξ(i) ∼ N (0, Iₖ×k)     (11.24)

Here x ⊙ y denotes the entry-wise product of two vectors of the same dimension. Here we used the fact that x ∼ N (μ, σ²) is equivalent to that x = μ + ξσ with ξ ∼ N (0, 1). We mostly just used this fact in every dimension simultaneously for the random variable z(i) ∼ *Qi*.





With this re-parameterization, we have that

Ez(i)∼Q [log p(x(i), z(i); θ)]                                   (11.25)

i        [Qⁱ(z(i))                                   ]

= Eξ(i)∼N(0,1)   log p(x(i), q(x(i); φ) + v(x(i); ψ)  ξ(i); θ)

Qi(q(x(i); φ) + v(x(i); ψ)           ξ(i))

It follows that

∇φEz(i)∼Q [log p(x(i), z(i); θ)]

i    [Qⁱ(z(i))                                  ]

= ∇φEξ(i)∼N(0,1)   log p(x(i), q(x(i); φ) + v(x(i); ψ)  ξ(i); θ)

[        Qi(q(x(i); φ) + v(x(i); ψ)  ξ(i))   ]

= Eξ(i)∼N(0,1)        ∇φ log p(x(i), q(x(i); φ) + v(x(i); ψ)  ξ(i); θ)

Qi(q(x(i); φ) + v(x(i); ψ)        ξ(i))

We can now sample multiple copies of ξ(i)’s to estimate the expectation in the RHS of the equation above.⁹

We can estimate the gradient with respect to ψ similarly, and with these, we can implement the gradient ascent algorithm to optimize the ELBO over φ, ψ, θ.

There are not many high-dimensional distributions with analytically computable density function are known to be re-parameterizable. We refer to Kingma and Welling [2013] for a few other choices that can replace Gaussian distribution.

9Empirically people sometimes just use one sample to estimate it for maximum computational efficiency.




# Chapter 12

# Principal components analysis

In this set of notes, we will develop a method, Principal Components Analysis (PCA), that tries to identify the subspace in which the data approximately lies. PCA is computationally efficient: it will require only an eigenvector calculation (easily done with the eig function in Matlab).

Suppose we are given a dataset {x(i); i = 1, . . . , n} of attributes of n different types of automobiles, such as their maximum speed, turn radius, and so on. Let x(i) ∈ Rᵈ for each i (d &#x3C; n). But unknown to us, two different attributes—some xi and xj—respectively give a car’s maximum speed measured in miles per hour, and the maximum speed measured in kilometers per hour. These two attributes are therefore almost linearly dependent, up to only small differences introduced by rounding off to the nearest mph or kph. Thus, the data really lies approximately on an n − 1 dimensional subspace. How can we automatically detect, and perhaps remove, this redundancy?

For a less contrived example, consider a dataset resulting from a survey of pilots for radio-controlled helicopters, where x(i) is a measure of the piloting skill of pilot i, and x₂(i) captures how much he/she enjoys flying. Because RC helicopters are very difficult to fly, only the most committed students, ones that truly enjoy flying, become good pilots. So, the two attributes x₁ and x₂ are strongly correlated. Indeed, we might posit that the data actually lies along some diagonal axis (the u₁ direction) capturing the intrinsic piloting “karma” of a person, with only a small amount of noise lying off this axis. (See figure.) How can we automatically compute this u₁ direction?



# 166

(enjoyment)

(skill)

We will shortly develop the PCA algorithm. But prior to running PCA per se, typically we first preprocess the data by normalizing each feature to have mean 0 and variance 1. We do this by subtracting the mean and dividing by the empirical standard deviation:

x(i) − μj
x(i)
←
j
σj

where μj = 1/n ∑i=1n x(i) and σ² = 1/n ∑i=1n (x(i) − μj)² are the mean variance of feature j, respectively.

Subtracting μj zeros out the mean and may be omitted for data known to have zero mean (for instance, time series corresponding to speech or other acoustic signals). Dividing by the standard deviation σj rescales each coordinate to have unit variance, which ensures that different attributes are all treated on the same “scale.” For instance, if x₁ was cars’ maximum speed in mph (taking values in the high tens or low hundreds) and x₂ were the number of seats (taking values around 2-4), then this renormalization rescales the different attributes to make them more comparable. This rescaling may be omitted if we had a priori knowledge that the different attributes are all on the same scale. One example of this is if each data point represented a grayscale image, and each x(i) took a value in {0, 1, . . . , 255} corresponding to the intensity value of pixel j in image i.

Now, having normalized our data, how do we compute the “major axis of variation” u—that is, the direction on which the data approximately lies? One way is to pose this problem as finding the unit vector u so that when




the data is projected onto the direction corresponding to u, the variance of the projected data is maximized. Intuitively, the data starts off with some amount of variance/information in it. We would like to choose a direction u so that if we were to approximate the data as lying in the direction/subspace corresponding to u, as much as possible of this variance is still retained. Consider the following dataset, on which we have already carried out the normalization steps:

Now, suppose we pick u to correspond the the direction shown in the figure below. The circles denote the projections of the original data onto this line.




We see that the projected data still has a fairly large variance, and the points tend to be far from zero. In contrast, suppose had instead picked the following direction:

Here, the projections have a significantly smaller variance, and are much closer to the origin.

We would like to automatically select the direction u corresponding to the first of the two figures shown above. To formalize this, note that given a





unit vector u and a point x, the length of the projection of x onto u is given by xᵀ u. I.e., if x(i) is a point in our dataset (one of the crosses in the plot), then its projection onto u (the corresponding circle in the figure) is distance xᵀ u from the origin. Hence, to maximize the variance of the projections, we would like to choose a unit-length u so as to maximize:

∑                      ∑

1 n      (x(i)ᵀ u)² = 1  n uᵀ x(i)x(i)ᵀ u

n i=1           n i=1                     )

( ∑

= uᵀ           1 n  x(i)x(i)ᵀ    u.

n i=1

We easily recognize that the maximizing this subject to ‖u‖₂ = 1 gives the principal eigenvector of Σ = 1 ∑n           x(i)x(i)ᵀ , which is just the empirical covariance matrix of the     n     i=1                          1 data (assuming it has zero mean).

To summarize, we have found that if we wish to find a 1-dimensional subspace with which to approximate the data, we should choose u to be the principal eigenvector of Σ. More generally, if we wish to project our data into a k-dimensional subspace (k &#x3C; d), we should choose u₁, . . . , uₖ to be the top k eigenvectors of Σ. The ui’s now form a new, orthogonal basis for the data.²

Then, to represent x(i) in this basis, we need only compute the corresponding vector:

uT x(i)

 1         

uᵀ x(i)

y(i) =  2 .      ∈ Rᵏ .

   .       

.

uᵀ x(i)

k

Thus, whereas x(i) ∈ Rᵈ, the vector y(i) now gives a lower, k-dimensional, approximation/representation for x(i). PCA is therefore also referred to as a dimensionality reduction algorithm. The vectors u₁, . . . , uₖ are called the first k principal components of the data.

Remark. Although we have shown it formally only for the case of k = 1, using well-known properties of eigenvectors it is straightforward to show that

1 If you haven’t seen this before, try using the method of Lagrange multipliers to maximize uT Σu subject to that uT u = 1. You should be able to show that Σu = λu, for some λ, which implies u is an eigenvector of Σ, with eigenvalue λ.

2 Because Σ is symmetric, the ui’s will (or always can be chosen to be) orthogonal to each other.




of all possible orthogonal bases u1, . . . , uk, the one that we have chosen maximizes ∑ ‖yi‖². Thus, our choice of a basis preserves as much variability possible in the original data.

PCA can also be derived by picking the basis that minimizes the approximation error arising from projecting the data onto the k-dimensional subspace spanned by them. (See more in homework.)

PCA has many applications; we will close our discussion with a few examples. First, compression—representing xi’s with lower dimension yi’s—is an obvious application. If we reduce high dimensional data to k = 2 or 3 dimensions, then we can also plot the yi’s to visualize the data. For instance, if we were to reduce our automobiles data to 2 dimensions, then we can plot it (one point in our plot would correspond to one car type, say) to see what cars are similar to each other and what groups of cars may cluster together.

Another standard application is to preprocess a dataset to reduce its dimension before running a supervised learning algorithm with the xi’s as inputs. Apart from computational benefits, reducing the data’s dimension can also reduce the complexity of the hypothesis class considered and help avoid overfitting (e.g., linear classifiers over lower dimensional input spaces will have smaller VC dimension).

Lastly, as in our RC pilot example, we can also view PCA as a noise reduction algorithm. In our example it, estimates the intrinsic “piloting karma” from the noisy measures of piloting skill and enjoyment. In class, we also saw the application of this idea to face images, resulting in eigenfaces method. Here, each point xi ∈ R100×100 was a 10000 dimensional vector, with each coordinate corresponding to a pixel intensity value in a 100x100 image of a face. Using PCA, we represent each image xi with a much lower-dimensional yi. In doing so, we hope that the principal components we found retain the interesting, systematic variations between faces that capture what a person really looks like, but not the “noise” in the images introduced by minor lighting variations, slightly different imaging conditions, and so on.

We then measure distances between faces i and j by working in the reduced dimension, and computing ‖yi − yj‖₂. This resulted in a surprisingly good face-matching and retrieval algorithm.



# Chapter 13

# Independent components analysis

Our next topic is Independent Components Analysis (ICA). Similar to PCA, this will find a new basis in which to represent our data. However, the goal is very different.

As a motivating example, consider the “cocktail party problem.” Here, d speakers are speaking simultaneously at a party, and any microphone placed in the room records only an overlapping combination of the d speakers’ voices. But lets say we have d different microphones placed in the room, and because each microphone is a different distance from each of the speakers, it records a different combination of the speakers’ voices. Using these microphone recordings, can we separate out the original d speakers’ speech signals?

To formalize this problem, we imagine that there is some data s ∈ Rd that is generated via d independent sources. What we observe is

x = As,

where A is an unknown square matrix called the mixing matrix. Repeated observations gives us a dataset {x(i); i = 1, . . . , n}, and our goal is to recover the sources s(i) that had generated our data (x(i) = As(i)).

In our cocktail party problem, s(i) is an d-dimensional vector, and sj(i) is the sound that speaker j was uttering at time i. Also, x(i) is an d-dimensional vector, and xj(i) is the acoustic reading recorded by microphone j at time i.

Let W = A−1 be the unmixing matrix. Our goal is to find W, so that given our microphone recordings x(i), we can recover the sources by computing s(i) = W x(i). For notational convenience, we also let wᵀ denote





the i-th row of W , so that

 — wT — 

    .¹     

W =   .           .

.

— wᵀ —

d

Thus, wi ∈ Rᵈ, and the j-th source can be recovered as s(i) = wᵀ x(i).

# 13.1  ICA ambiguities

To what degree can W = A−1 be recovered? If we have no prior knowledge about the sources and the mixing matrix, it is easy to see that there are some inherent ambiguities in A that are impossible to recover, given only the x(i)’s.

Specifically, let P be any d-by-d permutation matrix. This means that each row and each column of P has exactly one “1.” Here are some examples of permutation matrices:

 0  1  0        [ 0   1 ]       [ 1     0 ]

P =  1  0  0  ;  P =  1   0    ;    P =  0  1  .

0  0  1

If z is a vector, then P z   is another vector that contains a permuted version of z’s coordinates. Given only the x(i)’s, there will be no way to distinguish between W and P W . Specifically, the permutation of the original sources is ambiguous, which should be no surprise. Fortunately, this does not matter for most applications.

Further, there is no way to recover the correct scaling of the wi’s. For instance, if A were replaced with 2A, and every s(i) were replaced with (0.5)s(i), then our observed x(i) = 2A · (0.5)s(i) would still be the same. More broadly, if a single column of A were scaled by a factor of α, and the corresponding source were scaled by a factor of 1/α, then there is again no way to determine that this had happened given only the x(i)’s. Thus, we cannot recover the “correct” scaling of the sources. However, for the applications that we are concerned with—including the cocktail party problem—this ambiguity also does not matter. Specifically, scaling a speaker’s speech signal s(i) by some positive factor α affects only the volume of that speaker’s speech. Also, sign changes do not matter, and s(i) and −s(i) sound identical when played on a speaker. Thus, if the wᵀ found by an algorithm is scaled by any non-zero real number, the corresponding recovered source si = wᵀ x will be scaled by the.





173

same factor; but this usually does not matter. (These comments also apply to ICA for the brain/MEG data that we talked about in class.)

Are these the only sources of ambiguity in ICA? It turns out that they are, so long as the sources s are non-Gaussian. To see what the difficulty is with Gaussian data, consider an example in which n = 2, and s ∼ N (0, I). Here, I is the 2x2 identity matrix. Note that the contours of the density of the standard normal distribution N (0, I) are circles centered on the origin, and the density is rotationally symmetric.

Now, suppose we observe some x = As, where A is our mixing matrix. Then, the distribution of x will be Gaussian, x ∼ N (0, AAᵀ), since

Es∼N(0,I)[x] = E[As] = AE[s] = 0

Cov[x] = Es∼N(0,I)[xxᵀ] = E[Assᵀ Aᵀ] = AE[ssᵀ]Aᵀ = A · Cov[s] · Aᵀ = AAᵀ

Now, let R be an arbitrary orthogonal (less formally, a rotation/reflection) matrix, so that RRᵀ = Rᵀ R = I, and let A′ = AR. Then if the data had been mixed according to A′ instead of A, we would have instead observed x′ = A′s. The distribution of x′ is also Gaussian, x′ ∼ N (0, AAᵀ), since

Es∼N(0,I)[x′(x′)ᵀ] = E[A′ssᵀ (A′)ᵀ] = E[ARssᵀ (AR)ᵀ] = ARRᵀ Aᵀ = AAᵀ.

Hence, whether the mixing matrix is A or A′, we would observe data from a N (0, AAᵀ) distribution. Thus, there is no way to tell if the sources were mixed using A and A′. There is an arbitrary rotational component in the mixing matrix that cannot be determined from the data, and we cannot recover the original sources.

Our argument above was based on the fact that the multivariate standard normal distribution is rotationally symmetric. Despite the bleak picture that this paints for ICA on Gaussian data, it turns out that, so long as the data is not Gaussian, it is possible, given enough data, to recover the d independent sources.

# 13.2 Densities and linear transformations

Before moving on to derive the ICA algorithm proper, we first digress briefly to talk about the effect of linear transformations on densities.

Suppose a random variable s is drawn according to some density pₛ(s). For simplicity, assume for now that s ∈ R is a real number. Now, let the random variable x be defined according to x = As (here, x ∈ R, A ∈ R). Let pₓ be the density of x. What is pₓ?

Let W = A⁻¹. To calculate the “probability” of a particular value of x, it is tempting to compute s = W x, then then evaluate pₛ at that point, and





conclude that “pₓ(x) = pₛ(W x).” However, this is incorrect. For example, let s ∼ Uniform[0, 1], so pₛ(s) = 1{0 ≤ s ≤ 1}. Now, let A = 2, so x = 2s. Clearly, x is distributed uniformly in the interval [0, 2]. Thus, its density is given by pₓ(x) = (0.5)1{0 ≤ x ≤ 2}. This does not equal pₛ(W x), where W = 0.5 = A−1. Instead, the correct formula is pₓ(x) = pₛ(W x)·|W|.

More generally, if s is a vector-valued distribution with density pₛ, and x = As for a square, invertible matrix A, then the density of x is given by

pₓ(x) = pₛ(W x) · |W|,

where W = A−1.

Remark. If you’ve seen the result that A maps [0, 1]d to a set of volume |A|, then here’s another way to remember the formula for pₓ given above, that also generalizes our previous 1-dimensional example. Specifically, let A ∈ Rd×d be given, and let W = A−1 as usual. Also let C₁ = [0, 1]d be the d-dimensional hypercube, and define C₂ = {As : s ∈ C₁} ⊆ Rd to be the image of C₁ under the mapping given by A. Then it is a standard result in linear algebra (and, indeed, one of the ways of defining determinants) that the volume of C₂ is given by |A|. Now, suppose s is uniformly distributed in [0, 1]d, so its density is pₛ(s) = 1{s ∈ C₁}. Then clearly x will be uniformly distributed in C₂. Its density is therefore found to be pₓ(x) = 1{x ∈ C₂}/vol(C₂) (since it must integrate over C₂ to 1). But using the fact that the determinant of the inverse of a matrix is just the inverse of the determinant, we have 1/vol(C₂) = 1/|A| = |A−1| = |W|. Thus, pₓ(x) = 1{x ∈ C₂}|W| = 1{W x ∈ C₁}|W| = pₛ(W x)|W|.

# 13.3 ICA algorithm

We are now ready to derive an ICA algorithm. We describe an algorithm by Bell and Sejnowski, and we give an interpretation of their algorithm as a method for maximum likelihood estimation. (This is different from their original interpretation involving a complicated idea called the infomax principal which is no longer necessary given the modern understanding of ICA.) We suppose that the distribution of each source sj is given by a density pₛ, and that the joint distribution of the sources s is given by

∏j=1d p(s) = ∏j=1d pₛ(sj).





Note that by modeling the joint distribution as a product of marginals, we capture the assumption that the sources are independent. Using our formulas from the previous section, this implies the following density on *x = As = W−1s:*

∏j=1d p(x) = ps(wTx) · |W|.

All that remains is to specify a density for the individual sources ps. Recall that, given a real-valued random variable z, its cumulative distribution function (cdf) F is defined by F(z0) = P(z ≤ z0) = ∫−∞z0 pz(z)dz and the density is the derivative of the cdf: pz(z) = F′(z).

Thus, to specify a density for the si’s, all we need to do is to specify some cdf for it. A cdf has to be a monotonic function that increases from zero to one. Following our previous discussion, we cannot choose the Gaussian cdf, as ICA doesn’t work on Gaussian data. What we’ll choose instead as a reasonable “default” cdf that slowly increases from 0 to 1, is the sigmoid function g(s) = 1/(1 + e−s). Hence, ps(s) = g′(s).1

The square matrix W is the parameter in our model. Given a training set {x(i); i = 1, . . . , n}, the log likelihood is given by

∑i=1n (∑j=1d log g′(wTx(i)) + log |W|).

We would like to maximize this in terms W. By taking derivatives and using the fact (from the first set of notes) that ∇W|W| = |W|(W−1)T, we easily derive a stochastic gradient ascent learning rule. For a training example x(i), the update rule is:

W := W + α *(1 − 2g(wTx(i)))*
*(1 − 2g(wTx(i)))*
*1 − 2g(wTx(i))*
x(i)T + (WT)−1.

1If you have prior knowledge that the sources’ densities take a certain form, then it is a good idea to substitute that in here. But in the absence of such knowledge, the sigmoid function can be thought of as a reasonable default that seems to work well for many problems. Also, the presentation here assumes that either the data x(i) has been preprocessed to have zero mean, or that it can naturally be expected to have zero mean (such as acoustic signals). This is necessary because our assumption that ps(s) = g′(s) implies E[s] = 0 (the derivative of the logistic function is a symmetric function, and hence gives a density corresponding to a random variable with zero mean), which implies E[x] = E[As] = 0.




where α is the learning rate.

After the algorithm converges, we then compute s(i) = W x(i) to recover the original sources.

# Remark

When writing down the likelihood of the data, we implicitly assumed that the x(i)’s were independent of each other (for different values of i; note this issue is different from whether the different coordinates of x(i) are independent), so that the likelihood of the training set was given by ∏i p(x(i); W ). This assumption is clearly incorrect for speech data and other time series where the x(i)’s are dependent, but it can be shown that having correlated training examples will not hurt the performance of the algorithm if we have sufficient data. However, for problems where successive training examples are correlated, when implementing stochastic gradient ascent, it sometimes helps accelerate convergence if we visit training examples in a randomly permuted order. (I.e., run stochastic gradient ascent on a randomly shuffled copy of the training set.)



Chapter 14

# Self-supervised learning and foundation models

Despite its huge success, supervised learning with neural networks typically relies on the availability of a labeled dataset of decent size, which is sometimes costly to collect. Recently, AI and machine learning are undergoing a paradigm shift with the rise of models (e.g., BERT [Devlin et al., 2019] and GPT-3 [Brown et al., 2020]) that are pre-trained on broad data at scale and are adaptable to a wide range of downstream tasks. These models, called foundation models by Bommasani et al. [2021], oftentimes leverage massive unlabeled data so that much fewer labeled data in the downstream tasks are needed. Moreover, though foundation models are based on standard deep learning and transfer learning, their scale results in new emergent capabilities. These models are typically (pre-)trained by self-supervised learning methods where the supervisions/labels come from parts of the inputs. This chapter will introduce the paradigm of foundation models and basic related concepts.

# 14.1 Pretraining and adaptation

The foundation models paradigm consists of two phases: pretraining (or simply training) and adaptation. We first pretrain a large model on a massive unlabeled dataset (e.g., billions of unlabeled images).1 Then, we adapt the pretrained model to a downstream task (e.g., detecting cancer from scan images). These downstream tasks are often prediction tasks with limited or

1 Sometimes, pretraining can involve large-scale labeled datasets as well (e.g., the ImageNet dataset).

177




even no labeled data. The intuition is that the pretrained models learn good representations that capture intrinsic semantic structure/information about the data, and the adaptation phase customizes the model to a particular downstream task by, e.g., retrieving the information specific to it. For example, a model pretrained on massive unlabeled image data may learn good general visual representations/features, and we adapt the representations to solve biomedical imaging tasks.

# Pretraining

Suppose we have an unlabeled pretraining dataset {x(1), x(2), · · · , x(ⁿ)} that consists of n examples in Rᵈ. Let φθ be a model that is parameterized by θ and maps the input x to some m-dimensional representation φθ(x). (People also call φθ(x) ∈ Rᵐ the embedding or features of the example x.) We pretrain the model θ with a pretraining loss, which is often an average of loss functions on all the examples: Lₚᵣₑ(θ) = 1/n ∑i=1n `ₚᵣₑ(θ, x(i)). Here ` is a so-called self-supervised loss on a single datapoint x(i), because as shown later, e.g., in Section 14.3, the “supervision” comes from the data point x(i) itself. It is also possible that the pretraining loss is not a sum of losses on individual examples. We will discuss two pretraining losses in Section 14.2 and Section 14.3.

We use some optimizers (mostly likely SGD or ADAM [Kingma and Ba, 2014]) to minimize Lpre(θ). We denote the obtained pretrained model by ˆθ.

# Adaptation

For a downstream task, we usually have a labeled dataset {(x(1), y(1)), · · · , (x(ⁿₜₐₛₖ), y(ⁿₜₐₛₖ))} with nₜₐₛₖ examples. The setting when nₜₐₛₖ = 0 is called zero-shot learning—the downstream task doesn’t have any labeled examples. When nₜₐₛₖ is relatively small (say, between 1 and 50), the setting is called few-shot learning. It’s also pretty common to have a larger nₜₐₛₖ on the order of ranging from hundreds to tens of thousands.

An adaptation algorithm generally takes in a downstream dataset and the pretrained model ˆθ, and outputs a variant of θ that solves the downstream task. We will discuss below two popular and general adaptation methods, linear probe and finetuning. In addition, two other methods specific to language problems are introduced in 14.3.2.

The linear probe approach uses a linear head on top of the representation to predict the downstream labels. Mathematically, the adapted model outputs w>φ(x), where w ∈ Rᵐ is a parameter to be learned, and ˆθ is exactly the pretrained model (fixed). We can use SGD (or other optimizers) to train.





179

minimize

∑i=1ntask `task(ytask, wTφˆ(xtask)) (14.1)

w ∈ ℝm

E.g., if the downstream task is a regression problem, we will have

`task(ytask, wTφˆ(xtask)) = (ytask − wTφˆ(xtask))².

The finetuning algorithm uses a similar structure for the downstream prediction model, but also further finetunes the pretrained model (instead of keeping it fixed). Concretely, the prediction model is wTφθ(x) with parameters w and θ. We optimize both w and θ to fit the downstream data, but initialize θ with the pretrained model θˆ. The linear head w is usually initialized randomly.

minimize

∑i=1ntask `task(ytask, wTφθ(xtask)) (14.2)

w, θ

with initialization

w ← random vector (14.3)

θ ← θˆ (14.4)

Various other adaptation methods exist and are sometimes specialized to the particular pretraining methods. We will discuss one of them in Section 14.3.2.

# 14.2 Pretraining methods in computer vision

This section introduces two concrete pretraining methods for computer vision: supervised pretraining and contrastive learning.

# Supervised pretraining.

Here, the pretraining dataset is a large-scale labeled dataset (e.g., ImageNet), and the pretrained models are simply a neural network trained with vanilla supervised learning (with the last layer being removed). Concretely, suppose we write the learned neural network as U φ (x), where U is the last (fully-connected) layer parameters, θˆ corresponds to the parameters of all the other layers, and φˆ(x) are the penultimate activations layer (which serves as the representation). We simply discard U and use φˆ(x) as the pretrained model.

# Contrastive learning.

Contrastive learning is a self-supervised pretraining method that uses only unlabeled data. The main intuition is that a good representation function φθ(·) should map semantically similar images to similar representations, and that random pairs of images should generally have





distinct representations. E.g., we may want to map images of two huskies to similar representations, but a husky and an elephant should have different representations. One definition of similarity is that images from the same class are similar. Using this definition will result in the so-called supervised contrastive algorithms that work well when labeled pretraining datasets are available.

Without labeled data, we can use data augmentation to generate a pair of “similar” augmented images given an original image x. Data augmentation typically means that we apply random cropping, flipping, and/or color transformation on the original image x to generate a variant. We can take two random augmentations, denoted by *xˆ and x˜, of the same original image x, and call them a positive pair. We observe that positive pairs of images are often semantically related because they are augmentations of the same image. We will design a loss function for θ such that the representations of a positive pair, φ(xˆ), φ(x*˜), as close to each other as possible.

On the other hand, we can also take another random image z from the pretraining dataset and generate an augmentation *zˆ from z. Note that (xˆ, zˆ) are from different images; therefore, with a good chance, they are not semantically related. We call (xˆ, zˆ) a negative or random pair. We will design a loss to push the representation of random pairs, φ(xˆ), φ(z*ˆ), far away from each other.

There are many recent algorithms based on the contrastive learning principle, and here we introduce SIMCLR [Chen et al., 2020] as a concrete example. The loss function is defined on a batch of examples (*x¹, · · · , x(B)) with batch size B. The algorithm computes two random augmentations for each example x(i) in the batch, denoted by x(i)ˆ and x(i)˜. As a result, we have the augmented batch of 2B examples: x(¹)ˆ, · · · , x(B)ˆ, x(¹)˜, · · · , x*(B)˜. The SIMCLR loss is defined as:

*Lpᵣₑ(θ) = − ∑i=1B log (exp(φ(x(i)ˆ) φ(x(i)˜)) / ∑j=iB exp(φ(x(i)ˆ) φ(x(j)))*

The intuition is as follows. The loss is increasing in φ(*x(i)ˆ) φ(x(j)˜), and thus minimizing the loss encourages φ(x(i)) φ(x(j)˜) to be small, making φ(x(i)) far away from φ(x(j)˜). On the other hand, the loss is decreasing in φ(x(i)ˆ) φ(z*(i)ˆ).

2Random pair may be a more accurate term because it’s still possible (though not likely) that x and z are semantically related, so are *xˆ and z*ˆ. But in the literature, the term negative pair seems to be also common.

3This is a variant and simplification of the original loss that does not change the essence (but may change the efficiency slightly).





181

φ (ˆ(i)  >     (i)                                             (i) >  (i) x  ) φ (˜ x      ), and thus minimizing the loss encourages φ (ˆ x     ) φ (˜ θ           θ                                                  θ       θ x                      ) to be large, resulting in φ (ˆ(i)       (i)                    4                              x      ) and φ (˜ θ      θ  x      ) to be close.

# 14.3 Pretrained large language models

Natural language processing is another area where pretraining models are particularly successful. In language problems, an example typically corresponds to a document or generally a sequence/trunk of words,⁵ denoted by x  = (x₁, · · ·  , xT ) where T is the length of the document/sequence, xi ∈ {1, · · ·   , V } are words in the document, and V is the vocabulary size.⁶ A language model is a probabilistic model representing the probability of a document, denoted by p(x₁, · · · , xT ). This probability distribution is very complex because its support size is V T — exponential in the length of the document. Instead of modeling the distribution of a document itself, we can apply the chain rule of conditional probability to decompose it as follows:

p(x₁, · · · , xT ) = p(x₁)p(x₂|x₁) · · · p(xT |x₁, · · · , xT −1).        (14.5)

Now the support size of each of the conditional probability p(xₜ|x₁, · · · , xt−1) is V. We will model the conditional probability p(xₜ|x₁, · · · , xt−1) as a function of x₁, . . . , xt−1 parameterized by some parameter θ. A parameterized model takes in numerical inputs and therefore we first introduce embeddings or representations of the words. Let ei ∈ Rᵈ be the embedding of the word i ∈ {1, 2, · · · , V }. We call [e₁, · · · , eV ] ∈ Rᵈ×V the embedding matrix.

The most commonly used model is Transformer [Vaswani et al., 2017]. In this subsection, we will introduce the input-output interface of a Transformer, but treat the intermediate computation in the Transformer as a blackbox. We refer the students to the transformer paper or more advanced courses for more details. As shown in Figure 14.1, given a document (x₁, · · · , xT ), we first translate the sequence of discrete variables into a sequence of corresponding.

4To see this, you can verify that the function − log p p  is decreasing in p, and increasing in q when p, q > 0. +q

5In the practical implementations, typically all the data are concatenated into a single sequence in some order, and each example typically corresponds a sub-sequence of consecutive words which may corresponds to a subset of a document or may span across multiple documents.

6Technically, words may be decomposed into tokens which could be words or sub-words (combinations of letters), but this note omits this technicality. In fact most commons words are a single token themselves.





# 182

word embeddings (eₓ₁, · · · , eₓT ). We also introduce a fixed special token x₀ = ⊥ in the vocabulary with corresponding embedding eₓ₀ to mark the beginning of a document. Then, the word embeddings are passed into a Transformer model, which takes in a sequence of vectors (eₓ₀, eₓ₁, · · · , eₓT ) and outputs a sequence of vectors (u₁, u₂, · · · , uT +1), where uₜ ∈ Rⱽ will be interpreted as the logits for the probability distribution of the next word. Here we use the autoregressive version of the Transformers which by design ensures uₜ only depends on x₁, · · · , xt−1 (note that this property does not hold in masked language models [Devlin et al., 2019] where the losses are also different.) We view the whole mapping from x’s to u’s a blackbox in this subsection and call it a Transformer, denoted it by fθ, where θ include both the parameters in the Transformer and the input embeddings. We write uₜ = fθ(x₀, x₁, . . . , xt−1) where fθ denotes the mapping from the input to the outputs.

| 𝑢!         | 𝑢"     | 𝑢'  | … | 𝑢#(! |
| ----------- | ------- | ---- | - | ----- |
| Transformer | 𝑓%(𝑥) |      |   |       |
| 𝑒$        | 𝑒$!    | 𝑒$" | … | 𝑒$#  |
| 𝑥&         | 𝑥!     | 𝑥"  |   | 𝑥#   |

Figure 14.1: The inputs and outputs of a Transformer model.

The conditional probability p(xₜ|x₁, · · · , xt−1) is the softmax of the logits:

 p(xₜ = 1|x₁ · · · , xt−1) 

 p(xₜ = 2|x₁ · · · , xt−1) 

          .                    

          .                    

p(xₜ = V |x₁ · · · , xt−1) = softmax(fθ(x₀, . . . , xt−1))

We train the Transformer parameter θ by minimizing the negative log-likelihood of seeing the data under the probabilistic model defined by θ,





which is the cross-entropy loss on the logitis.

∑

loss(θ) = 1      T       − log(pθ(xₜ|x₁, . . . , xt−1))              (14.8)

T t=1

∑

= 1             T       `ce(fθ(x₀, x₁, · · · , xt−1), xₜ)

T t=1

∑

= 1             T       − log(softmax(fθ(x₀, x₁, · · · , xt−1))ₓ ) .

T t=1                                                    t

# 14.3 Autoregressive text decoding / generation

Given a autoregressive Transformer, we can simply sample text from it sequentially. Given a pre-fix x₁, . . . xₜ, we generate text completion xₜ₊₁, . . . xT sequentially using the conditional distribution.

xₜ₊₁ ∼ softmax(fθ(x₀, x₁, · · · , xₜ))                             (14.9)

xₜ₊₂ ∼ softmax(fθ(x₀, x₁, · · · , xₜ₊₁))                           (14.10)

. . .                                                 (14.11)

xT ∼ softmax(fθ(x₀, x₁, · · · , xT −1)) .                          (14.12)

Note that each generated token is used as the input to the model when generating the following tokens. In practice, people often introduce a parameter τ > 0 named temperature to further adjust the entropy/sharpness of the generated distribution,

xₜ₊₁ ∼ softmax(fθ(x₀, x₁, · · · , xₜ)/τ )                            (14.13)

xₜ₊₂ ∼ softmax(fθ(x₀, x₁, · · · , xₜ₊₁)/τ )                          (14.14)

. . .                                                  (14.15)

xT ∼ softmax(fθ(x₀, x₁, · · · , xT −1)/τ ) .                       (14.16)

When τ = 1, the text is sampled from the original conditional probability defined by the model. With a decreasing τ, the generated text gradually becomes more “deterministic”. τ → 0 reduces to greedy decoding, where we generate the most probable next token from the conditional probability.

# 14.3.1 Zero-shot learning and in-context learning

For language models, there are many ways to adapt a pretrained model to downstream tasks. In this notes, we discuss three of them: finetuning, zero-shot learning, and in-context learning.





# 14.4 Finetuning and Zero-Shot Learning

Finetuning is not very common for the autoregressive language models that we introduced in Section 14.3 but much more common for other variants such as masked language models which has similar input-output interfaces but are pretrained differently [Devlin et al., 2019]. The finetuning method is the same as introduced generally in Section 14.1—the only question is how we define the prediction task with an additional linear head. One option is to treat *cT+1 = φθ(x1, · · · , xT) as the representation and use wTcT+1 = wTφθ(x1, · · · , xT) to predict task label. As described in Section 14.1, we initialize θ to the pretrained model ˆθ and then optimize both w and θ*.

Zero-shot adaptation or zero-shot learning is the setting where there is no input-output pairs from the downstream tasks. For language problems tasks, typically the task is formatted as a question or a cloze test form via natural language. For example, we can format an example as a question:

*xtask = (xtask,1, · · · , xtask,T) = “Is the speed of light a universal constant?”*

Then, we compute the most likely next word predicted by the language model given this question, that is, computing *argmaxxT+1 p(xT+1 | xtask,1, · · · , xtask,T). In this case, if the most likely next word xT+1* is “No”, then we solve the task. (The speed of light is only a constant in vacuum). We note that there are many ways to decode the answer from the language models, e.g., instead of computing the argmax, we may use the language model to generate a few words. It is an active research question to find the best way to utilize the language models.

In-context learning is mostly used for few-shot settings where we have a few labeled examples *(x(1), y(1)), · · · , (x(ntask), y(ntask)). Given a test example x, we construct a document (x1, · · · , xT)*, which is more commonly called a “prompt” in this context, by concatenating the labeled examples and the text example in some format. For example, we may construct the prompt as follows:

x1, · · · , xT = “Q: 2 ∼ 3 = ?          x(1)
A: 5                   y(1)
Q: 6 ∼ 7 = ?           x(2)
A: 13                  y(2)
· · ·
Q: 15 ∼ 2 = ?”         xtest




Then, we let the pretrained model generate the most likely xT + 1, xT + 2, · · · .

In this case, if the model can “learn” that the symbol ∼ means addition from the few examples, we will obtain the following which suggests the answer is 17.

xT + 1, xT + 2, · · · = “A: 17”.

The area of foundation models is very new and quickly growing. The notes here only attempt to introduce these models on a conceptual level with a significant amount of simplification. We refer the readers to other materials, e.g., Bommasani et al. [2021], for more details.


# Part V

# Reinforcement Learning and Control

186


# Chapter 15

# Reinforcement learning

We now begin our study of reinforcement learning and adaptive control. In supervised learning, we saw algorithms that tried to make their outputs mimic the labels y given in the training set. In that setting, the labels gave an unambiguous “right answer” for each of the inputs x. In contrast, for many sequential decision making and control problems, it is very difficult to provide this type of explicit supervision to a learning algorithm. For example, if we have just built a four-legged robot and are trying to program it to walk, then initially we have no idea what the “correct” actions to take are to make it walk, and so do not know how to provide explicit supervision for a learning algorithm to try to mimic.

In the reinforcement learning framework, we will instead provide our algorithms only a reward function, which indicates to the learning agent when it is doing well, and when it is doing poorly. In the four-legged walking example, the reward function might give the robot positive rewards for moving forwards, and negative rewards for either moving backwards or falling over. It will then be the learning algorithm’s job to figure out how to choose actions over time so as to obtain large rewards.

Reinforcement learning has been successful in applications as diverse as autonomous helicopter flight, robot legged locomotion, cell-phone network routing, marketing strategy selection, factory control, and efficient web-page indexing. Our study of reinforcement learning will begin with a definition of the Markov decision processes (MDP), which provides the formalism in which RL problems are usually posed.



# 15.1 Markov decision processes

A Markov decision process is a tuple (S, A, {Pₛₐ}, γ, R), where:

- S is a set of states. (For example, in autonomous helicopter flight, S might be the set of all possible positions and orientations of the helicopter.)
- A is a set of actions. (For example, the set of all possible directions in which you can push the helicopter’s control sticks.)
- Pₛₐ are the state transition probabilities. For each state s ∈ S and action a ∈ A, Pₛₐ is a distribution over the state space. We’ll say more about this later, but briefly, Pₛₐ gives the distribution over what states we will transition to if we take action a in state s.
- γ ∈ [0, 1) is called the discount factor.
- R: S × A → R is the reward function. (Rewards are sometimes also written as a function of a state S only, in which case we would have R: S → R).

The dynamics of an MDP proceeds as follows: We start in some state s₀, and get to choose some action a₀ ∈ A to take in the MDP. As a result of our choice, the state of the MDP randomly transitions to some successor state s₁, drawn according to s₁ ∼ Pₛ₀ₐ₀. Then, we get to pick another action a₁. As a result of this action, the state transitions again, now to some s₂ ∼ Pₛ₁ₐ₁. We then pick a₂, and so on. . . . Pictorially, we can represent this process as follows:

s₀


Our goal in reinforcement learning is to choose actions over time so as to maximize the expected value of the total payoff:

E [R(s₀) + γR(s₁) + γ²R(s₂) + · · · ]

Note that the reward at timestep t is discounted by a factor of γᵗ. Thus, to make this expectation large, we would like to accrue positive rewards as soon as possible (and postpone negative rewards as long as possible). In economic applications where R(·) is the amount of money made, γ also has a natural interpretation in terms of the interest rate (where a dollar today is worth more than a dollar tomorrow).

A policy is any function π : S → A mapping from the states to the actions. We say that we are executing some policy π if, whenever we are in state s, we take action a = π(s). We also define the value function for a policy π according to:

V π(s) = E [R(s₀) + γR(s₁) + γ²R(s₂) + · · · | s₀ = s, π].

V π(s) is simply the expected sum of discounted rewards upon starting in state s, and taking actions according to π.1

Given a fixed policy π, its value function V π satisfies the Bellman equations:

V π(s) = R(s) + γ ∑s′∈S Psπ(s)(s′)V π(s′).

This says that the expected sum of discounted rewards V π(s) for starting in s consists of two terms: First, the immediate reward R(s) that we get right away simply for starting in state s, and second, the expected sum of future discounted rewards. Examining the second term in more detail, we see that the summation term above can be rewritten Es′∼Psπ(s) [V π′(s′)]. This is the expected sum of discounted rewards for starting in state s, where s′ is distributed according Psπ(s), which is the distribution over where we will end up after taking the first action π(s) in the MDP from state s. Thus, the second term above gives the expected sum of discounted rewards obtained after the first step in the MDP.

Bellman’s equations can be used to efficiently solve for V π. Specifically, in a finite-state MDP (|S| &#x3C; ∞), we can write down one such equation for V π(s) for every state s. This gives us a set of |S| linear equations in |S| variables (the unknown V π(s)’s, one for each state), which can be efficiently solved for the V π(s)’s.

1This notation in which we condition on π isn’t technically correct because π isn’t a random variable, but this is quite standard in the literature.



We also define the optimal value function according to

V ∗(s) = max V π(s). (15.1)

In other words, this is the best possible expected sum of discounted rewards that can be attained using any policy. There is also a version of Bellman’s equations for the optimal value function:

V ∗(s) = R(s) + max γ ∑ Pₛₐ(s′)V ∗(s′). (15.2)

The first term above is the immediate reward as before. The second term is the maximum over all actions a of the expected future sum of discounted rewards we’ll get upon after action a. You should make sure you understand this equation and see why it makes sense.

We also define a policy π∗ : S → A as follows:

π∗(s) = arg max ∑ Pₛₐ(s′)V ∗(s′). (15.3)

Note that π∗(s) gives the action a that attains the maximum in the “max” in Equation (15.2).

It is a fact that for every state s and every policy π, we have

V ∗(s) = V π∗(s) ≥ V π(s).

The first equality says that the V π∗, the value function for π∗, is equal to the optimal value function V ∗ for every state s. Further, the inequality above says that π∗’s value is at least as large as the value of any other policy. In other words, π∗ as defined in Equation (15.3) is the optimal policy.

Note that π∗ has the interesting property that it is the optimal policy for all states s. Specifically, it is not the case that if we were starting in some state s then there’d be some optimal policy for that state, and if we were starting in some other state s′ then there’d be some other policy that’s optimal policy for s′. The same policy π∗ attains the maximum in Equation (15.1) for all states s. This means that we can use the same policy π∗ no matter what the initial state of our MDP is.

# 15.2 Value iteration and policy iteration

We now describe two efficient algorithms for solving finite-state MDPs. For now, we will consider only MDPs with finite state and action spaces (|S| &#x3C;





# 191

∞, |A| &#x3C; ∞). In this section, we will also assume that we know the state transition probabilities {Pₛₐ} and the reward function R.

The first algorithm, value iteration, is as follows:

# Algorithm 4 Value Iteration

1. For each state s, initialize V (s) := 0.
2. for until convergence do
3. For every state, update

V (s) := R(s) + max γ ∑ Pₛₐ(s′)V (s′). (15.4)

This algorithm can be thought of as repeatedly trying to update the estimated value function using Bellman Equations (15.2).

There are two possible ways of performing the updates in the inner loop of the algorithm. In the first, we can first compute the new values for V (s) for every state s, and then overwrite all the old values with the new values. This is called a synchronous update. In this case, the algorithm can be viewed as implementing a “Bellman backup operator” that takes a current estimate of the value function, and maps it to a new estimate. (See homework problem for details.) Alternatively, we can also perform asynchronous updates. Here, we would loop over the states (in some order), updating the values one at a time.

Under either synchronous or asynchronous updates, it can be shown that value iteration will cause V to converge to V ∗. Having found V ∗, we can then use Equation (15.3) to find the optimal policy.

Apart from value iteration, there is a second standard algorithm for finding an optimal policy for an MDP. The policy iteration algorithm proceeds as follows:

Thus, the inner-loop repeatedly computes the value function for the current policy, and then updates the policy using the current value function. (The policy π found in step (b) is also called the policy that is greedy with respect to V .) Note that step (a) can be done via solving Bellman’s equations as described earlier, which in the case of a fixed policy, is just a set of |S| linear equations in |S| variables.

After at most a finite number of iterations of this algorithm, V will converge to V ∗, and π will converge to π∗.²

²Note that value iteration cannot reach the exact V ∗ in a finite number of iterations,





# 15.3 Learning a model for an MDP

So far, we have discussed MDPs and algorithms for MDPs assuming that the state transition probabilities and rewards are known. In many realistic problems, we are not given state transition probabilities and rewards explicitly, but must instead estimate them from data. (Usually, S, A and γ are known.)

For example, suppose that, for the inverted pendulum problem (see prob-

whereas policy iteration with an exact linear system solver, can. This is because when the actions space and policy space are discrete and finite, and once the policy reaches the optimal policy in policy iteration, then it will not change at all. On the other hand, even though value iteration will converge to the V ∗, but there is always some non-zero error in the learned value function.

# Algorithm 5 Policy Iteration

1. Initialize π randomly.
2. for until convergence do
3. Let V := V π. . typically by linear system solver
4. For each state s, let

Both value iteration and policy iteration are standard algorithms for solving MDPs, and there isn’t currently universal agreement over which algorithm is better. For small MDPs, policy iteration is often very fast and converges with very few iterations. However, for MDPs with large state spaces, solving for V π explicitly would involve solving a large system of linear equations, and could be difficult (and note that one has to solve the linear system multiple times in policy iteration). In these problems, value iteration may be preferred. For this reason, in practice value iteration seems to be used more often than policy iteration. For some more discussions on the comparison and connection of value iteration and policy iteration, please see Section 15.5.





lem set 4), we had a number of trials in the MDP, that proceeded as follows:

| s(1) | a(1) | (1)  | a(1)     | (1) | a(1) | (1) | a(1) |
| ---- | ---- | ---- | -------- | --- | ---- | --- | ---- |
| −→ s | −→ s | −→ s | −→ . . . |     |      |     |      |
| 0    | 1    | 2    | 3        |     |      |     |      |
| s(2) | a(2) | (2)  | a(2)     | (2) | a(2) | (2) | a(2) |
| −→ s | −→ s | −→ s | −→ . . . |     |      |     |      |
| 0    | 1    | 2    | 3        |     |      |     |      |

Here, s(ʲ) is the state we were at time i of trial j, and a(ʲ) is the corresponding action that was taken from that state.

In practice, each of the trials above might be run until the MDP terminates (such as if the pole falls over in the inverted pendulum problem), or it might be run for some large but finite number of timesteps.

Given this “experience” in the MDP consisting of a number of trials, we can then easily derive the maximum likelihood estimates for the state transition probabilities:

Pₛₐ(s′) = #times took we action a in state s and got to s′

#times we took action a in state s

Or, if the ratio above is “0/0”—corresponding to the case of never having taken action a in state s before—the we might simply estimate Pₛₐ(s′) to be 1/|S|. (I.e., estimate Pₛₐ to be the uniform distribution over all states.)

Note that, if we gain more experience (observe more trials) in the MDP, there is an efficient way to update our estimated state transition probabilities using the new experience. Specifically, if we keep around the counts for both the numerator and denominator terms of (15.5), then as we observe more trials, we can simply keep accumulating those counts. Computing the ratio of these counts then gives our estimate of Pₛₐ.

Using a similar procedure, if R is unknown, we can also pick our estimate of the expected immediate reward R(s) in state s to be the average reward observed in state s.

Having learned a model for the MDP, we can then use either value iteration or policy iteration to solve the MDP using the estimated transition probabilities and rewards. For example, putting together model learning and value iteration, here is one possible algorithm for learning in an MDP with unknown state transition probabilities:

1. Initialize π randomly.
2. Repeat {
3. (a) Execute π in the MDP for some number of trials.





(b) Using the accumulated experience in the MDP, update our estimates for Pₛₐ (and R, if applicable).

(c) Apply value iteration with the estimated state transition probabilities and rewards to get a new estimated value function V.

(d) Update π to be the greedy policy with respect to V.

We note that, for this particular algorithm, there is one simple optimization that can make it run much more quickly. Specifically, in the inner loop of the algorithm where we apply value iteration, if instead of initializing value iteration with V = 0, we initialize it with the solution found during the previous iteration of our algorithm, then that will provide value iteration with a much better initial starting point and make it converge more quickly.

# 15.4 Continuous state MDPs

So far, we’ve focused our attention on MDPs with a finite number of states. We now discuss algorithms for MDPs that may have an infinite number of states. For example, for a car, we might represent the state as (x, y, θ, x, ˙ y, ˙ θ), comprising its position (x, y); orientation θ; velocity in the x and y directions x ˙ and y ˙; and angular velocity θ. Hence, S = R6 is an infinite set of states, because there is an infinite number of possible positions and orientations for the car.3 Similarly, the inverted pendulum you saw in PS4 has states (x, θ, x, ˙ θ), where θ is the angle of the pole. And, a helicopter flying in 3d space has states of the form (x, y, z, φ, θ, ψ, x, ˙ y, ˙ z, ˙ φ, θ, ψ), where here the roll φ, pitch θ, and yaw ψ angles specify the 3d orientation of the helicopter. In this section, we will consider settings where the state space is S = Rd, and describe ways for solving such MDPs.

# 15.4.1 Discretization

Perhaps the simplest way to solve a continuous-state MDP is to discretize the state space, and then to use an algorithm like value iteration or policy iteration, as described previously. For example, if we have 2d states (s₁, s₂), we can use a grid to discretize the state space:

Technically, θ is an orientation and so the range of θ is better written θ ∈ [−π, π) than θ ∈ R; but for our purposes, this distinction is not important.





# 195

# S2

Here, each grid cell represents a separate discrete state s¯. We can then approximate the continuous-state MDP via a discrete-state one (S, A, {Psa}, γ, R), where S is the set of discrete states, {Psa} are our state transition probabilities over the discrete states, and so on. We can then use value iteration or policy iteration to solve for the V*(s¯) and π(s¯) in the discrete state MDP (S, A, {Psa}, γ, R). When our actual system is in some continuous-valued state s ∈ S and we need to pick an action to execute, we compute the corresponding discretized state s¯, and execute action π(s¯).

This discretization approach can work well for many problems. However, there are two downsides. First, it uses a fairly naive representation for V* (and π*). Specifically, it assumes that the value function takes a constant value over each of the discretization intervals (i.e., that the value function is piecewise constant in each of the grid cells).

To better understand the limitations of such a representation, consider a supervised learning problem of fitting a function to this dataset:

|   | 1.5 | 2 | 3   | 4 | 5   | 6 | 7   | 8 |
| - | --- | - | --- | - | --- | - | --- | - |
| y | 5.5 | 5 | 4.5 | 4 | 3.5 | 3 | 2.5 | 2 |




Clearly, linear regression would do fine on this problem. However, if we instead discretize the x-axis, and then use a representation that is piecewise constant in each of the discretization intervals, then our fit to the data would look like this:

|   | 1.5 | 2 | 3   | 4 | 5   | 6 | 7   | 8 |
| - | --- | - | --- | - | --- | - | --- | - |
| y | 5.5 | 5 | 4.5 | 4 | 3.5 | 3 | 2.5 | 2 |

This piecewise constant representation just isn’t a good representation for many smooth functions. It results in little smoothing over the inputs, and no generalization over the different grid cells. Using this sort of representation, we would also need a very fine discretization (very small grid cells) to get a good approximation.

A second downside of this representation is called the curse of dimensionality. Suppose S = Rd, and we discretize each of the d dimensions of the state into k values. Then the total number of discrete states we have is kd. This grows exponentially quickly in the dimension of the state space d, and thus does not scale well to large problems. For example, with a 10d state, if we discretize each state variable into 100 values, we would have 10010 = 1020 discrete states, which is far too many to represent even on a modern desktop computer.

As a rule of thumb, discretization usually works extremely well for 1d and 2d problems (and has the advantage of being simple and quick to implement). Perhaps with a little bit of cleverness and some care in choosing the discretization method, it often works well for problems with up to 4d states. If you’re extremely clever, and somewhat lucky, you may even get it to work for some 6d problems. But it very rarely works for problems any higher dimensional than that.



# 15.4.2 Value function approximation

We now describe an alternative method for finding policies in continuous-state MDPs, in which we approximate V ∗ directly, without resorting to discretization. This approach, called value function approximation, has been successfully applied to many RL problems.

# Using a model or simulator

To develop a value function approximation algorithm, we will assume that we have a model, or simulator, for the MDP. Informally, a simulator is a black-box that takes as input any (continuous-valued) state sₜ and action aₜ, and outputs a next-state sₜ₊₁ sampled according to the state transition probabilities Pₛₜₐₜ:

St  Simulator        St+1        Pₛₐ

↑

[t]          at

There are several ways that one can get such a model. One is to use physics simulation. For example, the simulator for the inverted pendulum in PS4 was obtained by using the laws of physics to calculate what position and orientation the cart/pole will be in at time t + 1, given the current state at time t and the action a taken, assuming that we know all the parameters of the system such as the length of the pole, the mass of the pole, and so on. Alternatively, one can also use an off-the-shelf physics simulation software package which takes as input a complete physical description of a mechanical system, the current state sₜ and action aₜ, and computes the state sₜ₊₁ of the system a small fraction of a second into the future.4

An alternative way to get a model is to learn one from data collected in the MDP. For example, suppose we execute n trials in which we repeatedly take actions in an MDP, each trial for T timesteps. This can be done picking actions at random, executing some specific policy, or via some other way of.

4Open Dynamics Engine (http://www.ode.com) is one example of a free/open-source physics simulator that can be used to simulate systems like the inverted pendulum, and that has been a reasonably popular choice among RL researchers.





# 15. Learning Models of Dynamics

Choosing actions. We would then observe n state sequences like the following:

(1) a(1)       (1) a(1)      (1) a(1)     a(1)    (1)
s      −→ s           −→ s          −→ · · · T −1
0       0      1   1         2        2   −→ sT
(2) a(2)       (2) a(2)      (2) a(2)     a(2)    (2)
s      −→ s           −→ s          −→ · · · T −1
0       0      1   1         2        2   −→ sT
· · ·
(n)    a(n)     (n)   a(n)      (n) a(n)     a(n)  (n)
s      −→ s           −→ s          −→ · · · T −1
0          0    1     1         2        2   −→ sT

We can then apply a learning algorithm to predict st+1 as a function of st and at.

For example, one may choose to learn a linear model of the form

st+1 = Ast + Bat,             (15.6)

using an algorithm similar to linear regression. Here, the parameters of the model are the matrices A and B, and we can estimate them using the data collected from our n trials, by picking

∑ ∑ ∥                    (              )∥
arg min                n T −1 ∥s(i)    −   As(i) + Ba(i)  ∥² .
A,B     i=1    t=0 ∥    t+1              t      t  ∥₂

We could also potentially use other loss functions for learning the model. For example, it has been found in recent work Luo et al. [2018] that using ‖ · ‖2 norm (without the square) may be helpful in certain cases.

Having learned A and B, one option is to build a deterministic model, in which given an input st and at, the output st+1 is exactly determined. Specifically, we always compute st+1 according to Equation (15.6). Alternatively, we may also build a stochastic model, in which st+1 is a random function of the inputs, by modeling it as

st+1 = Ast + Bat +             t,

where here t is a noise term, usually modeled as t ∼ N (0, Σ). (The covariance matrix Σ can also be estimated from data in a straightforward way.)

Here, we’ve written the next-state st+1 as a linear function of the current state and action; but of course, non-linear functions are also possible. Specifically, one can learn a model st+1 = Aφs(st) + Bφa(at), where φs and φa are some non-linear feature mappings of the states and actions. Alternatively, one can also use non-linear learning algorithms, such as locally weighted linear regression, to learn to estimate st+1 as a function of st and at. These approaches can also be used to build either deterministic or stochastic simulators of an MDP.





# 199

# Fitted value iteration

We now describe the fitted value iteration algorithm for approximating the value function of a continuous state MDP. In the sequel, we will assume that the problem has a continuous state space S = Rd, but that the action space A is small and discrete.5

Recall that in value iteration, we would like to perform the update

V (s) :=         R(s) + γ max             ∫ Ps,a(s′)V (s′)ds′
(15.7)

=         R(s) + γ max Es′∼Ps,a[V (s′)]
(15.8)

(In Section 15.2, we had written the value iteration update with a summation V (s) := R(s) + γ maxa ∑s′ Ps,a(s′)V (s′) rather than an integral over states; the new notation reflects that we are now working in continuous states rather than discrete states.)

The main idea of fitted value iteration is that we are going to approximately carry out this step, over a finite sample of states s(1), . . . , s(n). Specifically, we will use a supervised learning algorithm—linear regression in our description below—to approximate the value function as a linear or non-linear function of the states:

V (s) = θᵀ φ(s).

Here, φ is some appropriate feature mapping of the states.

For each state s in our finite sample of n states, fitted value iteration will first compute a quantity y(i), which will be our approximation to R(s) + γ maxa Es′∼Ps,a[V (s′)] (the right hand side of Equation 15.8). Then, it will apply a supervised learning algorithm to try to get V (s) close to R(s) + γ maxa Es′∼Ps,a[V (s′)] (or, in other words, to try to get V (s) close to y(i)).

In detail, the algorithm is as follows:

1. Randomly sample n states s(1), s(2), . . . s(n) ∈ S.
2. Initialize θ := 0.
3. Repeat {
4. For i = 1, . . . , n {

5In practice, most MDPs have much smaller action spaces than state spaces. E.g., a car has a 6d state space, and a 2d action space (steering and velocity controls); the inverted pendulum has a 4d state space, and a 1d action space; a helicopter has a 12d state space, and a 4d action space. So, discretizing this set of actions is usually less of a problem than discretizing the state space would have been.





# Fitted Value Iteration

For each action a ∈ A {

Sample s′ , . . . , s′ ∼ Pₛ(i)ₐ (using a model of the MDP).

Set q(a) = kj=1 R(si) + γV (sj)

// Hence, q(a) is an estimate of R(s(i)) + γEₛ′∼Pₛ(i)ₐ [V (s′)].

}

Set y(i) = maxₐ q(a).

// Hence, y(i) is an estimate of R(s(i)) + γ maxₐ Eₛ′∼Pₛ(i)ₐ [V (s′)].

}

// In the original value iteration algorithm (over discrete states)

// we updated the value function according to V (s(i)) := y(i).

// In this algorithm, we want V (s(i)) ≈ y(i), which we’ll achieve

// using supervised learning (linear regression).

Set θ := arg minθ 1/2 ∑i=1n (θT φ(s(i)) − y(i))2

}

Above, we had written out fitted value iteration using linear regression as the algorithm to try to make V (s(i)) close to y(i). That step of the algorithm is completely analogous to a standard supervised learning (regression) problem in which we have a training set (x(1), y(1)), (x(2), y(2)), . . . , (x(n), y(n)), and want to learn a function mapping from x to y; the only difference is that here s plays the role of x. Even though our description above used linear regression, clearly other regression algorithms (such as locally weighted linear regression) can also be used.

Unlike value iteration over a discrete set of states, fitted value iteration cannot be proved to always to converge. However, in practice, it often does converge (or approximately converge), and works well for many problems.

Note also that if we are using a deterministic simulator/model of the MDP, then fitted value iteration can be simplified by setting k = 1 in the algorithm. This is because the expectation in Equation (15.8) becomes an expectation over a deterministic distribution, and so a single example is sufficient to exactly compute that expectation. Otherwise, in the algorithm above, we had to draw k samples, and average to try to approximate that expectation (see the definition of q(a), in the algorithm pseudo-code).





Finally, fitted value iteration outputs V, which is an approximation to V∗. This implicitly defines our policy. Specifically, when our system is in some state s, and we need to choose an action, we would like to choose the action

arg max Eₛ′∼Pₛₐ[V (s′)]

(15.9)

The process for computing/approximating this is similar to the inner-loop of fitted value iteration, where for each action, we sample s′, . . . , s′ ∼ Pₛₐ to approximate the expectation. (And again, if the simulator is deterministic, we can set k = 1.)

In practice, there are often other ways to approximate this step as well. For example, one very common case is if the simulator is of the form sₜ₊₁ = f (sₜ, aₜ) + t, where f is some deterministic function of the states (such as f (sₜ, aₜ) = Asₜ + Baₜ), and t is zero-mean Gaussian noise. In this case, we can pick the action given by

arg max V (f (s, a)).

(15.10)

In other words, here we are just setting t = 0 (i.e., ignoring the noise in the simulator), and setting k = 1. Equivalent, this can be derived from Equation (15.9) using the approximation

Eₛ′ [V (s′)] ≈ V (Eₛ′ [s′])

(15.10)

= V (f (s, a)),

(15.11)

where here the expectation is over the random s′ ∼ Pₛₐ. So long as the noise terms t are small, this will usually be a reasonable approximation. However, for problems that don’t lend themselves to such approximations, having to sample k states using the model, in order to approximate the expectation above, can be computationally expensive.

# 15.5 Connections between Policy and Value Iteration (Optional)

In the policy iteration, line 3 of Algorithm 5, we typically use linear system solver to compute V π. Alternatively, one can also the iterative Bellman updates, similarly to the value iteration, to evaluate V π, as in the Procedure VE(·) in Line 1 of Algorithm 6 below. Here if we take option 1 in Line 2 of the Procedure VE, then the difference between the Procedure VE from the





# 202

# Algorithm 6 Variant of Policy Iteration

1. procedure VE(π, k) . To evaluate V π
2. Option 1: initialize V (s) := 0; Option 2: Initialize from the current V in the main algorithm.
3. for i = 0 to k − 1 do
4. For every state s, update
5. V (s) := R(s) + γ ∑ Psπ(s)(s′)V (s′). (15.12)

return V
6. Require: hyperparameter k.
7. Initialize π randomly.
8. for until convergence do
9. Let V = VE(π, k).
10. For each state s, let




value iteration (Algorithm 4) is that on line 4, the procedure is using the action from π instead of the greedy action.

Using the Procedure VE, we can build Algorithm 6, which is a variant of policy iteration that serves an intermediate algorithm that connects policy iteration and value iteration. Here we are going to use option 2 in VE to maximize the re-use of knowledge learned before. One can verify indeed that if we take k = 1 and use option 2 in Line 2 in Algorithm 6, then Algorithm 6 is semantically equivalent to value iteration (Algorithm 4). In other words, both Algorithm 6 and value iteration interleave the updates in (15.13) and (15.12). Algorithm 6 alternate between k steps of update (15.12) and one step of (15.13), whereas value iteration alternates between 1 steps of update (15.12) and one step of (15.13). Therefore generally Algorithm 6 should not be faster than value iteration, because assuming that update (15.12) and (15.13) are equally useful and time-consuming, then the optimal balance of the update frequencies could be just k = 1 or k ≈ 1.

On the other hand, if k steps of update (15.12) can be done much faster than k times a single step of (15.12), then taking additional steps of equation (15.12) in group might be useful. This is what policy iteration is leveraging — the linear system solver can give us the result of Procedure VE with k = ∞ much faster than using the Procedure VE for a large k. On the flip side, when such a speeding-up effect no longer exists, e.g., when the state space is large and linear system solver is also not fast, then value iteration is more preferable.


Chapter 16
# Chapter 16

# LQR, DDP and LQG

# 16.1  Finite-horizon MDPs

In Chapter 15, we defined Markov Decision Processes (MDPs) and covered Value Iteration / Policy Iteration in a simplified setting. More specifically we introduced the optimal Bellman equation that defines the optimal value function V π∗ of the optimal policy π∗.

V π∗(s) = R(s) + max γ ∑ Ps,a(s′)V π∗(s′)

a∈A              s′∈S

Recall that from the optimal value function, we were able to recover the optimal policy π∗ with

π∗(s) = argmaxa∈A ∑ Ps,a(s′)V ∗(s′)

s′∈S

In this chapter, we’ll place ourselves in a more general setting:

1. We want to write equations that make sense for both the discrete and the continuous case. We’ll therefore write

Es′∼Ps,a [V π∗(s′)]   instead of

∑ Ps,a(s′)V π∗(s′)

s′∈S

meaning that we take the expectation of the value function at the next state. In the finite case, we can rewrite the expectation as a sum over



205

states. In the continuous case, we can rewrite the expectation as an integral. The notation s′ ∼ Pₛₐ means that the state s′ is sampled from the distribution Pₛₐ.

1. We’ll assume that the rewards depend on both states and actions. In other words, R : S × A → R. This implies that the previous mechanism for computing the optimal action is changed into

π∗(s) = argmaxₐ∈A R(s, a) + γEₛ′∼Pₛₐ [V π∗(s′)]

Instead of considering an infinite horizon MDP, we’ll assume that we have a finite horizon MDP that will be defined as a tuple

(S , A, Pₛₐ, T, R)

with T > 0 the time horizon (for instance T = 100). In this setting, our definition of payoff is going to be (slightly) different:

R(s₀, a₀) + R(s₁, a₁) + · · · + R(sT , aT)

instead of (infinite horizon case)

R(s₀, a₀) + γR(s₁, a₁) + γ²R(s₂, a₂) + . . .

∑∞ R(sₜ, aₜ)γᵗ

t=0

What happened to the discount factor γ? Remember that the introduction of γ was (partly) justified by the necessity of making sure that the infinite sum would be finite and well-defined. If the rewards are bounded by a constant ¯R, the payoff is indeed bounded by

∑∞ R(s )γt ≤ ¯R ∑∞ γ

t=0

and we recognize a geometric sum! Here, as the payoff is a finite sum, the discount factor γ is not necessary anymore.





# 206

In this new setting, things behave quite differently. First, the optimal policy *π∗* might be non-stationary, meaning that it changes over time. In other words, now we have

*π(t) : S → A*

where the superscript (t) denotes the policy at time step t. The dynamics of the finite horizon MDP following policy *π(t) proceeds as follows: we start in some state s₀, take some action a₀ := π(0)(s₀) according to our policy at time step 0. The MDP transitions to a successor s₁, drawn according to Pₛ₀ₐ₀. Then, we get to pick another action a₁ := π(1)(s₁)* following our new policy at time step 1 and so on...

Why does the optimal policy happen to be non-stationary in the finite-horizon setting? Intuitively, as we have a finite number of actions to take, we might want to adopt different strategies depending on where we are in the environment and how much time we have left. Imagine a grid with 2 goals with rewards +1 and +10. At the beginning, we might want to take actions to aim for the +10 goal. But if after some steps, dynamics somehow pushed us closer to the +1 goal and we don’t have enough steps left to be able to reach the +10 goal, then a better strategy would be to aim for the +1 goal...

4. This observation allows us to use time dependent dynamics

sₜ₊₁ ∼ P (t)

st,at

meaning that the transition’s distribution *P (t) changes over time. The same thing can be said about R(t). Note sᵗ,aᵗ* that this setting is a better model for real life. In a car, the gas tank empties, traffic changes, etc. Combining the previous remarks, we’ll use the following general formulation for our finite horizon MDP

(S , A, P (t), T, R(t))

sa

Remark: notice that the above formulation would be equivalent to adding the time into the state.





The value function at time t for a policy π is then defined in the same way as before, as an expectation over trajectories generated following policy π starting in state s.

Vₜ(s) = E [R(t)(sₜ, aₜ) + · · · + R(ᵀ )(sT , aT ) | sₜ = s, π]

Now, the question is

In this finite-horizon setting, how do we find the optimal value function

Vₜ∗(s) = max Vₜπ(s)

It turns out that Bellman’s equation for Value Iteration is made for Dynamic Programming. This may come as no surprise as Bellman is one of the fathers of dynamic programming and the Bellman equation is strongly related to the field. To understand how we can simplify the problem by adopting an iteration-based approach, we make the following observations:

1. Notice that at the end of the game (for time step T), the optimal value is obvious

∀s ∈ S : V ∗(s) := max R(ᵀ )(s, a)  (16.1)

For another time step 0 ≤ t &#x3C; T, if we suppose that we know the optimal value function for the next time step V ∗, then we have

∀t &#x3C; T, s ∈ S : Vₜ∗(s) := max [R(t)(s, a) + Es′∼Psa [V ∗(s′)]] (16.2)

With these observations in mind, we can come up with a clever algorithm to solve for the optimal value function:

1. compute V ∗ using equation (16.1).
2. for t = T − 1, . . . , 0:

compute Vₜ∗ using V ∗ using equation (16.2)

t+1





208

Side note We can interpret standard value iteration as a special case of this general case, but without keeping track of time. It turns out that in the standard setting, if we run value iteration for T steps, we get a γᵀ approximation of the optimal value iteration (geometric convergence). See problem set 4 for a proof of the following result:

If Vt denotes the value function at the t-th step, then

||Vt+1 − V∗||∞ = ||B(Vt) − V∗||∞

≤ γ||Vt − V∗||∞

≤ γt||V1 − V∗||∞

In other words, the Bellman operator B is a γ-contracting operator.

# 16.2 Linear Quadratic Regulation (LQR)

In this section, we’ll cover a special case of the finite-horizon setting described in Section 16.1, for which the exact solution is (easily) tractable. This model is widely used in robotics, and a common technique in many problems is to reduce the formulation to this framework.

First, let’s describe the model’s assumptions. We place ourselves in the continuous setting, with

S = Rd, A = Rd

and we’ll assume linear transitions (with noise)

st+1 = Atst + Btat + wt

where At ∈ Rd×d, Bt ∈ Rd×d are matrices and wt ∼ N(0, Σt) is some gaussian noise (with zero mean). As we’ll show in the following paragraphs, it turns out that the noise, as long as it has zero mean, does not impact the optimal policy!

We’ll also assume quadratic rewards

R(t)(st, at) = −sTUtst − aTWtat

t





where Uₜ ∈ Rᵈ×n, Wₜ ∈ Rᵈ×d are positive definite matrices (meaning that the reward is always negative).

Remark Note that the quadratic formulation of the reward is equivalent to saying that we want our state to be close to the origin (where the reward is higher). For example, if Uₜ = Id (the identity matrix) and Wₜ = Id, then Rₜ = −||sₜ||² − ||aₜ||², meaning that we want to take smooth actions (small norm of aₜ) to go back to the origin (small norm of sₜ). This could model a car trying to stay in the middle of lane without making impulsive moves...

Now that we have defined the assumptions of our LQR model, let’s cover the 2 steps of the LQR algorithm

1. Step 1 suppose that we don’t know the matrices A, B, Σ. To estimate them, we can follow the ideas outlined in the Value Approximation section of the RL notes. First, collect transitions from an arbitrary policy. Then, use linear regression to find

argminA,B ∑i=1n ∥s(i) − A s(i) + B a(i)∥₂

Finally, use a technique seen in Gaussian Discriminant Analysis to learn Σ.
2. Step 2 assuming that the parameters of our model are known (given or estimated with step 1), we can derive the optimal policy using dynamic programming.

In other words, given

{

sₜ₊₁ = Aₜsₜ + Bₜaₜ + wₜ Aₜ, Bₜ, Uₜ, Wₜ, Σₜ known

R(t)(sₜ, aₜ) = −s>Uₜsₜ − a>Wₜaₜ

we want to compute Vₜ∗. If we go back to section 16.1, we can apply dynamic programming, which yields

1. Initialization step For the last time step T,

V ∗(sT) = maxaT ∈ A RT(sT, aT)

= maxaT ∈ A −s>UₜsT − a>WₜaT

= −s>UₜsT (maximized for aT = 0)





# 2. Recurrence step

Let t &#x3C; T. Suppose we know V*t+1.

Fact 1: It can be shown that if V*t+1 is a quadratic function in st, then V*t is also a quadratic function. In other words, there exists some matrix Φ and some scalar Ψ such that

if V*(st+1) = s> Φt+1st+1 + Ψt+1

then Vt*(st) = s>Φtst + Ψt

For time step t = T, we had Φt = −UT and ΨT = 0.

Fact 2: We can show that the optimal policy is just a linear function of the state. Knowing V*t+1 is equivalent to knowing Φt+1 and Ψt+1, so we just need to explain how we compute Φt and Ψt from Φt+1 and Ψt+1 and the other parameters of the problem.

Vt*(st) = s>Φtst + Ψt

= max R(t)(st, at) + Est+1∼Pst,at[V*(st+1)]

= max −st Utst − at Vtat + Est+1∼N(Atst + Btat, Σt)[st+1Φt+1st+1 + Ψt+1]

where the second line is just the definition of the optimal value function and the third line is obtained by plugging in the dynamics of our model along with the quadratic assumption. Notice that the last expression is a quadratic function in at and can thus be (easily) optimized1. We get the optimal action a*t

a*t = [(B>Φt+1Bt − Vt)−1BtΦt+1At] · st

= Lt · st

where Lt := [(B>Φt+1Bt − Wt)−1BtΦt+1At]

1Use the identity Es[w>Φt+1w] = Tr(ΣtΦt+1) with wt ∼ N(0, Σt)





which is an impressive result: our optimal policy is linear in st. Given a* we can solve for Φt and Ψt. We finally get the Discrete Ricatti equations

Φt = A> (Φt+1 − Φt+1Bt (B>Φt+1Bt − Wt)−1 BtΦt+1) At − Ut
Ψt = − tr (ΣtΦt+1) + Ψt+1

Fact 3: we notice that Φt depends on neither Ψ nor the noise Σt! As Lt is a function of At, Bt and Φt+1, it implies that the optimal policy also does not depend on the noise! (But Ψt does depend on Σt, which implies that Vt* depends on Σt.)

Then, to summarize, the LQR algorithm works as follows

1. (if necessary) estimate parameters At, Bt, Σt
2. initialize ΦT := −UT and ΨT := 0.
3. iterate from t = T − 1 . . . 0 to update Φt and Ψt using Φt+1 and Ψt+1 using the discrete Ricatti equations. If there exists a policy that drives the state towards zero, then convergence is guaranteed!

Using Fact 3, we can be even more clever and make our algorithm run (slightly) faster! As the optimal policy does not depend on Ψt, and the update of Φt only depends on Φt, it is sufficient to update only Φt!

# 16.3 From non-linear dynamics to LQR

It turns out that a lot of problems can be reduced to LQR, even if dynamics are non-linear. While LQR is a nice formulation because we are able to come up with a nice exact solution, it is far from being general. Let’s take for instance the case of the inverted pendulum. The transitions between states look like

xt+1      xt       
xt+1                 
˙                  x
 t+1 = F  ˙ t , a 
 θt+1      θt    t
˙                  ˙
θt+1               θt

where the function F depends on the cos of the angle etc. Now, the question we may ask is

Can we linearize this system?





# 16.3.1  Linearization of dynamics

Let’s suppose that at time t, the system spends most of its time in some state s̄ and the actions we perform are around ā. For the inverted pendulum, if we reached some kind of optimal, this is true: our actions are small and we don’t deviate much from the vertical.

We are going to use Taylor expansion to linearize the dynamics. In the simple case where the state is one-dimensional and the transition function F does not depend on the action, we would write something like

st+1 = F (st) ≈ F (s̄) + F (s̄) · (st − s̄)

In the more general setting, the formula looks the same, with gradients instead of simple derivatives

st+1 ≈ F (s̄, ā) + ∇ F (s̄, ā) · (st − s̄) + ∇ F (s̄, ā) · (at − ā)  (16.3)

and now, st+1 is linear in st and at, because we can rewrite equation (16.3) as

st+1 ≈ A st + B st + κ

where κ is some constant and A, B are matrices. Now, this writing looks awfully similar to the assumptions made for LQR. We just have to get rid of the constant term κ! It turns out that the constant term can be absorbed into st by artificially increasing the dimension by one. This is the same trick that we used at the beginning of the class for linear regression...

# 16.3.2    Differential Dynamic Programming (DDP)

The previous method works well for cases where the goal is to stay around some state s* (think about the inverted pendulum, or a car having to stay in the middle of a lane). However, in some cases, the goal can be more complicated.

We’ll cover a method that applies when our system has to follow some trajectory (think about a rocket). This method is going to discretize the trajectory into discrete time steps, and create intermediary goals around which we will be able to use the previous technique! This method is called Differential Dynamic Programming. The main steps are





# 213

step 1 come up with a nominal trajectory using a naive controller, that approximate the trajectory we want to follow. In other words, our controller is able to approximate the gold trajectory with

s∗, a∗ → s∗, a∗ → . . .

0                       0       1  1

step 2 linearize the dynamics around each trajectory point s∗, in other words

t

sₜ₊₁ ≈ F (s∗, a∗) + ∇ₛF (s∗, a∗)(sₜ − s∗) + ∇ₐF (s∗, a∗)(aₜ − a∗)

t  t                    t  t                       t    t  t    t

where sₜ, aₜ would be our current state and action. Now that we have a linear approximation around each of these points, we can use the previous section and rewrite

sₜ₊₁ = Aₜ · sₜ + Bₜ · aₜ

(notice that in that case, we use the non-stationary dynamics setting that we mentioned at the beginning of these lecture notes)

Note We can apply a similar derivation for the reward R(t), with a second-order Taylor expansion.

R(sₜ, aₜ) ≈ R(s∗, a∗) + ∇ₛR(s∗, a∗)(sₜ − s∗) + ∇ₐR(s∗, a∗)(aₜ − a∗)

t    t                  t          t    t               t    t    t

+ 1 (sₜ − s∗)>Hₛₛ(sₜ − s∗) + (sₜ − s∗)>Hₛₐ(aₜ − a∗)

2            t                        t         t                    t

+ 1 (aₜ − a∗)>Hₐₐ(aₜ − a∗)

2            t                  t

where Hxy refers to the entry of the Hessian of R with respect to x and y evaluated in (s∗, a∗) (omitted for readability). This expression can be re-written as     t    t

Rₜ(sₜ, aₜ) = −s>Uₜsₜ − a>Wₜaₜ

t         t

for some matrices Uₜ, Wₜ, with the same trick of adding an extra dimension of ones. To convince yourself, notice that

(1   x) · (a         b) · (1) = a + 2bx + cx₂

b     c       x





step 3 Now, you can convince yourself that our problem is strictly re-written in the LQR framework. Let’s just use LQR to find the optimal policy πₜ. As a result, our new controller will (hopefully) be better! Note: Some problems might arise if the LQR trajectory deviates too much from the linearized approximation of the trajectory, but that can be fixed with reward-shaping...

step 4 Now that we get a new controller (our new policy πₜ), we use it to produce a new trajectory

s∗, π₀(s∗) → s∗, π₁(s∗) → . . . → s∗

0             0  1  1                T

note that when we generate this new trajectory, we use the real F and not its linear approximation to compute transitions, meaning that

s∗  = F (s∗, a∗)

t+1  t  t

then, go back to step 2 and repeat until some stopping criterion.

# 16.4  Linear Quadratic Gaussian (LQG)

Often, in the real word, we don’t get to observe the full state sₜ. For example, an autonomous car could receive an image from a camera, which is merely an observation, and not the full state of the world. So far, we assumed that the state was available. As this might not hold true for most of the real-world problems, we need a new tool to model this situation: Partially Observable MDPs.

A POMDP is an MDP with an extra observation layer. In other words, we introduce a new variable oₜ, that follows some conditional distribution given the current state sₜ

oₜ|sₜ ∼ O(o|s)

Formally, a finite-horizon POMDP is given by a tuple

(S , O, A, Pₛₐ, T, R)

Within this framework, the general strategy is to maintain a belief state (distribution over states) based on the observation o₁, . . . , oₜ. Then, a policy in a POMDP maps this belief states to actions.





# 215

In this section, we’ll present a extension of LQR to this new setting. Assume that we observe *yₜ ∈ Rⁿ with m &#x3C; n* such that

*yₜ = C · sₜ + vₜ*

*sₜ₊₁ = A · sₜ + B · aₜ + wₜ*

where *C ∈ Rⁿ×d is a compression matrix and vₜ is the sensor noise (also gaussian, like wₜ). Note that the reward function R(t)* is left unchanged, as a function of the state (not the observation) and action. Also, as distributions are gaussian, the belief state is also going to be gaussian. In this new framework, let’s give an overview of the strategy we are going to adopt to find the optimal policy:

# Step 1

first, compute the distribution on the possible states (the belief state), based on the observations we have. In other words, we want to compute the mean *st|ₜ and the covariance Σt|ₜ* of

*sₜ|y₁, . . . , yₜ ∼ N (st|ₜ, Σt|ₜ)*

to perform the computation efficiently over time, we’ll use the Kalman Filter algorithm (used on-board Apollo Lunar Module!).

# Step 2

now that we have the distribution, we’ll use the mean *st|ₜ as the best approximation for sₜ*

# Step 3

then set the action *aₜ := Lₜst|ₜ where Lₜ* comes from the regular LQR algorithm.

Intuitively, to understand why this works, notice that *st|ₜ is a noisy approximation of sₜ* (equivalent to adding more noise to LQR) but we proved that LQR is independent of the noise!

Step 1 needs to be explicated. We’ll cover a simple case where there is no action dependence in our dynamics (but the general case follows the same idea). Suppose that

*sₜ₊₁ = A · sₜ + wₜ,      wₜ ∼ N (0, Σₛ)*

*yₜ = C · sₜ + vₜ,      vₜ ∼ N (0, Σy)*

As noises are Gaussians, we can easily prove that the joint distribution is also Gaussian.





216

s1
.
.
s
t ∼ N (μ, Σ) for some μ, Σ

y1
.
.
yt

then, using the marginal formulas of gaussians (see Factor Analysis notes), we would get

st | y1, . . . , yt ∼ N (st | t, Σt | t)

However, computing the marginal distribution parameters using these formulas would be computationally expensive! It would require manipulating matrices of shape t × t. Recall that inverting a matrix can be done in O(t3), and it would then have to be repeated over the time steps, yielding a cost in O(t4)!

The Kalman filter algorithm provides a much better way of computing the mean and variance, by updating them over time in constant time in t! The Kalman filter is based on two basic steps. Assume that we know the distribution of st | y1, . . . , yt:

- predict step compute st+1 | y1, . . . , yt
- update step compute st+1 | y1, . . . , yt+1

and iterate over time steps! The combination of the predict and update steps updates our belief states. In other words, the process looks like

(s | y1, . . . , yt) predict



217

{
st+1|ₜ = A · st|ₜ
Σt+1|ₜ = A · Σt|ₜ · A> + Σₛ
}

update step given st+1|ₜ and Σt+1|ₜ such that

sₜ₊₁|y₁, . . . , yₜ ∼ N (st+1|ₜ, Σt+1|ₜ)

we can prove that

sₜ₊₁|y₁, . . . , yₜ₊₁ ∼ N (st+1|ₜ₊₁, Σt+1|ₜ₊₁)

where

{
st+1|ₜ₊₁ = st+1|ₜ + Kₜ(yₜ₊₁ − Cst+1|ₜ)
Σt+1|ₜ₊₁ = Σt+1|ₜ − Kₜ · C · Σt+1|ₜ
}

with

Kₜ := Σt+1|ₜC >(C Σt+1|ₜC > + Σy)−1

The matrix Kₜ is called the Kalman gain.

Now, if we have a closer look at the formulas, we notice that we don’t need the observations prior to time step t! The update steps only depend on the previous distribution. Putting it all together, the algorithm first runs a forward pass to compute the Kₜ, Σt|ₜ and st|ₜ (sometimes referred to as sˆ in the literature). Then, it runs a backward pass (the LQR updates) to compute the quantities Ψₜ, Ψₜ and Lₜ. Finally, we recover the optimal policy with a∗ = Lₜst|ₜ.





Chapter 17

# Policy Gradient (REINFORCE)

We will present a model-free algorithm called REINFORCE that does not require the notion of value functions and Q functions. It turns out to be more convenient to introduce REINFORCE in the finite horizon case, which will be assumed throughout this note: we use τ = (s₀, a₀, . . . , sT −1, aT −1, sT ) to denote a trajectory, where T &#x3C; ∞ is the length of the trajectory. Moreover, REINFORCE only applies to learning a randomized policy. We use πθ(a|s) to denote the probability of the policy πθ outputting the action a at state s. The other notations will be the same as in previous lecture notes.

The advantage of applying REINFORCE is that we only need to assume that we can sample from the transition probabilities {Ps,a} and can query the reward function R(s, a)1 at state s and action a, but we do not need to know the analytical form of the transition probabilities or the reward function. We do not explicitly learn the transition probabilities or the reward function either.

Let s₀ be sampled from some distribution μ. We consider optimizing the expected total payoff of the policy πθ over the parameter θ defined as:

η(θ) = Eτ   [∑t=0T−1 γt R(st, at)]   (17.1)

Recall that st ∼ Pst−1,at−1} and at ∼ πθ(·|st). Also note that η(θ) = Es₀∼P[Vπθ (s₀)] if we ignore the difference between finite and infinite horizon.

1 In this notes we will work with the general setting where the reward depends on both the state and the action.

218




We aim to use gradient ascent to maximize η(θ). The main challenge we face here is to compute (or estimate) the gradient of η(θ) without the knowledge of the form of the reward function and the transition probabilities. Let P (τ) denote the distribution of τ (generated by the policy π), and let f (τ, θ) = ∑t=0T −1 γ R(st, at). We can rewrite η(θ) as

η(θ) = Eτ∼Pθ[f (τ)]                                 (17.2)

We face a similar situation in the variational auto-encoder (VAE) setting covered in the previous lectures, where we need to take the gradient w.r.t to a variable that shows up under the expectation — the distribution Pθ depends on θ. Recall that in VAE, we used the re-parametrization techniques to address this problem. However, it does not apply here because we do not know how to compute the gradient of the function f. (We only have an efficient way to evaluate the function f by taking a weighted sum of the observed rewards, but we do not necessarily know the reward function itself to compute the gradient.)

The REINFORCE algorithm uses another approach to estimate the gradient of η(θ). We start with the following derivation:

∫

∇θEτ∼Pθ[f (τ)] = ∇θ     Pθ(τ)f (τ)dτ

∫

=     ∇θ(Pθ(τ)f (τ))dτ  (swap integration with gradient)

∫

=     (∇θPθ(τ))f (τ)dτ            (because f does not depend on θ)

∫

=     Pθ(τ)(∇θ log Pθ(τ))f (τ)dτ

(because ∇ log Pθ(τ) = ∇Pθ(τ))

Pθ(τ)

= Eτ∼Pθ[(∇θ log Pθ(τ))f (τ)]        (17.3)

Now we have a sample-based estimator for ∇θEτ∼Pθ[f (τ)]. Let τ(1), . . . , τ(n) be n empirical samples from Pθ (which are obtained by running the policy πθ for n times, with T steps for each run). We can estimate the gradient of η(θ) by

∇θEτ∼Pθ[f (τ)] = Eτ∼Pθ[(∇θ log Pθ(τ))f (τ)]        (17.4)

∑

≈ 1                  n (∇θ log Pθ(τ(i)))f (τ(i))    (17.5)

n i=1





The next question is how to compute log Pθ(τ). We derive an analytical formula for log Pθ(τ) and compute its gradient w.r.t θ (using auto-differentiation). Using the definition of τ, we have

Pθ(τ) = μ(s0)πθ(a0|s0)Ps0a0(s1)πθ(a1|s1)Ps1a1(s2) · · · PsT − 1aT − 1(sT)

(17.6)

Here recall that μ is used to denote the density of the distribution of s0. It follows that

log Pθ(τ) = log μ(s0) + log πθ(a0|s0) + log Ps0a0(s1) + log πθ(a1|s1) + log Ps1a1(s2) + · · · + log PsT − 1aT − 1(sT)

(17.7)

Taking gradient w.r.t to θ, we obtain

∇θ log Pθ(τ) = ∇θ log πθ(a0|s0) + ∇θ log πθ(a1|s1) + · · · + ∇θ log πθ(aT − 1|sT − 1)

Note that many of the terms disappear because they don’t depend on θ and thus have zero gradients. (This is somewhat important — we don’t know how to evaluate those terms such as log Ps0a0(s1) because we don’t have access to the transition probabilities, but luckily those terms have zero gradients!)

Plugging the equation above into equation (17.4), we conclude that

∇θη(θ) = ∇θEτ∼Pθ[f(τ)] = Eτ∼Pθ∇θ log πθ(at|st) · f(τ)

[(∑ )]

T − 1

= Eτ∼Pθ∇θ log πθ(at|st) · γtR(st, at)

[( t=0 ) ( )]

∑

T − 1

= Eτ∼Pθ∇θ log πθ(at|st) ·

∑

γtR(st, at)

(17.8)

We estimate the RHS of the equation above by empirical sample trajectories, and the estimate is unbiased. The vanilla REINFORCE algorithm iteratively updates the parameter by gradient ascent using the estimated gradients.

Interpretation of the policy gradient formula (17.8). The quantity ∇θPθ(τ) = ∑t=0T − 1 ∇θ log πθ(at|st) is intuitively the direction of the change of θ that will make the trajectory τ more likely to occur (or increase the probability of choosing action a0, . . . , aT − 1), and f(τ) is the total payoff of this trajectory. Thus, by taking a gradient step, intuitively we are trying to improve the likelihood of all the trajectories, but with a different emphasis or weight for each τ (or for each set of actions a0, a1, . . . , aT − 1). If τ is very rewarding (that is, f(τ) is large), we try very hard to move in the direction





that can increase the probability of the trajectory τ (or the direction that increases the probability of choosing a₀, . . . , at−1), and if τ has low payoff, we try less hard with a smaller weight.

An interesting fact that follows from formula (17.3) is that

[∑]
T −1
Eτ∼Pθ ∇θ log πθ(aₜ|sₜ) = 0
(17.9)

To see this, we take f (τ ) = 1 (that is, the reward is always a constant), then the LHS of (17.8) is zero because the payoff is always a fixed constant ∑T γᵗ. Thus the RHS of (17.8) is also zero, which implies (17.9).

In fact, one can verify that Eₐₜ∼πθ(·|sₜ)∇θ log πθ(aₜ|sₜ) = 0 for any fixed t and sₜ.² This fact has two consequences. First, we can simplify formula (17.8) to

∑
T −1
∇θη(θ) = Eτ∼Pθ ∇θ log πθ(aₜ|sₜ) · ∑
T −1
γʲ R(sj , aj )
(17.10)

where the second equality follows from

Eτ∼Pθ [∇θ log πθ(aₜ|sₜ) · ( ∑ γʲ R(sj , aj ))]

= E [E [∇ log π (a |s )|s 0≤j<t (="" ∑="" j="" )]&#x3C;="" span="">
θ θ t t 0, a0, . . . , st−1, at−1, st] · γ R(sj , aj )
0≤j<t< span="">
</t<></t>

= 0 (because E [∇θ log πθ(aₜ|sₜ)|s₀, a₀, . . . , st−1, at−1, sₜ] = 0)

Note that here we used the law of total expectation. The outer expectation in the second line above is over the randomness of s₀, a₀, . . . , at−1, sₜ, whereas the inner expectation is over the randomness of aₜ (conditioned on s₀, a₀, . . . , at−1, sₜ.) We see that we’ve made the estimator slightly simpler.

The second consequence of Eₐₜ∼πθ(·|sₜ)∇θ log πθ(aₜ|sₜ) = 0 is the following: for any value B(sₜ) that only depends on sₜ, it holds that

Eτ∼Pθ [∇θ log πθ(aₜ|sₜ) · B(sₜ)] = E [E [∇θ log πθ(aₜ|sₜ)|s₀, a₀, . . . , st−1, at−1, sₜ] B(sₜ)] = 0 (because E [∇θ log πθ(aₜ|sₜ)|s₀, a₀, . . . , st−1, at−1, sₜ] = 0)

²In general, it’s true that Ex∼pθ [∇ log pθ(x)] = 0.





Again here we used the law of total expectation. The outer expectation in the second line above is over the randomness of s₀, a₀, . . . , aₜ₋₁, sₜ, whereas the inner expectation is over the randomness of aₜ (conditioned on s₀, a₀, . . . , aₜ₋₁, sₜ.) It follows from equation (17.10) and the equation above that

∑ T − 1 [ (∑ ) ]

∇θη(θ) =  Eτ∼Pθ ∇θ log πθ(aₜ|sₜ) · γʲ R(sj , aj ) − γᵗB(sₜ)

t=0 j≥t

∑ T − 1 [ (∑ ) ]

=  Eτ∼Pθ ∇θ log πθ(aₜ|sₜ) · γᵗ     γʲ−tR(sj , aj ) − B(sₜ)

t=0 j≥t

(17.11)

Therefore, we will get a different estimator for estimating the ∇η(θ) with a difference choice of B(·). The benefit of introducing a proper B(·) — which is often referred to as a baseline — is that it helps reduce the variance of the estimator.³ It turns out that a near optimal estimator would be the expected future payoff E [∑ᵀ −1 γʲ−tR(sj , aj )|sₜ], which is pretty much the same as the value function V πθ(sₜ) (if we ignore the difference between finite and infinite horizon.) Here one could estimate the value function V πθ(·) in a crude way, because its precise value doesn’t influence the mean of the estimator but only the variance. This leads to a policy gradient algorithm with baselines stated in Algorithm 7.⁴

³As a heuristic but illustrating example, suppose for a fixed t, the future reward ∑ T−1 γj−tR(sj , aj ) randomly takes two values 1000 + 1 and 1000 − 2 with equal probability, and the corresponding values for ∇θ log πθ(at|st) are vector z and −z. (Note that because E [∇θ log πθ(at|st)] = 0, if ∇θ log πθ(at|st) can only take two values uniformly, then the two values have to two vectors in an opposite direction.) In this case, without subtracting the baseline, the estimators take two values (1000 + 1)z and −(1000 − 2)z, whereas after subtracting a baseline of 1000, the estimator has two values z and 2z. The latter estimator has much lower variance compared to the original estimator.

⁴We note that the estimator of the gradient in the algorithm does not exactly match the equation 17.11. If we multiply γt in the summand of equation (17.13), then they will exactly match. Removing such discount factors empirically works well because it gives a large update.





# 223

# Algorithm 7 Vanilla policy gradient with baseline

for i = 1, · · · do

Collect a set of trajectories by executing the current policy. Use R as a shorthand for ∑t=0T-1 γj-tR(sj, aj)

Fit the baseline by finding a function B that minimizes

∑τ ∑t(R≥t − B(st))2 (17.12)

Update the policy parameter θ with the gradient estimator

∑τ ∑t ∇θ log πθ(at | st) · (R≥t − B(st)) (17.13)




# Bibliography

Mikhail Belkin, Daniel Hsu, Siyuan Ma, and Soumik Mandal. Reconciling modern machine-learning practice and the classical bias–variance trade-off. Proceedings of the National Academy of Sciences, 116(32):15849–15854, 2019.

Mikhail Belkin, Daniel Hsu, and Ji Xu. Two models of double descent for weak features. SIAM Journal on Mathematics of Data Science, 2(4):1167–1180, 2020.

David M Blei, Alp Kucukelbir, and Jon D McAuliffe. Variational inference: A review for statisticians. Journal of the American Statistical Association, 112(518):859–877, 2017.

Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International Conference on Machine Learning, pages 1597–1607. PMLR, 2020.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, 2019.

224



225

Jeff Z HaoChen, Colin Wei, Jason D Lee, and Tengyu Ma. Shape matters: Understanding the implicit bias of the noise covariance. arXiv preprint arXiv:2006.08680, 2020.

Trevor Hastie, Andrea Montanari, Saharon Rosset, and Ryan J Tibshirani. Surprises in high-dimensional ridgeless least squares interpolation. 2019.

Trevor Hastie, Andrea Montanari, Saharon Rosset, and Ryan J Tibshirani. Surprises in high-dimensional ridgeless least squares interpolation. The Annals of Statistics, 50(2):949–986, 2022.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani. An introduction to statistical learning, second edition, volume 112. Springer, 2021.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Yuping Luo, Huazhe Xu, Yuanzhi Li, Yuandong Tian, Trevor Darrell, and Tengyu Ma. Algorithmic framework for model-based deep reinforcement learning with theoretical guarantees. In International Conference on Learning Representations, 2018.

Song Mei and Andrea Montanari. The generalization error of random features regression: Precise asymptotics and the double descent curve. Communications on Pure and Applied Mathematics, 75(4):667–766, 2022.

Preetum Nakkiran. More data can hurt for linear regression: Sample-wise double descent. 2019.

Preetum Nakkiran, Prayaag Venkat, Sham Kakade, and Tengyu Ma. Optimal regularization can mitigate double descent. 2020.

Manfred Opper. Statistical mechanics of learning: Generalization. The handbook of brain theory and neural networks, pages 922–925, 1995.

Manfred Opper. Learning to generalize. Frontiers of Life, 3(part 2):763–775, 2001.





# References

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. arXiv preprint arXiv:1706.03762, 2017.

Blake Woodworth, Suriya Gunasekar, Jason D Lee, Edward Moroshko, Pedro Savarese, Itay Golan, Daniel Soudry, and Nathan Srebro. Kernel and rich regimes in overparametrized models. arXiv preprint arXiv:2002.09277, 2020.


