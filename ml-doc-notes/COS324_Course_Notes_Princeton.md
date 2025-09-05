
S A N J E E V A R O R A , S I M O N PA R K , D E N N I S J A - C O B , D A N Q I C H E N

# INTRODUCTION TO MACHINE LEARNING

# LECTURE NOTES FOR COS 324 AT PRINCETON UNIVERSITY



# Copyright © 2024 Sanjeev Arora, Simon Park, Dennis Jacob, Danqi Chen

# Published by

tufte-latex.github.io/tufte-latex

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 2.0 Generic License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/2.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

September 8, 2024





# Contents

- I   Supervised Learning           13
- 1     Linear Regression: An Introduction                15
- 1.1  A Warm-up Example    15
- 1.2  Using Linear Regression for Sentiment Prediction   18
- 1.3  Importance of Featurization     22
- 1.4  Linear Regression in Python Programming        24
- 2     Statistical Learning: What It Means to Learn            29
- 2.1  A Warm-up Example    29
- 2.2  Summary of Statistical Learning   31
- 2.3  Implications for Applications of Machine Learning  31
- 3     Optimization via Gradient Descent                 33
- 3.1  Gradient Descent     33
- 3.2  Implications of the Linearity of a Gradient    37
- 3.3  Regularizers  38
- 3.4  Gradient Descent in Python Programming         42
- 4     Linear Classification       47
- 4.1  General Form of a Linear Model    47
- 4.2  Logistic Regression  48





# 4.3 Support Vector Machines

# 4.4 Multi-class Classification (Multinomial Regression)

# 4.5 Regularization with SVM

# 4.6 Linear Classification in Python Programming

# 5 Exploring “Data Science” via Linear Regression

# 5.1 Boston Housing: Machine Learning in Economics

# 5.2 fMRI Analysis: Machine Learning in Neuroscience

# II Unsupervised Learning

# 6 Clustering

# 6.1 Unsupervised Learning

# 6.2 Clustering

# 6.3 k-Means Clustering

# 6.4 Clustering in Programming

# 7 Low-Dimensional Representation

# 7.1 Low-Dimensional Representation with Error

# 7.2 Application 1: Stylometry

# 7.3 Application 2: Eigenfaces

# 8 n-Gram Language Models

# 8.1 Probabilistic Model of Language

# 8.2 n-Gram Models

# 8.3 Start and Stop Tokens

# 8.4 Testing a Language Model

# 9 Matrix Factorization and Recommender Systems

# 9.1 Recommender Systems

# 9.2 Recommender Systems via Matrix Factorization

# 9.3 Implementation of Matrix Factorization





# III  Deep Learning

# Introduction to Deep Learning

10.1 A Brief History

10.2 Anatomy of a Neural Network

10.3 Why Deep Learning?

10.4 Multi-class Classification

# Feedforward Neural Network and Backpropagation

11.1 Forward Propagation: An Example

11.2 Forward Propagation: The General Case

11.3 Backpropagation: An Example

11.4 Backpropagation: The General Case

11.5 Feedforward Neural Network in Python Programming

# Convolutional Neural Network

12.1 Introduction to Convolution

12.2 Convolution in Computer Vision

12.3 Backpropagation for Convolutional Nets

12.4 CNN in Python Programming

# IV   Reinforcement Learning

# Introduction to Reinforcement Learning

13.1 Basic Elements of Reinforcement Learning

13.2 Useful Resource: MuJoCo-based RL Environments

13.3 Illustrative Example: Optimum Cake Eating

# Markov Decision Process

14.1 Markov Decision Process (MDP)

14.2 Policy and Markov Reward Process

14.3 Optimal Policy





# 15 Reinforcement Learning in Unknown Environments

# 15.1 Model-Free Reinforcement Learning

# 15.2 Atari Pong (1972): A Case Study

# 15.3 Q-learning

# 15.4 Applications of Reinforcement Learning

# 15.5 Deep Reinforcement Learning

# V Advanced Topics

# 16 Machine Learning and Ethics

# 16.1 Facebook’s Suicide Prevention

# 16.2 Racial Bias in Machine Learning

# 16.3 Conceptions of Fairness in Machine Learning

# 16.4 Limitations of the ML Paradigm

# 16.5 Final Thoughts

# 17 Deep Learning for Natural Language Processing

# 17.1 Word Embeddings

# 17.2 N-gram Model Revisited

# VI Mathematics for Machine Learning

# 18 Probability and Statistics

# 18.1 Probability and Event

# 18.2 Random Variable

# 18.3 Central Limit Theorem and Confidence Intervals

# 18.4 Final Remarks

# 19 Calculus

# 19.1 Calculus in One Variable

# 19.2 Multivariable Calculus




# 20 Linear Algebra

# 20.1 Vectors

245

# 20.2 Matrices

249

# 20.3 Advanced: SVD/PCA Procedures

253


NO_CONTENT_HERE



# Preface

# Introduction

These lecture notes accompany a junior-level machine learning course (COS 324) at Princeton University. This course provides a broad introduction to machine learning paradigms including supervised, unsupervised, deep learning, and reinforcement learning as a foundation for further study or independent work in ML, AI, and data science. Topics include linear models for classification and regression, clustering, low rank representations (PCA), n-gram language models, matrix factorization, feedforward neural nets and convolutional neural nets, Markov decision process, and reinforcement learning. Interesting applications are presented for all these models. The course design was shaped by some constraints that may not exist at other universities.

Background assumed: The formal prerequisites are the following courses, which all our majors have taken through sophomore year: An introduction to computer science course (COS 126), Data Structures and Algorithms (COS 226), Single-variable Calculus (MAT 103, 104) and Linear Algebra (MAT 202/204/217). While many majors also take multi-variable calculus and a probability course, not all do. Hence we do not include them in the list of prerequisites, although we do assume some exposure to elementary probability at high-school level. All our majors take a course on proof-based reasoning (COS 240) but many don’t take it before junior year. Hence that course is not a prerequisite and our course doesn’t rely on formal proofs per se. A side benefit of assuming minimal math and programming prerequisites is that this also makes our course accessible to hundreds of non-majors, many of whom only take introductory CS courses.

Should provide a broad introduction to today’s AI and machine learning: Since AI and Machine Learning has been transformed by deep learning and related methods in the past decade, we wanted to provide a reasonable competence in deep learning as well as





reinforcement learning. Thus students who don’t take another AI/ML course still leave with a skill set appropriate for applying ML techniques in their careers. Of course, devoting the second half of our course to these “advanced” topics required leaving out several topics that are classically taught in introductory ML. Essentially, the classical topics are condensed into the first half of the term.

Provide a taste of today’s programming environments: Students can start with zero Python background, and slowly graduate to comfort in Python as well as dipping their toes into deep learning on Google CoLab and RL on OpenAI gym.

Taste of interesting applications + discussion of broader issues of societal interest: An introductory course needs to introduce students to these issues, given machine learning’s ubiquitous role in research across disciplines, as well as throughout our economy and our society. Sometimes we invite guest lecturers to provide such perspectives.

Finally, note that all the above material has to fit in Princeton’s 12-week term. We do not have 13-15 weeks as at other universities.

# Structure of the Notes

These notes are divided into six main parts.

# Part I

introduces supervised learning. Topics include linear regression, linear classification, and gradient descent.

# Part II

is about unsupervised learning. Topics include clustering, dimensionality reduction, n-gram language models, and matrix factorization.

# Part III

covers the basics of deep learning. Topics include feedforward neural networks and convolutional neural networks.

# Part IV

presents reinforcement learning. Topics include Markov decision process and Q-learning.

# Part V

introduces some advanced applications of machine learning. Topics include ethics of machine learning and deep learning for natural language processing.

# Part VI

provides some mathematical background that is useful for machine learning. Topics include probability, calculus, and linear algebra.




# Basic Ingredients of Machine Learning

Machine learning is the discipline of creating decision-making programs that improve themselves automatically, based on data or repeated experience. The entire setup consists of the following elements.

# Data

We have a set of data to learn from. The data may have an additional field called a label. If the data is labeled, the goal of the learning is to predict the label of newly seen data. Such learning is called supervised learning; examples include regression (Chapter 1, Chapter 5) or classification (Chapter 4). If the data is unlabeled, the goal of the learning is to extract the structure of the current data. Such learning is called unsupervised learning; examples include clustering (Chapter 6) or dimensionality reduction (Chapter 7).

# Model

We have a model that we want to learn. A model is a mapping from a data point to a desired answer or output. For supervised learning, the answer will be the label of the data; for unsupervised learning, the output will be the structure of the data.

# Model parameters

Each model is defined by a number of adjustable values, or internal parameters. The goal of the learning is to find the values of these parameters that yield the best model. Throughout the training process, the values of the parameters will be updated or reset.

# Model fitting

The process of finding the best (or good enough) values of model parameters, based on the provided data, is called fitting the model. There are many different ways to assess how “good” a model is. In many cases, a loss function is defined to measure how far the predictions of the model are from the actual output. In Chapter 4, we see how different loss functions are defined for different models.

# Testing

In many cases, especially in supervised learning, we want to predict how “good” the model will be for a newly seen data. This process is called testing the model. For this purpose, it is customary to use only a fraction (e.g., 80%) of the data to fit the model, and use the rest (e.g., 20%) to test the model. The portion of the data that is used for testing and not for fitting the model is called the held-out data and is assumed to be a good sample of the population data.


NO_CONTENT_HERE


# Part I

# Supervised Learning


NO_CONTENT_HERE



# Linear Regression: An Introduction

This chapter introduces least squares linear regression, one of the simplest and most popular models in data science. Several of you may have seen it in high school. In particular, we focus on understanding linear regression in the context of machine learning. Using linear regression as an example, we will introduce the terminologies and ideas (some of them were mentioned in the Preface) that are widely applicable to more complicated models in ML.

# 1.1 A Warm-up Example

Figure 1.1: A dataset of heights and weights of some male adults. The figure on the right shows the least squares regression line that fits the data.

Suppose we have a dataset of heights and weights of some male individuals randomly chosen from the population. We wish to determine a relationship between heights and weights. One simple relationship would be a linear relationship; namely:

w = a0 + a1h (1.1)

where w is the weight, h is the height, and a0, a1 are constant coefficients. We can think of this as a predictor that maps height h to a predicted weight a0 + a1h, and we want this value to be similar to the actual weight w. Obviously, a linear relationship won’t describe the data exactly but we hope it is a reasonable fit to data.1 In an ML setting, this relationship between h and w is called a model — a linear model to be more specific.

1 Similar linear models are used in many disciplines. For instance, the famous Philips model in economics suggests a linear relationship between inflation rate and unemployment rate, at least when the inflation rate is high.





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

Based on the values of a0 and a1, there are infinitely many different choices of this linear model. Therefore, it is natural that we want to find the values of a0, a1 that yield the “best” model. In an ML setting, finding these optimal values of a0, a1 is known as fitting the model. One can posit different criteria for defining “goodness” of the model.

Here we use classic least squares fit, invented by Gauss. Given a dataset {(h1, w1), (h2, w2), . . . , (hn, wn)} of n pairs of heights and weights, the “goodness” of the model in (1.1) is

1

n ∑ (wi − a0 − a1hi)2

i=1

Notice that wi − a0 − a1hi is the difference between the actual weight wi and the predicted weight a0 + a1hi. This difference is called the residual for the data point (hi, wi), and the full term in (1.2) is called the average squared residuals, or equivalently the mean squared error (MSE), of the dataset. The smaller the MSE, the closer the model’s predictions are to actual weights, and the more accurate the model is. Therefore, the “best” model according to the least squares method would be the one defined by the values of a0, a1 that minimize (1.2).

In an ML setting, a mathematical expression like (1.2) that captures the “badness” of the model is called a loss function. In general, we find the “best” model by minimizing the loss function.

# Example 1.1.1

If the data points (h, w) are given as {(65, 130), (58, 120), (73, 160)}, the least squares fit will find the values of a0, a1 that minimize

1

3

((130 − a0 − 65a1)2 + (120 − a0 − 58a1)2 + (160 − a0 − 73a1)2)

which are a0 = −510, a1 = 35.

# Problem 1.1.2

Between the two lines in Figure 1.2, which is more preferred by the least squares regression method?

# Problem 1.1.3

Using calculus, give an exact expression for a0, a1 that minimize (1.2). (Hint: (1.2) is quadratic in both a0 and a1. Fix the value of)





# linear regression: an introduction

# 1.1.1 Multivariate Linear Regression

One can generalize the above example to multi-variable settings. In general, we have k predictor variables and one effect variable. 3 The data points consist of k + 1 coordinates, where the last coordinate is the value y of the effect variable and the first k coordinates contain values of the predictor variables x1, x2, . . . , xk. Then the relationship we are trying to fit has the form

y = a0 + a1x1 + a2x2 + · · · + akxk (1.3)

and the least squares fit method will find the values of a0, a1, · · · , ak that minimize

1 n
∑(yi − a0 − a1xi − a2xi − · · · − akxi)² (1.4)
n i=1

where (x1, x2, . . . xk, yi) is the i-th data point.

We can simplify the notation by rewriting everything above in a vectorized notation. If we set x = (1, x1, x2, · · · , xk) 4 and a = (a0, a1, · · · , ak), then the relationship we are trying to fit has the form

y = a · x (1.5)

and the least squares fit method will find a ∈ Rk+1 that minimize

1 n
n ∑(yi − a · xi)² (1.6)
i=1

where (xi, yi) is the i-th data point. We discuss how to find the best values of a0, a1, . . . , ak later in Chapter 3; for now just assume that the solution can be found.

# 1.1.2 Testing a model (Held-out Data)

A crucial step in machine learning is to test the trained/fitted model on newly seen data, or held-out data, that was not used during training. If we were to test the model in the above example, we would hold out a portion of the data points (say 20%) — i.e., not use them during training — and check the average squared residual of the model on the held-out data points.

We can think of the average squared residual of the held-out data as an estimate of the average squared residual of the fitted model on the entire population of male adults. 5 The reason is that if the training data points were a random sample of the adult male population, be sure to report the RMSE error, which is the square root of the average square residual on held-out data. Also report the R2 value, which is closely related.





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

Then so is the set of held-out data points. This is quite analogous to opinion polls, where the opinions of a few thousand randomly sampled individuals can be a reasonable estimate for the average opinion across the US. The math for such sampling estimates is covered in Chapter 18.

# 1.1.3 More about Linear Regression

In the above example, we used the least squares method, which uses the average squared residual to assess the model. The least squares fit is very common but other notions of fit may also be used. For instance, instead of taking the sum of squares of residuals, one could consider the sum of absolute values, or expressions using logarithms, etc. We see some of these examples in Chapter 4.

It is important to note that the relationship learnt via regression — and machine learning in general — is (a) approximate and (b) only holds for the population that the data was drawn from. Therefore, we cannot use the relationship to predict the output of a data that is not from the same distribution. Additionally, if the distribution of the data is shifted, the relationship no longer holds. We will discuss more about this in depth in Chapter 2.

# 1.2 Using Linear Regression for Sentiment Prediction

# 1.2.1 Introduction

While you might have seen linear regression as early as in high school, you probably did not see this cool application. In sentiment classification, we are given a piece of text and have to label it with +1 if it expresses positive sentiment and −1 otherwise.

Example 1.2.1. Consider the following dataset, collected by showing snippets of text to humans and asking them to label them as positive (+1) or negative (−1):

| The film’s performances are thrilling. | +1 |
| -------------------------------------- | -- |
| It’s not a great monster movie.        | −1 |
| It is definitely worth seeing.         | +1 |
| Unflinchingly bleak and desperate.     | −1 |

Table 1.1: Data from Stanford Sentiment Treebank (SST). https://nlp.stanford.edu/sentiment/treebank.html

How can we train a model to label text snippets with the correct sentiment value, given a dataset of training examples? Here is an idea to try to solve it using a linear regression model. We first enumerate all English words and assign a real-valued score to each word, where the score for the i-th word is denoted by wi. These scores will




# linear regression: an introduction

be the parameters of the model. The output of the model, given a training example, is defined as ∑j∈S wj where S is the multiset of indices of words in the text.6 Then the least squares method needs to solve the following optimization problem for a dataset of (text, sentiment) pairs.

6 Unlike in a set, an element can appear multiple times in a multiset. For example, if the word good appears twice in a text, then S contains two copies of good.

minimize ∑i (yi − ∑j∈Si wj) (1.7)

where Si is the multiset of words in the i-th piece of text. Each of the values yi − ∑j∈S wj is called a least squares error or more generally the loss for that particular training example. The full summation is called a training loss of the dataset.

# Example 1.2.2.

Assume we are training a sentiment prediction model on a dataset. Table 1.1 shows some of the model parameter values. Then the output of the model on the sentence “I like this movie” from the training data will be 0.15 + 0.55 + 0.03 − 0.07 = 0.66. The output for “I dislike this movie” from the training data will be 0.15 − 0.74 + 0.03 − 0.07 = −0.63.

| i | word    | wi    |
| - | ------- | ----- |
| 1 | I       | 0.15  |
| 2 | like    | 0.55  |
| 3 | dislike | −0.74 |
| 4 | this    | 0.03  |
| 5 | movie   | −0.07 |
| 6 | a       | 0     |

We can also cast this in the standard formulation of linear regression as follows. The bag of words (BoW) representation of a piece of text is a vector in RN where N is the number of dictionary words. The i-th coordinate is the number of times the i-th word appears in the piece of text. This represents the text as a very long vector, one coordinate per one English word in the dictionary. The vector usually contains a lot of zeros, since most words probably do not appear in this piece of text. If we denote the BoW vector as x, the output of the model is seen to be ∑j wj = ∑i wi xi, which shows that the linear model we have proposed for sentiment prediction is just a subcase of linear regression (see (1.5)).

# Example 1.2.3.

Consider the same model in Example 1.2.2. The BoW representation for the sentence “I like this movie” is (1, 1, 0, 1, 1, 0 · · · ). The BoW representation for the sentence “I dislike this movie” is (1, 0, 1, 1, 1, 0 · · · ).


20  introduction to machine learning lecture notes for cos 324 at princeton university

# 1.2.2   Testing the Model

Here we use the model from Example 1.2.2 to illustrate the training and testing process of a model. Assume that the following four sentences were a part of the training dataset.

| I like this movie.    | +1 |
| --------------------- | -- |
| I dislike this movie. | −1 |
| I like this.          | +1 |
| I dislike this.       | −1 |

Table 1.3: A portion of the training data for a sentiment prediction model.

Assuming that the model parameters are the same as reported in Table 1.2, we can calculate the training loss of the sentence “I like this movie” as (+1 − 0.66)2 ≃ 0.12. Similarly, the squared residual for each of the four training sentences in Table 1.3 can be calculated as

| I like this movie.    | 0.12 |
| --------------------- | ---- |
| I dislike this movie. | 0.14 |
| I like this.          | 0.07 |
| I dislike this.       | 0.19 |

Table 1.4: The squared residual for four training examples.

Now it is time to test the model. Assume that the sentence “I like a movie” is provided to the model as a test data. The test loss can be calculated in a way similar to the training loss as (+1 − 0.63)2 ≃ 0.14. But to actually test if the model produces the correct sentiment label for this newly seen data, we now wish the model to output either +1 or −1, the only two labels that exist in the population. An easy fix is to change the output of the model at test time to be sign(∑j∈S wj). For this test data, the model will output sign(0.63) = +1.

On the Stanford Sentiment Treebank, this approach of training a least squares model yields a success rate of 78%7. By contrast, the state-of-the-art deep learning methods yield success rates exceeding 96%!7

To be more exact, this result is from a model called ridge regression model, which is linear regression model augmented by an ℓ2 regularizer, which will be explained in Chapter 3.

One thing to note is that while the training loss is calculated and explicitly used in the training process, the test loss is only a statistic that is generated after the training is over. It is a metric to assess if the model fitted on the training data also performs well for a more general data.

# 1.2.3   Test Loss, Generalization, and Test accuracy

As mentioned already, the goal of training a model is that it should make good predictions on new, previously-unseen data. Most models will exhibit a low training loss, but not all of them show a low test loss. This observation motivates the following definition:

Generalization Error = |training loss − test loss|





# linear regression: an introduction

# 21

A trained model is said to generalize well if the generalization error is small. In our case, the loss is the average squared residual. Thus good generalization means that the average squared residual on test data points is similar to that on the training data. Let us see what happens on our sentiment model when it is fitted and tested on the SST dataset.

| Train MSE         | 0.0727 |
| ----------------- | ------ |
| Test MSE          | 0.7523 |
| Training accuracy | 99.55% |
| Test accuracy     | 78.09% |

Table 1.5: Accuracy refers to the classification accuracy when we make the model to output only ±1 labels.

# Example 1.2.4.

The generalization error above is the difference between MSE on test points and the MSE on training points, namely 0.75 − 0.07 = 0.68.

Let’s try to understand the relationship between low test loss (the squared residual) and high test accuracy (for what fraction of test data points the sentiment was correct). Heuristically, the test loss (average squared residual) being 0.75 means that the absolute value of the residual on a typical point is √0.75 ≈ 0.87. This means that for a data point with an actual positive sentiment (i.e., label +1), the output of the model is roughly expected to lie in the interval [1 − 0.87, 1 + 0.87], and similarly, for a data point with an actual negative sentiment (i.e., label −1), the output of the model is roughly expected to lie in the interval [−1 − 0.87, −1 + 0.87]. Once we take the sign sign(∑j∈S wj) of the output of the model, the output is thus likely to be rounded off to the correct label. We also note that the training accuracy is almost 100%. This usually happens in settings where the number of parameters (i.e., number of predictor variables) exceeds the number of training data points (or is close to it). The following problem explores this.

# Problem 1.2.5.

An expert on TV claims to have a formula to predict the outcome of presidential elections. It uses 31 measurements of various economic and societal quantities (inflation, divorce rate, etc). The formula correctly predicts the winner of all elections 1928-2020. Should you believe the formula’s prediction for the 2024 election? (Hint: Under fairly general conditions, T + 1 completely nonsense variables — i.e., having nothing to do with presidential politics — can be used to perfectly fit (via linear regression) the outcomes for T past presidential elections. 8)

8 If a model does not generalize well, then it is said to overfit the training data.

# 1.2.4  Interpreting the Model

In many settings (e.g., medicine), an important purpose of regression modeling is to understand the data or the phenomenon a bit





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

Better. In this case, the phenomenon is “sentiment” and we are naturally curious about what positive or negative sentiment amounts to. Specifically, what caused the model’s output to be +1 or −1 given a specific sentence?

Figure 1.3: A histogram of the learned parameters wi of a sentiment prediction model trained on the Stanford Sentiment Treebank.

Positive values of wi imply that the words carry a positive sentiment, while negative values of wi imply that the words carry a negative sentiment. Also, the greater the absolute value of wi is, the stronger the sentiment. Notice that most words have a value of wi close to zero, meaning the model views most words as neutral. The model “pays attention” to only a tiny set of words.

Words with high positive wi values (i.e., positive words) include enjoyable, fun, and remarkable. Words with high negative values (i.e., negative words) include suffers, dull, and worst. Words with wi values close to 0 (i.e., neutral words) include duty and desire.

# 1.3 Importance of Featurization

In the sentiment model, we chose a particular method to represent a piece of text with a vector. The coordinates of this vector are often referred to as features and this process of converting data into vectors is called featurization. One can conceive of other choices for featurizing text. For example, bigram featurization consists of the following: the coordinates of the vector correspond to pairs of words and the coordinate contains the number of times this pair of words appeared consecutively in the piece of text. In contrast, the choice of featurization from the earlier example matches each coordinate with a single word, and is called a unigram featurization.

| Score | Frequency |
| ----- | --------- |
| -0.4  | 1500      |
| -0.2  | 1750      |
| 0.0   | 1000      |
| 0.2   | 750       |
| 0.4   | 500       |

Page 22




# linear regression: an introduction

Bigram features allow the model to access information about phrases that were present in the text. For instance, in isolation “pretty” is a positive word and “bad” is a negative word. If they both occur in text one would imagine that they cancel each other out as far as overall sentiment is concerned. But the phrase “pretty bad” is more negative than “bad.” Thus bigram features can improve the model’s ability to capture sentiment.

The required number of dimension for bigram representations can get rather large. If the number of words is N, then the number of coordinates is N2. Realize that the number of model parameters in linear regressions is the same as the number of coordinates. Thus if N is 30,000 then the number of coordinates in bigram feature vector (and hence the number of model parameters) is close to a billion, which is a rather large number. In practice one might throw away information for all pairs except say the 10,000 most common ones in the dataset. Usually models that incorporate bigram features do better than unigram-only models.

If one is trying to do studies of medical treatment with regression, there can be many potential featurizations of patient data. Doctors’ annotations, test results, X-ray scans, etc. all have to be converted somehow into real-valued features, and the experimenter uses their prior knowledge and intuitions while featurizing the data.

# Example 1.3.1

Patients’ raw data might include height and weight. If we use linear regression, the effect variable can only depend upon a linear combination of height and weight. But it is known that several health outcomes are better modeled using Body Mass Index, defined as weight/(height2). Thus the experimenter may include a separate coordinate for BMI, even though it duplicates information already present in the other coordinates.

# Example 1.3.2

In Example 1.3.1, the weight in pounds may span a range of [90, 350], whereas cholesterol ratio may span a range of [1, 10]. It is often a good idea to normalize the coordinate, which means to replace x with (x − μ)/σ where μ is the mean of the coordinate values in the dataset and σ is the standard deviation.

Thus the same raw dataset can have multiple featurizations, with different number of coordinates. Problem 1.2.5 may make us wary of using featurizations with too many coordinates. We will learn a technique called regularization in Chapter 3, which helps mitigate the issue identified in Problem 1.2.5.


24    introduction to machine learning lecture notes for cos 324 at princeton university

# 1.4    Linear Regression in Python Programming

In this section, we briefly discuss how to write the Python code to perform linear regression (e.g., sentiment prediction). Python is often the language of choice for many machine learning applications due to its relative ease of use and the large variety of external packages available to automate the process. Here, we introduce a few of these packages:

- numpy: This package is ubiquitous throughout the machine learning community. It provides access to specialized array data structures which are implemented in highly optimized C code. Linear algebra computations and array restructuring operations are significantly faster with numpy compared to using Python directly. 9 Documentation is available at https://numpy.org/
- matplotlib: This package enables Python programmers to create high quality plots and graphs. Visualizations are highly configurable and interoperable with several other Python packages. 10 Documentation is available at https://matplotlib.org/
- sklearn: This package provides a potpourri of machine learning and data science models through an easy to use object-oriented API. In addition to linear regression, sklearn makes it possible to implement SVMs, clustering, neural networks, and much more; you will learn about some of these models later in the course. 11 Documentation is available at https://scikit-learn.org/stable/index.html

Throughout this course, you will be asked to make use of functions defined in some of these external packages. You may not always be familiar with the usage of these functions. It is important to check the official documentation to learn about the usage and the signature of the functions.

The code snippet below uses the three aforementioned packages to perform linear regression on any given dataset.

# import necessary packages
import numpy as np
from  sklearn.model_selection import train_test_split
from  sklearn.linear_model import LinearRegression
from  sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

# prepare train, test data
X = ...
y = ...
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# perform linear regression on train data
linreg = LinearRegression().fit(X_train, y_train)
pred_train = linreg.predict(X_train)
pred_test = linreg.predict(X_test)

# print train results
print(’Train MSE: ’, ’{0:.4f}’.format(mse(y_train, pred_train)))





# linear regression: an introduction

print(’Test MSE: ’, ’{0:.4f}’.format(mse(y_test, pred_test)))
print(’Train Acc: ’, ’{0:.2f}’.format(100*(np.sign(pred_train)==y_train).
mean()))
print(’Test Acc: ’, ’{0:.2f}’.format(100*(np.sign(pred_test)==y_test).mean()
))

# plot gold (actual) vs predicted value
plt.scatter(y_test, pred_test, c="red")
plt.xlabel("actual y value (y)")
plt.ylabel("predicted y value (y hat)")
plt.title("y vs y hat")

For readers who are not familiar with Python, we discuss some key details. In the first section of the code, we import the relevant packages

import numpy as np
from  sklearn.model_selection import train_test_split
from  sklearn.linear_model import LinearRegression
from  sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

As seen in this example, there are two ways to load a package. The first option is to import the full package with the import keyword

import numpy as np

Notice that the we can assign the imported package a customized name with the as keyword. In this case, we decided refer to the package numpy with the name np throughout the rest of the code. This is indeed the case when we call np.sign()

Here we refer to the method sign() of the numpy package with the customized name np. Alternatively, we can selectively import particular methods or classes with the from keyword

from  sklearn.model_selection import train_test_split

The next part of the code is preparing the train, test data.

X = ...
y = ...
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X will have to be an array of arrays, and y will have to be an array of values, with the same length as X. These arrays can be defined directly by specifying each of their entries, or they could be read from some external data (most commonly a csv file). Here, we present an example dataset where x ∈ R2:

X = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
y = [1, 1, 1, -1, -1]

Then we call the train_test_split() method to split the dataset into data for model training and testing. Alternatively, we can split the




# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

Dataset by manually slicing the data arrays.

In general, slicing a Python array involves the : operator along with start and end indices. For instance, consider an arbitrary array a. Then, the output of a[i:j] will be a subarray of a from the index i (inclusive) to the index j (exclusive). In the following code sample, we slice the data by specifying the number of training data points:

train_size = ...
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

Note that we have omitted some of the bounding indices. If the start index is omitted, Python assumes it to be 0 (so that the subarray is from the start of the array); for example, X[:train_size] is the first train_size entries of X. If the end index is omitted, Python assumes it to be n, the length of the array (so that the subarray ends at the end of the array); for instance, X[train_size:] is the remaining entries of X, once we remove the first train_size entries. Another way to slice the arrays is by specifying the number of test data points:

test_size = ...
X_train = X[:-test_size]
X_test = X[-test_size:]
y_train = y[:-test_size]
y_test = y[-test_size:]

Here, notice that the index -test_size is a negative number. In this case, Python interprets this as n - test_size, where n is the size of the array. In other words, it is the index of the test_size-th element from the back of the array.

The third part of the code is fitting the linear regression model:

linreg = LinearRegression().fit(X_train, y_train)
pred_train = linreg.predict(X_train)
pred_test = linreg.predict(X_test)

The first line will generate the least squares fit model based on the train data. Then we can have the model make predictions on the train and test data.

Next, we print out the mean squared loss and the accuracy for the train and test data:

print('Train MSE: ', '{0:.4f}'.format(mse(y_train, pred_train)))
print('Test MSE: ', '{0:.4f}'.format(mse(y_test, pred_test)))
print('Train Acc: ', '{0:.2f}'.format(100*(np.sign(pred_train)==y_train).mean()))
print('Test Acc: ', '{0:.2f}'.format(100*(np.sign(pred_test)==y_test).mean()))

Notice that we use the mse() method that we imported from the sklearn package. Also notice that when computing the accuracy, we...


# linear regression: an introduction

changed the output of the model to be the sign of the predicted values, so that we can compare them with the gold values. In many cases, there are packages that perform these elementary operations for machine learning.

Finally, we plot the actual and predicted values using the matplotlib package.

plt.scatter(y_test, pred_test, c="red")
plt.xlabel("actual y value (y)")
plt.ylabel("predicted y value (y hat)")
plt.title("y vs y hat")
The first line draws a scatter plot with the y_test in the x-axis and pred_test in the y-axis. Notice that you can specify the color of the data points by specifying the value of the parameter c. In general, parameters are optional values you can provide to Python functions. If the values to parameters are omitted, the functions will use their default values. The second and third lines specify the labels that will be written next to the axes. The final line specifies the title of the plot.


NO_CONTENT_HERE



# Statistical Learning: What It Means to Learn

Students often get confused about the meaning and significance of a relationship learnt via fitting a model to data. Some of them think such relationships are analogous to, say, a law of nature like F = ma, which applies every time force is applied to a mass anywhere in the universe. The main goal of this chapter is to explain the statistical nature of machine learning — models are fitted on a particular distribution of data points, and its predictions are valid only for data points from the same distribution. (See Chapter 18.)

# 2.1 A Warm-up Example

We work through a concrete example1 before enunciating the general properties of statistical learning. Suppose we are studying the relationship between the following quantities for the population of Princeton: height (H), number of exercise hours per week (E), amount of calories consumed per week (C), and weight (W). After collecting information from 200 randomly sampled residents, and using a 80/20 train/test split, we perform a linear regression on the training dataset to come up with the following relationship:

W = 50 + H + 0.1C − 4E (2.1)

Let’s also say that the average squared residual on train and test data were both 100. This means that the relationship (2.1) holds with an error of 10 lbs on a typical test data point.2

Question 2.1.1. Alice was one of the Princeton residents in the study, but the prediction of the model is very off from her actual value (squared residual is 300). Does this prove the model wrong?

The answer is no. The least squares linear regression finds the model that minimizes the average squared residual across all training data points. The residual could be large for a particular individual.




# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

# Question 2.1.2

There was a follow-up research for every Princeton resident who is taller than 7 feet. All of them reported squared residual of 500. Does this prove the model wrong?

The answer is still no. People who are taller than 7 feet make up a tiny fraction of the entire population. Their residuals have a very small effect on the average squared residual. The residual could be large for a small subset of the population.

# Question 2.1.3

There was a follow-up survey that tested the model on every single Princeton resident. Is it possible that the average squared residual is 200 for the entire population?

The answer is yes, although it is unlikely. Consider the distribution of 4-tuples (H, E, C, W) over the entire Princeton population. This is some distribution over a finite set of 4-dimensional vectors. 3 The 200 3 31,000 vectors to be more exact. The population of Princeton is 31,000.

Out of these 200 data points, 40 were randomly chosen to be held-out as test data, while the remaining 160 were used as training data. We can also say that these 40 data points were chosen at random from the distribution over the entire population of Princeton. Thus when we test the model in (2.1) on held-out data, we’re testing this relationship over a random sample of 40 data points drawn from the population. 40 is a large enough number to give us some confidence that the average squared residual of the test data is a good estimate of the squared residual in the population, but just as polling errors happen during elections, there is some chance that this estimate is off. In this case, we would say that the 40 test samples were unrepresentative of the full population.

It is important to remember that the training and test data are sampled from the same distribution as the population. Therefore, the average squared residual of the training and test data are only a good estimate of the squared residual of the distribution they were sampled from. This also means that the relationship found from the training data only holds (with small residue) for that particular distribution. If the population is different, or if the distribution shifts within the same population, the relationship is not guaranteed to hold. For example, the relationship in (2.1) is not expected to hold for people from Timbuktu, Mali (a different population), or for residents of Princeton who are taller than 7 feet (a tiny subpopulation that is likely unrepresentative of the population).

# Question 2.1.4

It becomes fashionable in Princeton to try to gain weight. Based on the relationship in (2.1), everyone decides to increase their value of C and reduce their value of E. Does the model predict that many of them will gain weight?



# Statistical Learning: What It Means to Learn

The answer is no. The model was fitted to and tested on the distribution obtained before everyone tried to gain weight. It has not been fitted on the distribution of data points from people who changed their values of C and E. In particular, note that if everyone reduces their E and increases their C, then the distribution has definitely changed — the average value of the E coordinate in this distribution has decreased, whereas the average value of the C coordinate has increased.

In general, a relationship learned from a fitted model illustrates correlation and need not imply causation. The values of H, C, E in (2.1) do not cause W to take a specific value. The equation only shows that the values are connected via this linear relationship on average (with some bounded square residuals).

# 2.2 Summary of Statistical Learning

The above discussion leads us to summarize properties of Statistical Learning. Note that these apply to most methods of machine learning, not just linear regression.

- Training/test data points are sampled from some distribution D: In the above example, 200 residents were randomly sampled from the entire population of Princeton residents.
- The learnt relationship holds only for the distribution D that the data was sampled from. The performance of the model on test data is an estimate of the performance of the model on the full distribution D.
- There is a small probability that the estimate using test data is off. This is analogous to polling errors in opinion polls. The computation of “confidence bounds” is discussed in Chapter 18.

# 2.3 Implications for Applications of Machine Learning

The above framework and its limitations have real-life implications.

1. Results of medical studies may not apply to minority populations. This can happen if the minority population is genetically distinct and constitutes only a small fraction of the population. Then test error could be large on the minority population even if it is small on average. In fact, there have been classic studies about heart disease in the 1960s whose conclusions and recommendations fail to apply well to even a group that is half of the population: females! In those days heart disease was thought to largely strike males (which subsequently turned out to be quite false) and so




# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

The studies were done primarily on males. It turns out that heart diseases in female patients behave differently. Many practices that came out of those studies turned out to be harmful to female patients.4

See https://www.theatlantic.com/health/archive/2015/10/heart-disease-women/412495/.

Classifiers released by tech companies in the recent past were found to have high error rates on certain minority populations. It was quickly recognized that relying on test error alone can lead to adverse outcomes on subpopulations.5

See https://time.com/5520558/artificial-intelligence-racial-gender-bias/.

Creating interactive agents is difficult. In an interactive setting (e.g., an online game), a decision-making program is often called an agent. When an agent has to enter an extended number of interactions6 with a human (or another agent designed by a different group of researchers, as happens in Robocup soccer7), then statistical learning requires that the agent to have been exposed to similar situations/interactions during training (i.e., from a fixed distribution). It is quite unclear if this is true.

Later in the book we encounter Reinforcement Learning, which deals with such settings.

See https://www.robocup.org/.



# 3 Optimization via Gradient Descent

This chapter discusses how to train model parameters through optimization techniques that help find the best (or fairly good) model that has low training loss. We assume that you have seen simple root-finding techniques in high school or in calculus. Optimization in machine learning often uses a procedure called gradient descent. This chapter assumes your knowledge of basic multivariable calculus. If you have not taken a course in multivariable calculus, read Chapter 19 to familiarize yourself with the basic definitions.

# 3.1 Gradient Descent

In general, an ML model has an associated loss function. The “best” model is the one that minimizes the training loss. In most cases, it is impossible or difficult to find the minimum analytically; instead, we use a numerical method called the gradient descent algorithm to find the (approximate) optimum.

# 3.1.1 Univariate Example

Let’s start with an univariate example to motivate the topic. Let f (w) = 4w2 − 6w − 9 be a quadratic function. Figure 3.1 shows the graph of this function.

Figure 3.1: The graph of f (w) = 4w2 − 6w − 9

-4              -2    4

-10

Let’s say that f attains its minimum at some point w = w*. How




34   introduction to machine learning lecture notes for cos 324 at princeton university

should we find the value of w∗? Here is an idea. Let’s start from some random point on the curve and “walk down” the curve. Notice from the graph that f ′(w∗) = 0. Also, f is decreasing (i.e., f ′(w) &#x3C; 0) when w &#x3C; w∗ and increasing (i.e., f ′(w) > 0) when w > w∗. So if we examine a point w and find that f ′(w) = 0, then we have arrived at our minimum. If f ′(w) > 0, then we are currently on the right side of the minimum, so we need to decrease w. On the other hand, if f ′(w) &#x3C; 0, then we need to increase w.

For example, we start with the point w = 0. Since f ′(w) = −6 &#x3C; 0, we know that we are on the left side of the minimum, so we update w ← 1. Since f ′(w) = 2 > 0, we are now on the right side of the minimum, so we update w ← 1. When we iterate this process, we hope that we eventually slide down to the bottom of the curve. Observe that the change of the value of w has the opposite sign from f ′(w) at that point. That is, for each step of this iteration, we can always find a η > 0 which decreases the value of w when w ← w − η f ′(w). This is not a mere coincidence — a similar result holds for a multivariate function.

# 3.1.2    Gradient Descent (GD)

Let f : Rd → R be a multivariate function. If we want to “walk down” the curve of f as in the univariate case, we need to find a direction from the current point w that decreases f. A generalization of the Taylor expansion in the multivariable setting shows that the value of f in a small neighborhood around x = (x1, x2, . . . , xd) can be approximated as a linear function in terms of the gradient.

f (w + h) ≈ f (w) + ∇ f (w) · h

where h ∈ Rd is small enough (i.e., h ≈ 0). If ∇ f is nonzero and we choose h = −η∇ f where η is a sufficiently small positive number, then

f (w − η∇ f ) ≈ f (w) − η∥∇ f ∥²

Since ∥∇ f ∥² is positive, being the squared length of the vector ∇ f, we conclude that the update w ← w − η∇ f causes a decrease in value of f. This discussion motivates the gradient descent algorithm, which iteratively decreases the value of f until ∇ f ≈ 0.

In fact, the gradient ∇ f is known as the direction of steepest increase of f. Hence, the opposite direction −∇f is the direction of steepest decrease of f.

# Definition 3.1.1 (Gradient Descent).

Gradient descent is an iterative algorithm that updates the weight vector w with the following rule:

w ← w − η∇ f (w) (3.1)





# 3.1.2 Optimization via Gradient Descent

where η > 0 is a sufficiently small positive constant, called the learning rate or step size.

We illustrate with an example.

# Example 3.1.2

Let f (w1, w2) = (w2 + w2)4 − 7(w2 + w2)3 + 13(w2 + w1)2. From Figure 3.2, we see that it attains a global minimum at (0, 0).

The partial derivatives of f can be calculated as:

∂f  = 2w1(w2 + w2)(4(w2 + w2)2 − 21(w2 + w2) + 26)

∂w1

∂f  = 2w2(w2 + w2)(4(w2 + w2)2 − 21(w2 + w2) + 26)

∂w2

Now imagine initiating the gradient descent algorithm from the point (0.5, 1) where the gradient vector is (7.5, 15). One iteration of gradient descent with η = 0.01 would move from (0.5, 1) to (0.425, 0.85). The gradient vector at (0.425, 0.85) is (7.90, 15.81) and the next iteration of GD will move the point from (0.425, 0.85) to (0.35, 0.69). After 200 iterations, the algorithm moves the point to (0.03, 0.06), which is very close to the global minimum of f.

Figure 3.2: The graph of f (w1, w2) = (w2 + w2)4 − 7(w2 + w2)3 + 13(w2 + w1)2. The function attains a global minimum at (0, 0).

20      1.5

1.0

0.5

0.0  w1

0.5

1.0

w2   0   -1.5

# 3.1.3 Learning Rate (LR)

Choosing an appropriate learning rate is crucial for GD. Figure 3.3 shows the result of two iterations of gradient descent with a different learning rate. On the left, we see the result when η is too small. The change of w is too small, and the loss function converges to the minimum very slowly. On the right, we see the result when η is too big. The change of w is too large, and the algorithm “shoots past” the minimum. If η is even larger, the algorithm may even fail to converge to the minimum.





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

f(w)    f(w)     Figure 3.3: Two iterations of gradient descent with different learning rates.

w         w    w  w

Too small: converge    Too big: overshoot and

very slowly              even diverge

The natural question to ask is: what is the appropriate learning rate? There is some theory, and the best setting is known in some cases. But in general, it has to be set by trial and error, especially for non-convex loss functions. For instance, we start with some learning rate, say 0.5 and decrease η by 1 if we do not observe a steady decrease in the training loss. Such heuristics are called training schedules and they are derived via trial and error on that particular dataset and model. 2

Constants whose values are decided by trial and error based on dataset and model are called hyperparameters. Modern ML models have several hyperparameters. Often optimization packages will suggest a default value and a fine-tuning method.

# 3.1.4 Non-convex Functions

For convex functions that are “bowl shaped,” gradient descent with a small enough learning rate provably converges to the minimum solution. But for non-convex functions, the best we can hope for is converging to a point where ∇ f = 0.3 Finding the global minimum of a non-convex function is NP-hard in the worst case. In practice, loss functions are non-convex and have multiple local minima. Then, the gradient descent algorithm may converge to a different local minimum based on the initialization of the parameter vector w.

There is advanced theory on how to escape saddle points, which will not be covered in this course.

0.51.0                          yo 2                                                 Figure 3.4: An example of a convex and a non-convex function in two variables. For non-convex functions, GD will reach a stationary point, where the gradient is zero. Figure from https://www.kdnuggets.com/2016/12/hard-thing-about-deep-learning.html.

Computed by Wolfram(Alpha                   Computed by WolframjAlpha

# Example 3.1.3

Consider the function f(w) = 1 w4 − 1 w3 − w2 + w, which has two local minima at (−1, −1) and (2, −1)3.

As seen in Figure 3.5, the





# 3.2 Implications of the Linearity of a Gradient

The fact that gradient is a linear operator (i.e., ∇( f1 + f2) = ∇ f1 + ∇ f2) has great practical importance in machine learning. Just like in (1.4), the training loss of a machine learning model is usually defined as the average (or the sum) of the loss on individual training data points. By the linearity of gradient, the gradient of the entire loss can be found by taking the sum of the gradient of the loss on individual data points.

# 3.2.1 Stochastic Gradient Descent

Since computing the gradient of the loss involves some computation on each of the data points, the computation can be quite slow for today’s large data sets, which can contain millions of data points. A simple workaround is to estimate the gradient at each step by randomly sampling k data points and averaging the corresponding loss gradients. This is very analogous to opinion polls, which can also be seen as sampling from a distribution on vectors and using the average of the sample as a substitute for the population average. This algorithm is called Stochastic Gradient Descent (SGD). 4 This technique works for two reasons: (1) all training data points are assumed to be sampled from the same distribution; (2) the overall training loss is just the sum/average of loss for individual data points.

# 3.2.2 Mini-batch Stochastic Gradient Descent

Today, large scale machine learning is done using special-purpose processors called Graphical Processing Units (GPUs). 5 These highly efficient processors were originally developed for computer graphics operations, which were most oftenly used in computer games. Around 2012, deep learning experts realized their usefulness for deep learning. At that time, writing code for GPUs was extremely difficult, but today’s environments have made this much easier.

| Initial Point A |   | Initial Point B |
| --------------- | - | --------------- |
| -2              | 2 | -1              |
| 1               | 3 |                 |
| Local Minimum A |   | Local Minimum B |





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

Specialized architectures have the ability to perform fast parallel computations. To exploit these special capabilities, a special variant of SGD — Mini-batch SGD — can be used. Here the dataset is randomly partitioned into mini batches whose size is dictated by the degree of parallelism available in the GPU, usually a power of 2, such as 256. The members of each batch are loaded onto a different processor. Together the processors compute the gradient for one mini-batch in one go, add up the gradients to perform a single iteration for the gradient descent. Then they move on to the next batch, perform another update step, and so on.

# 3.2.3 Federated Learning

This is a conceptual framework for training an ML model on data belonging to different parties, some of whom do not wish to hand the data over to a central server. Consider the following two examples:

1. Hospitals who wish to train an ML model on their pooled data, but who are forbidden by privacy laws to hand the data to other organizations.
2. Owners of Internet of Things (IoT) devices, who wish to benefit from training on their data but do not wish to submit the data.

In Federated Learning, the model is trained at a central server, whereas data remains with the data owners, who actively participate in the training. Users retrieve the current model parameters from the server and calculate the gradients locally. They send only the gradients, but not the data, to the server, and the overall gradient is calculated at the server as the weighted sum (or average) of the user gradients.

# 3.3 Regularizers

This section describes regularization, a useful idea that often improves generalization of the model. The main idea is that instead of minimizing the training loss function ℓ(w), we minimize the function

ℓ(w) + λR(w)


                                                                optimization via gradient descent              39

Example 3.3.1. Recall the sentiment prediction model using least squares
loss. Suppose the training data consists of two data points: (x1, y1) =
((1, 0, 1), −1) and (x2, y2) = ((1, 1, 0), +1). Then the least squares loss,
without any regularizer, can be written as
      1 ((−1 − (w0 + w2))2 + (1 − (w0 + w1))2)                         (3.3)
      2
A little thought suggests that the minimum value of this loss is 0 provided
there exists (w0, w1, w2) such that

      (−1 − (w0 + w2))2 = 0 = (1 − (w0 + w1))2.

You can verify that infinitely many solutions exist: all w∗ = (w0, w1, w2)
that lie on the line (0, 1, −1) + t(1, −1, −1) where t ∈ R. In other words,
the loss has infinitely many minimizers.
 Now if impose an ℓ2 regularizer, the loss becomes
 1 ((−1 − (w0 + w2))2 + (1 − (w0 + w1))2) + λ(w2 + w2 + w2)            (3.4)
 2                                                0         1   2
Any minimizer of this loss must make the gradient zero. In other words, the
minimizer will satisfy the following system of linear equations:

       (2 + 2λ)w0 + w1 + w2 = 0
       w0 + (1 + 2λ)w1 = 1
       w0 + (1 + 2λ)w2 = −1

You can verify that w∗∗ =  0,  1   , −    1  is the unique minimizer for
any λ > 0. For a sufficiently  1+2λ     1+2λ                     ∗∗
                               small value of λ, the corresponding w   is
close enough to the line (0, 1, −1) + t(1, −1, −1). That is, it has a non-zero
training loss, but the value is very close to zero. Combined with the fact it
has a relatively small norm, w∗∗ becomes the minimizer for the regularized
loss.

                                                                              Figure 3.6: The graph of the line
                                                                              (0, 1, −1) + t(1, −1, −1) and the point
                                                                              w∗∗ = (0,  1  , −  1  ) when λ = 0.01
                                                                                         1+2λ    1+2λ

1.2

Note that if w∗ is the minimizer of ℓ(w) and w∗∗ the minimizer
of the regularized loss, then by definition of a minimizer, it always

0.8

0.6

0.4

0.2    0.8
       0.6

-1.2    -0.8 -0.6  -0.4  -0.20.2  0 2 0.4

-0.6 -0.4  0  0.2 0.4

-1.2  -0.8    -0.2    0.6 0.8  1.2 1.4 1.6
              -0.4                        1.8  2 2.2 2.4 2.6

-2.4-2.2 2     -0.6    2.8
-2.6           -0.8
-2.8


40  introduction to machine learning lecture notes for cos 324 at princeton university

holds that ℓ(w∗) ≤ ℓ(w∗∗). In general, regularization ends up leading to training models with a higher value of ℓ(w). This is considered acceptable because the models often generalize better. In other words, a slightly higher training loss is considered a price worth paying for a significantly lower test loss. This is illustrated by the example of sentiment prediction from Chapter 1. As hinted there, the results shown used a model trained with an ℓ2 regularizer. The dataset involves 15k distinct words, so that is the number of model variables. There are 8k data points. Recall from Problem 1.2.5 that in such settings, there usually will exist a linear model that perfectly fits the data points. Indeed, we see in Table 3.1 that this is the case when we don’t use a regularizer. However, using a regularizer prevents the model from perfectly fitting the training data. But the test loss drops tenfold with regularization.

|                   | No regularizer | With ℓ2-regularizer |
| ----------------- | -------------- | ------------------- |
| Train MSE         | 0.0000         | 0.0727              |
| Test MSE          | 7.9469         | 0.7523              |
| Training accuracy | 100.00%        | 99.55%              |
| Test accuracy     | 61.67%         | 78.07%              |

Table 3.1: Training sentiment model on the SST with and without ℓ2 regularizer.

# 3.3.1  Effects of Regularization

Here we briefly list some benefits of regularization.

1. Regularizers often help improve generalization. Above we saw a concrete example with the sentiment prediction model.
2. Adding a scalar multiple of ∥w∥2 to a function can speed up optimization by slightly reshaping the optimization landscape. The mathematical treatment of this is beyond the scope of this course.
3. Without a regularizer term, models such as logistic regression and soft-margin SVMs begin to lose their power. This will be explained when we discuss these models in Chapter 4.

# 3.3.2  Why Does Regularization Help?

The simplest answer is that we do not fully understand this concept yet. In this section, we present some intuitions derived from simple models, but keep in mind that these ideas might be misleading in more complicated models. The usual explanation given is that the norm of the parameter vector controls the expressiveness or complexity of the model. Here “complexity” is being used in the sense of “complicatedness”. By





# optimization via gradient descent

trying to minimize loss as well as the norm of the parameter vector, the learned model tends to stay simple. 6 Whereas this discussion can be made fairly rigorous for linear models, it does not seem to apply to more complicated models; for instance, regularization often helps a lot in deep learning, but rigorous explanations appear to be at best incomplete and at worst incorrect there. 7

Another explanation 8 is that a regularizer serves as a penalty for large weights and forces the model to choose smaller absolute values of parameters. According to this explanation, adding regularizers to a model penalizes higher-order terms or unnecessary variables and is able to avoid overfitting. Indeed, Figure 3.7 shows that the weights of the parameters in our sentiment model are significantly smaller when trained with a regularizer. But one lingering question with this explanation is: How come attaching the same penalty to all variables forces the model to identify variables that are needed, and those that are not? What causes this disparate treatment of the variables?

1600                            5000                                           Figure 3.7: The histogram of weights of the parameters in the sentiment prediction model with (right) or without (left) an ℓ2 regularizer.

Now consider this explanation — ℓ2 regularization introduces a new dynamic to gradient descent, whereby gradient updates have to constantly battle against a rescaling that is always trying to whittle all variables down to zero. The effort succeeds only for variables where gradient updates are pushing hardest to make them nonzero. Therefore, the weights for “necessary” variables survive, while “unnecessary” variables are thrown away. To say this more precisely, consider the regularized loss ℓ(w) + λ ∥w∥2 whose gradient is

∇ℓ + 2λw2

Thus the update rule in gradient descent can be written as

wt+1 ← wt − η(∇ℓ + 2λwt)

where wt denotes the weight vector at the t-th time step. This update rule can be rewritten as

wt+1 ← wt(1 − 2ηλ) − η∇ℓ                                      (3.5)

The first term is down-scaling: if for example η = λ = 0.1, this amounts to multiplying the current vector by 0.98, and this of course will make w very small in a few hundred iterations.




42  introduction to machine learning lecture notes for cos 324 at princeton university

The second term is the gradient update to w. It can counteract the down-scaling by making the variables larger. But notice that the amount of change is based on how much each of the coordinates contribute to reducing the loss. Variables that are not useful will tend not to get increased by the gradient update and thus will keep getting down-scaled to low values.9 The choice of λ mediates between these two processes.9 It is one of those “use it or lose it” situations!

# 3.4 Gradient Descent in Python Programming

In this section, we briefly discuss how to implement the Gradient Descent algorithm in Python. It is customary to use the numpy package to speed up computation and the matplotlib package for visualization.

# import necessary packages
import  numpy as np
import  matplotlib.pyplot as plt
from matplotlib import cm # colormap

# initialize variables
num_iter = ...
x = np.zeros(num_iter + 1)
y = np.zeros(num_iter + 1)
x[0], y[0], eta = ...

# define functions to calculate f and grad_f
def  f(x, y):
...
return f

def  grad_f(x, y):
...
return grad_f

# run Gradient Descent
for  i in range(num_iter):
grad_x, grad_y = grad_f(x[i], y[i])
x[i + 1] = x[i] - eta * grad_x
y[i + 1] = y[i] - eta * grad_y

# plot the surface
xmin, xmax, ymin, ymax, n = ...
X, Y = np.meshgrid(np.linspace(xmin, xmax, n),
np.linspace(ymin, ymax, n))
Z = f(X, Y)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(projection=’3d’)
ax.set_xlabel(’X’)
ax.set_ylabel(’Y’)
ax.set_zlabel(’Z’)
ax.view_init(elev=ax.elev, azim=ax.azim)
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.5)

# plot the trajectory of Gradient Descent
ax.plot(x, y, f(x, y), color=’orange’, markerfacecolor=’black’,
markeredgecolor=’k’, marker=’o’, markersize=5)





# Optimization via Gradient Descent

We first start off by importing necessary packages and initializing variables. The following code initializes numpy arrays of length num_iter + 1, with all entries initialized to 0

x = np.zeros(num_iter + 1)
y = np.zeros(num_iter + 1)
Sometimes, it is useful to make use of np.ones(), which will generate arrays filled with entries equal to 1. We then define functions that will calculate the values of f and ∇ f given an array of data points (x, y).

def  f(x, y):
...
return f

def  grad_f(x, y):
...
return grad_f
This allows us to run the Gradient Descent algorithm as in

for  i in range(num_iter):
grad_x, grad_y = grad_f(x[i], y[i])
x[i + 1] = x[i] - eta * grad_x
y[i + 1] = y[i] - eta * grad_y
Here we iteratively update the value of (x, y) using ∇ f (x, y) and store each of the points in the array x and y. We next plot the surface of the function f (x, y). To start, we first create a grid of (x, y) points to evaluate f (x, y) at.

X, Y = np.meshgrid(np.linspace(xmin, xmax, n),
np.linspace(ymin, ymax, n))
Z = f(X, Y)
The function call np.linspace(min, max, n) generates an array of n equally spaced values from min to max. For example, the code

np.linspace(-2, 2, 5)
will create an array [−2, −1, 0, 1, 2]. Then np.meshgrid(x, y) will create a grid from the array of x values and the array of y values. We can now perform the 3D plotting with the following code.

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(projection=’3d’)
ax.set_xlabel(’X’)
ax.set_ylabel(’Y’)
ax.set_zlabel(’Z’)
ax.view_init(elev=ax.elev, azim=ax.azim)
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.5)
Feel free to change the values of the optional parameters to understand their purpose. Unlike the code for plotting a scatter plot of a





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

# Linear Regression

In Chapter 1, here we create an object of the Axes class with the function plt.figure().gca().10 Then we call its instance methods to add features to it (e.g., x-, y-, z-labels).

Finally, we can plot the trajectory of the Gradient Descent algorithm with the code:

ax.plot(x, y, f(x, y), color='orange', markerfacecolor='black', markeredgecolor='k', marker='o', markersize=5)
You can alternatively call:

ax.scatter(x, y, f(x, y))
but the names of optional parameters might be slightly different.

# 3.4.1 Using Machine Learning Packages

When the function f is simple and it is possible to calculate ∇ f by hand, we can implement the Gradient Descent algorithm by hand as in the previous subsection. However, in most ML programs, the loss function f is very high-dimensional, and it is difficult to write a single function to directly compute the gradient ∇ f. Instead, we can make use of functions defined in popular ML packages. Here, we introduce one such package called PyTorch:

- torch: This is a popular package used for designing and training machine learning models. PyTorch uses an object-oriented interface for user convenience and provides access to optimized array data structures called tensors to make computations faster and more efficient. The package also provides support for GPU training.11

Using PyTorch, Gradient Descent can be implemented in just a few lines:

import torch
model = ...
opt = torch.optim.SGD(model.parameters(), lr=0.1)
The code above will create an instance of the Optimizer class, which has pre-defined methods that will compute the gradients and automate the Gradient Descent process.

# 3.4.2 Beyond Vanilla Gradient Descent

If you visit the documentation for torch.optim,12 you may notice that there are other algorithms listed as an alternative to the Stochastic Gradient Descent. A lot of these algorithms are extensions of the GD algorithm we explained throughout this chapter, which have proven to be more effective than the vanilla GD algorithm in certain cases.

10 You can read more about the differences between these two matplotlib interfaces at this link.

11 Documentation is available at this link.

12 https://pytorch.org/docs/stable/optim.html




# Optimization via Gradient Descent

(e.g., Adam, Adagrad, Nesterov momentum). For example, these algorithms may choose to add a momentum to the gradient, so that the rate of change of f will be accelerated if it has been updating in the same direction in the recent few steps. These algorithms may also choose to use a different learning rate for each of the model parameters. In particular, an appropriate learning rate can be computed based on the mean and the variance of the gradient values from the recent few steps.


NO_CONTENT_HERE



# 4 Linear Classification

Multi-way Classification is a task of learning to predict a label on newly seen data out of k possible labels. In binary classification, there are only two possible labels, say ±1. Sentiment prediction in Chapter 1 was an example of a binary classification task. In this chapter, we introduce two other linear models that perform binary classification: logistic regression and Support Vector Machines (SVMs). From these two models, we learn more about the thought process of designing loss functions that are appropriate to the task.1

In this chapter, we are interested in using linear models to perform classification. In a binary classification problem, the training dataset consists of (point, label) pairs (x, y) where y can take two values (e.g., {±1} or {0, 1}). In a more general multi-class classification problem, the data has one of k labels, drawn from {0, 1, . . . , k − 1}.

# 4.1 General Form of a Linear Model

You already encountered a linear model in Chapter 1 — the least squares regression model for sentiment prediction. Given an input x, we learned a parameter vector w that minimizes the loss ∑i(yi − w · xi)². The model can be seen as mapping an input vector x to a real value w · x. For sentiment classification, we changed this real-valued output at test time to ±1 by outputting sign(w · x).

You probably wondered there: Why don’t we simply use sign(w · x) directly as the output of the model while training? In other words, why not do training on the following loss:

∑i(yi − sign(w · xi))² (4.1)

The answer is that using the sign(z) function in the loss makes gradient-based optimization ill-behaved. The derivative of sign(z) is 0 except at z = 0 (where the derivative is discontinuous) and thus the gradient is uninformative about how to update the weight vector.




48   introduction to machine learning lecture notes for cos 324 at princeton university

So the work-around in Chapter 1 (primarily for ease of exposition) was to train the sentiment classification model using the least squares loss ∑i(yi − w · xi)², which in practice is used more often in settings where the desired output yi is real-valued output as opposed to binary. This gave OK results, but in practice one would use either of the two linear models introduced in this chapter: Logistic Regression and Support Vector Machines. These are similar in spirit to the linear regression model — (1) given an input x, the models learn a parameter vector w that minimizes a loss, defined as a differentiable function on w · x; (2) at test-time, the model outputs sign(w · x). The main difference, however, is that the loss for the linear models introduced in this chapter is designed specifically for the binary classification task. Pay close attention to our “story” for why the loss makes sense. This will prepare you to understand any new loss functions you come across in your future explorations.

# 4.2  Logistic Regression

The logistic regression model arises from thinking of the answer as being probabilistic: the model assigns a “probability” to each of the two labels, with the sum of the two probabilities being 1. This paradigm of a probabilistic answer is a popular way to design loss functions in a host of ML settings, including deep learning.

Definition 4.2.1 (Logistic model). Given the input x, the model assigns the “Probability that the output is +1” to be

σ(w · x) = 1
1 + exp(−w · x) (4.2)

where σ is the sigmoid function (see Chapter 19). This implies that “Probability that the output is −1” is given by

1 − σ(w · x) = exp(−w · x) = 1 (4.3)
1 + exp(−w · x) 1 + exp(w · x)

See Figure 4.1. Note that “the probability that the output is +1” is greater than 1 precisely if w · x > 0. Furthermore, increasing the value of w · x causes the probability to rise towards 1. Conversely, if w · x &#x3C; 0, then “the probability of label −1” is greater than 1. When w · x = 0, the probability of label +1 and −1 are both equal to ½. In this way, the logistic model can be seen as a continuous version of the sign(w · x).

Example 4.2.2. If x = (1, −3) and w = (0.2, −0.1), then the probability of label +1 is

σ(w · x) = 1
1 + exp(−0.2 − 0.3) ≃ 0.62





# linear classification

# 1.5

Figure 4.1: The graph of the probability that the output of a logistic model is +1 (red) or -1 (blue) given w · x.

-1

Pr[label-is-+1]

0.5

Pr[label-is--1]

-5        0                                     5  W·X

-0.5

# 4.2.1 Defining Goodness of Probabilistic Predictions

Thus far, we explained how the logistic model generates its output given an input vector x and the current weight vector w. But we have not yet talked about how to train the model. To define a loss function, we need to decide what are the “good” values for w. Specifically, we formulate a definition of “quality” of probabilistic predictions.

Definition 4.2.3 (Maximum Likelihood Principle). Given a set of observed events, the goodness of a probabilistic prediction model6 is the probability it assigned to the set of observed events.

6 This is a definition of goodness, not the consequence of some theory.

We illustrate with an example.

Example 4.2.4. You often see weather predictions that include an estimate of the probability of rain.

|         | M   | T   | W   | Th  | F   |
| ------- | --- | --- | --- | --- | --- |
| Model 1 | 60% | 20% | 90% | 50% | 40% |
| Model 2 | 70% | 50% | 80% | 20% | 60% |
| Rained? | Y   | N   | Y   | N   | N   |

We can answer this question by seeing which model assigns higher likelihood to the events that were actually observed (i.e., whether or not it rained). For instance, the likelihood of the observed sequence according to Model 1 is

0.6 × (1 − 0.2) × 0.9 × (1 − 0.5) × (1 − 0.4) = 0.1296




50  introduction to machine learning lecture notes for cos 324 at princeton university

The corresponding number for Model 2 is 0.0896 (check this!). So Model 1 was a “better” model for this week.

# 4.2.2  Loss Function for Logistic Regression

We employ the Maximum Likelihood Principle from the previous part to define the loss function for the logistic model. Suppose we are provided the labeled dataset {(x1, y1), (x2, y2), . . . , (xN, yN)} for training where yi is a ±1 label for the input xi. By the description given in Definition 4.2.1, the probability assigned by the model with the weights w to the i-th labeled data point is

1

1 + exp(−yiw · xi)

which means that the total probability (“likelihood”) assigned to the dataset is

N

P = ∏ 1 + exp(−yiw · xi)           (4.4)

i=1

We desire the model w that maximizes P. Since log(x) is an increasing function, the best model is also the one that maximizes log P, hence the one that minimizes − log P = log 1. This leads to the logistic loss function:

N

log     ∏ (1 + exp(−yiw · xi))  = ∑ log(1 + exp(−yiw · xi))   (4.5)

i=1                     i=1

Note that this expression involves a sum over training data points, which as discussed in Section 3.2, is a very desirable and practical property of loss in machine learning.

# Problem 4.2.5.

Verify that the gradient for the logistic loss function is

N

∇ℓ = ∑ 1 + exp(yiw · xi)                              (4.6)

i=1

# 4.2.3      Using Logistic Regression for Roommate Matching

In this part, we use the following example to illustrate some of the material covered in the previous parts.

# Example 4.2.6.

Suppose Princeton University decides to pair up newly admitted undergraduate students as roommates. All students are asked to fill a questionnaire about their sleep schedule and their music taste. The questionnaire is used to generate a compatibility score in [0, 1] for each of the two attributes, for each pair of students. Table 4.2 shows the calculated





# 4.2 Sample data of compatibility scores for four pairs of students.

| Sleep (S) | Music (M) | Compatible? |
| --------- | --------- | ----------- |
| 1         | 0.5       | +1          |
| 0.75      | 1         | +1          |
| 0.25      | 0         | −1          |
| 0         | 1         | −1          |

Compatibility scores for four pairs of roommates from previous years and whether or not they turned out to be compatible (+1 for compatible, −1 for incompatible).

We wish to train a logistic model to predict if a pair of students will be compatible based on their sleep and music compatibility scores. To do this, we first convert the data in Table 4.2 into a vector form.

x1 = (1, 1, 0.5)            y1 = +1
x2 = (1, 0.75, 1)           y2 = +1             (4.7)
x3 = (1, 0.25, 0)           y3 = −1
x4 = (1, 0, 1)              y4 = −1

Where the first coordinate xi of xi is a dummy variable to introduce a constant bias term, and the second and third coordinates are respectively for sleep and music compatibility scores.

# Figure 4.2: Graph representing the points in Table 4.2.

The x-, y-axis in the graph correspond to the Sleep and Music compatibility scores, or the second and third coordinates in (4.7).

Consider two models — Model 1 with the weight vector w1 = (0, 1, 0) and Model 2 with the weight vector w2 = (0, 0, 1). Model 1 only looks at the sleep compatibility score to calculate the probability that a pair of students will be compatible as roommates, whereas





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

# Model Comparisons

Model 2 only uses the music compatibility score. For example, Model 1 assigns the probability that the first pair of students are compatible as:

σ(w1 · x1) = 1 + 1 (− 1) ≃ 0.73

We can calculate the probability for the other pairs and for Model 2 and fill out the following Table 4.3:

|             | Pair 1 | Pair 2 | Pair 3 | Pair 4 |
| ----------- | ------ | ------ | ------ | ------ |
| Model 1     | 0.73   | 0.68   | 0.56   | 0.50   |
| Model 2     | 0.62   | 0.73   | 0.50   | 0.73   |
| Compatible? | Y      | Y      | N      | N      |

Then the likelihood of the observations (YYNN) according to Model 1 can be calculated as:

0.73 × 0.68 × (1 − 0.56) × (1 − 0.50) ≃ 0.11

whereas the likelihood of the observations according to Model 2 is:

0.62 × 0.73 × (1 − 0.50) × (1 − 0.73) ≃ 0.06

Therefore, the Maximum Likelihood Principle tells us that Model 1 is a “better” model than Model 2.

# 4.2.4 Testing the Model

After training the model on the training data, we can use it to define label probabilities on any new data point. However, the probabilities do not explicitly tell us what label to output on a new data point. There are two options:

1. (Probabilistic) If p is the probability of the label +1 according to (4.2), then use a random number generator to output +1 with probability p and −1 with probability 1 − p.
2. (Deterministic) Output the label with a higher probability.





# Linear Classification

Recall from an earlier discussion that Pr[+1] ≥ Pr[−1] if and only if w · x ≥ 0. In other words, the second deterministic option is equivalent to the sign(z) function: sign(w · x)! We conclude that logistic regression is quite analogous to what we did in Chapter 1, except instead of least squares loss, we are using logistic loss to train the model. The logistic loss is explicitly designed with binary classification in mind.7

7 Using logistic loss (and ℓ2 regularizer) instead of least squares in our sentiment dataset boosts test accuracy from 78.1% to 80.7%.

# 4.3 Support Vector Machines

A Support Vector Machine (SVM)8 is also a linear model. It comes in several variants, including a more powerful kernel SVM that we will not study here. But this rich set of variants made it an interesting family of models, and it is fair to say that in the 1990s its popularity was somewhat analogous to the popularity of deep nets today. It remains a very useful model for your toolkit. The version we are describing is a so-called soft margin SVM.

As in the least squares regression, the main idea in designing the loss is that the label should be +1 or −1 according to sign(w · x). But we want to design a loss with a well-behaved gradient that provides a clearer direction of improvement. To be more specific, we want the model to have more “confident” answers, and we will penalize the model if it comes up with a correct answer but with a low degree of “confidence.”

For z ∈ R, let us define

Hinge(z) = max{0, 1 − z} (4.8)

Note that this function is always at least zero, and strictly positive for z &#x3C; 1. When z decreases to negative infinity, there is no finite upper bound to the value. The derivative is zero for z > 1 and 1 for z &#x3C; 1. The derivative is currently undefined at z = 1, but we can arbitrarily choose between 0 or 1 as the newly defined value.

For a single labeled data point (x, y) where y ∈ {−1, 1}, the SVM loss is defined as

ℓ = Hinge(yw · x) (4.9)

and its gradient is

∇ℓ =
−yx


# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

That is, the sum of the SVM loss on each of the training data points. Suppose y = +1. Then this loss is 0 only when w · x > 1. In other words, making loss zero not only requires w · x to be positive, but also be comfortably above 0. If w · x dips below 1, the loss is positive and increases towards +∞ as w · x → −∞. (Likewise if the label y = −1, then the loss is 0 only when w · x &#x3C; −1.)

Recall that the goal of a gradient-based optimization algorithm is to minimize the loss. Therefore, the loss gives a clear indication of the direction of improvement until the data point has been classified correctly with a comfortable margin away from 0, out of the zone of confusion.

# Example 4.3.1

Recall the roommate compatibility data from Table 4.2. Consider the soft-margin SVM with the weight vector w = (−1.5, 3, 0). This means the decision boundary — the set of points where w · x = 0 — is drawn at Sleep = 1, and the margins — the set of points where w · x = ±1 — are drawn at Sleep = 6 and Sleep = 6. Figure 4.4 shows the decision boundary and the two margin lines of the model. The SVM loss is zero for the point (1, 0.5) because it is labeled +1 and away from the decision boundary with enough margin. Similarly, the loss is zero for the point (0, 1). The loss for the point (0.75, 1), however, can be calculated as

Hinge(1 · (−1.5 · 1 + 3 · 0.75)) = 0.25

and similarly, the loss for the point (0.25, 0) is 0.25.

Figure 4.4: The decision boundary of a soft-margin SVM on the roommate matching example. The region to the left of the two dotted lines is where the model confidently classifies as −1; the region to the right is where it confidently classifies as +1; and the region between the two dotted lines is the zone of confusion.

| Music |          |        |
| ----- | -------- | ------ |
| 0     | 0.5      | 1      |
| 1.5   | W·X=+1   |        |
| 0.5   | W·x!= -1 | W·X= 0 |

The gradient of the loss at the point (0.75, 1) is

−yx = (−1, −0.75, −1)



linear classification 55

and the update rule for a gradient descent algorithm will be written as

w ← (−1.5, 3, 0) − 0.1(−1, −0.75, −1) = (−1.4, 3.075, 0.1)

where η = 0.1, and the new SVM loss will be

Hinge(1 · (−1.4, 3.075, 0.1) · (1, 0.75, 1)) = 0

which is now lower than the SVM loss before the update.

# 4.4 Multi-class Classification (Multinomial Regression)

So far, we have only seen problems where the model has to classify using two labels ±1. In many settings there are k possible labels for each data point 9 and the model has to assign one of them. The 9 This is the case in most settings in modern machine learning. For instance in the famous ImageNet challenge, each image belongs to one of 1000 classes.

The conceptual framework is similar to logistic regression, except the model defines a nonzero probability for each label as follows. The notation assumes data is d-dimensional and the model parameters consist of k vectors θ1, θ2, . . ., θk ∈ Rd. We define a new vector z ∈ Rk where each coordinate zi satisfies zi = θi · x. Then the probability of a particular label is defined through the softmax function (see Chapter 19):

Pr[label i on input x] = softmax(z)

= k exp(θi · x) (4.11)

∑j=1 exp(θj · x)

This distribution can be understood as assigning a probability to label i such that it is exponentially proportional to the value of θi · x.

# Problem 4.4.1.

Using the result of Problem 19.2.4, verify that the definition of logistic regression as in (4.2), (4.3) are equivalent to the definition of multi-class regression as in (4.11).

# Problem 4.4.2.

Reasoning analogously as in logistic regression, derive a training loss for this model using Maximum Likelihood Principle.

Since exp(z) > 0 for every real number z the model above assigns a nonzero probability to every label. In some settings that may be appropriate. But as in case of logistic regression, at test time we also have the option of extracting a deterministic answer out of the model: the i ∈ {1, 2, . . . , k} that has the largest value of θj · x.

# 4.5 Regularization with SVM

It is customary to use a regularizer, typically ℓ2, with logistic regression models and SVMs. When a ℓ2 regularizer is applied, the full





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

SVM loss is rewritten as

∑ Hinge(yⁱw · xⁱ) + λ ∥w∥²       (4.12)

i                          2

Let’s see why regularization is sensible for SVMs, and even needed. The Hinge function (4.8) treats the point z = 1 as special. In terms of the SVM loss, this translates to the thought that having w · x > 1 is a more “confident” classification of x than just having the sign(w · x) be correct (i.e., w · x > 0). But this choice is arbitrary because we have not specified the scale of w. If w · x = 1/10 then scaling w by a factor 10 ensures w · x > 1. Thus the training algorithm has cheap and meaningless ways of reducing the training loss. By applying an ℓ2 regularizer, we are able to prevent this easy route for the model, and instead, force the training to find optimal weights w with a small norm.

# Problem 4.5.1

Write a justification for why it makes sense to limit the ℓ2 norm of the classifier during logistic regression. How can large norm lead to false confidence (i.e., unrealistically low training loss)?

# 4.6 Linear Classification in Python Programming

In this section, we briefly discuss how to implement the logistic regression model in Python. It is customary to use the numpy package to speed up computation and the matplotlib package for visualization.

# import necessary packages
import  numpy as np
import  matplotlib.pyplot as plt

# prepare dataset
X = ...  # array of shape (n, d), each row is a d-dimensional data point
y = ...  # array of shape (n), each value = -1 or +1
w = ...  # array of shape (d), each value is a weight for each dimension
X_train, X_test, y_train, y_test, eta = ...

# define functions
def  loss(X, y, w):
# returns the logistic loss
# return sum log(1 + exp(-y*w*x))
...

def  grad_loss(X, y, w):
# returns the gradient of the logistic loss
# return sum (-y*x)/(1 + exp(y*w*x))
...

def  gradient_descent(X, y, w0, eta)
...
return w

# run Gradient Descent
w = gradient_descent(X_train, y_train, w, eta)





# linear classification

plot the learned classifier

assuming data is 2-dimensional

colors = {1: 'blue', -1: 'red'}
xmin, xmax, ymin, ymax = ...
plt.scatter(X[:,0], X[:,1], c=np.array([colors[y_i] for y_i in y]))
plt.plot([xmin, xmax], [ymin, ymax], c='black')
We have already discussed how to implement the majority of the code sample above in previous chapters. The only parts that are new are the functions to calculate the logistic loss and its gradient. This is consistent with the theme of this chapter — to discuss how to design loss functions that are appropriate for the task. Nevertheless, while the content of this code sample is familiar, some sections of the code introduce new Python functionality and syntax. We first consider the logistic loss and gradient functions:

def loss(X, y, w):
# returns the logistic loss
# return sum log(1 + exp(-y*w*x))
...

def grad_loss(X, y, w):
# returns the gradient of the logistic loss
# return sum (-y*x)/(1 + exp(y*w*x))
...
In Java, the programming language you learned in earlier programming classes, you would have to rely on a for loop to account for the array inputs in the loss() and grad_loss() functions. However, Python and numpy support many vectorized operations, including matrix multiplication and element-wise multiplication. These operations are far more concise to read and will also improve the runtime of the program by a great margin. Note that the code snippet above does not contain these operations; it is simply pseudo-code for your intuition. You will be introduced to these vectorized operations during the precept, and you will be expected to implement the loss function with these new tools in your programming assignments.

Next, we use a Python dictionary to store information corresponding to the plot’s coloring scheme:

colors = {1: 'blue', -1: 'red'}
This is equivalent to a hash table from Java. Here, 1 and −1 are the keys and “blue” and “red” are respectively their values.

We will now discuss multi-dimensional arrays in Python. There are multiple ways to perform array indexing. For example, if X is a 2-dimensional array, both X[i][j] and X[i, j] can be used to extract the entry at the i-th row, j-th column. It is also possible to provide a set of rows or a set of columns to extract. The following code snippet generates an array of shape (2, 2), where each entry is from the row 0




58   introduction to machine learning lecture notes for cos 324 at princeton university

or 1 and column 0 or 2:

X[[0, 1], [0, 2]]

Note that similar to the 1D case, the : operator is used to perform array slicing. Bounding indices can be omitted as shown in the following code snippet:

X[:,0]

This extracts the full set of rows and the column 0, or in other words, the first column of X.

Finally, we use a list comprehension to specify the plotting color for each data point:

[colors[y_i] for y_i in y]

This is Python syntactic sugar that allows the user to create an array while iterating over the elements of an iterator. The code snippet here is equivalent to the following code.

list = []
for  y_i in y:
list.append(colors[y_i])



# Exploring “Data Science” via Linear Regression

So far, our treatment of machine learning has been from the perspective of a computer scientist. It is important to note, however, that models such as linear regression are useful in a variety of other fields including the physical sciences, social sciences, etc. In this chapter, we present case studies from different fields. Here, the inputs xi are considered to be explanatory variables, the output y is considered to be the effect variable, and the weights wi quantify the causal significance of the associated inputs xi on the output y. The interpretation of weights as a type of causality is crucial; often, the ideal method of determining causality through a set of rigorous randomized control trials is too expensive.

# 5.1  Boston Housing: Machine Learning in Economics

Our first case study comes from the field of economics. In 1978, Harrison and Rubinfeld released a classic study on the willingness to pay for clean air in the Boston metropolitan area. Their methodology involved an economic model called hedonic pricing,1 which essentially estimates the value of a good by breaking it down into “constituent characteristics.” It turns out we can use linear regression to help determine which of these attributes are most important. Specifically, suppose we have a dataset of house sales where y represents the price of the house and x ∈ R15 represents a set of house attributes.2 Then, we aim to find an optimum set of weights w for the linear model:

y ≈ ∑ wi xi (5.1)

Table 5.1 lists all 14 attributes that were used in the linear regression model. Before fitting the model with these attributes, it is useful to intuitively reason about some of the attributes. For instance, we expect the weight w5 corresponding to RM, the number of bedrooms,

| Attribute | Description             |
| --------- | ----------------------- |
| x0        | Dummy variable          |
| x1        | Attribute 1             |
| x2        | Attribute 2             |
| x3        | Attribute 3             |
| x4        | Attribute 4             |
| x5        | RM (Number of bedrooms) |
| x6        | Attribute 6             |
| x7        | Attribute 7             |
| x8        | Attribute 8             |
| x9        | Attribute 9             |
| x10       | Attribute 10            |
| x11       | Attribute 11            |
| x12       | Attribute 12            |
| x13       | Attribute 13            |
| x14       | Attribute 14            |




60  introduction to machine learning lecture notes for cos 324 at princeton university

Index

|    | Code    | Description                                                             |   |   |
| -- | ------- | ----------------------------------------------------------------------- | - | - |
| 1  | ZN      | proportion of residential land zoned for lots over 25,000 ft2           |   |   |
| 2  | INDUS   | proportion of non-retail business acres per town                        |   |   |
| 3  | CHAS    | (1 Charles River indicator variable if tract bounds river; 0 otherwise) |   |   |
| 4  | NOX     | nitric oxides concentration (parts per 10 million)                      |   |   |
| 5  | RM      | average number of rooms per dwelling                                    |   |   |
| 6  | AGE     | proportion of owner-occupied units built prior to 1940                  |   |   |
| 7  | DIS     | weighted distances to five Boston employment centres                    |   |   |
| 8  | RAD     | index of accessibility to radial highways                               |   |   |
| 9  | TAX     | full-value property-tax rate per $10,000                                |   |   |
| 10 | MEDV    | Median value of owner-occupied homes (in $1,000s)                       |   |   |
| 11 | CRIM    | per capita crime rate in town                                           |   |   |
| 12 | PTRATIO | pupil-teacher ratio by town                                             |   |   |
| 13 | LSTAT   | % lower status of the population                                        |   |   |
| 14 | B       | 1000(Bk − 0.63)2 where Bk is the proportion of black population in town |   |   |

to be positive because larger houses typically sell for more. Conversely, we expect the weight w4 corresponding to NOX, the amount of air pollution, to be negative as people would prefer not to live in a polluted environment. After running the regression, it indeed turns out that these intuitions are correct.3 In general, it can be useful to double-check that the calculated weights align with intuition: if they do not, it could be a sign that a modeling assumption is incorrect.

3 The regression weights can be found on page 100 of the original paper. https://deepblue.lib.umich.edu/bitstream/handle/2027.42/22636/0000186.pdf?sequence=1&#x26;isAllowed=y.

# 5.1.1  The Strange Math of Feature B

The headline result of the paper is that the willingness to pay for cleaner air increases both when income level is higher and when the current pollution level is higher. However, if you read the paper closely, you may notice the presence of a curious parameter B, which is defined in terms of Bk, the proportion of black population in the neighborhood. This parameter is meant to represent a social segregation effect present within the Boston housing market. The authors of the paper speculated that (1) at a lower level of Bk, the housing price will decrease as Bk increases since the white population tend to avoid black population, but (2) at a very high level of Bk, the housing price will increase as Bk increases because black population prefer predom-





# exploring “data science” via linear regression

inantly black neighborhoods. To capture this intuition, they defined the attribute B in the parabolic expression B = 1000(Bk − 0.63)². It indeed turns out that the weight w14 corresponding to B is positive as shown in Figure 5.1.

Figure 5.1: The graph of B = 1000(Bk − 0.63)². This is an example of featurization as discussed in Chapter 1. It encodes prevailing discrimination of that period. The term “black” is not favored today either.

# 5.1.2 Ethnic Concerns Behind a Model

It seems strange to have such a sensitive attribute as B have an influence on the model. We might wonder about the social harm that could arise if the model was used by real-life sellers or buyers (e.g., the buyers could demand a house for a lower price based on the proportion of black population in the neighborhood). On the other hand, the fitted model confirms that there is an underlying segregation effect already present in the society. Also, we cannot guarantee that the model would be race-neutral even if we eliminated the parameter B. For instance, maybe one or more of the other variables (e.g., air quality variables) is highly correlated with B.4

Ultimately, the primary takeaway from this case study is that implementing machine learning models in real life is a challenge itself. At a technical level, the model may make sense and make good predictions of house prices. But one has to consider the social effects of an ML model on the phenomenon being studied: in particular, whether it supports or extends prevailing inequities. The following are some important considerations to keep in mind:

1. If the world has a problem, the data will reflect it and so will our models
2. If a problematic model later gets used in real life, it can exacerbate the existing problem
3. The choices of attributes when making a model might bias the outcome





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

# 4. Carelessly Using Data Can Later Lead to Modeling Issues

# 5.2 fMRI Analysis: Machine Learning in Neuroscience

We next consider an application of ML in a vastly different field. One of the most important tools in contemporary neuroscience is Functional Magnetic Resonance Imaging (fMRI). fMRI has been used successfully to map human functionality (e.g., speech, memory) to brain regions. In a more active role, it can assist with tumor surgery or “decoding” thoughts and emotions.

Figure 5.2: A sample image of an fMRI reading. Source: Wikipedia

fMRI experiments often involve presenting a set of stimuli (e.g., images of human faces) to the subject in order to elicit a neurological response, which is then captured through an fMRI reading. Each reading reveals the concentration of oxygen in the blood stream throughout the brain, which is used as a proxy for brain activity.5 Through the result of the reading, we are able to conclude if a particular voxel responds to a particular stimulus. The naive way of conducting these experiments is to present one stimulus at a time and wait until we get a reading of the brain response before we move on to the next stimulus.

But if you have previously taken a course in neuroscience, you may recall that fMRI is unfortunately a double-edged sword. It features excellent spatial resolution, with each voxel as small as 1 mm3. However, it has poor temporal resolution: often, readings require several seconds for blood flow to stabilize! Coupled with the fact that regulations limit the amount of time human subjects can spend in the scanner, it becomes clear that methodologies based on sequential presentation of stimuli are too inefficient. In this section, we explore





# exploring “data science” via linear regression

# 5.2.1 Linear Superposition

The key intuition involves a concept called linear superposition: if a subject is shown multiple stimuli in quick succession, the strength of the voxel’s response is the sum of the strength of its response to each of the individual stimuli.6 Instead of waiting until we have the image of one stimulus to move on, considering showing a new stimulus every 1 or 2 seconds. Each fMRI reading will now capture the composite brain response to the stimuli from the past few seconds. We will use linear regression to disentangle the information, and extract which voxel responded to which stimulus.7

Consider the following example.

# Example 5.2.1.

See Figure 5.3. The graph on the top left represents a voxel’s response when the subject is shown the image of a face. The graph on the top right represents the response when the subject is shown the image of a flower. The bottom graph represents the response when the subject is shown the image of a flower 1 second after the image of a face. Notice that the first two graphs have been superposed to create the third graph. In practice, we are interested in the problem of extracting the individual graphs when given the superposed graph.

| Face                                                                    | Flower |
| ----------------------------------------------------------------------- | ------ |
| Figure 5.3: Three graphs explaining the effect of linear superposition. |        |
| Time                                                                    | Time   |
| 5s                                                                      | 5s     |
| Face                                                                    | Flower |
| Time                                                                    |        |
| 6s                                                                      |        |

# 5.2.2 Linear Regression

Now let us describe how to formulate this problem in terms of linear regression. First assume that the subject is shown one of k types of stimuli at each time step t where t ∈ {1, 2, . . . , T}. Let yt be the





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

# 5.2.3 Neural Correlates of Thought

Now we know how to find the values of ws for a specific voxel. That is, we can test if a particular voxel responds to a particular stimulus. Combining this method with a spatial smoothing (i.e., applying the principle that nearby voxels behave similarly)8, we are able to identify which region of a brain is associated to which stimulus. So far, more than 1,000 regions of the brain have been identified and mapped.

Figure 5.4: A detailed map labeling areas of the brain with corresponding stimuli.

# 5.2.4 Locational Brain-Computer Interface (BCI)

We finish off with a tangible example of how our studies can help people. Patients who are suffering from Locked-in Syndrome (LIS) are aware of their surroundings and have normal reasoning capacities.

| PC1       | PC3      | emotional | communal | social       |
| --------- | -------- | --------- | -------- | ------------ |
| emotional | communal | emotional | communal | professional |
| mental    | social   | 03        | ona      | mental       |
| SMFA      | SMFA     |           |          |              |
| MHA       | M1F      | S1F       | RSC      | PC2          |
| RSC       | M1F      |           |          |              |
| FEF       | M1H      | S1H       |          |              |
| V3B       | 3B       | S1H       |          |              |
| S1M       | V3A      | EBA       |          |              |
| Superior  | OFA      | RH        | Superior | Anterior     |
| S6        | S5       |           |          |              |




# Exploring “Data Science” via Linear Regression

but have no way of communicating with others through speech or facial movements. Using a combination of a technology called Brain-Computer Interface and a linear regression model, we are able to communicate with these patients.

Brain-Computer Interface is an electrode sensor implanted near the motor cortex that can detect the electric signal that LIS patients are trying to send to the motor cortex. We can teach the patients to visualize writing with their dominant hand if they want to answer “no” and visualize writing with their non-dominant hand if they want to answer “yes.” Since the neural correlates of the two movements are very different, BCI will pick up essentially disjoint signals, and we can use linear regression model to distinguish between them.

Note: training also requires labeled data, which can be produced by asking the patient questions about known facts (e. g., birth date, marital status, etc.). This technique has been used to communicate with patients in deep coma and presumed to be in a vegetative state. See *Science of Mind Reading*, New Yorker, December 6 2021, which also profiles several Princeton researchers.


NO_CONTENT_HERE


# Part II

# Unsupervised Learning


NO_CONTENT_HERE



# 6 Clustering

So far, we have considered ML models which require labeled data in order to learn. However, there is a large class of models which can learn from unlabeled data. From this chapter, we will begin to introduce models from this modeling paradigm, called unsupervised learning. In this chapter, we focus on one application of unsupervised learning, called clustering algorithm.

# 6.1 Unsupervised Learning

Unsupervised learning is a branch of machine learning which only uses unlabeled data. Examples of unlabeled data include a text corpus containing the works of William Shakespeare (Chapter 8) or a set of unlabeled images (Chapter 7). Some key goals in this setting include:

- Learn the structure of data: It is possible to learn if the data consists of clusters, or if it can be represented in a lower dimension.
- Learn the probability distribution of data: By learning the probability distribution where the training data came from, it is possible to generate synthetic data which is “similar” to real data.
- Learn a representation for data: We can learn a representation that is useful in solving other tasks later. With this new representation, for example, we can reduce the need for labeled examples for classification.

# 6.2 Clustering

Clustering is one of the main tasks in unsupervised learning. It is the process of detecting clusters in the dataset. Often the membership of a cluster can replace the role of a label in the training dataset. In general, clusters reveal a lot of information about the underlying structure of the data.





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

# Heights and Weights of Female and Male Basketball Players

Figure 6.1: Height vs weight scatter plot of basketball players. In the plot on the right, the points in green and blue respectively correspond to female and male players.

| 225 |    |         | \*  | 225  |    |    |    | \*\* |   |   |
| --- | -- | ------- | --- | ---- | -- | -- | -- | ---- | - | - |
| 220 |    | \* \*\* | 220 | \*\* | \* | \* |    |      |   |   |
| 215 |    |         | \*  | 215  |    |    |    |      |   |   |
| 210 | \* | 210     |     |      |    |    |    |      |   |   |
| 205 |    | 205     |     |      |    |    |    |      |   |   |
| 200 |    | 200     |     |      |    |    |    |      |   |   |
| 195 | \* | \*      | 195 | \*   |    |    |    |      |   |   |
| 190 |    | 190     |     |      |    |    | \* |      |   |   |

5.50  5.75  6.00  6.25  6.50  6.75  7.00  7.25  7.50

Height (feet)

In Figure 6.1, we see a scatter plot of measurements of height and weight of basketball players. If you look at the plot on the left, it is easy to conclude that there is a usual linear relationship between the height and the weight of the athletes. However, upon further inspection, it seems like there are two clusters of the data points, separated around the middle of the plot. In fact, this is indeed the case! If we label the dataset with the additional information of whether the data point is from a male or female athlete, the plot on the right shows something more than just the linear relationship. In practice, however, we do not always have access to this additional label. Instead, one uses clustering algorithms to find natural clusterings of the data. This raises the question of what a “clustering” is, in the first place. Technically, any partition of the dataset D into k subsets C1, C2, . . . , Ck can be called a clustering. That is,

k

Ci = D and Ci = ∅

i=1

i=1

But we intuitively understand that not all partitions are a natural clustering of the dataset; our goal therefore will be to define what a “good” clustering is.

# 6.2.1 Some Attempts to Define a “Good” Cluster

The sample data in Figure 6.1 suggests that our vision system has evolved to spot natural clusterings in two or three dimensional data. To do machine learning, however, we need a more precise definition in Rd: specifically, for any partition of the dataset into clusters, we try to quantify the “goodness” of the clusters.

Definition 6.2.1 (Cluster: Attempt 1). A “good” cluster is a subset of points which are closer to each other than to all other points in the dataset.

But this definition does not apply to the clusters in Figure 6.1. The points in the middle of the plot are far away from the points on the top right corner or the bottom left corner. So whichever cluster we assign the middle points to, they will be farther away from some.





clustering 71

points in their assigned cluster than to some of the points on the other cluster. Ok, so that did not work. Consider the following definition.

# Definition 6.2.2 (Cluster: Attempt 2).

A “good” cluster is a subset of points which are closer to the mean of their own cluster than to the mean of other clusters.

# Definition 6.2.3 (Mean and Variance of Clusters).

Let Ci be one of the clusters for a dataset D. Let mi = |Ci| denote the cluster size. The mean of the cluster Ci is

yi = (1/mi) ∑x∈Ci x

and the variance within the cluster Ci is

σi2 = (1/mi) ∑x∈Ci ||x − yi||2

You may notice that Definition 6.2.2 appears to be using circular reasoning: it defines clusters using the mean of the clusters, but the mean can only be calculated once the clusters have been defined. Such circular reasoning occurs in most natural formulations of clustering. Look at the Wikipedia page on clustering for some other formulations.

# 6.3 k-Means Clustering

In this section, we present a particular partition of the dataset called the k-means clustering. Given k, the desired number of clusters, the k-means clustering partitions D into k clusters C1, C2, . . . , Ck so as to minimize the cost function:

∑i=1k ∑x∈Ci ||x − yi||22 (6.1)

This can be seen as minimizing the average of the individual cost of the k clusters, where cost of Ci is ∑x∈Ci ||x − yi||22. Notice that each cluster cost is the cluster size times the variance. This idea is similar in spirit to our earlier attempt in Definition 6.2.2 — we want the distance of each data point to the mean of the cluster to be small. But this method is able to circumvent the problem of circular reasoning.

The process of finding the optimal solution for (6.1) is called the k-means clustering problem.

# 6.3.1 k-Means Algorithm

Somewhat confusingly, the most famous algorithm that is used to solve the k-means clustering problem is also called k-means. It is technically a heuristic, meaning it makes intuitive sense but it is not




# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

Guaranteed to find the optimum solution.4 The following is the k-means algorithm. It is given some initial clustering (we discuss some choices for initialization below) and we repeat the following iteration until we can no longer improve the cost function:

- Maintain clusters C1, C2, . . . , Ck
- For each cluster Ci, find the mean yi
- Initialize new clusters C' ← ∅
- for x ∈ D do
- x' = arg mini ∥x − yi∥2
- Ci ← C' ∪ {x}
- end for
- Update clusters Ci ← C'

At each iteration, we find the mean of each current cluster. Then for each data point, we assign it to the cluster whose mean is the closest to the point, without updating the mean of the clusters. In case there are multiple cluster means that the point is closest to, we apply the tie-breaker rule that the point gets assigned to the current cluster if it is among the closest ones; otherwise, it will be randomly assigned to one of them. Once we have assigned all points to the new clusters, we update the current set of clusters, thereby updating the mean of the clusters as well. We repeat this process until there is no point that is mis-assigned.

# 6.3.2 Why Does k-Means Algorithm Terminate in Finite Time?

The k-means algorithm is actually quite akin to Gradient Descent, in the sense that the iterations are trying to improve the cost.

# Lemma 6.3.1.

Given a set of points x1, x2, . . . , xm, their mean y = (1/m) ∑i=1m xi is the point that minimizes the average squared distance to the points.

Proof. For any vector z, let C(z) denote the sum of squared distance to the set of points. That is,

C(z) = ∑i=1m ∥z − xi∥2 = ∑i=1m ((z − xi) · (z − xi))

= ∑i=1m (z · z − 2z · xi + xi · xi)

= ∑i=1m (∥z∥2 − 2z · xi + ∥xi∥2)

To find the optimal z, we set the gradient ∇C to 0

∇C(z) = ∑i=1m (2z − 2xi) = 0



# clustering 73

which yields the solution

z = m ∑i=1m xi

We are ready to prove the main result.

# Theorem 6.3.2.

Each iteration of the k-means Algorithm 6.3.1, possibly except for the last iteration before termination, strictly decreases the total cluster cost (6.1).

Proof. We follow the same notation as in Algorithm 6.3.1. The total cost at the end of one iteration is:

∑i=1k ∑x∈Ci ‖x − y′‖2

where y′ is the mean of the newly defined cluster C′. Notice that each of the cluster cost ∑x∈Ci ‖x − yi‖2 is the sum of the squared distance between a set of points x∈Ci and their mean. By Lemma 6.3.1, this sum is smaller than the sum of squared distance between the same set of points to any other point. In particular, we can compare with yi, the mean of Ci before the update. That is,

∑x∈C′ ‖x − y′‖2 ≤ ∑x∈C′ ‖x − yi‖2 for any 1 ≤ i ≤ k.

If we sum over all clusters, we see that

∑i=1k ∑x∈Ci ‖x − y′‖2 ≤ ∑i=1k ∑x∈Ci ‖x − yi‖2

Now notice that the summand ‖x − yi‖2 in the right hand side of the inequality is the squared distance between the point x and the mean yi (before update) of the cluster C′ that x is newly assigned to. In other words, we can rewrite this term as ‖x − yiₓ‖2 and instead sum over all points x in the dataset. That is,

∑i=1k ∑x∈Ci ‖x − yi‖2 = ∑x∈D ‖x − yiₓ‖2

Finally, recall that the index ix was defined as ix = arg mini ‖x − yi‖2. In particular, if j was the index of the cluster that a data point x originally belonged to, then ‖x − yi‖2 ≤ ‖x − yj‖2. Therefore, we have the following inequality,

∑x∈D ‖x − yiₓ‖2 ≤ ∑j=1k ∑x∈Cj ‖x − yj‖2





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

The equality holds in the inequality above if and only if when

∥x − yi ∥2 = ∥x − yj∥2 for each point x, which means that the original cluster Cj was one of the closest clusters to x. By the tie-breaker rule, ix would have been set to j. This is exactly the case when the algorithm terminates immediately after this iteration since no point is reassigned to a different cluster. In all other cases, we have a strict inequality:

∑ ∥x − yix∥2 &#x3C; ∑j=1k ∑ ∥x − yj∥2

x∈D

Notice that the right hand side of the inequality is the total cost at the beginning of the iteration.

Now we are ready to prove that the k-means algorithm is guaranteed to terminate in finite time. Since each iteration strictly reduces the cost, we conclude that the current clustering (i.e., partition) will never be considered again, except at the last iteration when the algorithm terminates. Since there is only a finite number of possible partitions of the dataset D, the algorithm must terminate in finite time.

# 6.3.3 k-Means Algorithm and Digit Classification

You might be familiar with the MNIST hand-written digits dataset. Here, each image, which depicts some digit between 0 and 9, is represented as an 8 × 8 matrix of pixels and each pixel can take on a different luminosity value from 0 to 15.

We can apply k-means clustering to differentiate between images depicting the digit “1” and the digit “0.” After running the model with k = 2 on 360 images of the two digits, we achieve the clusters in Figure 6.2.5 Note the presence of two colored regions: a point is colored red if a hypothetical held-out data point at that location would get assigned a “0;” otherwise it is colored blue. This assignment is based on which cluster center is closer.

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| - | - | - | - | - | - | - | - | - | - |
| 2 | 3 | Y | S |   | 7 | 8 | 7 |   |   |
| 2 | 3 | 4 | 5 |   | 7 |   | 9 |   |   |
| 2 | 5 |   | 5 |   | 8 | 9 |   |   |   |
|   | 3 | 4 | 5 |   | 7 |   | 9 |   |   |
| 2 | 3 | 4 | 5 | 6 | 7 | 8 |   |   |   |
|   | 2 | 3 | 4 | 5 |   | 7 | 8 |   |   |
| 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |   |   |
| 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |   |   |

Figure 6.2: Sample images from the MNIST dataset (left) and 2D visualization of the k-means clusters differentiating between the digits “1” and “0” (right). Only two images were misclassified!





# 6.3.4 Implementation Detail: How to Pick the Initial Clustering

The choice of initial clusters greatly influences the quality of the solution found by the k-means algorithm. The most naive method is to pick k data points randomly to serve as the initial cluster centers and create k clusters by assigning each data point to the closest cluster center. However, this approach can be problematic. Suppose there exists some “ground truth” clustering of the dataset. By picking the initial clusters randomly, we may end up splitting one of these ground truth clusters (e.g., two initial centers are drawn from within the same ground truth cluster), and the final clustering ends up being very sub-optimal. Thus one tries to select the initial clustering more intelligently. For instance the popular k-means++ initialization procedure6 is the following:

1. Choose one center uniformly at random among all data points.
2. For each data point x compute D(x), the distance between x and the nearest center which has already been chosen.
3. Choose a new data point at random as a new center, where a point x is chosen with probability proportional to D(x)2.
4. Repeat steps 2 and 3 until k centers have been chosen.

In COS 324, we will not expect you to understand why this is a good initialization procedure, but you may be expected to be able to implement this or similar procedures in code.

# 6.3.5 Implementation Detail: Choice of k

Above we assumed that the number of clusters k is given, but in practice you have to choose the appropriate number of clusters k.

Example 6.3.3. Is there a value of k that guarantees an optimum cost of 0? Yes! Just set k = n (i.e., each point is its own cluster). Of course, this is useless from a modeling standpoint!

Problem 6.3.4. Argue that the optimum cost for k + 1 clusters is no more than the optimum cost for k clusters.




76  introduction to machine learning lecture notes for cos 324 at princeton university

Note that Problem 6.3.4 only concerns the optimum cost, which as we discussed may not be attained by the k-means algorithm. Nevertheless, it does suggest that we can try various values of k and see when the cost is low enough to be acceptable.

A frequent heuristic is the elbow method: create a plot of the number of clusters vs. the final value of the cost as in Figure 6.3 and look for an “elbow” where the objective tapers off. Note that if the dataset is too complicated for a simple Euclidean distance cost, the data might not be easy to cluster “nicely” meaning there is no “elbow” shown on the plot.

| number of clusters vs. score (2 digits) |       | number of clusters vs. score (10 digits) |       |   |   |
| --------------------------------------- | ----- | ---------------------------------------- | ----- | - | - |
| 8000                                    | 20000 |                                          |       |   |   |
| 7000-                                   |       |                                          |       |   |   |
| 6000 -                                  |       |                                          |       |   |   |
| 5000                                    |       |                                          |       |   |   |
| 4000                                    |       |                                          |       |   |   |
| 3000                                    |       |                                          |       |   |   |
|                                         |       | 2000                                     | 15000 |   |   |
|                                         |       | 1000                                     |       |   |   |
| 0 -                                     | 10000 |                                          |       |   |   |
|                                         |       | 1                                        | 5000  |   |   |
|                                         | 2     | 0                                        |       |   |   |
| 3                                       | ↓     |                                          |       |   |   |
| 4                                       | 8     |                                          |       |   |   |
|                                         | 5     | 9                                        |       |   |   |
|                                         | 6     | 2.5                                      |       |   |   |
|                                         |       | 5.0                                      |       |   |   |
|                                         |       | 7.5                                      |       |   |   |
|                                         |       | 10.0                                     |       |   |   |
|                                         |       | 12.5                                     |       |   |   |
|                                         |       | 15.0                                     |       |   |   |
|                                         |       |                                          | 17.5  |   |   |
|                                         |       |                                          | 20.0  |   |   |

# 6.4 Clustering in Programming

In this section, we briefly discuss how to implement k-means algorithm for digit classification in Python. As usual, we use the numpy package to speed up computation and the matplotlib package for visualization. Additionally, we use the sklearn package to help perform the clustering.

# import necessary packages
import  numpy as np
import  matplotlib.pyplot as plt
from  sklearn.datasets         import                                  load_digits
from  sklearn.preprocessing          import                                  scale
from  sklearn.decomposition          import                                  PCA

# prepare dataset
X, y = load_digits(n_class=2, return_X_y=True)
X = scale(X)
X_train, X_test = ...

# define functions
def  initialize_cluster_mean(X, k):
# X: array of shape (n, d), each row is a d-dimensional data point
# k: number of clusters
# returns Y: array of shape (k, d), each row is the center of a cluster

def  assign_cluster(X, Y):
# X: array of shape (n, d), each row is a d-dimensional data point
# Y: array of shape (k, d), each row is the center of a cluster
# returns loss, the sum of squared distance from each point to its
assigned cluster





# Clustering

returns C: array of shape (n), each value is the index of the closest cluster

def update_cluster_mean(X, k, C):
# X: array of shape (n, d), each row is a d-dimensional data point
# k: number of clusters
# C: array of shape (n), each value is the index of the closest cluster
# returns Y: array of shape (k, d), each row is the center of a cluster

def k_means(X, k, max_iters=50, eps=1e-5):
Y = initialize_cluster_mean(X, k)

for i in range(max_iters):
loss, C = assign_cluster(X, Y)
Y = update_cluster_mean(X, k, Y)

if loss_change &#x3C; eps:
break

return loss, C, Y

def scatter_plot(X, C):
plt.figure(figsize=(12, 10))

k = int(C.max()) + 1
from itertools import cycle
colors = cycle('bgrcmk')

for i in range(k):
idx = (C == i)
plt.scatter(X[idx, 0], X[idx, 1], c=next(colors))
plt.show()

# run k-means algorithm and plot the result

loss, C, Y = k_means(X_train, 2)
low_dim = PCA(n_components=2).fit_transform(X_train)
scatter_plot(low_dim, C)

We start by importing outside packages.

from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

The load_digits() method loads the MNIST digits dataset, with around 180 data points per digit. The scale() method linearly scales each of the data points such that the mean is 0 and variance is 1. The PCA() method helps visualize the MNIST digits data points, which are 64-dimensional, in the Cartesian plane (i.e., R2). See the next Chapter 7 for details on this process.

Next we prepare the dataset by calling the load_digits() method.

X, y = load_digits(n_class=2, return_X_y=True)
X = scale(X)
X_train, X_test = ...

Notice that we discard the target array y because we are performing clustering, a type of unsupervised learning. If we were to perform supervised learning instead, we would need to make use of y.




# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

# K-Means Algorithm Functions

def initialize_cluster_mean(X, k):
return Y

def assign_cluster(X, Y):
return loss, C

def update_cluster_mean(X, k, C):
return Y

def k_means(X, k, max_iters=50, eps=1e-5):
Y = initialize_cluster_mean(X, k)

for i in range(max_iters):
loss, C = assign_cluster(X, Y)
Y = update_cluster_mean(X, k, Y)

if loss_change &#x3C; eps:
break

return loss, C, Y

In practice, it is common to limit the number of cluster update iterations (i.e., the parameter max_iters) and specify the smallest amount of loss change allowed for one iteration (i.e., the constant ϵ). By terminating the algorithm once either one of the conditions is reached, we can get an approximate solution within a reasonable amount of time.

# Helper Function for Plotting

def scatter_plot(X, C):
plt.figure(figsize=(12, 10))

k = int(C.max()) + 1
from itertools import cycle
colors = cycle('bgrcmk')

for i in range(k):
idx = (C == i)
plt.scatter(X[idx, 0], X[idx, 1], c=next(colors))
plt.show()

The cycle() method from the itertools package lets you iterate through an array indefinitely, with the index wrapping around back to the start, once it reaches the end of the array.

# Boolean Array Generation

Now, consider the for loop section in the helper function above. We first use Boolean conditions to concisely generate a new array.

idx = (C == i)

This generates a Boolean array with the same length as C, where each entry is either True/False based on whether the corresponding entry in C is equal to i. The following code is equivalent.

idx = np.zeros(C.size)
for j in range(C.size):


# Clustering

idx[j] = (C[j] == i)

We then use a technique called Boolean masking to extract a particular subset of rows of X.

X[idx, 0]

Notice that in place of a list of indices of rows to extract, we are indexing with the Boolean array we just defined. The code will extract only the rows where the Boolean value is True. For example, if the value of idx is [True, False, True], then the code above is equivalent to

X[[0, 2], 0]

Finally, we make use of the helper functions we defined earlier to run the k-means algorithm and plot results.

_, C, _ = k_means(X_train, 2)

low_dim = PCA(n_components=2).fit_transform(X_train)

scatter_plot(low_dim, C)

The first line of this code snippet shows how we can use the _ symbol to selectively disregard individual return values of a function call. The second line of code uses the PCA() method to transform the 64-dimensional data X_train into 2-dimensional data so that we can visualize it with the scatter_plot() method. We will learn the details of this process in the next Chapter 7.


NO_CONTENT_HERE



# 7 Low-Dimensional Representation

High-dimensional datasets arise in quite a few settings. This chapter concerns a phenomenon that arises frequently: the data points (i.e., vectors) collectively turn out to be “approximately low rank.” A running theme in this chapter is that arrays and matrices, which in introductory courses like COS 126 and COS 226 were thought of as data structures (i.e., an abstraction from programming languages), are treated now as objects that we can pass through some remarkable (but simple) mathematical procedures.

If a large dataset of N vectors in Rd has rank k, then we can think of a natural compression method. Let U be the k-dimensional subspace spanned by the vectors, and identify k basis vectors for U. For each of the N vectors, find the k coefficients of their representation in terms of the basis vectors. Following this method, instead of specifying the N vectors using Nd real numbers, we can represent them using k(N + d) real numbers, which is a big win if d is much larger than k.

| 6    | 4     | 1D0  |
| ---- | ----- | ---- |
| 2    | 0.75  |      |
| 0    |       |      |
| -2   | 0.50  |      |
| 4    | 0.25  |      |
| -6   |       |      |
| 5.0  | 7.5   |      |
| -8   | -6    | -4   |
| 0.0  | -0.25 |      |
| -2.5 | -0.50 |      |
| 2    | 4     | -5.0 |
| 6    | 8     | -7.5 |
|      | -1.00 |      |
|      | 1.00  |      |
|      | -0.75 |      |
|      | -0.50 |      |
|      | -0.25 |      |
|      | 0.00  |      |
|      | 0.25  |      |
|      | 0.50  |      |
|      | 0.75  |      |
|      | 1.0   |      |

Example 7.0.1. Figure 7.1 shows three vectors v1 = (3.42, −1.33, 6.94), v2 = (7.30, 8.84, 1.95), v3 = (−7.92, −6.37, −5.66) in R3. The three vectors have rank 2 — they are all in the 2-dimensional linear subspace generated.





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

by u1 = (8, 8, 4) and u2 = (1, −4, 6). Specifically,

v1 = 0.31u1 + 0.95u2

v2 = 0.95u1 − 0.31u2

v3 = −0.95u1 − 0.31u2

Therefore, we can represent these vectors in a 2-dimensional plane, as

v1 = (0.31, 0.95), v2 = (0.95, −0.31), v3 = (−0.95, −0.31)

# 7.1 Low-Dimensional Representation with Error

Of course, in general, high dimensional datasets are not exactly low rank. We’re interested in datasets which have low-dimension representations once we allow some error.

Definition 7.1.1 (Low-dimensional Representation with Error). We say a set of vectors v1, v2, . . . , vN ∈ Rd has rank k with mean-squared error ϵ if there exist some basis vectors u1, u2, . . . , uk ∈ Rd and N vectors v1, v2, . . . , vN ∈ span(u1, u2, . . . , uk) such that

1/N ∑i ||vi − vi||2 ≤ ϵ (7.1)

We say that v1, . . . , vN are the low-rank or low-dimensional approximation of v1, . . . , vN. Typically we will assume without loss of generality that the basis vectors are orthonormal (i.e., have ℓ2 norm equal to 1 and are pairwise orthogonal).

Definition 7.1.1 can be thought of as a lossy compression of the dataset of vectors since the low-dimensional representation of vectors is roughly correct, but with a bound of ϵ on the MSE. This compression view will be used in Section 7.3.

Figure 7.2: v1, v2, v3 ∈ R3 (left) and their 2-dimensional approximations v1, v2, v3 represented in the 2-dimensional subspace spanned by u1, u2.

| 6  | 4     | 1D0   |
| -- | ----- | ----- |
| 2  | 0.75  | 0     |
| -2 | 0.50  | 4     |
| -6 | 0.25  | -8    |
| -6 | -4    | 0.02  |
| -2 | -2.5  | -0.75 |
| 2  | 4     | -5.0  |
| 6  | 8     | -7.5  |
|    | -1.00 | 1.00  |
|    | -0.75 | -0.50 |
|    | -0.25 | 0.00  |
|    | 0.25  | 0.50  |
|    | 0.75  | 1D0   |





low-dimensional representation

Example 7.1.2. Figure 7.2 shows three vectors v1 = (3.42, −1.33, 6.94), v2 = (7.30, 8.84, 1.95), v3 = (−6.00, −7.69, −6.86) in R3. The three vectors have rank 2 with mean-squared error 2.5. If you choose the basis vectors u1 = (8, 8, 4), u2 = (1, −4, 6) and the low-rank approximations v1 = v1, v2 = v2, v3 = (−7.92, −6.37, −5.66) ∈ span(u1, u2) then,

1 ∑i vi − vi 2 ≃ 2.28 ≤ 2.5

3

Note that the basis vectors in this example are only orthogonal and not orthonormal, but it is easy to set them as orthonormal by normalizing them.

# Problem 7.1.3.

Show that if u1, u2, . . . , uk ∈ Rd is any set of orthonormal vectors and v ∈ Rd then the vector v in span(u1, u2, . . . , uk) that minimizes v − v2 is

2

k

∑(v · uj)uj (7.2)

j=1

(Hint: If α1, α2, . . . , αk minimize v − ∑j αjuj 2 then the gradient of this expression with respect to α1, α2, . . . , αk must be zero.)

Problem 7.1.3 illustrates how to find the low-dimensional representation of the vectors, once we specify the k basis vectors. Notice that (7.2) is the vector projection of v onto the subspace U spanned by the vectors u1, u2, . . . , uk. Therefore, the approximation error v − v2 is the squared norm of the component of v that is orthogonal to U.12 Also known as the vector rejection of v from U.

Problem 7.1.4 is only for more advanced students but all students should read its statement to understand the main point. It highlights how miraculous it is that real-life datasets have low-rank representations. It shows that generically one would expect ϵ in (7.1) to be 1 − k/n, which is almost 1 when k ≪ n. And yet in real life ϵ is small for fairly tiny k.

# Problem 7.1.4.

Suppose the vi’s are unit vectors2 and the vectors u1, u2, . . . , uk were the basis vectors of a random k-dimensional subspace in Rd. (That is, chosen without regard to the vi’s.) Heuristically argue that the ϵ one would need in (7.1) would be 1 − k/n.

2 Note: the maximum possible value of ϵ when vi’s are unit vectors is 1. Convince yourself!

# 7.1.1   Computing the Low-Dimensional Representation with Error

In Problem 7.1.3, we have already seen how to find the low-dimension representation with error, once we are given the basis vectors. All there remains is to identify a suitable value of k and find the corresponding basis vectors that will minimize the error.

There is a simple linear algebraic procedure, the Singular Value Decomposition (SVD). Given a set of vectors vi and a positive integer





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

k, SVD can output the best orthonormal basis in sense of Definition 7.1.1 that has the lowest possible value of ϵ. In practice, we treat k as a hyperparameter and use trial and error to find the most suitable k. Problem 7.1.5 shows that the accuracy of the low-dimensional representation will decrease when we choose a smaller number of dimensions. So we are making a choice between the accuracy of the representations against how condensed our compression is.

# Problem 7.1.5

Show that as we decrease k in Definition 7.1.1, the corresponding ϵ can only increase (i.e., cannot decrease).

Formally, SVD takes a matrix as its input; the rows of this matrix are the vector vi’s. The procedure operates on this matrix to output a low-rank approximation. We discuss details in Section 20.3. To follow the rest of this chapter, you do not need to understand details of the procedure. You just need to remember the fact that the best k-dimensional representation is computable for each k. In practice, programming languages have packages that will do the calculations for you. Below is a Python code snippet that will calculate the SVD.

import sklearn.decomposition.TruncatedSVD

# n * n matrix
data = ...
# prepare transform on dataset matrix "data"
svd = TruncatedSVD(n_components=k)
svd.fit(data)
# apply transform to dataset and output an n * k matrix
transformed = svd.transform(data)

Now we see some fun applications.

# 7.2 Application 1: Stylometry

In many cases in old literature, the identity of the author is disputed. For instance, the King James Bible (i.e., the canonical English bible from the 17th century) was written by a team whose identities and work divisions are not completely known. Similarly the Federalist Papers, an important series of papers explicating finer points of the US government and constitution, were published in the early days of the republic with the team of authors listed as Alexander Hamilton, James Madison, and John Jay. But it was not revealed which paper was written by whom. In such cases, can machine learning help identify who wrote what?

Here we present a fun example about the books in the Wizard of Oz series. 3 L. Frank Baum was the author of the original Wonderful Wizard of Oz, which was a best-seller in its day and remains highly popular to this day. The publisher saw a money-making opportunity.

Original paper at http://dh.obdurodon.org/Binongo-Chance.pdf.

A survey paper by Erica Klarreich in Science News Dec 2003: Statistical tests are unraveling knotty literary mysteries at http://web.mit.edu/allanmc/www/stylometrics.pdf.





# low-dimensional representation

and convinced Baum to also write 15 follow-up books. After Baum’s death the publisher managed to pass on the franchise to Ruth Plumly Thompson, who wrote many other books. However, the last of the Baum books, Royal Book of Oz (RBOO), always seemed to Oz readers closer in style to Thompson’s books than to Baum’s. But with all the principals in the story now dead, there seemed to be no way to confirm the suspicion. Now we describe how simple machine learning showed pretty definitively that this book was indeed written by Ruth Plumly Thompson. The main idea is to represent the books vectors in some way and then find their low-rank representations.

The key idea is that different authors use English words at different frequencies. Surprisingly, the greatest difference lies in frequencies of function words such as with, however, upon, rather than fancy vocabulary words (the ones found on your SAT exam).

# Example 7.2.1.

Turns out Alexander Hamilton used upon about 10 times more frequently than James Madison. We know this from analyzing their individual writing outside their collaboration on the Federalist Papers. Using these kinds of statistics, it has been determined that Hamilton was the principal author or even the sole author of almost all of the Federalist Papers.

The statistical analysis of the Oz books consisted of looking at the frequencies of 50 function words. All Oz books except RBOO were divided into text blocks of 5000 words each. For each text block, the frequency (i.e., number of occurrences) of each function word was computed, which allows us to represent the block as a vector in R50. There were 223 text blocks total, so we obtain 223 vectors in R50.

| the (6.7%)        | with (0.7%)       | up (0.3%)    | into (0.2%)  | just (0.2%)   |
| ----------------- | ----------------- | ------------ | ------------ | ------------- |
| and (3.7%)        | but (0.7%)        | no (0.3%)    | now (0.2%)   | very (0.2%)   |
| to (2.6%)         | for (0.7%)        | out (0.3%)   | down (0.2%)  | where (0.2%)  |
| a/an (2.3%)       | at (0.6%)         | what (0.3%)  | over (0.2%)  | before (0.2%) |
| of (2.1%)         | this/these (0.5%) | then (0.3%)  | back (0.2%)  | upon (0.1%)   |
| in (1.3%)         | so (0.5%)         | if (0.3%)    | or (0.2%)    | about (0.1%)  |
| that/those (1.0%) | all (0.5%)        | there (0.3%) | well (0.2%)  | after (0.1%)  |
| it (1.0%)         | on (0.5%)         | by (0.3%)    | which (0.2%) | more (0.1%)   |
| not (0.9%)        | from (0.4%)       | who (0.3%)   | how (0.2%)   | why (0.1%)    |
| as (0.7%)         | one/ones (0.3%)   | when (0.2%)  | here (0.2%)  | some (0.1%)   |

Then we compute a rank 2 approximation of these 223 vectors. Figure 7.5 shows the scatter plot in the 2-dimensional visualization. The two axes correspond to the two basis vectors we found for the rank 2 approximation. It becomes quickly clear that the vectors from the Baum books are in a different part of the space than those from the Thompson books. It is also clear that RBOO vectors fall in the





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

# Figure 7.5: Rank-2 visualization of the text block vectors from books of Glinda Oz.

The dots on the left correspond to vectors from Oz books known to be written by Ruth Plumly Thompson. The hearts on the left correspond to vectors from RBOO. The ones on the right correspond to ones written by L. Frank Baum. Figure from Binongo’s paper.

# Conclusion

Ruth Plumly Thompson was the true author of Royal Book of Oz! By the way, if one takes the non-Oz writings of Baum and Thompson and plot their vectors in the 2D-visualization in Figure 7.6, they also fall on the appropriate side. So the difference in writing style came across clearly even in non-Oz books!

# Figure 7.6: Rank-2 visualization of text block vectors from books written by Baum and Thompson outside of the Oz series.

Figure from Binongo’s paper.


# 7.3 Application 2: Eigenfaces

This section uses the lossy compression viewpoint of low-rank representations. As you may remember from earlier computer science courses (e.g., Seam Carver from COS 226), images are vectors of pixel values. In this section, let us only consider grayscale (i.e., B&#x26;W) images where each pixel has an integer value in [−127, 127]. −127 corresponds to the pixel being pitch black; 0 corresponds to middle gray; and 127 corresponds to total white. We can also reorganize the entries to form a single vector:

a11    a12     · · ·    a1n
a21    a22     · · ·    a2n  → (a11, a12, · · ·, a1n, a21, a22, · · ·, a2n, · · ·, am1, am2, · · ·, amn)
.      .                .
.      .                .
.      .                .
am1    am2     · · ·    amn

Once we have this vectorized form of an image, it is possible to perform linear algebraic operations on the vectors. For example, we can take 0.3 times the first image and add it to −0.8 times the second image, etc. See Figure 7.7 for some of these examples.

Figure 7.7: Example of linear algebra on images.

Eigenfaces was an idea for face recognition4. The dataset in this lecture is from a classic Olivetti dataset from 1990s. Researchers took images of people facing the camera in good light, downsized to 64 × 64 pixels. This makes them vectors in R4096. Now we can find a 64-rank approximation of the vectors using procedures we will explore in more detail in Section 20.3.

Figure 7.8 shows four basis vectors in the low-rank approximation of the images.

The first image looks like a generic human with a ill-defined nose and lips; the second image looks like having glasses and a wider nose; the third image potentially looks like a female face; the fourth image looks like having glasses, a moustache, and a beard. All images in the dataset can be approximated as a linear combination of these basis vectors.

| 0   | 0   |
| --- | --- |
| 100 | 100 |
| 200 | 200 |
| 300 | 300 |
| 400 | 400 |
| 500 | 500 |

0.5v1 + 0.5v2     0.8v1 + 0.2v2     0.4v1 - 0.6v2

4 L. Sirovich; M. Kirby (1987). Low-dimensional procedure for the characterization of human faces. Journal of the Optical Society of America.



# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

# Figure 7.8

Some basis vectors (which turn out to be face-like) in the low-rank approximation of the Olivetti dataset.

# Figure 7.9

4 original images in the Olivetti dataset (left), compared with their 64-rank approximations (middle) and 128-rank approximations (right).

| Original Images | Reconstructed Images | Reconstructed Images |
| --------------- | -------------------- | -------------------- |
|                 |                      |                      |

From Figure 7.9, we also see that the approximations are more accurate when the corresponding value of k is larger. In fact, Figure 7.10 shows the average value of ∥vⁱ−vⁱ∥² as a function of the rank of the approximation. Note that this value roughly represents the fraction of v lost in the approximation. It can be seen that the error is a decreasing function in terms of k. However, note that doing machine learning — specifically face recognition — on low-rank representations is computationally more efficient particularly because the images are compressed to a lower dimension. With a smaller value of k, we can improve the speed of the learning.

# Figure 7.10

What fraction of norm of the image is not captured in the low-dimensional representation, plotted versus the rank k.

| Average | lvi vill2 | ll vil2 |
| ------- | --------- | ------- |
| 0.008   | 0.006     | 0.004   |
| 0.002   | 0.000     |         |

k (number of dimensions in approximation)



# 8 n-Gram Language Models

In this chapter, we continue our investigation into unsupervised learning techniques and now turn our attention to language models. You may have heard of natural language processing (NLP) and models such as GPT-3 in the news lately. The latter is quite impressive, being able to write and publish its own opinion article on a reputable news website!1 While most of these models are trained using state-of-the-art deep learning techniques which we will discuss later on in this text, this chapter explores a key idea, which is to view language as the output of a probabilistic process, which leads to an interesting measure of the “goodness” of the model. Specifically, we will investigate the so-called n-gram language model.

# 8.1 Probabilistic Model of Language

Classical linguistics focused on the syntax or the formal grammar of languages. The linguists believed that a language can be modeled by a set of sentences, constructed from a finite set of vocabularies and a finite set of grammatical rules. But this approach in language modeling had limited success in machine learning.

Instead, the approach of machine learning in language models,





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

Pioneered by Claude Shannon, has been to learn the distribution of pieces of text. In other words, the model assigns a probability to all conceivable finite pieces of English text (even those that have not yet been spoken or written). For example, the sentence “how can I help you” will be assigned some probability, most likely larger than the probability assigned to the sentence “can I how you help.” Note that we don’t expect to find a “correct” model; all models found to date are approximations. But even an approximate probabilistic model can have interesting uses, such as the following:

1. Speech recognition: A machine processes a recording of a human speech that sounds somewhere between “I ate a cherry” and “eye eight a Jerry.” If the model assigns a higher probability score to the former, speech recognition can still work in this instance.
2. Machine translation: “High winds tonight” should be considered a better translation than “large winds tonight.”
3. Context sensitive spelling correction: We can compare the probabilities of sentences that are similar to the following sentence — “Their are problems wit this sentence.” — and output the corrected version of the sentence.
4. Sentence completion: We can compare the probabilities of sentences that will complete the following phrase — “Please turn off your ...” — and output the one with the highest probability.

# 8.2 n-Gram Models

Say we are in the middle of the process of assigning a probability distribution over all English sentences of length 5. We want to find the probability of the sentence “I love you so much.” If we let Xi be the random variable that represents the value of the i-th word, the probability we are looking for is the joint probability

Pr[X1 = “I”, X2 = “love”, X3 = “you”, X4 = “so”, X5 = “much”] (8.1)

By the Chain Rule, we can split this joint probability into the product of a marginal probability and four conditional probabilities:

(8.1) = Pr[X1 = “I”] (8.2)

× Pr[X2 = “love” | X1 = “I”]

× Pr[X3 = “you” | X1 = “I”, X2 = “love”]

× Pr[X4 = “so” | X1 = “I”, X2 = “love”, X3 = “you”]

× Pr[X5 = “much” | X1 = “I”, X2 = “love”, X3 = “you”, X4 = “so”]





# n-gram language models

If we estimate all components of the product in (8.2), we will be able to estimate the joint probability (8.1).

Now consider the bigram model, which has the following two assumptions:

1. The probability of a word is only dependent on the immediately previous word.
2. That probability does not depend on the position of the word in the sentence.

The first assumption says that, for example, the conditional probability

Pr[X3 = "you" | X1 = "I", X2 = "love"]

can be simplified as

Pr[X3 = "you" | X2 = "love"]

The second assumption says that

Pr[X3 = "you" | X2 = "love"] = Pr[Xi+1 = "you" | Xi = "love"] for any 1 ≤ i ≤ 4.

We abuse notation and denote any of these probabilities as Pr["you" | "love"].

Applying these assumptions to (8.2), we can simplify it as

(8.1) = Pr["I"] × Pr["love" | "I"] × Pr["you" | "love"] × Pr["so" | "you"] × Pr["much" | "so"] (8.3)

Now we are going to estimate each component of (8.3) from a large corpus of text. The estimation for the marginal probability of the word "I" is given as

Pr["I"] ≈ Count("I") (8.4)

total number of words

where Count refers to the number of occurrences of the word in the text. In other words, this is the proportion of the occurrence of the word "I" in the entire corpus. Similarly, we can estimate the conditional probability of the word "love" given its previous word is "I" as

Pr["love" | "I"] ≈ Count("I love") (8.5)

∑ Count("I" + w)

w

where in the denominator, we sum over all possible words in the dictionary. This is the proportion of the word "love" occurring immediately after the word "I" out of every time some word w in the dictionary occurring immediately after the word "I." 2

Notice that there is no word occurring immediately after the word "I" when "I" is at the end of the sentence in the training corpus. Therefore, the denominator in (8.5) is equal to the Count of "I" minus the Count of "I" at the end of a sentence. This is not necessarily the case when we introduce the sentence stop tokens in Section 8.3.





# 92

# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

can estimate the following conditional probability as

Pr[wi+1 | wi] ≈ Count(wiwi+1)                          (8.6)

∑ Count(wiw)

w

where wi is the i-th word of the sentence. Once we calculate these estimates from the corpus, we are able to define the probability of the sentence "I love you so much."

# 8.2.1 Defining n-Gram Probabilities

We can extend the example above to a more general setting. Now we want to define the probability distribution over all sentences of length k (grammatical or not). Say we want to find the joint probability of the sentence w1w2 . . . wk where wi is the i-th word of the sentence. We will employ an n-gram model which has two assumptions:

1. The probability of a word is only dependent on the immediately previous n − 1 words. 3
2. That probability does not depend on the position of the word in the sentence.

3 If n = 1, the model is called a unigram model, and the probability is not dependent on any previous word. When n = 2 and n = 3, the model is respectively called a bigram and a trigram model.

By a similar logic from the earlier example, we abuse notation and denote the joint probability of the sentence w1w2 · · · wk as Pr[w1w2 . . . wk]; the marginal probability of the first word being w1 as Pr[w1]; and so on. We can apply the Chain Rule again to define the n-gram model.

# Definition 8.2.1 (n-Gram Model)

An n-gram model assigns the following probability to the sentence w1w2 . . . wk if n > 1: 4

4 max(1, i − n + 1) in the third line is to ensure that we access the correct indices for the first n − 1 words, where there are less than n − 1 previous words to look at.

Pr[w1w2 . . . wk] = Pr[w1] Pr[w2 | w1] · · · Pr[wk | w1w2 . . . wk−1]

= Pr[w1] × ∏ Pr[wi | w1 . . . wi−1]                         (8.7)

i=2

= Pr[w1] × ∏ Pr[wi | wmax(1,i−n+1) . . . wi−1]

i=2

and the following probability if n = 1:

k

Pr[w1w2 . . . wk] = ∏  Pr[wi]                         (8.8)

i=1

where the n-gram probabilities are estimated from a training corpus as the following

Pr[wi] ≈             Count(wi)

total number of words

Pr[wj | wi . . . wj−1] ≈    Count(wi . . . wj−1wj)

∑ Count(wi . . . wj−1w)

w



# n-gram language models

This defines the “best” possible probabilistic model in terms of the Maximum Likelihood Principle from Subsection 4.2.1.5 We now turn to the following example.

# Example 8.2.2

We investigate a cowperson language which has two words in the dictionary: {Yee, Haw}. Suppose the training corpus is given as “Yee Haw Haw Yee Yee Yee Haw Yee.” Then the unigram probabilities can be estimated as

Pr[”Yee”] = 5
Pr[”Haw”] = 3

# Table 8.1: Bigram frequency table of the cowperson language.

| previous | next  |       | Total |   |
| -------- | ----- | ----- | ----- | - |
|          | “Yee” | “Haw” |       |   |
| “Yee”    | 2     | 2     | 4     |   |
| “Haw”    | 2     | 1     | 3     |   |

# Table 8.2: Bigram probability table of the cowperson language.

| previous | next  |       | Total |   |
| -------- | ----- | ----- | ----- | - |
|          | “Yee” | “Haw” |       |   |
| “Yee”    | 2/4   | 2/4   | 1     |   |
| “Haw”    | 2/3   | 1/3   | 1     |   |

From Table 8.2, we get the following bigram probabilities:

Pr[”Yee” | ”Yee”] = 2/4
Pr[”Haw” | ”Yee”] = 2/4
Pr[”Yee” | ”Haw”] = 2/3
Pr[”Haw” | ”Haw”] = 1/3

Then by the bigram model, the probability that we see the sentence “Yee Haw Yee” out of all sentences of length 3 can be calculated as

Pr[”Yee”] × Pr[”Haw” | ”Yee”] × Pr[”Yee” | ”Haw”] = 5 × 2/4 × 2/3 ≃ 0.21

# 8.2.2 Maximum Likelihood Principle

Recall the Maximum Likelihood Principle introduced in Subsection 4.2.1. It gave a way to measure the “goodness” of a model with probabilistic outputs. Now we formally prove that the estimation methods given in Definition 8.2.1 satisfy the Maximum Likelihood Principle for the n = 1 case. A probabilistic model is “better” than another if it assigns more probability to the actual outcome. Here, the actual outcome is the training corpus, which also consists of words. So let us denote




94   introduction to machine learning lecture notes for cos 324 at princeton university

The training corpus as a string of words w1w2 . . . wT. By definition, a unigram model will assign the probability

Pr[w1w2 . . . wT] = ∏ Pr[wi]

i=1

to this string. Remember that each of the wi’s are a member of a finite set of dictionary words. If we let V be the size of the dictionary, then the model is defined by the choice of V values, the probabilities we assign to each of the dictionary words. Let pi be the probability that we assign to the i-th dictionary word, and let ni be the number of times that the i-th dictionary word appears in the training corpus. Then the above equation can be rewritten as

Pr[w1w2 . . . wT] = ∏ pini

i=1

We want to maximize this value under the constraint ∑i=1V pi = 1. A solution to this type of a problem can be found via the Lagrange multiplier method. We will illustrate with an example.

# Example 8.2.3

We revisit the cowperson language from Example 8.2.2. Here V = 2 and T = 8. Let p1 = Pr[“Yee”] and p2 = Pr[“Haw”]. Then the probability assigned to the training corpus by the unigram model is

Pr[“Yee Haw Haw Yee Yee Yee Haw Yee”] = p15 p23

We want to maximize this value under the constraint p1 + p2 = 1. Therefore, we want to find the point where the gradient of the following is zero.

f (p1, p2) = p15 p23 + λ(p1 + p2 − 1)

for some λ. The gradients are given as

∂f/∂p1 = 5p14 p23 + λ

∂f/∂p2 = 3p15 p22 + λ

From 5p14 p23 + λ = 3p15 p22 + λ = 0, we get p1 = 5. Combined with the fact that p1 + p2 = 1, we get the optimal solution p1 = 8 and p2 = 8.

# Problem 8.2.4

Following the same Lagrange multiplier method as in Example 8.2.3, verify that the heuristic solution pi = ni (the empirical frequency) is the optimal solution that maximizes the above equation under the constraint ∑i=1V pi = 1.

# 8.3 Start and Stop Tokens

In this section, we present a convention that is often useful: start token ⟨s⟩ and stop token ⟨/s⟩. They signify the start and the end




# n-gram language models

of each sentence in the training corpus. They are a special type of vocabulary item that will be augmented to the dictionary, so you will want to pay close attention to the way they contribute to the vocabulary size, number of words, and the n-gram probabilities. Also, by introducing these tokens, we are able to define a probability distribution over all sentences of finite length, not just a given length of k. For the sake of exposition, we will only consider the bigram model for most parts of this section.

# 8.3.1  Re-estimating Bigram Probabilities

Consider the cowperson language again.

# Example 8.3.1.

The training corpus “Yee Haw Haw Yee Yee Yee Haw Yee” actually consists of three different sentences: (1) "Yee Haw," (2) "Haw Yee Yee," and (3) "Yee Haw Yee." We can append the start and stop tokens to the corpus and transform it into

⟨s⟩ Yee Haw ⟨/s⟩
⟨s⟩  Haw Yee Yee ⟨/s⟩
⟨s⟩  Yee Haw Yee ⟨/s⟩

With these start and stop tokens in mind, we slightly relax the Assumption 2 of the n-gram model and investigate the probability of a word w being the first or the last word of a sentence, separately from other probabilities. We will denote these probabilities respectively as Pr[w | ⟨s⟩] and Pr[⟨/s⟩ | w]. The former probability will be estimated as

Pr[w | ⟨s⟩] ≈ Count(⟨s⟩ w)

total number of sentences

which is the proportion of sentences that start with the word w in the corpus. The latter probability is estimated as

Pr[⟨/s⟩ | w] ≈ Count(w ⟨/s⟩)

Count(w)

which is the proportion of the occurrence of w that is at the end of a sentence in the corpus. Also, notice that other bigram probabilities are also affected when introducing the stop tokens. In (8.6), the denominator originally did not include the occurrence of the substring at the end of the sentence because there was no word to follow that substring. However, if we consider ⟨/s⟩ as a word in the dictionary, the denominator can now include the case where the substring is at the end of the sentence. Therefore, the denominator is just equivalent to the Count of the substring in the corpus. Therefore, the bigram probabilities after introducing start, stop tokens can be estimated instead as

6 If we consider ⟨s⟩ , ⟨/s⟩ as vocabularies of the dictionary, (8.13) can also include (8.11), (8.12).



# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

Pr[wj | wj−1] ≈ Count(wj−1wj) (8.13)

# Example 8.3.2

We revisit Example 8.2.2. The bigram frequency table and the bigram probability table can be recalculated as in Table 8.3 and Table 8.4.

Note that the values in the Total column now correspond to the unigram count of that word.

| previous | next | “Yee” | “Haw” | ⟨/s⟩ | Total |   |
| -------- | ---- | ----- | ----- | ---- | ----- | - |
| ⟨s⟩      |      | 2     | 1     | 0    | 3     |   |
| “Yee”    |      | 1     | 2     | 2    | 5     |   |
| “Haw”    |      | 2     | 0     | 1    | 3     |   |

| previous | next | “Yee” | “Haw” | ⟨/s⟩ | Total |   |
| -------- | ---- | ----- | ----- | ---- | ----- | - |
| ⟨s⟩      |      | 2/3   | 1/3   | 0/3  | 1     |   |
| “Yee”    |      | 1/5   | 2/5   | 2/5  | 1     |   |
| “Haw”    |      | 2/3   | 0/3   | 1/3  | 1     |   |

Therefore, the bigram probabilities of the cowperson language, once we introduce the start and stop tokens, are given as:

Pr[”Yee” | ⟨s⟩] = 2/3     Pr[”Haw” | ⟨s⟩] = 1/3

Pr[”Yee” | ”Yee”] = 1/5     Pr[”Haw” | ”Yee”] = 2/5     Pr[⟨/s⟩ | ”Yee”] = 2/5

Pr[”Yee” | ”Haw”] = 2/3     Pr[”Haw” | ”Haw”] = 0     Pr[⟨/s⟩ | ”Haw”] = 1/3

# 8.3.2 Redefining the Probability of a Sentence

The biggest advantage of introducing stop tokens is that now we can assign a probability distribution over all sentences of finite length, not just a given length k. Say we want to assign a probability to the sentence w1w2 . . . wk (without the start and stop tokens). By introducing start and stop tokens, we can interpret this as the probability of w0w1 . . . wk+1 where w0 = ⟨s⟩ and wk+1 = ⟨/s⟩. Following the similar logic from (8.2), we can define this probability by the Chain Rule.

# Definition 8.3.3 (Bigram Model with Start, Stop Tokens)

A bigram model, once augmented with start, stop tokens, assigns the following probability to a sentence w1w2 . . . wk:

Pr[w1w2 . . . wk ] = ∏i=1k+1 Pr[wi | wi−1] (8.14)

where the bigram probabilities are estimated as in (8.13).



n-gram language models 97

# Example 8.3.4.

The probability that we see the sentence “Yee Haw Yee” in the cowperson language can be calculated as

Pr[”Yee” | ⟨s⟩] × Pr[”Haw” | ”Yee”] × Pr[”Yee” | ”Haw”] × Pr[⟨/s⟩ | ”Yee”]

= 2 × 2 × 2 × 2 ≃ 0.07

3 5 3 5

Note that this probability is taken over all sentences of finite length.

# Problem 8.3.5.

Verify that (8.14) defines a probability distribution over all sentences of finite length.

# 8.3.3   Beyond Bigram Models

In general, if we have an n-gram model, then we may need to introduce more than 1 start or stop tokens. For example, in a trigram model, we will need to define the probability that the word is the first word of the sentence as Pr[w | ⟨s⟩ ⟨s⟩]. Based on the number of start and stop tokens introduced, the n-gram probabilities will need to be adjusted accordingly.

# 8.4     Testing a Language Model

So far, we discussed how to define an n-gram language model given a corpus. This is analogous to training a model given a training dataset. Naturally, the next step is to test the model on a newly seen held-out data to ensure that the model generalizes well. In this section, we discuss how to test a language model.

# 8.4.1   Shakespeare Text Production

First consider a bigram text generator — an application of the bigram model. The algorithm initiates with the start token ⟨s⟩. It then outputs a random word w1 from the dictionary, according to the probability Pr[w1 | ⟨s⟩]. It then outputs the second random word w2 from the dictionary, according to the probability Pr[w2 | w1]. It repeats this process until the newly generated word is the stop token ⟨/s⟩. The final output of the algorithm will be the concatenated string of all outputted words.

It is possible to define a text generator for any n-gram model in general. Figure 8.4 shows the output of the unigram, bigram, trigram, quadrigram text generators when the models were trained on all Shakespeare texts.

Notice the sentence “I will go seek the traitor Gloucester.” in the output of the quadrigram text generator. This exact line appears in King Lear, Act 3 Scene 7. This is not a coincidence. Figure 8.5 presents





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

# Figure 8.3: An example run of the bigram text generator.

I want

want to

to eat

eat tasty

tasty food

I want to eat tasty food

# Unigram

To him swallowed confess hear both. Which. Of save on trail for are ay device and rote life have

Every enter now severally so, let

Hill he late speaks; or! a more to leg less first you enter

Are where exeunt and sighs have rise excellency took of.. Sleep knave we. near; vile like

# Bigram

What means, sir. I confess she? then all sorts, he is trim, captain.

Why dost stand forth thy canopy, forsooth; he is this palpable hit the King Henry. Live king. Follow.

What we, hath got so she that I rest and sent to scold and nature bankrupt, nor the first gentleman?

# Trigram

Sweet prince, Falstaff shall die. Harry of Monmouth's grave.

This shall forbid it should be branded, if renown made it empty.

Indeed the duke; and had a very good friend.

Fly, and will rid me these news of price. Therefore the sadness of parting, as they say, 'tis done.

# Quadrigram

King Henry. What! I will go seek the traitor Gloucester. Exeunt some of the watch. A great banquet serv'd in;

Will you not tell me who I am?

It cannot be but so.

Indeed the short and the long. Marry, 'tis a noble Lepidus.

The snapshot of the bigram, trigram, and quadrigram text generators once they have outputted the phrase “go seek the.” You can see that bigram models and trigram models assign very small probabilities to the word “traitor” because there are many more instances of phrases “the” or “seek the” in the corpus than “go seek the.” On the other hand, the quadrigram model assigns a very large probability to the word “traitor” because there is only a limited number of times that the phrase “go seek the” appears in the corpus.

# Figure 8.5: The probability of the next word given the previous three words are “go seek the” in the n-gram model, where n = 2, 3, 4.

|     | n=2  | n=3  | n=4     |      |       |
| --- | ---- | ---- | ------- | ---- | ----- |
| 0.5 |      |      |         |      |       |
| 0.4 |      |      |         |      |       |
| 0.3 |      |      |         |      |       |
| 0.2 | LLI  |      |         |      |       |
| 0.1 |      |      |         |      |       |
| 0.0 | king | duke | traitor | lion | world |





# n-gram language models

phrase “go seek the,” the problem is even worse. As can be seen in Figure 8.6, the quadrigram model assigns probability of 1 to the word “Gloucester” meaning that the phrase “seek the traitor” only appears before the word “Gloucester.” So the model has memorized one completion of the phrase from the training text. From this example, we can see that text production based on n-grams is sampling and remixing text fragments seen in the training corpus.

# Figure 8.6: The probability of the next word given the previous three words are “seek the traitor.”

| 1.0 |            |        |         | n=1      |    |
| --- | ---------- | ------ | ------- | -------- | -- |
| 0.8 |            |        |         | n=2      |    |
| 0.6 |            |        |         | n=3      |    |
| 0.4 |            |        |         | n=4      |    |
| 0.2 |            |        |         |          |    |
| 0.0 | I          | in     | and     | the      |    |
|     | Gloucester | stands | breathe | murderer | to |

The Shakespeare corpus consists of N = 884, 647 words and V = 29, 066 distinct words from the dictionary. There are about V2 ≈ 845 million possible combinations of bigrams, but Shakespeare only used around 300, 000 of them in his text. So 99.96% of the possible bigrams were never seen. The percentage is much higher for quadrigrams! Furthermore, for the quadrigrams that do appear in the corpus, most do not even repeat. Thus what comes out of the quadrigram model looks like Shakespeare because it is a memorized fragment of Shakespeare.9

9 Do this remind you of overfitting?

# 8.4.2 Perplexity

Having described a way to train a simple language model, we now turn our attention to a formal way of testing10 a language model. Just like any other model in ML, a language model will be given a corpus w1w2 . . . wT. Then we can assess the performance of the model by its perplexity on the corpus.

Definition 8.4.1 (Perplexity). The perplexity of a language model on the corpus w1w2 . . . wT is defined as

Pr[w1w2 . . . wT]-1 = T1 (8.15)

Note that, perplexity is defined for any probabilistic language model: the Chain Rule of joint probability applies to every model.





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

and does not require the n-gram assumptions. That is, 11 Assume for now that start and stop tokens do not exist in the corpus.

Pr[w1w2 . . . wT] = Pr[w1] × ∏i=2 Pr[wi | w1 . . . wi−1]

Then the perplexity of the model can be rewritten as

1  T    1

T Pr[w1] × ∏i=2 Pr[wi | w1 . . . wi−1] (8.16)

# Example 8.4.2

Consider the uniform (“clueless”) model which assumes that the probability of all words are equal in any given situation. That is, if V is the vocabulary size (i.e., size of the dictionary),

Pr[wi] = Pr[wi | w1 . . . wi−1] =   1 / V

for any given w1, . . . , wi ∈ V. This model assigns 1 / V to every sequence of T words, including the corpus. Therefore, the perplexity of the model is

T − 1

1 T = V

Now we try to understand perplexity at an intuitive level. (8.16) is the geometric mean 12 of the following T values:

1 , 1 , . . . , 1

Pr[w1] Pr[w2 | w1] Pr[wT | w1 . . . wT−1]

Now note that a probabilistic model splits the total probability of 1 to fractions and distributes them to the potential options for the next word. So the inverse of an assigned probability for a word can be thought roughly as the number of choices the model considered for the next word. With this viewpoint, perplexity as written in (8.16) means: how much has the model narrowed down the number of choices for the next word on average? The clueless model had not narrowed down the possibilities at all and had the worst-possible perplexity equal to the number of vocabulary words.

# Example 8.4.3

Consider a well-trained language model. At any given place of text, it can identify a set of 20 words and assigns probability 1 / 20 to each of them to be the next word. It happens that the next word is always one of the 20 words that the model identifies. The perplexity of the model is

T − 1

1 T = 20

Interestingly enough, the true perplexity of English is believed to be between 15 and 20. That is, if at an “average” place in text, you ask humans to predict the next word, then they are able to narrow down the list of potential next words to around 15 to 20 words.

13 The perplexity of state of the art language models is under 20 as well.





# 8.4.3  Perplexity on Test Corpus

The perplexity of a language model is analogous to a loss of an ML model.14 Similar to ML models we have been studying so far, it is possible to define a train perplexity and a test perplexity. The “goodness” of the model will be defined by how low the perplexity was on a previously unseen, held-out data.

For example, when n-gram models are trained on 38 million words and tested on 1.5 million words from Wall Street Journal articles, they show the following test perplexities in Table 8.5.15 Note that the state-of-the-art deep learning models achieve a test perplexity of around 20 on the same corpus.

| Unigram | Bigram | Trigram |
| ------- | ------ | ------- |
| 962     | 170    | 109     |

# 8.4.4   Perplexity With Start and Stop Tokens

When start and stop tokens are introduced to a corpus, we also need to redefine how to calculate the perplexity of the model. Again, we will only focus on a bigram model for the sake of exposition.

Say the corpus consists of t sentences:

⟨s⟩ w1,1 w1,2, . . . , w1,T₁ ⟨/s⟩
⟨s⟩ w2,1 w2,2, . . . , w2,T₂ ⟨/s⟩
.
.
.
⟨s⟩ wt,1wt,2, . . . , wt,Tₜ ⟨/s⟩

The probability of the corpus w1,1 w1,2 . . . wt,Tₜ is redefined as the product of the probability of each of the sentences:

t
Pr[w1,1 w1,2 . . . wt,Tₜ ] = ∏ Pr[wi,1wi,2 . . . wi,Ti]
i=1
t                                          Ti+1
= ∏ ∏ Pr[wi,j | wi,j−1]             (8.17)
i=1                                         j=1

Now we apply the interpretation of the perplexity that it is the geometric mean of probabilities of each word. Notice that we multiplied t (T + 1) probabilities to calculate the probability of the corpus.

If we let T = ∑i=1t Ti denote the total number of words (excluding start and stop tokens) of the corpus, the number of probabilities we multiplied can be written as T* = T + t.16

16 This can also be thought as adding the number of stop tokens to the number of words in the corpus.





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

# 8.4.4 Perplexity with Start, Stop Tokens

The perplexity of a bigram model with start, stop tokens can be redefined as

T*
t
Ti+1
1

∏ ∏ Pr[wi,j | wi,j−1]

i=1
j=1

# 8.4.5 Smoothing

One big problem with our naive definition of the perplexity of a model is that it does not account for a zero denominator. That is, if the model assigns probability exactly 0 to the corpus, then the perplexity of the model will be ∞!

Mathematically, it is undefined, but here assume that the result is a positive infinity that is larger than any real number.

Example 8.4.5. Suppose the phrase “green cream” never appeared in the training corpus, but the test corpus contains the sentence “You like green cream.” Then a bigram model will have a perplexity of ∞ because it assigns probability 0 to the bigram “green cream.”

To address this issue, we generally apply smoothing techniques, which never allow the model to output a zero probability. By reducing the naive estimate of seen events and increasing the naive estimate of unseen events, we can always assign nonzero probabilities to previously unseen events.

The most commonly used smoothing technique is the 1-add smoothing (a.k.a, Laplace smoothing). We describe how the smoothing works for a bigram model. The main idea of the 1-add smoothing can be summarized as “add 1 to all bigram counts in the bigram frequency table.” Then the bigram probability as defined in Definition 8.2.1 can be redefined as

Pr[wj | wj−1] ≈ Count(wj−1wj) + 1 = Count(wj−1wj) + 1

∑ (Count(wj−1w) + 1)

∑ (Count(wj−1w)) + V

w

w

where V is the size of the dictionary. If we had augmented the corpus with the start and the stop tokens, the denominator in (8.19) is just equal to Count(wj−1) + V*

V* = V + 1 is the size of the dictionary after adding the start and the stop tokens. It is customary to add only one to the vocabulary count. It may help to look at the number of rows and columns in the bigram frequency table.

# 8.4.6 Example

Recall the cowperson language with the start and stop tokens from Example 8.3.2. Upon further research, it turns out the language actually consists of three words: {Yee, Haw, Moo}, but the training corpus




# n-gram language models

“Yee Haw Haw Yee Yee Yee Haw Yee” left out one of the vocabularies in the dictionary. By applying add-1 smoothing to the bigram model, we can recalculate the bigram frequency and the bigram probability table as in Table 8.6 and Table 8.7.

| previous | next | “Yee” | “Haw” | “Moo” | ⟨/s⟩ | Total |   |
| -------- | ---- | ----- | ----- | ----- | ---- | ----- | - |
| ⟨s⟩      |      | 3     | 2     | 1     | 1    | 7     |   |
| “Yee”    |      | 2     | 3     | 1     | 3    | 9     |   |
| “Haw”    |      | 3     | 1     | 1     | 2    | 7     |   |
| “Moo”    |      | 1     | 1     | 1     | 1    | 4     |   |

| previous | next | “Yee” | “Haw” | “Moo” | ⟨/s⟩ | Total |
| -------- | ---- | ----- | ----- | ----- | ---- | ----- |
| ⟨s⟩      | 3/7  | 2/7   | 1/7   | 1/7   | 1    |       |
| “Yee”    | 2/9  | 3/9   | 1/9   | 3/9   | 1    |       |
| “Haw”    | 3/7  | 1/7   | 1/7   | 2/7   | 1    |       |
| “Moo”    | 1/4  | 1/4   | 1/4   | 1/4   | 1    |       |

The probability that we see the sentence "Moo Moo" in the cowperson language, which would have been 0 before smoothing, is now assigned a non-zero value:

Pr[”Moo” | ⟨s⟩] × Pr[”Moo” | ”Moo”] × Pr[⟨/s⟩ | ”Moo”] = 1 × 1 × 1 ≃ 0.01

Problem 8.4.7. Verify that (8.19) defines a proper probability distribution over the conditioned event. That is, show that

∑ Pr[w | w′] = 1 for any w in the dictionary.

Another smoothing technique is called backoff smoothing. The intuition is that n-gram probabilities are less likely to be zero if n is smaller. So when we run into an n-gram probability that is zero, we replace it with a linear combination of n-gram probabilities of lower values of n.

Example 8.4.8. Recall Example 8.4.5. The bigram probability of “green cream” can be approximated instead as

Pr[“cream′′ | “green′′] ≈ Pr[“cream′′]

Also, say we want to calculate the trigram probability of “like green cream,” which is also zero in the naive trigram model. We can approximate it instead as

Pr[“cream′′ | “like green′′] ≈ α Pr[“cream′′] + (1 − α) Pr[“cream′′ | “green′′]


104 Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University
where α is a hyperparameter for the model.

There are other variants of the backoff smoothing, 19 with some theory for what the “best” choice is, but we will not cover it in these notes.

For instance, Good-Turing and Kneser-Ney smoothing.





# Matrix Factorization and Recommender Systems

# 9.1 Recommender Systems

Cataloging and recommender systems have always been an essential asset for consumers who find it difficult to choose from the vast scale of available goods. As early as 1876, the Dewey decimal system was invented to organize libraries. In 1892, Sears released their famed catalog to keep subscribers up to date with the latest products and trends, which amounted to 322 pages. Shopping assistants at department stores or radio disc jockeys in the 1940s are also examples of recommendations via human curation. In more contemporary times, bestseller lists at bookstores, or Billboard Hits list aim to capture what is popular among people. The modern recommender system paradigm now focuses on recommending products based on what is liked by people “similar” to you. In this long history of recommender systems, the common theme is that people like to follow trends, and recommender systems can help catalyze this process.

# 9.1.1 Movie Recommendation via Human Curation

Suppose we want to design a recommender system for movies. A human curator identifies r binary attributes that they think are important for a movie (e.g., is a romance movie, is directed by Steven Spielberg, etc.) Then they assign each movie an r-dimensional attribute vector, where each element represents whether the movie has the corresponding attribute (e.g., coordinate 2 will have value 1 if a movie is a “thriller” and 0 otherwise). Now, using a list of movies that a particular user likes, the curator assigns an r-dimensional taste vector to a given user in a similar manner (e.g., coordinate w will have value 1 if a user likes “thrillers” and 0 otherwise). With these concepts in mind, we can start with defining the affinity of a user for a particular movie:




# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

# Definition 9.1.1 (User Affinity)

Given a taste vector Ai = (ai,1, ai,2, . . . , ai,r) for user i and the genre vector Bj = (b1,j, b2,j, . . . , br,j) for movie j, we define the affinity of user i for movie j as

r

Ai · Bj = ∑ ai,k bk,j (9.1)

k=1

Intuitively, this metric counts the number of attributes which are 1 in both vectors, or equivalently how many of the user’s boxes are “checked off” by the movie. Mathematically, the affinity is defined as a dot product, which can be extended to matrix multiplication. Thus if we have a matrix A ∈ Rm×r where each of m rows is a taste vector for a user and a matrix B ∈ Rr×n where each of n columns is a genre vector for a movie, the (i, j) entry of the matrix product M = AB represents the affinity score of user i for movie j.

# Definition 9.1.2 (Similarity Metric)

Given taste vector Ai for user i and taste vector Aj for user j, we define the similarity of user i and user j as

r

∑ ai,k aj,k (9.2)

k=1

Similarly, the similarity of movie i and movie j is defined as

r

∑ bk,i bk,j (9.3)

k=1

Finally, in practice, each individual is unique and has a different average level of affinity for movies (for example, some users like everything while others are very critical). This means that directly comparing the affinity of one user to another might not be helpful. One way to circumvent this problem is to augment (9.1) in Definition 9.1.1 as

r

∑ ai,k bk,j + ai,0 (9.4)

k=1

with a bias term ai,0.

Based on the affinity scores or similarity scores, the human curator will be able to recommend movies to users. This model design seems like it does the job as a recommender system. In practice, developing such models through human curation comes with a set of pros and cons:

- Pros: Using human curation allows domain expertise to be leveraged and this intuition can be critical in the development of a good model (i.e., which attribute is important). In addition, a human curated model will naturally be interpretable.



# matrix factorization and recommender systems

•  Cons: The process is tedious and expensive; thus it is difficult to scale. In addition, it can be difficult to account for niche demographics and genres and this becomes a problem for companies with global reach.

We conclude that while human curated models can certainly be useful, the associated effort is often too great.

# 9.2    Recommender Systems via Matrix Factorization

In this section, we provide another technique that can be used for recommender systems — matrix factorization. This method started to become popular since 2005.

# 9.2.1  Matrix Factorization

Matrix factorizations are a common theme throughout linear algebra. Some common techniques include LU and QR decomposition, Rank Factorization, Cholesky Decomposition, and Singular Value Decomposition.

Definition 9.2.1 (Matrix Factorization). Suppose we have some matrix M ∈ Rm×n. A matrix factorization is the process of finding matrices A ∈ Rm×r, B ∈ Rr×n such that M = AB for some r &#x3C; m, n.

Unfortunately, these techniques become less directly applicable once we consider the case where most of the entries of M are missing (i.e., a missing-data setting). As we saw in Section 9.1.1, this is very common in real-world applications — for example, if the (m, n) entry of M represents the rating of user m for movie n, most entries in M are missing because not everyone has seen every movie. What can we do in such a case?

In turns out, if we assume that M is a low-rank matrix (which is true for many high-dimensional datasets, as noted in Chapter 7), then we can consider an approximate factorization M ≈ AB on the known entries. We express this as the following optimization problem:

Definition 9.2.2 (Approximate Matrix Factorization). Suppose we have some matrix M ∈ Rm×n where Ω ⊂ [m] × [n] is the subset of (i, j) where Mij is known. An approximate matrix factorization is the process of finding matrices A ∈ Rm×r, B ∈ Rr×n for some r &#x3C; m, n that minimize the loss function:

L(A, B) = | 1 |  ∑ (Mij − (AB)ij)²               (9.5)

Ω (i,j)∈Ω

We denote the approximation as M ≈ AB.





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

Notice this form is familiar: we are effectively trying to find optimal matrices A, B which will minimize the MSE between known entries of M and corresponding entries in the matrix product AB! One thing to note is that by calculating the matrix product AB, we can “predict” entries of M that are unknown. You can take the following result from linear algebra as granted.

# Theorem 9.2.3

Given M ∈ Rm×n, we can find the matrix factorization M = AB, with A ∈ Rm×r and B ∈ Rr×n if and only if M has rank at most r. Also, we can find the approximate matrix factorization M ≈ AB, with A ∈ Rm×r, B ∈ Rr×n if and only if M is “close to” rank r.

# 9.2.2 Matrix Factorization as Semantic Embeddings

Recall the setup in Section 9.1.1. But instead of calculating the affinity matrix M as the product of the matrices A, B, we will approach from the opposite direction. We will start with an affinity matrix M ∈ Rm×n (which is only partially known) and find its approximate matrix factorization M ≈ AB. We can understand that A ∈ Rm×r represents a set of users and that B ∈ Rr×n represents a set of movies.

Specifically, if we let Ai∗ denote the i-th row of A and B∗j denote the j-th column of B, then Ai∗ can be understood as the taste vector of user i and B∗j can be understood as the attribute vector of movie j. One difference to note is that the output of a matrix factorization is real-valued, unlike the 0/1 valued matrices A, B from Section 9.1.1. We can then use the vectors Ai∗ and B∗j to find similar users or movies and make recommendations.

# Example 9.2.4

Assume all columns of B have ℓ2 norm 1. That is, B∗j2 = 1 for all j. When the inner product B∗j B∗j′ of two movie vectors is actually 1, the two vectors are exactly the same! They have the same inner product with every user vector Ai∗ — in other words these movies have the same appeal to all users. Now suppose B∗j · B∗j′ is not quite 1 but close to 1, say 0.9. This means the movie vectors are quite close but not the same. Still, their inner product with typical user vectors will not be too different.

| Movies n | Movies n  |
| -------- | --------- |
| Users m  | ≈ Users m |




matrix factorization and recommender systems

# 9.2.3 Netflix Prize Competition: A Case Study

During 2006-09, DVDs were all the rage. Companies like Netflix were quite interested in recommending movies as accurately as possible in order to retain clients. At the time, Netflix was using an algorithm which had stagnated around RMSE = 0.95.1 Seeking fresh ideas, Netflix curated an anonymized database of 100M ratings (each rating was on a 1 − 5 scale) of 0.5 M users for 18K movies. Adding a cash incentive of $1,000,000, Netflix challenged the world to come up with a model that could achieve a much lower RMSE!2 It turned out that matrix factorization would be the key to achieving lower scores. In this example, m = 0.5 M, n = 18k, and Ω corresponds to the 100M ratings out of m · n = 10B affinities.3

After a lengthy competition,4 the power of matrix factorization is on full display when we consider the final numbers:

- Netflix’s algorithm: RMSE = 0.95
- Plain matrix factorization: RMSE = 0.905
- Matrix factorization and bias: RMSE = 0.9
- Final winner (an ensemble of many methods): RMSE = 0.856

Figure 9.2: 2D visualization of embeddings of film vectors. Note that you see clusters of “artsy” films on top right, and romantic films on the bottom. Credit: Koren et al., Matrix Factorization Techniques for Recommender Systems, IEEE Computer 2009.

| MabyD   | We       | H     | B Anihal | Citnfane | Tidd  | HrSked Saface | Sp once |   |
| ------- | -------- | ----- | -------- | -------- | ----- | ------------- | ------- | - |
| Fe      |          |       |          |          |       |               |         |   |
| Ro p    | THNt to. | Monch |          |          |       |               |         |   |
| Thed    | Te       | p     | An       | Gayoian  | Stema | pbie          |         |   |
| CompyHa |          | SA    |          |          |       |               |         |   |





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

# 9.2.4 Why Does Matrix Factorization Work?

In general, we need mn entries to completely describe a m × n matrix M. However, if we find factor M into the product M = AB of m × r matrix A and r × n matrix B, then we can describe M with essentially only (m + n)r entries. When r is small enough such that (m + n)r ≪ mn, some entries of M (including the missing entries) are not truly “required” to understand M.

# Example 9.2.5

Consider the matrix

| 1 | 1  | \* | 2  |
| - | -- | -- | -- |
| 1 | 1  | \* | \* |
| 4 | \* | 8  | \* |
| 4 | \* | \* | \* |

Is it possible to fill in the missing elements such that the rank of M is 1? Since r = 1, it means that all the rows/columns of M are the same up to scaling. By observing the known entries, the second row should be equal to the first row, and the third and the fourth row should be equal to the first row multiplied by 4. Therefore, we can fill in the missing entries as

| 1 | 1 | 2 | 2 |
| - | - | - | - |
| 1 | 1 | 2 | 2 |
| 4 | 4 | 8 | 8 |
| 4 | 4 | 8 | 8 |

It is not hard to infer that M = AB where A = (1, 1, 4, 4)T and B = (1, 1, 2, 2).

# Example 9.2.6

Consider another matrix

| 1  | 1  | \* | \* |
| -- | -- | -- | -- |
| 1  | 7  | \* | \* |
| 4  | \* | \* | 2  |
| \* | 4  | \* | \* |

Is it possible to fill in the missing elements such that the rank of M is 1? This time, the answer is no. Following a similar logic from Example 9.2.5, the second row should be equal to the first row multiplied by a constant. This is not feasible since M2,1/M1,1 = 1 and M2,2/M1,2 = 7.

# 9.3 Implementation of Matrix Factorization

In this section, we look more deeply into implementing matrix factorization in an ML setting. As suggested in Definition 9.2.2, we can consider the process of approximating a matrix factorization to be an optimization problem. Therefore, we can use gradient descent.




matrix factorization and recommender systems

# 9.3.1  Calculating the Gradient of Full Loss

Recall that for an approximate matrix factorization of a matrix M, we want to find matrices A, B that minimize the following loss:

L(A, B) = | 1 |         ∑ (Mij − (AB)ij)²              (9.5 revisited)

Ω (i,j)∈Ω

Here (AB)ij  = Ai* · B*j. Now we find the gradient of the loss L(A, B) by first finding the derivatives of L with respect to elements of A (a total of mr derivatives), then finding the derivatives of L with respect to elements of B (a total of nr derivatives).

First, consider an arbitrary element Ai′k′:

∂ ∂   L(A, B) = | 1 |         ∑ 2(Mij − (AB)ij)  ∂       (−(AB)ij)

Ai′k′            Ω (i,j)∈Ω                 ∂Ai′k′

= | 1 |          ∑    2(Mi′j − (AB)i′j)     ∂   (−(AB)i′j)

Ω j: (i′,j)∈Ω                            ∂Ai′k′

= | 1 |          ∑    2(Mi′j − (AB)i′j) · (−Bk′j)

Ω j: (i′,j)∈Ω

= | 1 |          ∑    −2Bk′j(Mi′j − (AB)i′j)         (9.6)

Ω j: (i′,j)∈Ω

Note that the second step is derived because (AB)ij = ∑k AikBkj and if i = i′, then ∂(AB)i′j  = 0. Enumerating (i, j) ∈ Ω can be changed to only enumerating (i′, j) ∈ Ω. Similarly, we can consider an arbitrary element Bk′j′:

∂ ∂        L(A, B) = | 1 |  ∑ 2(Mij − (AB)ij)    ∂     (−(AB)ij)

Bk′j′             Ω (i,j)∈Ω                    ∂Bk′j′

= | 1 |          ∑    2(Mij′ − (AB)ij′)     ∂   (−(AB)ij′)

Ω i: (i,j′)∈Ω                            ∂Bk′j′

= | 1 |          ∑    2(Mij′ − (AB)ij′) · (−Aik′)

Ω i: (i,j′)∈Ω

= | 1 |          ∑    −2Aik′(Mij′ − (AB)ij′)         (9.7)

Ω i: (i,j′)∈Ω

Whew! That’s a lot of derivatives, but we now have ∇L(A, B) at our disposal.

# 9.3.2   Stochastic Gradient Descent for Matrix Factorization

Of course, we could use ∇L(A, B) for a plain gradient descent as shown in Chapter 3. However, given that each derivative in the gradient involves a sum over a large number of indices, it would be





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

Worthwhile to use stochastic gradient descent in order to estimate the overall gradient via a small random sample (as shown in Section 3.2). If we take a sample S ⊂ Ω of the known entries at each iteration, the loss becomes

L(A, B) = |1|  ∑ (Mij − (AB)ij)²

S i,j∈S

and the gradient becomes

∂ L(A, B) / ∂ Ai'k' = |1|  ∑ −2Bk'j(Mi'j − (AB)i'j)

S j: (i′,j)∈S

∂ L(A, B) / ∂ Bk'j' = |1|  ∑ −2Aik'(Mij' − (AB)ij')

S i: (i,j′)∈S

However, if we take a uniform sample S of Ω, the computation will not become much cheaper, since (i, j) ∈ S can spread into many different rows and columns. One clever (and common) way to do so is to select a set of columns C by sampling k out of the overall n columns. This method is called column sampling. We then only need to consider entries (i, j) ∈ Ω where j ∈ C and compute gradients only for the entries Bk,j where j ∈ C. We can also perform row sampling in a very similar manner. In practice, whether we should use column sampling or row sampling, or gradient descent of full loss, depends on the actual sizes of m and n.




# Part III

# Deep Learning


NO_CONTENT_HERE


# Introduction to Deep Learning

Deep learning is currently the most successful machine learning approach, with notable successes in object recognition, speech and language understanding, self-driving cars, automated Go playing, etc. It is not easy to give a single definition to such a broad and influential field; nevertheless here is a recent definition by Chris Manning:1

Deep Learning is the use of large multi-layer (artificial) neural networks that compute with continuous (real number) representations, a little like the hierarchically-organized neurons in human brains. It is currently the most successful ML approach, usable for all types of ML, with better generalization from small data and better scaling to big data and compute budgets.

Deep learning does not represent a specific model per se, but rather categorizes a group of models called (artificial) neural networks (NNs) (or deep nets) which involve several computational layers. Linear models studied in earlier chapters, such as logistic regression in Section 4.2, can be seen as special sub-cases involving only a single layer. The main difference, however, is that general deep nets employ nonlinearity in between each layer, which allows a much broader scale of expressivity. Also, the multiple layers in a neural net can be viewed as computing “intermediate representations” of the data, or “high level features” before arriving at its final answer. By contrast, a linear model works only with the data representation it was given.

Deep nets come in various types, including Feed-Forward NNs (FFNNs), Convolutional NNs (CNNs), Recurrent NNs (RNNs), Residual Nets, and Transformers.2 Training uses a variant of Gradient Descent, and the gradient of the loss is computed using an algorithm called backpropagation.

Due to the immense popularity of deep learning, a variety of software environments such as Tensorflow and PyTorch allow quick implementation of deep learning models. You will encounter them in the homework.

1 Source: https://hai.stanford.edu/sites/default/files/2020-09/AI-Definitions-HAI.pdf

2 Interestingly, a technique called Neural Architecture Search uses deep learning to design custom deep learning architectures for a given task.


116   introduction to machine learning lecture notes for cos 324 at princeton university

# 10.1 A Brief History

Neural networks are inspired by the biological processes present within the brain. The concept of an artificial neuron was first outlined by Warren MuCulloch and Walter Pitts in the 1940s.3 The basic frameworks for CNNs and modern training soon followed in the 1980s.4 Later in 1986, backpropagation was discovered as a new procedure to efficiently apply gradient-based training methods to these models.5 However, by the 21st century deep learning had gone out of fashion. This changed in 2012, when Krizhevsky, Sutskever, and Hinton leveraged deep learning techniques through their AlexNet model and set new standards for performance on the ImageNet dataset.6 Deep learning has since begun a resurgence throughout the last decade, boosted by some key factors:

- Hardware, such as GPU and TPU (Tensor Processing Unit, specifically developed for neural network machine learning) technology has made training faster.
- The development of novel neural network architectures as well as better algorithms for training neural networks.
- A vast amount of data collection, boosted by the spread of the internet, has augmented the performance of NN models.
- Popular frameworks, such Tensorflow and PyTorch, have made it easier to prototype and deploy NN architectures.
- Commercial payoff has caused tech corporations to invest more financial resources.

Each of the reasons listed above has interfaced in a positively reinforcing cycle, causing the acceleration of this technology into the foreseeable future.

# 10.2 Anatomy of a Neural Network

# 10.2.1 Artificial Neuron

An artificial neuron, or a node, is the main component of a neural network. Artificial neurons were inspired by early work on neurons in animal brains, with the analogies in Table 10.1.

Formally, a node is a computational unit which receives m scalar inputs and outputs 1 scalar. This scalar output can be used as an input for a different neuron.

Consider the vector x = (x1, x2, . . . , xm) of m inputs. A neuron internally maintains a trainable weight vector w = (w1, w2, . . . , wm)





# Introduction to Deep Learning

# 10. Biological Neuron vs Artificial Neuron

| Biological neuron   | Artificial neuron |
| ------------------- | ----------------- |
| Dendrites           | Input             |
| Cell Nucleus / Soma | Node              |
| Axon                | Output            |
| Synapse             | Interconnections  |

Figure 10.1: A comparison between a brain neuron and an artificial neuron.

Figure 10.2: A sample artificial neuron.

An artificial neuron can optionally have a nonlinear activation function f : R → R and outputs the following value:

y = f (w · x) (10.1)

If no activation function is chosen, we can assume that f is an identity function f (z) = z.

We can also add a scalar bias b before applying the activation function f (z) in which case the output will look like the following:

y = f (w · x + b)

# 10.2.2 Activation Functions

An artificial neuron can choose its nonlinear activation function f (z) from a variety of options. One such choice is the sigmoid function:

σ(z) = 1 / (1 + e^(-z)) (10.2)

7 If no activation function is chosen, we can assume that f is an identity function f (z) = z.

8 If we introduce a dummy variable for the constant bias term as in Chapter 1 we can absorb the bias term into the equation in (10.1).





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

Note that in this case, the neuron represents a logistic regression unit.9 Another popular activation function is the hyperbolic tangent, which is similar to the sigmoid function:

tanh(z) =
ez - e-z
(10.3)
ez + e-z

In fact, we can rewrite the hyperbolic tangent in terms of sigmoid:

tanh(z) =
2σ(2z) - 1
(10.3 revisited)

According to this expression, the tanh function can be viewed as a rescaled sigmoid function. The key difference is: the range of σ(z) is (0, 1) and the range of tanh(z) is (−1, 1).

Arguably the most commonly used activation function is the Rectified Linear Unit, or ReLU:

ReLU(z) =
[z]+ = max{z, 0}
(10.4)

There are several benefits to the ReLU activation function. It is far cheaper to compute than the previous two alternatives and avoids the “vanishing gradient” problem.10 With sigmoid and hyperbolic tangent activation functions, the vanishing gradient problem happens when z = x · w has high absolute values, but ReLU avoids this problem because the derivative is exactly 1 even for high values of z.

Example 10.2.1. Consider a vector x = (−2, −1, 0, 1, 2) of inputs and a neuron with the weights w = (1, 1, 1, 1, 1). If the activation function of this neuron is the sigmoid, then the output will be:

y =
σ(w · x) = σ(0) = 1
(2)

If the activation is ReLU, it will output:

[w · x]+ =
[0]+ = 0

Problem 10.2.2. Consider a neuron with the weights w = (1, 1, 5, 1, 1) and the ReLU activation function. What will the outputs y1 and y2 be for the inputs x1 = (−2, −2, 0, 1, 2) and x2 = (2, −1, 0, 1, 2) respectively?

# 10.2.3 Neural Network

A neural network consists of nodes connected with directed edges, where each edge has a trainable parameter called its “weight” and each node has an activation function as well as associated parameter(s). There are designated input nodes and output nodes. The input nodes are given some input values, and the rest of the network then computes as follows: each node produces its output by taking the




introduction to deep learning

# 10.3 Why Deep Learning?

Now that we are aware of the basic building blocks of neural networks, let’s consider why we prefer these models over techniques explored in previous chapters. The key understanding is that the models previously discussed are fundamentally linear in nature. For instance, if we do binary classification, where the data point x is mapped to a label based on sign(w · x), then this corresponds to separating the points with label +1 from the points with label −1 via a linear hyperplane w · x = 0. But such models are not a good choice for datasets which are not linearly separable. Deep learning is inherently nonlinear and is able to do classification in many settings where linear classification cannot work.

# Deep neural network

Figure 10.3: A sample neural network design. Each circle represents one artificial neuron. Two nodes being connected by an edge means that the output of the node on the left is being used as one of the inputs for the node on the right.

# Values produced by all nodes

Values produced by all nodes that have a directed edge to it. If the directed graph of connections is acyclic — which is the case in most popular architectures — this process of producing the values takes finite time and we end up with a unique value at each of the output nodes.

11 The term hidden nodes is used for nodes that are not input or output nodes.

11 We will not study Recurrent Neural Nets (RNNs), where the graph contains cycles. These used to be popular until a few years ago, and present special difficulties due to the presence of directed loops. For instance, can you come up with instances where the output is not well-defined?

# Figure 10.4

Some examples of datasets that are not linearly separable.

0 on $

0

2

C

0

o

O

-

C

•





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

# 10.3.1 The XOR Problem

Consider the boolean function XOR with the truth table in Table 10.2.

| x1 | x2 | XOR |
| -- | -- | --- |
| 0  | 0  | 0   |
| 0  | 1  | 1   |
| 1  | 0  | 1   |
| 1  | 1  | 0   |

Let us first attempt to represent the XOR function with a single linear neuron. That is, consider a neuron that takes two inputs x1, x2 with weights w1, w2, a bias term b, and the following Heaviside step activation function:

g(z) =

0 if z ≤ 0

1 if z > 0

This neuron is called a linear perceptron. It uses a nonlinear activation function, but the nonlinearity is strictly for the binary classification in the final step. The boundary of the classification is still linear.

# Proposition 10.3.1

There are no values of w1, w2, b such that the linear neuron defined by the values represent the XOR function.

# Proof

Assume to the contrary that there are such values. Let x1 = (0, 0), x2 = (0, 1), x3 = (1, 0), x4 = (1, 1). Then we know that

g(w · x1 + b) = g(w · x4 + b) = 0

g(w · x2 + b) = g(w · x3 + b) = 1

which implies that

w · x1 + b ≤ 0,     w · x4 + b ≤ 0

w · x2 + b > 0,     w · x3 + b > 0

Now let x = (1, 1). Since we have x = (1/2)(x1 + x4), we should have

w · x + b = (1/2) · ((w · x1 + b) + (w · x4 + b)) ≤ 0

since we are taking the average of two non-positive numbers. But at the same time, since x = (1/2)(x2 + x3), we should have

w · x + b = (1/2) · ((w · x2 + b) + (w · x3 + b)) > 0

since we are taking the average of two positive numbers. This leads to a contradiction.

# Problem 10.3.2

Verify that the AND, OR Boolean functions can be represented by a single linear node.




# Introduction to Deep Learning

Figure 10.5 visualizes the truth table for XOR in the 2D plane. The two axes represent the two inputs x1 and x2; blue circles denote that y = 1; and white circles denote that y = 0. A single linear neuron can be understood as drawing a red line that can separate white points from blue points. Notice that it is possible to draw such a line for AND and OR functions, but the data points that characterize the XOR function are not linearly separable.

Figure 10.5: The data points that characterize the XOR function are not linearly separable.

|           | x2 |   |   | x1 |
| --------- | -- | - | - | -- |
|           | 1  | 0 | 0 | 1  |
| x1 AND x2 |    |   |   |    |
| x1 OR x2  |    |   |   |    |
| x1 XOR x2 |    |   |   |    |

Instead, we will leverage neural networks to solve this problem. Let us design an architecture with inputs x1, x2, a hidden layer with two nodes h1, h2, and a final output layer with one node y1. We assign the ReLU activation function to the hidden nodes and define weights and biases as shown in Figure 10.6.

Figure 10.6: A sample neural network which computes the XOR of its inputs x1 and x2. The weights for inputs are shown by black arrows, while bias terms are shown by grey arrows.

To be more explicit, the neural network is defined by the following three neurons:

- h1 = ReLU(x1 + x2)
- h2 = ReLU(x1 + x2 − 1)
- y1 = ReLU(h1 − 2h2)

Problem 10.3.3. Verify that the model in Figure 10.6 represents the XOR function by constructing a truth table.

The main difference between the single linear neuron approach and the neural network for the XOR function is that the network now...



# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

has two layers of neurons. If we only focus on the final layer of the neural network, we expect the boundary of the binary classification to be linear in the values of h1, h2. However, the values of h1, h2 are not linear in the input values x1, x2 because the hidden nodes utilize a nonlinear activation function. Hence the boundary of the classification is also not linear in the input values x1, x2. The nonlinear activation function transforms the input space into a space where the XOR operation is linearly separable. As shown in Figure 10.7, the h space is quite clearly linearly separable in contrast to the original x space.

Figure 10.7: Unlike the x space, after applying the nonlinear ReLU activation function, the mapped h space is linearly separable.

|            | **h2** |
| ---------- | ------ |
| **$x\_2$** | 1      |
| 0          | 0      |
| 1          | $h$    |
|            | 1      |
|            | 2      |

a) The original x space

b) The new (linearly separable) h space

# 10.4 Multi-class Classification

Neural networks, like multi-class regression in Chapter 4, can be used for classification tasks where the number of possible labels is larger than 2. Real-life scenarios include hand-written digit recognition on the MNIST dataset, where the model designer could use ten different classes to correspond to each possible digit. Another possible example is a speech recognition language model, where the model is trained to distinguish between sounds of vocabularies. It turns out that such functionality can be added by simply including the same number of output neurons as the desired number of classes in the output layer. Then, the values of the output neurons will be converted into a probability distribution Pr[y = k] over the number of classes.

# 10.4.1 Softmax Function

Just as in Chapter 4, we use the softmax function for the purpose of the multi-class classification. See Chapter 19 for the definition of the softmax function.

Example 10.4.1. Say o = (3, 0, 1) are the values of the output neurons of a neural network before applying the activation function. If we decide to




# Introduction to Deep Learning

Apply the softmax function as the activation function, the final outputs of the network will be softmax(o) ≃ (0.84, 0.04, 0.11). If the network was trying to solve a multi-class classification task, we can understand that the given input is most likely to be of class 1, with probability 0.84 according to the model.

One notable property of the softmax function is that the output of the function is the same if all coordinates of the input vector is shifted by the same amount; that is softmax(z) = softmax(z + c · 1) for any c ∈ R, where 1 = (1, 1, . . . , 1) is a vector of all ones.

# Example 10.4.2

Consider two vectors z1 = (5, 2, 3) and z2 = (3, 0, 1). Then softmax(z1) = softmax(z2) because z2 = z1 − (2, 2, 2).

# Problem 10.4.3

Prove the property that softmax(z) = softmax(z + c · 1) for any c ∈ R. (Hint: multiply both the numerator and the denominator of softmax(z)k by exp(c).)


NO_CONTENT_HERE



# 11 Feedforward Neural Network and Backpropagation

Feedforward Neural Networks (FFNNs) are perhaps the simplest kind of deep nets and are characterized by the following properties:

- There are nodes connected with no cycles.
- Nodes are partitioned into layers numbered 1 to k for some k. The nodes in the first layer receive input of the model and output some values. Then the nodes in layer i + 1 receive output of the nodes in layer i as their input and output some values. The output of the model can be computed with the output of the nodes in layer k.
- No outputs are passed back to lower layers.

Now, we only consider fully-connected layers — a special case of a layer in feedforward neural networks.

# Definition 11.0.1 (Fully-Connected Layer)

A fully-connected layer is a neural network layer in which all the nodes from one layer are fully connected to every node of the next layer.

Note that not all layers of feedforward neural networks are necessarily fully-connected (a typical case is a Convolutional Neural Network, which we will explore in Chapter 12). However, feedforward neural networks with fully-connected layers are very common and also easy to implement.

# 11.1 Forward Propagation: An Example

Forward propagation refers to how the network converts a specific input to the output, specifically the calculation and storage of intermediate variables from the input layer to the output layer. In this section, we use concrete examples to motivate the topic. We will provide a more general formula in the next section. Readers who have a stronger background in math may feel to skip this section altogether.





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

# 11.1.1 One Output Node

We start with the network in Figure 11.1 as an example. The network receives three inputs x1, x2, x3 and has a first hidden layer with two nodes h(1)1, h(1)2, a second hidden layer with two nodes h(2)1, h(2)2, and a final output layer with one node o. We assign the ReLU activation function to the hidden units, and define weights as shown in Figure 11.1.

The two hidden nodes in the first hidden layer are characterized by the following equations:

h(1)1 = ReLU(2x1 − 3x2)

h(1)2 = ReLU(−x1 + x2 + 2x3)

and the two hidden nodes in the second hidden layer are characterized by the following equations:

h(2)1 = ReLU(h(1)1 + 2h(1)2)

h(2)2 = ReLU(2h(1)1 − 2h(1)2)

and the output node is characterized by the following equation:

o = −h(2)1 + 2h(2)2

Therefore, if we know the input values x1, x2, x3, we can first calculate the values h(1)1, h(1)2, then using these values, calculate h(2)1, h(2)2, and finally using these values, we can calculate the output o of the network.

# Example 11.1.1

If the provided input vector to the neural network in Figure 11.1 is x = (1, 1, 1), we can calculate the first hidden layer as

h(1)1 = ReLU(2 − 3) = 0

h(1)2 = ReLU(−1 + 1 + 2) = 2



feedforward neural network and backpropagation

and the second hidden layer as

h(2) = ReLU(0 + 2 · 2) = 4

h(2) = ReLU(0 − 2 · 2) = 0

and the output as

o = −4 + 2 · 0 = −4

# 11.1.2 Multiple Output Nodes

Networks can have more than one output node. An example is the network in Figure 11.2.

Figure 11.2: A sample feedforward neural network with two hidden layers and three output nodes.

| x1 | ReLU | ReLU |   |    |
| -- | ---- | ---- | - | -- |
| -1 | 2    | -1   |   |    |
| x2 | -3   | 2    | 2 |    |
| 1  | h0   | -2   | 1 |    |
| x3 | 0    | 2    | 2 | 03 |

The networks in Figure 11.1 and Figure 11.2 are the same except for the output layer; the former has one output node, while the latter has three output nodes. Now the output values of the network in Figure 11.2 can be calculated as:

o1 = −h(2) + 2h(2)

o2 = 2h(2) + h(2)

o3 = h(2) + 2h(2)

Recall from the previous Chapter 10 that a FFNN with multiple output nodes is used for multi-class classification. After the naive output values are calculated, the output nodes will use the softmax activation function to transform the values into the probabilities for each of the three classes. That is, the probability for predicting each class will be calculated as:

o1 = softmax(o1, o2, o3)1

o2 = softmax(o1, o2, o3)2

o3 = softmax(o1, o2, o3)3





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

# 11.1.2

If the provided input vector to the neural network in Figure 11.1 is x = (1, 1, 1), we can calculate the output layer as

o1 = −4 + 0 = −4

o2 = 2 · 4 + 0 = 8

o3 = 4 + 0 = 4

and the probabilities of each class as

o1 = softmax(−4, 8, 4)1 =
−4 e−4 ≃ 0.00

e−4 + e8 + e4

o2 = softmax(−4, 8, 4)2 =
e−4 + e8 + e4 ≃ 0.98

o3 = softmax(−4, 8, 4)3 =
e−4 + e8 + e4 ≃ 0.02

# 11.1.3 Matrix Notation

Let w(1) be the weight between the i-th node h(1) in the first hidden layer and the j-th input xj. Then (11.1) can be rewritten as

h(1) = ReLU(w(1)x1 + w(1)x2 + w(1)x3)1

h(1) = ReLU(w(1)x1 + w(1)x2 + w(1)x3)2

Notice that if we set x = (x1, x2, x3) ∈ R3 and h(1) = (h(1), h(1)) ∈ R2 and define a matrix W(1) ∈ R2×3 where its (i, j) entry is w(1)i,j, then we can further rewrite (11.1) as

h(1) = ReLU(W(1)x) (11.5)

This will be a convention throughout this chapter.

where the ReLU function is applied element-wise.

Similarly, if we let w(2) be the weight between the i-th node h(2) in the second hidden layer and the j-th node h(1) in the first hidden layer, (11.2) can be rewritten as

h(2) = ReLU(w(2)h(1) + w(2)h(1))1

h(2) = ReLU(w(2)h(1) + w(2)h(1))2

or in a matrix notation as

h(2) = ReLU(W(2)h(1)) (11.6)

where h(2) = (h(2), h(2)) ∈ R2 and W(2) ∈ R2×2 is a matrix whose (i, j) entry is w(2)i,j.





# feedforward neural network and backpropagation

Next, if we let w(o) be the weight between the i-th output node oi (before softmax) and the j-th node h(2) in the second hidden layer, (11.3) can be rewritten as

o1 = w(o)h(2) + w(o)h(2
1,1  1             1,2       2
o2 = w(o)h(2) + w(o)h(2
2,1  1             2,2       2
o3 = w(o)h(2) + w(o)h(2
3,1  1             3,2       2

or in a matrix notation as

o = W(o)h(2)                                        (11.7)

where o = (o1, o2, o3) ∈ R3 and W(o) ∈ R3×2 is a matrix whose (i, j) entry is w(o).

Finally, if we let o = (o1, o2, o3) ∈ R3, then (11.4) can be rewritten as

o = so f tmax(o)                                        (11.8)

We summarize the results above into the following matrix equations

h(1) = ReLU               W(1)x
h(2) = ReLU( )W(2)h(1)                                         (11.9)
o = W(o)h(2)
o = so f tmax(o)

# Example 11.1.3.

If the provided input vector to the neural network in Figure 11.2 is x = (1, 1, 1), we can calculate the first hidden layer as

h(1) = W(1)x = ReLU                 2          −3        0         1     =     0
−1            1         2         1           2
1

and the second hidden layer as

h(2) = W(2)h(1) = ReLU                     1        2         0          =    4
2        −2        2               0

and the output layer o (before the softmax) as

o = W(o)h(2) =  −1            2         4     =        −4
2             1         0                   8
1             2                             4

The probability distribution o of the three classes can then be calculated as

o = so f tmax(o) =      −4  e−4        ,    −       e8                  ,  −  e4
e       + e8 + e4     e    4 + e8 + e4                 e  4 + e8 + e4




130   introduction to machine learning lecture notes for cos 324 at princeton university

# 11.2    Forward Propagation: The General Case

We now consider an arbitrary feedforward neural network with L ≥ 1 layers. Let x ∈ Rd₀ be the vector of d0 inputs to the network. For k = 1, 2, . . . , L, let h(k) = (h1(k), h2(k), . . . , hdk(k)) ∈ Rdk represent the dk outputs (values of each of the nodes) of the k-th layer. The k-th layer is also known as the output layer, and we alternatively denote d0 = din and dL = dout to emphasize that they are respectively the number of inputs and the number of output nodes. Each of the k-th layer where 1 ≤ k ≤ L − 1 is considered a hidden layer, but for convenience, we may abuse notation and refer to the input/output layers as respectively the 0-th and L-th hidden layers as well.

Additionally, we consider W(k) ∈ Rdk×dk−1 to represent the weights for the k-th hidden layer. Its (i, j) entry is the weight between the i-th node hi(k) of the k-th hidden layer and the j-th node hj(k−1) of the (k − 1)-th hidden layer. We also alternatively denote W(L) = W(o) to emphasize that it represents the weights for the output layer.

Finally, let f(k) be the nonlinear activation function for layer k. For instance, consider the output layer. If dout = 1 (i.e., there is one output node), we can assume that f(L) is the identity function. On the other hand, if dout > 1 (i.e., there are multiple output nodes), we can assume that f(L) is the softmax function. It is also possible to use different activation functions for each layer.

With all these new notations in mind, we can express the nodes of layer k as:

h(k) = f(k)(W(k)h(k−1))

for each k = 1, 2, . . . , L. If dout = 1, we let o = W(L)hL−1 denote the final output of the model. If dout > 1, we let o = W(L)hL−1 denote the output layer before the softmax and o = f(L)(o) denote the output layer after the softmax.

# 11.2.1    Number of Weights

We now briefly consider the number of weights in a feedforward network. There are din · d1 weights (or variables) for the first hidden layer. Similarly, there are d1 · d2 weights for the second hidden layer. In total, the number of weights is:

∑i=0L−1 di · di+1

# Example 11.2.1.

The number of weights in the model in Figure 11.2 can be calculated as:

3 × 2 + 2 × 2 + 2 × 3 = 16





# 11.2.2 What If We Remove Nonlinearity?

If we removed the nonlinear activation function ReLU in our model from (11.9), we would have the following forward propagation equations:

- h(1) = W(1)x
- h(2) = W(2)h(1)
- o = W(o)h(2)
- o = softmax(o)

Notice that if we set W′ = W(o)W(2)W(1) ∈ R3×3, then

o = softmax(W′x)

We have thus reduced our neural net to a standard multi-classification logistic regression model! As we have discussed the limitation of linear models earlier, this indicates the importance of having nonlinear activation functions between layers.

# 11.2.3 Training Loss

Just like we have defined a loss function for ML models so far, we also define an appropriate loss function for neural networks, where the objective of the network becomes finding a set of weights that minimize the loss. While there are many different definitions of loss functions, here we present two — one that is more appropriate when there is a single output node, and another that is more appropriate for multi-class classification.

When there is only one scalar node in the output layer (i.e., dout = 1), we can use a squared error loss, similar to the least squares loss from (1.4):

∑(x,y)∈D (y − o)2 (Squared Error Loss)

where x ∈ Rdin is the input vector, y is its gold value (i.e., actual value in the training data), and o = W(o)h(L−1) is the final output (i.e., prediction) of the neural network.

# Example 11.2.2

If the provided input vector to the neural network in Figure 11.1 is x = (1, 1, 1) and the training output is y = 0, we can calculate the squared error loss as

(y − o)2 = (0 − (−4))2 = 16

When there are multiple nodes in the output layer (e.g., for multi-class classification), we can use a loss function that is similar to the logistic loss. Recall that in logistic regression, we defined the logistic





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

# Loss as a Sum of Log Loss Over a Set of Data Points:

∑(x,y)∈D − log Pr[label y on input x] (4.5 revisited)

where y ∈ {−1, 1} denotes the gold label. For neural networks, we can analogously define the cross-entropy loss:

∑(x,y)∈D − log oy (Cross-Entropy Loss)

where y ∈ {1, . . . , dout} denotes the gold label, and oy denotes the probability that the model assigns to class y — that is, the y-th coordinate of the output vector o = softmax(o) after applying the softmax function.

# Example 11.2.3:

If the provided input vector to the neural network in Figure 11.2 is x = (1, 1, 1) and the training output is y = 2, we can calculate the cross-entropy loss for this data point as

− log oy = − log e−4 + e48 + 4 ≃ 4.02

# 11.3 Backpropagation: An Example

Just like in previous ML models we have learned, the process of training a neural network model involves three steps:

1. Defining an appropriate loss function.
2. Calculating the gradient of the loss with respect to the training data and the model parameters.
3. Iteratively updating the parameters via the gradient descent algorithm.

But once a neural network grows in size, the second step of calculating the gradients starts to become a problem. Naively trying to calculate each of the gradients separately becomes inefficient. Instead, Backpropagation2 is an efficient way to calculate the gradients with respect to the network parameters such that the number of steps for the computation is linear in the size of the neural network.

The key idea is to perform the computation in a very specific sequence — from the output layer to the input layer. By the Chain Rule, we can use the already computed gradient value of a node in a higher layer in the computation of the gradient of a node in a lower layer. This way, the gradient values propagate back from the top layer to the bottom layer.

Reference: https://www.nature.com/articles/323533a0





# 11.3.1 The Delta Method: Reasoning from First Principles

The main goal of backpropagation is to compute the partial derivative ∂ f /∂w where f is the loss and w is the weight of an edge. This will allow us to apply the gradient descent algorithm and appropriately update the weight w. Students often find backpropagation a confusing idea, but it is actually just a simple application of the Chain Rule in multivariate calculus.

In this subsection, we motivate the topic with the Delta Method, which is an intuitive way to compute ∂ f /∂w. We perturb a weight w by a small amount ∆ and measure how much the output changes. In doing so, we also measure how the rest of the network changes. As we will see later, the process of computing the partial derivative of the form ∂ f /∂w requires us to also compute the partial derivative of the form ∂ f /∂h where h is the value at a node.

Readers who are familiar with the Chain Rule can quickly browse through the rest of this subsection.

# Example 11.3.1

Consider the model from Figure 11.2 but now with the inputs x = (3, 1, 2). We use the same notation for the nodes and the weights that we used throughout Section 11.1. The goal of this simple example is to illustrate what the derivatives mean.

Suppose we want to take the partial derivative of first output node o1 with respect to the weight w(1)2,2 (i.e., the weight on the edge between the second input x2 and the second node h(1) of first hidden layer). This is denoted as ∂o /∂w(1)2,2. Its definition involves considering how changing w2,2 by an infinitesimal amount ∆ changes o1, whose current value is −3.

Adding ∆ to w(1)2,2 will change the values of the first hidden layer to

h(1) = ReLU((−1) · 3 + (1 + ∆) · 1 + 2 · 2) = 2 + ∆

Letting ∆ → 0, we see that the rate of change of h(1) and h(1) with respect to

| 1  | -3 | 3 | 1 | 2  | 7 | -1 | 2  | -3 |   |   |
| -- | -- | - | - | -- | - | -- | -- | -- | - | - |
| 0  | 2  | 2 | 2 | 1  | 1 | 2  | 16 |    |   |   |
| -2 |    |   |   |    |   |    |    |    |   |   |
|    |    | 2 | 2 | 11 |   |    |    |    |   |   |





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

# Figure 11.4

The model from Figure 11.2 with inputs x = (3, 1, 2). When w(1) is changed by a small number ∆, the first hidden layer is also affected.

# Figure 11.5

The model from Figure 11.2 with inputs x = (3, 1, 2). When w(1) is changed by a small number ∆, the second hidden layer is also affected.

Using the updated values of the first hidden layer, the second hidden layer will be calculated as:

| h(2) | ReLU(1 · 3 + 2 · (2 + ∆))    | = 7 + 2∆ |
| ---- | ---------------------------- | -------- |
| h(2) | ReLU(2 · 3 + (−2) · (2 + ∆)) | = 2 − 2∆ |

This shows that the rate of change of h(2) and h(2) with respect to change of w(1) is 2 and −2 respectively.

Finally, the output layer can now be calculated as:

| o1 | = (−1) · (7 + 2∆) + 2 · (2 − 2∆) | = −3 − 6∆ |
| -- | -------------------------------- | --------- |
| o2 | = 2 · (7 + 2∆) + 1 · (2 − 2∆)    | = 16 + 2∆ |
| o3 | = 1 · (7 + 2∆) + 2 · (2 − 2∆)    | = 11 − 2∆ |

This shows that the rate of change of o1 with respect to change of w(1) is −6.

# Example 11.3.2

Now we consider the meaning of ∂o1/∂h1: how changing the value of h(2) by an infinitesimal ∆ affects o1. Note that this is a thought experiment.



# Feedforward Neural Network and Backpropagation

Figure 11.6: The model from Figure 11.2 with inputs x = (3, 1, 2). When w(1) is changed by a small number ∆, the output layer is also affected.

Experiment that does not correspond to a change that is possible if the net were a physical object constructed of nodes and wires — the value of h(2) is completely decided by the previous layers and cannot be changed in isolation without touching the previous layers. However, treating these values as variables, it is possible to put on a calculus hat and think about the rate of change of one with respect to the other.

If only the value of h(2) is changed from 7 to 7 + ∆ in Figure 11.3, then o1 can be calculated as

o1 = (−1) · (7 + ∆) + 2 · 2 = −3 − ∆

which shows that ∂o1/∂h(2) = −1.

# Problem 11.3.3

Following the calculations in Example 11.3.1 and Example 11.3.2, calculate ∂o1/∂h(2), ∂h(2)/∂w(1), and ∂h(2)/∂w(1). Verify that

∂o1 / ∂w1 = (∂h2 / ∂h2) · ∂w(1) + ∂h(2) / ∂w(1) · (7 + 2Δ)

∂o1 = −1

# Problem 11.3.4

Following the calculations in Example 11.3.1 and Example 11.3.2, calculate ∂h(2)/∂h(1), ∂h(2)/∂h(1), and ∂h(1)/∂w(1). Verify that

∂o1 / ∂w2,2 = (∂h2 / ∂h1) · ∂h(1) + ∂o1 / ∂h(2) · ∂h(1)

Some readers may notice that (11.10) is just the result of the Chain Rule in multivariable calculus. It shows that the effect of w(1) on o1 is the sum of its effect through all possible paths that the values propagate through the network, and the amount of effect for each path can be calculated by multiplying the appropriate partial derivative between each layer.

In this section, we calculated by hand one partial derivative ∂o1/∂w(1). But in general, to compute the loss gradient, we see below that we want to calculate the partial derivative of each output node.

1 + 16 + 2Δ

Δ

2

2 + Δ - 2 2 - 2Δ 2 X

2 11 - 2Δ




136   introduction to machine learning lecture notes for cos 324 at princeton university

oi with respect to each weight in the network. Manually applying the Chain Rule for each partial derivative as in (11.10) is too inefficient.3 Instead, in the next section, we will learn how to utilize matrix operations to combine the computation for multiple partial derivatives into one process.4

11.4    Backpropagation: The General Case

# 11.4.1  Jacobian Matrix

Suppose some vector y = (y1, y2, . . . , ym) ∈ Rm is a function of x = (x1, x2, . . . , xn) ∈ Rn — that is, there is a mapping f  : Rn → Rm such that y = f (x), or equivalently, there are m functions fi : Rn → R for each i = 1, 2, . . . , m such that yi =  fi(x).

Then the Jacobian matrix of y with respect to x, denoted as J(y, x), is an m × n matrix whose (i, j) entry is the partial derivative ∂yi/∂xj. Note that each entry of this matrix is itself a function of x. A bit confusingly, a Jacobian matrix is also often denoted as ∂y/∂x when it is clear from the context that x, y are vectors and hence this object is not a partial derivative or gradient.5

The mathematical interpretation of the Jacobian matrix is that if we change x such that each coordinate xi is updated to xi + δi for an infinitesimal value δi, then the output y changes to y + J(y, x)δ.

Example 11.4.1. Suppose y is a linear function of x — that is, there exists a matrix A ∈ Rm×n such that y = Ax. Then notice that yi, the i-th coordinate of y, can be expressed as

yi = Ai,∗x = Ai,1x1 + Ai,2x2 + . . . + Ai,nxn

Notice that the partial derivative ∂yi/∂xj is equal to Aij. This means that the (i, j) entry of the Jacobian matrix is the (i, j) entry of the matrix A, and hence J(y, x) = A.

Problem 11.4.2. If y ∈ R2 is a function of x ∈ R3 such that

y1 = 2x1 − x2 + 3x3

y2 = −x1 + 2x3

then what is the Jacobian matrix J(y, x)?

Example 11.4.3. If x ∈ Rn and y = ReLU(x) ∈ Rn, then notice that

∂yi    =   1     xi > 0

∂xi        0     otherwise

We can also denote this with an indicator function 1(xi > 0). Also for any j  = i, we see that ∂yi/∂xj = 0. Therefore, the Jacobian matrix J(y, x) is a





feedforward neural network and backpropagation 137

diagonal matrix whose entry down the diagonal is 1(xi > 0); that is

J(y, x) = diag(1(x > 0))

where we take the indicator function element-wise to the vector x.

# Definition 11.4.4 (Jacobian Chain Rule).

Suppose vector z ∈ Rk is a function of y ∈ Rm and y is a function of x ∈ Rn, then by the Chain Rule, the Jacobian matrix J(z, x) ∈ Rk×n is represented as the matrix product:

J(z, x) = J(z, y)J(y, x) (11.11)

In context of the feedforward neural network, each hidden layer is a function of the previous layer. Specifically, vector of activations of a hidden layer is a function of the vector of activations of the previous layer as well as of the trainable weights within the layer.

# Example 11.4.5 (Gradient calculation for a single layer with ReLU’s).

If x ∈ Rn, A ∈ Rm×n, y = Ax ∈ Rm and z = ReLU(y) ∈ Rm, then the Jacobian matrix J(z, x) can be calculated as

J(z, x) = J(z, y)J(y, x) = diag(1(Ax > 0))A

# 11.4.2 Backpropagation — Efficiency Using Jacobian Viewpoint

Now we return to backpropagation, and show how the Jacobian viewpoint allows computing the gradient of the loss (with respect to network parameters) with a number of mathematical operations (i.e., additions and multiplications) proportional to the size of the fully connected net.

Recall that we want to find the weights W(1), W(2), . . . , W(o) that minimize the cross-entropy loss ℓ. To apply the standard/stochastic gradient descent algorithm, we need to find the partial derivative

∂ℓ / ∂W(k)i,j

of the loss function with respect to each weight W(k) of each layer k.

To simplify notations, we introduce a new matrix ∂ℓ / ∂W(k) which has the same dimensions as W(k) (e.g., ∂ℓ / ∂W(1) ∈ R2×3 the (i, j) entry of the matrix is: ∂ℓ / ∂W(k)i,j in Figure 11.2) and

∂ℓ / ∂W(k) = ∂ℓ / ∂W(k)i,j (11.12)

for any layer k. The matrix ∂ℓ / ∂W(k) will be called the gradient with respect to the weights of the k-th layer. Now the update rule for the gradient descent algorithm can be written as the following:

W(k) → W(k) − η · ∂ℓ / ∂W(k) (11.13)

6 Alternatively, you can think of flattening W(k) into a single vector, then finding the Jacobian matrix ∂ℓ/∂W(k), and later converting it back to a matrix form.





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

where η is the learning rate. Now the question remains as how to calculate these gradients. As the name “backpropagation” suggests, we will first compute the gradient of the loss ℓ with respect to the output nodes; we then inductively compute the gradient for the previous layers, until we reach the input layer.

# 1. Output Layer:

First recall that the cross-entropy loss due to one data point is

ℓ = − log         dout
∑          eoi
i=1
dout
= − log(eoy) + log          ∑ eoi
i=1
dout
= −oy + log            ∑ eoi
i=1

where y ∈ {1, 2, . . . , dout} is the ground truth value. Therefore, the gradient with respect to the output layer is

∂ℓ              =     −1 + oi     y = i
∂oi  1≤i≤dout          oi          y = i

To simplify notations, we introduce a one-hot encoding vector ey, which has 1 only at the y-th coordinate and 0 everywhere else. Then, we can rewrite the equation above as:

∂ℓ    = o − ey               ⊺ ∈ R1×dout    (11.14)       R1×dout. But o and ey, the terms on the right hand side, are both column vectors, or equivalently a dout × 1 matrix. This is the Jacobian matrix of the loss ℓ with respect to the output layer o.

We resolve the problem by taking the transpose.

# 2. Jacobian With Respect To Hidden Layer:

We first compute ∂ℓ/∂h(L−1), the Jacobian matrix with respect to the last hidden layer before the output layer. Since o = W(o)h(L−1), we can apply the result from Example 11.4.1 and get

∂ℓ           = ∂ℓ J(o, h(L−1))
∂h(L−1)                                         ∂o
= ∂ℓ W(o)                  (11.15)
∂o

Now as an inductive hypothesis, assume that we have already computed the gradient (or Jacobian matrix) ∂ℓ/∂h(k+1). We now compute





# Feedforward Neural Network and Backpropagation

∂ℓ/∂h(k) using the result from Example 11.4.5.

∂ℓ = ∂ℓ J(h(k+1), h(k))

∂h(k) ∂h(k+1) = ∂ℓ diag(1(W(k+1)h(k) > 0))W(k+1)  (11.16)

# 3. Gradient With Respect to Weights:

We first compute ∂ℓ/∂W(o), the gradients with respect to the weights of the output layer. Notice that a particular weight w(o) is only used in computing oi out of all output nodes:

oi = w(o)h(L−1) + . . . + w(o)h(L−1) + . . . + w(o) h(L−1)

i,1  1                  i,j  j         i,dL−1          dL−1

Therefore, the gradient with respect to w(o) can be calculated as

∂ℓ = ∂ℓ · ∂oi = ∂ℓ · h(L−1)

∂w(o) ∂oi ∂w(o)   ∂oi  j

We can combine these results into the following matrix form

∂ℓ = ∂ℓ ⊺ h(L−1) ⊺ (11.17)

Now as an inductive hypothesis, assume that we have already computed the gradient (or Jacobian) ∂ℓ/∂h(k). We now compute ∂ℓ/∂W(k).

To do this, we introduce an intermediate variable z(k) = W(k)h(k−1) such that h(k) = ReLU(z(k)). Then the gradient with respect to a particular weight w(k) can be calculated as

∂ℓ = ∂ℓ ∂z(k) ∂ℓ (k−1)

∂w(k) = ∂z(k) · ∂wi(k) = ∂z(k) · hj

We can combine these results into the following matrix form

∂ℓ = ∂ℓ ∂z(k) ⊺ h(k−1) ⊺

= ∂ℓ ∂h(k) J(h(k), z(k)) ⊺ h(k−1) ⊺

= J(h(k), z(k)) ⊺ ∂ℓ ⊺ h(k−1) ⊺

= diag(1(W(k)h(k−1) > 0)) ∂ℓ ⊺ h(k−1) ⊺ (11.18)

# 4. Full Backpropagation Process

We summarize the results above into the following four steps:




140    introduction to machine learning lecture notes for cos 324 at princeton university

1. Compute the Jacobian matrix with respect to the output layer,

∂ℓ ∈ R1×dout:

∂o

∂ℓ = o − ey ⊺         ((11.14) revisited)

∂o
2. Compute the Jacobian matrix with respect to the last hidden layer,

(∂ℓ−) ∈ R1×dL−1:

∂hL

∂ℓ = ∂ℓ W(o)          ((11.15) revisited)

∂h(L−1)                  ∂o

Then, compute the gradient with respect to the output weights,

∂ℓ     ∈ Rdout×dL−1:

∂W(o)

∂ℓ                 = ∂ℓ  ⊺  h(L−1)   ⊺    ((11.17) revisited)

∂W(o)                  ∂o
3. For each successive layer k, calculate the Jacobian matrix with
respect to the k-th hidden layer       ∂ℓ  ∈ R1×dk:

∂h(k)

∂ℓ       = ∂ℓ      diag(1(W(k+1)h(k) > 0))W(k+1) ((11.16) revisited)

∂h(k)        ∂h(k+1)

Next, compute the gradient with respect to the (k + 1)-th hidden
layer weights    ∂ℓ    ∈ Rdk+1×dk:

∂W(k+1)

∂ℓ          = diag(1(W(k+1)h(k) > 0))           ∂ℓ  ⊺  h(k) ⊺

∂W(k+1)                                     ∂h(k+1)

((11.18) revisited)
4. Repeat Step 3 until we reach the input layer.

Note that these instructions are based on a model that adopts the
cross-entropy loss and the ReLU activation function. Using alterna-
tive losses and/or activation functions would result in a similar form,
although the actual derivatives may be slightly different.

Problem 11.4.6.    (i) Show that if A is an m × n matrix and h ∈ Rn then
computing Ah requires mn multiplications and m vector additions. (ii)
Using the previous part, argue that the number of arithmetic operations
(additions or multiplications) in backpropagation algorithm on a fully
connected net with ReLU activations is proportional to the number of
parameters in the net.

While the above calculation is in line with your basic algorithmic
training, it doesn’t exactly describe running time in modern ML
environments with GPUs, since certain operations are parallelized,
and compilers are optimized to run backpropagation as fast as
possible.





# 11.4.3 Using a Different Activation Function

We briefly consider what happens if we choose a different activation function for the hidden layers. Consider the sigmoid activation function σ(z) = 1 / 1 + e-z. Its derivative is given by:

σ′(z) = e / 1 + e-z · 1 − z / 1 + e

= e-z / (1 + e-z)2 (11.19)

= σ(z) · e-z / 1 + e-z

= σ(z) · (1 − σ(z))

There is also the hyperbolic tangent function tanh(z) = e2z - 1 / e2z + 1.

# Problem 11.4.7.

Compute f ′(z) for f (z) = tanh(z); show how f ′(z) can be written in terms of f (z).

# Problem 11.4.8.

Say a neural network uses an activation function f (z) at layer k + 1 such that f ′(z) is a function of f (z). That is, f ′(z) = g(f (z)) for some function g. Then verify that (11.16, 11.18) can be rewritten as:

∂ℓ / ∂h(k) = diag(g(W(k+1)h(k)))W(k+1)

∂ℓ / ∂h(k+1) = diag(g(W(k+1)h(k))) ∂ℓ / ∂h(k+1)

# 11.4.4 Final Remark

Directly following the steps of backpropagation is complicated and involves a lot of calculations. But remember that backpropagation is simply computing gradients by the Chain Rule. At a high level, we can think of the loss as a function of inputs and all the weights and note that backpropagation simply entails calculating derivatives with respect to each variable. The good news is that modern deep learning software does all the gradient calculations for users. All the model designer needs to do is to determine the neural network architecture (e.g., choose number of layers, number of hidden units, and the activation functions).

One note of caution is that the loss function for deep neural nets is highly non-convex with respect to the parameters of the network. Just as we discussed in Chapter 3, the gradient descent algorithm is not guaranteed to find the actual minimizer in such situation, and the choice of the initial values of the parameters matter a lot.




142  introduction to machine learning lecture notes for cos 324 at princeton university

# 11.5 Feedforward Neural Network in Python Programming

In this section, we discuss how to write Python code to build neural network models and perform forward propagation and backpropagation. As usual, we use the numpy package to speed up computation. Additionally, we use the torch package to easily design and train the neural network.

# Import necessary packages

import  numpy as np
import  torch
import  torch.nn as nn

# Define the neural network

class  Net(nn.Module):
def __init__(self, input_size=2, hidden_dim1=2, hidden_dim2=3, hidden_dim3=2):
super(Net, self).__init__()

self.hidden1 = nn.Linear(input_size, hidden_dim1, bias=False)
self.hidden1.weight = nn.Parameter(torch.tensor([[-2., 1.], [3., -1.]]))

self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2, bias=False)
self.hidden2.weight = nn.Parameter(torch.tensor([[0., 1.], [2., -1.], [1., 2.]]))

self.hidden3 = nn.Linear(hidden_dim2, hidden_dim3, bias=False)
self.hidden3.weight = nn.Parameter(torch.tensor([[-1., 2., 1.], [3., 0., 0.]]))

self.activation = nn.ReLU()

# single step of forward propagation
def forward(self, x):
h1 = self.hidden1(x)
h1 = self.activation(h1)
h2 = self.hidden2(h1)
h2 = self.activation(h2)
h3 = self.hidden3(h2)
return h3

net = Net()

# Forward propagation with sample input

x = torch.tensor([3., 1.])
y_pred = net.forward(x)
print('Predicted value:', y_pred)

# Backpropagation with sample input

loss = nn.functional.cross_entropy(y_pred.unsqueeze(0), torch.LongTensor([1]))
loss.backward()
print(loss)
print(net.hidden1.weight.grad)

We start the code by importing all necessary packages.

import  numpy as np
import  torch



# Feedforward Neural Network and Backpropagation

import torch.nn as nn

With PyTorch, we can design the architecture of any neural network by defining the corresponding class.
class Net(nn.Module):
def __init__(self, input_size=2, hidden_dim1=2, hidden_dim2=3, hidden_dim3=2):
super(Net, self).__init__()

self.hidden1 = nn.Linear(input_size, hidden_dim1, bias=False)
self.hidden1.weight = nn.Parameter(torch.tensor([[-2., 1.], [3., -1.]]))

self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2, bias=False)
self.hidden2.weight = nn.Parameter(torch.tensor([[0., 1.], [2., -1.], [1., 2.]]))

self.hidden3 = nn.Linear(hidden_dim2, hidden_dim3, bias=False)
self.hidden3.weight = nn.Parameter(torch.tensor([[-1., 2., 1.], [3., 0., 0.]]))

self.activation = nn.ReLU()

def forward(self, x):
h1 = self.hidden1(x)
h1 = self.activation(h1)
h2 = self.hidden2(h1)
h2 = self.activation(h2)
h3 = self.hidden3(h2)
return h3

In the constructor, we define all the layers and activation functions we are going to use in the network. In particular, we specify that we need fully-connected layers by making instances of the nn.Linear class and that we need ReLU activation function by making an instance of the nn.ReLU class. Then in the forward() function, we specify the order in which to apply the layers and activations. See Figure 11.7 for a visualization of this neural network architecture.

ReLU Activation    ReLU Activation                      Figure 11.7: A sample feedforward neural network with two hidden layers and two output nodes.

We can simulate one step of forward propagation by calling the forward() function of the class Net we defined.
x = torch.tensor([3., 1.])
y_pred = net.forward(x)
print('Predicted value:', y_pred)




144 Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

# Backpropagation Implementation

Similarly, we can implement backpropagation by specifying which loss function we want to use, and calling its backward() function.

loss = nn.functional.cross_entropy(y_pred.unsqueeze(0), torch.LongTensor([1]))

loss.backward()

print(loss)

Each function call of backward() will evaluate the gradients of loss with respect to every parameter in the network. Gradients can be manually accessed through the following code.

print(net.hidden1.weight.grad)

Note that calling backward() multiple times will cause gradients to accumulate. While we do not update model weights in this code sample, it is important to periodically clear the gradient buffer when doing so to prevent unintended training.8 We will discuss how to do this in the next chapter.

For more information, see PyTorch Documentation.





# 12 Convolutional Neural Network

In Chapter 11, we focused on a type of a neural network called feed-forward neural networks. But different data has different structure (e.g., image, text, audio, etc.) and we need better ways of exploiting them. This can help reduce the number of parameters needed in the network, which may allow easier or more data-efficient training. In this chapter, we present a type of a neural network common in image processing called Convolutional Neural Network (CNN); these models use a mathematical technique called convolution in order to extract important visual features from input images.

# 12.1 Introduction to Convolution

Roughly speaking, convolution refers to a mathematical operation where two functions are “mixed” to output a new function. In machine learning, the main idea of convolution is to reuse the same set of parameters on different portions of input. This is particularly effective at exploiting the structure of images. It was originally motivated by studies of the structure of cortical cells in the V1 visual cortex of the human brain (Hubel and Wiesel won the Nobel Prize in 1981 for this breakthrough discovery). 1 Let’s first consider an example of a 1D convolution.

1 Paper: https://www.jstor.org/stable/24965293.

Figure 12.1: The effects of 1D convolution on graph of COVID-19 positive cases.

X1X_{2X_{3X_{4X5  xt-2Xt-1x{X{t+1X{t+2 yt =3 (xt−1 + xt + xt+1)

$yt+1=  (xt + xt+1 + xt+2)

y1$y{2 y{3 $y 4 y5  yt-2 Yt-1 yt yt+1 Yt+2




146   introduction to machine learning lecture notes for cos 324 at princeton university

# 12.1 Moving Averages and Convolution

Example 12.1.1. Consider the 3-day moving average of daily COVID-19 cases as shown in Figure 12.1. Let xt denote the number of daily cases on day t. We can then take three consecutive values, compute their average and create a new output sequence from averages: yt = (xt−1 + xt + xt+1) / 3.

Then if we set w1 = w2 = w3 = 1 and denote w = (w1, w2, w3), we can write:

yt = w · (xt−1, xt, xt+1)

Then:

- yt+1 = w1xt + w2xt+1 + w3xt+2 = w · (xt, xt+1, xt+2)
- yt+2 = w1xt+1 + w2xt+2 + w3xt+3 = w · (xt+1, xt+2, xt+3)

Notice that we are reusing the same weights and applying them to multiple different values of xt to calculate yt. It is almost like sliding a filter down the array of xt’s and applying it to every set of 3 consecutive inputs. For this reason, we call w the convolution filter weight of length 3.

Example 12.1.2. Consider an input sequence x = (2, 1, 1, 7, −1, 2, 3, 1) and a convolution filter w = (3, 2, −1). The first two output values will be:

- y1 = 2 × 3 + 1 × 2 + 1 × (−1) = 7
- y2 = 1 × 3 + 1 × 2 + 7 × (−1) = −2

Following a similar calculation for the other values, we see that the full output sequence is y = (7, −2, 18, 17, −2, 11). Note that the length of y should be |x| − |w| + 1 = 8 − 3 + 1 = 6.

Problem 12.1.3. If yt = 2xt−1 − xt+1, yt+1 = 2xt − xt+2, and yt+2 = 2xt+1 − xt+3, what is the convolution filter weight?

# 12.2 Convolution in Computer Vision

In this section, we now focus on the application of convolution in computer vision. By the nature of image data, we will be primarily dealing with 2D convolution. Generally, 2D convolution filters are called kernels.

Figure 12.2: The effect of local smoothing on a sample image. (The person depicted here is Admiral Grace Murray Hopper, a computing pioneer.)

Example 12.2.1 (Local Smoothing (Blurring)). An image can be blurred by constructing a filter that replaces each pixel by the average of neighboring pixels:

yi,j = (1/9) ∑b1,b2∈{−1,0,1} xi+b1,j+b2



# Convolutional Neural Network

An example is shown in Figure 12.2.

Figure 12.3: The effect of local sharpening on a sample image

Before

After

# Example 12.2.2 (Local Sharpening (Edge Detection))

The edge of objects in an image can be detected by constructing a filter that replaces each pixel by its difference with the average of neighboring pixels:

yi,j = xi,j − 1       ∑           xi+b ,j+b
9 b₁,b₂∈{−1,0,1}      1    2

An example is shown in Figure 12.3.

# 12.2.1 Convolution Filters for Images

Computationally, we perform 2D convolution on an image by "sliding" the filter around every possible location in the image and taking the inner product:

yi,j =    ∑    wr,sxi+r,j+s               (12.1)
−k≤r,s≤k

The result is a new image and we can view each filter as a transformation which takes an image and returns an image. In the above equation, the filter size is (2k + 1) × (2k + 1). For example, if k = 1, we can consider the convolution weight filter to be

w−1,−1    w−1,0    w−1,+1
w0,−1      w0,0    w0,+1
w+1,−1    w+1,0    w+1,+1

The filter can only be applied to an image of size m × n at a location where the filter completely fits inside the image. Therefore, the locations in the input image where the center of the filter can be placed are k &#x3C; i ≤ m − k, k &#x3C; j ≤ n − k and the size of the output image is (m − 2k) × (n − 2k).

# Example 12.2.3

If the input and convolution filter are given as follows:

1    1  1   0    0
0    1  1   1    0                       1  0  1
X =     0    0  1   1    1    and  W =           0  1  0
0    0  1   1    0                       1  0  1
0    1  1   0    0





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

Then the pixel at (1, 1) of the resulting image can be calculated by applying the filter at the top left corner of the input image. That is, we take the inner product of the following parts (in red) of the two matrices:

| 1 | 1 | 1 | 0 | 0 |
| - | - | - | - | - |
| 0 | 1 | 1 | 1 | 0 |
| 0 | 0 | 1 | 1 | 1 |
| 0 | 0 | 1 | 1 | 0 |
| 0 | 1 | 1 | 0 | 0 |

Which is:

1 × 1 + 1 × 0 + 1 × 1 + 0 × 0 + 1 × 1 + 1 × 0 + 0 × 1 + 0 × 0 + 1 × 1 = 4

Therefore, the (1, 1) entry of the resulting image is 4. Similarly, the remaining pixels of the resulting image can be calculated by moving around the filter as in Figure 12.4. The output image is given as:

| 4 | 3 | 4 |
| - | - | - |
| 2 | 4 | 3 |
| 2 | 3 | 4 |

In this example, X ∈ R5×5, W ∈ R3×3, k = 1 and Y ∈ R3×3.

Figure 12.4: Visual representation of applying a 3 × 3 convolutional filter to a 5 × 5 image.

# Problem 12.2.4

Suppose we have a 10 × 10 image and a 5 × 5 filter. What is the size of the output image?

Figure 12.5 shows some common filters used in image processing. Note that all these filters are hand-crafted and require domain-specific knowledge. However, in convolutional neural networks, we don’t set these weights by hand and we learn all the filter weights from the data!

# 12.2.2 Padding

In standard 2D convolution, the size of the output image is not equal to the size of the input image because we only consider locations where the filter fits completely in the image. However, sometimes we may want their sizes to be the same. In such a case, we apply a





# Convolutional Neural Network

# Figure 12.5: Some common filters

| Box blur (normalized)                                                           |     | 1 | O  | 1    | 1  | 1 | 1 | 1 |
| ------------------------------------------------------------------------------- | --- | - | -- | ---- | -- | - | - | - |
| Gaussian blur 3 × 3 (approximation)                                             | 1   | 1 | 2  | 1    | P  |   |   |   |
|                                                                                 | 16  | 2 | 4  | 2    |    |   |   |   |
| 1                                                                               | 2   | 1 |    |      |    |   |   |   |
|                                                                                 |     | 1 | 4  | 6    | 4  | 1 | P |   |
| Gaussian blur 5 × 5 (approximation)                                             | 1   | 4 | 16 | 24   | 16 | 4 |   |   |
|                                                                                 | 256 | 6 | 24 | 36   | 24 | 6 |   |   |
|                                                                                 |     | 4 | 16 | 24   | 16 | 4 |   |   |
| 4                                                                               | 6   | 4 | 1  |      |    |   |   |   |
|                                                                                 | -1  | 4 | 16 | 24   | 16 | 4 | 4 |   |
| Unsharp masking 5 × 5                                                           | 256 | 6 | 24 | -476 | 24 | 6 |   |   |
| Based on Gaussian blur with amount as 1 and threshold as 0 (with no image mask) |     | 4 | 16 | 24   | 16 | 4 |   |   |
|                                                                                 | \[1 | 4 | 6  | 4    | 1] |   |   |   |

The technique called padding. The idea is to pad pixels to all four edges of the input image (left, right, up, and down) so that the number of valid locations for the filter is the same as the number of pixels in the original image. In particular, if the filter size is (2k + 1) × (2k + 1), we need to pad k pixels on each side.

There are multiple ways to implement padding. Zero padding is when the values at all padded pixels are set to 0. “Same” padding is when the values at padded pixels are set to the value of the nearest pixel at the edge of the input image. In practice, zero padding is a more common form of padding (it is equivalent to adding a black frame around the image).

# 12.2.3 Downsampling Input with Stride

Another common operation in convolutional neural networks is called stride. Stride controls how the filter convolves around the input image. Instead of moving the filter by 1 pixel every time, we can also move the filter every 2 (or in general, s) pixels. Essentially, we are applying each of the filter weights at fewer locations of the image than before. This can be viewed as a downsampling strategy, which gives a smaller output image while greatly preserving the information from the original input.




150  introduction to machine learning lecture notes for cos 324 at princeton university

# Figure 12.6: A visual comparison between two common types of padding: zero padding and “same” padding

| Zero Padding | "Same" Padding |
| ------------ | -------------- |

Suppose we have an input image of size m × n and a filter of size (2k + 1) × (2k + 1). If padding is applied to the image, the output image size is ⌊(m + s − 1)/s⌋ × ⌊(n + s − 1)/s⌋. 3 If padding is not applied to the image, the output image size is ⌊(m + s − 2k − 1)/s⌋ × ⌊(n + s − 2k − 1)/s⌋; this is because convolution with stride is performed directly on the input image itself, making the effective input image size (m − 2k) × (n − 2k).

# Example 12.2.5

Suppose we have an input image of size 5 × 5 and a filter of size 3 × 3. If we apply padding and take stride size s = 2, then output size is 3 × 3.

# Figure 12.7: Visual representation of applying a 3 × 3 convolutional filter to a 5 × 5 image with padding and stride size 2.

# 12.2.4  Nonlinear Convolution

For each location in the image (original or padded) and a single convolution filter, we can apply a nonlinear activation function after the convolution

yi,j = g ∑ wr,sxi+r,j+s                   (12.2)

−k≤r,s≤k

where g is some function like ReLU, sigmoid, tanh. The intuition is similar to what we had earlier in feedforward neural networks — if we don’t add non-linear activation functions, a multi-layered convolutional neural network can be easily reduced to a linear model!




# Convolutional Neural Network

# 12.2.5 Channels

In general, we do not only use one convolution filter. We construct a network of multiple layers, and for each of the layers, we apply multiple convolutional filters. Different filters will be able to detect different features of the image (e.g., one filter detects edges and one filter detects dots), and we want to apply different filters independently on the input image. The result of applying a given filter on a single input image is called a channel. We can stack the channels to create a 3D structure, as shown in Figure 12.8.

Figure 12.8: Each filter creates one channel. The output of a convolutional layer has multiple output channels.

Next, let’s imagine that we want to build a deep neural network with multiple convolutional layers (state-of-the-art CNNs have 100 or even 1000 layers!). A typical convolutional layer in the middle of the network will have several input channels (equivalent to the number of output channels from the previous layer) and multiple output channels. How can we determine the number of filters needed?

Figure 12.9: A convolutional layer which has multiple input and multiple output channels.

| Layer k     | Channel 6     |
| ----------- | ------------- |
| Layer k + 1 | Channel 5     |
| Channel 4   | Out Channel 3 |
| Channel 3   |               |
| Channel 2   | Out Channel 2 |
| Channel 1   | Out Channel 1 |

In Channel 1



# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

Image in the v-th output channel as:

(v)    nin              (u,v) (u)
yi,j   = g  ∑  ∑        wr,s  xi+r,j+s    (12.3)
u=1 −k≤r,s≤k

where nin is the number of input channels, X(u) is the image at the u-th input channel, Y(v) is the image at the v-th output channel, and W(u,v) is the filter between the u-th input channel and the v-th output channel.

# Example 12.2.6

Assume there are 6 input channels and 3 output channels, and the filter size is 5 × 5. Then for every 6 × 3 pair of input and output channel, we have a kernel of weights of size 5 × 5, so there are a total of 6 × 3 × 5 × 5 = 450 weights.

# 12.2.6 Pooling

Pooling is another popular way to reduce the size of the output of a convolutional layer. In contrast to stride, which applies convolution operation every s pixels, pooling partitions each image (channel) to patches of size ∆ × ∆ and performs a reduction operation on each patch. You can think of this as similar to what happens when you lower the resolution of an image. The reduction operation can either involve taking the max of all the values in the patch (“max-pooling”):

yi,j   = max   X(i−1)·∆+r,(j−1)·∆+s
1≤r,s≤∆

or taking the average of all the values in the patch (“mean-pooling”):

1  ∆
yi,j   = ∆2 ∑ X(i−1)·∆+r,(j−1)·∆+s
r,s=1

The pooling operation can reduce the image size by a factor of ∆2. If the input image is of size m × n, the size of the image after pooling will be ⌊m/∆⌋ × ⌊n/∆⌋.

# Example 12.2.7

If the size of an input image to a pooling layer is 6 × 6 and ∆ = 2, then the output is of size 3 × 3.

# 12.2.7 A Full Convolutional Neural Network

Let’s put everything together and consider a full convolutional neural network. Figure 12.11 shows a typical example of a convolutional neural network. A convolutional neural network typically begins by stacking multiple convolutional layers and pooling layers. Each convolutional layer has its own kernel size and number of output channels; similarly, each pooling layer has its own kernel size. This is




# Convolutional Neural Network

# Figure 12.10: Max-pooling vs mean-pooling.

| Max-Pooling                          | Avg-Pooling |
| ------------------------------------ | ----------- |
| 5 1 -2 5 -2 0 2 3 1 4 4 -2 1 4 4 3 2 | 2 0 4 1     |

# Figure 12.11: A illustration of a full convolutional neural network.

| Conv\_1                    | Conv\_2                    |
| -------------------------- | -------------------------- |
| Convolution (5 x 5) kernel | Convolution (5 x 5) kernel |
| Max-Pooling (2x 2)         | Max-Pooling (2x 2)         |
| ReLU activation            | ReLU activation            |

INPUT: (28 x 28 x 1) → (24 x 24 x n1) → (12 x 12 x n1) → (8 x 8 x n2) → (4 x 4 x n2) → OUTPUT: n3 units

Followed by several fully-connected layers at the end. Since the output images of convolutional layers are 2-dimensional, it is customary to “flatten” the images into a 1D vector (i.e., append one row after another) before applying the fully connected layers. Intuitively, we can think of the convolutional and pooling layers as learning interesting image features (e.g., stripes or dots) while the fully-connected layers map these features to output classes (e.g., zebras have a lot of stripes).

All the weights in a convolutional neural network (including weights in kernels, fully-connected layers) can be learned via the backpropagation algorithm in Chapter 11. Again, modern deep learning libraries (e.g., PyTorch, TensorFlow) have all the convolutional and pooling layers implemented and can compute gradients.



# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

Finally, the above convolutional neural network is still a simple network, compared to modern convolutional neural networks. Interested students can look up architectures such as AlexNet, Inception, VGG, ResNet and DenseNet.

# 12.2.8 Designing a Convolutional Network

While we described convolutional nets above, we did not give a good explanation of why they are well-suited to solve vision tasks. Working through the next few examples will help you understand their power. The idea is that convolution is a parameter-efficient architecture that can “search for patterns” anywhere in the image. For example, suppose the net has to decide if the input image has a triangle anywhere in it. If the net were fully connected, it would have to learn how to detect triangles centered at every possible pixel location (i, j). By contrast, if a simple convolutional filter can detect a triangle, we can just replicate this filter in all patches to detect a triangle anywhere in the image.

Now consider the CNN architecture in Figure 12.12. The architecture has two convolutional layers, the first with a ReLU activation function, and the second with a sigmoid activation function. We will choose an appropriate convolutional weight and bias such that the architecture can detect a particular simple visual pattern.

Example 12.2.8. The input to the network is a gray-scale image of size 8 × 8 (1 channel), and each pixel takes an integer value between 0 and 255, inclusive. If at least one pixel of the image has value exactly 255, the output of the CNN should have a value close to 1 and otherwise the output should have a value close to 0.

We will now solve Example 12.2.8 by individually configuring the parameters for each convolutional layer in Figure 12.12.

| Input X  | Conv 1 | Image Y | Conv 2 | Output o | σ    |
| -------- | ------ | ------- | ------ | -------- | ---- |
| Output o |        |         |        |          | ReLU |





# Convolutional Neural Network

The convolutional layer will have a 1 × 1 filter of weight 1, a bias of −254, and a ReLU activation function. The convolution will be applied with no padding, and with stride 1. That is, the (i, j)-th entry of the output image of the first convolutional layer will be

yi,j = ReLU(xi,j − 254)            1 ≤ i, j ≤ 8

where xi,j is the (i, j)-th entry of the input image. Notice that this value is zero everywhere, except if xi,j = 255, in which case yi,j takes the value 1. That is,

yi,j =  1       xi,j = 255

0       otherwise

See Figure 12.13 to see the effect of this choice of convolutional layer on a sample image. We see that we have now successfully identified the pixels in the input image that take the value 255.

Figure 12.13: The effect of the choice of the first convolutional layer for Example 12.2.8 on a sample image. Only a portion of the image is shown.

| 0  | 100 | 200 | −254 | −154 | −54 | 0  | 0 | 0 |
| -- | --- | --- | ---- | ---- | --- | -- | - | - |
| 50 | 150 | 250 | −204 | −104 | −4  | −→ | 0 | 0 |
| 55 | 155 | 255 | −199 | −99  | 1   | 0  | 0 | 1 |

Next, consider the second convolutional layer with a 8 × 8 filter of all weights equal to 10, a bias of −5, and a sigmoid activation function. The convolution will be applied with no padding, and with stride 1.6 The output, before the sigmoid, will be

o = ∑i,j=1 8 10yi,j − 5

Notice that this value is −5 if and only if there is no pixel such that yi,j = 1 (i.e., xi,j = 255). If there is one such pixel, the output is 5; if there are two, the output is 15. The important thing is, the output is at least 5 if there is at least one pixel in the input image whose value is 255, and otherwise the output is ≤ 0. That is

o =     ≥ 5    ∃(i, j) :    xi,j = 255

−5     otherwise

Finally, when we apply the sigmoid function, the final output of the model will be

o = σ(o) =            ≥ 0.99       ∃(i, j) : xi,j = 255

0.01         otherwise

This is exactly what we wanted in Example 12.2.8.





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

# Example 12.2.9

The input to the network is a gray-scale image of size 8 × 8 (1 channel), and each pixel takes an integer value between 0 and 255, inclusive. If any part of the input image contains the following pattern:

∗       255       ∗
255     255      255                           (12.4)
∗       255       ∗

the output of the CNN should have a value close to 1 and otherwise the output should have a value close to 0.

We use the same architecture as in Figure 12.12, but now with a different choice of parameters for the convolutional layers. The first convolutional layer will have a 3 × 3 filter with the following weights:

0     1      0
1     1      1
0     1      0

a bias of −1274, and a ReLU activation function. The convolution will be applied with no padding, and with stride 1. That is, the (i, j)-th entry of the output image of the first convolutional layer will be

yi,j = ReLU(xi−1,j + xi,j−1 + xi,j + xi,j+1 + xi+1,j − 1274)  2 ≤ i, j ≤ 7

where xi,j is the (i, j)-th entry of the input image. 7 Notice that this value is zero everywhere, except if xi,j + xi−1,j + xi,j−1 + xi,j+1 + xi+1,j = 1275, in which case yi,j takes the value 1. This can only happen if xi−1,j = xi,j−1 = xi,j = xi,j+1 = xi+1,j = 255. That is, if the input image has the pattern in (12.4) centered around (i, j).

yi,j =  1  Pattern in (12.4) exists at (i, j)
0  otherwise

See Figure 12.14 to see the effect of this choice of convolutional layer on two sample images.

Figure 12.14: The effect of the choice of first convolutional layer for Example 12.2.9 on two sample images. Only a portion of the images is shown.

|     |     | 0   | 255 | 0  | \* | \* | \* | \* | \* | \* |
| --- | --- | --- | --- | -- | -- | -- | -- | -- | -- | -- |
| 255 | 250 | 255 | −→  | \* | −4 | \* | −→ | \* | 0  | \* |
|     |     | 0   | 255 | 0  | \* | \* | \* | \* | \* | \* |
|     |     | 0   | 255 | 0  | \* | \* | \* | \* | \* | \* |
| 255 | 255 | 255 | −→  | \* | +1 | \* | −→ | \* | 1  | \* |
|     |     | 0   | 255 | 0  | \* | \* | \* | \* | \* | \* |




# Convolutional Neural Network

Next, consider the second convolutional layer with a 6 × 6 filter of all weights equal to 10, a bias of −5, and a sigmoid activation function. The convolution will be applied with no padding, and with stride 1. The output, before the sigmoid, will be

7

o = ∑ 10yi,j − 5

i,j=2

Notice that this value is −5 if and only if there is no pixel such that yi,j = 1 (i.e., the pattern exists at (i, j)). If there is one such pixel, the output is 5; if there are two, the output is 15. The important thing is, the output is at least 5 if there is at least one copy of the given pattern, and otherwise the output is ≤ 0. That is

o =

≥ 5  ∃(i, j) : Pattern in (12.4) exists at (i, j)

−5   otherwise

Finally, when we apply the sigmoid function, the final output of the model will be

o = σ(o) =

≥ 0.99  ∃(i, j) :  Pattern in (12.4) exists at (i, j)

0.01    otherwise

This is exactly what we wanted in Example 12.2.9.

# 12.3 Backpropagation for Convolutional Nets

A convolutional neural network is a special case of a feedforward neural network where we use convolutional layers, instead of fully-connected layers as in Chapter 11. Therefore, we can apply the same basic idea of backpropagation so that we can run the gradient descent algorithm, although the details of the calculation are slightly different.

The biggest difference is that in a fully-connected layer, each weight is used exactly once, while in a convolutional layer, each weight is applied multiple times throughout the input image.8 This phenomenon is also known as weight sharing. But the basic idea is the same — identify all paths through which the corresponding weight affects the output of the model and add up the amount of effect for each path.

Figure 12.15 shows a portion of a sample neural network where weight sharing occurs. That is, the same weight w is used between the following four pairs of nodes: (x1, y1), (x2, y2), (x2, y3), (x3, y4). If we wanted to find the gradient ∂o/∂w, we need to consider the four paths that the weight w affects the output: w → yi → o where 1 ≤ i ≤ 4.





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

(1) y1 · · · Figure 12.15: A sample neural network where weight sharing occurs.

w1, w2, w3, w4 are the copies of the same weight w.

x

w2 y2 · · ·

x2 w3 o

y3 · · ·

x3 w4

y4 · · ·

What we will do is consider the four copies of the weight w as separate weights that will be denoted as w(i) where 1 ≤ i ≤ 4. Since these weights are only used in one place in the layer, we are already familiar with computing the gradients ∂o/∂w(i). Then we will add (or pool) these values to get the gradient ∂o/∂w. This works because we can think of each w(i) as a function of w where w(i) = w. Then by Chain Rule, we have

∂o / ∂w = ∑i=14 ∂o / ∂w(i) · ∂w(i) / ∂w = ∑i=14 ∂o / ∂w(i)

# 12.3.1 Deriving Backpropagation Formula for Convolutional Layers

In this subsection, we derive the backpropagation formula for a convolutional layer. (As in many other places, if your instructor did not teach it in COS 324, consider this to be advanced reading.) Recall that in a fully-connected layer (without an activation function), which computes hk = W(k)h(k−1), the gradient with respect to a particular weight w(k) can be simply computed as

∂ℓ / ∂w(k) = ∂h(k) / ∂w(i) · ∂ℓ / ∂h(k) = ∂h(k) / ∂w(i) · hj

This is because the weight w(k) is only used to compute h(k) out of all nodes in the next hidden layer. In comparison, consider a convolutional layer, which computes an output image Y ∈ Rn×n from an input image X ∈ Rm×m and filter W ∈ R(2k+1)×(2k+1). Notice that the weight wi,j is used to compute all of the pixels in the output image. Therefore, we just need to add (or pool) the gradient flow from each of these paths. The gradient with respect to a particular weight





# Convolutional Neural Network

wi,j will be

∂∂ℓ         =  ∂ℓ      · ∂y1,1            + ∂ℓ     ·     ∂y1,2    + . . . +  ∂ℓ     · ∂y1,n
wi,j   ∂y1,1       ∂wi,j            ∂y1,2          ∂wi,j               ∂y1,n    ∂wi,j

+   ∂∂ℓ    · ∂y2,1            +    ∂ℓ  · ∂y2,2        + . . . +  ∂ℓ     · ∂y2,n
y2,1  ∂wi,j          ∂y2,2    ∂wi,j                   ∂y2,n    ∂wi,j

+ . . .

+   ∂∂ℓ          · ∂yn,1 +      ∂ℓ     · ∂yn,2 + . . . +         ∂ℓ       · ∂yn,n
yn,1    ∂wi,j      ∂yn,2      ∂wi,j                   ∂yn,n      ∂wi,j

Assuming there is zero padding, this can be calculated as

∂∂ℓ        =       ∂ℓ       · xi+1,j+1 +  ∂ℓ            · xi+1,j+2 + . . . +      ∂ℓ         · xi+1,j+n
wi,j        ∂y1,1                  ∂y1,2                                     ∂y1,n

+     ∂∂ℓ        · xi+2,j+1 +         ∂ℓ     · xi+2,j+2 + . . . +     ∂ℓ        · xi+2,j+n
y2,1                       ∂y2,2                           ∂y2,n

+ . . .

+     ∂∂ℓ        · xi+n,j+1 +  ∂ℓ            · xi+n,j+2 + . . . +      ∂ℓ         · xi+n,j+n
yn,1                ∂yn,2                                     ∂yn,n

Notice that the equation above can be rewritten as

∂∂ℓ     =         ∑    ∂ℓ        · xi+r,j+s                     (12.5)
wi,j       1≤r,s≤n ∂yr,s

That is, the Jacobian matrix ∂ℓ/∂W is the output when applying a convolution filter ∂ℓ/∂Y to the input matrix X.

Similarly, we can try to calculate the Jacobian matrix with respect to the input matrix X. Each input pixel xi,j is used to calculate the output pixels yi+r,j+s where −k ≤ r, s ≤ k. The gradient with respect to a particular input pixel will be

∂∂ℓ     =            ∂ℓ           · ∂yi−k,j−k       + . . . +          ∂ℓ  · ∂yi−k,j+k
xi,j             ∂yi−k,j−k            ∂xi,j                    ∂yi−k,j+k       ∂xi,j

+        ∂y ∂ℓ              · ∂yi−k+1,j−k       + . . . +          ∂ℓ            · ∂yi−k+1,j+k
i−k+1,j−k               ∂xi,j                       ∂yi−k+1,j+k           ∂xi,j

+ . . .

+        ∂y ∂ℓ            · ∂yi+k,j−k    + . . . +             ∂ℓ  · ∂yi+k,j+k
i+k,j−k            ∂xi,j                    ∂yi+k,j+k          ∂xi,j




160    introduction to machine learning lecture notes for cos 324 at princeton university

Assuming zero padding, this is calculated as

∂∂ℓ      =   ∂ℓ     · wk,k + . . . +    ∂ℓ   · wk,−k
xi,j    ∂yi−k,j−k                      ∂yi−k,j+k

+  ∂y −∂ℓ  · wk−1,k + . . . + ∂y        ∂ℓ  · wk−1,−k
i  k+1,j−k                   i−k+1,j+k
+ . . .

+  ∂y ∂ℓ   · w−k,k + . . . + ∂y ∂ℓ      · w−k,−k
i+k,j−k                 i+k,j+k

which can be rewritten as

∂∂ℓ     =  ∑            ∂ℓ  · w−r,−s       (12.6)
xi,j       −k≤r,s≤k ∂yi+r,j+s

That is, the Jacobian matrix ∂ℓ/∂X is the output when applying the horizontally and vertically inverted image of W as the convolutional filter to the input matrix ∂ℓ/∂Y.

# 12.4    CNN in Python Programming

In this section, we discuss how to write Python code to implement Convolutional Neural Networks (CNN). As usual, we use the numpy package to speed up computation and the torch package to easily design and train the neural network. We also introduce the torchvision package:

- torchvision: This package focuses on computer vision applications and is integrated with the broader PyTorch framework. It provides access to pre-built models, popular datasets, and a variety of image transform capabilities. 9 Documentation is available at https://pytorch.org/vision/stable/index.html

The following code sample implements a CNN and trains it on a single image.

# import necessary packages
import  random
import  numpy as np
import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torchvision
import  torchvision.transforms as transforms
from torch.utils.data     import     DataLoader

# set random seeds to ensure reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)





# Convolutional Neural Network

# Load CIFAR10 Data

transform = transforms.Compose(
[transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
transform=transform)

# Helps Iterate Through the Train/Test Data in Batches

train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True,
num_workers=0)
test_loader = DataLoader(dataset=test_data, batch_size=8, shuffle=False,
num_workers=0)

# Define the CNN Architecture

class ConvNet(nn.Module):
def __init__(self):
super(ConvNet, self).__init__()
# Conv2d takes # input channels, # output channels, kernel size
self.conv1 = nn.Conv2d(3, 3, 5)
self.pool1 = nn.MaxPool2d(2, 2)
self.conv2 = nn.Conv2d(3, 16, 5)
self.pool2 = nn.AvgPool2d(2, 2)
self.fc1 = nn.Linear(16*5*5, 120)
self.fc2 = nn.Linear(120, 10)

def forward(self, x):
x = F.relu(self.conv1(x))
x = self.pool1(x)
x = F.relu(self.conv2(x))
x = self.pool2(x)
x = x.view(-1, 16*5*5)
x = F.relu(self.fc1(x))
x = self.fc2(x)

return x

# Extract One Image from the Dataset

images, labels = next(iter(train_loader))
image = images[0].unsqueeze(0)

# Forward Propagation

net = ConvNet()
output = net(image)

# Choose the Optimization Technique to Invoke

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Backpropagation

loss = torch.norm(output - torch.ones(output.shape[1]))**2
loss.backward()
optimizer.step()
optimizer.zero_grad()

# As Usual, We Start by Importing Packages

import random
import numpy as np
import torch
import torch.nn as nn





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

The DataLoader class helps iterate through a dataset in batches. Next, we fix all random seeds to ensure reproducibility.

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

Recall that programming languages on a classical computer can only implement pseudorandom methods, which always produce the same result for a given seed. Then we load the CIFAR-10 dataset.

transform = transforms.Compose(
[transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
transform=transform)

The CIFAR-10 dataset contains simple images of a single object, and the images are labeled with the category of the objects they contain. Note that we normalize the dataset with a mean of 0.5 and standard deviation of 0.5 per color channel. Figure 12.16 shows a sampling of images from the dataset after the normalization.

| plane | horse | horse | deer | bird | at | horse | deer |
| ----- | ----- | ----- | ---- | ---- | -- | ----- | ---- |

Figure 12.16: Sample images from the CIFAR10 dataset.

Next we create DataLoader objects to help iterate through the dataset in batches. Each batch will consist of 8 images and 8 labels.

train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True,
num_workers=0)
test_loader = DataLoader(dataset=test_data, batch_size=8, shuffle=False,
num_workers=0)

Then we define our CNN architecture in the ConvNet class.

class ConvNet(nn.Module):
def __init__(self):
super(ConvNet, self).__init__()
# conv2d takes # input channels, # output channels, kernel size
self.conv1 = nn.Conv2d(3, 3, 5)
self.pool1 = nn.MaxPool2d(2, 2)
self.conv2 = nn.Conv2d(3, 16, 5)
self.pool2 = nn.AvgPool2d(2, 2)



# Convolutional Neural Network

self.fc1 = nn.Linear(16*5*5, 120)

self.fc2 = nn.Linear(120, 10)

def forward(self, x):

x = F.relu(self.conv1(x))

x = self.pool1(x)

x = F.relu(self.conv2(x))

x = self.pool2(x)

x = x.view(-1, 16*5*5)

x = F.relu(self.fc1(x))

x = self.fc2(x)

return x

Just like the FFNN code from the previous chapter, we define all the layers and activations we are going to use in the constructor. Note that in addition to instances of the nn.Linear class and the nn.ReLU class, we also make use of classes like nn.Conv2d and nn.MaxPool2d which are specifically designed for CNNs.

We extract one training image with the following code.

images, labels = next(iter(train_loader))

image = images[0].unsqueeze(0)

The unsqueeze() function adds one dimension to the training data. This is called a batch dimension. Normally, we would run the training in batches, and the size of the data along the batch dimension will be equal to the number of images in each batch. Here, we only use one image for the sake of exposition.

We can now run forward propagation on a sample image with the code below.

net = ConvNet()

output = net(image)

We then implement the squared error loss. Alternatively, we could have chosen the cross-entropy loss or any other valid loss function.

loss = torch.norm(output - torch.ones(output.shape[1]))**2

Next, we calculate the gradients of the loss with the following line of code

loss.backward()

and update each of the parameters according to the Gradient Descent algorithm with the following line.

optimizer.step()

Finally, we reset the values of the gradients to zero with the following code.

optimizer.zero_grad()





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

Recall as discussed in the previous chapter that failing to do so will cause unintended training during subsequent iterations of backpropagation. Here, we called the zero_grad() function at the end of one iteration of backpropagation, but it may be a good idea to call this function right before calling backward(), just in case there are already gradients in the buffer before program execution (e.g., if someone was working with the model beforehand in the interpreter).

In this section, we only showed how to run forward propagation and backpropagation on a single data point. In general, we train the model on the entire dataset multiple times. A single pass over the entire dataset is called an epoch.




# Part IV

# Reinforcement Learning


NO_CONTENT_HERE



# 13

# Introduction to Reinforcement Learning

This part of the course concerns Reinforcement Learning (RL), the conceptual underpinning of several modern technologies such as self-driving technologies in new cars. It is the third major category of machine learning, in addition to the two previously seen categories of supervised and unsupervised learning. In class we saw a video of robots (made by Boston Dynamics) doing parkour, dancing, and overall doing a pretty good job of imitating the peak human physique. That is also achieved via RL.

The basic idea of RL involves the concept of an agent learning to make a sequence of actions in a dynamic environment. At each discrete time step, the agent is able to take one of a menu of actions. Each choice of action leads to changes in the state of the world (i.e., the agent and its surroundings). The agent has an internal representation of the current and potential states of the world (e.g., using vision or other sensing modules). Under this setting, the agent takes a sequence of actions towards a certain goal.

The world contains uncertainty due to a variety of factors. For instance, there may be other agents in the environment that also take actions to their own benefit, or the sensing modules may be imperfect. Thus taking the same action from the same state of the world may lead to different evolution of state in the future — that is, RL is non-deterministic.

In this chapter, we introduce the basic elements of RL using real-world examples, and what it means for the agent to act optimally. Chapter 14 focuses on the setting where the underlying environment (e.g., the number of states, the current state, the probability distribution) is completely known to the agent.1 In Chapter 15, we will present the case where the environment is not fully available to the agent, and the agent learns about the environment while also learning to act in it.2

1 Think of playing a game where you know the complete set of rules.

2 Think of playing a Role-playing Game (RPG) where you need to unlock parts of the map by advancing the story.




168   introduction to machine learning lecture notes for cos 324 at princeton university

# 13.1 Basic Elements of Reinforcement Learning

Now we formalize several of the basic elements of reinforcement learning that were sketched above.

# 13.1.1 States and Actions

There is a finite set S of states, and the entire system agent + environment exists in one of these states at any time. At each state s ∈ S, the agent makes an action a ∈ As, where As is the set of allowed actions at state s. We denote A = s∈S As to be the set of all possible actions in the whole RL environment.

# Example 13.1.1.

Consider a game of chess. Each state s can be represented as a pair (C, p) where C denotes the current configuration of pieces and p denotes the player to play next. For example, “white king at e1, black king at e8, and it is white turn to move” would be a possible state s of the game. An action a is a valid movement of a piece, given a state of the game. For example, “white king to e2” (i.e., Ke2) would be a possible action of the agent playing white in state s.

# Example 13.1.2.

Self-driving cars, like those built by Tesla, are becoming increasingly popular. Let’s imagine how we could construct a state diagram for the task of driving autonomously. Each state can be represented by the current configuration of a number of factors (e.g., the car speed, distance from lane boundaries, distance to nearest vehicle, etc.) Possible actions include increasing/decreasing speed, changing gear, changing direction, changing lane, etc.

# 13.1.2 Modeling Uncertainty via Transition Probabilities

As mentioned, the agent has many sources of uncertainty in its knowledge about the environment, and we can use concepts from probability to model uncertainty. Suppose S = {s1, s2, . . . , sn} contains n states. When the agent takes action a while in state s, it will transition into another (potentially the same) state s′. The catch is: the agent does not know exactly which state it will end up in. Instead, there is a probability pi of ending up in state si for each si ∈ S. Here ∑i pi = 1, meaning each (state, action) pair is associated with a probability distribution over the next state that the agent will enter. Formally, we define it as follows:

# Definition 13.1.3 (Transition Probabilities).

Given a state s ∈ S and an action a ∈ As, there is an associated transition probability p(∗ | s, a) distributed over S such that state s′ ∈ S happens with probability p(s′ | s, a)




# Introduction to Reinforcement Learning

When action a is taken at state s and ∑ p(s′ | s, a) = 1. If p(s′ | s, a) > 0, we say that the state s′ is reachable from s when action a is taken.

In general, not all states are reachable, given a state s and an action a. That is, some transition probability p(s′ | s, a) is zero. For these states, it is conventional to leave out the corresponding transitions when representing the RL environment as a state diagram as in Figure 13.1 or Figure 13.2.

# Example 13.1.4

Consider the state diagram shown in Figure 13.1. This is a special case where there is only one action a in the set A. In other words, the agent is not making any choices; instead, it is just following probabilistic transition over time steps. To calculate the probability of reaching state s3 from s0, we note there are two different paths. The first path is s0 − s1 − s3 and the second path is s0 − s2 − s3. We thus calculate the probabilities of each of these paths and note that the overall probability of reaching s3 will be the sum of both: 0.2 · 0.7 + 0.8 · 0.4 = 0.46.

|    | S1  | S4  |
| -- | --- | --- |
| S0 | 0.2 | 0.7 |
| S2 | 0.6 | S3  |
|    | 0.8 | 0.4 |

Now we consider an example where there is more than one action to make. In this case, each action induces a different probability distribution on the set of states, so we need to draw a diagram for each option.

# Example 13.1.5

We can model a baby learning to walk through RL. As shown in Figure 13.2, we can define the state s0 = standing but feeling unsteady, s1 = standing and feeling secure, and s2 = on ground. The baby has two actions to take: a = not grab onto nearest support and a′ = grab onto nearest support. The state diagram on the top represents the transition probabilities when the baby takes the action a. See that the baby has a high chance of entering state s2 — falling to the ground. On the other hand, the state diagram on the bottom represents the transition probabilities when the baby takes the action a′. The baby now has a high chance of entering state s1 — standing securely on the ground. The two actions have different probability distributions associated with the relevant transitions.

# Example 13.1.6

Mechanical ventilators are used to stabilize breathing for patients. Suppose we wished to construct a state diagram. We could define



# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

# Figure 13.2: An example diagram showing how an action determines the probability of outgoing transitions.

| S0 | 0.2 | S1 | 0.8 | S2 |
| -- | --- | -- | --- | -- |
| S0 | 0.7 | S1 |     | S2 |
| S0 | 0.3 |    |     |    |

States that consider the pressure and CO2 level in the patient’s level for the past k seconds. Actions might include adjusting the flow rate of oxygen via valve settings as needed. Possible transitions might include the typical mechanical response of the lungs, or unexpected spasms. Finally, we can define the goal to be maintaining steady pressure in the patient without “overshooting” and causing damage.

# 13.1.3 Agent’s Motivation/Goals

In general, an agent is a participant in RL models driven by the need to maximize “rewards.” In a probabilistic setting, the agent wishes to maximize their expected rewards. In a natural setting, the “rewards” could be innate satisfaction, such as getting to eat food, being entertained, etc. But in the usual artificial settings such as robots and self-driving cars, rewards are sprinkled by the system designer into the framework. Some examples appear later.3

At each step, the agent takes an action, and is given a reward (which could be negative, i.e., is a punishment) based on the action, current state, and next state.

# Definition 13.1.7 (Reward).

For each valid 3-tuple (a, s, s′) where s′ ∈ S is a state reachable from state s ∈ S by taking action a ∈ A, we define a corresponding reward r(a | s, s′) ∈ R.

# Example 13.1.8 (Example 13.1.5 revisited).

When the baby stands and feels secure after grabbing onto something, the parents applaud the baby, and the baby receives a positive reward: r(a′ | s0, s1) = 5. When the baby feels secure without grabbing onto the nearest support, the parents feel even prouder and the baby gets a more positive reward: r(a | s0, s1) = 10. When the baby falls to the ground, the baby feels pain and receives negative reward:





# Introduction to Reinforcement Learning

r(a | s0, s2) = r(a′ | s0, s2) = −5.

Typically, the designer of an RL model gets to define the rewards throughout the framework based on the designer’s judgment. For instance, in Example 13.1.2, we might design an RL model such that if the car drifts into an adjacent lane, we assign a negative reward. If another vehicle is in the lane, we might assign an even larger negative reward. This will induce an RL model to “learn” the proper way to driving — staying in lane.

# 13.1.4 Comparison with NFA

Recall the Non-deterministic Finite Automata (NFA) you learned in COS 126. In an NFA, there is a finite number of states, and for each state, we know the set of next possible states, based on the next input character.

Figure 13.3: A sample Non-deterministic Finite Automata. Source: Introduction to the Theory of Computation by Michael Sipser.

We can consider the following analogy between RL and NFA — there is someone behind an NFA, who can observe its current state and type in the next input character. This person will be called an agent, and the choice of input character that is typed in will be called an action. Each action can lead to a finite set of next possible states, but because of some uncertainty in the world, the agent cannot specify which particular state will be the next one. This is similar to an NFA in the sense that the actions are non-deterministic. Also, just like in an NFA, the change in the current state is also referred to as a transition. One major difference between RL and NFA is that while an NFA only cares about the final state of the automata (i.e., whether it is an accept state or a reject state), in RL, the agent is given a reward after each transition. The goal of the agent will be to take a sequence





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

of actions so as to maximize the sum of the reward throughout the sequence of actions.

# 13.2 Useful Resource: MuJoCo-based RL Environments

Real-life robots with precise and reliable hardware can get very expensive to buy, let alone train. An easier playground for students (especially those trying to work with a single GPU on CoLab) is doing RL in a virtual environment. MuJoCo is a famous physics engine that allows creating virtual objects with somewhat realistic “joints” that can be commanded to move similar to real-life robots. OpenAI and DeepMind have open-source environments that allow experimentation in the MuJoCo environment. The official website gives a pretty good overview of the software:

Figure 13.4: An example of a MuJoCo L Walker.

MuJoCo is a physics engine that aims to facilitate research and development in robotics, biomechanics, graphics and animation, and other areas where fast and accurate simulation is needed. MuJoCo offers a unique combination of speed, accuracy and modeling power, yet it is not merely a better simulator. Instead it is the first full-featured simulator designed from the ground up for the purpose of model-based optimization, and in particular optimization through contacts.4

Source: https://mujoco.org.

One aspect of MuJoCo simulation involves a representation of a humanoid figure (i.e., the agent) learning how to navigate an obstacle course (i.e., the environment). Training videos are readily available online and show how the agent learns over time (sometimes, to comedic effect).

# Example 13.2.1

Let’s analyze the example of an agent navigating an obstacle course through an RL framework. The states can be considered to be




introduction to reinforcement learning

The set of coordinates, velocity, and acceleration for each limb, the velocity and acceleration for the motors in each joint, and the environment itself straight ahead. The actions can include the agent increasing or decreasing motor speed in their joints. Finally, the final goal is to stay upright, run forward at a reasonable pace, and avoid obstacles.

# 13.3  Illustrative Example: Optimum Cake Eating

Let’s consider an extended example which ties together the elements of RL discussed previously. Suppose you buy a small cake with three slices. The reward of eating one slice at one sitting is 1, but eating two or three slices at one sitting is 1.5 and 1.8 respectively. 5 This sense of diminishing rewards is known as the satiation effect.

Problem 13.3.1. Suppose you plan to eat the cake over a period of three days. What eating schedule will maximize the internal reward?

Now let’s introduce your roommate, who is oblivious to basic understandings of ownership and adheres to the “finders keepers” faith. We define the probability Pr[sneakily eats a slice overnight] = 1. To account for this uncertainty, we can create a look-ahead tree for different initial actions. We first consider the action where you decide to eat two out of the three slices on the first night. Successive states and associated probabilities are shown in the Figure 13.5.

Figure 13.5: The diagram (look-ahead tree) of the cake problem where you decide to eat two slices on the first night. Each state represents the number of slices remaining, and each action represents the number of slices eaten on one night.

Even though you eat only two out of the three slices during the first night, there is a 1/2 chance that your roommate eats the remaining slice overnight. Therefore, the action of “eating 2 slices” can lead to two possible states — “1 slice left” or “0 slices left” — each with probability 1/2.

# Example 13.3.2.

We can calculate the expected reward associated with eating two slices on the first night by analyzing the Figure 13.5. You first gain

| Action = "eat 2 slices" | Reward = 1.5 |           |   | 0.5 | 0.5 |
| ----------------------- | ------------ | --------- | - | --- | --- |
| Action = "eat 1 slice"  | 0            | 2nd night |   |     |     |
|                         | 0            | 3rd night |   |     |     |




# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

Reward of 1.5 by eating the two slices on the first night. Then with probability 1 (where the roommate does not eat the remaining slice overnight), you get to eat the last slice on the second night and gain additional reward of 1. With probability of 1 (where the roommate eats the remaining slice), you cannot gain anymore reward. That is, the expected reward is 1.5 + 0.5 · 1 + 0.5 · 0 = 2.

# Problem 13.3.3

Consider the result of Example 13.3.2. Would you prefer to take two slices on the first night or three slices?

We next consider the action where you decide to eat one out of the three slices on the first night. Successive states and associated probabilities are shown in Figure 13.6.

State = # slices remaining

Action = "eat 1 slice"

|            |     | 3   | 1st night |   |   |   |
| ---------- | --- | --- | --------- | - | - | - |
| Reward = 1 | 0.5 | 0.5 |           |   |   |   |
|            | 1   | 2   | 2nd night |   |   |   |

Figure 13.6: The diagram (look-ahead tree) of the cake problem where you decide to eat one slice on the first night.

The difficulty in this example in contrast to the Figure 13.5 is that if the roommate does not eat a slice after the first night, you have two slices at your disposal on the second night. You have two actions you can take in this “2 slices left” state — “eat 1 slice” (and hope the third slice is still there on the third night) or “eat 2 slices” — and it is not immediately obvious which one is more optimal. It turns out that the expected reward you can get from the remaining 2 slices is 1.5 for both options.

# Problem 13.3.4

Verify the previous claim that both options on the second night have the same expected reward.

# Example 13.3.5

Given the previous analysis and the look-ahead tree in the Figure 13.6, we note that the total expected reward is 1 + 0.5 · 1 + 0.5 · 1.5 = 2.25. You first receive a reward of 1 by eating 1 slice on the first night. Then with probability 1, the roommate eats one slice overnight, and you gain reward of 1 by eating the last slice on the second night. With the remaining probability 1, the roommate does not eat a slice, and you are expected to gain reward of 1.5 from the remaining 2 slices, regardless of the action you choose to take on the second night.

# Problem 13.3.6

Consider the result of Example 13.3.5. Would you prefer to take two slices on the first night or one slice?





# 14 Markov Decision Process

In this chapter, we formally introduce the Markov Decision Process (MDP), a way to formulate an RL environment. We then present ways to find the optimal strategy of an agent, provided that the agent knows the full details of the MDP — that is, knows everything about the environment.

# 14.1 Markov Decision Process (MDP)

Let’s review the key ingredients of RL. We have the agent, who senses the environment and captures it as the current state. There is a finite number of actions available at any given state, and taking an action a in state s will cause a transition to s′ with probability p(s′ | s, a). Each transition is accompanied by a reward r(a | s, si) ∈ R. Finally, the goal of the agent is to maximize the expected reward via a sequence of actions.

A Markov Decision Process (MDP) is a formalization of these concepts. It is a directed graph which consists of four key features:

- A set S which contains all possible states
- A set A which contains all possible actions
- For each valid tuple of action a and states s1, s2, there is an assigned probability p(s2 | s1, a) of transition to s2 if action a is taken in s1
- For each valid tuple of action a and states s1, s2, there is an assigned reward r(a | s1, s2), which is obtained if action a is taken to transition from s1 to s2

If a designed MDP has M actions and N states, we can specify the MDP by a table of transition probabilities (with MN2 numbers) and a table for rewards (with MN2 numbers).




176  introduction to machine learning lecture notes for cos 324 at princeton university

# 14.1.1  Revisiting the Cake Eating Example

Let’s return to the case study on eating cake from Subsection 13.3, and formally express it through a MDP. The set of states is given as S = {0, 1, 2, 3}, where each state represents the number of slices left. The set of actions is given as A = {1, 2, 3}, where each action represents the number of slices you choose to eat on a given night. Notice that reward only depends on how many slices you take, not how many slices are left after your roommate goes through the fridge. That is, we can define the reward r(a | s, ∗) for each a ∈ A to be the same for every s ∈ S where a is feasible.1

1 We still need to include the previous state s because not all actions are feasible at each state. For example, you can’t eat 2 slices when there is only 1 slice left.

# Example 14.1.1

Let’s revisit Example 13.3.2 as a motivating example. If we let a = 2, s1 = 3, and s2 = 0, then the probability of the specified transition is p(s2 | s1, a) = 0.5. The associated reward is r(a | s1, s2) = 1.5 as discussed earlier.

We are now ready to generalize to the a more complete MDP, which is shown in Figure 14.1. Note that every transition is labeled with its probability, associated action, and associated reward.

Figure 14.1: A more complete diagram of the cake problem when described as a MDP.

| 0          | 1        | 3        | 0.5 | 1        |
| ---------- | -------- | -------- | --- | -------- |
| 0.5        | a=1; r=1 | 0.5      | 0.5 |          |
| a=2; r=1.5 | 1        | 2        | 0.5 | a=1; r=1 |
| a=2; r=1.5 | 0.5      | a=1; r=1 |     |          |

# 14.1.2   Discounting the Future

The MDP describing cake eating in the previous subsection was acyclic.2 However, in general, MDPs can have directed cycles, and the agent’s actions can allow it to continuously collect rewards along that cycle. For instance, continuing our cake theme, we may have a scenario in which you receive a fresh cake every 3 days. But now we run into a problem: how can we calculate the expected reward when there is an unbounded number of steps?

The solution lies in the concept of future discounting. The basic idea is to reduce, or discount, the amount of reward we get from





# Markov Decision Process

Future steps. In an MDP, we represent this through a discount factor 0 &#x3C; γ ≤ 1 and an associated infinite sum. 3 This is related to notions of discounting commonly considered in economics.

# Definition 14.1.2 (Future Discounting)

If a reward rt is received at time t = 0, 1, 2, . . . , then the perceived value of these rewards rd, or the discounted reward, at t = 0 is:

rd = r0 + γr1 + γ2r2 + γ3r3 + · · ·

# Example 14.1.3

Consider the cake eating problem again and let rt denote the reward we get on night t. If the reward is discounted by a factor of γ every night, the total expected discounted reward E[total] can be rewritten as:

E[total] = E[r1] + γ · E[r2] + γ2 · E[r3]

Consider taking the action a = 2 on the first night. If γ = 0.9, then the expected discounted reward is:

1.5 + 0.9 · (0.5 · 1 + 0.5 · 0) = 1.95

This is the same as in Example 13.3.2 except the reward taken from the second night is discounted by a factor of 0.9. Now consider taking the action a = 1 on the first night and on the second night. If γ = 0.9, the expected discounted reward is:

1 + 0.9 · (0.5 · 1 + 0.5 · 1) + 0.92 · (0.52 · 1) = 2.1025

Here, we first take the reward of 1 on the first night without any discount factor. Then, we calculate the expected reward from the second night — 1 whether or not the roommate eats a slice — and discount it by a factor of 0.9. Finally, we calculate the expected reward from the third night — 1 only if the roommate did not eat any slice on the first two nights — and discount it by a factor of 0.92.

Note that in Definition 14.1.2, if each rt ∈ [−R, R] and if γ &#x3C; 1, then the magnitude of discounted reward of the infinite sequence has the following upper bound:

|rd| ≤ R(1 + γ + γ2 + · · · ) = R (14.1)

(14.1) is derived by considering the formula for the sum of an infinite geometric series, which we can invoke if γ &#x3C; 1. In general, γ is up to the system designer. A lower γ would imply that the agent places little importance on future rewards, whereas γ = 1 would imply that there is effectively no discounting.

# 14.2 Policy and Markov Reward Process

Now that we have discussed what an action is and what it does in an MDP, we want to specify what action an agent has to take in each state. This is known as a policy.




178  introduction to machine learning lecture notes for cos 324 at princeton university

# Example 14.2.1

Consider again the cake eating MDP example without a discount factor. We already established through Example 13.3.2 and Example 13.3.5 that to maximize the expected reward, you need to eat one slice per day until all slices are gone. That is, in any state j where j = 1, 2, 3, you need to take action 1.

In general, if S is the set of states, and A is the set of actions, then a policy (not necessarily the optimum) π can be defined as a function π : S → A

# Definition 14.2.2 (Policy)

If S is the set of states, and A is the set of actions, any function π : S → A is called a policy that describes which action to take at each state. In particular, each state s should only be mapped to a valid action a ∈ As at that state.

Recall that if there are M actions and N states, there are at most MN2 transitions in the graph of the MDP. Because a policy specifies one action per state, there are at most N2 transitions that remain when we choose a specific policy. Therefore, it can be understood that a policy trims out the MDP.

# 14.2.1  Markov Reward Process (MRP)

When we have an MDP and a fixed policy, we have what is called a Markov Reward Process (MRP). There are no more decisions to make; instead, all we need to do is take the action specified by the policy; probabilistically follow a transition into a new state; and collect the associated reward.

# Example 14.2.3

Let’s revisit Figure 14.1. If we fix the policy to be π(s) = 1 for any s ∈ S, we can focus our attention to the action a = 1. Then there are three trajectories that will lead from state 3 to state 0, based on what the roommate does overnight. The first trajectory is 3 → 1 → 0 with probability 0.5 × 1 and reward 1 + 1. The second trajectory is 3 → 2 → 0 with probability 0.5 × 0.5 and reward 1 + 1. The last trajectory is 3 → 2 → 1 → 0 with probability 0.5 × 0.5 × 1 and reward 1 + 1 + 1.

In general, when we fix a policy π and an initial state s, we can redraw the transition diagram of an MDP into a tree diagram for the MRP, where each node corresponds to a state, and each edge corresponds to a probabilistic transition. The top node represents the initial state, and each subsequent row of the tree represents the set of possible states after taking an action from their parent node.

# Example 14.2.4

We revisit Example 14.2.3. We now transform Figure 14.1 into a tree diagram for the MRP as shown in Figure 14.2. The top node is the initial state 3. The second row of the tree is all states that can be achieved by taking the action 1 at state 3, and so on.





# Markov Decision Process

# Figure 14.2: A tree representing the MRP in Example 14.2.3.

|       | r = 1   | p = 0.5 | 1 |   |   |
| ----- | ------- | ------- | - | - | - |
|       |         |         |   |   | 2 |
| r = 1 | p = 1   | 0       |   |   |   |
| r = 1 | p = 0.5 | 0       |   |   |   |
|       |         |         |   |   | 1 |
| r = 1 | p = 1   | 0       |   |   |   |

Note that in an MRP tree, the same state can appear multiple times, but each copy of the same state is identical — that is, the subtree rooted at each copy must be identical. In Figure 14.2, the state 1 appears twice in the tree. Every time it appears, it can only lead to state 0 with probability 1. This is simply the result of fixing a policy π — once we know the state we are in, we only have one choice for the action to take.

The policy also induces a value function on this tree. The value function assigns a value to each node of the tree, and each value intuitively measures how much reward the agent should expect to collect once the agent knows they have arrived at that node. By the observation from the previous paragraph, this expected reward is the same for two nodes if they are copies of the same state. Therefore, we can equivalently define the value function for each state s instead.

# Definition 14.2.5 (Value Function).

vπ(s), the value of state s under the policy π, is the expected discounted reward of a random trajectory starting from s. We can define this value by using the following recursive formula:

vπ(s) = ∑ p(s′ | s, π(s)) · r(π(s) | s, s′) + γvπ(s′) (14.2)

Computing the value function as in (14.2) is also known as the Bellman equation.

Let us unpack the intuition behind (14.2). Once we take action π(s) at state s, it will bring us to state s′ with probability p(s′ | s, a), immediately giving us a reward r(a | s, s′). Then, the expected reward from that point on is already captured by the value vπ(s′). We just need to apply the discount factor γ because we already took one time step to reach s′ from s.





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

On the other hand, if we pick any random trajectory starting from s, its next node will be some state s′ that is reachable from s. Therefore, the contribution of this particular trajectory to vπ(s) is accounted for when we sum over that particular s′.

# 14.2.2 Connection with Dynamic Programming

In COS 226, you may have seen an implementation of a bottom-up dynamic programming.

int[] opt = new int[n+1];
for (int v = 1; v &#x3C;= V; v++)
{
opt[v] = Integer.MAX_VALUE;                   ∞                       if v &#x3C; 0
for (int i = 1; i &#x3C;= n; i++)        OPT(ν) =  0                        if = 0
{
if (d[i] > v) continue;                      min { 1 + OPT(ν − di) } if υ > 0
if (opt[v] > 1 + opt[v − d[i]])              1 ≤ i ≤ n
opt[v] = 1 + opt[v − d[i]];
}
}

In such implementations, the algorithm divides the problem into subproblems arranged as directed acyclic graphs and computes “bottom-up.” The MDP from the cake eating problem is acyclic and our method using a look-ahead tree is similar to the dynamic programming algorithms. Therefore, it seems like we can apply a similar algorithm to the cake eating problem.

# Example 14.2.6

Consider Example 14.2.3 again, but now with a discount factor of 0.9. We will find the value vπ(s) of each state s by going bottom-up from the tree in Figure 14.2. We start by noticing that vπ(0) = 0 as can be seen from the bottom row. Then from the third node of the third row, we can calculate:

vπ(1) = 1 · (1 + 0.9 · 0) = 1

From the second node of the second row, we can calculate:

vπ(2) = 0.5 · (1 + 0.9 · 0) + 0.5 · (1 + 0.9 · 1) = 1.45

Finally, from the top node, we can calculate:

vπ(3) = 0.5 · (1 + 0.9 · 1) + 0.5 · (1 + 0.9 · 1.45) = 2.1025

But in general, the dynamic programming approach does not completely apply to MDP. The biggest assumption for dynamic programming algorithms is that the graph is acyclic, but MDPs are generally allowed to have directed cycles if we can return to the same state after a sequence of actions. Therefore, computing the expected reward for even a single policy π involves solving a system of linear equations.





# 14.3  Optimal Policy

Out of all choices for a policy, we are interested in the optimal policy, the one that maximizes the expected (discounted) reward. Surprisingly, it is known that there always exists a policy π* that obtains the maximum expected reward from all initial states simultaneously; that is π* = arg max vπ(s) for every state s. 4 The value function of the optimal policy is called the optimal value function and is often denoted as v*(s). Then we can express the optimal value function using (14.2) as:

v*(s) = max ∑ p(s′ | s, π(s))(r(π(s) | s, s′) + γvπ(s′))

This is just restating the fact that the optimal value of state s is the maximum of all possible values vπ(s) of s under a policy π — i.e., the Bellman equation evaluated with the values vπ(s′) of each child node s′ under that specific policy π.

But we can even go further than this result. It is known that the optimal value also satisfies the following:

v*(s) = max ∑ p(s′ | s, π(s))(r(π(s) | s, s′) + γv*(s′))  (14.3)




182  introduction to machine learning lecture notes for cos 324 at princeton university

Notice that vπ (s′) in the summation has now been replaced with v∗(s′). This property, known as the Bellman Optimality condition, states that the optimal value is even the maximum when the Bellman equation is evaluated with the values v∗(s′), regardless of the choice of the policy π.

Notice that the right-hand side of (14.3) only depends on the choice of the action a of the given state s, not any other states. Therefore, we can rewrite (14.3) as:

v∗(s) = max ∑ p(s′ | s, a)(r(a | s, s′) + γv∗(s′))



# Markov Decision Process

# Figure 14.5: Visual representation of the Gridworld.

| A   | B | +5 |
| --- | - | -- |
| +10 | B |    |

policy that looks at least near-optimal, and use its value as a lower bound for the optimal policy.

First, let v∗(s) denote the value vπ∗ (s) of state s for an optimal policy π∗. Since there is only one action to choose from at state A, we know that

v∗(A) = 10 + γv∗(A′) (14.6)

Now, at the state A′, one possible trajectory you can follow is “go up four steps” (each with reward 0) back to A. We know that the optimal value has to be at least as great as this value. That is

v∗(A′) ≥ γ4v∗(A) (14.7)

Combining (14.6) and (14.7), we get

v∗(A) ≥ 10 + γ5v∗(A)

If we solve for v∗(A), we get

v∗(A) ≥ 10 / (1 − γ) ≈ 24.4

The value iteration method discussed below is based on this intuition — we can provide a lower bound for the optimal policy by suggesting some potential policy. If we repeat this process, the lower bound for the optimal policy can only go up. At the end of the section, we will prove that this process converges to the actual optimal value.

# 14.3.2 Value Iteration Method

Value Iteration is a method guaranteed to find the optimal policy. At each step of the iteration, we are given a lower bound on the optimal values of each state s. Using the values of the immediate children nodes in the tree, we can compute an improved lower bound on v∗(s).





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

# Example 14.3.2

See Figure 14.6. Suppose there are two actions to take at state s. The first action, labeled as blue, will lead to state s1 with reward −1 with probability 0.5 and s3 with reward −1 with probability 0.5. The second action, labeled as red, will lead to state s2 with reward 2 with probability 1. The discount factor is given as 0.6. Now assume that someone tells us that they know a way to get an expected reward of 12 starting from s1, 1 from s2, and 4 from s3, regardless of the choice of initial action at s. In other words, the optimal values for these three states are lower bounded by:

v*(s1) ≥ 12, v*(s2) ≥ 1 and v*(s3) ≥ 4.

Using this fact, we consider two strategies7 — (1) first take action blue at state s and play optimally thereon based on the other person’s knowledge; (2) first take action red at state s and play optimally thereon. The lower bound for the expected reward for each of the two strategies can be computed as:

vblue(s) ≥ 0.5 × (−1 + 0.6 × 12) + 0.5 × (−1 + 0.6 × 4) = 3.8

vred(s) ≥ 1.0 × (2 + 0.6 × 1) = 2.6

The Bellman Optimality condition in (14.4) guarantees that the optimal policy is at least as good as either of these strategies. Therefore v*(s) has to be larger than both vblue, vred; that is, v*(s) ≥ 3.8.

| State       | Action |    |   |   |   |
| ----------- | ------ | -- | - | - | - |
| Probability | Reward |    |   |   |   |
| s1          | 0.5    | -1 |   |   |   |
| s2          | 1.0    | 2  |   |   |   |
| s3          | 0.5    | -1 |   |   |   |

In general, the value iteration algorithm looks like:

1. Initialize some values v0(s) for each state s such that we are guaranteed v0(s) ≤ v*(s)
2. For each time step k = 1, 2, . . ., and for each state s, use the values vk(s′) of the immediate children s′8 to compute an updated value vk+1(s) such that vk+1(s) ≤ v*(s).
3. When k → ∞, each vk(s) will converge to the optimal value v*(s).

Recall from (14.1) that if all transition rewards are within [−R, R], then the expected rewards at any state for any policy lies in −R/(1−γ), R/(1−γ).

7 This is not necessarily a policy because the second part of playing optimally may require you to return to state s and take an action that is inconsistent with your initial choice of action.

8 These values vk(s) maintained by the algorithm is not necessarily associated with a specific policy. They are just a lower bound for the optimal value v*(s) that will be improved over time.





# Markov Decision Process

Therefore, we can set the initial value v0(s) = − R to be the lower bound for each state s.9 After the k-th iteration of the algorithm, we will maintain a value vk(s) for state s, where the condition vk(s) ≤ v∗(s) is maintained as an invariant. Now at the (k + 1)-th iteration, the algorithm will update the values at each state s as the following:

vk+1(s) = max ∑ p(s′ | s, π(s)) · r(π(s) | s, s′) + γvk(s′) (14.8)

This is just the Bellman equation evaluated with the values vk(s′) of each children node.

# Example 14.3.3 (Example 14.3.1 revisited)

Say we start the value iteration on the gridworld with all values equal to zero. Now let us compute v1(A), the value of A after the first iteration. Recall that A has only one action to choose from: moving to A′. Denote this action by a. Therefore,

v1(A) = p(A′ | A, a) · r(a | A, A′) + γv0(A′)

= 1.0 · (10 + 0.9 · 0) = 10

# Problem 14.3.4 (Example 14.3.1 revisited)

Start value iteration with all values equal to zero. What is v2((1, 3)), the value of (1, 3) after second iteration?

# 14.3.3 Why Does Value Iteration Find an Optimum Policy?

Assume γ &#x3C; 1. We prove that the values vk(s) maintained by the value iteration method converge to the optimal values vπ(s). We break this proof down into two parts. We first prove that the invariant vk(s) ≤ v∗(s) holds throughout the algorithm. Then we prove that in general, vk+1(s) is a tighter lower bound for v∗(s) than vk(s).

# Proposition 14.3.5

For each time step k = 1, 2, ..., and for each state s, the invariant vk(s) ≤ v∗(s) holds.

Proof. Proof by mathematical induction. As discussed earlier, our choice of initial values v0(s) = − R satisfies the invariant. Now assume that the invariant holds for some k. Now consider the update rule of the value iteration algorithm:

vk+1(s) = max ∑ p(s′ | s, π(s)) · r(π(s) | s, s′) + γvk(s′)

Notice that for any specific policy π and for any next state s′, we have

p(s′ | s, π(s)) · r(π(s) | s, s′) + γvk(s′) ≤ p(s′ | s, π(s)) · r(π(s) | s, s′) + γv∗(s′)




186    introduction to machine learning lecture notes for cos 324 at princeton university

because of the inductive hypothesis that vk (s′) ≤ v∗(s′). Therefore, if we sum over all state s′, we have

∑ p(s′ | s, π(s)) · r(π(s) | s, s′) + γvk (s′)



# Markov Decision Process

Subtracting (14.10) from (14.9), we get

v∗(s) − vk+1(s) ≤ γ  ∑ p(s′ | s, a∗)(v∗(s′) − vₖ (s′))



# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

# Example 14.3.9 (Example 14.3.1 revisited)

Say we ran the value iteration algorithm on the Gridworld. The output of the algorithm (the optimal values of each state) is given in Table 14.1.

| 22.0 | 24.4 | 22.0 | 19.4 | 17.5 |
| ---- | ---- | ---- | ---- | ---- |
| 19.8 | 22.0 | 19.8 | 17.8 | 16.0 |
| 17.8 | 19.8 | 17.8 | 16.0 | 14.4 |
| 16.0 | 17.8 | 16.0 | 14.4 | 13.0 |
| 14.4 | 16.0 | 14.4 | 13.0 | 11.7 |

Table 14.1: Optimal values v∗(s) of the Gridworld.

Consider the state A′ = (5, 2). There are four actions to take: left/right/up/down. Each action would yield the following values when evaluating the Bellman equation:

vle f t(A′) = 0 + 0.9 × 14.4 = 13.0

vright(A′) = 0 + 0.9 × 14.4 = 13.0

vup(A′) = 0 + 0.9 × 17.8 = 16.0

vdown (A′) = −1 + 0.9 × 16.0 = 13.4

The only action that maximizes the value is the action “go up.” Therefore, we can conclude that the optimal policy π∗ will adopt the action “go up” for the state A′.

# Problem 14.3.10 (Example 14.3.1 revisited)

Verify that an optimal policy can assign either the action “go up” or the action “go left” for the state (5, 3).





# 15 Reinforcement Learning in Unknown Environments

In the previous Chapter 14, we established the principles of reinforcement learning using a Markov Decision Process (MDP) with set of states S, set of actions A, transition probabilities p(s′ | a, s), and the rewards r(a | s, s′). We saw a method (value iteration) to find the optimal policy that will maximize the expected reward for every state. The main assumption of the chapter was that the agent has access to the full description of the MDP — the set of states, the set of actions, the transition probabilities, rewards, etc.

But what can we do when some of the parameters of the MDP are not available to the agent in advance — specifically, the transition probabilities and the rewards? Instead, the agent makes actions and observes the new state and the reward it just received. Using such experiences it begins to learn the reward and transition structure, and then to translate this incremental knowledge into improved actions.

The above scenario describes most real-life agents: the system designer does not know a full description of the probabilities and transitions. For instance, think of the sets of possible states and transitions in the MuJoCo animals and walkers that we saw. Even with a small number of joints, the total set of scenarios is too vast. Thus the designer can set up an intuitive reward structure and let the learner figure out from experience (which is tractable since it involves a simulation).

Settings where agent must determine (or “figure out”) the MDP through experience, specifically by taking actions and observing the effects, is called the “model-free” setting of RL. This chapter will introduce basic concepts, including the famous Q-learning algorithm. In many settings today, the underlying MDP is too large for the agent to reconstruct completely, and the agent uses deep neural networks to represent its knowledge of the environment and its own policy.





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

# 15.1 Model-Free Reinforcement Learning

In model-free RL, we know the set of states S and the set of actions A, but the transition probabilities and rewards are unknown. The agent now needs to explore the environment to estimate the transition probabilities and rewards. Suppose the agent is originally in state s1, chooses to take an action a, and ends up in state s2. The agent immediately observes some reward r(a | s1, s2), but we need more information to figure out p(s2, |s1, a).

One way we can estimate the transition probabilities is through the Maximum Likelihood Principle. This concept has been used before when considering estimating unigram probabilities in Chapter 8. In model-free RL, an agent can keep track of the number of times they took action a at state s1 and ended up in state s2 — denote this as #(s1, a, s2). Then the estimate of the transition probability p(s′|s, a) is:

p(s2|s1, a) =  #(s1, a, s2)

∑ #(s1, a, s′)

s′

The Central Limit Theorem (see Chapter 18) guarantees that estimates will improve with more observations and quickly converge to underlying state-action transition probabilities and rewards.

# 15.1.1 Groundhog Day

Groundhog Day is an early movie about a “time loop” and the title has even become an everyday term. The film tracks cynical TV weatherman Phil Connors (Bill Murray) who is tasked with going to the small town of Punxsutawney and filming its annual Groundhog Day celebration. He ends up reliving the same day over and over again, and becomes temporarily trapped. Along the way, he attempts to court his producer Rita Hanson (Andie MacDowell), and is only released from the time loop after a concerted effort to improve his character.

Sounds philosophically deep! On the internet you can find various interpretations of the movie: Buddhist interpretation (“many reincarnations ending in Nirvana”) and psychoanalysis (“revisiting of the same events over and over again to reach closure”). The RL interpretation is that Phil is in a model-free RL environment,1 revisiting the same events of the day over and over again and figuring out his optimal actions.

1 Specifically a model-free RL environment with an ability to reset to an initial state. This happens for example with a robot vacuum that periodically returns to its charging station. After charging, it starts exploring the MDP from the initial state again.





# Reinforcement Learning in Unknown Environments

# 15.2 Atari Pong (1972): A Case Study

In 1972, the classic game of Pong was released by Atari. This was the first commercially successful video game, and had a major cultural impact on the perception of video games by the general public. The rules of the game are simple: each player controls a virtual paddle which can move vertically in order to rally a ball back and forth (one participant may be an AI agent). If a player misses the ball, the other player wins a point. We can consider the total number of points accumulated by a player to be their reward so far. While technology and video games have become far more advanced in the present, it is still useful to analyze Pong today. This is because it is a simple example of a physics-based system, similar to (but far less advanced than) the MuJoCo stick figure simulations discussed in Chapter 13. It thus provides a useful case study to demonstrate how an agent can learn basic principles of physics through random exploration and estimation of transition probabilities.

Let’s apply some simplifications in the interest of brevity. We define the pong table to be 5 × 5 pixels in size, the ball to have a size of 1 pixel, and the paddles to be 2 pixels in height. We define the state at a time t as the locations of the two paddles at time t, and the locations of the ball at time t and time t − 1. Storing the location of the ball at time t − 1 and time t allows us to calculate the difference between the two locations and thus gives an estimate for the velocity.

We additionally restrict attention to the problem of tracking and returning the ball, also known as “Pico-Pong.” Thus, we define the game to begin when the opponent hits the ball. The agent gets a reward of +1 if they manage to hit the ball, −1 if they miss, and 0 if the ball is still in play. As soon as the agent either hits the ball or misses, we define that the game ends. Of course, these additional rules of the game are not available to the agent playing the game. The agent needs to “learn” these rules by observing the possible states, transitions, and corresponding rewards.

In general, these simplifications remove complications of modeling the opponent and makes the MDP acyclic; an explanatory diagram is shown in Figure 15.1. Throughout this section, we will build intuition about different aspects of our Pico-Pong model through some examples.

# 15.2.1 Pico-Pong Modeling: States

Suppose the agent is playing random paddle movements. Consider the possible states of the game shown in Figure 15.2. We note that out of the three, the third option is never seen. By the definition of the game, the ball can never move away from the agent. Of course, the agent is oblivious to this fact at first, but once the game proceeds,




192  introduction to machine learning lecture notes for cos 324 at princeton university

# Figure 15.1:

The simplified Pico-Pong setup which will be considered in this case study.

Our / paddle Location of ball at time t-1

Location of ball at time t

The agent will be able to implicitly “learn” that the ball can never move away from them.

# Figure 15.2:

Out of these possible states, the third option is never seen.

# 15.2.2  Pico-Pong Modeling: Transitions

Let us now add another restriction to the game that the ball always moves at a speed of 1 pixel every time step (i.e., moves to one of the 8 adjacent pixels) and in a straight linear path unless being bounced against the top/bottom wall. Consider the possible transitions shown in Figure 15.3. We note that out of the three, the third option is never seen. By the restriction of the game, the ball cannot move 2 pixels in one time step. The agent thus implicitly “learns” that the ball moves at a constant speed of 1 pixel per time step.

# Figure 15.3:

Out of these possible transitions, the third option is never seen.


reinforcement learning in unknown environments

# 15.2.1

Suppose the agent is playing randomly and the ball is traveling at a speed of 1 pixel per step. Which of the transitions in Figure 15.4 is never seen, and why?

Figure 15.4: Out of these possible transitions, one option is never seen.

# 15.2.3  Pico-Pong Modeling: Rewards

Suppose the agent is playing randomly and the ball is traveling at a speed of 1 pixel per step. Consider the action in Figure 15.5. We note that the associated reward will be +1 because in the resulting state the agent has “hit” the ball. The agent thus implicitly learns that if the ball is 1 pixel away horizontally, it should move to intercept it to obtain a positive reward.

Figure 15.5: Taking action ↓ results in a reward of +1.

# 15.2.2

Suppose the agent is playing randomly and the ball is traveling at a speed of 1 pixel per step. What reward is achieved given the current state and chosen action in Figure 15.6, and why?

Figure 15.6: What reward will result when taking action ↓?

# 15.2.4  Playing Optimally in the Learned MDP

After allowing the agent to explore enough, the agent has “learned” some information about the underlying MDP of the Pico-Pong model. The first thing the agent can learn is that, out of all possible states, there is a subset of states that never appear in the game (e.g., ball





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

Moving away from the agent or ball moving too fast). The agent will be able to ignore these states, while learning how to play optimally in states that did occur while exploring.

|   | -1 | Figure 15.7: An example look-ahead tree for the Pico-Pong model. |                    |                            |
| - | -- | ---------------------------------------------------------------- | ------------------ | -------------------------- |
|   | +1 |                                                                  |                    |                            |
|   |    | (Agent stays in place)                                           | (Agent moves away) | (Agent moves towards ball) |

Also, the agent has now “learnt” the transition probabilities and rewards of the MDP. Using these estimates, the agent is able to build up a representation of the MDP. Since the underlying MDP for the simplified Pico-Pong model is acyclic, the optimal policy can be determined using a simple look-ahead tree. An example diagram is shown in Figure 15.7.

We provide a specific example to aid the exposition. Suppose an agent finds themselves in the state shown in Figure 15.8. Since the path of the ball is already determined, the next possible state is uniquely determined by the choice of the action — “go down” or “stay in place” or “go up.” If the agent chooses to “go down,” the game will end with a reward of +1. If the agent chooses to “stay in place” or “go up,” the game continues for another time step, but no matter the choice of action on that step, the game will end with a reward of −1. Therefore, the agent will learn that the optimal policy will assign the action of “go down” in the state shown in Figure 15.8.

Problem 15.2.3. Draw out the look-ahead tree from the state shown in Figure 15.8.

Figure 15.8: A sample state in the game play of Pico-Pong.

Problem 15.2.4. Suppose we start from the state shown in Figure 15.9. Assuming optimal play, what is the expected reward for the agent? (Hint: consider if the agent will be able to reach the ball in time.)





# Reinforcement Learning in Unknown Environments

Figure 15.9: A sample state in the game play of Pico-Pong.

Impressive! The agent has learnt how to return the ball in Pico-Pong by first building up the MDP and its transitions/rewards through repeated observations, and then computing the optimum policy for the constructed MDP through a look-ahead tree.3

How would you extend these ideas to design a rudimentary ping pong robot which can track and return the ball?

# 15.3 Q-learning

# 15.3.1 Exploration vs. Exploitation

Let us analyze the case study with Pico-Pong more deeply. We can separate the process of learning into two different stages — exploration and exploitation:

- Exploration: This pertains to what the agent did in the first phase. Random paddle movements were used to help build up previously unknown knowledge of the MDP — transition probabilities and rewards.
- Exploitation: This pertains to what the agent did in the second phase. Specifically, the agent used the learnt MDP to play optimally.

In general, an RL environment is more complicated than Pico-Pong, and there is no clear-cut boundary of when an agent has explored “sufficiently.” It is best to combine the two stages (i.e., exploration and exploitation) into one and “learn as you go.” Also, it is difficult to balance between these two processes, and how to find the correct trade-off between exploration and exploitation is a recurring topic in RL.

# 15.3.2 Q-function

We now introduce the Q-function, an important concept that helps tie together concepts of exploration and exploitation when considering general MDPs with discounted rewards.

Definition 15.3.1 (Q-function): We define the Q-function Q : S × A → R as a table which assigns a real value Q(s, a) to each pair (s, a) where s ∈ S and a ∈ A.




196  introduction to machine learning lecture notes for cos 324 at princeton university

# Introduction

Intuitively, the value Q(s, a) is the current estimate of the expected discounted reward when we take action a from state s. In other words, it is the estimate of the value vπ(s) if π is any policy that will assign the action a to state s. Using the currently stored values of the Q-function, we can define a canonical policy πQ. For each state s, the policy will assign the action a that maximizes the Q(s, a) value; that is,

πQ(s) = arg max Q(s, a)



# Reinforcement Learning in Unknown Environments

# 15.3.3 Q-learning

Now that we have defined the Q-function and the optimal Q-function, it is time for us to study how to learn the optimal Q-function. This process is called Q-learning. The basic idea is to probabilistically choose between exploration or exploitation: we define some probability ϵ ∈ [0, 1] such that we choose a random action a with probability ϵ (exploration) or choose the action a according to the current canonical policy πQ with probability 1 − ϵ (exploitation). If we choose the exploration option, we use its outcome to update the Q(s, a) table. But how should we define the update rule?

Let’s take a step back and consider a (plausibly?) real life scenario. You are a reporter for the Daily Princetonian at Princeton, and want to estimate the average wealth of alumni at a Princeton Reunions event. The alumni, understandably vexed by such a request, strike a compromise that you are only allowed to ask one alum about their net worth. Can you get an estimate of the average? Well, you could pick an alum at random and ask them their net worth!4

With this intuition, we return to the world of Q-learning. Suppose you start at some state st, take an action at, receive a reward of rt, and arrive at state st+1. We call this process an experience. Now, when we update the current estimate of Q(st, at), we ideally want to mimic the behavior of the optimal Q-function in (15.2) and update it to:

Q(st, at) = ∑ p(s′ | st, at) · (r(at | st, s′) + γ max Q(s′, b))  (15.3)

Notice that this is the weighted average of the expected reward r(at | st, s′) + γ maxb Q(s′, b) over all possible next states s′ given the action at. But in practice, the agent only has the ability to take a single experience; they lack the ability to “reset” and retake the step to try all states s′ according to the transition probability p(s′ | st, at). We thus must consider an alternative idea — we define the estimate for Q(st, at) according to the experience at time step t as

Q′ = rt + γ max Q(st+1, b)

This estimate can be calculated using the observed reward rt and looking up the Q values of the state st+1 on the Q-function table. Note that the expectation of Q′ is exactly the right hand side of (15.3). That is,

E[Q′ ] = ∑ p(s′ | st, at) · (r(at | st, s′) + γ max Q(s′, b))

This is because the agent took a transition to state st+1 with probability p(st+1 | st, at) (of course, the agent does not know this value).





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

is thus analogous to the single-sample estimate of average alumni wealth at the Princeton Reunions event. We can now define the following update rule of the Q-learning process:

Q(st, at) ← Q(st, at) + η(Q′ − Q(st, at))

= ( − ) (  ) t ′ (15.4)

1 η Q st, at + ηQt

for some learning rate η > 0. You can understand this update rule in two different ways. First, we are gently nudging the value of Q(st, at) towards the estimate Q′ from the most recent experience. We can alternatively think of the updated value of Q(st, at) as the weighted average of the previous value of Q(st, at) and the estimate Q′. In either approach, the most important thing to note is that we combine both the previous Q value and the new estimate to compute the updated Q value. This is because the new estimate is just a single sample that can be far off from the actual expectation, and also because after enough iterations, we can assume the previous Q value to contain information from past experience.

# Example 15.3.3

Let’s return to our adventures in Pico-Pong and consider the situation in Figure 15.10. Denote the state in the left diagram as st and the state in the right as st+1. Suppose the current value of Q(st, a) = 0.4 with a = ↑. Assuming that Q(st+1, a) = 0 for all a, we can compute the estimate Q′ from this experience as

Q′ = rt + γ max Q(st+1, b) = 1

t b

Then the Q value will be increased to 0.4 + 0.6η.

Figure 15.10: The diagram representing two states in a game of Pico-Pong.

# 15.3.4 Deep Q-learning

Note that the update rule in (15.4) looks similar to the Gradient Descent algorithm. They are both iterative processes which incorporate a learning rate η. In fact, you can consider the Q-learning update rule to be trying to minimize the squared difference between Q(st, at) and Q′. The similarity between the Q-learning update rule and the Gradient Descent algorithm allows us to utilize a deep neural network.


reinforcement learning in unknown environments

to learn the optimal Q-function. Such a network is called the Deep Q Network (DQN).

In a DQN, the Q-function can be represented by the parameters W of the network. We emphasize this by denoting the Q-function as QW(s, a). Now instead of directly updating the Q-function as in the update rule

Q(st, at) ← Q(st, at) + η(Q′ − Q(st, at))    (15.4 revisited)
t

we instead update the parameters W such that the Q-function is updated accordingly.

First consider the case that Q′ > Q(st, at). That is, the estimated Q-value is larger than the currently stored value. Then the update rule (15.4) will increase the value of Q(st, at). To mimic this behavior, we want to find an update rule for W that will increase the Q-value. This is given as:

W ← W + β · ∇WQW(st, sa )

for some learning rate β > 0.

Problem 15.3.4. Suppose Q′ &#x3C; Q(st, at). How should we design the weight updates?

One final thing to note is a technique called experience replay. Experiencing the environment can be expensive (i.e., computation time, machine wear, etc.). Therefore, it is customary to keep a history of old experiences and their rewards, and periodically take a random sample out of the old experiences to update the Q values. In particular, experience replay ensures that DQNs are efficient and avoid “catastrophic forgetting.” 5

5 Catastrophic forgetting is a phenomenon where a neural network, after being exposed to new information, “forgets” information it had learned earlier.

# 15.4 Applications of Reinforcement Learning

# 15.4.1 Q-learning for Breakout (1978)

We previously considered using reinforcement learning for Pong. We can also use it for another famous Atari game called Breakout. One particular design uses a CNN to process the screen and uses the "score" as a reward. As shown in Figure 15.11, the model becomes quite successful after several epochs.

# 15.4.2 Self-help Apps

Self-help apps are designed to aid in recovery of the user from addiction, trauma, heart disease, etc. A typical design involves an RL algorithm which determines the next advice/suggestion based upon reversals, achieved milestones, etc. so far. These can be a helpful supplement to expensive therapy/consultation.





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

# 15.4.3 Content Recommendation

At reputable websites, we might imagine that there exists a page creation system designed to capture the “reward” of user engagement. We can use MDP techniques to model this situation. Specifically, we can define s0 as the outside link which brought the user to the landing page and/or the past history of the user on the site. If the user clicks on a link, a new page is created and we can define s1 as a concatenation of s0 and the new link. If the user again clicks on a link, another new page is created and we can define s2 as the concatenation of s1 and the new link.

# 15.5 Deep Reinforcement Learning

Deep Reinforcement Learning is a subfield of machine learning that combines the methods of Deep Learning and Reinforcement Learning that we have discussed earlier. 6 The goal of it is to create an artificial agent with human-level intelligence (Artificial General Intelligence, AGI). In general, Reinforcement Learning defines the objective and Deep Learning gives the mechanism for optimizing that objective. Deep RL combines the problem given by the RL specification with the solution given by the DL technique. In the cited source video, RL expert David Silver made three broad conjectures related to this topic.

1. RL is enough to formalize the problem of intelligence
2. Deep neural networks can represent and learn any computable function
3. Deep RL can solve the problem of intelligence

Many Deep RL models are trained to play games (e.g., chess, Go) because it is easy to evaluate progress. By letting them compete against

Figure 15.11: An application of Q-learning to the famous Atari game Breakout.





# Reinforcement Learning in Unknown Environments

Humans, we can easily compare them to human-level intelligence. As an example, Google Deepmind trained a Deep RL model called DQN to play 49 arcade games.7 The computer is not given the explicit set of rules; instead, given only the pixels and game score as input, it learns by using deep reinforcement learning to maximize its score. Amazingly, on about half of the games, the model played at least at a human level of intelligence!

# 15.5.1 Chess: A Case Study

Founders of AI considered chess to be the epitome of human intelligence. In principle, the best next move can be calculated via a look-ahead tree (similar to Figure 13.5 from the cake-eating example). Since chess is a two-player game, we can use an algorithm called the min-max search on the look-ahead game tree.8 Usually, RL agents are playing against the nature that causes them to take random transitions according to the MDP’s transition probabilities. But in chess, the agent plays against an opponent that is trying to make the agent take the largest possible loss (the largest possible gain for the opponent). That is why we need a min-max evaluation of the look-ahead tree.

Figure 15.12: An example look-ahead game tree for chess with depth 3. White will choose the right option.

In Figure 15.12, the numbers at the leaf nodes represent a static evaluation of how good the game configuration is for white. This is an approximation for the actual value of the node. An example metric in chess would be the difference in the number of pieces (# white − # black). These numbers are evaluated either when the game terminates or when the algorithm has reached the specified number of steps to look ahead. If the game ever reaches the specified node, the white has two options to choose from: if white chooses the left child node, it will end up with reward of −1; whereas if it chooses the right child node, the reward will be 3. Then to maximize reward,

white to move A

| 1 | 3 |
| - | - |





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

# Best Move of White

The best move of white will be to choose 3. 9 For those who are familiar with chess or game theory in general, this is known as the best response.

Figure 15.13: Black will choose the left option.

In Figure 15.13, it is now black’s turn to choose. Note that the reward for black is the opposite of the reward for white, so black wants to minimize the value on the tree. Therefore, black will want to choose the left child node.

So whenever we are at a configuration, we can create a look-ahead tree for a reasonable number of steps and try to calculate the best move. But the size of a game tree is astronomical, so it is computationally infeasible to search all levels of the tree. There is an optimization method called alpha-beta pruning. Consult the video referenced above for an implementation on the game of chess.

# 15.5.2 AlphaGo: A Case Study

Go is a game invented in China around 500 BC. It is played by 2 players on a 19 × 19 grid. Players take turns placing stones on the grid, and if any set of stones is entirely surrounded by opponent stones, the enclosed stones are taken away from the board and awarded to the opponent as points. Even though the rules are very simple, no computer could beat a good human amateur at Go until 2015. In comparison, IBM’s Deep Blue model beat the world chess champion Kasparov in 1997.

How can we utilize RL concepts to play this game? In general, we can create a Deep Policy Net (DPN) to learn W, which is a function that takes state s as an input and outputs a probability distribution pW (a | s) over the next possible actions from s. AlphaGo is an example of a DPN engineered by the Google Deepmind lab. It takes the current board position as the input and uses ConvNet to learn the internal weights, and outputs the value given by a softmax function.

In its initial setup, the DPN was trained using a big dataset of past games. To be more specific, AlphaGo used supervised learning from human data to learn the optimal policy (action to take at each game setting). In other words, it used convolutional layers to replicate the moves of professional players as closely as possible.

Source: https://www.youtube.com/watch?v=Wujy7OzvdJk

black to move


reinforcement learning in unknown environments
# 203

Figure 15.14: The diagram representing the process of training AlphaGo.

is just mimicking human players, it cannot beat human champions. Training AlphaGo However, it can be used to search the full game tree more efficiently than the alpha-beta search. Formally, this method is called the Monte Carlo Tree Search, where the CNN is used to decide the order in which to explore the tree. After the policy network was sufficiently trained, reinforcement learning was used to train the value network for position evaluation. Given a board setting, the network was trained to estimate the value (i.e., likelihood of winning) of that setting.

AlphaGo Zero is a newer version of the model that does not depend on any human data or features. In this model, policy and value networks are combined into one neural network, and the model does not use any randomized Monte-Carlo simulations. It learns solely by self-play reinforcement learning and uses neural network (ResNet) to evaluate its performance. Within 3 days of training, AlphaGo Zero surpassed an earlier version of AlphaGo that beat Lee Se Dol, the holder of 8 world titles; within 21 days, it surpassed the version that beat Ke Jie, the world champion. Interestingly enough, AlphaGo Zero adopted some opening patterns commonly played by human players, but it also discarded some common human patterns and it also discovered patterns unknown to humans.

The newest version of AlphaGo is called AlphaZero. It is a model that can be trained to play not just Go but simultaneously Chess and Shogi (Japanese chess). After just a few hours of training, AlphaZero surpassed the previous computer world champions (Stockfish in Chess, Elmo in Shogi, and AlphaGo Zero in Go). Just as AlphaGo Zero did, AlphaZero was able to dynamically adopt or discard known openings in chess.

| Supervised Learning, | Reinforcement Learning |
| -------------------- | ---------------------- |




NO_CONTENT_HERE


# Part V

# Advanced Topics


NO_CONTENT_HERE



# Machine Learning and Ethics

Throughout this course, we have discussed the technical aspects of model design, training, and testing in depth. However, we have not yet discussed some of the social implications of this technology. What are some ethical and legal issues in deployment of ML techniques in society? What are the caveats and limitations to temper our exuberance about the possibilities of ML? This brief chapter addresses these issues, and we hope as technologists you will continue to investigate and consider such issues throughout your career.

# 16.1 Facebook’s Suicide Prevention

| Figure 16.1: A visualization of the Facebook model to predict suicides. |             |              |
| ----------------------------------------------------------------------- | ----------- | ------------ |
| Time of day                                                             | Day of week | Post type... |
| "Goodbye...                                                             | Main text   | 0.89         |
| "Where are you??                                                        | Comment     | 0.92         |

In 2017, Facebook launched a program to use a machine learning algorithm to predict suicide risk amongst its user population. It has continued with various iterations over the years. Figure 16.1 gives a visualization of the four-step process:

1. ML algorithm automatically analyzes a post by processing its text content and comments


208   introduction to machine learning lecture notes for cos 324 at princeton university

2. Algorithm additionally uses spatial-temporal context of the post to perform a risk prediction.

3. A human behind the algorithm performs a personal review to finally verify if a threshold is reached.

4. If the post poses a serious risk, Facebook performs a wellness check through the person’s contacts, community organizations, etc.

At first sight, this may appear to be very good idea: even if it saves just one life, surely the project is worth it? But the announcement of the project cause a lot of controversy among people. The following are some of the potential problems that people identified:

1. False positives may result in stigmatization.
2. Many people who contemplate suicide do not end up going through with it. Facebook’s reporting could lead to criminal penalties (in regions where suicide is a crime), involuntary hospitalization, stigmatization, etc.
3. Involvement of authorities (e.g., law enforcement) raises risk of illegal seizures.
4. Should Facebook be liable for any problem caused by mis-detection?

Beyond these points, there are deep philosophical questions associated with the concept of suicide as well. For instance, is suicide actually immoral? Even if it is immoral, is it the responsibility of Facebook to get involved? Is it moral for Facebook to use personal information to assess suicide risk? Opinions differ.

# 16.2  Racial Bias in Machine Learning

Suppose we are designing a machine learning approach for loan approval. The general approach will be to take a dataset of (x, y), where x is a vector of the individual’s attributes (e.g., age, education, alma mater, address, etc.) who got a loan and y ∈ {−1, 1} indicates whether they actually paid off the loan or not. Using the approaches we learned in Section 4.2, we could train a binary classifier through logistic regression. Civil rights legislation forbids using the individual’s race in many of these decisions, so while training we could simply mask out any coordinates which identify race. However, this does not guarantee that the classifier will be entirely “race-neutral.”

1 The reason is that race happens to be correlated with many other attributes. Thus if a classifier uses any of the correlated attributes, it may be implicitly using racial information in the decision making process.

2 Machine Bias, by Angwin et al., in Pro Publica 2016.





# Machine Learning and Ethics

Serious crime, disproportionately tags African-American as being likely to commit crimes — in the sense that African-Americans who were tagged as likely to commit another crime were only half as likely to actually commit a crime than a similarly-tagged person of another race.

| White                                   | African-American                   | Table 16.1: COMPAS correctly predicts recidivism 61 percent on average. But African-Americans are almost twice as likely as whites to be labeled a higher risk but not actually re-offend. Conversely, whites are twice as likely as African-Americans to be labeled lower risk but go on to commit other crimes. |   |   |
| --------------------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | - | - |
| Labeled Higher Risk & Did Not Re-offend | Labeled Lower Risk & Did Re-offend |                                                                                                                                                                                                                                                                                                                   |   |   |
|                                         | 23.5%                              | 44.9%                                                                                                                                                                                                                                                                                                             |   |   |
|                                         | 47.7%                              | 28.0%                                                                                                                                                                                                                                                                                                             |   |   |

# 16.3 Conceptions of Fairness in Machine Learning

We will briefly consider possible ways to formulate fairness in machine learning. Keep in mind that this task is intrinsically difficult, as we are attempting to assign a quantifiable objective to a fundamentally normative problem. The first property we might want an ML classifier to have is called demographic parity, which effectively enforces that the output of classifier does not depend on a protected attribute (e.g., race, ethnicity, gender).

# Definition 16.3.1 (Demographic Parity)

We say that a binary classifier that outputs y ∈ {−1, 1} satisfies demographic parity if Pr[y | xi = a] = Pr[y | xi = b] where a, b are any two values that a protected attribute xi can take.

Figure 16.2: A hypothetical application of ML to a loan approval application. Race has been made a protected attribute in an attempt to prevent bias during training.

| X (features) | A (protected attribute) | LOAN |
| ------------ | ----------------------- | ---- |
| 0 0 1        | 1                       | Y    |
| 1 1 0        | 1                       | N    |
| 1 1 0        | 0                       | N    |

A visualization of how a protected attribute could be specified in a dataset is shown in Figure 16.2. Consider the loan approval example.




210   introduction to machine learning lecture notes for cos 324 at princeton university

from the previous section. If the binary classification model for the loan approval satisfies the demographic parity property, then the model approves loans for different races at similar rates. One way to achieve this condition is to use a regularizer term λ(Pr[y | xi = a] − Pr[y | xi = b])3 during training. 3 Does this seem like a good formulation of fairness?

Another property we might want a “fair” model to satisfy is called predictive parity. This is the property that the model in Table 16.1 failed to satisfy.

# Definition 16.3.2 (Predictive Parity).

We say that a binary classifier that outputs y ∈ {−1, 1} satisfies predictive parity if the true negative, false negative, false positive, true positive rates are the same for any values of a protected attribute.

Figure 16.3: A table of all possible outcomes based on the model output and the ground truth outcome. This is also known as a confusion matrix.

| Classifier's output | True     | False    |
| ------------------- | -------- | -------- |
| -1                  | Negative | Negative |
| 1                   | False    | True     |
|                     | Positive | Positive |
| -1                  |          | 1        |
| Outcome             |          |          |

Ideally, we want an ML model to satisfy both the demographic parity and predictive parity. However, it turns out that these two notions are incompatible!

# Theorem 16.3.3 (Fairness Impossibility Theorem).

Under fairly general conditions, demographic parity and predictive parity are incompatible.4 There are other formulations of fairness, but it is difficult to find a combination of these notions that are compatible with each other. So one way or another, we need to sacrifice some notions of “fairness.”

# 16.4  Limitations of the ML Paradigm

The predictive power of ML seems immense, but is it true that if we have enough data and the right algorithm, then everything becomes predictable? If yes, then one could imagine societal programs leveraging this to precisely target help to where it would be more effective. We first consider a famous — and somewhat amusing — example of a study5 that turned out to be false.

5 Extraneous factors in judicial decisions, Danziger et al., PNAS 2011.




# machine learning and ethics

# 16.4.1 Hungry Judge Effect

The study analyzed the parole decisions made by 8 Israeli judges in over 1,100 cases. The data in Figure 16.4 shows that prisoners were much more likely to be granted parole after the judge took a lunch break or a coffee break. The study therefore suggested that judges tend to be stricter before a break (maybe because they are “hangry”) but more lenient when they return from the break.

# Figure 16.4: Data from the study

| 0.8              |              |
| ---------------- | ------------ |
| 0.7              |              |
| 0.6              |              |
| 0.5              |              |
| 0.4              |              |
| 0.3              |              |
| 0.2              |              |
| 0.1              |              |
| 0                |              |
| Ordinal position |              |
| Lunch Break      | Coffee Break |

Nevertheless, it turns out that this “hungry judge effect” can be explained by a completely different reason. A followup study6 found that the ordering of cases presented to the judge was not random: prisoners with attorneys were scheduled at the beginning of each session, while prisoners without an attorney were scheduled at the end of a session. The former group were let on parole with a rate of 67%, while the rate was just 39% for those without attorneys. Another important observation was that attorneys tended to present their cases in decreasing order of strength of case, with the average attorney having 4.1 clients. Computer simulations of hunger-immune judges faced with cases presented according to these percentages showed the same see-saw effect of Figure 16.4.

# 16.4.2 Fragile Families Challenge

The Fragile Families Challenge is a collaborative project initiated by the Center for Research on Child Wellbeing at Princeton University. A brief description of the initiative’s motivation is provided on the website:7

The Fragile Families Challenge is a mass collaboration that combines predictive modeling, causal inference, and in-depth interviews to yield insights that can improve the lives of disadvantaged children in the United States.


212   introduction to machine learning lecture notes for cos 324 at princeton university

By working together, we can discover things that none of us can discover individually.

The Fragile Families Challenge is based on the Fragile Families and Child Wellbeing Study, which has followed thousands of American families for more than 15 years. During this time, the Fragile Families study collected information about the children, their parents, their schools, and their larger environments.

| Birth to age 9   | Age 15     |
| ---------------- | ---------- |
| 12,942 variables | 6 outcomes |

Figure 16.5: Diagram illustrating the dataset of the Fragile Families Challenge. After training a model on the training data, participants made predictions on held-out data and submitted the results to a leaderboard.

# 20 Background

The initiative has collected immense data on multiple families, including interviews with mothers, fathers, and/or primary caregivers at several ages. Interviewees were inquired as to attitudes, relationships, parenting behavior, economic and employment status, etc. Additionally, in-home assessments of children and their home environments were performed to assess cognitive and emotional development, health, and home environment. The goal was to predict six key outcomes at age 15 (e.g., whether or not the child is attending school) given background data from birth to age 9 as shown in 16.5. However, up to this point no method has done better than random guessing.

This is food for thought: what is going on?

# 16.4.3 General Limits to Prediction

Matt Salganik and Arvind Narayanan, professors at Princeton University, recently started a course 8 which aims to explore the extent to which interdisciplinary problems in social science and computer science can be predictable. In general, the following are some major themes that can make prediction difficult:

1. The distribution associated with data can shift over time
2. The relationship between input data and desired outputs can change over time

8 The course, COS 597E/SOC 555 is a seminar first offered in Fall 2020.




# machine learning and ethics

3. There is a possibility for undiscovered coordinates to be unintentionally ignored (i.e., as in the hungry judge effect)

4. The “8 billion problem,” which describes how data available in the real world is fundamentally finite and limited

# 16.5 Final Thoughts

As described in the preceding sections, users and designers of machine learning will often face ethical dilemmas. Designers may have to operate without moral clarity or easy technical fixes. In fact, technical solutions may even be impossible. To appropriately acknowledge these limitations, it is important to embrace a culture of measuring and openly discussing the impact of the system being built. Indeed, a general principle to follow is to avoid harm when trying to do good.


NO_CONTENT_HERE



# Deep Learning for Natural Language Processing

# 17.1 Word Embeddings

In traditional NLP, each word is regarded as a distinct symbol each with a single value of weight. For example, in Chapter 1, we learned how to use linear regression on sentiment prediction. But with this approach, it is hard for the computer to learn the meaning of the word; instead, each of the words remains as some abstract symbol with numeric weights.

But how do computers know the meaning of words? We can easily think of one solution: we can look up words in a dictionary. For example, WordNet is a project that codes the meaning of the words and the relationship between the words, so that the data can be used for computers to parse.1 But resources like WordNet require human labor to create and adapt, and it is impractical to keep up-to-date (because new words are coined and new meanings appear out of existing words).

An alternative approach is to represent words as short (50 - 300 dimensions2), real-valued vectors. These vectors encode the meaning and other properties of words. In this representation, the distance between vectors represents the similarity between words. This vectorized form of the words is much easier to be used as input in modern ML systems (especially neural networks). These vector forms of words are known as word embedding. In this section, we explore the process of how to learn a good word embedding.

# 17.1.1 Distributional Hypothesis

Word embedding is based on a concept called the distributional hypothesis, a theory developed by John Rupert Firth. The hypothesis, one of the most successful ideas of modern statistical NLP, says that words that occur in similar contexts tend to have similar meaning.

1 For more information, check http://wordnetweb.princeton.edu.

2 The dimension of word vectors is a hyperparameter that needs to be decided first.




216  introduction to machine learning lecture notes for cos 324 at princeton university

# 17.1 Context and Co-occurrence

Definition 17.1.1 (Context). When a word w appears in a text, its context is the set of words that appear nearby (within a fixed-size window).

Example 17.1.2. Assume that you first heard the word tejuino and have no idea what the word means. But you learn that the word may appear in the following four contexts.

- C1: A bottle of      is on the table.
- C2: Everybody likes  .
- C3: Don’t have       before you drive.
- C4: We make      out of corn.

Based on these contexts, it is reasonable to conclude that the word “tejuino” refers to some form of alcoholic drink made from corn.

Problem 17.1.3. To find words with similar meanings as “tejuino,” we tried filling out the contexts from Example 17.1.2 with 5 other words. The results are given in Table 17.1, where 1 means that a native speaker deemed the word was appropriate to be used in that context, and 0 means that it was inappropriate.

|           | C1 | C2 | C3 | C4 |
| --------- | -- | -- | -- | -- |
| tejuino   | 1  | 1  | 1  | 1  |
| loud      | 0  | 0  | 0  | 0  |
| motor-oil | 1  | 0  | 0  | 0  |
| tortillas | 0  | 1  | 0  | 1  |
| choices   | 0  | 1  | 0  | 0  |
| wine      | 1  | 1  | 1  | 0  |

Which word is closest to “tejuino”?

# 17.1.2 Word-word Co-occurrence Matrix

Given a very large collection of documents with words from a dictionary V, we construct a |V| × |V| matrix X, where the entry at the i-th row, j-th column denotes the number of times (i.e., frequency) that wj appears in the context window of wi. This matrix is called the word-word co-occurrence matrix.

Example 17.1.4. Table 17.2 shows a portion of a word-word co-occurrence matrix. Each row corresponds to the center word wi, and each column corresponds to the context word wj. The value Xij at the (i, j) entry means that the context word wj appeared Xij times in the context (of length 4) of wi in total.

Although the portion shown in Table 17.2 mostly has non-zero entries, in general, the entries of the matrix are mostly zero.




deep learning for natural language processing
217

|             | · · · | computer | data | result | pie | sugar | · · · |   |
| ----------- | ----- | -------- | ---- | ------ | --- | ----- | ----- | - |
| cherry      | · · · | 2        | 8    | 9      | 442 | 25    | · · · |   |
| strawberry  | · · · | 0        | 0    | 1      | 60  | 19    | · · · |   |
| digital     | · · · | 1670     | 1683 | 85     | 5   | 4     | · · · |   |
| information | · · · | 3325     | 3982 | 378    | 5   | 13    | · · · |   |

# 17.1.3 Factorization of Word-word Co-occurrence Matrix

Recall the example of movie recommendation through matrix factorization in Chapter 9. In that example m × n matrix M was factored into M ≈ AB where the i-th row of A was a d-dimensional vector that represented user i and the j-th column of B was a d-dimensional vector that represented movie j.

We can imagine a similar factorization on the word-word co-occurrence matrix. That is, we can represent each center word and each context word as a d-dimensional vector such that Xij ≈ Ai∗ · B∗j. But this particular idea does not work on the word-word co-occurrence matrix. The key difference is that X is a complete matrix with no missing entries (although most entries are zero). Therefore we instead use other standard matrix factorization techniques.

One popular choice of factorization is running the Singular Value Decomposition (SVD) on a weighted co-occurrence matrix. 3 This idea originates from a concept called Latent Semantic Analysis. 4 If the SVD returns the following decomposition,

σ1    0        · · ·    0
X    =        W  0     σ2       · · ·    0    V⊺
.     .        .        .
.     .         . .     .
.     .                 .
0     0        · · ·    σd

where X is a V × V matrix, W is a V × d matrix, and V is a d × V matrix, then the i-th row of matrix W can be regarded as the embedding for word wi.

Other modern approaches tend to treat word vectors as parameters to be optimized for some objective function and apply the gradient descent algorithm. But the principle is the same: “words that occur in similar contexts tend to have similar meanings.” Some of the popular algorithms with this approach include: word2vec (Mikolov et al., 2013), GloVe (Pennington et al., 2014), and fastText (Bojanowski et al., 2017).

Here we briefly explain the GloVe algorithm. Given the co-occurrence table X, we will construct a center word vector ui ∈ Rd and a context word vector vj ∈ Rd such that they optimize the follow-





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

# Objective:

J(θ) = ∑ f (Xij) ui · vj + bi + bj − log Xij2 (17.1)

where f is some non-linear function and bi, bj are bias terms, and θ is the set of all entries in ui, vj and bi, bj. This is within the same line of logic as optimizing

L(A, B) = | 1 | ∑ (Mij − (AB)ij)² ((9.5) revisited)

# 17.1.4 Properties of Word Embeddings

A good word embedding should represent the meaning of the words and their relationship with other words as accurately as possible. Therefore there are some properties that we would like a word embedding to preserve. We will discuss three such properties and see how the current algorithms for word embedding perform on preserving those properties.

# 1. Similar words should have similar word vectors:

This is the most important property we can think of.

Example 17.1.5. In a certain word embedding, the following is the list of 9 most nearest words to the word “sweden.”

| Word        | Cosine distance |
| ----------- | --------------- |
| norway      | 0.760124        |
| denmark     | 0.715460        |
| finland     | 0.620022        |
| switzerland | 0.588132        |
| belgium     | 0.585835        |
| netherlands | 0.574631        |
| iceland     | 0.562368        |
| estonia     | 0.547621        |
| slovenia    | 0.531408        |

Notice Scandanavian countries are the top 3 entries on the list, and the rest are also European country names.

# 2. Vector difference should encode the relationship between words:

If there are two or more pairs of words where each pair of words are distinguishable by the same attribute, you can imagine that the vector difference within each pair is nearly the same.




deep learning for natural language processing

# Example 17.1.6.

In Figure 17.1, notice that vman − vwoman ≈ vking − vqueen. The vector difference in common can be understood as representing the male-female relationship. Similarly, there seems to be a common vector difference for representing the difference in verb tense.

| man     | walked  |         | woman  |
| ------- | ------- | ------- | ------ |
| king    | walking | swam    | queen  |
| Turkey  | Ankara  | Canada  | Russia |
| Germany | Rome    | Spain   | Italy  |
| Japan   | Tokyo   | Vietnam | Hanoi  |
| China   | Beijing |         |        |

Male-Female

Verb tense

Country-Capital

# 3. The embeddings should be translated between different languages:

When we independently find the word embedding in different languages, we can expect to have a bijective mapping that preserves the structure of the words in each language.5

Example 17.1.7. In Figure 17.2, notice that if we let W to be the mapping from English to Spanish word embeddings, vcuatro ≈ W ◦ vfour

| 0.15p | O four | n3r  | cuatro (four) | Ouno (one)    |
| ----- | ------ | ---- | ------------- | ------------- |
| 0.1   |        | 0.1  |               |               |
| 0.06  | O five | oone | −0.1          | Ocinco (five) |
| 0     |        | -02  | Otres (three) |               |
| -0.05 | Othree |      | -0.3          |               |
| −0.15 |        | -0.4 |               |               |
| -0.07 |        | −0.5 | odos (two)    |               |
| −0.2  | Ofwo   | 0.1  | 0.2           | 0.3           |
| 0.4   | 0.5    | 0.6  | 0.7           | → -0.2$       |
| 0     | 0.2    | 0.4  | 0.6           | 0.8           |
| 1.2   |        |      |               |               |

| 0.2r  | horse |       |      | 0.5r        | caballo (horse) |             |      |      |
| ----- | ----- | ----- | ---- | ----------- | --------------- | ----------- | ---- | ---- |
| 0.15  |       |       |      | 0.4         |                 |             |      |      |
| 0.1   |       | cow   |      | 0.3         | vaca (cow)      | perro (dog) |      |      |
| 0.05  |       | pig   | dog  | 0.2         |                 |             |      |      |
| 0     |       | 0.1   |      | cerdo (pig) |                 |             |      |      |
| -0.05 |       | -0.1  |      |             |                 |             |      |      |
| -0.15 |       |       | -0.2 |             |                 |             |      |      |
| -0.25 | cat   |       | -0.4 | gato (cat)  |                 |             |      |      |
| -0.25 | -0.2  | -0.15 | -0.1 | -0.05       |                 |             |      |      |
|       | 0     | 0.05  | 0.1  | 0.15        |                 |             |      |      |
|       |       |       |      | -0.55       | -0.4            | -0.3        | -0.2 | -0.1 |
| 0     | 0.1   | 0.2   | 0.3  | 0.4         | 0.5             |             |      |      |

# 17.2 N-gram Model Revisited

Recall the n-gram model from Chapter 8. It assigned a probability Pr[w1w2 . . . wn] to every word sequence w1w2 . . . wn. We discussed the concept of perplexity of the model to compare the performance.




220    introduction to machine learning lecture notes for cos 324 at princeton university

of unigram, bigram, and trigram models. While the n-gram model is impressive, it has obvious limitations.

# Problem 17.2.1.

“The students opened their  .” Can you guess the next word?

# Problem 17.2.2.

“As the proctor started the clock, the students opened their .” Can you guess the next word?

In a lot of cases, words in a sentence are closely related to other words and phrases that are far away. But the n-gram model cannot look beyond the specified frame.

# Example 17.2.3.

The following is a text generated by a 4-gram model:

Today the price of gold per tan, while production of shoe
lasts and shoe industry, the bank intervened just after it
considered and rejected an imf demand to rebuild depleted
european stocks, sept 30 and primary 76 cts a share.

The generated text is surprisingly grammatical, but incoherent.

Example 17.2.3 shows that we need to consider more than three words at a time if we want to model language well. But if we use a larger value of n for the n-gram model, the data will become too sparse to estimate the probabilities. But even when we restrict ourselves to words that appear in the dictionary, there are 1021 distinct sequences of 4 words.

# 17.2.1  Feedforward Neural Language Model

The idea of the feedforward neural language model was proposed by Bengio et al. in 2003 in a paper called A Neural Probabilistic Language Model. The intuition is to use a neural network to learn the probabilistic distribution of language, instead of estimating raw probabilities. The key ingredient in this model is the word embeddings we discussed earlier.

# Example 17.2.4.

Assume we are given two contexts “You like green ” and “You like yellow ” to fill the blanks in. A n-gram model will try to calculate the raw probabilities Pr[w | You like green] and Pr[w |  You like yellow]. However, if the word embeddings showed that vgreen ≈ vyellow, then we can imagine that the two contexts are similar enough. Then we may be able to estimate the probabilities better.

Now we show how to use the feedforward neural language model on a n-gram model. Assume we want to estimate the probability Pr[wn+1 | w1 . . . wn]. Then the first step is to find a word embedding

v1, v2, . . . , vn ∈ Rd


deep learning for natural language processing

of each word w1, w2, . . . , wn. Then we concatenate the word embeddings into

x = (v1, . . . , vn) ∈ Rnd the order of the input vectors cannot change

This will be the input layer. Then we define the fully connected hidden layer as

h = tanh(Wx + b) ∈ Rh where W ∈ Rh×nd and b ∈ Rh. Then we define the output layer as

z = Uh ∈ R|V|

where U ∈ R|V|×h. Then finally, the probability will be calculated with the softmax function:

Pr[wn+1 = i | w1 . . . wn] = softmaxi(z) =  ezi / ∑k ezk where k ∈ V

So the total number of parameters to train in this network is

d |V| + ndh + h + h |V| where the terms are respectively for the input embeddings, W, b, U.

When d = h, sometimes we tie the input and output embeddings. That is, we can consider U to be the parameters required for the output embeddings. At this point, the language model reduces to a |V|-way classification, and we can create lots of training examples by sliding the input-output indices. That is, when given a huge text, we can create lots of input-output tuples as follows:

((w1, . . . , wn), wn+1), ((w2, . . . , wn+1), wn+2), . . .

# 17.2.2 Beyond Feedforward Neural Language Model

But the feedforward language model still has its limitations. The main reason is that W ∈ Rh×nd scales linearly with the window size. Of course, this is better than the traditional n-gram model which scales exponentially with n. Another limitation of the neural LM is that the model learns separate patterns for the same item. That is, a substring wk wk+1, for example, will correspond to different parameters in W when trained on (wk wk+1 . . . wk+n−1) or on (wk−1wk . . . wk+n−2).

To mitigate these limitations, we can choose to use similar modeling ideas but use better and bigger neural network architectures like recurrent neural networks (RNN) or transformers.

Here we briefly explain the core ideas of a RNN. RNNs are a family of neural networks that handle variable length inputs. Whereas feedforward NNs map a fixed-length input to a fixed-length output, recurrent NNs map a sequence of inputs to a sequence of outputs. The





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

# Figure 17.3: A visual representation of an RNN architecture.

| Input layer | Hidden layers | Output layer |
| ----------- | ------------- | ------------ |
| Time        |               |              |

Sequence length can vary and the key is to reuse the weight matrices at different time steps. When the inputs are given as x1, x2, . . . xT ∈ ℝd and we want to find outputs h1, h2, . . . hT ∈ ℝh, we train the parameters

W ∈ ℝh×h, U ∈ ℝh×d, b ∈ ℝh such that

ht = g(W ht−1 + U xt + b) ∈ ℝ where g is some non-linear function (e.g., ReLU, tanh, sigmoid). We can also set h0 = 0 for simplicity.




# Part VI

# Mathematics for Machine Learning


NO_CONTENT_HERE



# Probability and Statistics

# 18.1 Probability and Event

# 18.1.1 Sample Space and Event

Probability is related to the uncertainty and randomness of the world. It measures the likelihood of some outcome or event happening. To formalize this concept, we introduce the following definition:

Definition 18.1.1 (Sample Space and Event). A set S of all possible outcomes of a random phenomenon in the world is called a sample space. Each element x ∈ S is called an outcome. A subset A ⊂ S is called an event.

Example 18.1.2. The sample space of “the outcome of tossing two dice” is the set S = {(1, 1), (1, 2), . . . , (6, 6)} of 36 elements. The event “the sum of the numbers on the two dice is 5” is the subset A = {(1, 4), (2, 3), (3, 2), (4, 1)} of 4 elements.

# 18.1.2 Probability

Given a sample space S, we define a probability for each event A of that space. This probability measures the likelihood that the outcome of the random phenomenon belongs to the set A.

Definition 18.1.3 (Probability). A probability Pr : S → R≥0 is a mapping from each event A ⊂ S to a non-negative real number Pr[A] ≥ 0 such that the following properties are satisfied:

1. 0 ≤ Pr[A] ≤ 1 for any A ⊂ S
2. Pr[S] = 1
3. For any countable collection {A1, A2, . . .} of events that are pairwise disjoint (i.e., Ai ∩ Aj = ∅ for any i = j),

∞             ∞

Pr            Ai  = ∑ Pr[Ai]

i=1           i=1




226  introduction to machine learning lecture notes for cos 324 at princeton university

When the sample space is finite or countably infinite, 1 the properties above can be simplified into the following condition:

∑ Pr[{x}] = 1

x∈S

Example 18.1.4. Consider the sample space of “the outcome of tossing two dice” again. Assuming the two dice are fair, the probability of each outcome can be defined as 1/36. Then the probability of the event “the sum of the numbers on the two dice is 5” is 4/36.

Example 18.1.5. We are picking a point uniformly at random from the sample space [0, 2] × [0, 2] in the Cartesian coordinate system. The probability of the event that the point is drawn from the bottom left quarter [0, 1] × [0, 1] is 1/4.

# 18.1.3  Joint and Conditional Probability

In many cases, we are interested in not just one event, but multiple events, possibly happening in a sequence.

Definition 18.1.6 (Joint Probability). For any set of events A = {A1, . . . , An} of a sample space S, the joint probability of A is the probability Pr[A1 ∩ . . . ∩ An] of the intersection of all of the events. The probability Pr[Ai] of each of the events is also known as the marginal probability.

Example 18.1.7. Consider the sample space of “the outcome of tossing two dice” again. Let A1 be the event “the number on the first die is 1” and let A2 be the event “the number on the second die is 4.” The joint probability of A1, A2 is 1/36. The marginal probability of each of the events is 1/6.

It is also useful to define the probability of an event A, based on the knowledge that other events A1, . . . , An have occurred.

Definition 18.1.8 (Conditional Probability). For any event A and any set of events A = {A1, . . . , An} of a sample space S, where Pr[A1 ∩ . . . ∩ An] > 0, the conditional probability of A given A is

Pr[A | A1, . . . , An] = Pr[A ∩ A1 ∩ . . . ∩ An]

Pr[A1 ∩ . . . ∩ An]

Example 18.1.9. Consider the sample space of “the outcome of tossing two dice” again. Let A1 be the event “the number on the first die is 1” and let A2 be the event “the sum of the numbers on the two dice is 5.” The conditional probability of A1 given A2 is 1/4. The conditional probability of A2 given A1 is 1/6.

Using the definition of a conditional probability, we can define a formula to find the joint probability of a set A of events of a sample space.





# Probability and Statistics

# Proposition 18.1.10 (Chain Rule for Conditional Probability)

Given a set A = {A1, . . . , An} of events of a sample space S, where all appropriate conditional probabilities are defined, we have the following:

Pr[A1 ∩ . . . ∩ An] = Pr[A1 | A2 ∩ . . . ∩ An] · Pr[A2 ∩ . . . ∩ An]

= Pr[A1 | A2 ∩ . . . ∩ An] · Pr[A2 | A3 ∩ . . . ∩ An] · Pr[A3 ∩ . . . ∩ An]

.

.

.

= Pr[A1 | A2 ∩ . . . ∩ An] · Pr[A2 | A3 ∩ . . . ∩ An] · · · Pr[An]

Finally, from the definition of a conditional probability, we see that:

Pr[B | A] Pr[A] = Pr[A ∩ B] = Pr[A | B] Pr[B]

This shows that:

Pr[B | A] = Pr[A | B] Pr[B]

Pr[A]

This is known as the Bayes’s Rule.

# 18.1.4 Independent Events

# Definition 18.1.11 (Independent Events)

Two events A, B are independent if Pr[A], Pr[B] > 0 and:

Pr[A] = Pr[A | B]

or equivalently:

Pr[B] = Pr[B | A]

or equivalently:

Pr[A ∩ B] = Pr[A] · Pr[B]

# Example 18.1.12

Consider the sample space of “the outcome of tossing two dice” again. Let A1 be the event “the number on the first die is 1” and let A2 be the event “the number on the second die is 4.” A1 and A2 are independent.

# Example 18.1.13

Consider the sample space of “the outcome of tossing two dice” again. Let A1 be the event “the number on the first die is 1” and let A2 be the event “the sum of the numbers on the two dice is 5.” A1 and A2 are not independent.

# 18.2 Random Variable

In the previous section, we only learned how to assign a probability to an event, a subset of the sample space. But in general, we can assign a probability to a broader concept called a random variable, associated to the sample space.





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

Definition 18.2.1 (Random Variable). Given a sample space S, a mapping X : S → R that maps each outcome x ∈ S to a value r ∈ R is called a random variable.

Example 18.2.2. Consider the sample space of “the outcome of tossing two dice” again. Then the random variable X = “sum of the numbers on the two dice” maps the outcome (1, 4) to the value 5.

Definition 18.2.3 (Sum and Product of Random Variables). If X, X1, . . . , Xn are random variables defined on the same sample space S such that X(x) = X1(x) + . . . + Xn(x) for every outcome x ∈ S, then we say that X is the sum of the random variables X1, . . . , Xn and denote

X = X1 + . . . + Xn

If X(x) = X1(x) × . . . × Xn(x) for every outcome x ∈ S, then we say that X is the product of the random variables X1, . . . , Xn and denote

X = X1 · · · Xn

Example 18.2.4. Consider the sample space of “the outcome of tossing two dice” again. Then the random variable X = “sum of the numbers on the two dice” is the sum of the two random variables X1 = “the number on the first die” and X2 = “the number on the second die.”

# 18.2.1 Probability of Random Variable

There is a natural relationship between the definition of an event and a random variable. Given a sample space S and random variable X : S → R, the “event that X takes a value in B” is denoted Pr[X ∈ B]. It is the total probability of all outcomes x ∈ S such that X(x) ∈ B. In particular, the event that X takes a particular value r ∈ R is denoted as X = r and the event that X takes a value in the interval [a, b] is denoted as a ≤ X ≤ b and so on.

Example 18.2.5. Consider the sample space of “the outcome of tossing two dice” and the random variable X = “sum of the numbers on the two dice” again. Then

Pr[X = 5] = Pr[{(1, 4), (2, 3), (3, 2), (4, 1)}] = 4/36

Often we are interested in the probability of the events of the form X ≤ x. Plotting the values of Pr[X ≤ x] with respect to x completely identifies the distribution of the values of X.

Definition 18.2.6 (Cumulative Distribution Function). Given a random variable X, there is an associated cumulative distribution function (cdf) FX : R → [0, 1] defined as

FX (x) = Pr[X ≤ x]



# Probability and Statistics

# 18.2.7 Proposition

The following properties hold for a cumulative distribution function FX:

1. FX is increasing
2. lim FX(x) = 0 and lim FX(x) = 1

x→−∞

x→∞

# 18.2.2 Discrete Random Variable

If the set of possible values of a random variable X is finite or countably infinite, we call it a discrete random variable. For a discrete random variable, the probability Pr[X = i] for each value i that the random variable can take completely identifies the distribution of X. In view of this fact, we denote the probability mass function (pmf) by

pX(i) = Pr[X = i]

# 18.2.8 Proposition

The following properties hold for a probability mass function pX:

1. ∑ pX(i) = 1

i
2. FX(x) = ∑ pX(i)

i≤x

# 18.2.3 Continuous Random Variable

We now consider the case where the set of all possible values of a random variable X is an interval or a disjoint union of intervals in R. We call such X a continuous random variable. In this case, the probability of the event X = i is zero for any i ∈ R. Instead, we care about the probability of the events of the form a ≤ X ≤ b.

# Definition 18.2.9 (Probability Density Function)

Given a continuous random variable X, there is an associated probability density function (pdf) fX : R → R≥0 such that

Pr[a ≤ X ≤ b] = ∫ab fX(x)dx

for any a, b ∈ R.

# 18.2.10 Proposition

The following properties hold for a probability density function fX:

1. ∫−∞∞ fX(x)dx = 1
2. FX(x) = ∫−∞x fX(y)dy




230  introduction to machine learning lecture notes for cos 324 at princeton university

# 18.2.4  Expectation and Variance

# Definition 18.2.11 (Expectation).

The expectation or the expected value of a discrete random variable X is defined as

E[X] = ∑i · pX(i) = ∑i · Pr[X = i]

where pX is its associated probability mass function. Similarly, the expectation for a continuous random variable X is defined as

E[X] = ∫−∞∞ x · fX(x)dx

where fX is the associated probability density function. In either case, it is customary to denote the expected value of X as μX or just μ if there is no source of confusion.

# Example 18.2.12.

Consider the sample space of “the outcome of tossing one die.” Then the expected value of the random variable X = “the number on the first die” can be computed as

E[X] = 1 · 1⁄6 + 2 · 1⁄6 + 3 · 1⁄6 + 4 · 1⁄6 + 5 · 1⁄6 + 6 · 1⁄6 = 3.5

36        36      36         36      36         36

# Proposition 18.2.13 (Linearity of Expectation).

If X is the sum of the random variables X1, . . . , Xn, then the following holds:

E[X] = E[X1] + . . . + E[Xn]

Also, if a, b ∈ R and X is a random variable, then

E[aX + b] = aE[X] + b

# Example 18.2.14.

Consider the sample space of “the outcome of tossing two dice.” Then the expected value of the random variable X = “the sum of the numbers of the two dice” can be computed as

E[X] = 3.5 + 3.5 = 7

since the expected value of the number on each die is 3.5.

# Definition 18.2.15 (Variance).

The variance of a random variable X, whose expected value is μ, is defined as

Var[X] = E[(X − μ)2]

Its standard deviation is defined as

σX = √Var[X]

It is customary to denote the variance of X as σ2X.





# Probability and Statistics 231

# Proposition 18.2.16.

If a ∈ R and X is a random variable, then

Var[aX] = a2Var[X]       σaX = |a| σX

# Problem 18.2.17.

Prove Chebyshev’s inequality:

Pr[|X − μ| ≥ kσ] ≤ 1/k2 for any k > 0. (Hint: Suppose the probability was greater than 1/k2. What could you conclude about E[(X − μ)2]?)

# 18.2.5 Joint and Conditional Distribution of Random Variables

Just as in events, we are interested in multiple random variables defined on the sample space.

# Definition 18.2.18 (Joint Distribution).

If X, Y are discrete random variables defined on the same sample space S, the joint probability mass function pX,Y is defined as

pX,Y(i, j) = Pr[X = i, Y = j]

where the event X = i, Y = j refers to the intersection (X = i) ∩ (Y = j). If X, Y are continuous random variables defined on S, there is an associated joint probability density function fX,Y: R → R≥0 such that Pr[a ≤ X ≤ b, c ≤ Y ≤ d] = ∫cd ∫ab fX,Y(x, y)dxdy

The joint probability mass/density function defines the joint distribution of the two random variables.

# Definition 18.2.19 (Marginal Distribution).

Given a joint distribution pX,Y or fX,Y of two random variables X, Y, the marginal distribution of X can be found as

pX(i) = ∑ pX,Y(i, j)

if X, Y are discrete and

fX(x) = ∫−∞∞ fX,Y(x, y)dy

if continuous. We can equivalently define the marginal distribution of Y.

# Definition 18.2.20 (Conditional Distribution).

Given a joint distribution pX,Y or fX,Y of two random variables X, Y, we define the conditional distribution of X given Y as

pX | Y(i | j) = pX,Y(i, j) / pY(j)





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

# 18.2.6 Bayes’ Rule for Random Variables

Sometimes it is easy to calculate the conditional distribution of X given Y, but not the other way around. In this case, we can apply the Bayes’ Rule to compute the conditional distribution of Y given X. Here, we assume that X, Y are discrete random variables. By a simple application of Bayes’ Rule, we have:

Pr[Y = j | X = i] = Pr[X = i | Y = j] Pr[Y = j] / Pr[X = i]

Now by the definition of a marginal distribution, we have:

Pr[X = i] = ∑ Pr[X = i, Y = j] = ∑ Pr[X = i | Y = j′] Pr[Y = j′] for all possible values j′ that Y can take. If we plug this into the denominator above:

Pr[Y = j | X = i] = Pr[X = i | Y = j] Pr[Y = j] / ∑ Pr[X = i | Y = j ] Pr[Y = j ]

# Example 18.2.21

There is a coin, where the probability of Heads is unknown and is denoted as θ. You are told that there is a 50% chance that it is a fair coin (i.e., θ = 0.5) and 50% chance that it is biased to be θ = 0.7. To find out if the coin is biased, you decide to flip the coin. Let D be the result of a coin flip. Then it is easy to calculate the conditional distribution of D given θ. For example:

Pr[D = H | θ = 0.5] = 0.5

But we are more interested in the probability that the coin is fair/biased based on the observation of the coin flip. Therefore, we can apply the Bayes’ Rule:

Pr[θ = 0.7 | D = H] = Pr[D = H | θ = 0.7] Pr[θ = 0.7] / Pr[D = H]

which can be calculated as:

Pr[D = H | θ = 0.7] Pr[θ = 0.7] / (Pr[D = H | θ = 0.7] Pr[θ = 0.7] + Pr[D = H | θ = 0.5] Pr[θ = 0.5])

= (0.7 * 0.5) / (0.7 * 0.5 + 0.5 * 0.5) ≃ 0.58





# 18.2.7 Independent Random Variables

So if we observe one Heads, there is a 58% chance that the coin was biased and a 42% chance that it was fair.

Problem 18.2.22. Consider Example 18.2.21 again. This time, we decide to throw the coin 10 times in a row. Let N be the number of observed Heads. What is the probability that the coin is biased if N = 7?

# Definition 18.2.23 (Independent Random Variables)

Two discrete random variables X, Y are independent if for every i, j, we have

pX(i) = pX | Y(i | j)

or equivalently,

pY(j) = pY | X(j | i)

or equivalently

pX,Y(x, y) = pX(x) · pY(y)

Two continuous random variables X, Y are independent if the analogous conditions hold for the probability density functions.

# Definition 18.2.24 (Mutually Independent Random Variables)

If any pair of n random variables X1, X2, . . . , Xn are independent of each other, then the random variables are mutually independent.

# Proposition 18.2.25

If X1, . . . , Xn are mutually independent random variables, the following properties are satisfied:

1. E[X1 · · · Xn] = E[X1] · · · E[Xn]
2. Var[X1 + . . . + Xn] = Var(X1) + . . . + Var(Xn)

We are particularly interested in independent random variables that have the same probability distribution. This is because if we repeat the same random process multiple times and define a random variable for each iteration, the random variables will be independent and identically distributed.

# Definition 18.2.26

If X1, . . . , Xn are mutually independent random variables that have the same probability distribution, we call them independent, identically distributed random variables, which is more commonly denoted as iid or i.i.d. random variables.




234   introduction to machine learning lecture notes for cos 324 at princeton university

# 18.3 Central Limit Theorem and Confidence Intervals

Now we turn our attention to two very important topics in statistics: Central Limit Theorem and confidence intervals. You may have seen confidence intervals or margin of error in the context of election polls. The pollster usually attaches a caveat to the prediction, saying that there is some probability that the true opinion of the public is ±ϵ of the pollster’s estimate, where ϵ is typically a few percent. This section is about the most basic form of confidence intervals, calculated using the famous Gaussian distribution. It also explains why the Gaussian pops up unexpectedly in so many settings.

A running example in this chapter is estimating the bias of a coin we have been given. Specifically, Pr[Heads] = p where p is unknown and may not be 1/2. We wish to estimate p by repeatedly tossing the coin. If we toss the coin n times, we expect to see around np Heads. Confidence intervals ask the converse question: after having seen the number of heads in n tosses, how “confidently” can we estimate p?

# 18.3.1 Coin Tossing

Suppose we toss the same coin n times. For each i = 1, 2, . . . , n, define the random variable Xi as an indicator random variable such that

Xi =

1      i-th toss was Heads

0      otherwise

It is easily checked that X1, . . . , Xn are iid random variables, each with E[Xi] = p and Var[Xi] = p(1 − p). Also if we have another random variable X = “number of heads,” notice that X is the sum of X1, . . . , Xn. Therefore, E[X] = np and Var[X] = np(1 − p).

Problem 18.3.1. Show that if Pr[Heads] = p then E[X] = np and Var[X] = np(1 − p). (Hint: use linearity of expectation and the fact that Xi’s are mutually independent.)

Suppose p = 0.8. What is the distribution of X? Figure 18.1 gives the distribution of X for different n’s. Let’s make some observations about Figure 18.1.

Expected value may not happen too often. For n = 10, the expected number of Heads is 8, but that is seen only with probability 0.3. In other words, with probability 0.7, the number of Heads is different from the expectation. 2 In such cases, expectation can be a misleading term. It may in fact be never seen. The highly likely values fall in a smaller and smaller band around the expected value, as n increases.

For n = 10, there is a good chance that the number of Heads is somewhere between 1 and 2 but no individual has a non-integral number of eyes. Thus mean value is a more intuitive term.




probability and statistics

# 18.3.2 Gaussian Distribution

We say that a real-valued random variable X is distributed according to N (μ, σ2), the Gaussian distribution with mean μ and variance σ2, if

FX(x) = 1 - (x−μ)2 / (2σ2)

It is hard to make an intuitive sense of this expression. The following figure gives us a better handle.

Figure 18.2: Cheatsheet for the Gaussian distribution with mean μ and variance σ2. It is tightly concentrated in the interval [μ − kσ, μ + kσ] for even k = 1 and certainly for k = 2, 3. Source: Wikipedia

| -3σ  | -2σ  | -1σ   | 0     | 1σ    | 2σ    | 3σ   |      |
| ---- | ---- | ----- | ----- | ----- | ----- | ---- | ---- |
| 0.1% | 2.1% | 13.6% | 34.1% | 34.1% | 13.6% | 2.1% | 0.1% |

Figure 18.2 shows that X concentrates very strongly around the mean μ. The probability that X lies in various intervals around μ of the type μ ± kσ are as follows: (i) For k = 1 it is 68.3%; (ii) For k = 2 it is 95.5%; (iii) For k = 3 it is 99.7%.




236  introduction to machine learning lecture notes for cos 324 at princeton university

# 18.3.3  Central Limit Theorem (CLT)

This fundamental result explains our observations in Subsection 18.3.1.

Theorem 18.3.2 (Central Limit Theorem, informal statement). Suppose X1, X2, . . . , is a sequence of random variables that are mutually independent and each of whose variance is upper bounded by some constant C. Then as n → ∞, the sum X1 + X2 + . . . + Xn tends to N (μ, σ2) where μ = ∑i E[Xi] and σ2 = ∑i Var(Xi).

We won’t prove this theorem. We will use it primarily via the “cheatsheet” of Figure 18.2.

# 18.3.4  Confidence Intervals

We return to the problem of estimating the bias of a coin, namely p = Pr[Heads]. Suppose we toss it n times and observe X heads. Then X  = ∑i Xi where Xi is the indicator random variable that signifies if the i-th toss is Heads.

Since the Xi’s are mutually independent, we can apply the CLT and conclude that X will approximately follow a Gaussian distribution as n grows. This is clear from Figure 18.1, where the probability histogram (which is a discrete approximation to the probability density) looks quite Gaussian-like for n = 1000. In this course we will assume for simplicity that CLT applies exactly. Using the mean and variance calculations from Problem 18.3.1, X is distributed like N (μ, σ2) where μ = np, σ2 = np(1 − p). Using the cheatsheet of Figure 18.2, we can conclude that

Pr[X ∈ [np − 2σ, np + 2σ]] ≤ 4.6%

Since X ∈ [np − 2σ, np + 2σ] if and only if np ∈ [X − 2σ, X + 2σ], some students have the following misconception:

Given the observation of X heads in n coin tosses, the probability that np ∈ [X − 2σ, X + 2σ] is at most 4.6%.

But there is no a priori distribution on p. It is simply some (unknown) constant of nature that we’re trying to estimate. So the correct inference should be:

If np ∈ [X − 2σ, X + 2σ], then the probability (over the n coin tosses) that we would have seen X heads is at most 4.6%.

The above is an example of confidence bounds. Of course, you may note that σ also depends on p, so the above conclusion doesn’t give us a clean confidence interval. In this course we use a simplifying assumption: to do the calculation we estimate σ2 as np′(1 − p′) where p′ = X/n. (The intuitive justification is that we expect p to be close to X/n.)





# 18.3 Confidence Intervals

# Example 18.3.3

Suppose X = 0.8n. Using our simplified calculation, σ2 ≈ n(0.8)(0.2), implying σ = 0.4√n. Thus we conclude that if p ∈ [0.8 − 0.4/√n, 0.8 + 0.4/√n], then the probability of observing this many Heads in n tosses would have been less than 100 − 68.2%, that is, less than 31.8%.

The concept of confidence intervals is also relevant to ML models.

# Example 18.3.4

A deep neural network model was trained to predict cancer patients’ chances of staying in remission a year after chemotherapy, and we are interested in finding out its accuracy p. When the model is tested on n = 1000 held-out data points, this problem is equivalent to the coin flipping problem. For each of the held-out data point, the probability that the model makes the correct prediction is p. By observing the number of correct predictions on the held-out data, we can construct a confidence interval for p. Say the test accuracy was p′ = 70%. Then the 68% confidence interval can be written as

np ∈ [np′ − σ, np′ + σ]

Substituting p′ = 0.7, σ ≈ np′(1 − p′), n = 1000, we get

1000p ∈ [685.5, 714.5]

or equivalently,

p ∈ [0.6855, 0.7145]

# 18.3.5 Confidence Intervals for Vectors

In the above settings, sampling was being used to estimate a real number, namely, Pr[Heads] for a coin. How about estimating a vector? For instance, in an opinion poll, respondents are being asked for opinions on multiple questions. Similarly, in stochastic gradient descent (Chapter 3), the gradient vector is being estimated by sampling a small number of data points. How can we develop confidence bounds for estimating a vector in Rk from n samples?

The confidence intervals for the coin toss setting can be easily extended to this case using the so called Union Bound:

Pr[A1 ∪ A2 ∪ · · · ∪ Ak] ≤ Pr[A1] + Pr[A2] + · · · + Pr[Ak] (18.2)

This leads to the simplest confidence bound for estimating a vector in Rk. Suppose the probability of the estimate being off by δi in the i-th coordinate is at most qi. Then

Pr[estimate is off by δ] ≤ q1 + q2 + · · · + qk

where δ = (δ1, δ2, . . . , δk)





# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

# 18.4 Final Remarks

The CLT applies to many settings, but it doesn’t apply everywhere. It is useful to clear up a couple of frequent misconceptions that students have:

1. Not every distribution involving a large number of samples is Gaussian. For example, scores on the final exam are usually not distributed like a Gaussian. Similarly, human heights are not really distributed like Gaussians.
2. Not everything that looks Gaussian-like is a result of the Central Limit Theorem. For instance, we saw that the distribution of weights in the sentiment model in Chapter 1 looked vaguely Gaussian-like, but they are not the sum of independent random variables as far as we can tell.





# 19

# Calculus

# 19.1 Calculus in One Variable

In this section, we briefly review calculus in one variable.

# 19.1.1 Exponential and Logarithmic Functions

When we multiply the same number a by itself n times, we denote it as an. The exponential function is a natural extension of this concept.

Definition 19.1.1 (Exponential Function). There is a unique function f : R → R such that f (n) = en for any n ∈ N and f (x + y) = f (x) f (y) for any x, y ∈ R. This function is called the exponential function and is denoted as ex or exp(x).

Proposition 19.1.2. The following properties hold for the exponential function:

1. exp(x) > 0 for any x ∈ R
2. exp(x) is increasing
3. limx→−∞ exp(x) = 0
4. limx→∞ exp(x) = ∞
5. exp(−x) = 1/exp(x)

We are also interested in the inverse function of the exponential function.

Definition 19.1.3 (Logarithmic Function). The logarithmic function log : (0, ∞) → R is defined as the inverse function of the exponential function. That is, log(x) = y where x = ey.

Proposition 19.1.4. The following properties hold for the logarithmic function:




240   introduction to machine learning lecture notes for cos 324 at princeton university

# 1. log(x) is increasing

2. lim  log(x) = −∞

x→0+

3. lim  log(x) = ∞

x→∞

4. log(xy) = log(x) + log(y)

# 19.1.2 Sigmoid Function

In Machine Learning, a slight variant of the exponential function, known as the sigmoid function is widely used.

Definition 19.1.5 (Sigmoid Function). The sigmoid function denoted as σ : R → R is defined as

σ(x) = 1 / 1 + exp(−x)

# Proposition 19.1.6.

The following properties hold for the sigmoid function:

- 1. 0 &#x3C; σ(x) &#x3C; 1 for any x ∈ R
- 2. σ(x) is increasing
- 3. lim  σ(x) = 0

x→−∞
- 4. lim  σ(x) = 1

x→∞
- 5. The graph of σ is symmetrical to the point (0, 1/2). In particular, σ(x) + σ(−x) = 1

Because of the last property in Proposition 19.1.6, the sigmoid function is well suited for binary classification (e.g., in logistic regression in Chapter 1). Given some output value x of a classification model, we interpret it as the measure of confidence that the input is of label 1, where we implicitly assume that the measure of confidence that the input is of label 2 is −x. Then we apply the sigmoid function to translate this into a probability distribution over the two labels.

# 19.1.3 Differentiation

Definition 19.1.7 (Derivative). Given a function f : R → R, its derivative f ′ is defined as

f ′(x) = lim (f (x + h) − f (x))

h→0     h

We alternatively denote f ′(x) as d f (x) / dx.





# Calculus 241

Example 19.1.8. The derivative of the exponential function is itself:

exp′(x) = exp(x)

and the derivative of the logarithmic function is:

log′(x) = 1

x

In general, there are more than two variables, that are related to each other through a composite function. The chain rule helps us find the derivative of the composite function.

# Definition 19.1.9 (Chain Rule).

If there are functions f, g : R → R such that y = f(x) and z = g(y), then

(g ◦ f)′(x) = g′(f(x)) f′(x) = d g(f(x)) · d f(x)

dy dx

or equivalently

dz = dz · dy

dx dy dx

# 19.2 Multivariable Calculus

In this section, we introduce the basics of multivariable calculus, which is widely used in Machine Learning. Since this is a generalization of the calculus in one variable, it will be useful to pay close attention to the similarity with the results from the previous section.

# 19.2.1 Mappings of Several Variables

So far, we only considered functions of the form f : R → R that map a real value x to a real value y. But now we are interested in mappings f : Rn → Rm that map a vector x = (x1, . . . , xn) with n coordinates to a vector y = (y1, . . . , ym) with m coordinates. In general, a function is a special case of a mapping where the range is R. If the mappings are of the form f : Rn → R (i.e., m = 1), it can still be called a function of several variables.

First consider an example where m = 1.

Example 19.2.1. Let f(x1, x2) = x2 + x2 be a function in two variables. This can be understood as mapping a point x1, x2 in the Cartesian coordinate system to its squared distance from the origin. For example, f(3, 4) = 25 shows that the squared distance between the point (3, 4) and the origin (0, 0) is 25.

When m > 1, we notice that each coordinate y1, . . . , ym is a function of x1, . . . , xn. Therefore, we can decompose f into m functions f1, . . . , fm : Rn → R such that

f(x) = (f1(x), . . . , fm(x))




242  introduction to machine learning lecture notes for cos 324 at princeton university

# Example 19.2.2.

Let f (x1, x2) = (x2x2, x1x2) be a mapping from R2 to R2.

Then we can decompose f into two functions f1, f2 in two variables where

- f1(x1, x2) = x2x2
- f2(x1, x2) = x1x2

# 19.2.2  Softmax Function

The softmax function is a multivariable function widely used in Machine Learning, especially for multi-class classification (see Chapter 4, Chapter 10). It takes in a vector of k values, each corresponding to a particular class, and outputs a probability distribution over the k classes — that is, a vector of k non-negative values that sum up to 1. The resulting probability is exponentially proportional to the input value of that class. We formally write this as:

# Definition 19.2.3 (Softmax Function).

Given a vector z = (z1, z2, . . . , zk) ∈ Rk, we define the softmax function as a probability function so f tmax : Rk → [0, 1]k where the “probability of predicting class i” is:

so f tmax(z)i = ∑j=1k ezi ezj (19.1)

# Problem 19.2.4.

Show that for k = 2, the definition of the softmax function is equivalent to the sigmoid function (after slight rearrangement/renaming of terms).

The sigmoid function is used for binary classification, where it takes in a single real value and converts it to a probability of one class (and the probability of the other class can be inferred as its complement). The softmax function is used for multi-class classification, where it takes in k real values and converts them to k probabilities, one for each class.

# 19.2.3  Differentiation

Just like with functions in one variable, we can define differentiation for mappings in several variables. The key point is that now we will define a partial derivative for each pair (xi, yj) of coordinate xi of the domain and coordinate yj of the range.

# Definition 19.2.5 (Partial Derivative).

Given a function f : Rn → Rm, the partial derivative of yj with respect to xi at the point x is defined as

∂yj = limh→0 fj(x1, . . . , xj + h, . . . , xn) − fj(x1, . . . , xj, . . . , xn) / h





# Calculus

Definition 19.2.6 (Gradient). If f : Rn → R is a function of several variables, the gradient of f is defined as a mapping ∇f : Rn → Rn that maps each input vector to the vector of partial derivatives at that point:

∇f(x) = ∂f, . . . , ∂f

∂x1 x

∂xn

Similarly to the chain rule in one variable, we can define a chain rule for multivariable settings. The key point is that there are multiple ways that a coordinate xj can affect the value of zi. Definition 19.2.7 can be thought as applying the chain rule for one variable in each of the paths, and adding up the results.

Figure 19.4: A visualization of the chain rule in multivariable settings. Notice that x2 can affect the value of z1 in three different paths. The amount of effect from each path will respectively be calculated as (∂z1/∂y1)(∂y1/∂x2) (red), (∂z1/∂y2)(∂y2/∂x2) (blue), and (∂z1/∂y3)(∂y3/∂x2) (cyan).

Definition 19.2.7 (Chain Rule). If f : Rn → Rm and g : Rm → Rℓ are mappings of several variables, where y = f(x) and z = g(y), the following chain rule holds for each 1 ≤ i ≤ ℓ and 1 ≤ j ≤ n:

∂zi / ∂xj = ∑k=1m ∂zi / ∂yk · ∂yk / ∂xj

Example 19.2.8. Suppose we define the functions h = s + t2, s = 3x, and t = x − 2. Then, we can find the partial derivative ∂h using the chain rule:

∂h / ∂x = ∂s / ∂x + ∂(t2) / ∂x

= ∂s / ∂x + ∂(t2) / ∂t · ∂t / ∂x

= 3 + 2t · 1

= 2x − 1

Problem 19.2.9. Suppose we define the functions h = s + t2, s = xy, and t = x − 2y. Compute the partial derivative ∂h / ∂x.




NO_CONTENT_HERE



# 20 Linear Algebra

# 20.1 Vectors

Vectors are a collection of entries (here, we focus only on real numbers). For example, the pair (1, 2) is a real vector of size 2, and the 3-tuple (1, 0, 2) is a real vector of size 3. We primarily categorize vectors by their size. For example, the set of all real vectors of size n is denoted as Rn. Any element of Rn can be thought of as representing a point (or equivalently, the direction from the origin to the point) in the n-dimensional Cartesian space. A real number in R is also known as a scalar, as opposed to vectors in Rn where n > 1.

# 20.1.1 Vector Space

We are interested in two operations defined on vectors — vector addition and scalar multiplication. Given vectors x = (x1, x2, . . . , xn) and y = (y1, y2, . . . , yn) and a scalar c ∈ R, the vector addition is defined as

x + y = (x1 + y1, x2 + y2, . . . , xn + yn) ∈ Rn

where we add each of the coordinates element-wise. As shown in Figure 20.2, vector addition is the process of finding the diagonal of the parallelogram made by the two vectors x and y. The scalar multiplication is similarly defined as

cx = (cx1, cx2, . . . , cxn) ∈ Rn

As shown in Figure 20.3, scalar multiplication is the process of scaling one vector up or down. Rn is closed under these two operations — i.e., the resulting vector of either operation is still in Rn. Any subset S of Rn that is closed under vector addition and scalar multiplication is known as a subspace of Rn.

Figure 20.1: A visualization of a vector v = (2, 1) in R2.

Figure 20.2: A visualization of x + y where x = (2, 1) and y = (1, 2).

Figure 20.3: A visualization of 0.5x where x = (4, 2).




246  introduction to machine learning lecture notes for cos 324 at princeton university

# 20.1.2  Inner Product

The inner product is defined as

n

x · y = x1y1 + x2y2 + . . . + xnyn = ∑ xiyi ∈ R

i=1

Closely related to the inner product is the norm of a vector, which measures the length of it. It is defined as ∥x∥ = √x · x. 1 There are many other definitions of a norm. This particular one is called an ℓ2 norm.

# Proposition 20.1.1.

The inner product satisfies the following properties:

- Symmetry: x · y = y · x
- Linearity: (a1x1 + a2x2) · y = a1(x1 · y) + a2(x2 · y)

and the norm satisfies the following property:

- Absolute Homogeneity: ∥ax∥ = |a| ∥x∥

# 20.1.3  Linear Independence

Any vector of the form

a1x1 + a2x2 + . . . + akxk

where ai’s are scalars and xi’s are vectors is called a linear combination of the vectors xi’s. Notice that the zero vector 0 (i.e., the vector with all zero entries) can always be represented as a linear combination of an arbitrary collection of vectors, if all ai’s are chosen as zero. This is known as a trivial linear combination, and any other choice of ai’s is known as a non-trivial linear combination.

# Definition 20.1.2.

k vectors x1, x2, . . . , xk ∈ Rn are called linearly dependent if 0 can be represented as a non-trivial linear combination of the vectors x1, . . . , xk; or equivalently, if one of the vectors can be represented as a linear combination of the remaining k − 1 vectors. The vectors that are not linearly dependent with each other are called linearly independent.

Consider the following analogy. Imagine trying to have a family style dinner at a fast food restaurant, where the first person orders a burger, the second person orders a chilli cheese fries, and the third person orders a set menu with a burger and a chili cheese fries. The third person’s order did not contribute to the diversity of the food on the dinner table. Similarly, if some set of vectors are linearly dependent, it means that at least one of the vectors is redundant.

# Example 20.1.3.

The set {(−1, 2), (3, 0), (1, 4)} of three vectors is linearly dependent because

(1, 4) = 2 · (−1, 2) + (3, 0)

can be represented as the linear combination of the remaining two vectors.




# Linear Algebra

# 20.1.4 Span

Example 20.1.4. The set {(−1, 2, 1), (3, 0, 0), (1, 4, 1)} of three vectors is linearly independent because there is no way to write one vector as a linear combination of the remaining two vectors.

# Definition 20.1.5

The span of a set of vectors x1, . . . , xk is the set of all vectors that can be represented as a linear combination of xi’s.

Example 20.1.6. (1, 4) is in the span of {(−1, 2), (3, 0)} because

(1, 4) = 2 · (−1, 2) + (3, 0)

Example 20.1.7. (1, 4, 1) is not in the span of {(−1, 2, 1), (3, 0, 0)} because there is no way to choose a1, a2 ∈ R such that

(1, 4, 1) = a1(−1, 2, 1) + a2(3, 0, 0)

The span is also known as the subspace generated by the vectors x1, . . . , xk. This is because if you add any two vectors in the span, or multiply one by a scalar, it is still in the span (i.e., the span is closed under vector addition and scalar multiplication).

Example 20.1.8. In the R3, the two vectors (1, 0, 0) and (0, 1, 0) span the 2-dimensional XY-plane. Similarly, the vectors (1, 0, 1) and (0, 2, 1) span the 2-dimensional plane 2x + y − 2z = 0. The term dimension will be formally defined soon. Here, we rely on your intuition.

In Example 20.1.8, we see examples where 2 vectors span a 2-dimensional subspace. In general, the dimension of the subspace spanned by k vectors can go up to k, but it can also be strictly smaller than k. This is related to the linear independence of the vectors.

# Proposition 20.1.9

Given k vectors, x1, . . . , xk ∈ Rn, there is a maximum number d ≥ 1 such that there is some subcollection xi₁, . . . , xid of these vectors that are linearly independent. Then

span(x1, . . . , xk) = span(xi₁, . . . , xid) (20.1)

is a d-dimensional subspace of Rn.

Conversely, if we know that the span of the k vectors is a d-dimensional subspace, then the maximum number of vectors that are linearly independent with each other is d, and any subcollection of linearly independent d vectors satisfies (20.1).

Proposition 20.1.9 states that the dimension of the span of some set of k vectors is equivalent to the maximum number d of linearly independent vectors. It also states that the span of the k vectors is equal to the span of the linearly independent d vectors, meaning all of the information is captured by the d vectors; the remaining k − d



# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

Vectors are just redundancies. But trying to directly compute the maximum number of linearly independent vectors is inefficient — it may require checking the linear independence of an exponential number of subsets of the vectors. In the next section, we discuss a concept called matrix rank that is very closely related to this topic.

# 20.1.5 Orthogonal Vectors

Definition 20.1.10. If vectors x1, . . . , xk ∈ Rn satisfy xi · xj = 0 for any i ≠ j, then they are called orthogonal vectors. In particular, if they also satisfy the condition that ∥xi∥ = 1 for each i, then they are also orthonormal.

In Rn, orthogonal vectors form a 90 degree angle with each other.

Example 20.1.11. The two vectors (1, 0), (0, 2) are orthogonal. So are the vectors (1, 2), (−2, 1).

Given any set of orthogonal vectors, it is possible to transform it into a set of orthonormal vectors, by normalizing each vector (i.e., scale it such that the norm is 1).

Figure 20.4: A visualization of orthogonal vectors x = (1, 2) and y = (−2, 1).

# 20.1.6 Basis

Definition 20.1.12. A collection {x1, . . . , xk} of linearly independent vectors in Rn that span a set S is known as a basis of S. In particular, if the vectors of the basis are orthogonal/orthonormal, the basis is called an orthogonal/orthonormal basis of S.

The set S in Definition 20.1.12 can be the entire vector space Rn, but it can also be some subspace of Rn with a lower dimension.

Example 20.1.13. The set {(1, 0, 0), (0, 1, 0), (0, 0, 1)} of three vectors is a basis for R3. When we exclude the last vector (0, 0, 1), the set {(1, 0, 0), (0, 1, 0)} is a basis of the 2-dimensional XY-plane in R3.

Given some subspace S, the basis of S is not unique. However, every basis of S must have the same size — this size is called the dimension of S. For a finite dimensional space S, it is known that there exists an orthogonal basis of S. There is a well-known algorithm — Gram-Schmidt process — that can transform an arbitrary basis into an orthogonal basis (and eventually an orthonormal basis via normalization).

# 20.1.7 Projection

Vector projection is the key concept used in the Gram-Schmidt process that computes an orthogonal basis. Given a fixed vector a, it decomposes any given vector x into a sum of two components — one that is





# Linear Algebra

orthogonal to a (“distinct information”) and the other that is parallel to a (“redundant information”).

# Definition 20.1.14 (Vector Projection)

Fix a vector a ∈ Rn. Given another vector x, the projection of x on a is defined as

proa(x) = x · a a

a · a

and is parallel to the fixed vector a. The remaining component

x − proa(x)

is called the rejection of x from a and is orthogonal to a.

# Proposition 20.1.15 (Pythagorean Theorem)

If x, y are orthogonal, then

∥x + y∥2 = ∥x∥2 + ∥y∥2

In particular, given two vectors a, x, we have

∥x − proa(x)∥2 = ∥x∥2 − ∥proa(x)∥2

Now assume we are given a space S and a subspace T ⊂ S. Then a vector x ∈ S in the larger space does not necessarily belong in T. Instead, we can find a vector x′ ∈ T that is “closest” to x using vector projection.3 We ask you to prove this in Problem 7.1.3.

# Definition 20.1.16 (Vector Projection on Subspace)

Given a space S, its subspace T with an orthogonal basis {t1, . . . , tk}, and a vector x ∈ S, the projection of x on T is defined as

projT(x) = ∑i=1k projti(x) = ∑i=1k ti · ti} ti

the sum of projection of x on each of the basis vectors of T.

# 20.2 Matrices

Matrices are a generalization of vectors in 2-dimension — a m × n matrix is a collection of numbers assembled in a rectangular shape of m rows and n columns. The set of all real matrices of size m × n is denoted as Rm×n. A vector of size n is customarily understood as a column vector — that is, a n × 1 matrix. Also, if m = n, then the matrix is known as a square matrix.

# 20.2.1 Matrix Operation

Similarly to vector operations, we are interested in four matrix operations — matrix addition, scalar multiplication, matrix multiplication,




# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

Given a scalar c ∈ R and matrices X, Y ∈ Rm×n such that

| x1,1 | x1,2 | · · · | x1,n |
| ---- | ---- | ----- | ---- |
| x2,1 | x2,2 | · · · | x2,n |
| .    | .    | .     | .    |
| .    | .    | .     | .    |
| .    | .    | .     | .    |
| xm,1 | xm,2 | · · · | xm,n |

and

| y1,1 | y1,2 | · · · | y1,n |
| ---- | ---- | ----- | ---- |
| y2,1 | y2,2 | · · · | y2,n |
| .    | .    | .     | .    |
| .    | .    | .     | .    |
| .    | .    | .     | .    |
| ym,1 | ym,2 | · · · | ym,n |

The matrix addition is defined as

| x1,1 + y1,1 | x1,2 + y1,2 | · · · | x1,n + y1,n |
| ----------- | ----------- | ----- | ----------- |
| x2,1 + y2,1 | x2,2 + y2,2 | · · · | x2,n + y2,n |
| .           | .           | .     | .           |
| .           | .           | .     | .           |
| .           | .           | .     | .           |
| xm,1 + ym,1 | xm,2 + ym,2 | · · · | xm,n + ym,n |

where we add each of the coordinates element-wise. The scalar multiplication is similarly defined as

| c x1,1 | c x1,2 | · · · | c x1,n |
| ------ | ------ | ----- | ------ |
| c x2,1 | c x2,2 | · · · | c x2,n |
| .      | .      | .     | .      |
| .      | .      | .     | .      |
| .      | .      | .     | .      |
| c xm,1 | c xm,2 | · · · | c xm,n |

The matrix multiplication XY is defined for a matrix X ∈ Rℓ×m and a matrix Y ∈ Rm×n; that is, when the number of columns of the first matrix is equal to the number of rows of the second matrix. The output XY of the matrix multiplication will be a ℓ × n matrix. The (i, j) entry of the matrix XY is defined as

(XY)i,j = ∑k=1m xi,k yk,j

That is, it is defined as the inner product of the i-th row of X and the j-th column of Y.

# Proposition 20.2.1

The above matrix operations satisfy the following properties:

- c(XY) = (cX)Y = X(cY)
- (X1 + X2)Y = X1Y + X2Y
- X(Y1 + Y2) = XY1 + XY2

Finally, the transpose X⊺ ∈ Rn×m of a matrix X ∈ Rm×n is the resulting matrix when the entries of X are reflected down the diagonal. That is,

(X⊺)i,j = Xj,i


linear algebra 251

# Proposition 20.2.2.

The transpose of a matrix satisfies the following properties:

- (X + Y)⊺ = X⊺ + Y⊺
- (cX)⊺ = c(X⊺)
- (XY)⊺ = Y⊺ X⊺

# 20.2.2 Matrix and Linear Transformation

Recall that a vector of size n is often considered a n × 1 matrix. Therefore, given a matrix A ∈ Rm×n and a vector x ∈ Rn, we can define the following operation

y = Ax ∈ Rm through matrix multiplication. This shows that A can be understood as a mapping from Rn to Rm. We see that ai,j (the (i, j) entry of the matrix A) is the coefficient of xj (the j-th coordinate of the input vector) when computing yi (the i-th coordinate of the output vector). Since each yi is linear in terms of each xj, we say that A is a linear transformation.

# 20.2.3 Matrix Rank

Matrix rank is one of the most important concepts in basic linear algebra.

# Definition 20.2.3.

Given a matrix A ∈ Rm×n of m rows and n columns, the number of linearly independent rows is known to be always equal to the number of linearly independent columns. This common number is known as the rank of A and is denoted as rank(A).

The following property of rank is implied in the definition, but we state it explicitly as follows.

# Proposition 20.2.4.

The rank of a matrix is invariant to reordering rows/columns.

# Example 20.2.5.

Consider the matrix M =

| 1  | 1  | −2 | 0 |
| -- | -- | -- | - |
| −1 | −1 | 2  | 0 |

we notice that the second row is simply the first row negated, and thus the rank of M is 1.

# Example 20.2.6.

Consider the matrix M =

| 1 | 0 | 0 |
| - | - | - |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

the rank of M is 3 because all the row (or column) vectors are linearly independent (they form basis vectors of R3).





# Introduction to Machine Learning Lecture Notes for COS 324 at Princeton University

Example 20.2.7. Consider the matrix M =
| -2 | -3 | 1 |
| -- | -- | - |
| 3  | 3  | 0 |

the rank of M is 2 because the third row can be expressed as the second row subtracted from the first row.

When we interpret a matrix as a linear transformation, the rank measures the dimension of the output space.

Proposition 20.2.8. A ∈ Rm×n has rank k if and only if the image of the linear transformation, i.e., the subspace

{Ax | x ∈ Rn}

of Rm, has dimension k.

There are many known algorithms to compute the rank of a matrix. Examples include Gaussian elimination or certain decompositions (expressing a matrix as the product of other matrices with certain properties). Given m vectors in Rn, we can find the maximum number of linearly independent vectors by constructing a matrix with each row equal to each vector4 and finding the rank of that matrix.

4 By Proposition 20.2.4, the order of the rows can be arbitrary.

# 20.2.4 Eigenvalues and Eigenvectors

Say we have a square matrix A ∈ Rn×n. This means that the linear transformation expressed by A is a mapping from Rn to itself. For most vectors x ∈ Rn, x is mapped to a very “different” vector Ax under this mapping. However, some vectors are “special” and they are mapped to another vector with the same direction.

Definition 20.2.9 (Eigenvalue/Eigenvector). Given a square matrix A ∈ Rn×n, if a vector v ∈ Rn satisfies

Av = λv

for some scalar λ ∈ R, then v is known as an eigenvector of A, and λ is its corresponding eigenvalue.

Each eigenvector can only be associated with one eigenvalue, but each eigenvalue may be associated with multiple eigenvectors.

Proposition 20.2.10. If x, y are both eigenvectors of A for the same eigenvalue λ, then any linear combination of them is also an eigenvector for A with the same eigenvalue λ.

Proposition 20.2.10 shows that the set of eigenvectors for a particular eigenvalue forms a subspace, known as the eigenspace of that eigenvalue. The dimension of this subspace is known as the geometric multiplicity of the eigenvalue. The following result ties together some of the concepts we discussed so far.





# 20.2.11 Rank-Nullity Theorem

Given a square matrix A ∈ Rn×n, the eigenspace of 0 is the set of all vectors that get mapped to zero vector 0 under the linear transformation A. This subspace is known as the null space of A and its dimension (i.e., the geometric multiplicity of 0) is known as the nullity of A and is denoted as nullity(A). Then

rank(A) + nullity(A) = n

# 20.3 Advanced: SVD/PCA Procedures

Now we briefly introduce a procedure called Principal Component Analysis (PCA), which is commonly used in low-dimensional representation as in Chapter 7. We are given vectors v1, v2, . . . , vN ∈ Rd and a positive integer k and wish to obtain the low-dimensional representation in the sense of Definition 7.1.1 that minimizes ϵ. This is what we mean by “best” representation.

# Theorem 20.3.1

The best low-dimensional representation consists of k eigenvectors corresponding to the top k eigenvalues (largest numerical values) of the matrix AA⊺ where the columns of A are v1, v2, . . . , vN.

Theorem 20.3.1 shows what the best low-dimensional representation is, but it does not show how to compute it. It turns out something called the Singular Value Decomposition (SVD) of the matrix A is useful. It is known that any matrix A can be decomposed into the following product

A = UΣV⊺

where Σ is a diagonal matrix with entries equal to the square root of the nonzero eigenvalues of AA⊺ and the columns of U are the orthonormal eigenvectors of AA⊺, where the i-th column is the eigenvector that corresponds to the eigenvalue at the i-th diagonal entry of Σ. There are known computationally efficient algorithms that will perform the SVD of a matrix.

In this section, we will prove Theorem 20.3.1 for the case where k = 1. To do this, we need to introduce some preliminary results.

# Theorem 20.3.2

If a square matrix A ∈ Rn×n is symmetric (i.e., A = A⊺), then there is an orthonormal basis of Rn consisting of n eigenvectors of A.5

Proof. A real symmetric matrix is known to be diagonalizable, and diagonalizable matrices are known to have n eigenvectors that form a basis for Rn. In particular, the eigenvectors are linearly independent, meaning the eigenvectors corresponding to a particular eigenvalue λ will form a basis for the corresponding eigenspace. Through the Gram-Schmidt process, we can replace some of these eigenvectors.




254    introduction to machine learning lecture notes for cos 324 at princeton university

such that the eigenvectors for λ are orthogonal to each other. That is, if u, v are eigenvectors for the same eigenvalue λ, then u · v = 0. Now assume u, v are two eigenvectors with distinct eigenvalues λ, μ respectively. Then

λu · v = (λu) · v = (Au) · v = ∑i,j=1n ai,j uj vi

= u · (A⊺ v) = u · (Av) = u · (μv) = μu · v

where the third and the fourth equality can be verified by direct computation. Since λ = μ, we conclude u · v = 0. We have now showed that u · v = 0 for any pair of eigenvectors u, v — this means that the basis of eigenvectors is also orthogonal. After normalization, the basis can be made orthonormal.

The following result is not necessarily needed for the proof of Theorem 20.3.1, but the proofs are similar.

# Theorem 20.3.3.

If A ∈ Rn×n is symmetric, then the unit vector x that maximizes ∥Ax∥ is an eigenvector of A with an eigenvalue whose absolute values is the largest out of all eigenvalues.

# Proof.

By Theorem 20.3.2, there is an orthonormal basis {u1, . . . , un} of Rn consisting of eigenvectors of A. Then any vector x is in the span of the eigenvectors and can be represented as the linear combination

x = α1u1 + α2u2 + . . . + αnun

for some scalars αi’s. Then

∥x∥2 = x · x = (α1u1 + α2u2 + . . . + αnun) · (α1u1 + α2u2 + . . . + αnun)

= ∑i,j=1n αiαj(ui · uj) = ∑i=1n αi2

where for the last equality, we use the fact that ui’s are orthonormal — that is, ui · uj = 0 if i ≠ j and ui · ui = 1. Since x has norm 1, we see that ∑i=1n αi2 = 1. Now notice that

Ax = A(α1u1 + α2u2 + . . . + αnun) = α1Au1 + α2Au2 + . . . + αnAun

= α1λ1u1 + α2λ2u2 + . . . + αnλnun



# linear algebra

where λi is the eigenvalue for the eigenvector ui. Following a similar computation as above,

∥Ax∥2 = ∑i=1n αi2λi2

The allocation of weights αi that will maximize ∑i=1n αi2λi2 while maintaining ∑i=1n αi = ±1 is assigning αi = ±1 to the eigenvalue λi that has the highest value of λi. This shows that the unit vector x = ui is an eigenvector with the eigenvalue λi.

We now prove one last preliminary result.

# Theorem 20.3.4.

For a matrix A ∈ Rm×n, the matrix AA⊺ is symmetric and its eigenvalues are non-negative.

Proof. The first part can be verified easily by observing that

(AA⊺)⊺ = (A⊺)⊺A⊺ = AA⊺

Now assume x is an eigenvector of A with eigenvalue λ. Then

AA⊺x = λx

We multiply by x⊺ on the left on both sides of the equation.

x⊺AA⊺x = x⊺(λx) = λ∥x∥2

At the same time, notice that

x⊺AA⊺x = (A⊺x)⊺(A⊺x) = ∥A⊺x∥2

which shows that λ∥x∥2 = ∥A⊺x∥2

Since ∥x∥2, ∥A⊺x∥2 are both non-negative, λ is also non-negative.

We are now ready to (partially) prove the main result of this section.

# Proof of Theorem 20.3.1.

We prove the case where k = 1. Recall that we want to find a vector u that minimizes the error of the low-dimensional representation:

∑i=1N ∥vi − vi∥2

where vi is the low-dimensional representation of vi that can be computed as

vi = (vi · u)u




# Introduction to Machine Learning

# Lecture Notes for COS 324 at Princeton University

By the result of Problem 7.1.3. Now by Proposition 20.1.15, we see that

N
∑ ∥vi − (vi · u)u∥
= ∑ ∥vi∥² − ∥(vi · u)u∥
i=1

N
= ∑ ∥vi∥² − (vi · u)²
i=1

Since we are already given a fixed set of vectors vi, we cannot change the values of ∥vi∥². Therefore, minimizing the last term of the equation above amounts to maximizing

N
∑ (vi · u)²
= ∥A u∥² = u⊺AA u
i=1

By Theorem 20.3.2 and by Theorem 20.3.4, there is an orthonormal basis {u1, . . . , un} of Rn that consist of the eigenvectors of the matrix AA⊺. Let λi be the eigenvalue corresponding to the eigenvector ui. Then similarly to the proof of Theorem 20.3.3, we can represent any vector u as a linear combination of the eigenvectors as

u = α1u1 + α2u2 + . . . + αnun

Then we have

n
∑ αi² = 1
i=1

u⊺AA⊺u = (α1u1 + α2u2 + . . . + αnun)⊺AA⊺(α1u1 + α2u2 + . . . + αnun)

= (α1u1 + α2u2 + . . . + αnun)⊺(α1λ1u1 + α2λ2u2 + . . . + αnλnun)

n
= ∑ αiαjλj(ui · uj)
i,j=1

n
= ∑ αi²λi
i=1

Again, the allocation of α’s that maximize

n
∑ αi²λi
while maintaining
∑ αi² = 1
i=1

is assigning αi = ±1 to the eigenvector corresponding to the highest value of λi.
