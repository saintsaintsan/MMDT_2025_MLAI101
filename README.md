# MMDT_2025_MLAI101

# Gradient Descent: An Essential Algorithm in Data Science

Gradient Descent is a core optimization algorithm used to minimize a function, typically called a "Cost Function" or "Loss Function." This function measures how much error your machine learning model has. The goal is to find the set of model parameters that makes this error as small as possible.

It's a foundational algorithm used to train many machine learning models like Linear Regression, Logistic Regression, and Neural Networks.

## How It Works
Imagine you're trying to find the lowest point in a valley. Gradient Descent works like this:

Start Somewhere: You begin at any random spot on the "hill." (These are your model's starting values).
Find the Downhill Path: From where you are, figure out which direction is the steepest way down. (This "downhill path" is called the "gradient.")
Take a Step: You take a step down that path. How big is your step? That's controlled by the "Learning Rate."
A small Learning Rate means tiny steps; it's slow.
A large Learning Rate means big steps; it might be fast but could jump over the lowest point.
Repeat: You keep repeating steps 2 and 3, taking one step at a time, until you gradually reach the very bottom of the valley (the point of minimum error).
In Simple Terms:
New Value = Current Value - (Learning Rate * Downhill Direction)

## Types of Gradient Descent

The main types differ by how much data they use to calculate the gradient at each step:

1.  **Batch Gradient Descent (BGD):**
2.  **Stochastic Gradient Descent (SGD):**
3.  **Mini-Batch Gradient Descent (MBGD):**
   
## Where Is It Used?

Gradient Descent is widely used in:
* Linear Regression
* Logistic Regression
* Neural Networks (the backpropagation algorithm is a form of gradient descent)
* Any machine learning problem where you need to minimize a differentiable cost function to train a model.



