
# Polynomial Regression Reflection

a) **Multiple Linear Regression:**
r2-score: 0.423
mean squared error: 61619674038.75
mean absolute error: 164763.13

Order 5 polynomial Regression:
r2-score: 0.62
mean squared error: 40567034684.0
mean absolute error: 117757.0

Observing from these matrices, Order 5 Polynomial regression performs better, determining by higher r2-score and lesser mse/mae scores.

b) **Deciding the optimal degree**

To determine the optimal degree for polynomial regression, I compared the model’s performance metrics (r2-score, mean squared error, and mean absolute error) for polynomial degrees 2 through 5:

Degree: 2 | r2-score: 0.449 | mse: 58,809,893,909.87 | mae: 162,638.92  
Degree: 3 | r2-score: 0.453 | mse: 58,397,966,523.22 | mae: 163,806.28  
Degree: 4 | r2-score: 0.538 | mse: 49,324,162,524.08 | mae: 141,803.82  
Degree: 5 | r2-score: 0.620 | mse: 40,567,034,684.26 | mae: 117,756.92

I observed that as the degree increases, the r2-score improves and both MSE and MAE decrease.

Degree 5 gives the best performance on the training data, with the highest r2-score and lowest errors.

 However, there are risks of choosing between too low and too high degree. The model may underfit, failing to capture important patterns in the data when we choose too low degree. In the contradictory, the model may overfit, capturing noise rather than the underlying trend. This can lead to poor generalization on new, unseen data.

In conclusion, in order to identify the optimal degree in this problem, it is essential to choose base on using cross validation or test set performance not only using the training data. Moreover, we might capture better using more training data instead of using only 50 rows.
