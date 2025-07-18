a) Which model gives you better performance: Multiple Linear Regression or Polynomial Regression?
Provide evidence from your analysis (such as R-squared, Mean Squared Error, or other performance metrics) and explain your reasoning. Be sure to compare the results from both models and justify why one might perform better than the other based on the dataset and problem you're solving.

Based on the analysis, Multiple Linear Regression performs better than Polynomial Regression.The r2 score for Linear Regression is 0.417, meaning it explains about 42% of the changes in house prices.The r2 score for Polynomial Regression is 0.0, which means it doesn't explain the data well.Although Polynomial Regression has slightly lower MSE and MAE, the difference is very small.So, Linear Regression is the better model for predicting house prices based on sqft_living in this case.

 
b)How do you decide the optimal degree for Polynomial Regression in this case?
Explain how you determined the degree of the polynomial and what criteria you used to decide whether a higher degree improves the model's performance. Discuss the potential risks of choosing too high or too low a degree for the polynomial.

To find the best degree for Polynomial Regression, I tested different degrees and checked the model performance using r2,MSE, MAE on the test data.When I used degree 2, I got r2= 0.25.I chose this degree because it gives a good balance between accuracy and simplicity.If the degree is too low, the model is too simple (underfitting).If the degree is too high, the model may fit noise and perform poorly on new data (overfitting).So, degree 2 is a good choice for this dataset.
 

