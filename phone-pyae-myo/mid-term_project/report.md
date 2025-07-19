# Body Fat Prediction Using Polynomial Regression

**Why I chose this topic:**
I chose the Body Fat Percentage Prediction project because it provides an opportunity to explore how various physiological and anthropometric features relate to body fat percentage. This topic is both practical and relevant, especially in health, fitness, and medical fields where body composition assessment is crucial. This project seemed like a perfect fit because it had multiple numerical features that could potentially have non-linear relationships with the target variable (body fat percentage).

**Dataset and approach:**
I used a body fat dataset from Kaggle that had 252 people with various body measurements. What I liked about it was that it was already clean with no missing data. I was initially about to use only the four features but I had to take all of the listed measurements to predict body fat percentage because later I found out that taking all of these features would give me a really accurate model.

**How I built the model:**
I set up a pipeline that first scaled all the data, then transformed it into polynomial features, and finally train the data. Instead of just guessing what polynomial degree to use, I used GridSearchCV to test degrees from 2 to 9. It turned out that degree 2 worked best. I split my data 70-30 for training and testing.

**Findings & Evaluations:**
At first, my model did pretty well on training data with an R² of 0.77, meaning it explained about 77% of the variance. But when I tested it on test data, it dropped to 0.59, which shows my model was overfitting a bit. I tried playing with the dependents by adding and removing to see if I could get a better result. After tweaking and changing the features for a while with getting a lot of overfitting and underfitting results, I put all of the columns (except BodyFat which is the target) and got the most accurate model with an R² of 0.9968 on training set and 0.75 on test data.

**What I learned:**
This project really makes me understand how much goes into machine learning beyond just the algorithms. I learned about overfitting firsthand when I firstly tested using only four features (Weight, Height, Abdomen, Thigh) and saw how my model performed worse on test data. I tried tweaking the features many times and faced with both overfitting and underfitting when I compared the train and test data. I finally got the best score by putting all of the features, mainly **Density** as there is actually a well-known formula in physiology called the Siri equation that directly converts body density to body fat percentage. Also using GridSearchCV taught me the importance of systematic parameter selection rather than just guessing. Most importantly, I learned that machine learning involves making lots of decisions throughout the process, and each choice affects the final results.
