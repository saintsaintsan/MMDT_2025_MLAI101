
# Project Report: Predicting Life Expectancy Using Regression Models

## Problem Description

In this project, the main objective was to predict **life expectancy** using various socioeconomic and health-related indicators. Life expectancy is a crucial measure used by governments, public health organizations, and researchers to understand the overall health and development status of a country. It encapsulates a range of factors such as healthcare infrastructure, income levels, education, and environmental conditions.

Although I initially did not have a specific reason for choosing this topic, I realized it presents an ideal problem for regression modeling. The relationship between life expectancy and its influencing factors is both complex and continuous, making it well-suited for models like linear and polynomial regression. Additionally, predicting life expectancy can have real-world impact, informing policies and resource allocation to improve public health outcomes.

## Dataset

The dataset was obtained from Kaggle, titled **“Life Expectancy (WHO)”**, which includes data from multiple countries and years. The dataset contains over 20 numerical features ranging from immunization coverage and GDP to adult mortality and BMI.

For this project, I decided to drop three columns: **country**, **year**, and **status**. The decision to drop `country` and `year` was based on their nature as identifiers rather than predictive numerical inputs; including them could introduce unnecessary noise or bias into the model. As for `status`, which indicates whether a country is developing or developed, I excluded it to focus purely on quantitative indicators and to keep the modeling process straightforward without introducing categorical encoding in this iteration.

## Modeling Approach

To model life expectancy, I used two regression techniques: **Multiple Linear Regression** and **Polynomial Regression**.

The data preprocessing steps included:

- Cleaning the dataset by checking and correcting data types.
- Filling missing values using the **mean** of the respective columns to preserve data integrity.
- Dropping the aforementioned non-numeric or less relevant columns.
- **Standardizing** the numerical features using `StandardScaler` to ensure all variables had equal weight and scale during training.
- Splitting the dataset into **80% training** and **20% testing** sets to validate model performance fairly.

I first trained a multiple linear regression model to establish a baseline. Then, I used polynomial regression with degree **2**. Initially, I considered trying higher degrees (up to 5), but the degree-2 model already showed a significantly better fit compared to linear regression, and increasing the degree further risked **overfitting**.

## Evaluation

Model performance was assessed using several metrics, including **R² score**, **Mean Absolute Error (MAE)**, and **Mean Squared Error (MSE)**. The linear regression model achieved an **R² score of 0.81**, while the polynomial regression model performed better with an **R² score of 0.89**. This improvement suggests that the relationship between life expectancy and its predictors is nonlinear, and capturing the interaction between features helped improve the model’s predictive power.

## Reflection

Initially, I used a much smaller dataset (with only around 60 rows) for regression, and surprisingly obtained perfect R² and near-zero error scores. After consulting with my mentor, I learned that this was likely due to **overfitting caused by insufficient data**. This was a key learning point that led me to switch to the life expectancy dataset, which had a more substantial size and structure.

Additionally, I learned the importance of evaluating model complexity. Although polynomial regression can capture more complex patterns, higher degrees can easily lead to overfitting. In this case, using a polynomial degree of 2 struck a good balance between accuracy and generalization.
