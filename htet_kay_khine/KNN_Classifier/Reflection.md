a) What is fraud detection, and why is it important?

Fraud detection is the process of identifying and preventing fraudulent activities. It is important for minimizing financial losses, protecting assets and data, maintaining customer trust, ensuring regulatory compliance, and preserving company reputation.

b) If you change the training and testing split to 70% training and 30% testing, how does the model's performance change?

After changing the train-test split from 60%–40% to 70%–30%, the model’s overall performance remained almost the same. The AUC score slightly improved from 0.9186 to 0.9287, indicating slightly better prediction quality. Accuracy and precision remained very high, but recall on the test set slightly decreased from 0.80 to 0.79. This suggests the model benefits from more training data but still struggles slightly with identifying all fraud cases.

c) Keeping the test size fixed at 40%, try changing the number of neighbors (in KNN). How does the model’s performance vary with different K values?
Which value gives the best result, and how do you define what makes it the "best"?

I tested different values of k(number of neighbors) in KNN while keeping the test size at 40%. The model performed best at k = 5, achieving a high F1-score (0.8755) and strong AUC (0.9186). This value gave the best balance between precision and recall for detecting fraud. Therefore, we selected k = 5 as the optimal choice based on its performance in identifying fraudulent cases.
