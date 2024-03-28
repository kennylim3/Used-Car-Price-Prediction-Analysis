# Machine Learning Project Report - Used Cars Price Prediction

## Project Domain
The automotive industry continues to evolve, and in recent years, the used car market has become increasingly important. Many consumers choose to buy used cars due to economic factors and better investment value. 
Currently, if consumers want to sell their cars, they have to take their cars to their respective company workshops or make appointments to get price estimates. This process involves a lot of time and resources. 
With machine learning technology, we can create a model to predict the prices of used cars based on certain features. A machine learning model that can predict the prices of used cars will undoubtedly have several 
benefits, such as decision-making efficiency, improved buyer experience, and sales optimization.

**Reference**: Gajera, P., Gondaliya, A., & Kavathiya, J. (2021). Old car price prediction with machine learning. Int. Res. J. Mod. Eng. Technol. Sci, 3, 284-290.

## Business Understanding
### Problem Statements
- How can we improve the accuracy of predicting used car prices?
- Is it possible to provide clearer and more accurate pricing guidance to buyers and sellers of used cars?
- How can we enhance transaction efficiency and customer satisfaction in the used car market?

### Goals
- Enhance the accuracy of predicting used car prices using machine learning models.

With a machine learning model that can provide more accurate price predictions, it is expected to provide better understanding of the value of used cars, reduce uncertainty, and increase customer trust.
- Provide clearer and more accurate pricing guidance to buyers and sellers.

The machine learning model is expected to provide clearer and measurable pricing guidance to buyers and sellers, helping them make more informed decisions.
- Improve transaction efficiency and customer satisfaction in the used car market.

By using machine learning models to simplify and expedite the transaction process, it is expected to improve efficiency and provide a better customer experience in the used car market.

### Solution Statements
- Utilizing several algorithms, including K-Nearest Neighbor, Random Forest, and Adaptive Boosting.
- Comparing models using the MSE (Mean Squared Error) metric to select the best-performing model.

## Data Understanding
The data used in this project is the Used Car Price Prediction dataset obtained from Kaggle. The dataset contains 6019 records of used cars.

Source: https://www.kaggle.com/datasets/colearninglounge/used-cars-price-prediction

### Variables in the Used Car Price Prediction dataset are as follows:
- Name: car name
- Location: Location where the car is sold or registered.
- Year: year of manufacture of the car
- Kilometers_Driven: number of kilometers driven by the car
- Fuel_Type: type of fuel used (Diesel, Petrol, CNG, LPG, Electric)
- Transmission: type of car transmission (Manual, Automatic)
- Owner_Type: Car ownership type (First, Second, Third, Fourth & Above)
- Mileage: fuel consumption in kilometers per liter
- Engine: engine capacity in CC
- Power: engine power in bhp
- Seats: number of seats
- New_Price: price of the car in new condition
- Price: car price

### Exploratory Data Analysis
Through visualization techniques and EDA, the following insights were obtained.
- Location has an influence on car prices, where Coimbatore has the highest average price, while Jaipur and Kolkata have the lowest average prices.
- The average price of used cars tends to decrease with increasing age.
- Cars with diesel fuel have the highest average price, followed by Petrol, CNG, and LPG.
- Automatic transmission cars tend to be more expensive than manual transmission cars.
- The more ownership transfers a car has, the lower its price.
- Kilometers driven and number of seats do not significantly affect car prices.
- Engine and power are positively correlated with car prices.
- Mileage is negatively correlated with car prices.

## Data Preparation
Some steps taken in the preparation phase are as follows.
- Encoding Categorical Features

The process of encoding categorical features is done using the one-hot encoding technique using OneHotEncoder. This process is carried out to convert categorical features into numeric form. 
Encoding helps to address the problem of representing categorical features in a form understandable by the model.
- Dimensionality Reduction with PCA
  
The PCA (Principal Component Analysis) process is carried out to reduce the dimensionality of features. This will speed up the model training time by reducing the number of features,
addressing multicollinearity, and improving overfitting issues in the model.
- Train Test Split
  
This process is done by dividing the dataset into training data and test data using train_test_split from scikit-learn. Considering the size of the dataset in this project,
train-test split is done with a ratio of 80:20.
- Standardization
  
The standardization process aligns the scale of features by subtracting the mean and dividing by the standard deviation. Standardization ensures that all features have a similar scale,
so no feature dominates the others. The process is done using StandardScaler.

## Modeling
For model creation, several algorithms used are K-Nearest Neighbor (with 10 neighbors), Random Forest (with n_estimators = 50, max_depth = 16, random_state=55, n_jobs=-1), and Adaptive Boosting (with learning_rate=0.05, random_state=55).

**K-Nearest Neighbor**

Advantages of the K-Nearest Neighbor algorithm include:
- Simple and Intuitive: The concept is easy to understand and implement.
- Non-Parametric: KNN does not make assumptions about the data distribution, making it able to handle complex and unstructured data.
- Suitable for Multiclass Data: KNN can be used for multiclass classification problems without needing special adjustments.

Disadvantages of the K-Nearest Neighbor algorithm include:
- Expensive Computation: The decision-making process requires calculating the distance to each data point, which can be expensive for large datasets.
- Sensitivity to Outliers: Outliers can have a significant impact on prediction results.
- Must Store the Entire Dataset: The model must store the entire training dataset, which can consume a lot of memory for large datasets.

**Random Forest**

Advantages of the Random Forest algorithm include:
- Robustness against Overfitting: Random Forest has a good ability to deal with overfitting because it builds many diverse trees.
- Can Handle Imbalanced Data: Random Forest can provide good results on imbalanced datasets.
- Feature Importance: Provides information about the importance of each feature in making predictions.

Disadvantages of the Random Forest algorithm include:
- Difficult to Interpret: Random Forest is more difficult to interpret than simple linear models.
- Expensive Computation: Involves training a large number of trees, which can take time and computational resources.
- Not Suitable for Drifting Data: Random Forest may face issues when applied to dynamically changing data.

**Adaptive Boosting**

Advantages of the Adaptive Boosting algorithm include:
- Robustness against Overfitting: Like Random Forest, AdaBoost tends to be robust against overfitting.
- Can Handle Imbalanced Data: Suitable for handling classification problems with minority classes.
- Utilizes Weak Models: Can utilize weak models (e.g., decision stump) and improve their performance.

Disadvantages of the Adaptive Boosting algorithm include:
- Sensitivity to Noise: Vulnerable to noise and outliers in the data.
- Hyperparameter Sensitivity: Sensitive to suboptimal hyperparameter configurations.
- Expensive Computation: Although faster than some complex algorithms, AdaBoost still requires significant training time.

**The best model selected as the solution is the random forest model. This is because the random forest model produces the lowest MSE (Mean Squared Error) among the three models. 
When tested, the random forest model often provides results that are closer compared to the other two models.**

## Evaluation
**Evaluation Metric Used: Mean Squared Error (MSE)**

MSE is the evaluation metric used to measure how far the predicted values by the model are from the actual values. In the context of predicting used car prices, 
MSE measures the average of the squares of the differences between the predicted price and the actual price.

The formula for MSE is as follows.

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2$$

Where:
- n is the number of samples in the dataset.
- $Y_i$ is the actual price of the used car.
- $\hat{Y}_i$ is the price of the used car predicted by the model.

How it works:
- MSE calculates the average of the squared differences between the predicted values and the actual values.
- The smaller the MSE, the closer the predictions are to the actual values.
- MSE penalizes larger differences more, making it suitable for price prediction cases.

The results of the three models created show promising results with low MSE, where the KNN model has a training MSE of 0.028183 and a test MSE of 0.042867, 
the random forest model has a training MSE of 0.004026 and a test MSE of 0.027424, and the AdaBoost model has a training MSE of 0.045626 and a test MSE of 0.05815. 
All three models show good results and can be considered a good fit.
