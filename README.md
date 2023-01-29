# Hotel-Booking-Cancellation

Model optimization to predict availabity with and without highly predictive parameter.


# Perfomed EDA - Refer code HBC_WO_Parameter

Treating with Duplicates, Nulls, & Ouliers 

Observed that hotel is getting booked travel agents, indivisually, by hotel, & by company 

We considered z score/ Z transformation to call a data point outlier. It is an outlier if it lies beyond 3 sigma

Feature selection, One hot encoding (Dummy Variables)

 # Observations

Hotel: Cancellation % doesn't change much with the hotel type.

Month: April, May, June, July, august have more than 30% booking of cancellation

Meal: BB, FB, HB constitute to majority of meals and there is very less change in
%cancellation in these type of meals

172 countries we can bucket this column according to continent or ignore the column. For now i will ignore it

market_segement: There is significcant difference in % with market segment
distribution_channel: here also % changes with distribution channel

reserved room type: We might want to bucket it and keep top 4 and classify remainings as 'other'. % is not chaging that much with reserved room type.

assigned room type: Again we might want to bucket it but % is not changing much
deposit type: even though almost all the records are for no deposit, non refundable records are being cancelled 95% of the time which is very big number

customer type: % is changing

booked_by: % is changing considerably

is_repeated guest: if guest is repeated then he/she cancel the booking only for 9% of the times

reservation_status: This variable has very high predictive power. Whenever status is check-out, then target variable is alwasy 0 and if status is not check-out then target will be always 1
Conclusion

Categorical variables to consider: 'Marcket_segment','deposit_type','customer_type','booked_by','is_repeated_guest'

Tasks

Create a column is_room_changed which specifies if assigned room is not same as reserved room

Reduce the unique values in country, reserved room type and assigned room type column by

keeping the top few recordsand merging the remaining records in 'other' category

Convert the text month column into month number

# Logistic Regression

Summary

Fitting the model using Logistic Regression Model on splitted Train & Test Data using 500 iterations.

Using 500 iterations because there are more chances of converging.

Now applying Cross Validation on train data with CV = 5

Predicting with Test Data

Finding out the number of important features (Feature is considered important if it's beta value is greter than 0.01)

Calculating performance metrics along with time required to compute the model processing.
Performance Metrics calculated -
Accuracy
Recall
Precision
False Negatives
False Positives

Determining the Probility Threshold for best Accuracy & Recall

# Probability Threshold Determination 

For selecting probability classification threshold we need to understand what TP, TN, FP, FN explains in business terms

-TP: Booking gets cancelled and it is predicted as cancelled

-TN: Booking is not cancelled and is predicted that booking is not canclled

-FP: Booking is not cancelled but it is predicted that is cancelled

-FN: Booking is cancelled and it is predicted that it is not cancelled

We can say that FN is a very costly parameter here because cancelled booking is getting predicted as not cancelled. Which means Even if the Hotel has vacancy it would be showing as Fully booked. This is bad as if there is any newbooking. The hotel might not consider this as it would be shown as fully booked.

Considering above scenario, we should try to decrease the False Negatives, as we decrease FN, we would be able to predict correctly if there are more vacancies or not.

If we are predicting correct vacancy we can increase the revenue, if not increase , we can atleast save the revenue loss for those rooms which are vacant and are predicted as not cancelled.

Since our focus id to decrease the False Negatives, we should choose the threshold in such a manner that we are dachieveing max Recall with maximum Accuracy.

By looking into the above table, we can see our model is working perfectly with threshold of 0.50, with an Accuracy of 1 and Recall of 1. I think we should consider threshold in such a manner that both Accuracy and Recall are Maximum.

# Decision Tree using GridSearchCV

Summary

Defining hyper parameters

Tested combination on different hyperparameters;

'max_depth': [5,10, 15, 20,25,30],

'min_samples_split': [2,3,4,5,10,15,20,25],

'min_impurity_decrease': [0, 0.001, 0.0001]

Apllied GridSearch Cv to find the best parameters, used cross validation with cv = 5.

Used Accuracy , Recall & Precision for scoring the matrix, & accuracy to find the best model.

We found out - -Max Depth : 5 -Minimum Sample : 5 -Impurity Decrease : 0

From feature impotance we found that reservation_status_Check-Out is the best feature.

Now predicting using the best estimator

Calculating performance metrics along with computing time. -Accuracy

Precision
Recall
False Negatives
False Positives
Minimum Samples Split
Max Depth

Here I am running Decision tree with Grid SearhCV, to predict the Booking is cancelled or not, along with different parameters perfoming hyper parameter tunning.

Now, Decision Tree is running 12 models with a cross validation of 5 which means total 60 models are cross checked to consider the best parametrer.

Best Model states that Max Depth should be 5 and Minimum Sample should be 5, to predict correctly.

Above is the best feature that is selected for the Decision Tree. From the Feature importance plot we can say that reservation_status_Check-Out is the best feature to predict whether the booking is canclled or not.

# Random Forest 

Max Depth :  3
Minimum Sample :  5
Impurity Decrease :  0
Criteria for Chosing the splitting :  gini
Here we are, performing Random Forest to predict the bookings using hyper parameter tunning to run 24 models with cross validation of 5, which means we are validating 120 models to consider the best results.

Here as we have to reduce the complexity of the model, I think the cost complexity for running the Random Forest would be very high compared to decision tree.

So we need to consider between Decision Tree and Random forest which is perfomring better.

From the above results we can see that best result for Max Depth is 3, MInimum Sample is 5, and criteria for choosing the splitting is Gini

# Artificial Neural Netwrok 
Neural Network is build using 2 hidden layers, Size = 25, 20, activation fucntion is relu.

# Comparison of all the models

If we look at the above table we can see that all the models are perfoming Good with an Accuracy of 1. But our goal is to have zero False Negatives. So we cannot consider Neural Networks as we are getting 1 False Negative.

Now we are left with three models, Logistic regression, Decsion Tree and Random Forest. We need to ceck whih model takes least time to process.

Processing Time -

Decisoin Tree - 1.31 Sec Random forest - 160.98 Logistic Regression - 0.66

From above it seems that Logistic Regression seems to be best.

But if we consider the GridSearch Cross Validation for Decision Tree, we can see in 1.31 seconds 12 models and 60 cross validations are taking place which is much higher compared to time taken to run one Logistic Regression model whih takes 0.66 secs.

Both the models are giiving similar results but Logistic Regression has a dependency of classification threshold but Decisoin Tree doesn't, both the models have their pros and cons. I think Decision Tree, is performing better.

Comparing Decison Tree to Random Forest we can that both the Accuracy & Recall is 1 but we need to consider such a model for which the processing & computing power is less. Here we should select Decision tree as it is performing equally better with same scores and less computing complexity. Comapred to Dt the RF has much much higher computing time, though validating 12 extra models.

For me comparing all models, I can say Decision Tree is performing better.

# Best Two Models

If we see the df_results we can say that, Decision Tree is using only one feature reservation_status_Check-Out, to give an accuracy of 1 and Recall of 1. With a Run time of 1.31 secs and using only 1 feature Decision Tree is the best. But we need to decide two best models so from the above discussions we can say that Logistic Regression is second best model as the computinhg time is less with best Accuracy and recall.

Two best models -

Decision Tree
Logistic Regression
From the above observations we can see that reservation_status_Check-Out is biasing the results, so we will try to train the model without using them as they are very close to Y variable "is_cancelled."

As discussed above, we need a Logistic Regresion Model where our Recall is max along with very high Accuracy.

From the above table we can see that for threshold between 0.3 & 0.35 we can achive high values for both of them.

Ideal probability threshold we can take 0.33, where we can achive a good accuracy of 76% with a pretty good Recall of 75%.

# Refitting Accuracy - decision tree

First we tried to fit the Decision Tree, by refitting the model with the best accuracy.

From the model we can see that the computing time is 10.36 secs in which we achived an accuracy of 82%, and with Recall of 66%. Running & cross validating 45 models.

We can see that the accuracy is pretty good.

# Refitting Recall

Above tried to Refit the model with increasing the Recall of the model, with max depth of 15, Minimum sample 2 of impurity decrese of 0. But the Computing time took more, than fitting the Accuracy.

While trying to increase the recall, we can see there is not much difference in the Accuracy.

Same Accuracy was achieved and hardly any increase in Recall was observed.

While fitting the Accuracy we found the Accuracy to be 82% also while fitting the Recall we found that Accuracy is 82%.

Found that With Decison Tree, we can achieve better accuracy but only an adequate recall.

But with Logistic Regression (Threshold = 0.33) we can achive a better Recall with a pretty good Accuracy.

# Recommendation for Hotel Mananagement

Employer can use this model to predict the booking cancellations, but we have a dominating feature (reservation_status_Check-Out) which is giving us the best accuracy and recall.
We can ask employer more information which can be converted into new features and helps us in better prediction.
If we drop the above mention feature most of the features available dont have correlation with booking cancelation. Model might generate considerable false negatives which will result in rooms being empty. This problem will be solved if hotel has high walk in customers
Employer can provide information about walk in guest booking. If number of walk in customers is considerable then employer can choose model which is performing better in accuracy but not that good for recall as walk in guests will can always fill the rooms that get empty last minute
Employer can collect feedback from guests who cancel the booking to understand the reason. From those reasons employer migh get more insights regarding what type of data might be useful in the prediction
Employer can stop collecting the data which is not required to reduce the cost
