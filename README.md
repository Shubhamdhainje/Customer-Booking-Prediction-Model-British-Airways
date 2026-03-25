### British Airways — Customer Booking Prediction
_____________________________________________________________________________________
### Overview
Customers today research and book holidays long before arriving at an airport — 
which means airlines must be proactive, not reactive. This project builds a machine 
learning pipeline to predict whether a customer will complete a holiday booking, 
enabling British Airways to identify and target high-intent customers early in the
buying journey using data-driven intelligence.
_____________________________________________________________________________________
### Objectives

•	Explore and prepare a dataset of 50,000 customer booking records

•	Engineer meaningful new features from raw behavioural data

•	Train a Random Forest classifier to predict booking completion

•	Evaluate model performance rigorously using 5-fold cross-validation

•	Interpret feature importance to understand what drives customer bookings
_____________________________________________________________________________________
### Machine Learning Algorithm: Random Forest Classifier
A Random Forest is an ensemble method that trains hundreds of decision trees on
random subsets of the data and aggregates their predictions via majority vote. 
It was selected for this task because it:

•	Naturally handles mixed numeric and categorical features without scaling

•	Provides reliable feature importance scores out of the box

•	Is robust to class imbalance via class_weight='balanced'

•	Generalises well with low risk of overfitting through bagging and feature randomness

Key configuration: n_estimators=200, max_depth=10, class_weight='balanced'
_____________________________________________________________________________________
### How a Single Decision Tree Works
A Decision Tree splits data at each node based on a feature threshold that best 
separates the two classes (booked vs. not booked). It asks questions like:

Is purchase_lead < 30 days?

├── YES → Is booking_origin == "Australia"?

│         ├── YES → Likely BOOKED ✅

│         └── NO  → Likely NOT BOOKED ❌

└── NO  → Is length_of_stay > 7 nights?

          ├── YES → Likely BOOKED ✅
          
          └── NO  → Likely NOT BOOKED ❌
          
Each split is chosen to maximise information gain (how much the split reduces 
uncertainty about the target). The tree keeps splitting until it reaches a maximum
depth or a stopping condition.
_____________________________________________________________________________________
### Hyperparameters Used & Why

Hyperparameter	Value	Reason

n_estimators	200	More trees = more stable predictions; 200 gives strong performance 
without excessive compute time

max_depth	10	Limits tree depth to prevent overfitting — trees can't memorise noise
past 10 levels

class_weight	'balanced'	Dataset is heavily imbalanced (85% not booked). This 
weight automatically adjusts so the minority class (booked) gets proportionally more 
influence during training

random_state	42	Ensures reproducibility — same results every run
n_jobs	-1	Uses all available CPU cores to train trees in parallel, speeding up 
training significantly
______________________________________________________________________________________
### Why Random Forest Was Chosen for This Task
Reason	Detail

Handles mixed data types	Works natively with numeric (e.g. purchase_lead) and 
label-encoded categorical features (e.g. route, booking_origin) without needing 
scaling or normalisation

Built-in feature importance	Each tree tracks how much each feature reduces impurity 
across all splits — aggregated across 200 trees, this gives a robust, reliable 
ranking of what matters most

Robust to outliers	Outliers in purchase_lead (up to 867 days) don't distort results 
because tree splits are threshold-based, not distance-based

Handles class imbalance	The class_weight='balanced' parameter penalises the model more 
for missing a booking (minority class), making it more sensitive to the class we care 
about most

No need for feature scaling	Unlike Logistic Regression or SVM, Random Forest doesn't 
require normalisation — decision boundaries are not affected by the scale of features

Low risk of overfitting	The combination of bagging + feature randomness + max_depth 
cap makes the ensemble generalise well to unseen data
_______________________________________________________________________________________
### Feature Engineering
Six new features were derived from the raw data to improve model signal — including
total_extras (sum of all add-on selections), booking_urgency (bucketed purchase lead
time), and binary flags for trip type, sales channel, and weekend departures.
_______________________________________________________________________________________
### Results — 5-Fold Cross-Validation

Metric	Score

ROC-AUC	0.767

Accuracy	71.3%

Recall	69.1%

F1 Score	0.419

•	booking_origin was the dominant predictor at 38.8% importance, followed by route
(13.5%) and length_of_stay (10.8%)

•	High recall ensures ~7 in 10 actual bookers are correctly identified — critical 
for proactive campaign targeting
_______________________________________________________________________________________
### Feature Importance (Top 5)

Rank	Feature	Importance	Insight

1	booking_origin	38.8%	Country of booking is the dominant signal — geography predicts 
behaviour

2	route	13.5%	Specific origin–destination pairs carry strong predictive information

3	length_of_stay	10.8%	Longer planned stays correlate with higher booking completion

4	flight_duration	9.9%	Long-haul vs. short-haul trips show distinct booking patterns

5	purchase_lead	6.9%	How far in advance a customer searches is a key behavioural signal
_______________________________________________________________________________________
### Business Impact
A Recall of 69% means British Airways can correctly identify 7 in 10 customers who will 
complete a booking — enabling personalised, proactive outreach campaigns before those 
customers book with a competitor.
_______________________________________________________________________________________
### Tools & Technologies
Python 3.8+ 

Scikit-learn 

Pandas 

NumPy 

Matplotlib 

Random Forest 

Stratified K-Fold Cross-Validation
_______________________________________________________________________________________

