==Overview==

The codes/packages in this repository is currently used to load the data, 'Adult' from UCI repository and perform the descriptive and exploratory data analysis. 
It manipulates the data and transforms some variables into more meaningful values ML likes.
I plan to use the same codes to perform the analysis and run various ML models with different datasets.

==What has done==
* Data Collection: Loads adult dataset directly from UCI repository. Total 32,561 instances loaded into Pandas dataframe.
  Since dataset is missing with a header, columns/features are named with data types. Definition of data types and names
  are in <tt>constants.py</tt>

* Data Cleaning: Dataset has no null variables. However, it has '?' in 3 different features (occupation, workclass, and native country).
  Removed instances containing '?'. After removal, total instances became 30,162. 
  Removed 23 duplicates. Total came down to 30,139 instances. 
  Dataset has a leading space in every values in all columns. Removed all of them.
  These are done in <tt>manipulate_data.py</tt> using pandas lib.
                 
* Analysis/Visualization: Visualization like relationship with a label, 'income' and various features done via 
  <tt>analyze_data.py</tt> and <tt>utils.py</tt> using matplotlib and seaborn libs. 
  Descriptive Analysis done via <tt>analyze_data_data.py</tt> using pandas lib. 
                          
* Feature Engineering/Tranformation: Categorical features like occupation (ex: Sales, Farming-fishing) were converted into
  indicator values via <tt>manipulate_data.py</tt> using pandas lib:  
  		         			      
* Run ML Models: Decision Tree, SVM, Logistic Regression, and XGBoost were performed via <tt>models.py</tt> using scikit-learn and xgboost libs.
  XGBoost performance is best
