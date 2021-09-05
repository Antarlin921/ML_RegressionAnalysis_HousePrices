# Regression Analysis Model for determining price of houses.Maison.csv file used as datafile
# # # Simple Linear Regression
# # Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# # Importing the dataset
# dataset = pd.read_csv('Salary_Data.csv')
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 1].values
# # Splitting the dataset into the Training set and Test set
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 42)
# # Fitting Simple Linear Regression to the Training set
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# # Predicting the Test set results
# y_pred = regressor.predict(X_test)
# print('Coefficients: \n', regressor.coef_)
# # The mean squared error
# print("Mean squared error: %.2f" % np.mean((regressor.predict(X_test) - y_test) ** 2))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % regressor.score(X_test, y_test)).

# ## Import the Libraries and Data

# Let's import our libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

# from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
# import pydotplus

# Since we are going to mae lot of visualization, let's set some visualization parameters in order to have same plots size
plt.rcParams['figure.figsize'] = [12,6]
sns.set_style('darkgrid')

#house = pd.read_excel('/home/utkarsh/LearnBay/ML_Python/Maison.xlsx') ## Reading the data
house = pd.read_csv("Maison.csv") 

house.shape


house.head()


# # Data Transformations & Analysis

# Since the columns are in french, in order to make them more readable, let's translate them into English
house = house.rename(index = str, columns = {'PRIX':'price','SUPERFICIE': 'area','CHAMBRES': 'rooms', 
                         'SDB': 'bathroom', 'ETAGES': 'floors','ALLEE': 'driveway',
                         'SALLEJEU':'game_room', 'CAVE': 'cellar', 
                         'GAZ': 'gas', 'AIR':'air', 'GARAGES': 'garage', 'SITUATION': 'situation'})

house.head()

# Let's see ig we have a linear relation between price and area
#sns.palettes(house['area'], house['price'], palette = 'viridis')
plt.scatter(house['area'], house['price'])
plt.show()

import warnings
warnings.filterwarnings('ignore')
sns.distplot(house['price'])
plt.show()
sns.distplot(house['area'])
plt.show()
house1=house
house1


# # EDAS To be done:0 variance check,Misssing Value check,correlation check,outlier removal

# # Missing Value Check

total = house1.isnull().sum().sort_values(ascending=False)
percent = (house1.isnull().sum()/house1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
# No missing data was found so no rows removed
#dealing with missing data
house1 = house1.drop((missing_data[missing_data['Total'] > 1]).index,axis=1)

house1.isnull().sum().max() #just checking that there's no missing data 

# Note we dont find any missing values so we dont drop any columns or rows

house1.dtypes


# # 0 Variance Check


house1.var()

# We see variance of gas column is very low.So we decide to drop it.

house1.drop('gas',axis=1,inplace=True)

house1


# # Outlier removal


from scipy import stats
import functools

def drop_numerical_outliers(df, z_thresh=3):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = df.select_dtypes(include=[np.number])         .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh)         .all(axis=1)
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constrains], inplace=True)
drop_numerical_outliers(house1, z_thresh=4)

house1


# # Correlation Check


#correlation matrix
corrmat = house1.corr()
print (corrmat)
f, ax = plt.subplots(figsize=(14, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

#saleprice correlation matrix
k = 12 #number of variables for heatmap
cols = corrmat.nlargest(k, 'price')['price'].index
cm = np.corrcoef(house1[cols].values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.3f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# # Data Standardization

from sklearn import *


from sklearn.preprocessing import StandardScaler
import seaborn as sb

house1stan=pd.DataFrame(StandardScaler().fit_transform(house1))

sb.kdeplot(house1stan[0])
sb.kdeplot(house1stan[1])
sb.kdeplot(house1stan[10])

house1stan.columns=house1.columns

house1stan


# # Regression Analysis

# Import the libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# We now instatiate a Linear Regression object
lm = LinearRegression()

# let's do the split of the dataset
house1stan.columns
X = house1stan[['area', 'rooms', 'bathroom', 'floors', 'driveway', 'game_room',
       'cellar', 'air', 'garage', 'situation']]
y = house1stan['price']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.22, random_state=42)

## Let's chec the head of some of these splits
X_test.head()
# We see that they are randomly selected


# Now let's build the model using sklearn
lm.fit(X_test,y_test)


# Now let's look at the coefficients
print(lm.coef_)
# it would be nicer if we can put them together in a dataframe
coef = pd.DataFrame(lm.coef_, X.columns, columns = ['Coefficients'])
coef

predictions = lm.predict(X_test)


# To check the quality of our model, let's plot it
sns.scatterplot(y_test, predictions)


# Evaluation metrics
# Mean Absolute Error (MAE)
# Mean Squared Error (MSE)
# Root Mean Squared Error(RMSE)
import numpy as np
from sklearn import metrics

print('MAE :', metrics.mean_absolute_error(y_test, predictions))
print('MSE :', metrics.mean_squared_error(y_test, predictions))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

MAE : 10248.782807401953
MSE : 188311345.17713058
RMSE : 13722.658094448414



print('Variance score: %.5f' % lm.score(X_test, y_test))

