# ## Step 1 : Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ## Step 2 : Load Dataset
dataset = pd.read_csv("dataset.csv")
dataset.head()


# ## Step 3 : Get General Information of Dataset
# Information on type of object, column names, data types of each column, memory usage
# non-null counts
dataset.info()


# ## Step 4 : Check Data Type and Change If Required
# Change object datatype into category if number 
# of categories are less than 5 percent of the total number of values
cols = dataset.select_dtypes(include='object').columns
for col in cols:
    ratio = len(dataset[col].value_counts()) / len(dataset)
    if ratio < 0.05:
        dataset.loc[:, col] = dataset.loc[:, col].astype('category')

dataset.info()


# ## Step 5 : Missing Data Management
# Percentage of Missing values in each column
nan_percentage = [(clm_names, dataset[clm_names].isna().mean() * 100) for clm_names in dataset]
nan_percentage = pd.DataFrame(nan_percentage, columns = ['columns', 'nan_percentages'])
nan_percentage

# Set threshold on missing values and if any column crosses that threshold 
# will be removed from dataset
threshold = len(dataset) * 0.7
dataset = dataset.dropna(axis = 1, thresh = threshold)

nan_percentage = [(clm_names, dataset[clm_names].isna().mean() * 100) for clm_names in dataset]
nan_percentage = pd.DataFrame(nan_percentage, columns = ['columns', 'nan_percentages'])
nan_percentage


# Handle missing value 
import warnings
warnings.filterwarnings('ignore')

dataset.Age = dataset.Age.replace(np.NaN, dataset.Age.mode()[0])
dataset.Salary  = dataset.Salary.replace(np.NaN, dataset.Salary.mode()[0])

nan_percentage = [(clm_names, dataset[clm_names].isna().mean() * 100) for clm_names in dataset]
nan_percentage = pd.DataFrame(nan_percentage, columns = ['columns', 'nan_percentages'])
nan_percentage


# ## Step 6 : Check Duplicate Data and Take Action
# Count of duplicate rows
dataset.duplicated().sum()

# Drop duplicate rows
dataset = dataset.drop_duplicates()


# ## Step 7 : Check Outliers and Take Action
# Using BoxPlot
dataset.plot(kind = 'box',
            subplots = True,
            layout = (7, 2),
            figsize = (15,20))
plt.show()


# Check quantile range for your outliers
low = np.quantile(dataset.Salary, 0)
high = np.quantile(dataset.Salary, 0.95)
dataset[dataset.Salary.between(low, high)]

# Check quantile range for your outliers
low = np.quantile(dataset.Age, 0)
high = np.quantile(dataset.Age, 0.9)
dataset[dataset.Age.between(low, high)]


# Make final changes in outliers of dataset
low = np.quantile(dataset.Salary, 0)
high = np.quantile(dataset.Salary, 0.95)
dataset = dataset[dataset.Salary.between(low, high)]

low = np.quantile(dataset.Age, 0)
high = np.quantile(dataset.Age, 0.9)
dataset = dataset[dataset.Age.between(low, high)]

dataset.plot(kind = 'box',
            subplots = True,
            layout = (7, 2),
            figsize = (15,20))
plt.show()


# ## Step 8 : Matrix Of Features
# Independent and Dependent Features/Variable
X = dataset.loc[:, ['Country', 'Age', 'Salary']]
Y = dataset.iloc[:,-1:]


# ## Step 9 : Check Imbalanced Dataset and Take Action
# Using Count Plot
sns.countplot(dataset.Country)
plt.show()

# Check each class weight
from sklearn.utils import compute_class_weight

class_weight = compute_class_weight('balanced', 
                    classes = dataset['Country'].unique() , 
                    y = dataset['Country'])
print("Ratio : {}".format(class_weight))


# ## Step 10 : Categorical Data Management
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

clm_x = ColumnTransformer([("Combine",
                            OneHotEncoder(),[0])], 
                            remainder="passthrough")
X = clm_x.fit_transform(X)

labelencoderY = LabelEncoder()
Y = labelencoderY.fit_transform(Y.values.ravel())


# ## Step 11 : Splitting Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                test_size = 0.2,
                                random_state = 0)


# ## Step 12 : Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)