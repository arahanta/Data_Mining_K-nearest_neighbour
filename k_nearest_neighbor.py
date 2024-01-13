
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('/kaggle/input/water-potability/water_potability.csv')


df.head()

missing_values = df.isnull().sum()
print(missing_values)

df.describe()

#Replaces empety values
def replace_NAN(d):
    missing_values = d.isnull().sum()
    means = d.mean()
    for col in d.columns:
        if missing_values[col] > 0:
            d[col].fillna(means[col], inplace=True)
    return d

means = df.mean()
print(means)

df = replace_NAN(df)

missing_values = df.isnull().sum()
print(missing_values)

df.describe()

#visiualizing the data and class distribution

sns.countplot(data=df, x= 'Potability', hue= 'Potability')
plt.legend(labels= ['Not Drinkable', 'Drinkable'])

cor = df.corr()
m = np.triu(cor)
plt.figure(figsize = (12,12))
sns.heatmap(cor, annot= True, mask =m)

for col in df.columns:
    sns.histplot(data=df, x=col, kde = 'True',  hue= 'Potability')
#     plt.legend(labels= ['Not Drinkable', 'Drinkable'])
    plt.show()



#Most of the attributes follow normal distribution

def train_eval(data):

    # Standardizing the dataset
    data = data.sample(frac=1)
    data_x= data.iloc[:, :-1]
    data_y = data.iloc[:,-1]
    scaler = StandardScaler()
    data_x = pd.DataFrame(scaler.fit_transform(data_x), columns = data_x.columns)

    #test train split
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state=41)

    #model creation
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)

    #evaluation
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions)
    print(report)
    matrix = confusion_matrix(y_test, predictions)
    sns.heatmap(matrix, annot= True)

train_eval(df)

#Poor accuracy, maybe due to imbalanced data

#Testing k-values
data=df
data = data.sample(frac=1)
data_x= data.iloc[:, :-1]
data_y = data.iloc[:,-1]
scaler = StandardScaler()
data_x = pd.DataFrame(scaler.fit_transform(data_x), columns = data_x.columns)

#test train split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state=41)




k_values = list(range(1,51, 2))
accu = []
for k in k_values:
    m = KNeighborsClassifier(n_neighbors =k)
    scores = cross_val_score(m, x_train, y_train, cv=10,  scoring='accuracy')
    accu.append(scores.mean())

# Plotting accuracy scores versus k values
plt.figure(figsize =(15,8))
plt.plot(k_values, accu, marker='o')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.title('Accuracy Scores for Different k values')
plt.grid(True)
plt.show()

#Try dropping sulphata colmn
df = df.drop(columns = 'Sulfate', axis =1 )
df.head(5)

train_eval(df)

#Lets tackel class imbalance problem, by downsampaling not drinkable instances(np)

p = data[data['Potability']==1]
not_p = data[data['Potability']==0]
sampled_data = not_p.sample(n=1300)

#Shuffle the sampled data
data_new = pd.concat([p, sampled_data], axis =0)
#shuffle
s_data = data_new.sample(frac=1)
s_data.head()

sns.countplot(data=s_data, x= 'Potability', hue= 'Potability')
# plt.legend(labels= ['Not Drinkable', 'Drinkable'])



#visiualizing the data and class distribution

#lets train model
#replacing NAN
# s_data = replace_NAN(s_data)
train_eval(s_data)


# Lets try dropping all the rows with missing values
data = pd.read_csv('/kaggle/input/water-potability/water_potability.csv')
data.dropna(inplace =True)

sns.countplot(data=data, x = 'Potability', hue = 'Potability' )
print(data.shape)

train_eval(data)

data = data.sample(frac=1)
data_x= data.iloc[:, :-1]
data_y = data.iloc[:,-1]
scaler = StandardScaler()
data_x = pd.DataFrame(scaler.fit_transform(data_x), columns = data_x.columns)

#test train split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state=41)




k_values = list(range(1,51, 2))
accu = []
for k in k_values:
    m = KNeighborsClassifier(n_neighbors =k)
    scores = cross_val_score(m, x_train, y_train, cv=10,  scoring='accuracy')
    accu.append(scores.mean())

# Plotting accuracy scores versus k values
plt.figure(figsize =(15,8))
plt.plot(k_values, accu, marker='o')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.title('Accuracy Scores for Different k values')
plt.grid(True)
plt.show()

dataset = prep.skewcorrect(dataset,except_columns=['Potability'])

#balancing the class
data[data['Potability']==1].shape

#Lets tackel class imbalance problem, by downsampaling not drinkable instances(np)

p = data[data['Potability']==1]
not_p = data[data['Potability']==0]
sampled_data = not_p.sample(n=811)

#Shuffle the sampled data
data_new = pd.concat([p, sampled_data], axis =0)
#shuffle
s_data = data_new.sample(frac=1)
s_data.head()

sns.countplot(data=s_data, x= 'Potability', hue= 'Potability')
# plt.legend(labels= ['Not Drinkable', 'Drinkable'])

train_eval(s_data)

data = s_data
data_x= data.iloc[:, :-1]
data_y = data.iloc[:,-1]
scaler = StandardScaler()
data_x = pd.DataFrame(scaler.fit_transform(data_x), columns = data_x.columns)

#test train split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state=41)




k_values = list(range(1,51, 2))
accu = []
for k in k_values:
    m = KNeighborsClassifier(n_neighbors =k)
    scores = cross_val_score(m, x_train, y_train, cv=10,  scoring='accuracy')
    accu.append(scores.mean())

# Plotting accuracy scores versus k values
plt.figure(figsize =(15,8))
plt.plot(k_values, accu, marker='o')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.title('Accuracy Scores for Different k values')
plt.grid(True)
plt.show()

