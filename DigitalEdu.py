import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

#Handling
df = pd.read_csv('train.csv')
df.drop(['id', 'bdate', 'has_photo', 'followers_count', 'graduation', 'education_form', 'relation', 'education_status', 'langs', 'life_main', 'people_main', 'city', 'last_seen', 'occupation_name', 'career_start', 'career_end', 'occupation_type'], axis = 1, inplace = True)

#Counting
male = 0
female = 0

def sex_count(row):
    global male, female
    if (row['sex'] == 1) and (row['result'] == 1):
        male += 1
    elif (row['sex'] == 2) and (row['result'] == 1):
        female += 1

df.apply(sex_count, axis = 1)

mobiley = 0
mobilen = 0

def mobile_count(row):
    global mobiley, mobilen
    if (row['has_mobile'] == 1) and (row['result'] == 1):
        mobiley += 1
    elif (row['has_mobile'] == 0) and (row['result'] == 1):
        mobilen += 1

df.apply(mobile_count, axis = 1)

#Output
print('CHART INFO 1')
print('Male:', male)
print('Female:', female)
print('CHART INFO 2')
print('Mobile included:', mobiley)
print('Moblie not included:', mobilen)

#Chart
s = pd.Series(data = [male, female], index = ['Male', 'Female'])
s.plot(kind = 'pie')
plt.title('Gender')
plt.xlabel(None)
plt.ylabel(None)
plt.show()

m = pd.Series(data = [mobiley, mobilen], index = ['Yes', 'No'])
m.plot(kind = 'pie')
plt.title('Mobile Inclusion')
plt.xlabel(None)
plt.ylabel(None)
plt.show()

#Sklearn
X = df.drop('result', axis = 1)
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
 
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)
 
y_pred = classifier.predict(X_test)
print('Coorect results:', round(accuracy_score(y_test, y_pred) * 100, 2), '%')
