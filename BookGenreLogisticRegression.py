import pandas as pd
from numpy import asarray
from sklearn import metrics
from sklearn import linear_model
from sklearn.metrics import classification_report
import cProfile

def efficiencyTest():
    datainput = pd.read_csv("book_data.csv", delimiter=";")
    giveWarning = False
    X = datainput[['age','gender','genre']].values


    # Data Preprocessing
    from sklearn import preprocessing

    age_value = ['young', 'middle', 'old']
    label_age = preprocessing.LabelEncoder()
    label_age.fit(age_value)

    for check in X[:, 0] :
        if check not in age_value :
            giveWarning = True

    if giveWarning :
        print("Ada kesalahan di penamaan umur")
        return False
    else:
        X[:, 0] = label_age.transform(X[:, 0])


    genderValue = ['male', 'female']
    label_gender = preprocessing.LabelEncoder()
    label_gender.fit(genderValue)
    for check in X[:, 1] :
        if check not in genderValue :
            giveWarning = True

    if giveWarning :
        print("Ada kesalahan di penamaan gender")
        return False
    else:
        X[:, 1] = label_gender.transform(X[:, 1])


    genreValue = ['Fantasy', 'Adventure', 'Romance', 'Contemporary' , 'Dystopian', 'Mystery',
    'Horror',
    'Thriller',
    'Paranormal',
    'Historical fiction',
    'Science Fiction',
    'Memoir',
    'Cooking',
    'Art',
    'Self-help / Personal',
    'Development',
    'Motivational',
    'Health',
    'History',
    'Travel',
    'Guide / How-to',
    'Families & Relationships',
    'Humor',
    'Children’s']
    label_genre = preprocessing.LabelEncoder()
    label_genre.fit(genreValue)
    for check in X[:, 2] :
        if check not in genreValue :
            giveWarning = True

    if giveWarning :
        print("Ada kesalahan di penamaan genre")
        return False
    else:

        X[:, 2] = label_genre.transform(X[:, 2])

    y = datainput["suggested"].values

    suggestedValue = ['Yes', 'No']
    decision_label = preprocessing.LabelEncoder()
    decision_label.fit(suggestedValue)
    for check in y:
        if check not in suggestedValue:
            giveWarning = True

    if giveWarning:
        print("Ada kesalahan di penamaan decision")
        return False
    else:
        y = decision_label.transform(y)

    # train_test_split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)


    bookLogistic = linear_model.LogisticRegression(solver='liblinear', random_state=0)

    bookLogistic.fit(X_train, y_train)

    predicted = bookLogistic.predict(X_test)

    print(predicted)

    print(classification_report(y_test, predicted))

    print("\nLogisticRegression's Accuracy: ", metrics.accuracy_score(y_test, predicted))
    # precision tp / (tp + fp)
    print("\n LogisticRegression's Precision: ", metrics.precision_score(y_test, predicted, average="macro"))
    # recall: tp / (tp + fn)
    recall = metrics.recall_score(y_test, predicted, average="macro")
    print("\n LogisticRegression's Recall: ", recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = metrics.f1_score(y_test, predicted, average="macro")
    print("\n LogisticRegression's F1: ", f1)

    from sklearn.preprocessing import MinMaxScaler
    scalar = MinMaxScaler()

    normArray = scalar.fit_transform(asarray(X))
    normDataInput = pd.DataFrame(normArray, columns= datainput[['age','gender','genre']].columns)
    print("Model Scalability: \n",normDataInput.head())



cProfile.run('efficiencyTest()')

