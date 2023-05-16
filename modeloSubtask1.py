import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV

# Lectura de los archivos de datos
with open("corpus_procesado.csv", "r") as f:
    corpus = f.read()
df = pd.DataFrame(corpus.splitlines())

with open("development_data_davincis23_V2_subtask2/train_labels_subtask_1.csv", "r") as f:
    labels = f.read()
df_target = pd.DataFrame(labels.splitlines()) 

# Representación vectorial por frecuencia
vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b|\.')
X = vectorizer.fit_transform(df[0])
y = df_target[0]

print('Representación vectorial por frecuencia')
print (vectorizer.get_feature_names_out())
print (X.toarray())
print('\n')

# Partición de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

''' GridSearch para encontrar los mejores parámetros de cada modelo
# Parámetros para Logistic Regression
lr_params = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
             'C': [0.001, 0.01, 0.1, 1, 10, 100],
             'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
             'max_iter': [100, 500, 1000, 5000]}

# Parámetros para SVM
svm_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'degree': [2, 3, 4, 5],
              'gamma': ['scale', 'auto']}

# Parámetros para Naive Bayes
nb_params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# GridSearch para Logistic Regression
lr_grid = GridSearchCV(LogisticRegression(), lr_params, cv=5, verbose=2)
lr_grid.fit(X_train, y_train)

# GridSearch para SVM
svm_grid = GridSearchCV(SVC(), svm_params, cv=5, verbose=2)
svm_grid.fit(X_train, y_train)


# GridSearch para Naive Bayes
nb_grid = GridSearchCV(MultinomialNB(), nb_params, cv=5, verbose=2)
nb_grid.fit(X_train, y_train)

print("Mejores parámetros para Logistic Regression: ", lr_grid.best_params_)
print("Mejores parámetros para SVM: ", svm_grid.best_params_)
print("Mejores parámetros para Naive Bayes: ", nb_grid.best_params_)
'''

# Entrenamiento del modelo de regresión logística con validación cruzada
cv, lr, svm, nb = 5, LogisticRegression(multi_class='ovr', C= 1, max_iter= 100, penalty= 'l2', solver= 'newton-cg'), SVC(C= 10, degree= 2, gamma= 'scale', kernel= 'poly', probability=True), MultinomialNB(alpha=1)
scoring = {'accuracy': 'accuracy', 'precision': make_scorer(precision_score, pos_label='1'), 'recall': make_scorer(recall_score, pos_label='1'), 'f1': make_scorer(f1_score, pos_label='1')}
models = [('Regresión logística:', lr), ('SVM:', svm), ('Naive Bayes:', nb)]

for model in models:
    results = cross_validate(model[1], X_train, y_train, cv=cv, scoring=scoring)
    print(model[0])
    print(f"Accuracy promedio en validación cruzada ({cv} particiones): {results['test_accuracy'].mean():.2f}")
    print(f"Precision en validación cruzada (Violent): {results['test_precision'].mean():.2f}")
    print(f"Recall en validación cruzada (Violent): {results['test_recall'].mean():.2f}")
    print(f"Puntuación F1 en validación cruzada (Violent): {results['test_f1'].mean():.2f}")
    model[1].fit(X_train, y_train)
    # Predice las etiquetas para los datos de prueba
    y_pred = model[1].predict(X_test)
    # Calcula e imprime el classification report
    report = classification_report(y_test, y_pred)
    print("Report de entrenamiento 90-10")
    print(report)
    print('\n')

voting_classifier = VotingClassifier(estimators=models, voting='soft')
voting_classifier.fit(X_train, y_train)
y_pred = voting_classifier.predict(X_test)
report = classification_report(y_test, y_pred)
print("Report del voting classifier entrenamiento 90-10")
print(report)   
print('\n')


# Representación vectorial binaria
# Vectorización de los datos
vectorizer = CountVectorizer(binary=True, token_pattern=r'(?u)\b\w+\b|\.')
X = vectorizer.fit_transform(df[0])
y = df_target[0]

print('Representación vectorial binaria')
print (vectorizer.get_feature_names_out())
print (X.toarray())
print('\n')
# Partición de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

for model in models:
    results = cross_validate(model[1], X_train, y_train, cv=cv, scoring=scoring)
    print(model[0])
    print(f"Accuracy promedio en validación cruzada ({cv} particiones): {results['test_accuracy'].mean():.2f}")
    print(f"Precision en validación cruzada (Violent): {results['test_precision'].mean():.2f}")
    print(f"Recall en validación cruzada (Violent): {results['test_recall'].mean():.2f}")
    print(f"Puntuación F1 en validación cruzada (Violent): {results['test_f1'].mean():.2f}")
    model[1].fit(X_train, y_train)
    # Predice las etiquetas para los datos de prueba
    y_pred = model[1].predict(X_test)
    # Calcula e imprime el classification report
    print("Report de entrenamiento 90-10")
    report = classification_report(y_test, y_pred)
    print(report)
    print('\n')

voting_classifier = VotingClassifier(estimators=models, voting='soft')
voting_classifier.fit(X_train, y_train)
y_pred = voting_classifier.predict(X_test)
report = classification_report(y_test, y_pred)
print("Report del voting classifier entrenamiento 90-10")
print(report)   
print('\n')

# Representación vectorial TF-IDF 
vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b|\.')
X = vectorizer.fit_transform(df[0])
y = df_target[0]

print('Representación vectorial TF-IDF')
print (vectorizer.get_feature_names_out())
print (X.toarray())
print('\n')

# Partición de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

for model in models:
    results = cross_validate(model[1], X_train, y_train, cv=cv, scoring=scoring)
    print(model[0])
    print(f"Accuracy promedio en validación cruzada ({cv} particiones): {results['test_accuracy'].mean():.2f}")
    print(f"Precision en validación cruzada (Violent): {results['test_precision'].mean():.2f}")
    print(f"Recall en validación cruzada (Violent): {results['test_recall'].mean():.2f}")
    print(f"Puntuación F1 en validación cruzada (Violent): {results['test_f1'].mean():.2f}")
    model[1].fit(X_train, y_train)
    # Predice las etiquetas para los datos de prueba
    y_pred = model[1].predict(X_test)
    # Calcula e imprime el classification report
    print("Report de entrenamiento 90-10")
    report = classification_report(y_test, y_pred)
    print(report)
    print('\n')

voting_classifier = VotingClassifier(estimators=models, voting='soft')
voting_classifier.fit(X_train, y_train)
y_pred = voting_classifier.predict(X_test)
report = classification_report(y_test, y_pred)
print("Report del voting classifier entrenamiento 90-10")
print(report)   
print('\n')
