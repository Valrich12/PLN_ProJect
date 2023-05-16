import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier 
from sklearn.base import BaseEstimator, ClassifierMixin
##Multioutput classifier

# Lectura de los archivos de datos
with open("corpus_procesado.csv", "r") as f:
    corpus = f.read()
df = pd.DataFrame(corpus.splitlines())

df_target = pd.read_csv("development_data_davincis23_V2_subtask2/train_labels_subtask_2.csv", sep=',', header=None)

# Representación vectorial por frecuencia
vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b|\.')
X = vectorizer.fit_transform(df[0])
y = df_target.values
print(y[22])

print('Representación vectorial por frecuencia')
print(vectorizer.get_feature_names_out())
print(X.toarray())
print('\n')

# Partición de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

''' GridSearch para encontrar los mejores parámetros para cada clasificador 
# Parámetros para Logistic Regression
lr_params = {'estimator__penalty': ['l1', 'l2', 'elasticnet', 'none'],
             'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100],
             'estimator__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
             'estimator__max_iter': [100, 500, 1000, 5000]}

# Parámetros para SVM
svm_params = {'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100],
              'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'estimator__degree': [2, 3, 4, 5],
              'estimator__gamma': ['scale', 'auto']}

# Parámetros para Naive Bayes
nb_params = {'estimator__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# GridSearch para Logistic Regression
lr_grid = GridSearchCV(MultiOutputClassifier(LogisticRegression()), lr_params, cv=5, verbose=2)
lr_grid.fit(X_train, y_train)
print("Mejores parámetros para Logistic Regression: ", lr_grid.best_params_)

# GridSearch para SVM
svm_grid = GridSearchCV(MultiOutputClassifier(SVC()), svm_params, cv=5, verbose=2)
svm_grid.fit(X_train, y_train)
print("Mejores parámetros para SVM: ", svm_grid.best_params_)

# GridSearch para Naive Bayes
nb_grid = GridSearchCV(MultiOutputClassifier(MultinomialNB()), nb_params, cv=5, verbose=2)
nb_grid.fit(X_train, y_train)
print("Mejores parámetros para Naive Bayes: ", nb_grid.best_params_)


'''
cv, lr, svm, nb = 5, LogisticRegression(multi_class='ovr', C= 1, max_iter= 100, penalty= 'l2', solver= 'newton-cg'), SVC(C= 10, degree= 2, gamma= 'scale', kernel= 'poly', probability=True), MultinomialNB(alpha=1)
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall': make_scorer(recall_score, average='weighted', zero_division=0),
    'f1': make_scorer(f1_score, average='weighted', zero_division=0)
}

models = [('Regresión logística:', lr), ('SVM optimizada:', svm), ('Naive Bayes:', nb)]

for model in models:
    multi_model = MultiOutputClassifier(model[1])
    results = cross_validate(multi_model, X_train, y_train, cv=cv, scoring=scoring)
    print(model[0])
    print(f"Accuracy promedio en validación cruzada ({cv} particiones): {results['test_accuracy'].mean():.2f}")
    print(f"Precision promedio ponderada en validación cruzada: {results['test_precision'].mean():.2f}")
    print(f"Recall promedio ponderado en validación cruzada: {results['test_recall'].mean():.2f}")
    print(f"Puntuación F1 promedio ponderada en validación cruzada: {results['test_f1'].mean():.2f}")
    multi_model.fit(X_train, y_train)
    # Predice las etiquetas para los datos de prueba
    y_pred = multi_model.predict(X_test)
    # Calcula e imprime el classification report
    report = classification_report(y_test, y_pred, zero_division=0)
    print("Report de entrenamiento 90-10")
    print(report)
    print('\n')

# Crear el Voting Classifier
voting_classifier = VotingClassifier(estimators=models, voting='soft')
multi_output_classifier = MultiOutputClassifier(voting_classifier)
multi_output_classifier.fit(X_train, y_train)
y_pred = multi_output_classifier.predict(X_test)
report = classification_report(y_test, y_pred, zero_division=0)
print("Report del voting classifier entrenamiento 90-10")
print(report)   
print('\n')

#for test, pred in zip(y_test, y_pred):
#    print("Valor real: ", test, " Predicción: ", pred)


# Representación vectorial binaria
# Vectorización de los datos
vectorizer = CountVectorizer(binary=True, token_pattern=r'(?u)\b\w+\b|\.')
X = vectorizer.fit_transform(df[0])
y = df_target.values

print('Representación vectorial binaria')
print (vectorizer.get_feature_names_out())
print (X.toarray())
print('\n')
# Partición de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

for model in models:
    multi_model = MultiOutputClassifier(model[1])
    results = cross_validate(multi_model, X_train, y_train, cv=cv, scoring=scoring)
    print(model[0])
    print(f"Accuracy promedio en validación cruzada ({cv} particiones): {results['test_accuracy'].mean():.2f}")
    print(f"Precision promedio ponderada en validación cruzada: {results['test_precision'].mean():.2f}")
    print(f"Recall promedio ponderado en validación cruzada: {results['test_recall'].mean():.2f}")
    print(f"Puntuación F1 promedio ponderada en validación cruzada: {results['test_f1'].mean():.2f}")
    multi_model.fit(X_train, y_train)
    # Predice las etiquetas para los datos de prueba
    y_pred = multi_model.predict(X_test)
    # Calcula e imprime el classification report
    report = classification_report(y_test, y_pred, zero_division=0)
    print("Report de entrenamiento 90-10")
    print(report)
    print('\n')

# Crear el Voting Classifier
voting_classifier = VotingClassifier(estimators=models, voting='soft')
multi_output_classifier = MultiOutputClassifier(voting_classifier)
multi_output_classifier.fit(X_train, y_train)
y_pred = multi_output_classifier.predict(X_test)
report = classification_report(y_test, y_pred, zero_division=0)
print("Report del voting classifier entrenamiento 90-10")
print(report)   
print('\n')

# Representación vectorial TF-IDF 
vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b|\.')
X = vectorizer.fit_transform(df[0])
y = df_target.values

print('Representación vectorial TF-IDF')
print (vectorizer.get_feature_names_out())
print (X.toarray())
print('\n')

for model in models:
    multi_model = MultiOutputClassifier(model[1])
    results = cross_validate(multi_model, X_train, y_train, cv=cv, scoring=scoring)
    print(model[0])
    print(f"Accuracy promedio en validación cruzada ({cv} particiones): {results['test_accuracy'].mean():.2f}")
    print(f"Precision promedio ponderada en validación cruzada: {results['test_precision'].mean():.2f}")
    print(f"Recall promedio ponderado en validación cruzada: {results['test_recall'].mean():.2f}")
    print(f"Puntuación F1 promedio ponderada en validación cruzada: {results['test_f1'].mean():.2f}")
    multi_model.fit(X_train, y_train)
    # Predice las etiquetas para los datos de prueba
    y_pred = multi_model.predict(X_test)
    # Calcula e imprime el classification report
    report = classification_report(y_test, y_pred, zero_division=0)
    print("Report de entrenamiento 90-10")
    print(report)
    print('\n')

# Crear el Voting Classifier
voting_classifier = VotingClassifier(estimators=models, voting='soft')
multi_output_classifier = MultiOutputClassifier(voting_classifier)
multi_output_classifier.fit(X_train, y_train)
y_pred = multi_output_classifier.predict(X_test)
report = classification_report(y_test, y_pred, zero_division=0)
print("Report del voting classifier entrenamiento 90-10")
print(report)   
print('\n')
