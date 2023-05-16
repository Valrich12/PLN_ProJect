from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import os, pickle
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import spacy
from sklearn.model_selection import train_test_split
import re

class data_set_attraction:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

def lemmatization(train, test):
	lemmatized_train = []
	lemmatized_test = []
	
	nlp = spacy.load("es_core_news_sm")
	for index in range(len(train)):
		# ~ print (index)
		line = train[index]
		
		string = ''
		title = str(line[0])
		opinion = str(line[1])
		opinion = re.sub('\n', ' ', opinion)
		string = title + ' ' + opinion
		
		lemmatized_string = ''
		doc = nlp(string)
	
		for token in doc:
		# ~ print(token.text, token.pos_, token.dep_, token.lemma_)
			lemmatized_string = lemmatized_string + token.lemma_ + " "
		lemmatized_string = lemmatized_string[:-1]
		
		lemmatized_train.append(lemmatized_string)
	
	for index in range(len(test)):
		# ~ print (index)
		line = test[index]
		
		string = ''
		title = str(line[0])
		opinion = str(line[1])
		opinion = re.sub('\n', ' ', opinion)
		string = title + ' ' + opinion
		
		lemmatized_string = ''
		doc = nlp(string)
	
		for token in doc:
		# ~ print(token.text, token.pos_, token.dep_, token.lemma_)
			lemmatized_string = lemmatized_string + token.lemma_ + " "
		lemmatized_string = lemmatized_string[:-1]
		
		lemmatized_test.append(lemmatized_string)
	
	return (lemmatized_train, lemmatized_test)

def generate_train_test(file_name):

	#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
	df = pd.read_excel(file_name)
	# ~ print (df)
	X = df.drop(['Polarity', 'Country', 'Type'],axis=1).values   
	y = df['Type'].values
	
	# ~ print(X)
	# ~ print (y)
	
	#~ #Separa el corpus cargado en el DataFrame en el 80% para entrenamiento y el 20% para pruebas
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
	# ~ print (X_train)
	# ~ print (X_test)
	# ~ print (X_train.shape)
	# ~ print (y_train)
	# ~ print (y_train.shape)
	# ~ print(X_test.shape)
	
	#Lematiza los conjuntos de datos de entrenamiento y prueba
	lemmatized_X_train, lemmatized_X_test = lemmatization(X_train, X_test)
	  
	# ~ print (lemmatized_X_train)
	# ~ print (lemmatized_X_test)
	
	#Se retorna el objeto creado para almacenar los conjuntos de datos de entrenamiento y prueba lematizados
	return (data_set_attraction(lemmatized_X_train, y_train, lemmatized_X_test, y_test))

def train_model(corpus):
	
	# Representación vectorial por frecuencia
	vectorizer = CountVectorizer()
	X_train = vectorizer.fit_transform(corpus.X_train)
	y_train = corpus.y_train
	print (vectorizer.get_feature_names_out())
	print (X_train.shape)#sparse matrix
	
	#Se crea un objeto clasificador
	# ~ clf = LogisticRegression()
	clf = LogisticRegression(max_iter=300)
	#Se entrena al clasificador
	clf.fit(X_train, y_train)
	
	#Se retorna el clasificador y el vectorizador
	return (clf, vectorizer)

def test_model(corpus, model, vectorizer):
	#Se vectoriza el conjunto de prueba
	X_test = vectorizer.transform (corpus.X_test)
	y_test = corpus.y_test
	
	#Se realizan las predicciones del conjunto de pruebas con el modelo entrenado
	predictions = model.predict(X_test)
	print (predictions)
	
	#Se calcula la exactitud del las predicciones del modelo entrenado
	print (accuracy_score(y_test, predictions))

if __name__=='__main__':
	#Verificar si el corpus ya ha sido creado
	if not (os.path.exists('corpus.pkl')):
		#Genera el conjunto de entrenamiento y prueba, y se guarda para su posterior carga
		corpus = generate_train_test('Rest_Mex_Recortado.xlsx')
		dataset_file = open ('corpus.pkl','wb')
		pickle.dump(corpus, dataset_file)
		dataset_file.close()
	else:#En caso de que el corpus exista se carga
		corpus_file = open ('corpus.pkl','rb')
		corpus = pickle.load(corpus_file)
		corpus_file.close()

	# ~ print (corpus.X_train[0])
	# ~ print (corpus.y_train[0])
	
	#Se entrena el modelo
	model, vectorizer = train_model(corpus)
	
	#Se prueba el modelo entrenado
	test_model(corpus, model, vectorizer)






