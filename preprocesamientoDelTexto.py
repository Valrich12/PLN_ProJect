
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
import pandas as pd

#Se define una lista de stop words, de acuerdo a lo que se pide en la práctica
nuevas_stop_words = { ##Articulos
                        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
                        'lo', 'al', 'del', 'este', 'ese', 'aquel', 'un poco de', 'mucho', 'poco',
                        'otro', 'cierto', 'algún', 'alguna', 'algunos', 'algunas', 'varios', 'varias',
                        'ambos', 'ambas', 'cada', 'cualquier', 'cualquieras', 'demasiado', 'demasiada', 
                        'demasiados', 'demasiadas', 'menos', 'más', 'medio', 'media', 'medios', 'medias',
                        'ningún', 'ninguna', 'ningunos', 'ningunas', 'varios', 'varias', 'poco', 'poca', 'a la',
                    ##Preposiciones
                        'a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en', 'entre',
                        'hacia', 'hasta', 'mediante', 'para', 'por', 'segun', 'sin', 'so', 'sobre', 'tras',
                        'versus', 'vía', 'a través de', 'a causa de', 'a pesar de', 'a propósito de', 'a raíz de',
                        'durante', 'excepto', 'frente a', 'junto a', 'menos', 'salvo', 'según', 'según con',
                        'sobre todo', 'dentro de', 'encima de', 'detrás de', 'fuera de', 'más allá de', 'debajo de',
                        'dentro de', 'de', 'DE',
                    ##Conjunciones
                        'y', 'e', 'ni', 'o', 'u', 'que', 'si', 'mas', 'pero', 'aunque', 'sino', 'para que', 'porque',
                        'ya que', 'pues', 'como', 'así que', 'mientras', 'cuando', 'después', 'antes', 'hasta que',
                        'siempre que', 'a menos que', 'en caso de que', 'con tal de que', 'sin que', 'por más que',
                        'a fin de que', 'a pesar de que', 'en tanto que', 'aunque no', 'por cuanto', 'sea que',
                        'de manera que', 'por lo tanto', 
                    ##Pronombres
                        'yo', 'tu', 'el', 'ella', 'usted', 'nosotros', 'nosotras', 'vosotros', 'vosotras', 'ellos',
                        'ellas', 'ustedes', 'quien', 'quienes', 'cual', 'cuales', 'cuanto', 'cuanta', 'cuantos',
                        'cuantas', 'que', 'esto', 'eso', 'aquello', 'nada', 'algo', 'alguien', 'nadie', 'quienquiera',
                        'quienesquiera', 'cualquiera', 'cualesquiera', 'cuantoquiera', 'cuantaquiera', 'cuantosquiera',
                        'cuantasquiera', 'quequiera', 'dondequiera', 'comoquiera', 'cuandoquiera', 'estos', 'esos',
                    ##Otros
                        'edd'
                    }

#Se agrega la lista de stop words anterior a la lista por defecto de Spacy
#Para ver la lista original de Spacy: verStopWords.py
#stop_words = STOP_WORDS.union(nuevas_stop_words)

# Ruta del archivo "train_data.csv"
csv_path = "development_data_davincis23_V2_subtask2/train_data.csv"

#Se reemplaza la lista de stop words por defecto de spacy 
stop_words = nuevas_stop_words

#print(stop_words)

#Se carga el corpus para el tagger en español
nlp = spacy.load('es_core_news_sm')

#Se lee el archivo corpus_noticias.txt y se almacena en un DataFrame de Pandas la tercer columna (noticias)"
df = pd.read_csv(csv_path, sep=",", header=None, engine="python")
noticias = df.iloc[:, 1]


# Tokenizar, lematizar y remover stop words en cada noticia
noticias_procesadas = []
for noticia in noticias:
    doc = nlp(noticia)
                                                                 #and not token.is_punct: signos de puntación
    tokens = [token.lemma_ for token in doc if token.lemma_.lower() not in stop_words]
    noticias_procesadas.append(" ".join(tokens))

# Crear un nuevo archivo con el corpus procesado
with open('corpus_procesado.csv', 'w', encoding='utf-8') as file:
    for noticia in noticias_procesadas:
        file.write(noticia + '\n')
