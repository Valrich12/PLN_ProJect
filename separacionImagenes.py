import csv
import os
import shutil
import pandas as pd
import re

i=0

# Ruta del archivo "train_data.csv"
csv_path = "development_data_davincis23_V2_subtask2/train_data.csv"
# Ruta de la carpeta "train_images"
img_path = "train_images/train_images"
# Ruta del archivo "train_labels_subtask_2.csv"
label_path = "development_data_davincis23_V2_subtask2/train_labels_subtask_2.csv"
# Ruta para binario
label_pathb="development_data_davincis23_V2_subtask2/train_labels_subtask_1.csv"

# Lee el archivo CSV utilizando Pandas
df = pd.read_csv(label_path, sep=",", header=None)
# Lectura del label binario
dfb = pd.read_csv(label_pathb, sep=",", header=None)

# Lee el archivo "train_data.csv" y recorre todas las filas
with open(csv_path, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        # Obtiene el nombre del archivo de la primera columna, quita los corchetes y las comillas simples
        file_name = row[0].strip("[]'\"")
        if ',' in file_name:
            # dividir el valor de la primera columna en una lista
            file_name_split = file_name.split(',')
            # tomar el primer elemento de la lista
            file_name_first = file_name_split[0]
            file_name = file_name_first
            file_name = file_name.strip("[]'\"")
            
        #print(file_name)
        # Verifica si el archivo existe en la carpeta "train_images"
        if os.path.isfile(os.path.join(img_path, file_name)):
            #binario dfb
            fila_i = dfb.iloc[i].tolist()   
            if(fila_i[0]==1):
                            #binario:violent
                category = "Violent"
            #para binario
            else:
                category = "NonViolent"
            #elif(fila_i[1]==1):
            #    category = "Murder"
            #elif(fila_i[2]==1):
            #    category = "Theft"
            #elif(fila_i[3]==1):
            #    category = "None"
            #else:
            #   category = "0"
            print(f'Archivo: {file_name}, category: {category}')
            # Copia el archivo a la carpeta correspondiente
            shutil.copy(os.path.join(img_path, file_name), os.path.join(img_path, category, file_name))
        i=i+1
            
            
            
                