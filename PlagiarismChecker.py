# José Luis Madrigal Sánchez A01745419
# Paulo Ogando Gulías A01751587
# César Emiliano Palome Luna A01746493
# Creado 21/05/2024

# Se requieren librerias nltk, sklearn, re, spacy
# Instalar con 'pip install nombre'
# Igualmente correr en terminal: python -m spacy download es_core_news_sm

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
import os


nltk.download('punkt')
nltk.download('omw-1.4')
nlp = spacy.load("es_core_news_sm") # Cargar modelo en español para lematización (wordnet solo funciona con inglés)

class PlagiarismChecker:
    def __init__(self):
        pass

    def lectura(self, nombre_archivo):
        '''Leer archivo para obtener su texto.
        @param nombre_archivo: nombre del txt a leer
        @return info: texto completo del archivo
        '''
        with open(nombre_archivo, encoding="utf-8") as archivo:
            info = archivo.read()
        return info
    
    def limpieza(self, texto):
        '''Remover puntuación de párrafo.
        @param texto: párrafo a limpiar.
        @return: párrafo sin puntuacion.
        '''
        return re.sub(r'[^\w\s]', '', texto)
    
    def stemming(self, oraciones):
        '''Utilización de Stemming para preprocesamiento
        @param oraciones: oraciones de párrafo limpio.
        @return: palabras con procesadas con stemming.
        '''
        ss = SnowballStemmer("spanish")
        palabras = word_tokenize(oraciones)
        stemmed_tokens = [ss.stem(palabra) for palabra in palabras]
        return ' '.join(stemmed_tokens)
    
    def lematizacion(self, oraciones):
        '''Utilización de lematización para preprocesamiento
        @param oraciones: oraciones de párrafo limpio.
        @return: palabras lematizadas.
        '''
        palabras = nlp(oraciones)
        lematized_tokens = [palabra.lemma_ for palabra in palabras]
        return ' '.join(lematized_tokens)
    
    def vectorizacion(self, tokens1, tokens2, n):
        '''Obtención de vectores por medio de tokenización.
        @param tokens1: palabras procesadas del primer párrafo.
        @param tokens2: palabras procesadas del segundo párrafo.
        @param n: cantidad de n gramas a usar.
        @return ngramas: vector comparado con corpus.
        '''
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(n, n))
        ngramas = vectorizer.fit_transform([tokens1, tokens2])
        # print(f"Corpus {n}grama: {vectorizer.get_feature_names_out()}")
        # print("------------------------------------")
        # print(f"Vector {n}grama: {ngrams.toarray()}")
        # print("------------------------------------")
        return ngramas
    
    def calcular_similitud(self, ngramas):
        '''Cálculo de similitud por coseno
        @param ngramas: vector con valores para similitud.
        @return: resultado de similitud por coseno.
        '''
        similitud = cosine_similarity(ngramas)[0, 1]
        return round(similitud * 100, 4)
    
    def generar_reporte(self, archivo, similitud):
        '''Determinar si hay plagio
        @param archivo: documento con el que se está comparando.
        @param similitud: porcentaje de similitud.
        @return: reporte de resultados de plagio.
        '''
        plagio = False
        if similitud > 59:
            plagio = True
            return [archivo, similitud, plagio]
        else:
            plagio = False
            return [archivo, similitud, plagio]
