# José Luis Madrigal Sánchez A01745419
# Paulo Ogando Gulías A01751587
# César Emiliano Palome Luna A01746493
# Creado 21/05/2024
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

nltk.download('punkt')

# Leer archivo
def lectura(nombre_archivo):
    with open(nombre_archivo, encoding="utf-8") as archivo:
        info = archivo.read()
    return info

# Remover puntuación
def limpieza(texto):
    return re.sub(r'[^\w\s]', '', texto)

# Utilización de Stemming para preprocesamiento
def stemming(oraciones):
    ps = PorterStemmer()
    palabras = word_tokenize(oraciones)
    stemmed_tokens = [ps.stem(palabra) for palabra in palabras]
    return ' '.join(stemmed_tokens)

# Obtención de vectores por medio de tokenización
def vectorizacion(tokens1, tokens2, n):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(n, n))
    ngramas = vectorizer.fit_transform([tokens1, tokens2])
    # print(f"Corpus {n}grama: {vectorizer.get_feature_names_out()}")
    # print("------------------------------------")
    # print(f"Vector {n}grama: {ngrams.toarray()}")
    # print("------------------------------------")
    return ngramas

# Cálculo de similitud por coseno
def calcular_similitud(ngramas):
    similitud = cosine_similarity(ngramas)[0, 1]
    return similitud

def main():
    # Primer párrafo
    parrafo1 = lectura("texto1.txt")
    # Primer párrafo parafraseado (plagio)
    parafraseado1 = lectura("texto2.txt")
    # Párrafos sin puntuación
    parrafo_limpio1 = limpieza(parrafo1)
    parrafo_limpio2 = limpieza(parafraseado1)
    # Parrafos preprocesados
    stemmed_oraciones1 = stemming(parrafo_limpio1)
    stemmed_oraciones2 = stemming(parrafo_limpio2)
    # Vectores
    vector_unigrama = vectorizacion(stemmed_oraciones1, stemmed_oraciones2, 1)
    vector_bigrama = vectorizacion(stemmed_oraciones1, stemmed_oraciones2, 2)
    vector_trigrama = vectorizacion(stemmed_oraciones1, stemmed_oraciones2, 3)
    # Similitudes
    similitud_unigrama = calcular_similitud(vector_unigrama)
    similitud_bigrama = calcular_similitud(vector_bigrama)
    similitud_trigrama = calcular_similitud(vector_trigrama)
    # Mostrar similitudes
    print(f"Porcentaje de similitud por unigrama: {similitud_unigrama * 100} %")
    print(f"Porcentaje de similitud por bigrama: {similitud_bigrama * 100} %")
    print(f"Porcentaje de similitud por trigrama: {similitud_trigrama * 100} %")

if __name__ == '__main__':
    main()
