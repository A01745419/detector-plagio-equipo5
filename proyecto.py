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

# Primer párrafo
parrafo1 = """La historia detrás de los orígenes de Batman es casi tan interesante en la vida real como en los cómics. El Hombre Murciélago nace de una idea original de Bob Kane, que tenía poco más que un nombre, y su colaborador habitual Bill Finger, que diseñó tanto el traje como la historia de Bruce Wayne. Un nombre que, por cierto, no es para nada casual."""
# Primer párrafo parafraseado (plagio)
parafraseado1 = """La historia del origen de Batman es casi tan interesante en la vida real como lo es en los cómics. Batman nació de la idea original de Bob Kane, que tiene un solo nombre, y su frecuente colaborador Bill Finger, quien diseñó el traje de Batman y la historia de Bruce Wayne. Por cierto, este nombre no es casual."""

# Remover puntuación
def limpieza(texto):
    return re.sub(r'[^\w\s]', '', texto)

# Párrafos sin puntuación
parrafo_limpio1 = limpieza(parrafo1)
parrafo_limpio2 = limpieza(parafraseado1)

# Utilización de Stemming para preprocesamiento
def stemming(oraciones):
    ps = PorterStemmer()
    palabras = word_tokenize(oraciones)
    stemmed_tokens = [ps.stem(palabra) for palabra in palabras]
    return ' '.join(stemmed_tokens)

# Parrafos preprocesados
stemmed_oraciones1 = stemming(parrafo_limpio1)
stemmed_oraciones2 = stemming(parrafo_limpio2)

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

# Vectores
vector_unigrama = vectorizacion(stemmed_oraciones1, stemmed_oraciones2, 1)
vector_bigrama = vectorizacion(stemmed_oraciones1, stemmed_oraciones2, 2)
vector_trigrama = vectorizacion(stemmed_oraciones1, stemmed_oraciones2, 3)

# Similitudes
similitud_unigrama = calcular_similitud(vector_unigrama)
similitud_bigrama = calcular_similitud(vector_bigrama)
similitud_trigrama = calcular_similitud(vector_trigrama)

print(f"Porcentaje de similitud por unigrama: {similitud_unigrama * 100} %")
print(f"Porcentaje de similitud por bigrama: {similitud_bigrama * 100} %")
print(f"Porcentaje de similitud por trigrama: {similitud_trigrama * 100} %")
