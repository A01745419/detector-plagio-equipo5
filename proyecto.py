# José Luis Madrigal Sánchez A01745419
# Paulo Ogando Gulías A01751587
# César Emiliano Palome Luna A01746493
# Creado 21/05/2024
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
import os

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
# Cargar modelo en español para lematización (wordnet solo funciona con inglés)
nlp = spacy.load("es_core_news_sm")
# correr en terminal: python -m spacy download es_core_news_sm

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

# Utilización de lematización para preprocesamiento
def lematizacion(oraciones):
    #lm = WordNetLemmatizer()
    #palabras = word_tokenize(oraciones)
    #lematized_tokens = [lm.lemmatize(palabra) for palabra in palabras]
    palabras = nlp(oraciones)
    lematized_tokens = [palabra.lemma_ for palabra in palabras]
    return ' '.join(lematized_tokens)

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
    return round(similitud * 100, 4)

# Determinar si hay plagio
def generar_reporte(archivo, similitud, similitud2, similitud3, similitud4, similitud5, similitud6):
    plagio = False
    if similitud > 80:
        plagio = True
    else:
        plagio = False
    return (f"{archivo}  |    {similitud}    |    {similitud2}   |    {similitud3}    |    {similitud4}    |    {similitud5}    |    {similitud6}    |    {plagio}")

def main():
    # Lectura del texto a comprobar su plagio
    parrafo_plagio = lectura("textoprueba.txt")
    #limpieza de texto
    plagio_limpio = limpieza(parrafo_plagio)
    #preprocesamiento de texto
    plagio_stemmed = stemming(plagio_limpio)
    plagio_lemmatized = lematizacion(plagio_limpio)

    # Lectura y preprocesamiento de todos los textos originales para comparar con el texto a comprobar
    lista_textos = os.listdir("originales")

    print(f'Texto       |  1grama stem  |  2grama stem  |  3grama stem  |  1grama lemm  |  2grama lemm  |  3grama lemm  |  Plagio')

    for texto in lista_textos:
        parrafo_original = lectura(f"originales/{texto}")
        original_limpio = limpieza(parrafo_original)
        original_stemmed = stemming(original_limpio)
        original_lemmatized = lematizacion(original_limpio)
        # Vectores
        vector_unigrama_stemmed = vectorizacion(plagio_stemmed, original_stemmed, 1)
        vector_bigrama_stemmed = vectorizacion(plagio_stemmed, original_stemmed, 2)
        vector_trigrama_stemmed = vectorizacion(plagio_stemmed, original_stemmed, 3)
        vector_unigrama_lemmatized = vectorizacion(plagio_lemmatized, original_lemmatized, 1)
        vector_bigrama_lemmatized = vectorizacion(plagio_lemmatized, original_lemmatized, 2)
        vector_trigrama_lemmatized = vectorizacion(plagio_lemmatized, original_lemmatized, 3)
        # Similitudes
        similitud_unigrama_stemmed = calcular_similitud(vector_unigrama_stemmed)
        similitud_bigrama_stemmed = calcular_similitud(vector_bigrama_stemmed)
        similitud_trigrama_stemmed = calcular_similitud(vector_trigrama_stemmed)
        similitud_unigrama_lemmatized = calcular_similitud(vector_unigrama_lemmatized)
        similitud_bigrama_lemmatized = calcular_similitud(vector_bigrama_lemmatized)
        similitud_trigrama_lemmatized = calcular_similitud(vector_trigrama_lemmatized)
        
        resultados = generar_reporte(texto, similitud_unigrama_lemmatized, similitud_bigrama_lemmatized, similitud_trigrama_lemmatized, similitud_unigrama_stemmed, similitud_bigrama_stemmed, similitud_trigrama_stemmed)
        print(resultados)

if __name__ == '__main__':
    main()
