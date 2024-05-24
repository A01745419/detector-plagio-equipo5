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
# Se requieren librerias nltk, sklearn, re, spacy
# Instalar con 'pip install nombre'
# Igualmente correr en terminal: python -m spacy download es_core_news_sm

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
# Cargar modelo en español para lematización (wordnet solo funciona con inglés)
nlp = spacy.load("es_core_news_sm")


def lectura(nombre_archivo):
    '''Leer archivo para obtener su texto.
    @param nombre_archivo: nombre del txt a leer
    @return info: texto completo del archivo
    '''
    with open(nombre_archivo, encoding="utf-8") as archivo:
        info = archivo.read()
    return info


def limpieza(texto):
    '''Remover puntuación de párrafo.
    @param texto: párrafo a limpiar.
    @return: párrafo sin puntuacion.
    '''
    return re.sub(r'[^\w\s]', '', texto)


def stemming(oraciones):
    '''Utilización de Stemming para preprocesamiento
    @param oraciones: oraciones de párrafo limpio.
    @return: palabras con procesadas con stemming.
    '''
    ps = PorterStemmer()
    palabras = word_tokenize(oraciones)
    stemmed_tokens = [ps.stem(palabra) for palabra in palabras]
    return ' '.join(stemmed_tokens)


def lematizacion(oraciones):
    '''Utilización de lematización para preprocesamiento
    @param oraciones: oraciones de párrafo limpio.
    @return: palabras lematizadas.
    '''
    palabras = nlp(oraciones)
    lematized_tokens = [palabra.lemma_ for palabra in palabras]
    return ' '.join(lematized_tokens)


def vectorizacion(tokens1, tokens2, n):
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


def calcular_similitud(ngramas):
    '''Cálculo de similitud por coseno
    @param ngramas: vector con valores para similitud.
    @return: resultado de similitud por coseno.
    '''
    similitud = cosine_similarity(ngramas)[0, 1]
    return round(similitud * 100, 4)


def generar_reporte(archivo, similitud):
    '''Determinar si hay plagio
    @param archivo: documento con el que se está comparando.
    @param similitud: porcentaje de similitud.
    @return: reporte de resultados de plagio.
    '''
    plagio = False
    if similitud > 80:
        plagio = True
    else:
        plagio = False
    return (f"{archivo}  |        {similitud}         | {plagio}")


def main():
    # Lectura del texto a comprobar su plagio
    parrafo_plagio = lectura("textoprueba.txt")
    # Limpieza de texto
    plagio_limpio = limpieza(parrafo_plagio)
    # Preprocesamiento de texto
    plagio_lemmatized = lematizacion(plagio_limpio)
    # Obtener todos los archivos con los cuales comparar
    lista_textos = os.listdir("originales")
    print("")
    print(f'Texto       |  % Unigrama lematizado | Plagio')

    for texto in lista_textos:
        parrafo_original = lectura(f"originales/{texto}")
        original_limpio = limpieza(parrafo_original)
        original_lemmatized = lematizacion(original_limpio)
        # Vector con corpus unigrama
        vector_unigrama_lemmatized = \
            vectorizacion(plagio_lemmatized, original_lemmatized, 1)
        # Similitud por coseno
        similitud_unigrama_lemmatized = \
            calcular_similitud(vector_unigrama_lemmatized)
        # Determinación de plagio
        resultados = generar_reporte(texto, similitud_unigrama_lemmatized)
        print(resultados)


if __name__ == '__main__':
    main()
