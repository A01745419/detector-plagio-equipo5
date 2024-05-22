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
    return similitud

def main():
    # Primer párrafo
    parrafo1 = lectura("texto1.txt")
    # Primer párrafo parafraseado (plagio)
    parrafo2 = lectura("texto2.txt")
    # Párrafos sin puntuación
    parrafo_limpio1 = limpieza(parrafo1)
    parrafo_limpio2 = limpieza(parrafo2)
    # Parrafos preprocesados
    stemmed_oraciones1 = stemming(parrafo_limpio1)
    stemmed_oraciones2 = stemming(parrafo_limpio2)
    lematized_oraciones1 = lematizacion(parrafo_limpio1)
    lematized_oraciones2 = lematizacion(parrafo_limpio2)
    #print(stemmed_oraciones1)
    #print(lematized_oraciones1)
    # Vectores
    vector_unigrama_stemmed = vectorizacion(stemmed_oraciones1, stemmed_oraciones2, 1)
    vector_bigrama_stemmed = vectorizacion(stemmed_oraciones1, stemmed_oraciones2, 2)
    vector_trigrama_stemmed = vectorizacion(stemmed_oraciones1, stemmed_oraciones2, 3)
    vector_unigrama_lemmatized = vectorizacion(lematized_oraciones1, lematized_oraciones2, 1)
    vector_bigrama_lemmatized = vectorizacion(lematized_oraciones1, lematized_oraciones2, 2)
    vector_trigrama_lemmatized = vectorizacion(lematized_oraciones1, lematized_oraciones2, 3)
    # Similitudes
    similitud_unigrama_stemmed = calcular_similitud(vector_unigrama_stemmed)
    similitud_bigrama_stemmed = calcular_similitud(vector_bigrama_stemmed)
    similitud_trigrama_stemmed = calcular_similitud(vector_trigrama_stemmed)
    similitud_unigrama_lemmatized = calcular_similitud(vector_unigrama_lemmatized)
    similitud_bigrama_lemmatized = calcular_similitud(vector_bigrama_lemmatized)
    similitud_trigrama_lemmatized = calcular_similitud(vector_trigrama_lemmatized)
    # Mostrar similitudes
    print(f"Similitud por unigrama stemmed: {similitud_unigrama_stemmed * 100} %")
    print(f"Similitud por bigrama stemmed: {similitud_bigrama_stemmed * 100} %")
    print(f"Similitud por trigrama stemmed: {similitud_trigrama_stemmed * 100} %")
    print(f"Similitud por unigrama lemmatized: {similitud_unigrama_lemmatized * 100} %")
    print(f"Similitud por bigrama lemmatized: {similitud_bigrama_lemmatized * 100} %")
    print(f"Similitud por trigrama lemmatized: {similitud_trigrama_lemmatized * 100} %")

if __name__ == '__main__':
    main()
