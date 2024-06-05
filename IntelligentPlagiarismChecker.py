import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

nltk.download('punkt')
nltk.download('omw-1.4')
# Cargar modelo en español para lematización (wordnet solo funciona con inglés)
nlp = spacy.load("es_core_news_sm")

class IntelligentPlagiarismChecker:
    def __init__(self):
        # Cargar etiquetas de plagio
        with open('plagio_labels.json', 'r', encoding='utf-8') as file:
            self.plagio_labels = json.load(file)['fraudulentos']
        
        # Entrenar modelo de clasificación de tipos de plagio
        self.modelo_plagio = self.entrenar_modelo_plagio()

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
        return ngramas

    def calcular_similitud(self, ngramas):
        '''Cálculo de similitud por coseno
        @param ngramas: vector con valores para similitud.
        @return: resultado de similitud por coseno.
        '''
        similitud = cosine_similarity(ngramas)[0, 1]
        return round(similitud * 100, 4)

    def generar_reporte(self, archivo, similitud, tipo_plagio):
        '''Determinar si hay plagio
        @param archivo: documento con el que se está comparando.
        @param similitud: porcentaje de similitud.
        @param tipo_plagio: tipo de plagio detectado.
        @return: reporte de resultados de plagio.
        '''
        plagio = False
        if similitud > 59:
            plagio = True
        return [archivo, similitud, plagio, tipo_plagio]

    def entrenar_modelo_plagio(self):
        '''Entrenar modelo de clasificación de tipos de plagio
        @return modelo: modelo entrenado
        '''
        textos = []
        labels = []

        for archivo, tipo_plagio in self.plagio_labels.items():
            texto = self.lectura(f"sospechosos/{archivo}")
            texto_limpio = self.limpieza(texto)
            texto_lema = self.lematizacion(texto_limpio)
            textos.append(texto_lema)
            labels.append(tipo_plagio)
        
        X_train, X_test, y_train, y_test = train_test_split(textos, labels, test_size=0.2, random_state=42)
        
        modelo = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        modelo.fit(X_train, y_train)
        print(f"Precisión del modelo de clasificación de plagio: {modelo.score(X_test, y_test)}")
        
        return modelo

    def predecir_tipo_plagio(self, texto):
        '''Predecir tipo de plagio de un texto
        @param texto: texto lematizado
        @return: tipo de plagio predicho
        '''
        tipo_plagio = self.modelo_plagio.predict([texto])[0]
        return tipo_plagio
