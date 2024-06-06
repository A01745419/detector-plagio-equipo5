import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
import numpy as np
import re
import spacy
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from langdetect import detect
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('stopwords')
# Cargar modelos en español e inglés
nlp_es = spacy.load("es_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")
stop_words_es = set(stopwords.words('spanish'))
stop_words_en = set(stopwords.words('english'))

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
        limpio = re.sub(r'[^\w\s]', '', texto)
        limpio = limpio.lower()
        return limpio

    def lematizacion(self, oraciones):
        '''Utilización de lematización para preprocesamiento
        @param oraciones: oraciones de párrafo limpio.
        @return: palabras lematizadas.
        '''
        idioma = detect(oraciones)
        if idioma == 'es':
            nlp = nlp_es
            stop_words = stop_words_es
        elif idioma == 'en':
            nlp = nlp_en
            stop_words = stop_words_en
        palabras = nlp(oraciones)
        lematized_tokens = [palabra.lemma_ for palabra in palabras if palabra.lemma_ not in stop_words]
        return ' '.join(lematized_tokens)

    def vectorizacion(self, tokens1, tokens2, n):
        '''Obtención de vectores por medio de tokenización.
        @param tokens1: palabras procesadas del primer párrafo.
        @param tokens2: palabras procesadas del segundo párrafo.
        @param n: cantidad de n gramas a usar.
        @return ngramas: vector comparado con corpus.
        '''
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, n))
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
        if similitud > 40:
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
        
        # Ajuste de hiperparámetros
        param_grid = {
            'tfidf__ngram_range': [(1, 7), (1, 8), (1, 9)],
            'clf__n_estimators': [100, 200, 300],
            'clf__max_depth': [None, 10, 20, 30],
            'clf__min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(modelo, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(f"Mejores parámetros: {grid_search.best_params_}")
        print(f"Precisión del modelo de clasificación de plagio: {best_model.score(X_test, y_test)}")
        print("")

        return best_model


    def predecir_tipo_plagio(self, texto):
        '''Predecir tipo de plagio de un texto
        @param texto: texto lematizado
        @return: tipo de plagio predicho
        '''
        tipo_plagio = self.modelo_plagio.predict([texto])[0]
        return tipo_plagio
