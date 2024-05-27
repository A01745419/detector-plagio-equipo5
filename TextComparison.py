from PlagiarismChecker import PlagiarismChecker
import os

lista_textos_originales = os.listdir("originales")
lista_textos_sospechoso = os.listdir("sospechosos")

#Instancia de la clase PlagiarismChecker
PlagiarismChecker = PlagiarismChecker()

#Lectura del texto a comprobar su plagio
parrafo_plagio = PlagiarismChecker.lectura(f"sospechosos/{lista_textos_sospechoso[0]}")

#Limpieza de texto
plagio_limpio = PlagiarismChecker.limpieza(parrafo_plagio)

#Preprocesamiento de texto
plagio_lemmatized = PlagiarismChecker.lematizacion(plagio_limpio)

#Obtener todos los archivos con los cuales comparar
print("")
print(f'   Texto    |  % Unigrama lematizado | Plagio')

for texto in lista_textos_originales:

    parrafo_original = PlagiarismChecker.lectura(f"originales/{texto}")
    original_limpio = PlagiarismChecker.limpieza(parrafo_original)
    original_lemmatized = PlagiarismChecker.lematizacion(original_limpio)

    #Vector con corpus unigrama
    vector_unigrama_lemmatized = PlagiarismChecker.vectorizacion(plagio_lemmatized, original_lemmatized, 1)

    #Similitud por coseno
    similitud_unigrama_lemmatized = PlagiarismChecker.calcular_similitud(vector_unigrama_lemmatized)

    #Determinación de plagio
    resultados = PlagiarismChecker.generar_reporte(texto, similitud_unigrama_lemmatized)
    print(resultados)
