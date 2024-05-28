# José Luis Madrigal Sánchez A01745419
# Paulo Ogando Gulías A01751587
# César Emiliano Palome Luna A01746493
# Creado 28/05/2024
from PlagiarismChecker import PlagiarismChecker
import os

# Archivos dentro de las carpetas
lista_textos_originales = os.listdir("originales")
lista_textos_sospechoso = os.listdir("sospechosos")

# Instancia de la clase PlagiarismChecker
PlagiarismChecker = PlagiarismChecker()


def lectura_y_preprocesamiento_texto(texto):
    '''Obtener texto de archivo y procesarlo.
    @param texto: nombre del archivo.
    @return plagio_lemmatized: texto lematizado.
    '''
    # Lectura del texto a comprobar su plagio
    parrafo_plagio = PlagiarismChecker.lectura(texto)
    # Limpieza de texto
    plagio_limpio = PlagiarismChecker.limpieza(parrafo_plagio)
    # Preprocesamiento de texto
    plagio_lemmatized = PlagiarismChecker.lematizacion(plagio_limpio)

    return plagio_lemmatized


def comparar_textos(plagio_lemmatized):
    '''Calcula similitud para comparar con umbrar y determinar plagio.
    @param plagio_lemmatized: texto lematizado.
    '''
    resultados = []
    for texto in lista_textos_originales:
        original_lemmatized = \
            lectura_y_preprocesamiento_texto(f"originales/{texto}")
        vector_unigrama_lemmatized = \
            PlagiarismChecker.vectorizacion(
                plagio_lemmatized, original_lemmatized, 1)
        similitud_unigrama_lemmatized = \
            PlagiarismChecker.calcular_similitud(
                vector_unigrama_lemmatized)
        resultados.append(
            PlagiarismChecker.generar_reporte(
                texto, similitud_unigrama_lemmatized))
        resultados_ordenados = sorted(
            resultados, key=lambda x: x[1], reverse=True)

    return resultados_ordenados[0][2], resultados_ordenados[1][2]


def evaluar_sospechosos():
    '''Hace la evaluación del protocolo calculando AUC.
    '''
    # Contadores
    tp_cont = 0
    tn_cont = 0
    fp_cont = 0
    fn_cont = 0

    for sospechoso in lista_textos_sospechoso:
        lematizado = lectura_y_preprocesamiento_texto(
            f"sospechosos/{sospechoso}")
        tabla = comparar_textos(lematizado)
        # Identificar los que se saben que son true positive
        es_tp = 'TP' in os.path.basename(sospechoso)

        for es_plagio in tabla:
            if es_tp:
                if es_plagio:
                    tp_cont += 1
                else:
                    fn_cont += 1
            else:
                if es_plagio:
                    fp_cont += 1
                else:
                    tn_cont += 1

    # Aplicar formulas
    if (tp_cont + fn_cont) == 0:
        tpr = 0
    else:
        tpr = tp_cont / (tp_cont + fn_cont)

    if (fp_cont + tn_cont) == 0:
        fpr = 0
    else:
        fpr = fp_cont / (fp_cont + tn_cont)

    resultados = {
        'Verdaderos Positivos': tp_cont,
        'Verdaderos Negativos': tn_cont,
        'Falsos Positivos': fp_cont,
        'Falsos Negativos': fn_cont
    }

    # Calcular area bajo la curva
    auc = (1 + tpr - fpr) / 2

    print(f'Resultados: {resultados}')
    print(f'AUC: {round(auc, 4)}')


evaluar_sospechosos()
