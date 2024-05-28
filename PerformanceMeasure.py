from PlagiarismChecker import PlagiarismChecker
import os

lista_textos_originales = os.listdir("originales")
lista_textos_sospechoso = os.listdir("sospechosos")

#Instancia de la clase PlagiarismChecker
PlagiarismChecker = PlagiarismChecker()

def lectura_y_preprocesamiento_texto(texto):
    #Lectura del texto a comprobar su plagio
    parrafo_plagio = PlagiarismChecker.lectura(texto)
    #Limpieza de texto
    plagio_limpio = PlagiarismChecker.limpieza(parrafo_plagio)
    #Preprocesamiento de texto
    plagio_lemmatized = PlagiarismChecker.lematizacion(plagio_limpio)

    return plagio_lemmatized

def comparar_textos(plagio_lemmatized):
    resultados = []
    for texto in lista_textos_originales:
        original_lemmatized = lectura_y_preprocesamiento_texto(f"originales/{texto}")
        vector_unigrama_lemmatized = PlagiarismChecker.vectorizacion(plagio_lemmatized, original_lemmatized, 1)
        similitud_unigrama_lemmatized = PlagiarismChecker.calcular_similitud(vector_unigrama_lemmatized)
        resultados.append(PlagiarismChecker.generar_reporte(texto, similitud_unigrama_lemmatized))
        resultados_ordenados = sorted(resultados, key=lambda x: x[1], reverse=True)

    print(f'{resultados_ordenados[0][0]} |        {resultados_ordenados[0][1]}%        |  {resultados_ordenados[0][2]}')
    print(f'{resultados_ordenados[1][0]} |        {resultados_ordenados[1][1]}%        |  {resultados_ordenados[1][2]}')
    return resultados_ordenados[0][2], resultados_ordenados[1][2]

def evaluar_sospechosos():
        tp_count = 0
        tn_count = 0
        fp_count = 0
        fn_count = 0

        carpeta_sospechosos = os.listdir("sospechosos")

        for sospechoso in carpeta_sospechosos:
            lematizado = lectura_y_preprocesamiento_texto(f"sospechosos/{sospechoso}")
            tabla = comparar_textos(lematizado)
            es_tp = 'TP' in os.path.basename(sospechoso)

            for es_plagio in tabla:
                if es_tp:
                    if es_plagio:
                        tp_count += 1
                    else:
                        fn_count += 1
                else:
                    if es_plagio:
                        fp_count += 1
                    else:
                        tn_count += 1

        resultados = {
            'True Positive': tp_count,
            'True Negative': tn_count,
            'False Positive': fp_count,
            'False Negative': fn_count
        }

        print(resultados)

        if (tp_count + fn_count) == 0:
            tpr = 0
        else:
            tpr = tp_count / (tp_count + fn_count)
    
        if (fp_count + tn_count) == 0:
         fpr = 0
        else:
            fpr = fp_count / (fp_count + tn_count)
    
        auc = (1 + tpr - fpr)/2

        print (f'AUC: {auc}')

        return auc

# print(f'   Texto    |  % Unigrama lematizado | Plagio')
# for texto_sospechoso in lista_textos_sospechoso:
#     plagio_lemmatized = lectura_y_preprocesamiento_texto(f"sospechosos/{texto_sospechoso}")
#     print(f'Texto a comparar: {texto_sospechoso}')
#     tabla = comparar_textos(plagio_lemmatized)
#     for i in tabla:
#         print(i)

# is_tp = 'TP' in os.path.basename("FID-001-TP.txt")
# print(is_tp)