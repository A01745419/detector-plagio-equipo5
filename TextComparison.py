# José Luis Madrigal Sánchez A01745419
# Paulo Ogando Gulías A01751587
# César Emiliano Palome Luna A01746493
# Creado 24/05/2024

from PlagiarismChecker import PlagiarismChecker
import os

# Archvivos dentro de las carpetas
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
            PlagiarismChecker.vectorizacion(plagio_lemmatized,
                                            original_lemmatized, 1)
        similitud_unigrama_lemmatized = \
            PlagiarismChecker.calcular_similitud(
                vector_unigrama_lemmatized)
        # Ordenar por porcentaje
        resultados.append(
            PlagiarismChecker.generar_reporte
            (texto, similitud_unigrama_lemmatized))
        resultados_ordenados = sorted(resultados,
                                      key=lambda x: x[1], reverse=True)

    # Mostrar los 2 archvios con mayor similitud
    # si existen archivos plagiados
    if resultados_ordenados[0][2] is False:
        print('               Este texto es genuino')
    else:
        print(f'{resultados_ordenados[0][0]} |\
            {resultados_ordenados[0][1]}%        |\
            {resultados_ordenados[0][2]}')
        print(f'{resultados_ordenados[1][0]} |\
            {resultados_ordenados[1][1]}%        |\
            {resultados_ordenados[1][2]}')


print("Elija una opción para comparar:")
print("1. Comparar 1 texto sospechoso con todos los textos originales")
print("2. Comparar un grupo de textos sospechosos con \
      todos los textos originales")
print("3. Comparar todos los textos sospechosos todos los textos originales")
opcion = int(input("Opción: "))
print("")
# Elegir un texto en específico
if opcion == 1:
    print("Textos sospechosos:")
    for i in range(len(lista_textos_sospechoso)):
        print(f"{i+1}. {lista_textos_sospechoso[i]}")
    print("")
    texto_sospechoso = int(input("Elija el numero del texto sospechoso: "))
    plagio_lemmatized = lectura_y_preprocesamiento_texto(
        f"sospechosos/{lista_textos_sospechoso[texto_sospechoso-1]}")
    print("")
    print(f'Texto a comparar: {lista_textos_sospechoso[texto_sospechoso-1]}')
    print(f'   Texto    |  % Unigrama lematizado |       Plagio')
    tabla = comparar_textos(plagio_lemmatized)

# Elegir un grupo de textos
elif opcion == 2:
    print("Textos sospechosos:")
    for i in range(len(lista_textos_sospechoso)):
        print(f"{i+1}. {lista_textos_sospechoso[i]}")
    print("")
    textos_sospechosos = input(
        "Elija los numeros de los textos sospechosos separados por coma: ")
    textos_sospechosos = textos_sospechosos.split(",")
    print("")
    print(f'   Texto    |  % Unigrama lematizado |       Plagio')
    for texto_sospechoso in textos_sospechosos:
        plagio_lemmatized = lectura_y_preprocesamiento_texto(
            f"sospechosos/{lista_textos_sospechoso[int(texto_sospechoso)-1]}")
        print(
            f'Texto a comparar: \
                {lista_textos_sospechoso[int(texto_sospechoso)-1]}')
        tabla = comparar_textos(plagio_lemmatized)

# Elegir todos los textos
elif opcion == 3:
    print(f'   Texto    |  % Unigrama lematizado |       Plagio')
    for texto_sospechoso in lista_textos_sospechoso:
        plagio_lemmatized = lectura_y_preprocesamiento_texto(
            f"sospechosos/{texto_sospechoso}")
        print(f'Texto a comparar: {texto_sospechoso}')
        tabla = comparar_textos(plagio_lemmatized)
