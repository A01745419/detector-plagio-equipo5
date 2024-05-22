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
# Párrafo similar que habla del mismo tema
parrafo2 ="""La historia de cómo se creó el personaje de Batman es una de las más fascinantes de los cómics. Originalmente creado por Bob Kane de forma muy somera lo que después trajo polémica por su autoría, no fue hasta que su colaborador Bill Finger llegó y desarrolló el concepto completo de lo que hoy entendemos como el personaje. Una de las contribuciones más significativas al mito de Batman hecha por Finger fue el propio nombre del personaje, Bruce Wayne."""
# Primer párrafo parafraseado (plagio)
parafraseado1 = """La historia del origen de Batman es casi tan interesante en la vida real como lo es en los cómics. Batman nació de la idea original de Bob Kane, que tiene un solo nombre, y su frecuente colaborador Bill Finger, quien diseñó el traje de Batman y la historia de Bruce Wayne. Por cierto, este nombre no es casual."""


def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

parrafo_limpio1 = remove_punctuation(parrafo1)
parrafo_limpio2 = remove_punctuation(parrafo2)
parrafo_limpio3 = remove_punctuation(parafraseado1)

# Utilizacion de Stemming
def stem_sentence(sentence):
    ps = PorterStemmer()
    palabras = word_tokenize(sentence)
    stemmed_tokens = [ps.stem(palabra) for palabra in palabras]
    return ' '.join(stemmed_tokens)

# Aplicar stemming
stemmed_sentence1 = stem_sentence(parrafo_limpio1)
stemmed_sentence2 = stem_sentence(parrafo_limpio2)
stemmed_sentence3 = stem_sentence(parrafo_limpio3)

# Ngramas y similitud
def calculate_similarity(tokens1, tokens2, n):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(n, n))
    ngrams = vectorizer.fit_transform([tokens1, tokens2])
    print(f"Corpus {n}grama: {vectorizer.get_feature_names_out()}")
    print("------------------------------------")
    print(f"Vector {n}grama: {ngrams.toarray()}")
    print("------------------------------------")
    similarity = cosine_similarity(ngrams)[0, 1]
    return similarity

# Similitudes entre párrafos con tema similar
unigram_similarity = calculate_similarity(stemmed_sentence1, stemmed_sentence2, 1)
bigram_similarity = calculate_similarity(stemmed_sentence1, stemmed_sentence2, 2)
trigram_similarity = calculate_similarity(stemmed_sentence1, stemmed_sentence2, 3)

# Similitudes entre original y parafraseado
unigram_similarity_paraphrasing = calculate_similarity(stemmed_sentence1, stemmed_sentence3, 1)
bigram_similarity_paraphrasing = calculate_similarity(stemmed_sentence1, stemmed_sentence3, 2)
trigram_similarity_paraphrasing = calculate_similarity(stemmed_sentence1, stemmed_sentence3, 3)

print(f"Unigram similarity between paragraphs with same topic: {unigram_similarity}")
print(f"Bigram similarity between paragraphs with same topic: {bigram_similarity}")
print(f"Trigram similarity between paragraphs with same topic: {trigram_similarity}")

print(f"Unigram similarity between original and paraphrased: {unigram_similarity_paraphrasing}")
print(f"Bigram similarity between original and paraphrased: {bigram_similarity_paraphrasing}")
print(f"Trigram similarity between original and paraphrased: {trigram_similarity_paraphrasing}")