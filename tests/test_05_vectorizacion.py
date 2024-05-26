# File: tests/test_05_vectorizacion.py

from unittest import TestCase
from PlagiarismChecker import PlagiarismChecker

class TestCleanParagraph(TestCase):

    def setUp(self):
        self.pc = PlagiarismChecker()

    def test_limpiar_parrafo(self):
        tokens1 = "La lluvia estaba tan fuerte que volvimos"
        tokens2 = "Es mejor estar en casa y ver unas películas"
        vector = self.pc.vectorizacion(tokens1, tokens2, 1)
        vector_array = vector.toarray()
        vector_string = "\n".join(" ".join(map(str, fila)) for fila in vector_array)
        resultado_correcto = "0 0 0 1 0 1 1 1 0 0 1 1 0 0 1\n1 1 1 0 1 0 0 0 1 1 0 0 1 1 0"
        self.assertEqual(vector_string, resultado_correcto)