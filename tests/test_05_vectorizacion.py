# File: tests/test_05_vectorizacion.py

from unittest import TestCase
from PlagiarismChecker import IntelligentPlagiarismChecker

class TestCleanParagraph(TestCase):

    def setUp(self):
        self.pc = IntelligentPlagiarismChecker()

    def test_vectorizacion(self):
        tokens1 = "el lluvia estar tanto fuerte que volver"
        tokens2 = "ser mejor estar en casa y ver uno película"
        vector = self.pc.vectorizacion(tokens1, tokens2, 1)
        vector_array = vector.toarray()
        vector_string = "\n".join(" ".join(map(str, fila)) for fila in vector_array)
        resultado_correcto = "0 1 0 1 1 1 0 0 1 0 1 0 0 1\n1 0 1 1 0 0 1 1 0 1 0 1 1 0"
        self.assertEqual(vector_string, resultado_correcto)
