# File: tests/test_06_calcular_similitud.py

from unittest import TestCase
from PlagiarismChecker import PlagiarismChecker

class TestCleanParagraph(TestCase):

    def setUp(self):
        self.pc = PlagiarismChecker()

    def test_calcular_similitud(self):
        tokens1 = "el lluvia estar tanto fuerte que volver"
        tokens2 = "ser mejor estar en casa y ver uno película"
        vector = self.pc.vectorizacion(tokens1, tokens2, 1)
        similitud = self.pc.calcular_similitud(vector)
        self.assertEqual(similitud, 13.3631)
