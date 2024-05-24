# File: tests/test_03_stemming.py

from unittest import TestCase
from PlagiarismChecker import PlagiarismChecker

class TestStemming(TestCase):

    def setUp(self):
        self.pc = PlagiarismChecker()

    def test_stemming(self):
        texto = 'This is a test for the plagiarism checker.\n'
        texto_limpio = self.pc.limpieza(texto)
        texto_stemming = self.pc.stemming(texto_limpio)
        self.assertEqual(
            texto_stemming,
            'thi is a test for the plagiar checker')
