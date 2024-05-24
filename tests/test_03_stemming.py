# File: tests/test_03_stemming.py

from unittest import TestCase
from PlagiarismChecker import PlagiarismChecker

class TestStemming(TestCase):

    def setUp(self):
        self.pc = PlagiarismChecker()

    def test_stemming(self):
        texto = 'This is a test for the plagiarism checker.'
        texto_limpio = self.pc.limpieza(texto)
        texto_stemming = self.pc.stemming(texto_limpio)
        self.assertEqual(
            texto_stemming,
            'thi is a test for the plagiar checker')

    def test_stemming_with_2_sentences(self):
        texto = 'This is a test for the plagiarism checker. This is a second sentence.'
        texto_limpio = self.pc.limpieza(texto)
        texto_stemming = self.pc.stemming(texto_limpio)
        self.assertEqual(
            texto_stemming,
            'thi is a test for the plagiar checker thi is a second sentenc')

    def test_stemming_with_3_sentences(self):
        texto = 'This is a test for the plagiarism checker. This is a second sentence.\n This is a third sentence.'
        texto_limpio = self.pc.limpieza(texto)
        texto_stemming = self.pc.stemming(texto_limpio)
        self.assertEqual(
            texto_stemming,
            'thi is a test for the plagiar checker thi is a second sentenc thi is a third sentenc')
