# File: tests/test_02_limpieza.py

from unittest import TestCase
from PlagiarismChecker import PlagiarismChecker

class TestCleanParagraph(TestCase):

    def setUp(self):
        self.pc = PlagiarismChecker()

    def test_clean_paragraph(self):
        texto = 'This is a test for the plagiarism checker.\n'
        texto_limpio = self.pc.limpieza(texto)
        self.assertEqual(
            texto_limpio,
            'This is a test for the plagiarism checker\n')
