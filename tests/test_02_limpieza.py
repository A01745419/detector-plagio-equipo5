# File: tests/test_02_limpieza.py


from unittest import TestCase

from PlagiarismChecker import PlagiarismChecker


class TestCleanParagraph(TestCase):


    def setUp(self):

        self.pc = PlagiarismChecker()


    def test_limpiar_parrafo(self):

        texto = 'La historia de, Batman es/ buena. Tiene #buenos cómics.'

        texto_limpio = self.pc.limpieza(texto)

        self.assertEqual(

            texto_limpio,

            'La historia de Batman es buena Tiene buenos cómics')
