# File: tests/test_01_lectura.py


from unittest import TestCase

from PlagiarismChecker import PlagiarismChecker


class TestReadArchive(TestCase):


    def setUp(self):

        self.pc = PlagiarismChecker()


    def test_leer_archivo(self):

        info = self.pc.lectura('tests/tests.txt')

        self.assertEqual(

            info,

            'Este es un ejemplo de un archivo txt. Incluye acentos y símbolos especiales $.')
