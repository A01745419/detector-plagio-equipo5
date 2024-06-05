# File: tests/test_03_stemming.py


from unittest import TestCase

from PlagiarismChecker import IntelligentPlagiarismChecker


class TestStemming(TestCase):


    def setUp(self):

        self.pc = IntelligentPlagiarismChecker()


    def test_stemming_variaciones_sufijos(self):

        texto_limpio = 'amare amoroso amante amorío'

        texto_stemming = self.pc.stemming(texto_limpio)

        self.assertEqual(

            texto_stemming,
            'amar amor amant amori')


    def test_stemming_tiempos_verbales(self):

        texto_limpio = 'compre comprando compraban compraré'

        texto_stemming = self.pc.stemming(texto_limpio)

        self.assertEqual(

            texto_stemming,
            'compr compr compr compr')


    def test_stemming_plural(self):

        texto_limpio = 'muchas personas inteligentes'

        texto_stemming = self.pc.stemming(texto_limpio)

        self.assertEqual(

            texto_stemming,

            'much person inteligent')
