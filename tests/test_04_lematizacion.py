# File: tests/test_04_lematizacion.py

from unittest import TestCase
from PlagiarismChecker import IntelligentPlagiarismChecker

class TestLematizacion(TestCase):

    def setUp(self):
        self.pc = IntelligentPlagiarismChecker()

    def test_lematizacion_articulo_verbo(self):
        texto_limpio = 'la mentira es mala'
        texto_lematizado = self.pc.lematizacion(texto_limpio)
        self.assertEqual(
            texto_lematizado,
            'el mentira ser malo')

    def test_lematizacion_con_plural(self):
        texto_limpio = 'muchas personas con diversas tareas'
        texto_lematizado = self.pc.lematizacion(texto_limpio)
        self.assertEqual(
            texto_lematizado,
            'mucho persona con diverso tarea')

    def test_lematizacion_tiempos_verbales(self):
        texto_limpio = 'juegan jugaron jugarán jugando'
        texto_lematizado = self.pc.lematizacion(texto_limpio)
        self.assertEqual(
            texto_lematizado,
            'jugar jugar jugar jugar')
