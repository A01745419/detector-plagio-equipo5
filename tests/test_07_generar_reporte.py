# File: tests/test_07_generar_reporte.py

from unittest import TestCase
from PlagiarismChecker import PlagiarismChecker

class TestCleanParagraph(TestCase):

    def setUp(self):
        self.pc = PlagiarismChecker()

    def test_generar_reporte_no_plagio(self):
        similitud = 40.1516
        reporte = self.pc.generar_reporte("no plagio", similitud)
        self.assertEqual(reporte, None)
    
    def test_generar_reporte_plagio(self):
        similitud = 87.5915
        reporte = self.pc.generar_reporte("plagio", similitud)
        self.assertEqual(reporte, \
                         (f"plagio  |        {87.5915}         | {True}"))