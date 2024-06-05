# File: tests/test_07_generar_reporte.py

from unittest import TestCase
from PlagiarismChecker import IntelligentPlagiarismChecker

class TestCleanParagraph(TestCase):

    def setUp(self):
        self.pc = IntelligentPlagiarismChecker()

    def test_generar_reporte_no_plagio(self):
        similitud = 40.1516
        reporte = self.pc.generar_reporte("no plagio", similitud)
        self.assertEqual(reporte, ['no plagio', 40.1516, False])
    
    def test_generar_reporte_plagio(self):
        similitud = 87.5915
        reporte = self.pc.generar_reporte("plagio", similitud)
        self.assertEqual(reporte, ['plagio', 87.5915, True])
