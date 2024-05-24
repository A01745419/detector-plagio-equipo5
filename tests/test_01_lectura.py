# File: tests/test_01_lectura.py

from unittest import TestCase
from PlagiarismChecker import PlagiarismChecker

class TestReadArchive(TestCase):

    def setUp(self):
        self.pc = PlagiarismChecker()

    def test_read_archive(self):
        info = self.pc.lectura('tests/tests.txt')
        self.assertEqual(
            info,
            'This is a test for the plagiarism checker.\n')
