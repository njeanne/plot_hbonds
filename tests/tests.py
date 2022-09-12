#!/usr/bin/env python3

__author__ = 'Nicolas Jeanne'
__license__ = 'GNU General Public License'
__version__ = '1.0.0'
__email__ = 'jeanne.n@chu-toulouse.fr'

import argparse
import os
import re
import shutil
import sys
import tempfile
import unittest
import uuid

from trajectories_contacts_regions import extract_roi

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILES_DIR = os.path.join(TEST_DIR, "test_files")
BIN_DIR = os.path.dirname(TEST_DIR)
sys.path.append(BIN_DIR)


class TestTrajectoriesContactsRegions(unittest.TestCase):

    def setUp(self):
        system_tmp_dir = tempfile.gettempdir()
        self.tmp_dir = os.path.join(system_tmp_dir, "tmp_tests_trajectories")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.format_output = "svg"
        self.atoms_dist = 3.0
        self.residues_dist = 10

    def tearDown(self):
        # Clean temporary files
        shutil.rmtree(self.tmp_dir)

    def test_extract_roi(self):
        self.assertRaises(argparse.ArgumentTypeError, extract_roi, "682_838")
        self.assertRaises(argparse.ArgumentTypeError, extract_roi, "422")


if __name__ == '__main__':
    unittest.main()
