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

import pandas as pd
from pandas.testing import assert_frame_equal
from trajectories_outliers_contacts import extract_bed, extract_roi, extract_contacts_with_atoms_distance, \
    outliers_contacts, update_regions

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILES_DIR = os.path.join(TEST_DIR, "test_files")
BIN_DIR = os.path.dirname(TEST_DIR)
sys.path.append(BIN_DIR)


class TestTrajectoriesOutliersContacts(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = os.path.join(tempfile.gettempdir(), "tmp_tests_trajectories")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.format_output = "svg"
        self.atoms_dist = 3.0
        self.residues_dist = 10
        self.roi = "682-838"
        self.col_distance = "whole_MD_median_distance"
        self.path_bed = os.path.join(TEST_FILES_DIR, "HEPAC-26_RPL6_ORF1.bed")
        self.input_contacts = os.path.join(TEST_FILES_DIR, "contacts_HEPAC-26_RPL6_ORF1_0_MD_0_to_30M.csv")
        self.extracted_bed = pd.read_csv(os.path.join(TEST_FILES_DIR, "extracted_bed_HEPAC-26_RPL6_ORF1.csv"),
                                         sep=",", index_col=0)
        self.expected_filtered_contacts = pd.read_csv(os.path.join(TEST_FILES_DIR,
                                                                   "filtered_contacts_HEPAC-26_RPL6_ORF1_0.csv"),
                                                      sep=",", index_col=0)
        self.expected_outliers_contacts = pd.read_csv(os.path.join(TEST_FILES_DIR,
                                                                   "outliers_contacts_HEPAC-26_RPL6_ORF1_0.csv"),
                                                      sep=",", index_col=0)
        self.expected_regions_contacts = pd.read_csv(os.path.join(TEST_FILES_DIR,
                                                                  "outliers_regions_contacts_HEPAC-26_RPL6_ORF1_0.csv"),
                                                     sep=",", index_col=0)

    def tearDown(self):
        # Clean temporary files
        shutil.rmtree(self.tmp_dir)

    def test_extract_bed(self):
        assert_frame_equal(self.extracted_bed, extract_bed(self.path_bed))

    def test_extract_roi(self):
        self.assertRaises(argparse.ArgumentTypeError, extract_roi, "682_838")
        self.assertRaises(argparse.ArgumentTypeError, extract_roi, "422")
        self.assertRaises(argparse.ArgumentTypeError, extract_roi, "toto")
        observed = extract_roi(self.roi)
        self.assertListEqual([682, 838], observed)

    def test_extract_contacts_with_atoms_distance(self):
        observed = extract_contacts_with_atoms_distance(self.input_contacts, self.roi, self.col_distance,
                                                        self.atoms_dist)
        assert_frame_equal(self.expected_filtered_contacts, observed)
        # with self.assertRaises(argparse.ArgumentTypeError):
        #     extract_contacts_with_atoms_distance(raw_contacts, "422", self.col_distance, self.atoms_dist)
        # with self.assertRaises(KeyError):
        #     extract_contacts_with_atoms_distance(raw_contacts, "682-838", "toto", self.atoms_dist)
        # with self.assertRaises(TypeError):
        #     extract_contacts_with_atoms_distance(raw_contacts, "682-838", "donor residue", self.atoms_dist)

    def test_outliers_contacts(self):
        observed = outliers_contacts(self.expected_filtered_contacts, self.residues_dist, self.col_distance)
        assert_frame_equal(self.expected_outliers_contacts, observed)

    def test_update_regions(self):
        observed = update_regions(self.expected_outliers_contacts, self.extracted_bed)
        assert_frame_equal(self.expected_regions_contacts, observed)


if __name__ == '__main__':
    unittest.main()
