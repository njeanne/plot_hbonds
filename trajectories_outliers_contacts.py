#!/usr/bin/env python3

"""
Created on 12 Sep. 2022
"""

__author__ = "Nicolas JEANNE"
__copyright__ = "GNU General Public License"
__email__ = "jeanne.n@chu-toulouse.fr"
__version__ = "1.0.0"

import argparse
import logging
import os
import re
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns


def create_log(log_path, level, out_dir):
    """Create the log as a text file and as a stream.

    :param log_path: the path of the log.
    :type log_path: str
    :param level: the level og the log.
    :type level: str
    :param out_dir: the results directory path.
    :type out_dir: str
    """
    os.makedirs(out_dir, exist_ok=True)
    if not log_path:
        log_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
    log_level_dict = {"DEBUG": logging.DEBUG,
                      "INFO": logging.INFO,
                      "WARNING": logging.WARNING,
                      "ERROR": logging.ERROR,
                      "CRITICAL": logging.CRITICAL}

    log_level = log_level_dict["INFO"]
    if level:
        log_level = log_level_dict[args.log_level]

    if os.path.exists(log_path):
        os.remove(log_path)

    logging.basicConfig(format="%(asctime)s %(levelname)s:\t%(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=log_level,
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logging.info(f"version: {__version__}")
    logging.info(f"CMD: {' '.join(sys.argv)}")


def extract_roi(roi_to_extract):
    """Extract the start and stop coordinates of the region of interest (roi).

    :param roi_to_extract: the coordinates ot the region of interest, as 100-200 i.e.
    :type roi_to_extract: str
    :raises ArgumentTypeError: is not between 0.0 and 100.0
    :return: the start and stop coordinates of the region of interest.
    :rtype: list
    """
    pattern_roi_to_extract = re.compile("(\\d+)-(\\d+)")
    match_roi_to_extract = pattern_roi_to_extract.search(roi_to_extract)
    if match_roi_to_extract:
        roi_extracted = [int(match_roi_to_extract.group(1)), int(match_roi_to_extract.group(2))]
    else:
        raise argparse.ArgumentTypeError(f"'{roi_to_extract}' argument is malformed, it should be two integers "
                                         f"separated by an hyphen, i.e: '100-200'.")
    return roi_extracted


def extract_contacts_with_atoms_distance(path_contacts, roi_str, col, thr):
    """
    Extract the contacts from the CSV file residues contacts and filter on the donor region of interest (if any) and on
    the threshold of the contacts distance column.

    :param path_contacts: the path to the contacts CSV path.
    :type path_contacts: str
    :param roi_str: the string defining the donors residues region of interest, the format should be i.e: '100-200'.
    :type roi_str: str
    :param col: the name of the column to use for the atoms contacts between two residues.
    :type col: str
    :param thr: the maximal atoms distance threshold between two residues.
    :type thr: float
    :return: the filtered contacts.
    :rtype: pd.Dataframe
    """
    # load the contacts file
    df = pd.read_csv(path_contacts, sep=",")
    logging.debug(f"{len(df)} contacts for the raw data.")
    # select the donors region of interest if any
    roi_text = ""
    if roi_str:
        try:
            roi = extract_roi(roi_str)
            # reduce to the donors region of interest limits
            df = df[df["donor position"].between(roi[0], roi[1])]
            logging.debug(f"{len(df)} contacts in the region of interest.")
            roi_text = f" in the donors region of interest {roi_str}"
        except argparse.ArgumentTypeError as exc:
            logging.error(exc, exc_info=True)
            sys.exit(1)
    try:
        df = df[df[col] <= thr]
        logging.debug(f"{len(df)} contacts for under the distance threshold of {thr} \u212B in column \"{col}\".")
    except KeyError as exc:
        logging.error(f"\"{col}\" column name does not exists in the CSV file.", exc_info=True)
        sys.exit(1)
    except TypeError as exc:
        logging.error(f"\"{col}\" column name may refers to a column that is not an atoms distances.", exc_info=True)
        sys.exit(1)
    logging.info(f"{len(df)} contacts detected{roi_text} for a maximal contacts distance of {thr} \u212B on column "
                 f"\"{col.replace('_', ' ')}\".")
    return df


def outliers_contacts(df, thr, col):
    """"""
    idx_to_remove = []
    for idx, row in df.iterrows():
        if abs(row["donor position"] - row["acceptor position"]) < thr:
            idx_to_remove.append(idx)
    # remove rows with too close distance between the residues
    df.drop(idx_to_remove, inplace=True, axis=0)
    # reduce to one row when more than one atom per residue
    return df


if __name__ == "__main__":
    descr = f"""
    {os.path.basename(__file__)} v. {__version__}

    Created by {__author__}.
    Contact: {__email__}
    {__copyright__}

    Distributed on an "AS IS" basis without warranties or conditions of any kind, either express or implied.

    From a CSV of the amino acids contacts file produced by the script trajectories
    (https://github.com/njeanne/trajectories), extract the information of the contacts outside the neighbourhood 
    contacts of an amino acid and determine to which region the acceptor amino acid belongs to using a BED file 
    registering the domains.
    """
    parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", required=True, type=str, help="the path to the output directory.")
    parser.add_argument("-b", "--bed", required=True, type=str,
                        help="the path to the BED file registering the amino acids regions positions.")
    parser.add_argument("-c", "--col-distance", required=True, type=str,
                        help="the name of the column to use for the atoms distances.")
    parser.add_argument("-i", "--roi", required=False, type=str,
                        help="the donors region of interest coordinates, the format should be two digits separated by "
                             "an hyphen, i.e: '100-200'.")
    parser.add_argument("-f", "--format", required=False, default="svg",
                        choices=["eps", "jpg", "jpeg", "pdf", "pgf", "png", "ps", "raw", "svg", "svgz", "tif", "tiff"],
                        help="the output plots format: 'eps': 'Encapsulated Postscript', "
                             "'jpg': 'Joint Photographic Experts Group', 'jpeg': 'Joint Photographic Experts Group', "
                             "'pdf': 'Portable Document Format', 'pgf': 'PGF code for LaTeX', "
                             "'png': 'Portable Network Graphics', 'ps': 'Postscript', 'raw': 'Raw RGBA bitmap', "
                             "'rgba': 'Raw RGBA bitmap', 'svg': 'Scalable Vector Graphics', "
                             "'svgz': 'Scalable Vector Graphics', 'tif': 'Tagged Image File Format', "
                             "'tiff': 'Tagged Image File Format'. Default is 'svg'.")
    parser.add_argument("-a", "--atoms-distance", required=False, type=float, default=3.0,
                        help="the atoms contacts distances threshold between two residues, default is 3.0 Angstroms.")
    parser.add_argument("-r", "--residues-distance", required=False, type=int, default=10,
                        help="when 2 atoms of different residues are in contact, the minimal distance in number of "
                             "residues that should separate them for a long range interaction, default is 10.")
    parser.add_argument("-l", "--log", required=False, type=str,
                        help="the path for the log file. If this option is skipped, the log file is created in the "
                             "output directory.")
    parser.add_argument("--log-level", required=False, type=str,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="set the log level. If the option is skipped, log level is INFO.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("input", type=str, help="the donors/acceptors contacts CSV file.")
    args = parser.parse_args()

    # create the logger
    create_log(args.log, args.log_level, args.out)

    # load the BED file
    bed = pd.read_csv(args.bed, sep="\t", header=None, names=["id", "start", "stop", "region"])
    print(bed)

    contacts = extract_contacts_with_atoms_distance(args.input, args.roi, args.col_distance, args.atoms_distance)
    print(contacts)
    contacts.to_csv(os.path.join(args.out, "filtered_contacts_HEPAC-26_RPL6_ORF1_0.csv"))

    outliers = outliers_contacts(contacts, args.residues_distance, args.col_distance)
    print(outliers)



