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


def extract_bed(bed_path):
    """
    Load the BED file and fill in regions that are not covered.

    :param bed_path: the path to the BED file.
    :type bed_path: str
    :return: the filled-in BED.
    :rtype: pd.Dataframe
    """
    raw = pd.read_csv(bed_path, sep="\t", header=None, names=["id", "start", "stop", "region"])
    data = {"id": [], "start": [], "stop": [], "region": []}
    pos_start = 1
    previous_region = None
    for idx, row in raw.iterrows():
        if pos_start < row["start"]:  # not in region
            # record the undefined region
            data["id"].append(row["id"])
            data["start"].append(pos_start)
            data["stop"].append(row["start"] - 1)
            if idx == 0:  # before first region
                data["region"].append(f"before {row['region']}")
            else:
                data["region"].append(f"between {previous_region} and {row['region']}")
        # record the region in the BED
        data["id"].append(row["id"])
        data["start"].append(row["start"])
        data["stop"].append(row["stop"])
        data["region"].append(row["region"])
        pos_start = row["stop"] + 1
        previous_region = row["region"]

    df = pd.DataFrame(data)
    return df


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
    logging.info(f"{len(df)} atoms contacts in the input data (residues pairs may have multiple atoms contacts).")
    # select the donors region of interest if any
    roi_text = ""
    if roi_str:
        roi = extract_roi(roi_str)
        # reduce to the donors region of interest limits
        df = df[df["donor position"].between(roi[0], roi[1])]
        logging.debug(f"{len(df)} atoms contacts in the region of interest.")
        roi_text = f" in the donors region of interest {roi_str}"

    df = df[df[col] <= thr]
    logging.debug(f"{len(df)} atoms contacts detected{roi_text} for a maximal contacts distance of {thr} \u212B on "
                  f"column \"{col.replace('_', ' ')}\".")
    return df


def unique_residues_pairs(df_not_unique, col):
    """
    Get the unique residues pairs contacts.

    :param df_not_unique: the dataframe of contacts.
    :type df_not_unique: pd.Dataframe
    :param col: the atoms distances column name.
    :type col: str
    :return: the dataframe of unique residues pairs contacts.
    :rtype: pd.Dataframe
    """
    donor_positions = []
    donor_residues = []
    acceptor_positions = []
    acceptor_residues = []
    nb_contacts = []
    max_atoms_dist = []
    contacts_ids = []
    for donor_position in sorted(list(set(df_not_unique["donor position"]))):
        df_donors = df_not_unique[df_not_unique["donor position"] == donor_position]
        for acceptor_position in sorted(list(set(df_donors["acceptor position"]))):
            df_acceptors = df_donors[df_donors["acceptor position"] == acceptor_position]
            donor_positions.append(donor_position)
            donor_residues.append(df_donors["donor residue"].iloc[0])
            acceptor_positions.append(acceptor_position)
            acceptor_residues.append(df_acceptors["acceptor residue"].iloc[0])
            nb_contacts.append(len(df_acceptors))
            max_atoms_dist.append(max(df_acceptors[col]))
            pair_contacts_ids = []
            for _, row in df_acceptors.iterrows():
                pair_contacts_ids.append(row["contact"])
            contacts_ids.append(" | ".join(pair_contacts_ids))

    df_uniques = pd.DataFrame.from_dict({"donor positions": donor_positions,
                                         "donor residues": donor_residues,
                                         "acceptor positions": acceptor_positions,
                                         "acceptor residues": acceptor_residues,
                                         "atoms contacts": nb_contacts,
                                         "maximum atoms distance": max_atoms_dist,
                                         "contacts ID": contacts_ids})
    return df_uniques


def outliers_contacts(df, res_dist_thr, col_dist):
    """
    Get the residues pairs contacts above the residues distance contacts, meaning contacts of distant residues.

    :param df: the dataframe of contacts.
    :type df: pd.Dataframe
    :param res_dist_thr: the residues distance threshold.
    :type res_dist_thr: int
    :param col_dist: the atoms distances column name.
    :type col_dist: str
    :return: the dataframe of unique residues pairs contacts.
    :rtype: pd.Dataframe
    """
    idx_to_remove_for_residue_distance = []
    for idx, row in df.iterrows():
        if abs(row["donor position"] - row["acceptor position"]) < res_dist_thr:
            idx_to_remove_for_residue_distance.append(idx)
    # remove rows with too close distance between the residues
    df.drop(idx_to_remove_for_residue_distance, inplace=True, axis=0)
    logging.debug(f"{len(df)} atoms contacts remaining with a minimal residues distance threshold of {res_dist_thr}.")
    # reduce to one row when more than one atom per residue
    unique = unique_residues_pairs(df, col_dist)
    return unique


def update_regions(df, bed, path):
    """
    Get the regions for pairs acceptor and donor.

    :param df: the dataframe of unique residues pairs contacts.
    :type df: pd.Dataframe
    :param bed: the regions in BED format.
    :type bed: pd.Dataframe
    :param path: the path of the outliers contacts updated with the regions CSV file.
    :type path: str
    :return: the pairs contacts dataframe updated with the regions.
    :rtype: pd.Dataframe
    """
    donors_regions = [None] * len(df)
    acceptors_regions = [None] * len(df)
    for idx, row_contacts in df.iterrows():
        for _, row_bed in bed.iterrows():
            if row_bed["start"] <= row_contacts["donor positions"] <= row_bed["stop"]:
                donors_regions[idx] = row_bed["region"]
            if row_bed["start"] <= row_contacts["acceptor positions"] <= row_bed["stop"]:
                acceptors_regions[idx] = row_bed["region"]
    df.insert(2, "donor regions", pd.DataFrame(donors_regions))
    df.insert(5, "acceptor regions", pd.DataFrame(acceptors_regions))
    df.to_csv(path, index=False)
    logging.info(f"Pairs residues contacts updated with regions saved: {path}")
    return df


def acceptors_regions_involved(df, bed, path, roi, atoms_dist, res_dist):
    """
    Create the plot of contacts by regions.

    :param df: the dataframe.
    :type df: pd.Dataframe
    :param bed: the regions BED file.
    :type bed: pd.Dataframe
    :param path: the path of the output plot.
    :type path: str
    :param roi: the region of interest.
    :type roi: str
    :param atoms_dist: the maximal atoms distance contact.
    :type atoms_dist: str
    :param res_dist: the maximal residues distance in the amino acids chain.
    :type res_dist: int
    """
    data = {}
    for _, row_bed in bed.iterrows():
        tmp = df[df["acceptor regions"] == row_bed["region"]]
        if not tmp.empty:
            if not row_bed["region"] in data:
                data[row_bed["region"]] = 0
            for _, row_tmp in tmp.iterrows():
                data[row_bed["region"]] += row_tmp["atoms contacts"]
    source = pd.DataFrame.from_dict({"region": data.keys(), "number of contacts": data.values()})

    # set the seaborn plots style and size
    sns.set_style("darkgrid")
    sns.set_context("poster", rc={"grid.linewidth": 2})
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.barplot(data=source, ax=ax, x="region", y="number of contacts", color="blue")
    ax.set_xticklabels(source["region"], rotation=45, horizontalalignment="right")
    ax.set_xlabel(None)
    ax.set_ylabel("Number of contacts", fontweight="bold")
    ax.text(x=0.5, y=1.1, s=f"Outliers contacts by regions{' (region of interest ' + roi + ')' if roi else ''}",
            weight="bold", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(x=0.5, y=1.05, s=f"Maximal atoms distance: {atoms_dist} \u212B, maximal residues distance: {res_dist}",
            alpha=0.75, ha="center", va="bottom", transform=ax.transAxes)
    fig.savefig(path, bbox_inches="tight")
    logging.info(f"Contacts by region plot saved: {path}")


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

    # get the input basename
    basename = os.path.splitext(os.path.basename(args.input))[0]
    # load and format the BED file
    bed_data = extract_bed(args.bed)

    # extract the contacts
    try:
        contacts = extract_contacts_with_atoms_distance(args.input, args.roi, args.col_distance, args.atoms_distance)
    except argparse.ArgumentTypeError as exc:
        logging.error(exc, exc_info=True)
        sys.exit(1)
    except KeyError as exc:
        logging.error(f"\"{args.col_distance}\" column name does not exists in the CSV file.", exc_info=True)
        sys.exit(1)
    except TypeError as exc:
        logging.error(f"\"{args.col_distance}\" column name may refers to a column that is not an atoms distances.",
                      exc_info=True)
        sys.exit(1)
    outliers = outliers_contacts(contacts, args.residues_distance, args.col_distance)
    logging.info(f"{len(outliers)} unique residues pairs contacts (<= {args.atoms_distance} \u212B) with a distance of "
                 f"at least {args.residues_distance} "
                 f"residues{' in the region of interest '+args.roi if args.roi else ''} (residues pair may have "
                 f"multiple atoms contacts).")

    # get the outliers contacts
    outliers = update_regions(outliers, bed_data, os.path.join(args.out, f"outliers_{basename}.csv"))

    # by acceptor region
    acceptors_regions_involved(outliers, bed_data, os.path.join(args.out, f"regions_{basename}.{args.format}"),
                               args.roi, args.atoms_distance, args.residues_distance)
