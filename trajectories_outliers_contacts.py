#!/usr/bin/env python3

"""
Created on 12 Sep. 2022
"""

__author__ = "Nicolas JEANNE"
__copyright__ = "GNU General Public License"
__email__ = "jeanne.n@chu-toulouse.fr"
__version__ = "2.0.0"

import argparse
import logging
import os
import re
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import seaborn as sns
import yaml


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


def get_domains(domains_path):
    """
    Load the domains file and fill in domains that are not covered.

    :param domains_path: the path to the domains file.
    :type domains_path: str
    :return: the filled-in BED.
    :rtype: pd.Dataframe
    """
    raw = pd.read_csv(domains_path, sep=",", header=0, names=["domain", "start", "stop", "color"])
    data = {"domain": [], "start": [], "stop": [], "color": []}
    pos_start = 1
    previous_region = None
    for idx, row in raw.iterrows():
        if pos_start < row["start"]:  # not in domain
            # record the undefined domain
            data["start"].append(pos_start)
            data["stop"].append(row["start"] - 1)
            data["color"].append(row["color"])
            if idx == 0:  # before first domain
                data["domain"].append(f"before {row['domain']}")
            else:
                data["domain"].append(f"between {previous_region} and {row['domain']}")
        # record the domain in the BED
        data["domain"].append(row["domain"])
        data["start"].append(row["start"])
        data["stop"].append(row["stop"])
        data["color"].append(row["color"])
        pos_start = row["stop"] + 1
        previous_region = row["domain"]

    df = pd.DataFrame(data)
    return df


def get_contacts_analysis_parameters(parameters_path):
    """
    Get the analysis parameters from the previous analysis from trajectories_contacts.py script.

    :param parameters_path: the path to the YAML parameters file.
    :type parameters_path: str
    :return: the parameters.
    :rtype: dict
    """
    with open(parameters_path, "r") as file_handler:
        parameters = yaml.safe_load(file_handler.read())
        logging.info("Parameters used for trajectory contacts search:")
        for p_key, p_value in parameters.items():
            if type(p_value) is dict:
                logging.info(f"\t{p_key}:")
                for p_key_2, p_value_2 in p_value.items():
                    logging.info(f"\t\t{p_key_2}:\t{p_value_2}")
            else:
                logging.info(f"\t{p_key}:\t{p_value}")
    return parameters


def extract_roi(roi_to_extract):
    """
    Extract the start and stop coordinates of the region of interest (roi).

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


def get_residues_in_contact(df):
    """
    Reduce the contacts to residues, if 2 residues have multiple atoms in contacts, the pair of atoms with the smallest
    distance will be kept, then create a column with the number of contacts between this 2 residues and finally sort
    the data by position of the donors.

    :param df: the contacts from the trajectory analysis.
    :type df: pandas.Dataframe
    :return: the reduced dataframe with the minimal distance value of all the couples of donors-acceptors and the
    column with the number of contacts.
    :rtype: pd.Dataframe
    """
    # convert the donor and acceptor positions columns to int
    df["donor position"] = pd.to_numeric(df["donor position"])
    df["acceptor position"] = pd.to_numeric(df["acceptor position"])
    # donors_acceptors is used to register the combination of donor and acceptor and select only the value with the
    # minimal contact distance and also the number of contacts
    donors_acceptors = []
    idx_to_remove = []
    donor_acceptor_nb_contacts = []
    for _, row in df.iterrows():
        donor = f"{row['donor position']}{row['donor residue']}"
        acceptor = f"{row['acceptor position']}{row['acceptor residue']}"
        if f"{donor}_{acceptor}" not in donors_acceptors:
            donors_acceptors.append(f"{donor}_{acceptor}")
            tmp_df = df[
                (df["donor position"] == row["donor position"]) & (df["acceptor position"] == row["acceptor position"])]
            # get the index of the minimal distance
            idx_min = tmp_df[["median distance"]].idxmin()
            # record the index to remove of the other rows of the same donor - acceptor positions
            tmp_index_to_remove = list(tmp_df.index.drop(idx_min))
            if tmp_index_to_remove:
                idx_to_remove = idx_to_remove + tmp_index_to_remove
            donor_acceptor_nb_contacts.append(len(tmp_df.index))
    df = df.drop(idx_to_remove)
    df["number contacts"] = donor_acceptor_nb_contacts
    df = df.sort_values(by=["donor position"])
    return df


def extract_residues_contacts(path_contacts, roi):
    """
    Extract the contacts from the CSV file residues contacts and filter on the region of interest (if any) for donors
    and acceptors.

    :param path_contacts: the path to the contacts CSV path.
    :type path_contacts: str
    :param roi: the list of the start and end positions of the Region Of Interest.
    :type roi: list
    :return: the filtered contacts.
    :rtype: pd.Dataframe
    """
    # load the contacts file
    df_contacts = pd.read_csv(path_contacts, sep=",")
    logging.info(f"{len(df_contacts)} atoms contacts in the input data (residues pairs may have multiple atoms "
                 f"contacts).")
    # select the rows of acceptors and donors within the region of interest if any
    roi_text = ""
    if roi:
        # reduce to the donors region of interest limits
        df_donors = df_contacts[df_contacts["donor position"].between(roi[0], roi[1])]
        print(f"donors:\n{df_donors}\n\n")
        df_acceptors = df_contacts[df_contacts["acceptor position"].between(roi[0], roi[1])]
        print(f"acceptors:\n{df_acceptors}\n\n")
        logging.debug(f"{len(df_contacts)} atoms contacts in the region of interest.")
        roi_text = f" for donors and acceptors in the region of interest {roi[0]}-{roi[1]}"
        df_contacts = pd.concat([df_donors, df_acceptors]).drop_duplicates()
        print(f"all:\n{df_contacts}\n\n")

    df_residues_contacts = get_residues_in_contact(df_contacts)
    logging.info(f"{len(df_residues_contacts)} extracted residues contacts for {len(df_contacts)} atoms contacts"
                 f"{roi_text}.")
    df_residues_contacts.to_csv("results/HEPAC-6_RNF19A_ORF1_0/tmp.csv")
    return df_residues_contacts


def get_df_distances_nb_contacts(df):
    """
    Create a distances and a number of contacts dataframes for the couples donors and acceptors.

    :param df: the initial dataframe
    :type df: pd.Dataframe
    :return: the dataframe of distances and the dataframe of the number of contacts.
    :rtype: pd.Dataframe, pd.Dataframe
    """
    # create the dictionaries of distances and number of contacts
    distances = {}
    nb_contacts = {}
    donors = []
    acceptors = []
    unique_donor_positions = sorted(list(set(df["donor position"])))
    unique_acceptor_positions = sorted(list(set(df["acceptor position"])))
    for donor_position in unique_donor_positions:
        donor = f"{donor_position}{df.loc[(df['donor position'] == donor_position), 'donor residue'].values[0]}"
        if donor not in donors:
            donors.append(donor)
        for acceptor_position in unique_acceptor_positions:
            acceptor = f"{acceptor_position}" \
                       f"{df.loc[(df['acceptor position'] == acceptor_position), 'acceptor residue'].values[0]}"
            if acceptor not in acceptors:
                acceptors.append(acceptor)
            # get the distance
            if acceptor_position not in distances:
                distances[acceptor_position] = []
            dist = df.loc[(df["donor position"] == donor_position) & (df["acceptor position"] == acceptor_position),
                          "median distance"]
            if not dist.empty:
                distances[acceptor_position].append(dist.values[0])
            else:
                distances[acceptor_position].append(None)
            # get the number of contacts
            if acceptor_position not in nb_contacts:
                nb_contacts[acceptor_position] = []
            contacts_found = df.loc[(df["donor position"] == donor_position) & (
                        df["acceptor position"] == acceptor_position), "number contacts"]
            if not contacts_found.empty:
                nb_contacts[acceptor_position].append(contacts_found.values[0])
            else:
                nb_contacts[acceptor_position].append(None)
    source_distances = pd.DataFrame(distances, index=donors)
    source_distances.columns = acceptors
    source_nb_contacts = pd.DataFrame(nb_contacts, index=donors)
    source_nb_contacts.columns = acceptors
    return source_distances, source_nb_contacts


def heatmap_contacts(contacts, params, bn, out_dir, output_fmt, lim_roi):
    """
    Create the heatmap of contacts between residues.

    :param contacts: the contacts by residues.
    :type contacts: dict
    :param params: the parameters used in the previous trajectory contacts analysis.
    :type params: dict
    :param bn: the basename.
    :type bn: str
    :param out_dir: the output directory.
    :type out_dir: str
    :param output_fmt: the output format for the heatmap.
    :type output_fmt: str
    :param lim_roi: the region of interest limits for the heatmap.
    :type lim_roi: list
    """
    # create the distances and number of contacts dataframes to produce the heatmap
    source_distances, source_nb_contacts = get_df_distances_nb_contacts(contacts)

    # increase the size of the heatmap if too much entries
    factor = int(len(source_distances) / 40) if len(source_distances) / 40 >= 1 else 1
    logging.debug(f"{len(source_distances)} entries, the size of the figure is multiplied by a factor {factor}.")
    rcParams["figure.figsize"] = 15 * factor, 12 * factor
    # create the heatmap
    heatmap = sns.heatmap(source_distances, annot=source_nb_contacts, cbar_kws={"label": "Distance (\u212B)"},
                          linewidths=0.5, xticklabels=True, yticklabels=True)
    heatmap.figure.axes[-1].yaxis.label.set_size(15)
    plot = heatmap.get_figure()
    title = f"Contact residues median distance: {bn}"
    plt.suptitle(title, fontsize="large", fontweight="bold")
    subtitle = "Number of residues atoms in contact are displayed in the squares."
    if lim_roi:
        subtitle = f"{subtitle}\nHeatmap focus on donor residues {lim_roi[0]} to {lim_roi[1]}"
    if params["frames"]:
        subtitle = f"{subtitle}\nMolecular Dynamics frames used: {params['frames']['min']} to {params['frames']['max']}"
    plt.title(subtitle)
    plt.xlabel("Acceptors", fontweight="bold")
    plt.ylabel("Donors", fontweight="bold")
    out_path = os.path.join(out_dir, f"heatmap_distances_{bn}.{output_fmt}")
    plot.savefig(out_path)
    # clear the plot for the next use of the function
    plt.clf()
    logging.info(f"\tmedian distance heatmap saved: {out_path}")


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


def update_domains(df, domains, path):
    """
    Get the domains for pairs acceptor and donor.

    :param df: the dataframe of unique residues pairs contacts.
    :type df: pd.Dataframe
    :param domains: the domains coordinates.
    :type domains: pd.Dataframe
    :param path: the path of the outliers contacts updated with the domains CSV file.
    :type path: str
    :return: the pairs contacts dataframe updated with the regions.
    :rtype: pd.Dataframe
    """
    donors_regions = [None] * len(df)
    acceptors_regions = [None] * len(df)
    for idx, row_contacts in df.iterrows():
        for _, row_domains in domains.iterrows():
            if row_domains["start"] <= row_contacts["donor positions"] <= row_domains["stop"]:
                donors_regions[idx] = row_domains["domain"]
            if row_domains["start"] <= row_contacts["acceptor positions"] <= row_domains["stop"]:
                acceptors_regions[idx] = row_domains["domain"]
    df.insert(2, "donor domains", pd.DataFrame(donors_regions))
    df.insert(5, "acceptor domains", pd.DataFrame(acceptors_regions))
    df.to_csv(path, index=False)
    logging.info(f"Pairs residues contacts updated with domains saved: {path}")
    return df


def acceptors_domains_involved(df, domains, out_dir, bn, roi, atoms_dist, res_dist, fmt):
    """
    Create the plot of contacts by regions.

    :param df: the dataframe.
    :type df: pd.Dataframe
    :param domains: the domains.
    :type domains: pd.Dataframe
    :param out_dir: the path of the output directory.
    :type out_dir: str
    :param bn: basename for the sample.
    :type bn: str
    :param roi: the region of interest.
    :type roi: str
    :param atoms_dist: the maximal atoms distance contact.
    :type atoms_dist: str
    :param res_dist: the maximal residues distance in the amino acids chain.
    :type res_dist: int
    :param fmt: the format for the plot.
    :type fmt: str
    """
    data = {}
    for _, row_domains in domains.iterrows():
        tmp = df[df["acceptor domains"] == row_domains["domain"]]
        if not tmp.empty:
            if not row_domains["domain"] in data:
                data[row_domains["domain"]] = 0
            for _, row_tmp in tmp.iterrows():
                data[row_domains["domain"]] += row_tmp["atoms contacts"]
    source = pd.DataFrame.from_dict({"domain": data.keys(), "number of contacts": data.values()})

    # set the seaborn plots style and size
    sns.set_style("darkgrid")
    sns.set_context("poster", rc={"grid.linewidth": 2})
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.barplot(data=source, ax=ax, x="domain", y="number of contacts", color="blue")
    ax.set_xticklabels(source["domain"], rotation=45, horizontalalignment="right")
    ax.set_xlabel(None)
    ax.set_ylabel("Number of contacts", fontweight="bold")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.text(x=0.5, y=1.1, s=f"{bn}: outliers contacts by domains{' (region of interest ' + roi + ')' if roi else ''}",
            weight="bold", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(x=0.5, y=1.05, s=f"Maximal atoms distance: {atoms_dist} \u212B, maximal residues distance: {res_dist}",
            alpha=0.75, ha="center", va="bottom", transform=ax.transAxes)
    path = os.path.join(out_dir, f"outliers_{bn}.{fmt}")
    fig.savefig(path, bbox_inches="tight")
    logging.info(f"Contacts by domain plot saved: {path}")


if __name__ == "__main__":
    descr = f"""
    {os.path.basename(__file__)} v. {__version__}

    Created by {__author__}.
    Contact: {__email__}
    {__copyright__}

    Distributed on an "AS IS" basis without warranties or conditions of any kind, either express or implied.

    From a CSV of the amino acids contacts file produced by the script trajectories
    (https://github.com/njeanne/trajectories), extract the information of the contacts outside the neighbourhood 
    contacts of an amino acid and determine to which domain the acceptor amino acid belongs to using a BED file 
    registering the domains.
    """
    parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", required=True, type=str, help="the path to the output directory.")
    parser.add_argument("-d", "--domains", required=False, type=str, default="",
                        help="the path to the CSV domains file. A comma separated file, the first column is the "
                             "annotation name, the 2nd is the residue start coordinate, the 3rd is the residue end "
                             "coordinate, the last one is the color to apply in hexadecimal format. The coordinate are "
                             "1-indexed.")
    parser.add_argument("-p", "--parameters", required=True, type=str,
                        help="the path to the trajectory contacts analysis parameters (the YAML file in the results "
                             "directory of the trajectory_contacts.py script.")
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
    parser.add_argument("-r", "--residues-distance", required=False, type=int, default=4,
                        help="when 2 atoms of different residues are in contact, the minimal distance in number of "
                             "residues that should separate them for a long range interaction. Default is 4 residues, "
                             "the number of residues in an alpha helix.")
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
    # load and format the domains file
    domains_data = get_domains(args.domains)
    # get the Region Of Interest if specified
    if args.roi:
        roi_limits = extract_roi(args.roi)
    else:
        roi_limits = None

    # load the contacts analysis parameters
    try:
        parameters_contacts_analysis = get_contacts_analysis_parameters(args.parameters)
    except ImportError as exc:
        logging.error(exc, exc_info=True)
        sys.exit(1)

    # extract the contacts
    try:
        residues_contacts = extract_residues_contacts(args.input, roi_limits)
        print(residues_contacts)
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

    # get the heatmaps of validated contacts by residues
    heatmap_contacts(residues_contacts, parameters_contacts_analysis, args.out, basename, args.format, roi_limits)
    sys.exit()

    outliers = outliers_contacts(contacts, args.residues_distance, args.col_distance)
    logging.info(f"{len(outliers)} unique residues pairs contacts (<= {args.atoms_distance} \u212B) with a distance of "
                 f"at least {args.residues_distance} "
                 f"residues{' in the region of interest '+args.roi if args.roi else ''} (residues pair may have "
                 f"multiple atoms contacts).")

    # get the outliers contacts
    outliers = update_domains(outliers, domains_data, os.path.join(args.out, f"outliers_{basename}.csv"))

    # by acceptor domain
    acceptors_domains_involved(outliers, domains_data, args.out, basename, args.roi, args.atoms_distance,
                               args.residues_distance, args.format)
