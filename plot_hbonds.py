#!/usr/bin/env python3

"""
Created on 12 Sep. 2022
"""

__author__ = "Nicolas JEANNE"
__copyright__ = "GNU General Public License"
__email__ = "jeanne.n@chu-toulouse.fr"
__version__ = "2.4.0"

import argparse
import logging
import os
import re
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


def create_log(log_path, level, out_dir):
    """
    Create the log as a text file and as a stream.

    :param log_path: the path of the log.
    :type log_path: str
    :param level: the level og the log.
    :type level: str
    :param out_dir: the result directory path.
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
        log_level = log_level_dict[level]

    if os.path.exists(log_path):
        os.remove(log_path)

    logging.basicConfig(format="%(asctime)s %(levelname)s:\t%(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=log_level,
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logging.info(f"version: {__version__}")
    logging.info(f"CMD: {' '.join(sys.argv)}")


def check_optional_args(args_domains, args_embedded_domains, args_residues_distance):
    """Check the combination of optional arguments

    :param args_domains: the domain's argument.
    :type args_domains: str
    :param args_embedded_domains: the embedded domain's argument.
    :type args_embedded_domains: bool
    :param args_residues_distance: the value of the residues distance argument.
    :type args_residues_distance: int
    """
    if args_embedded_domains and not args_domains:
        logging.warning("--embedded-domains will not be used as the --domains argument is missing.")
    if args_residues_distance != 4 and not args_domains:
        logging.warning(f"--residues-distance {args_residues_distance} will not be used as the --domains argument is "
                        f"missing.")


def get_domains(domains_path, use_embedded):
    """
    Load the domain file and fill in domains that are not covered.

    :param domains_path: the path to the domains file.
    :type domains_path: str
    :param use_embedded: if an embedded domain should be used as a domain or if only the embedding domain should be
    processed.
    :type use_embedded: bool
    :return: the filled-in domains data frame.
    :rtype: pd.Dataframe
    """
    logging.info(f"domains embedded in other domains will{' ' if use_embedded else ' not '}be used in the hydrogen "
                 f"bonds by domain plot.")
    raw = pd.read_csv(domains_path, sep=",", header=0, names=["domain", "start", "stop", "color"], index_col=False)

    # check for embedded entries
    embedded_raw_idx = []
    previous_stop = 0
    for idx, row in raw.iterrows():
        if row["stop"] < previous_stop:
            embedded_raw_idx.append(idx)
        else:
            previous_stop = row["stop"]

    # update the domains
    data = {"domain": [], "start": [], "stop": [], "color": []}
    expected_start = 1
    previous = {"embedded": False, "domain": None, "stop": None, "color": None}
    for idx, row in raw.iterrows():
        if idx in embedded_raw_idx:  # case of embedded entry
            if use_embedded:
                # modify the previous end of the embedding domain
                data["stop"][-1] = row["start"] - 1
                # register the embedded domain
                data["domain"].append(row["domain"])
                data["start"].append(row["start"])
                data["stop"].append(row["stop"])
                data["color"].append(row["color"])
                # record the end of the domain where the embedded is
                previous["embedded"] = True
                expected_start = row["stop"] + 1
        else:
            if previous["embedded"]:
                # register the end of the domain where the previous was
                data["domain"].append(previous["domain"])
                data["start"].append(expected_start)
                data["stop"].append(previous["stop"])
                data["color"].append(previous["color"])
                expected_start = previous["stop"] + 1
                previous["embedded"] = False
            if row["start"] > expected_start:  # between domains
                # record the undefined domain
                if idx == 0:  # before first domain
                    data["domain"].append(f"before {row['domain']}")
                else:
                    data["domain"].append(f"between {previous['domain']} and {row['domain']}")
                data["start"].append(expected_start)
                data["stop"].append(row["start"] - 1)
                data["color"].append("#cecece")
            # record the current row domain
            data["domain"].append(row["domain"])
            data["start"].append(row["start"])
            data["stop"].append(row["stop"])
            data["color"].append(row["color"])
            previous["domain"] = row["domain"]
            previous["stop"] = row["stop"]
            previous["color"] = row["color"]
            expected_start = row["stop"] + 1

    # add a last row for the case of a contact after the last domain
    data["domain"].append(f"after {row['domain']}")
    data["start"].append(row["stop"] + 1)
    data["stop"].append(row["stop"] + 10000)
    data["color"].append(None)

    df = pd.DataFrame(data)
    return df


def get_hydrogen_bonds_analysis_parameters(parameters_path):
    """
    Get the analysis parameters from the previous analysis from trajectories_hbonds.py script.

    :param parameters_path: the path to the YAML parameters file.
    :type parameters_path: str
    :return: the parameters.
    :rtype: dict
    """
    with open(parameters_path, "r") as file_handler:
        parameters = yaml.safe_load(file_handler.read())
        logging.info("Parameters used for trajectory hydrogen bonds search:")
        for p_key, p_value in parameters.items():
            if type(p_value) is dict:
                logging.info(f"\t{p_key}:")
                for p_key_2, p_value_2 in p_value.items():
                    logging.info(f"\t\t{p_key_2}:\t{p_value_2}")
            else:
                logging.info(f"\t{p_key}:\t{', '.join(p_value) if p_key == 'trajectory files processed' else p_value}")
    return parameters


def extract_roi(roi_to_extract):
    """
    Extract the region of interest (roi) start's and stop's coordinates.

    :param roi_to_extract: the coordinates ot the region of interest, as 100-200 i.e.,
    :type roi_to_extract: str
    :raises ArgumentTypeError: is not between 0.0 and 100.0
    :return: the region of interest (roi) start's and stop's coordinates.
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


def get_residues_forming_an_hydrogen_bond(df):
    """
    Reduce the hydrogen bonds to a residue hydrogen bonds if two residues have multiple atoms in hydrogen bonds. The
    residue in the ROI is used as reference, the median distance between the ROI (Region Of Interest) residue atoms and
    the atoms of its partner will be added.
    All their atoms involved in hydrogen bonds identifiers will be regrouped in a single column with semicolon
    separators.
    Same thing for their median distances during the molecular dynamics simulation and the type of the residue's atom
    in the ROI (donor or acceptor).

    :param df: the hydrogen bonds from the trajectory analysis.
    :type df: pandas.Dataframe
    :return: the reduced dataframe with the added columns for the couples ROI partner - second partner.
    :rtype: pd.Dataframe
    """
    # combinations were used to register the combination of ROI (Region Of Interest) partner and second partner, in
    # this specific order, then select only the value with the minimal contact distance and also the number of hydrogen
    # bonds for this pair of residues
    combinations = []
    idx_to_remove = []
    median_distances = []
    combinations_nb_hydrogen_bonds = []
    combinations_atoms_hydrogen_bonds = []
    combinations_atoms_hydrogen_bonds_distances = []
    first_partner_types = []
    for idx, row in df.iterrows():
        first = f"{row['ROI partner position']}{row['ROI partner residue']}"
        second = f"{row['second partner position']}{row['second partner residue']}"
        if f"{first}_{second}" not in combinations and f"{second}_{first}" not in combinations:
            combinations.append(f"{first}_{second}")
            tmp_df = df[(df["ROI partner position"] == row["ROI partner position"]) & (
                        df["second partner position"] == row["second partner position"]) | (
                                    df["second partner position"] == row["second partner position"]) & (
                                    df["ROI partner position"] == row["ROI partner position"])]
            # get the median distance of all the atom hydrogen bonds between the two residues
            median_distances.append(statistics.median(tmp_df["median distance"]))
            combinations_nb_hydrogen_bonds.append(len(tmp_df.index))
            hydrogen_bonds_str = None
            tmp_df_distance_hydrogen_bonds = []
            tmp_df_partner_types = []
            for _, tmp_df_row in tmp_df.iterrows():
                if hydrogen_bonds_str is None:
                    hydrogen_bonds_str = f"{tmp_df_row['hydrogen bond']}"
                else:
                    hydrogen_bonds_str = f"{hydrogen_bonds_str} | {tmp_df_row['hydrogen bond']}"
                tmp_df_distance_hydrogen_bonds.append(str(tmp_df_row["median distance"]))
                tmp_df_partner_types.append(tmp_df_row["ROI partner type"])
            combinations_atoms_hydrogen_bonds.append(hydrogen_bonds_str)
            combinations_atoms_hydrogen_bonds_distances.append(" | ".join(tmp_df_distance_hydrogen_bonds))
            first_partner_types.append(" | ".join(tmp_df_partner_types))
        else:
            idx_to_remove.append(idx)
    df = df.drop(idx_to_remove)
    df["hydrogen bond"] = combinations
    df["median distance"] = median_distances
    df["number atoms hydrogen bonds"] = combinations_nb_hydrogen_bonds
    df["atoms hydrogen bonds"] = combinations_atoms_hydrogen_bonds
    df["atoms hydrogen bonds distances"] = combinations_atoms_hydrogen_bonds_distances
    df["ROI partner types"] = first_partner_types
    # rename the columns
    df.rename(columns={"median distance": "median hydrogen bonds distance"}, inplace=True)
    df.rename(columns={"hydrogen bond": "residues hydrogen bond"}, inplace=True)
    # drop the "ROI partner type" column
    df.drop(["ROI partner type"], axis=1)
    # sort on the values of the column "ROI partner position"
    df = df.sort_values(by=["ROI partner position"])
    # set the column order
    cols = ["residues hydrogen bond", "ROI partner position", "ROI partner residue", "second partner position",
            "second partner residue", "median hydrogen bonds distance", "number atoms hydrogen bonds",
            "atoms hydrogen bonds", "atoms hydrogen bonds distances", "ROI partner types"]
    df = df[cols]
    return df


def extract_residues_hydrogen_bonds(path_hydrogen_bonds, roi):
    """
    Extract the hydrogen bonds from the CSV file residues hydrogen bonds and filter on the region of interest (if any)
    for donors and acceptors.

    :param path_hydrogen_bonds: the path to the hydrogen bonds CSV path.
    :type path_hydrogen_bonds: Str
    :param roi: the list of the start and end positions of the Region Of Interest.
    :type roi: List
    :return: the filtered hydrogen bonds.
    :rtype: Pd.Dataframe
    """
    # load the hydrogen bonds' file
    df_hydrogen_bonds_all = pd.read_csv(path_hydrogen_bonds, sep=",")
    logging.info(f"{len(df_hydrogen_bonds_all)} atoms involved in hydrogen bonds in the input data (residues pairs may "
                 f"have multiple atoms hydrogen bonds).")
    # select the rows of acceptors and donors within the region of interest if any
    roi_text = ""
    first_type = []
    first_position = []
    first_residue = []
    second_position = []
    second_residue = []
    # reduce to the donors region of interest limits
    df_donors = df_hydrogen_bonds_all[df_hydrogen_bonds_all["donor position"].between(roi[0], roi[1])]
    df_acceptors = df_hydrogen_bonds_all[df_hydrogen_bonds_all["acceptor position"].between(roi[0], roi[1])]
    logging.debug(f"{len(df_hydrogen_bonds_all)} atoms involved in hydrogen bonds in the region of interest.")
    roi_text = f" for donors and acceptors in the region of interest {roi[0]}-{roi[1]}"
    df_hydrogen_bonds_all = pd.concat([df_donors, df_acceptors]).drop_duplicates()
    for _, row in df_hydrogen_bonds_all.iterrows():
        if roi[0] <= row["donor position"] <= roi[1]:
            first_type.append("donor")
            first_position.append(row["donor position"])
            first_residue.append(row["donor residue"])
            second_position.append(row["acceptor position"])
            second_residue.append(row["acceptor residue"])
        else:
            first_type.append("acceptor")
            first_position.append(row["acceptor position"])
            first_residue.append(row["acceptor residue"])
            second_position.append(row["donor position"])
            second_residue.append(row["donor residue"])

    # lose the notions of donor and acceptor to use ROI and second partner
    df_tmp = pd.DataFrame({"hydrogen bond": df_hydrogen_bonds_all["contact"],
                           "ROI partner position": first_position,
                           "ROI partner residue": first_residue,
                           "second partner position": second_position,
                           "second partner residue": second_residue,
                           "median distance": df_hydrogen_bonds_all["median distance"],
                           "ROI partner type": first_type})
    # reduce to residue hydrogen bonds
    df_residues_hydrogen_bonds = get_residues_forming_an_hydrogen_bond(df_tmp)
    logging.info(f"{len(df_residues_hydrogen_bonds)} extracted residues forming hydrogen bonds with "
                 f"{len(df_hydrogen_bonds_all)} atoms (donors and acceptors) involved in hydrogen bonds{roi_text}.")
    return df_residues_hydrogen_bonds


def heatmap_distances_nb_hydrogen_bonds(df):
    """
    Create a distances and a number of hydrogen bonds dataframes for the couples first and second partner.

    :param df: the initial dataframe
    :type df: pd.Dataframe
    :return: the distances dataframe and the number of hydrogen bonds dataframe.
    :rtype: pd.Dataframe, pd.Dataframe
    """

    # create the dictionaries of distances and number of hydrogen bonds
    distances = {}
    nb_hydrogen_bonds = {}
    firsts = []
    seconds = []
    unique_first_positions = sorted(list(set(df["ROI partner position"])))
    unique_second_positions = sorted(list(set(df["second partner position"])))
    for first_position in unique_first_positions:
        first = f"{first_position}" \
                f"{df.loc[(df['ROI partner position'] == first_position), 'ROI partner residue'].values[0]}"
        if first not in firsts:
            firsts.append(first)
        for second_position in unique_second_positions:
            second = f"{second_position}" \
                     f"{df.loc[(df['second partner position'] == second_position), 'second partner residue'].values[0]}"
            if second not in seconds:
                seconds.append(second)
            # get the distance
            if second_position not in distances:
                distances[second_position] = []
            dist = df.loc[(df["ROI partner position"] == first_position) & (
                        df["second partner position"] == second_position), "median hydrogen bonds distance"]
            if not dist.empty:
                distances[second_position].append(dist.values[0])
            else:
                distances[second_position].append(None)
            # get the number of hydrogen bonds
            if second_position not in nb_hydrogen_bonds:
                nb_hydrogen_bonds[second_position] = []
            hydrogen_bonds = df.loc[(df["ROI partner position"] == first_position) & (
                        df["second partner position"] == second_position), "number atoms hydrogen bonds"]
            if not hydrogen_bonds.empty:
                nb_hydrogen_bonds[second_position].append(hydrogen_bonds.values[0])
            else:
                nb_hydrogen_bonds[second_position].append(None)
    source_distances = pd.DataFrame(distances, index=firsts)
    source_distances.columns = seconds
    source_nb_hydrogen_bonds = pd.DataFrame(nb_hydrogen_bonds, index=firsts)
    source_nb_hydrogen_bonds.columns = seconds
    return source_distances, source_nb_hydrogen_bonds


def heatmap_hydrogen_bonds(hydrogen_bonds, params, out_dir, output_fmt, roi_id):
    """
    Create the heatmap of hydrogen bonds between residues.

    :param hydrogen_bonds: the hydrogen bonds by residues.
    :type hydrogen_bonds: dict
    :param params: the parameters used in the previous trajectory hydrogen bonds analysis.
    :type params: dict
    :param out_dir: the output directory.
    :type out_dir: str
    :param output_fmt: the output format for the heatmap.
    :type output_fmt: str
    :param roi_id: the region of interest.
    :type roi_id: str
    """
    logging.info("Computing the hydrogen bonds heatmap..")
    # create the distances and number of hydrogen bonds dataframes to produce the heatmap
    source_distances, source_nb_hydrogen_bonds = heatmap_distances_nb_hydrogen_bonds(hydrogen_bonds)
    # get a mask for the Null values, it will be useful to color those heatmap cells in grey
    mask = source_distances.isnull()

    # increase the size of the heatmap if too many entries
    factor = int(len(source_distances) / 40) if len(source_distances) / 40 >= 1 else 1
    logging.debug(f"{len(source_distances)} entries, the size of the figure is multiplied by a factor {factor}.")
    matplotlib.rcParams["figure.figsize"] = 15 * factor, 12 * factor
    # create the heatmap
    heatmap = sns.heatmap(source_distances, annot=source_nb_hydrogen_bonds, cbar_kws={"label": "Distance (\u212B)"},
                          linewidths=0.5, xticklabels=True, yticklabels=True, mask=mask)
    # set the color of Null cells
    heatmap.set_facecolor("lavender")
    heatmap.figure.axes[-1].yaxis.label.set_size(15)
    plot = heatmap.get_figure()
    title = f"Hydrogen bonds residues median distance: {params['sample']} {roi_id} vs. whole protein"
    plt.suptitle(title, fontsize="large", fontweight="bold")
    md_duration = f"MD length: {params['MD duration']}. " if "MD duration" in params else ""
    subtitle = f"{md_duration}The number of residue atoms forming hydrogen bonds is shown in the squares."
    if params["frames"]:
        subtitle = (f"{subtitle}\nHydrogen bond validated if present in {params['parameters']['proportion hbonds']}% "
                    f"of the {params['frames']} frames.")
    plt.title(subtitle)
    plt.xlabel("Whole protein residues", fontweight="bold")
    plt.ylabel(f"{roi_id} residues", fontweight="bold")
    out_path = os.path.join(out_dir, f"heatmap_distances_{params['sample'].replace(' ', '_')}.{output_fmt}")
    plot.savefig(out_path)
    # clear the plot for the next use of the function
    plt.clf()
    logging.info(f"\tmedian distance heatmap saved: {out_path}")


def unique_residues_pairs(df_not_unique, col):
    """
    Get the unique residues pairs forming hydrogen bonds.

    :param df_not_unique: the dataframe of hydrogen bonds.
    :type df_not_unique: pd.Dataframe
    :param col: the atom distances column name.
    :type col: str
    :return: the dataframe of unique residues pairs forming hydrogen bonds.
    :rtype: pd.Dataframe
    """
    donor_positions = []
    donor_residues = []
    acceptor_positions = []
    acceptor_residues = []
    nb_hydrogen_bonds = []
    max_atoms_dist = []
    hydrogen_bonds_ids = []
    for donor_position in sorted(list(set(df_not_unique["donor position"]))):
        df_donors = df_not_unique[df_not_unique["donor position"] == donor_position]
        for acceptor_position in sorted(list(set(df_donors["acceptor position"]))):
            df_acceptors = df_donors[df_donors["acceptor position"] == acceptor_position]
            donor_positions.append(donor_position)
            donor_residues.append(df_donors["donor residue"].iloc[0])
            acceptor_positions.append(acceptor_position)
            acceptor_residues.append(df_acceptors["acceptor residue"].iloc[0])
            nb_hydrogen_bonds.append(len(df_acceptors))
            max_atoms_dist.append(max(df_acceptors[col]))
            pair_hydrogen_bonds_ids = []
            for _, row in df_acceptors.iterrows():
                pair_hydrogen_bonds_ids.append(row["hydrogen bond"])
            hydrogen_bonds_ids.append(" | ".join(pair_hydrogen_bonds_ids))

    df_uniques = pd.DataFrame.from_dict({"donor positions": donor_positions,
                                         "donor residues": donor_residues,
                                         "acceptor positions": acceptor_positions,
                                         "acceptor residues": acceptor_residues,
                                         "atoms hydrogen bonds": nb_hydrogen_bonds,
                                         "maximum atoms distance": max_atoms_dist,
                                         "hydrogen bonds ID": hydrogen_bonds_ids})
    return df_uniques


def outliers_hydrogen_bonds(df, res_dist_thr):
    """
    Get the residues pairs hydrogen bonds above the residues distance hydrogen bonds, meaning hydrogen bonds of distant
    residues.

    :param df: the dataframe of hydrogen bonds.
    :type df: pd.Dataframe
    :param res_dist_thr: the residues distance threshold.
    :type res_dist_thr: int
    :return: the dataframe of unique residues pairs forming hydrogen bonds.
    :rtype: pd.Dataframe
    """
    idx_to_remove_for_residue_distance = []
    for idx, row in df.iterrows():
        if abs(row["ROI partner position"] - row["second partner position"]) < res_dist_thr:
            idx_to_remove_for_residue_distance.append(idx)
    # remove rows with too close distance between the residues
    df.drop(idx_to_remove_for_residue_distance, inplace=True, axis=0)
    # reset the index of the dataframe from 0
    df.reset_index(inplace=True, drop=True)
    logging.debug(f"{len(df)} atoms involved in hydrogen bonds remaining with a minimal residues distance threshold of "
                  f"{res_dist_thr}.")
    return df


def extract_roi_id(domains, roi_coord):
    """
    Extract the Region Of Interest (ROI) identity from the domain file using the start and stop coordinates. If no
    match is performed, the ROI is returned as the start and stop coordinates.

    :param domains: the domain information.
    :type domains: pandas.dataframe
    :param roi_coord: the region of interest's start and stop coordinates.
    :type roi_coord: list
    :return: the ROI identity.
    :rtype: str
    """
    try:
        roi_id = domains.loc[(domains["start"] == roi_coord[0]) & (domains["stop"] == roi_coord[1])]["domain"].values[0]
    except IndexError:
        roi_id = f"{roi_coord[0]}-{roi_coord[1]}"
        logging.warning(f"no domains match with the coordinates {roi_id} in the domains CSV file provided, this "
                        f"coordinates will be used to named the Region Of Interest instead of a domain name.")
    return roi_id


def update_domains(df, domains, out_dir, params):
    """
    Get the domains for pairs acceptor and donor.

    :param df: the dataframe of unique residues pairs forming hydrogen bonds.
    :type df: pd.Dataframe
    :param domains: the domain's coordinates.
    :type domains: pd.Dataframe
    :param out_dir: the path output directory.
    :type out_dir: str
    :param params: the parameters used in the previous trajectory hydrogen bonds analysis.
    :type params: dict
    :return: the pairs hydrogen bonds dataframe updated with the regions.
    :rtype: pd.Dataframe
    """
    donors_regions = [None] * len(df)
    acceptors_regions = [None] * len(df)
    for idx, row_hydrogen_bonds in df.iterrows():
        for _, row_domains in domains.iterrows():
            if row_domains["start"] <= row_hydrogen_bonds["ROI partner position"] <= row_domains["stop"]:
                donors_regions[idx] = row_domains["domain"]
            if row_domains["start"] <= row_hydrogen_bonds["second partner position"] <= row_domains["stop"]:
                acceptors_regions[idx] = row_domains["domain"]
    df.insert(3, "ROI partner domain", pd.DataFrame(donors_regions))
    df.insert(6, "second partner domain", pd.DataFrame(acceptors_regions))
    out_path = os.path.join(out_dir, f"outliers_{params['sample'].replace(' ', '_')}.csv")
    df.to_csv(out_path, index=False)
    logging.info(f"Pairs residues hydrogen bonds updated with domains saved: {out_path}")
    return df


def acceptors_domains_involved(df, domains, out_dir, params, roi_id, fmt, res_dist):
    """
    Create the plot of hydrogen bonds by regions.

    :param df: the dataframe.
    :type df: pd.Dataframe
    :param domains: the domains.
    :type domains: pd.Dataframe
    :param out_dir: the path of the output directory.
    :type out_dir: str
    :param params: the parameters used in the previous trajectory hydrogen bonds analysis.
    :type params: dict
    :param roi_id: the region of interest name.
    :type roi_id: str
    :param fmt: the format for the plot.
    :type fmt: str
    :param res_dist: the maximal residues distance in the amino acids chain.
    :type res_dist: int
    """
    data = {}
    for _, row_domains in domains.iterrows():
        tmp = df[df["second partner domain"] == row_domains["domain"]]
        if not tmp.empty:
            if not row_domains["domain"] in data:
                data[row_domains["domain"]] = 0
            for _, row_tmp in tmp.iterrows():
                data[row_domains["domain"]] += row_tmp["number atoms hydrogen bonds"]
    source = pd.DataFrame.from_dict({"domain": data.keys(), "number of hydrogen bonds": data.values()})

    # set the seaborn plots style and size
    sns.set_style("darkgrid")
    sns.set_context("poster", rc={"grid.linewidth": 2})
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.barplot(data=source, ax=ax, x="domain", y="number of hydrogen bonds", color="blue")

    # modify the ticks labels for the X axis by adding new lines every 3 words
    modified_x_labels = [re.sub(r'(\w+ \w+ \w+)( )',
                                r'\1\n', x_label.get_text()) for x_label in ax.get_xticklabels()]
    # set the number of ticks for the X axis to avoid a matplotlib warning
    ax.set_xticks([num_tick for num_tick in range(len(modified_x_labels))])
    ax.set_xticklabels(modified_x_labels, rotation=45, horizontalalignment="right")

    ax.set_xlabel(None)
    ax.set_ylabel(f"Hydrogen bonds with {roi_id}", fontweight="bold")
    ax.text(x=0.5, y=1.1, s=f"{params['sample']}: Hydrogen bonds with {roi_id}",
            weight="bold", ha="center", va="bottom", transform=ax.transAxes)
    md_duration = f", MD: {params['parameters']['time']}" if "time" in params['parameters'] else ""
    ax.text(x=0.5, y=1.0,
            s=f"Maximal atoms distance: {params['parameters']['maximal atoms distance']} \u212B, minimal angle cut-off "
              f"{params['parameters']['angle cutoff']}°, minimal residues distance: {res_dist}\n"
              f"{params['parameters']['proportion hbonds']}% of hydrogen bonds in {params['frames']} "
              f"frames{md_duration}",
            alpha=0.75, ha="center", va="bottom", transform=ax.transAxes)
    path = os.path.join(out_dir, f"outliers_{params['sample'].replace(' ', '_')}.{fmt}")
    fig.savefig(path, bbox_inches="tight")
    logging.info(f"Contacts by domain plot saved: {path}")


if __name__ == "__main__":
    descr = f"""
    {os.path.basename(__file__)} v. {__version__}

    Created by {__author__}.
    Contact: {__email__}
    {__copyright__}

    Distributed on an "AS IS" basis without warranties or conditions of any kind, either express or implied.

    From a CSV file of the hydrogen bonds during a molecular dynamics simulation and a YAML parameters file, produced 
    by the script trajectories_hbonds.py (https://github.com/njeanne/trajectories_hbonds), a heatmap representing the 
    residues hydrogen bonds.
    
    A Region Of Interest (ROI) is defined with a range of amino acids selected in the protein, on the heatmap the 
    hydrogen bonds on the ordinate axis will be the ones belonging to this ROI.

    If a domains CSV file is used with the option "--domains", a plot and a CSV file of the hydrogen bonds by domains 
    will be produced. An example of the domains CSV file is provided in data/traj_test_domains.csv
    For this CSV, if some domains are embedded in other domains, they can be displayed in the outputs with the option 
    "--use-embedded".
    """
    parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", required=True, type=str, help="the path to the output directory.")
    parser.add_argument("-d", "--domains", required=False, type=str, default="",
                        help="the path to the CSV domains file. A comma separated file, the first column is the "
                             "annotation name, the 2nd is the residue start coordinate, the 3rd is the residue end "
                             "coordinate, the last one is the color to apply in hexadecimal format. The coordinate are "
                             "1-indexed.")
    parser.add_argument("-p", "--parameters", required=True, type=str,
                        help="the path to the trajectory hydrogen bonds analysis parameters (the YAML file in the "
                             "results directory of the trajectory_hbonds.py script.")
    parser.add_argument("-r", "--roi", required=True, type=str,
                        help="the donors Region Of Interest (ROI) amino acids coordinates, the format should be two "
                             "digits separated by an hyphen, i.e: '100-200'.")
    parser.add_argument("-f", "--format", required=False, default="svg",
                        choices=["eps", "jpg", "jpeg", "pdf", "pgf", "png", "ps", "raw", "svg", "svgz", "tif", "tiff"],
                        help="the output plots format: 'eps': 'Encapsulated Postscript', "
                             "'jpg': 'Joint Photographic Experts Group', 'jpeg': 'Joint Photographic Experts Group', "
                             "'pdf': 'Portable Document Format', 'pgf': 'PGF code for LaTeX', "
                             "'png': 'Portable Network Graphics', 'ps': 'Postscript', 'raw': 'Raw RGBA bitmap', "
                             "'rgba': 'Raw RGBA bitmap', 'svg': 'Scalable Vector Graphics', "
                             "'svgz': 'Scalable Vector Graphics', 'tif': 'Tagged Image File Format', "
                             "'tiff': 'Tagged Image File Format'. Default is 'svg'.")
    parser.add_argument("-x", "--residues-distance", required=False, type=int, default=4,
                        help="when 2 atoms of different residues are in contact, the minimal distance in number of "
                             "residues that should separate them for a long range interaction. Default is 4 residues, "
                             "the number of residues in an alpha helix.")
    parser.add_argument("-e", "--embedded-domains", required=False, action="store_true",
                        help="for the outliers plot of hydrogen bonds between a specific domain and the whole protein, "
                             "use the domains embedded in another domain. In example, if the domain 2 is in domain 1, "
                             "the plot will represent the domain 1 as: domain-1 domain-2 domain-1. If this option is "
                             "not used only the domain 1 will be used in the plot.")
    parser.add_argument("-l", "--log", required=False, type=str,
                        help="the path for the log file. If this option is skipped, the log file is created in the "
                             "output directory.")
    parser.add_argument("--log-level", required=False, type=str,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="set the log level. If the option is skipped, log level is INFO.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("input", type=str, help="the donors/acceptors hydrogen bonds CSV file.")
    args = parser.parse_args()

    # create the logger
    create_log(args.log, args.log_level, args.out)
    check_optional_args(args.domains, args.embedded_domains, args.residues_distance)

    # get the Region Of Interest if specified
    roi_limits = extract_roi(args.roi)

    # load the hydrogen bonds analysis parameters
    parameters_hydrogen_bonds_analysis = None
    try:
        parameters_hydrogen_bonds_analysis = get_hydrogen_bonds_analysis_parameters(args.parameters)
    except ImportError as exc:
        logging.error(exc, exc_info=True)
        sys.exit(1)

    # extract the hydrogen bonds
    residues_hydrogen_bonds = None
    try:
        residues_hydrogen_bonds = extract_residues_hydrogen_bonds(args.input, roi_limits)
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

    if args.domains:
        # load and format the domains file
        domains_data = get_domains(args.domains, args.embedded_domains)
        # match the ROI coordinates with a domain
        roi = extract_roi_id(domains_data, roi_limits)
    else:
        roi = f"{roi_limits[0]}-{roi_limits[1]}"
        domains_data = None

    # get the heatmaps of validated hydrogen bonds by residues
    heatmap_hydrogen_bonds(residues_hydrogen_bonds, parameters_hydrogen_bonds_analysis, args.out, args.format, roi)

    outliers = outliers_hydrogen_bonds(residues_hydrogen_bonds, args.residues_distance)
    logging.info(f"{len(outliers)} unique residues pairs forming hydrogen bonds (<= "
                 f"{parameters_hydrogen_bonds_analysis['parameters']['maximal atoms distance']} \u212B) with a "
                 f"distance of at least {args.residues_distance} residues"
                 f"{' in the region of interest '+args.roi if args.roi else ''} (residues pair may have multiple atoms "
                 f"hydrogen bonds).")

    if args.domains:
        # get the outliers hydrogen bonds
        outliers = update_domains(outliers, domains_data, args.out, parameters_hydrogen_bonds_analysis)
        # by acceptor domain
        acceptors_domains_involved(outliers, domains_data, args.out, parameters_hydrogen_bonds_analysis, roi, args.format,
                                   args.residues_distance)
