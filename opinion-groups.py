import numpy as np
import pandas as pd
import sys
import csv

def sort_groups(data):
    """Sorts each participant into their respective opinion group."""

    group0 = []
    group1 = []
    group2 = []

    for i in range(0, 73):
        row = nd[i,1:17]
        group = nd[i,17]

        if int(group) == 0:
            group0.append(row)
        if int(group) == 1:
            group1.append(row)
        if int(group) == 2:
            group2.append(row)

    return [group0, group1, group2]

def calculate_percentiles(group):
    """Calculates the percentage of agrees, disagrees, and passes for each question."""

    n_questions = len(group[0])

    agree_count = [0] * n_questions
    disagree_count = [0] * n_questions
    pass_count = [0] * n_questions

    for participant in group:
        for question in range(n_questions):
            if participant[question] == 1.0:
               agree_count[question] = agree_count[question] + 1
            if participant[question] == 0.0:
               disagree_count[question] = disagree_count[question] + 1
            if participant[question] == 0.5:
               pass_count[question] = pass_count[question] + 1

    agree_avg = [(i / len(group))*100 for i in agree_count]
    disagree_avg = [(i / len(group))*100 for i in disagree_count]
    pass_avg = [(i / len(group))*100 for i in pass_count]

    return [agree_avg, disagree_avg, pass_avg]

def calculate_consensus(groups):
    """Calculates consensus between three groups."""
    n_questions = len(groups[0])

    consensus = [0] * n_questions

    for i in range(n_questions):
        consensus[i] = abs(((groups[0][i]+groups[1][i]+groups[2][i])/3) - 50)

    return consensus

def calculate_individuality(main_group, comparison_groups):
    """Determines how one group's agree percentage compares to other groups."""

    diff_1 = []
    diff_2 = []
    diff = []

    for i in range(len(main_group)):
        diff_1.append(main_group[i]-comparison_groups[0][i])
        diff_2.append(main_group[i]-comparison_groups[1][i])
        diff.append((abs(main_group[i]-comparison_groups[0][i])+abs(main_group[i]-comparison_groups[1][i]))/2)

    return [diff_1, diff_2, diff]

def create_group_df(group, individuality):
    res = pd.DataFrame(group[0], columns=['Agree'])
    res['Disagree'] = group[1]
    res['Pass'] = group[2]
    res['Difference from First Group'] = individuality[0]
    res['Difference from Second Group'] = individuality[1]
    res['Average Difference'] = individuality[2]

    return res

def save_group_percentiles(data):
    """Saves the percentage of agrees, disagrees, and passes for each question for each group"""

    groups = sort_groups(data)

    group0 = calculate_percentiles(groups[0])
    group1 = calculate_percentiles(groups[1])
    group2 = calculate_percentiles(groups[2])

    group0_agree = group0[0]
    group1_agree = group1[0]
    group2_agree = group2[0]

    consensus = calculate_consensus([group0_agree, group1_agree, group2_agree])
    consensus = pd.DataFrame(consensus, columns=['Consensus'])
    consensus.to_csv(r'~/Documents/Consensus.csv')

    group0_individuality = calculate_individuality(group0_agree, [group1_agree, group2_agree])
    group1_individuality = calculate_individuality(group1_agree, [group0_agree, group2_agree])
    group2_individuality = calculate_individuality(group2_agree, [group0_agree, group1_agree])

    group0_res = create_group_df(group0, group0_individuality)
    group1_res = create_group_df(group1, group1_individuality)
    group2_res = create_group_df(group2, group2_individuality)

    group0_res.to_csv(r'~/Documents/Group0.csv')
    group1_res.to_csv(r'~/Documents/Group1.csv')
    group2_res.to_csv(r'~/Documents/Group2.csv')

    print("Saved to Documents folder")
    return

def save_majority_percentiles(data):
    """Saves the percentage of agrees, disagrees, and passes for each question for everyone"""

    majority = []

    for i in range(0, 73):
        row = nd[i,1:17]
        majority.append(row)

    majority = calculate_percentiles(majority)

    majority_res = pd.DataFrame(majority[0], columns=['Agree'])
    majority_res['Disagree'] = majority[1]
    majority_res['Pass'] = majority[2]

    majority_res.to_csv(r'~/Documents/Majority.csv')

    print("Saved to Documents folder")
    return


if __name__ == "__main__":
    filename = sys.argv[1]

    df = pd.read_csv(filename)
    nd = df.to_numpy()

    save_group_percentiles(nd)
