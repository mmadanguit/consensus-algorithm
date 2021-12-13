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

    res = pd.DataFrame(agree_avg, columns=['Agree'])
    res['Disagree'] = disagree_avg
    res['Pass'] = pass_avg

    return res

def save_group_percentiles(data):
    """Saves the percentage of agrees, disagrees, and passes for each question for each group"""

    groups = sort_groups(data)

    group0 = calculate_percentiles(groups[0])
    group1 = calculate_percentiles(groups[1])
    group2 = calculate_percentiles(groups[2])

    group0.to_csv(r'~/Documents/Group0.csv')
    group1.to_csv(r'~/Documents/Group1.csv')
    group2.to_csv(r'~/Documents/Group2.csv')

    print("Saved to Documents folder")
    return

def save_majority_percentiles(data):
    """Saves the percentage of agrees, disagrees, and passes for each question for everyone"""

    majority = []

    for i in range(0, 73):
        row = nd[i,1:17]
        majority.append(row)

    majority = calculate_percentiles(majority)
    majority.to_csv(r'~/Documents/Majority.csv')

    print("Saved to Documents folder")
    return


def determine_individuality(data):

    print(res['Disagree'].to_numpy())


def calculate_consensus(filename):
    consensus = [0]*16

    for i in range(17):
        consensus[i] = abs(((group0_agree_avg[i]+group1_agree_avg[i]+group2_agree_avg[i])/3) - 50)

    consensus_res = pd.DataFrame(consensus, columns=['Consensus'])
    consensus_res.to_csv(r'~/Documents/Consensus.csv')


if __name__ == "__main__":
    filename = sys.argv[1]

    df = pd.read_csv(filename)
    nd = df.to_numpy()

    save_majority_percentiles(nd)
