import numpy as np
import pandas as pd
from screenply import validate, constants


def get_validation_summary(data):
    df_copy = data.copy()
    # generate variable to calculate avg dialogue length later
    mask = df_copy.unit_type=='dialogue'
    df_copy.loc[mask, 'n_chars_dialogue'] = df_copy.loc[mask, 'n_chars']

    # generate n_spaces for each line to calculate spaces per char later
    df_copy['n_spaces'] = df_copy.text.apply(lambda s: s.count(' '))

    # generate summary table
    methods = {
        #     'character':pd.Series.nunique,
        'page': np.max,
        'left': pd.Series.nunique,
        'is_dialogue': np.sum,
        'is_action': np.sum,
        'n_chars_dialogue': np.mean,
        'n_spaces': np.sum,
        'n_chars': np.sum
    }

    return df_copy.aggregate(methods)


def validate(data):
    """
    A set of heuristics to identify bad PDFs, based on generous,
        approximate thresholds determined by looking at outliers.

    If a failure is detected, set dataframe to empty so that this PDF
        won't contribute bad data in a batch job.

    The different criteria are sorted in order of importance. For example,
        if a script is a bad scan, we want to log that as the error 
        (rather than all the other errors the script will trigger) 
        because the bad scan is the root of the problem.

    Other ideas:
        - scripts with odd left/right boundaries
        - is the line height really big / really small
        - characters per script
    """

    # first, check for empty dataframe
    if len(data) == 0:
        return 'Empty DataFrame'

    # if not, empty, get summary statistics for validation
    summary = get_validation_summary(data)

    # is the left alignment inconsistent? (detects badly scanned scripts mostly)
    # here we look at the unique count of left alignment values for units in the script
    # if summary.left > 100:
    #     return self.failure('unique number of left alignment positions', summary.left)

    # common error: pdf reader doesn't put spaces between words
    spaces_per_char = summary.n_spaces / summary.n_chars
    if spaces_per_char < .05:
        return 'Spaces per text: {}'.format(spaces_per_char)

    # is there enough text per page
    # many ill-scanned screenplays will read with a lot of empty text
    n_chars_per_page = summary.n_chars / summary.page
    if n_chars_per_page < 200:
        return 'Text per page: {}'.format(n_chars_per_page)

    # has units with multiple classifications 
    mask = data[constants.CLASSIFIERS].sum(axis=1) > 1
    if mask.sum() > 0:
        return 'Units with more than one classification: {}'.format(mask.sum())

    # is there enough dialogue
    dialogue_per_page = summary.is_dialogue / summary.page
    if dialogue_per_page < 1:
        return 'Dialogue per page: {}',format(dialogue_per_page)

    # is there enough action
    action_per_page = summary.is_action / summary.page
    if action_per_page < 1:
        return 'Action per page: {}'.format(action_per_page)

    # average dialogue length
    if summary.n_chars_dialogue < 10:
        return 'Avg length of dialogue {}'.format(summary.n_chars_dialogue)

    # if self.check_for_headers():
    #     return self.failure('headers detected')

    return False
