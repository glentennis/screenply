import subprocess
from bs4 import BeautifulSoup
import re
import os
import pandas as pd
import numpy as np
from uuid import uuid4
import datetime
import json

# general
STOPWORDS = ['']

# scene headings
SCENE_HEADING_INDICATORS = ['INT.', 'EXT.']
SCENE_HEADING_TIMES_OF_DAY = ['DAY', 'NIGHT']
SCENE_HEADING_STOPWORDS = ['-']

# ignore elements if futher left than (in pixels):
IGNORE_PAST_LEFT = 400

# minimum space between lines to be considered an 
# empty line (in pixels). Chosen semi-arbitrarily:
# too low -> caps dialogue gets labeled as character
WHITESPACE_MIN = 5

# rough maximum page counts for different formats
PAGE_MAX_BY_FORMAT = [
    ('other', 20),
    ('half_hour', 45),
    ('hour', 70),
    ('feature', 9999)
]

# page to pixel ratio (approximation, determined empirically)
PIXELS_PER_PAGE = 840.0

##
# SPECIFY COLUMNS
## 
DIV_LEVEL_COLS = ['div_id',
                'height',
                'left',
                'top',
                'width',
                'bottom',
                'prior_whitespace']

LINE_LEVEL_COLS = ['text', 'raw_text', 'line_num', 'div_id', 'n_chars']

ELEMENT_LEVEL_COLS = ['element_type']

CLASSIFIERS = ['is_action', 'is_dialogue', 'is_character', 'is_parenthetical', 'is_scene_heading']

# columns that point to a "parent" item the row belongs to (i.e. character, scene)
PARENT_COLUMNS = ['character', 'character_id', 'scene', 'scene_id']

OTHER_COLUMNS = ['title', 'page', 'format']


def run_pdf2txt(source, target):
    path = "screenply/pdf2txt.py"
    subprocess.call(["python", path, "-o", target, source])


def convert_position_to_num(x):
    if isinstance(x, str) or isinstance(x, unicode):
        x = re.sub('[^0-9]', '', x)
        if x:
            return float(x)
        else:
            return None
    return None


def get_format(script_length):
    for t in PAGE_MAX_BY_FORMAT:
        if script_length <= t[1]:
            return t[0]
    return t[0]


def split_br(div):
    return re.split('<.{0,1}br.{0,1}>', str(div))


def strip_parentheticals(s):
    pattern = r"\(.*\)"
    while re.search(pattern, s):
        s = re.sub(pattern, '', s)
    return s.strip()


def strip_html_tags(s):
    pattern = r"<.*>"
    while re.search(pattern, s):
        s = re.sub(pattern, '', s)
    return s.strip()


def get_scene_time_of_day(scene_heading):
    for t in SCENE_HEADING_TIMES_OF_DAY:
        n_chars = len(t)
        if scene_heading[:n_chars*-1] == t:
            return t.lower()
    return None


def clean_scene_heading(scene_heading):
    for t in SCENE_HEADING_TIMES_OF_DAY:
        scene_heading = scene_heading.replace(t, '')
    for w in SCENE_HEADING_STOPWORDS:
        scene_heading = scene_heading.replace(w, '')
    return scene_heading.strip()


def is_in_range(leftpos, min_pos, max_pos):
    return leftpos >= min_pos and leftpos <= max_pos

    
def get_div_attributes(div):
    div_attributes = {}
    style_attrs = div.attrs['style'].split('; ')
    leave_out = ['writing-mode', 'position', 'border']
    for attr in style_attrs:
        prop, value = attr.split(':')
        if prop not in leave_out:
            div_attributes[prop] = value
    return div_attributes


def title_from_path(path):
    return path.split('/')[-1].replace('.html', '').replace('.pdf', '')


class Screenplay(object):
    """
    Reads a properly-formatted screenplay from a PDF*, and returns a 
        pandas DataFrame containing tagged script elements. Also contains
        a validation step to "give up" on badly-formatted scripts and log
        errors.

    *If you have a previously parsed PDF and have saved the temp html file,
        you can instantiate a Screenplay object faster using the source_html
        parameter

    Uses pdf2txt.py to convert a PDF to an HTML document, then uses BeautifulSoup
        to extract data from the HTML.

    Parameters
    ----------
    source_pdf: str, default None
        path pointing to a PDF to be converted
    source_html: str, default None
        path pointing to an intermediate HTML file of a previously 
        converted PDF for faster processing. if specified, this source 
        file will be used instead of a PDF
    temp_html_path: str, default None
        path to store the intermediate html file
    debug_mode: bool, default False
        if true, the intermediate html file will not be delelted, so that
        it can be viewed and re-used
.
    Attributes
    ----------
    data : pandas DataFrame
        result DataFrame containing tagged screenplay elements.
        if validation fails, this will be empty
    failure_info : dict, default None
        if validation fails, this contains info on the failure
    title : str
        stripped title from file path
    soup : BeautifulSoup
        BeautifulSoup object containing HTML version of data
    soup : BeautifulSoup
        BeautifulSoup object containing HTML version of data
    soup : BeautifulSoup
        BeautifulSoup object containing HTML version of data

    Notes
    -----
    
    See Also
    --------
    """
    def __init__(self, source_pdf=None, source_html=None, temp_html_path=None, 
                 debug_mode=False):

        self.source_pdf = source_pdf
        self.source_html = source_html
        self.temp_html_path = temp_html_path
        self.debug_mode = debug_mode
        self.failure_info = {}


        if self.source_html:
            self.title = title_from_path(self.source_html)
            # print(self.title)
            self.soup = BeautifulSoup(open(self.source_html,'r'), "html.parser")
            self.data = self.soup_to_data(self.soup)
        elif source_pdf:
            self.title = title_from_path(self.source_pdf)
            # print(self.title)
            self.temp_html_path = self.temp_html_path or '{}.html'.format(self.title)
            self.soup = self.pdf_to_soup()
            self.data = self.soup_to_data(self.soup)

        self.drop_shared_lines()
        self.drop_empty_lines()
        if len(self.data) > 0:
            self.label_rows()
            self.collapse_multi_line_elements()
            self.identify_characters()
            self.identify_scenes()

        self.data['title'] = self.title
        self.data['page'] = self.data.top / PIXELS_PER_PAGE

        last_page = self.data.page.max()
        self.data['format'] = get_format(last_page)       

        self.validate()

    #############
    #######

    # PDF / HTML PARSERS 
    
    #######
    #############

    def pdf_to_soup(self):
        # print(self.source_pdf)
        run_pdf2txt(source=self.source_pdf, target=self.temp_html_path)
        if os.stat(self.temp_html_path).st_size == 0:
            return ''
        else:
            soup = BeautifulSoup(open(self.temp_html_path,'r'), "html.parser")
        if not self.debug_mode:
            os.remove(self.temp_html_path)
        return soup

    def get_div_data(self, divs):
        # parse div elements
        data = []
        for i, div in enumerate(divs):
            row = {}
            row['div_id'] = i
            row.update(get_div_attributes(div))
            data.append(row)

        df = pd.DataFrame(data, columns=DIV_LEVEL_COLS)

        # convert position columns to numeric
        positional_cols = ['left','height','top','width']
        for positional_col in positional_cols:
            df[positional_col] = df[positional_col].apply(convert_position_to_num)

        # drop elements too far left (transitions, page numbers)
        df = df[df.left < IGNORE_PAST_LEFT]
        
        # calculate prior whitespace
        df['bottom'] = df['top'] + df['height']
        df = df.sort_values('top')
        df['prior_whitespace'] = df.top - df.bottom.shift(1)

        # find the smallest** div height to determine PDF line height
        # **use .02 quantile to filter out anomalies
        self.line_height = df.height.quantile(0.02)
        # print("line height is: %s" % self.line_height)

        return df

    def div_to_lines(self, div, div_id):
        # separates a div into lines, each tagged with div metadata
        lines_data = []        
        lines = split_br(div)

        for line_num, line in enumerate(lines):
            line_data = {}
            line_data['div_id'] = div_id
            line_data['line_num'] = line_num
            try:
                line_data['text'] = strip_html_tags(line)
                line_data['raw_text'] = strip_html_tags(line)
            except:
                print(div)
                print(lines)
                return
            lines_data.append(line_data)

        return lines_data

    def soup_to_data(self, soup):
        divs = self.soup.findAll('div')
        divs_df = self.get_div_data(divs)

        lines_data = []
        for i, div in enumerate(divs):
            div_lines = self.div_to_lines(div, i)
            lines_data += div_lines

        lines_df = pd.DataFrame(lines_data, columns=LINE_LEVEL_COLS)
        lines_df.text = lines_df.text.apply(lambda s: s.strip())
        lines_df['n_chars'] = lines_df.text.apply(len).astype(int)

        return pd.merge(divs_df, lines_df, on='div_id')

    def drop_empty_lines(self):
        self.data = self.data[~self.data.left.isnull()]
        self.data = self.data[self.data.text.str.strip() != '']

    def drop_shared_lines(self):
        # simultaneous dialogue is very confusing for screenply,
        # this removes two divs with identical vertical positions
        shared_line_positions = self.data[self.data.duplicated(['top'])].top.unique()
        mask = self.data.top.isin(shared_line_positions)
        self.data = self.data[~mask]

    def collapse_multi_line_elements(self):
        by = DIV_LEVEL_COLS + ELEMENT_LEVEL_COLS + CLASSIFIERS

        methods = {
            'text': lambda arr: ' '.join(arr),
            'raw_text': lambda arr: ' '.join(arr),
            'line_num': np.max,
            'n_chars': np.sum,
        }
        self.data = self.data.groupby(by, as_index=False).agg(methods)

    def is_character(self, row):
        text = strip_parentheticals(row.text)
        is_upper = text == text.upper()
        first_line = row.line_num == 0
        in_middle = is_in_range(row.left, 120, 300)
        prior_whitespace = row.prior_whitespace >= WHITESPACE_MIN
        return is_upper and in_middle and first_line and prior_whitespace and not self.is_parenthetical(row)

    def is_action(self, row):
        is_left_align = is_in_range(row.left, 0, 120)
        no_int_ext = row.text[:4] not in SCENE_HEADING_INDICATORS
        return is_left_align and no_int_ext

    def is_parenthetical(self, row):
        in_middle = is_in_range(row.left, 120, 300)
        no_prior_whitespace = row.prior_whitespace < WHITESPACE_MIN
        not_first_line = row.line_num > 0
        parentheses = row.text[0]+row.text[-1] == '()'
        return in_middle and (no_prior_whitespace or not_first_line) and parentheses 

    def is_dialogue(self, row):
        in_middle = is_in_range(row.left, 120, 300)
        no_prior_whitespace = row.prior_whitespace < WHITESPACE_MIN
        not_first_line = row.line_num > 0
        return in_middle and (no_prior_whitespace or not_first_line) and not self.is_parenthetical(row)

    def is_scene_heading(self, row):
        is_left_align = is_in_range(row.left, 0, 120)
        has_int_ext = row.text[:4] in SCENE_HEADING_INDICATORS
        return is_left_align and has_int_ext

    def classify_element(self, row):
        if self.is_character(row):
            return 'character'
        if self.is_action(row):
            return 'action'
        if self.is_dialogue(row):
            return 'dialogue'
        if self.is_scene_heading(row):
            return 'scene_heading'
        if self.is_parenthetical(row):
            return 'parenthetical'
    
    def label_rows(self):
        self.data['element_type'] = self.data.apply(self.classify_element, axis=1)
        rules = {
            'is_character': self.is_character,
            'is_action': self.is_action,
            'is_parenthetical': self.is_parenthetical,
            'is_dialogue': self.is_dialogue,
            'is_scene_heading': self.is_scene_heading
        }
        for classifier in CLASSIFIERS:
            self.data[classifier] = self.data.apply(rules[classifier], axis=1)
        
    def identify_characters(self):
        # remove parentheticals from character names, so that
        # JIM (VO) and JIM are identified as the same character
        mask = self.data.element_type=='character'
        self.data.loc[mask, 'text'] = self.data.loc[mask, 'text'].apply(strip_parentheticals)

        # forward fill characters
        self.data = self.data.sort_values('top')
        self.data.loc[self.data.is_character, 'character'] = self.data.loc[self.data.is_character, 'text']
        self.data.character = self.data.character.fillna(method='ffill')

        # clear character for non-dialogue/non-parenthetical elements
        mask = self.data.element_type.isin(['action', 'scene_heading'])
        self.data.loc[mask, 'character'] = None

        # add a unique id
        characters = pd.DataFrame(self.data.character.unique()).reset_index()
        characters.columns = ['character_id', 'character']
        self.data = pd.merge(self.data, characters, on='character', how='left')

        # fill NA
        # self.data['character'] = self.data['character'].fillna('NA')
        # self.data['character_id'] = self.data['character_id'].fillna(9999)

    def identify_scenes(self):
        self.data = self.data.sort_values('top')
        self.data.loc[self.data.is_scene_heading, 'scene'] = self.data.loc[self.data.is_scene_heading, 'text']
        
        # add a unique id (different method from character
        # since two identical scene headings are still unique scenes
        self.data.loc[self.data.is_scene_heading, 'scene_id'] = range(len(self.data.loc[self.data.is_scene_heading]))

        self.data['scene'] = self.data.scene.fillna(method='ffill')
        self.data['scene_id'] = self.data.scene_id.fillna(method='ffill').fillna(9999)
        # self.data['time_of_day'] = self.data.location.apply(get_scene_time_of_day)
        # self.data['location'] = self.data.location.apply(clean_scene_heading)

        # fill NA
        # self.data['scene'] = self.data['scene'].fillna('NA')
        # self.data['scene_id'] = self.data.scene_id.fillna(9999)

    def character_summary(self):
        return self.data.pivot_table(['is_dialogue','is_parenthetical','n_chars'],'character',aggfunc='sum')

    def failure(self, reason, value=None):
        self.data = pd.DataFrame()
        now = str(datetime.datetime.now())
        self.failure_info = {'title': self.title, 'date': now, 'reason': reason, 'value': value}

    def get_validation_summary(self):
        df_copy = self.data.copy()
        # generate variable to calculate avg dialogue length (later)
        mask = df_copy.element_type=='dialogue'
        df_copy.loc[mask, 'n_chars_dialogue'] = df_copy.loc[mask, 'n_chars']

        # generate n_spaces for each line to calculate spaces per char (later)
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

    def validate(self):
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
        if len(self.data) == 0:
            return self.failure('empty dataframe')

        # if not, empty, get summary statistics for validation
        summary = self.get_validation_summary()

        # is the left alignment inconsistent? (detects badly scanned scripts mostly)
        # here we look at the unique count of left alignment values for elements in the script
        if summary.left > 100:
            return self.failure('unique number of left alignment positions', summary.left)

        # common error: pdf reader doesn't put spaces between words
        spaces_per_char = summary.n_spaces / summary.n_chars
        if spaces_per_char < .05:
            return self.failure('spaces per char', spaces_per_char)

        # is there enough text per page
        # many ill-scanned screenplays will read with a lot of empty text
        n_chars_per_page = summary.n_chars / summary.page
        if n_chars_per_page < 200:
            return self.failure('chars per page', n_chars_per_page)

        # has elements with multiple classifications 
        mask = self.data[CLASSIFIERS].sum(axis=1) > 1
        if mask.sum() > 0:
            return self.failure('elements with more than one classification', mask.sum())

        # is there enough dialogue
        dialogue_per_page = summary.is_dialogue / summary.page
        if dialogue_per_page < 1:
            return self.failure('dialogue per page', dialogue_per_page)

        # is there enough action
        action_per_page = summary.is_action / summary.page
        if action_per_page < 1:
            return self.failure('action per page', action_per_page)

        # average dialogue length
        if summary.n_chars_dialogue < 10:
            return self.failure('avg length of dialogue (chars)', summary.n_chars_dialogue)

    def view_around(self, top, window=200):
        return self.data[self.data.top.between(top-window, top+window)]

    def save_raw(self, filename=None):
        filename = filename or self.source_pdf.split('.pdf')[0].split('/')[-1]+'.json'
        save_cols = ['left', 'height', 'top', 'width', 'text']
        self.data[save_cols].to_json(filename)

    def save_full(self, filename=None):
        filename = filename or self.source_pdf.split('.pdf')[0].split('/')[-1]+'.json'
        self.data.to_json(filename)
