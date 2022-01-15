from bs4 import BeautifulSoup
from bs4.element import Tag
import re
import os
import pandas as pd
import numpy as np
from uuid import uuid4
import datetime
import json
from difflib import SequenceMatcher
import pdfminer.layout
import pdfminer.high_level
from screenply import validate, constants


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def run_pdf2txt(source, target, pages=[]):
    with open(target, 'w') as f_out:
        with open(source, 'rb') as f_in:
            pdfminer.high_level.extract_text_to_fp(
                f_in, 
                f_out, 
                laparams=pdfminer.layout.LAParams(), 
                output_type='html', 
                codec=None
            )


def convert_position_to_num(x):
    if isinstance(x, str) or isinstance(x, unicode):
        x = re.sub('[^0-9]', '', x)
        if x:
            return float(x)
        else:
            return None
    return None


def get_format(script_length):
    for t in constants.PAGE_MAX_BY_FORMAT:
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


def is_in_range(leftpos, min_pos, max_pos):
    return leftpos >= min_pos and leftpos <= max_pos

    
def get_style_attributes(element):
    element_attributes = {}
    style_attrs = element.attrs['style'].split('; ')
    leave_out = ['writing-mode', 'position']
    for attr in style_attrs:
        prop, value = attr.split(':')
        if prop not in leave_out:
            element_attributes[prop] = value
    return element_attributes


def get_text(c):
    if isinstance(c, BeautifulSoup) or isinstance(c, Tag):
        return c.text
    else:
        return c


def get_div_contents(div):
    return ''.join([get_text(c) for c in div.contents])


def validate_element(element_attributes):
    required_keys = ['top', 'left', 'height', 'width']
    for k in required_keys:
        if k not in element_attributes:
            return False
    return True


def is_page_break(element):
    style_attrs = get_style_attributes(element)
    a_tag = element.find('a')
    if a_tag:
        try:
            int(a_tag.attrs['name'])
            return True
        except:
            return False


def title_from_path(path):
    return path.split('/')[-1].replace('.html', '').replace('.pdf', '')


class Screenplay(object):
    """
    Reads a properly-formatted screenplay from a PDF*, and returns a 
        pandas DataFrame containing tagged script units. Also contains
        a validation step to "give up" on badly-formatted scripts and log
        errors.

    *If you have a previously parsed PDF and have saved the temp html file,
        you can instantiate a Screenplay object faster by passing that as the source.

    Uses pdfminer to convert a PDF to an HTML document, then uses BeautifulSoup
        to extract data from the HTML.

    Parameters
    ----------
    source: str, default None
        path pointing to a PDF to be converted
    source: str, default None
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
        result DataFrame containing tagged screenplay units.
        if validation fails, this will be empty
    failure_info : dict, default None
        if validation fails, this contains info on the failure
    title : str
        stripped title from file path
    soup : BeautifulSoup
        BeautifulSoup object containing HTML version of data

    Notes
    -----
    
    See Also
    --------
    """
    def __init__(self, source=None, temp_html_path=None, 
                 debug_mode=False, pages=[], validation=True,
                 failure_path=None):

        self.source = source
        self.temp_html_path = temp_html_path
        self.debug_mode = debug_mode
        self.failure_info = {}
        self.failure_path = failure_path
        self.raw_data = pd.DataFrame()
        self.pages = pages
        self.validation = validation

        if self.source.lower().endswith('.html'):
            self.title = title_from_path(self.source)
            self.soup = BeautifulSoup(open(self.source,'r'), "html.parser")
            self.raw_data = self.soup_to_data(self.soup)
        else:
            self.title = title_from_path(self.source)
            self.temp_html_path = self.temp_html_path or '{}.html'.format(self.title)
            self.soup = self.pdf_to_soup(self.pages)
            self.raw_data = self.soup_to_data(self.soup)

        self.drop_empty_lines()
        # self.drop_shared_lines()
        # check_for_headers = True
        # while check_for_headers:
        #     check_for_headers = self.drop_headers()
        
        if len(self.raw_data) > 0:
            self.label_lines()
            self.collapse_multi_line_units()
            self.identify_characters()
            self.identify_scenes()

        self.raw_data = self.raw_data.assign(title=self.title)
        last_page = self.raw_data.page.max()
        self.raw_data = self.raw_data.assign(format=get_format(last_page))
        self.data = self.raw_data[constants.DISPLAY_COLUMNS]

        if self.validation:
            self.validate()

    #############
    #######

    # PDF / HTML PARSERS 
    
    #######
    #############

    def pdf_to_soup(self, pages):
        run_pdf2txt(source=self.source, target=self.temp_html_path, pages=pages)
        if os.stat(self.temp_html_path).st_size == 0:
            return ''
        else:
            soup = BeautifulSoup(open(self.temp_html_path,'r'), "html.parser")
        if not self.debug_mode:
            os.remove(self.temp_html_path)
        return soup

    def soup_to_elements(self, elements):
        """
        input: HTML data
        output: dataframe containing info on each html element
        """
        data = []
        current_page = 0
        for i, element in enumerate(elements):
            if is_page_break(element):
                current_page += 1            
            row = {}
            row['element_id'] = i
            row['page'] = current_page
            row.update(get_style_attributes(element))
            if validate_element(row):
                data.append(row)

        df = pd.DataFrame(
            data, 
            columns=constants.ELEMENT_LEVEL_COLS
        )

        # convert position columns to numeric
        positional_cols = ['left','height','top','width']
        for positional_col in positional_cols:
            df[positional_col] = df[positional_col].apply(convert_position_to_num)

        # calculate prior/following whitespace
        df['bottom'] = df['top'] + df['height']
        df = df.sort_values('top')
        df['prior_whitespace'] = df.top - df.bottom.shift(1)
        df['following_whitespace'] = df.top.shift(-1) - df.bottom

        return df

    def element_to_lines(self, element, element_id):
        # separates an element into lines, each tagged with element metadata
        lines_data = []        
        lines = split_br(element)

        for line_num, line in enumerate(lines):
            line_data = {}
            line_data['element_id'] = element_id
            line_data['line_num'] = line_num
            line_data['line_id'] = "%s_%s" % (element_id, line_num)
            contents = get_div_contents(BeautifulSoup(line, "html.parser"))
            line_data['text'] = contents.strip()
            line_data['raw_text'] = line
            lines_data.append(line_data)

        return lines_data

    def soup_to_data(self, soup):
        elements = soup.find('body').find_all('div')
        self.elements_df = self.soup_to_elements(elements)

        lines_data = []
        for i, element in enumerate(elements):
            element_lines = self.element_to_lines(element, i)
            lines_data += element_lines

        lines_df = pd.DataFrame(
            lines_data, 
            columns=constants.LINE_LEVEL_COLS
        )
        lines_df.text = lines_df.text.apply(lambda s: s.strip())
        lines_df['n_chars'] = lines_df.text.apply(len).astype(int)


        data = pd.merge(self.elements_df, lines_df, on='element_id')
        return data

    #############
    #######

    # CLEAN UP DATAFRAME 
    
    #######
    #############

    def check_for_headers(self):
        page_tops = self.raw_data.groupby('page').min().top
        first_lines_mask = (self.raw_data.top.isin(page_tops)) & (self.raw_data.line_num == 0)

        firsts = self.raw_data[first_lines_mask].groupby('page').agg({'raw_text': lambda arr: ' '.join(arr)})

        # strip page numbers
        firsts['raw_text'] = firsts.raw_text.apply(lambda s: re.sub('[0-9]', '', s))

        firsts['last_page_header'] = firsts.raw_text.shift(1).fillna(' ')
        firsts['similarity'] = firsts.apply(lambda arr: similar(arr[0], arr[1]), axis=1)
        header_similarity = firsts.similarity.mean()
        if header_similarity > .75:
            self.raw_data = self.raw_data[~first_lines_mask]
            first_line_elements = self.raw_data[first_lines_mask].element_id
            mask = self.raw_data.element_id.isin(first_line_elements)
            # should I just index after this instead of reindexing here?
            self.raw_data.loc[mask, 'line_num'] = self.raw_data.loc[mask, 'line_num'] - 1
            self.raw_data = self.raw_data.reset_index(drop=True)
            return True
        return False

    def drop_empty_lines(self):
        self.raw_data = self.raw_data[~self.raw_data.left.isnull()]
        self.raw_data = self.raw_data[self.raw_data.text.str.strip() != '']

    def drop_shared_lines(self):
        # simultaneous dialogue is very confusing for screenply,
        # this removes two divs with identical vertical positions
        shared_line_positions = self.raw_data[self.raw_data.duplicated(['top'])].top.unique()
        mask = self.raw_data.top.isin(shared_line_positions)
        self.raw_data = self.raw_data[~mask]

    def collapse_multi_line_units(self):
        by = constants.ELEMENT_LEVEL_COLS + constants.UNIT_LEVEL_COLS + constants.CLASSIFIERS

        methods = {
            'text': lambda arr: ' '.join(arr),
            'raw_text': lambda arr: ' '.join(arr),
            'line_num': np.max,
            'n_chars': np.sum,
        }
        self.raw_data = self.raw_data.groupby(by, as_index=False).agg(methods)

    #############
    #######

    # CLASSIFY SCREENPLAY ELEMENTS
    
    #######
    #############

    def is_character(self, row):
        text = strip_parentheticals(row.text)
        is_upper = text == text.upper()
        first_line = row.line_num == 0
        in_middle = is_in_range(row.left, 130, 300) and is_in_range(row.width, 0, 300) # need to figure out that width max
        prior_whitespace = row.prior_whitespace >= constants.WHITESPACE_MIN
        return is_upper and in_middle and first_line and prior_whitespace and not self.is_parenthetical(row)

    def is_action(self, row):
        is_left_align = is_in_range(row.left, 0, 130)
        no_int_ext = row.text[:4] not in constants.SCENE_HEADING_INDICATORS
        return is_left_align and no_int_ext

    def is_parenthetical(self, row):
        in_middle = is_in_range(row.left, 130, 300)
        no_prior_whitespace = row.prior_whitespace < constants.WHITESPACE_MIN
        not_first_line = row.line_num > 0
        parentheses = row.text[0]+row.text[-1] == '()'
        return in_middle and (no_prior_whitespace or not_first_line) and parentheses 

    def is_dialogue(self, row):
        in_middle = is_in_range(row.left, 130, 300)
        no_prior_whitespace = row.prior_whitespace < constants.WHITESPACE_MIN
        not_first_line = row.line_num > 0
        return in_middle and (no_prior_whitespace or not_first_line) and not self.is_parenthetical(row)

    def is_scene_heading(self, row):
        is_left_align = is_in_range(row.left, 0, 120)
        has_int_ext = row.text[:4] in constants.SCENE_HEADING_INDICATORS
        return is_left_align and has_int_ext

    def is_page_header(self, row):
        # not guaranteed to work all the time
        not_left_align = is_in_range(row.left, 120, 800)
        next_line_is_new_line = row.following_whitespace > constants.WHITESPACE_MIN
        return not_left_align and next_line_is_new_line

    def classify_unit(self, row):
        if self.is_character(row):
            return 'character'
        # if self.is_page_header(row):
        #     return 'page_header'
        if self.is_action(row):
            return 'action'
        if self.is_dialogue(row):
            return 'dialogue'
        if self.is_scene_heading(row):
            return 'scene_heading'
        if self.is_parenthetical(row):
            return 'parenthetical'
    
    def label_lines(self):
        self.raw_data['unit_type'] = self.raw_data.apply(self.classify_unit, axis=1)
        rules = {
            'is_character': self.is_character,
            'is_action': self.is_action,
            'is_parenthetical': self.is_parenthetical,
            'is_dialogue': self.is_dialogue,
            'is_scene_heading': self.is_scene_heading
        }
        for classifier in constants.CLASSIFIERS:
            self.raw_data[classifier] = self.raw_data.apply(rules[classifier], axis=1)

    #############
    #######

    # IDENTIFY ELEMENT PARENTS
    
    #######
    #############
        
    def identify_characters(self):
        # remove parentheticals from character names, so that
        # JIM (VO) and JIM are identified as the same character
        mask = self.raw_data.unit_type=='character'
        self.raw_data.loc[mask, 'text'] = self.raw_data.loc[mask, 'text'].apply(strip_parentheticals)

        # forward fill characters
        self.raw_data = self.raw_data.sort_values('top')
        self.raw_data.loc[self.raw_data.is_character, 'character'] = self.raw_data.loc[self.raw_data.is_character, 'text']
        self.raw_data.character = self.raw_data.character.fillna(method='ffill')

        # clear character for non-dialogue/non-parenthetical units
        mask = self.raw_data.unit_type.isin(['action', 'scene_heading'])
        self.raw_data.loc[mask, 'character'] = None

        # add a unique id, ordered by frequency
        characters = self.raw_data.pivot_table(
            values=['height'], 
            index=['character'], 
            aggfunc='sum'
        )
        characters = characters[(~characters.index.isnull()) & (characters.index!='')]
        characters = characters.sort_values('height', ascending=False).reset_index()
        characters['character_id'] = characters.index
        self.raw_data = pd.merge(self.raw_data, characters[['character_id', 'character']], on='character', how='left')


    def identify_scenes(self):
        self.raw_data = self.raw_data.sort_values('top')
        self.raw_data.loc[self.raw_data.is_scene_heading, 'scene'] = self.raw_data.loc[self.raw_data.is_scene_heading, 'text']
        
        # add a unique id (different method from character
        # since two identical scene headings are still unique scenes
        self.raw_data.loc[self.raw_data.is_scene_heading, 'scene_id'] = range(len(self.raw_data.loc[self.raw_data.is_scene_heading]))

        self.raw_data['scene'] = self.raw_data.scene.fillna(method='ffill')
        self.raw_data['scene_id'] = self.raw_data.scene_id.fillna(method='ffill').fillna(9999)

    #############
    #######

    # VALIDATION
    
    #######
    #############

    def failure(self, reason, value=None):
        self.raw_data = pd.DataFrame()
        now = str(datetime.datetime.now())
        self.failure_info = {'title': self.title, 'date': now, 'reason': reason, 'value': value}
        if self.failure_path:
            with open(self.failure_path, 'a') as f:
                f.write(json.dumps(self.failure_info) + "\n")

    def validate(self):
        failure_msg = validate.validate(self.raw_data)
        if failure_msg:
            self.failure(failure_msg)

    #############
    #######

    # I/O STUFF
    
    #######
    #############

    def view_around(self, top, window=200):
        return self.data[self.data.top.between(top-window, top+window)]

    def save_raw(self, filename=None):
        filename = filename or self.source.split('.pdf')[0].split('/')[-1]+'.json'
        save_cols = ['left', 'height', 'top', 'width', 'text']
        self.data[save_cols].to_json(filename)

    def save_full(self, filename=None):
        filename = filename or self.source.split('.pdf')[0].split('/')[-1]+'.json'
        self.data.to_json(filename)
