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

##
# SPECIFY COLUMNS
## 
ELEMENT_LEVEL_COLS = ['element_id',
                'height',
                'left',
                'top',
                'width',
                'bottom',
                'prior_whitespace',
                'following_whitespace',
                'page']

LINE_LEVEL_COLS = ['line_id', 'text', 'raw_text', 'line_num', 'element_id', 'n_chars', 'is_first_on_page']

UNIT_LEVEL_COLS = ['unit_type'] # unit = piece of the script (i.e. a dialogue block or an action line)

CLASSIFIERS = ['is_action', 'is_dialogue', 'is_character', 'is_parenthetical', 'is_scene_heading']

# columns that point to the "parent" of an unit (i.e. dialogue belongs to character, 
#   character belongs to scene, etc)
PARENT_COLUMNS = ['character', 'character_id', 'scene', 'scene_id']

OTHER_COLUMNS = ['title', 'format']

DISPLAY_COLUMNS = [
    'element_id',
    'height',
    'left',
    'top',
    'width',
    'bottom',
    'page',
    'unit_type',
    'text',
    'raw_text',
    'line_num',
    'n_chars',
    'character',
    'character_id',
    'scene',
    'scene_id',
    'title',
    'format',
]
