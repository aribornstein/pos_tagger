
# Constants
SLASH = "/"
SPACE = " "
NEW_LINE = "\n"
TAB = "\t"
START_SIGHT = "*START*"
BUFFER_SIZE = 2
UNK = "*UNK*"
END = "*END*"

import re
HMM_PATTERNS =  [\
(re.compile(r'.*ing$'), 'VBG'),  # gerunds
(re.compile(r'.*ed$'), 'VBD'),  # simple past
(re.compile(r'.*es$'), 'VBZ'),  # 3rd singular present
(re.compile(r'.*ould$'), 'MD'),  # modals
(re.compile(r'.*s$'), 'NNS'),  # plural nouns
(re.compile(r'^-?[0-9]+(.[0-9]+)?$'), 'CD'),  # cardinal numbers
(re.compile(r'^[A-Z]+[a-z]*$'), 'NNP'),  # proper nouns
(re.compile(r'.*'), 'NN') # nouns (default)
]

FEATURE_PATTERNS = [
(re.compile(r'[0-9]+(.[0-9]+)$'), 'contains_number'),  # cardinal numbers
(re.compile(r'[A-Z\d]'),'contains_upper'),
(re.compile(r'\w+(?:-\w+)+'),'contains_hyphen')
]