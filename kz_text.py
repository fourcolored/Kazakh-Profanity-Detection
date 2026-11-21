import re, unicodedata

LATIN_TO_CYR = {'sh':'ш','ch':'ч','ya':'я','yu':'ю','a':'а','á':'ә','b':'б','d':'д','e':'е','f':'ф','g':'г','ǵ':'ғ',
                'h':'х','i':'і','ı':'и','j':'ж','k':'к','l':'л','m':'м','n':'н','ń':'ң','o':'о','ó':'ө','p':'п','q':'қ',
                'r':'р','s':'с','t':'т','u':'ұ','ú':'ү','v':'в','w':'у','x':'х','y':'ы','ý':'у','z':'з','c':'к'}
KAZAKH_TO_RUSSIAN = str.maketrans({'ә':'а','ғ':'г','қ':'к','ң':'н','ө':'о','ұ':'у','ү':'у','һ':'х','і':'и','ё':'е','й':'и'})

def clean_for_bert(text):
    text = unicodedata.normalize('NFKC', str(text)).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)            # links
    text = re.sub('[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]', ' ', text)   # emoji
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)                     # өтттте -> өтте
    text = re.sub(r'[^\w\s\*#@$]', ' ', text)                      # punctuation -> space (keep masks * # @ $)
    return re.sub(r'\s+', ' ', text).strip()

def latin_to_cyrillic(text):
    out, i = '', 0
    while i < len(text):
        if text[i:i+2] in LATIN_TO_CYR:   out += LATIN_TO_CYR[text[i:i+2]]; i += 2    # two-letter sounds first (sh, ch)
        elif text[i] in LATIN_TO_CYR:     out += LATIN_TO_CYR[text[i]];     i += 1
        else:                             out += text[i];                    i += 1
    return out

def clean_for_tfidf(text):
    text = latin_to_cyrillic(clean_for_bert(text)).translate(KAZAKH_TO_RUSSIAN)
    text = re.sub(r'(.)\1+', r'\1', text)                          # collapse all repeats
    return re.sub(r'\s+', ' ', text).strip()
