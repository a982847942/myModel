import re
import spacy
from spacy.lang.zh import Chinese

nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])

def clean_text(text):
    """
    This function cleans the text in the following ways:
    1. Replace websites with URL
    1. Replace 's with <space>'s (eg, her's --> her 's)

    """

    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "URL",
                  text)  # Replace urls with special token
    text = text.replace("\'s", "")
    text = text.replace("\'", "")
    text = text.replace("n\'t", " n\'t")
    text = text.replace("@", "")
    text = text.replace("#", "")
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("&amp;", "")
    text = text.replace("&gt;", "")
    text = text.replace("\"", "")
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("(", "")
    text = text.replace(")", "")

    text = ' '.join(text.split())

    return text.strip()


def clean_tokenized_text(text_lst):
    if len(text_lst) <= 1:
        return text_lst

    idx = 0
    cleaned_token_lst = []

    while idx < len(text_lst) - 1:

        current_token = text_lst[idx]
        next_token = text_lst[idx + 1]

        if current_token != next_token:
            cleaned_token_lst.append(current_token)
            idx += 1

        else:

            last_idx = max([i + idx for i, val in enumerate(text_lst[idx:]) if val == current_token]) + 1
            cleaned_token_lst.append(current_token)
            idx = last_idx

    if cleaned_token_lst[-1] != text_lst[-1]:
        cleaned_token_lst.append(text_lst[-1])

    return cleaned_token_lst


def tokenize_text(text):
    text = clean_text(text)
    # print(text)

    # token_lst = [token.text.lower() for token in nlp(text)]
    # token_lst = clean_tokenized_text(token_lst)
    return  get_token(text)


import string
from nltk.corpus import stopwords
def get_token(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic 保留只包含字母的token
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words 去除停用词
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens 去掉长度过短的词
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [token.lower() for token in tokens]
    return ' '.join(tokens)

print(tokenize_text("the sky is so3 blue http://www.baidu.com"))
