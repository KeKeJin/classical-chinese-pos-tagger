import conllu

tagSet = ['AUX', 'NUM', 'VERB', 'CCONJ', 'SYM', 'PRON', 'PROPN', 'INTJ', 'ADP', 'NOUN', 'ADV', 'SCONJ', 'PART']


def parse_data(sentences):
    tokens = []
    tagged_tokens = []
    for tokenList in sentences:
        tagsForSentence = []
        tokensForSentence = []
        for token in tokenList:
            tag = token['upos']
            word = token['form']
            tagsForSentence.append(tag)
            tokensForSentence.append(word)
        tagged_tokens.append(( tokensForSentence, tagsForSentence))
        tokens.extend(tokensForSentence)
    return tokens, list(tagged_tokens)

def get_tagged_tokens(dir):
    with open(dir, 'r', encoding = 'utf-8') as file:
        data = file.read()
    sentences = conllu.parse(data)
    tokens, tagged_tokens = parse_data(sentences)
    return tokens, tagged_tokens
