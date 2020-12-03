import conllu

tagSet = [  'AUX',      # v,助動詞, ex: '敢'問崇德脩慝辨惑
            'NUM',      # n,数詞,数, ex: 顏淵篇第'十二'
            'VERB',     # v,動詞,行為, ex: 顏淵問'仁'
            'CCONJ',    # p,助詞,接続, ex: 公若鼓行'而'西
            'SYM',      # s,記号,一般
            'PRON',     # n,代名詞, ex: '己'所不欲勿施於人
            'PROPN',    # n,名詞,人, ex: '顏淵'篇第十二
            'INTJ',     # p,感嘆詞, ex: '噫'嘻我死矣
            'ADP',      # v,前置詞, ex: '於'斯三者何先
            'NOUN',     # n,名詞, ex: 顏淵問'仁'
            'ADV',      # v,副詞, ex: 非'禮'勿'視'
            'SCONJ',    # p,助詞,接続, ex:孝弟也者其為人'之'本與
            'PART',     # p,助詞,句末,接続,提示. ex:未若貧而樂道富而好禮'者'也
            'X'
            ]
accTagSet = [  'a',        # adjective
            'c',        # conjection, then
            'd',        # adverb, not
            'f',        # locative, front
            'j',        # combined, at there
            'm',        # number
            'n',        # noun
            'nr',       # noun, person's name
            'ns',       # location
            'p',        # prepositional, at
            'q',        # classifier, counter
            'r',        # pronoun, me
            's',        # onomatopoeia, lol
            't',        # time
            'u',        # aux, of
            'v',        # verb
            'w',        # poncuation
            'y',        # modal, interrogative 
            'other'     
            ]
UD_to_ACC = {
    'AUX': ('u'),      # v,助動詞, ex: '敢'問崇德脩慝辨惑
    'NUM': ('m'),      # n,数詞,数, ex: 顏淵篇第'十二'
    'VERB': ('v'),     # v,動詞,行為, ex: 顏淵問'仁'
    'CCONJ': ('c'),    # p,助詞,接続, ex: 公若鼓行'而'西
    'SYM': ('w'),      # s,記号,一般
    'PRON': ('r'),     # n,代名詞, ex: '己'所不欲勿施於人
    'PROPN': ('nr', 'ns'),    # n,名詞,人, ex: '顏淵'篇第十二
    'INTJ': ('s'),     # p,感嘆詞, ex: '噫'嘻我死矣
    'ADP': ('p'),      # v,前置詞, ex: '於'斯三者何先
    'NOUN': ('f', 'n', 'q', 't'),     # n,名詞, ex: 顏淵問'仁'
    'ADV': ('a', 'd', 'y'),      # v,副詞, ex: 非'禮'勿'視'
    'SCONJ': ('u'),    # p,助詞,接続, ex:孝弟也者其為人'之'本與
    'PART': ('j')      # p,助詞,句末,接続,提示. ex:未若貧而樂道富而好禮'者'也
}

ACC_to_UD = {
    'u': 'AUX',
    'a': 'ADV',
    'c': 'CCONJ',
    'd': 'ADV',
    'f': 'NOUN',
    'j': 'PART',
    'm': 'NUM',
    'n': 'NOUN',
    'nr':'PROPN',
    'ns': 'PROPN', # location is a form of proper noun
    'p': 'ADP',
    'q': 'NOUN',
    'r': 'PRON',
    's': 'INTJ',
    't': 'NOUN',
    'v': 'VERB',
    'w': 'SYM',
    'y': 'ADV',
    'other': 'X'
}

def parse_data_UD(sentences, universal = True):
    tokens = []
    tagged_tokens = []
    for tokenList in sentences:
        tagsForSentence = []
        tokensForSentence = []
        for token in tokenList:
            tag = token['upos']
            word = token['form']
            # if not universal:
                
            tagsForSentence.append(tag)
            tokensForSentence.append(word)
        tagged_tokens.append(( tokensForSentence, tagsForSentence))
        tokens.extend(tokensForSentence)
    return tokens, list(tagged_tokens)

def parse_data_ACC(sentences, universal = True):
    tokens = []
    tagged_tokens = []
    for tokenList in sentences:
        tagsForSentence = []
        tokensForSentence = []
        for token in tokenList:
            try:
                tagMarker = token.index('/')
                tag = token[tagMarker+1:]
                if tag not in accTagSet:
                    tag = 'other'
                if universal:
                    tag = ACC_to_UD[tag]
                word = token[:tagMarker]
                tagsForSentence.append(tag)
                tokensForSentence.append(word)
            except:
                pass
        tagged_tokens.append(( tokensForSentence, tagsForSentence))
        tokens.extend(tokensForSentence)
    return tokens, list(tagged_tokens)

def get_tagged_tokens(dirs, universal = True):
    tokensList = []
    tagged_tokens_List = []
    for dir in dirs:
        if dir[len(dir)-4:] == '.txt':
            with open(dir, 'r', encoding = 'utf-8') as file:
                data = file.read()
            data = data.splitlines()
            dataList = [i.split(" ") for i in data]
            tokens, tagged_tokens = parse_data_ACC(dataList,universal)
        else:
            with open(dir, 'r', encoding = 'utf-8') as file:
                data = file.read()
            sentences = conllu.parse(data)
            tokens, tagged_tokens = parse_data_UD(sentences, universal)
        tokensList += tokens
        tagged_tokens_List += tagged_tokens
    return tokensList, tagged_tokens_List
