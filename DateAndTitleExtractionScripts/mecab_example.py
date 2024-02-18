import MeCab

# MeCab Initialization
mecab = MeCab.Tagger()

def filter_morphs(text):
    nodes = mecab.parseToNode(text)
    filtered_words = []
    while nodes:
        print(f"Surface: {nodes.surface}, Feature: {nodes.feature}")
        if nodes.feature.split(",")[0] in ["名詞", "動詞", "形容詞"]:
            filtered_words.append(nodes.surface)
        nodes = nodes.next
    return " ".join(filtered_words)

# Separate text to morpheme
text = "子供3人の大学無償化への日韓の反応の違い"
print(mecab.parse(text))
print(filter_morphs(text))