# -*- coding: utf-8 -*-
# import ijson
import bz2
import json
import ahocorasick


known_words = ahocorasick.Automaton()
known_words.add_word("первый второй", (1, "первый второй"))
known_words.add_word("второй", (3, "второй"))
known_words.add_word("второй", (2, "второй"))
known_words.make_automaton()
haystack = "На первый второй, рассчитайсь!"
for end_index, (insert_order, original_value) in known_words.iter(haystack):
    start_index = end_index - len(original_value) + 1
    print((start_index, end_index, (insert_order, original_value)))
    assert haystack[start_index:start_index + len(original_value)] == original_value

some_set = set()
some_set.add("word1")
some_set.add("word2")
some_set.add("второй")
for itms in some_set:
    result = known_words.get(itms, "nope")
    if result != "nope":
        print ("Yay!")
        print (result)

# def find_synonims(text):
# print ("lul")
# important = {}
# synonyms = bz2.BZ2File("C:/Users/2/Downloads/latest-all.json.bz2", 'rb')
# i = 0
# for obj in ijson.items(synonyms, 'item'):
#     try:
#         # if (obj['ru']['en']['value'] == text) or\
#         #         (obj['labels']['en-gb']['value'] == text):
#         name = obj['labels']['ru']['value']
#         aliases = obj['aliases']['ru']
#         important[name] = aliases
#         i+=1
#         if i%5000 == 0:
#             print ('Working number ', i)
#
#         #     break
#     except KeyError:
#         continue
# with open('synonyms.json', 'a') as f:
#     f.write(json.dumps(important))

# print find_synonims('Belgium')
# from nltk.stem.snowball import SnowballStemmer
# from nltk.tokenize.punkt import PunktSentenceTokenizer
# from nltk.tokenize import word_tokenize
# from nltk.tokenize.moses import MosesDetokenizer
# from nltk.tokenize.moses import MosesTokenizer
# import nltk
#
# # nltk.download()
# import re
# import string
#
# print "not lul"
# stemmer = SnowballStemmer("russian")
# s = "Хрень, со: знаками, препинания!"
# s = s[:-1]
# nltk.download()
# # s = re.sub('['+string.punctuation+']', '', s)
# print "seriously"
# tokens = word_tokenize("Хрень, со: знаками, препинания!")
# print tokens
# tokenizer = MosesTokenizer()
# tokenss = tokenizer.tokenize(s.decode('utf-8'))
# tokens = word_tokenize(s)
# print tokens
# print tokenss
# for token in tokens:
#     print stemmer.stem(token.decode('utf8'))
#
# detokenizer = MosesDetokenizer()
# # for tokensssss in tokenss:
# #     tokensssss = tokensssss.decode('utf8')
# print detokenizer.detokenize(tokenss)

