
#! .env/Scripts/python
# -*- coding: utf-8 -*-
import os
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.moses import MosesDetokenizer
from nltk.tokenize.moses import MosesTokenizer
from nltk.tokenize import word_tokenize
import codecs
import json
import ahocorasick # 1 is person,  2 is location, 3 is organization
import pymorphy2
# from rdflib import BNode, Literal, Namespace, Graph, term
# from rdflib.namespace import RDF
# text_annot = BNode() # для удобства обращения

# def create_rdf(text, g):
#
#     start = Literal(24)
#     end = Literal(42)
#     doc_text = Literal(text)
#     rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
#     persons = Namespace("http://www.abbyy.com/ns/BasicEntity#")
#     orgs = Namespace("http://www.abbyy.com/ns/Org#")
#     places = Namespace("http://www.abbyy.com/ns/Geo#")
#     annotations = Namespace("http://www.abbyy.com/ns/Aux#")
#     annotations.document_text = term.URIRef(u'http://www.abbyy.com/ns/Aux#document_text')
#     annotations.annotation = term.URIRef(u"http://www.abbyy.com/ns/Aux#annotation")
#     annotations.text_annotations = term.URIRef(u"http://www.abbyy.com/ns/Aux#TextAnnotations")
#     annotations.InstanceAnnotation = term.URIRef(u"http://www.abbyy.com/ns/Aux#InstanceAnnotation")
#     annotations.annotation_start = term.URIRef(u"http://www.abbyy.com/ns/Aux#annotation_start")
#     annotations.annotation_end = term.URIRef(u"http://www.abbyy.com/ns/Aux#annotation_end")
#     g.namespace_manager.bind("Aux", annotations)
#     g.namespace_manager.bind("Org", orgs)
#     g.namespace_manager.bind("BasicEntity", persons)
#     g.namespace_manager.bind("Geo", places)
#     g.namespace_manager.bind("rdfs", rdfs)
#     # g.add((bob, persons, persons))
#     # g.add((bob, FOAF.name, name))
#     # g.add((bob, FOAF.knows, linda))
#     # g.add((text, RDF.type, persons))
#     g.add((text_annot, RDF.type, annotations.text_annotation))
#     g.add((text_annot, annotations.document_text, doc_text))
#     # print (g.serialize(format='pretty-xml'))
#     return g

# def add_obj_with_annotation(link, offset, article_rdf, rdf):
#     to_add = BNode()
#     annotation = BNode()
#     if link[2] == 1:
#         rdf.add(to_add, RDF.type, persons)
#         rdf.add(text_annot, annotations.annotation, )


def sort_by_priority(dat):
    if dat[2] == first_priority:
        return 1
    else:
        return 0


def sort_by_first(x):
    return x[0]


def sort_by_length(x): # для сортировки tuple по длине
    return x[1] - x[0]
    # if isinstance(x, str):
    #     return (1, len(x))
    # elif isinstance(x, tuple):
    #     return (len(x), 0)
    # else:
    #     assert False, 'неизвестная зверушка'


def manual_link_remove(article, pos_in):
    index = article.find("]]", pos_in)
    text = article[pos_in + 2: index]
    article_out = article
    is_bad = False
    if text.startswith("Файл:") or text.startswith("Категория:"):
        is_bad = True

    is_prop = False
    if (article.find("|", pos_in + 1) > article.find("]]", pos_in + 1)) or (article.find("|", pos_in + 1) == -1):
        is_prop = True

    if is_bad:
        print ("Deleting bad link")
        if (article.find("[[", pos_in + 2) < article.find("]]", pos_in + 2)): # вложенные ссылки
            counter = 1
            pos_inner = pos_in
            while counter != 0:
                pos_inner_open = article.find("[[", pos_inner + 1)
                pos_inner_close = article.find("]]", pos_inner + 1)
                if pos_inner_open < pos_inner_close and pos_inner_open != -1:
                    counter += 1
                    pos_inner = pos_inner_open
                elif pos_inner_close != -1:
                    counter -= 1
                    pos_inner = pos_inner_close
                else:
                    assert False, "Something wrong in deleting link"
            article_out = article[:pos_in] + article[pos_inner + 2:]
        else:
            article_out = article[:pos_in] + article[article.find("]]", pos_in + 1) + 2:]
    else:
        if is_prop:
            # print ("Deleting proper link")
            # print (article[(pos_in+2):article.find("]]", pos_in + 2)])
            article_out = article[:pos_in] + article[(pos_in+2):article.find("]]", pos_in + 1)] + article[(article.find("]]", pos_in + 1) + 2):]
        else:
            # print ("Deleting inproper link")
            # print (article[(article.find("|", pos_in + 1) + 1): article.find("]]", pos_in + 1)])
            article_out = article[:pos_in] + article[(article.find("|", pos_in + 1) + 1): article.find("]]", pos_in + 1)] + article[(article.find("]]", pos_in + 1) + 2):]
    return article_out



def insert_links(sentence, links, sentence_tokens, first_priority ): #, offset, article_rdf, rdf
    print ("In insert_links")
    links.sort(key=sort_by_length, reverse=True)
    print(links)
    links.sort(key=sort_by_priority, reverse=True)
    links.sort(key=sort_by_first)
    print (links)
    is_already_linked_marker = [0] * len(sentence_tokens) # храним и помечаем проставленные ссылки, чтобы случайно не поставить короткую и длинную на одно и то же
    special_characters_list = [', ', ' (', ')', ' >', ' <', '% ', ': ', '; '] # Символы, которые увеличивают смещение и считаются отдельными токенами
    special_characters_sentence_list = [', ', ' (', ') ', ' >', ' <', ' \"', '% ', ': ', '; '] # ещё посмотреть
    for link in links:
        amount_of_spaces = sentence.count(' ', 0, link[0])
        if amount_of_spaces > len(sentence_tokens):
            assert False, "что-то не так во вставке ссылок"
        amount_of_special_tokens_in_sentence = sum(sentence.count(x, 0, link[0]) for x in special_characters_sentence_list)
        additional_tokens_in_link = sum(sentence.count(x, link[0], link[1]) for x in special_characters_list)
        spaces_in_link = sentence.count(" ", link[0], link[1])
        additional_tokens_in_link = spaces_in_link + additional_tokens_in_link
        additional_tokens_in_sentence = amount_of_spaces + amount_of_special_tokens_in_sentence
        if is_already_linked_marker[additional_tokens_in_sentence] == 1:
            # print (link, additional_tokens_in_sentence)
            continue
        # link[2] == 1 and
        if sentence_tokens[additional_tokens_in_sentence][0] == sentence_tokens[additional_tokens_in_sentence][0].lower():
            print ("LETTER", sentence_tokens[additional_tokens_in_sentence][0])
            print ("WORD", sentence_tokens[additional_tokens_in_sentence])
            continue
        is_already_linked_marker[additional_tokens_in_sentence : additional_tokens_in_sentence + additional_tokens_in_link + 1] = [1] * (additional_tokens_in_link + 1)
        # print (is_already_linked_marker)
        # add_obj_with_annotation(link, offset, article_rdf, rdf)
        if link[2] == 1:
            sentence_tokens[additional_tokens_in_sentence] = "<PER>" + sentence_tokens[additional_tokens_in_sentence]
            sentence_tokens[additional_tokens_in_sentence + additional_tokens_in_link] = sentence_tokens[additional_tokens_in_sentence + additional_tokens_in_link] + "</PER>"
        elif link[2] == 2:
            sentence_tokens[additional_tokens_in_sentence] = "<LOC>" + sentence_tokens[additional_tokens_in_sentence]
            sentence_tokens[additional_tokens_in_sentence + additional_tokens_in_link] = sentence_tokens[additional_tokens_in_sentence + additional_tokens_in_link] + "</LOC>"
        elif link[2] == 3:
            sentence_tokens[additional_tokens_in_sentence] = "<ORG>" + sentence_tokens[additional_tokens_in_sentence]
            sentence_tokens[additional_tokens_in_sentence + additional_tokens_in_link] = sentence_tokens[additional_tokens_in_sentence + additional_tokens_in_link] + "</ORG>"
    return ' '.join(detokenizer_with_fixes(sentence_tokens))
        



def add_to_automaton(container_with_data, type, automaton):
    for items in container_with_data:
        if not isinstance(items, str):
            item = ' '.join(detokenizer_with_fixes(items))
        else:
            item = items
        # print (item)
        automaton.add_word(item, (type, item))


def load_known(known_words, persons, pops, orgs):
    for lines in persons:
        known_words.add_word(lines.rstrip('\r\n'), (1, lines.rstrip('\r\n'))) # \r\n Windows-like ending
    for lines in orgs:
        known_words.add_word(lines.rstrip('\r\n'), (3, lines.rstrip('\r\n')))
    for lines in pops:
        known_words.add_word(lines.rstrip('\r\n'), (2, lines.rstrip('\r\n')))
    return known_words


def add_to_set_automaton(automaton, synonyms, person_link_set, pop_link_set, org_link_set):
    for itms in synonyms:
        result = automaton.get(itms, "nope")
        if result != "nope":
            if result[0] == 1:
                person_link_set = person_link_set.union(synonyms)
            if result[0] == 2:
                pop_link_set = pop_link_set.union(synonyms)
            if result[0] == 3:
                org_link_set = org_link_set.union(synonyms)
    return person_link_set, pop_link_set, org_link_set




def add_to_set(synonyms, person_set, org_set, pop_set, person_link_set, pop_link_set, org_link_set): # вынес добавление сетов наружу
    if first_priority != 0 and first_priority != 1:
        if first_priority == 2:
            if synonyms.intersection(pop_set):
                pop_link_set = pop_link_set.union(synonyms)
                return person_link_set, pop_link_set, org_link_set
            if synonyms.intersection(person_set):
                person_link_set = person_link_set.union(synonyms)
                return person_link_set, pop_link_set, org_link_set
            if synonyms.intersection(org_set):
                org_link_set = org_link_set.union(synonyms)
                return person_link_set, pop_link_set, org_link_set
        if first_priority == 3:
            if synonyms.intersection(org_set):
                org_link_set = org_link_set.union(synonyms)
                return person_link_set, pop_link_set, org_link_set
            if synonyms.intersection(pop_set):
                pop_link_set = pop_link_set.union(synonyms)
                return person_link_set, pop_link_set, org_link_set
            if synonyms.intersection(person_set):
                person_link_set = person_link_set.union(synonyms)
                return person_link_set, pop_link_set, org_link_set
    else:
        if synonyms.intersection(person_set):
            person_link_set = person_link_set.union(synonyms)
            return person_link_set, pop_link_set, org_link_set
        if synonyms.intersection(pop_set):
            pop_link_set = pop_link_set.union(synonyms)
            return person_link_set, pop_link_set, org_link_set
        if synonyms.intersection(org_set):
            org_link_set = org_link_set.union(synonyms)
            return person_link_set, pop_link_set, org_link_set
    return person_link_set, pop_link_set, org_link_set


def detokenizer_with_fixes(sentence_tokens): # небольшие костыли к детокенизатору (терял закрытие кавычек и не склеивал '(' с текстом после)
    quot_even = 0
    count = 0
    for token in sentence_tokens:
        try:
            if (token == '&quot;') and (quot_even == 0) and (count < len(sentence_tokens) - 1):
                if sentence_tokens[count + 1] == '&quot;':
                    sentence_tokens[count + 1] = '&quotquot;'
                else:
                    quot_even = 1
                    sentence_tokens[count + 1] = '\"' + sentence_tokens[count + 1]
                sentence_tokens.remove('&quot;')
                count -= 1
            elif token == '&quot;' and quot_even != 0: # out of range??
                quot_even = 0
                sentence_tokens[count] = sentence_tokens[count] + '\"'
                sentence_tokens.remove('&quot;')
                count -= 1
            elif token == '&quot;' and count >= len(sentence_tokens) - 1 and quot_even != 0:
                sentence_tokens[count] = sentence_tokens[count] + '\"'
            count += 1
        except ValueError:
            count += 1
            continue
    print (sentence_tokens)
    detokenized_sentence = detokenizer.detokenize(sentence_tokens)
    count = 0
    for item in detokenized_sentence:
        count += 1
        if item == '(':
            detokenized_sentence[count] = '(' + detokenized_sentence[count]
            detokenized_sentence.remove('(')
    count = 0
    for item in detokenized_sentence:
        count += 1
        if item == '((':
            detokenized_sentence[count] = '((' + detokenized_sentence[count]
            detokenized_sentence.remove('((')
    count = 0
    for item in detokenized_sentence:
        count += 1
        if item == '(((':
            detokenized_sentence[count] = '(((' + detokenized_sentence[count]
            detokenized_sentence.remove('(((')
    return detokenized_sentence


def find_synonims(text, data):
    terms = set()
    try:
        try:
            for val in data[str(text)]:
                terms.add(val['value'])
        except TypeError:
            for val in data[text]:
                terms.add(val['value'])
    except KeyError:
        pass
    return terms


def tokenize_and_stem_set(some_set, should_switch, should_use_parts): # получение стемов токенов по сету
    output_list = []
    for elem in some_set:
        tokenized = word_tokenize(str(elem))
        counter = len(tokenized)
        while counter > 0:
            counter -= 1
            # tokenized[counter] = stemmer.stem(tokenized[counter])
            tokenized[counter] = lemmatizer.parse(tokenized[counter].lower())[0].normal_form # Убрать Великий и прочие
        for el in tokenized:
            if el == "," or el == "-" or el == "(" or el == ")" or el == "."\
                        or el == '``' or el == "\"" or el == "\'"\
                    or el == "—" or el == '„': # Убираем знаки препинания и скобки. Вообще нужно ещё почистить от слов вида "река" и "город" # а может и не нужно
                tokenized.remove(el)
                counter -= 1
        output_list.append(tuple(tokenized))
        counter = len(tokenized) #switching to list of tuples now
        if should_switch:
            temp_list = tokenized
            output_list.append(tuple(temp_list.__reversed__()))
            while counter > 0:
                temp = temp_list.pop()
                if should_use_parts and not temp.endswith('.'):
                    output_list.append(temp,)
                    output_list.append(tuple(temp_list))
                temp_list.insert(0, temp)
                output_list.append(tuple(temp_list))
                counter -= 1
    temp_for_removing_duplicates = []
    for k in output_list:
        if k not in temp_for_removing_duplicates:
            temp_for_removing_duplicates.append(k)
    output_list = temp_for_removing_duplicates
    return output_list

def get_article_name(art):
    name_start = art.find("title=\"")
    name_start = art.find("\"", name_start + 1)
    name_end = art.find("\"", name_start + 1)
    return art[name_start+1:name_end]


# НАЧАЛО
print('Привет, мир!')
synonyms_all = codecs.open("synonyms.json", "r", encoding='utf-8')
synonyms_all_data = json.loads(synonyms_all.read())
dir_path = os.getcwd()
article_list = []
print(dir_path)
path = str(dir_path) + "/extracted/good_examples"
print(path)
persons = codecs.open("persons.txt", 'r', encoding='utf-8')
orgs = codecs.open("orgs.txt", 'r', encoding='utf-8')
pops = codecs.open("pops.txt", 'r', encoding='utf-8')
person_set = set()
org_set = set()
pop_set = set()
person_link_set = set()
org_link_set = set()
pop_link_set = set()
detokenizer = MosesDetokenizer()
tokenizer = MosesTokenizer()
stemmer = SnowballStemmer("russian")
lemmatizer = pymorphy2.MorphAnalyzer()
sentence_tokenizer = PunktSentenceTokenizer()
known_words = ahocorasick.Automaton()
known_words = load_known(known_words, persons, pops, orgs)
known_words.make_automaton()
for dirp, dirn, files in os.walk(path):
    for fl in files:
        p = str(dirp) + '/' + str(fl)
        print (p)
        with codecs.open(p, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            article_list = text.split(r"</doc>")
            print("LIST LENGTH = ", (len(article_list) - 1))
            pos = -1
            print (article_list)
            article_list = article_list[:-1] # Последняя это пробелы после последней осмысленной статьи
            for article in article_list:

                automaton= ahocorasick.Automaton()
                automaton = load_known(automaton, persons, pops, orgs)

                article_name = get_article_name(article)
                print ("ARTICLE = ", article_name)
                outpath = str(dir_path) + '/output/'
                out = codecs.open(outpath + article_name + ".txt", 'a', encoding='utf-8', errors='ignore')
                article_name = article_name
                pos = article.find('<')
                pos = article.find('>', pos)
                article = article[pos+1:] # удаляем шапку с <doc>
                pos = -1
                pos = article.find("[[")
                text = ''
                index = 0
                temp_set = set()
                temp_set.add(article_name)
                temp_set = temp_set.union(find_synonims(article_name, synonyms_all_data))
                # person_link_set, pop_link_set, org_link_set = add_to_set(temp_set, person_set, org_set, pop_set, person_link_set, pop_link_set, org_link_set)
                person_link_set, pop_link_set, org_link_set = add_to_set_automaton(known_words, temp_set, person_link_set, pop_link_set, org_link_set) # добавили в сеты
                first_priority = 0
                if len(person_link_set) > 0:
                    first_priority = 1
                elif len(pop_link_set) > 0:
                    first_priority = 2
                elif len(org_link_set) > 0:
                    first_priority = 3
                print ("PRIORITY", first_priority)
                old_pos = 0
                while pos > 0: #ищем все ссылки
                    old_pos = pos
                    isProper = False # совпадает ли текст ссылки и сама ссылка
                    if (article.find("|", pos) > article.find("]]", pos)) or (article.find("|", pos) == -1):
                        isProper = True
                    if not isProper:
                        index = article.find("|", pos)
                        text = article[pos + 2: index]
                    else:
                        index = article.find("]]", pos)
                        text = article[pos + 2: index]
                    if text.startswith("Файл:") or text.startswith("Категория:"):
                        article = manual_link_remove(article, pos)
                        pos = article.find("[[")
                        if old_pos > pos:
                            break
                        continue
                    synonyms = set()
                    synonyms.add(text)
                    # print (text)
                    pos0 = pos # to give into function
                    if not isProper:
                        pos = article.find("]]", index)
                        syn = article[index + 1: pos]
                        synonyms.add(syn)
                    else:
                        pos += 2 # for next offset consistency
                    article = manual_link_remove(article, pos0)
                    synonyms = synonyms.union(find_synonims(text, synonyms_all_data)) # поиск синонимов
                    # person_link_set, pop_link_set, org_link_set = add_to_set(synonyms, person_set, org_set, pop_set, person_link_set, pop_link_set, org_link_set)
                    person_link_set, pop_link_set, org_link_set = add_to_set_automaton(known_words, synonyms, person_link_set, pop_link_set, org_link_set) # добавили в сеты
                    pos = article.find("[[")
                    if old_pos > pos:
                        break
                pos = -1
                pop_link_list_stemmed = tokenize_and_stem_set(pop_link_set, True, False) # нужно перемешивать,  не использовать части как независимые
                org_link_list_stemmed = tokenize_and_stem_set(org_link_set, False, False) # не перемешиваем и не используем части как независимые
                per_link_list_stemmed = tokenize_and_stem_set(person_link_set, True, True) # и перемешиваем и рассматриваем части как независимые
                if first_priority != 0 and first_priority != 1:
                    if first_priority == 2:
                        add_to_automaton(org_link_list_stemmed, 3, automaton)
                        add_to_automaton(per_link_list_stemmed, 1, automaton)
                        add_to_automaton(pop_link_list_stemmed, 2, automaton)
                    elif first_priority == 3:
                        add_to_automaton(per_link_list_stemmed, 1, automaton)
                        add_to_automaton(pop_link_list_stemmed, 2, automaton)
                        add_to_automaton(org_link_list_stemmed, 3, automaton)
                else:
                    add_to_automaton(org_link_list_stemmed, 3, automaton)
                    add_to_automaton(pop_link_list_stemmed, 2, automaton)
                    add_to_automaton(per_link_list_stemmed, 1, automaton)

                automaton.make_automaton()
                # article_rdf = article
                # g = Graph()
                # rdf = create_rdf(text, g)
                article_sentences = sentence_tokenizer.tokenize(article) # разбили на предложения
                sentence_number = 0
                previous_sentence = ''
                # offset = 0
                for sentence_original in article_sentences:
                    sentence_to_work_with = sentence_original[:-1] # отрезали пунктуатор (!.?)
                    sentence_to_work_with = sentence_to_work_with.replace("„", "\"")
                    sentence_to_work_with = sentence_to_work_with.replace("“", "\"")
                    try:
                        sentence_tokens = tokenizer.tokenize(sentence_to_work_with)
                    except NameError:
                        sentence_tokens = tokenizer.tokenize(previous_sentence + ' ' + sentence_to_work_with)
                        article_sentences[sentence_number - 1] = ''
                    # print(sentence_tokens)
                    previous_sentence = sentence_to_work_with
                    token_number_in_sentence = 0
                    skip_how_many_tokens = 0 # в случае если обработали подпоследовательность
                    stemmed_sentence = []
                    for token in sentence_tokens:
                        # stemmed_sentence.append(stemmer.stem(token))
                        stemmed_sentence.append(lemmatizer.parse(token.lower())[0].normal_form)
                    sentence_to_work_with = ' '.join(detokenizer_with_fixes(stemmed_sentence))
                    sentence_links = []
                    roman_numbers = []
                    roman_numbers.append("I")
                    roman_numbers.append("V")
                    roman_numbers.append("X")

                    try:
                        for end_index, (typ, original_value) in automaton.iter(sentence_to_work_with):
                            start_index = end_index - len(original_value) + 1
                            if (end_index - start_index) <= 2 and not any((c in article[start_index:end_index])
                                                                          for c in roman_numbers ): # одинокие буквы в случае если не стоит точка при инициалах
                                continue
                            sentence_links.append((start_index, end_index, typ))
                    except AttributeError:
                        pass
                    print (sentence_to_work_with)
                    sentence_to_work_with = insert_links(sentence_to_work_with, sentence_links, sentence_tokens, first_priority) #, offset, article_rdf, rdf)
                    # sentence_to_work_with = ' '.join(detokenizer_with_fixes(sentence_tokens))
                    sentence_to_work_with = sentence_to_work_with + sentence_original[-1]
                    article_sentences[sentence_number] = sentence_to_work_with
                    sentence_number += 1
                    # offset += len(sentence_to_work_with) + 1
                person_link_set.clear()
                org_link_set.clear()
                pop_link_set.clear()
                article = ' '.join(article_sentences)
                out.write(article + "\n")
                out.close()
persons.close()
orgs.close()
pops.close()
