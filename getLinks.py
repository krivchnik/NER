
#! .env/Scripts/python
# -*- coding: utf-8 -*-
import os
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.moses import MosesDetokenizer
from nltk.tokenize.moses import MosesTokenizer
from nltk.tokenize import word_tokenize
import codecs
import json
import ahocorasick # 1 is person,  2 is location, 3 is organization


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


def insert_links(sentence, links, sentence_tokens): # всё ещё едут ссылки немного. закрывающиеся скобки?
    links.sort(key=sort_by_length, reverse=True)
    links.sort(key=sort_by_first)
    print ("In insert_links")
    print (links)
    is_already_linked_marker = [0] * len(sentence_tokens) # храним и помечаем проставленные ссылки, чтобы случайно не поставить короткую и длинную на одно и то же
    special_characters_list = [', ', ' (', ') ', ' >', ' <'] # Символы, которые увеличивают смещение и считаются отдельными токенами
    special_characters_sentence_list = [', ', ' (', ') ', ' >', ' <', ' \"'] # ещё посмотреть
    for link in links:
        amount_of_spaces = sentence.count(' ', 0, link[0])
        if amount_of_spaces > len(sentence_tokens):
            assert False, "что-то не так во вставке ссылок"
        amount_of_special_tokens_in_sentence = sum(sentence.count(x, 0, link[0]) for x in special_characters_sentence_list)
        additional_tokens_in_link = sum(sentence.count(x, link[0], link[1]) for x in special_characters_list)
        if is_already_linked_marker[amount_of_spaces + amount_of_special_tokens_in_sentence] == 1:
            continue
        # print (is_already_linked_marker)
        is_already_linked_marker[amount_of_spaces + amount_of_special_tokens_in_sentence :
        amount_of_spaces + amount_of_special_tokens_in_sentence + additional_tokens_in_link + 1] = [1] * (additional_tokens_in_link + 1)
        if link[2] == 1:
            sentence_tokens[amount_of_spaces + amount_of_special_tokens_in_sentence] = "<PER>" + sentence_tokens[amount_of_spaces + amount_of_special_tokens_in_sentence]
            sentence_tokens[amount_of_spaces + amount_of_special_tokens_in_sentence + additional_tokens_in_link] = sentence_tokens[amount_of_spaces + amount_of_special_tokens_in_sentence + additional_tokens_in_link] + "</PER>"
        elif link[2] == 2:
            sentence_tokens[amount_of_spaces + amount_of_special_tokens_in_sentence] = "<LOC>" + sentence_tokens[amount_of_spaces + amount_of_special_tokens_in_sentence]
            sentence_tokens[amount_of_spaces + amount_of_special_tokens_in_sentence + additional_tokens_in_link] = sentence_tokens[amount_of_spaces + amount_of_special_tokens_in_sentence + additional_tokens_in_link] + "</LOC>"
        elif link[2] == 3:
            sentence_tokens[amount_of_spaces + amount_of_special_tokens_in_sentence] = "<ORG>" + sentence_tokens[amount_of_spaces + amount_of_special_tokens_in_sentence]
            sentence_tokens[amount_of_spaces + amount_of_special_tokens_in_sentence + additional_tokens_in_link] = sentence_tokens[amount_of_spaces + amount_of_special_tokens_in_sentence + additional_tokens_in_link] + "</ORG>"
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
    for lines in pops:
        known_words.add_word(lines.rstrip('\r\n'), (2, lines.rstrip('\r\n')))
    for lines in orgs:
        known_words.add_word(lines.rstrip('\r\n'), (3, lines.rstrip('\r\n')))
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
    if synonyms.intersection(person_set):
        person_link_set = person_link_set.union(synonyms)
    if synonyms.intersection(org_set):
        org_link_set = org_link_set.union(synonyms)
    if synonyms.intersection(pop_set):
        pop_link_set = pop_link_set.union(synonyms)
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
            tokenized[counter] = stemmer.stem(tokenized[counter])
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
                pos = article.find("[[", pos + 2)
                text = ''
                index = 0
                temp_set = set()
                temp_set.add(article_name)
                temp_set = temp_set.union(find_synonims(article_name, synonyms_all_data))
                # person_link_set, pop_link_set, org_link_set = add_to_set(temp_set, person_set, org_set, pop_set, person_link_set, pop_link_set, org_link_set)
                person_link_set, pop_link_set, org_link_set = add_to_set_automaton(known_words, temp_set, person_link_set, pop_link_set, org_link_set) # добавили в сеты
                old_pos = 0
                while pos > 0: #ищем все ссылки
                    old_pos = pos
                    isProper = False # совпадает ли текст ссылки и сама ссылка
                    if (article.find("|", pos + 1) > article.find("]]", pos + 1)) or article.find("|", pos + 1) == -1:
                        isProper = True
                    if not isProper:
                        index = article.find("|", pos)
                        text = article[pos + 2: index]
                    else:
                        index = article.find("]]", pos)
                        text = article[pos + 2: index]
                    if text.startswith("Файл:") or text.startswith("Категория:"):
                        pos = article.find("[[", pos + 2)
                        continue
                    synonyms = set()
                    synonyms.add(text)
                    print (text)
                    if not isProper:
                        pos = article.find("]]", index)
                        syn = article[index + 1: pos]
                        synonyms.add(syn)
                    synonyms = synonyms.union(find_synonims(text, synonyms_all_data)) # поиск синонимов
                    # person_link_set, pop_link_set, org_link_set = add_to_set(synonyms, person_set, org_set, pop_set, person_link_set, pop_link_set, org_link_set)
                    person_link_set, pop_link_set, org_link_set = add_to_set_automaton(known_words, synonyms, person_link_set, pop_link_set, org_link_set) # добавили в сеты
                    pos = article.find("[[", pos + 1)
                    if old_pos > pos:
                        break
                pos = -1
                pop_link_list_stemmed = tokenize_and_stem_set(pop_link_set, True, False) # нужно перемешивать, не использовать части как независимые
                org_link_list_stemmed = tokenize_and_stem_set(org_link_set, False, False) # не перемешиваем и не используем части как независимые
                per_link_list_stemmed = tokenize_and_stem_set(person_link_set, True, True) # и перемешиваем и рассматриваем части как независимые
                # pop_link_list_stemmed.sort(key=sort_by_length, reverse = True)
                # per_link_list_stemmed.sort(key=sort_by_length, reverse = True)
                # org_link_list_stemmed.sort(key=sort_by_length, reverse = True)
                add_to_automaton(per_link_list_stemmed, 1, automaton)
                add_to_automaton(pop_link_list_stemmed, 2, automaton)
                add_to_automaton(org_link_list_stemmed, 3, automaton)
                automaton.make_automaton()
                # получили списки стемов синонимов
                wikilink_rx = re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]') # для удаления ссылок
                article = wikilink_rx.sub(r'\1', article)
                # print(article)
                article_sentences = sentence_tokenizer.tokenize(article) # разбили на предложения
                sentence_number = 0
                previous_sentence = ''
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
                        stemmed_sentence.append(stemmer.stem(token))
                    sentence_to_work_with = ' '.join(detokenizer_with_fixes(stemmed_sentence))
                    sentence_links = []
                    for end_index, (typ, original_value) in automaton.iter(sentence_to_work_with):
                        start_index = end_index - len(original_value) + 1
                        if (end_index - start_index) <= 2: # одинокие буквы в случае если не стоит точка при инициалах
                            continue
                        sentence_links.append((start_index, end_index, typ))
                    print (sentence_to_work_with)
                    sentence_to_work_with = insert_links(sentence_to_work_with, sentence_links, sentence_tokens)
                    sentence_to_work_with = sentence_to_work_with + sentence_original[-1]
                    article_sentences[sentence_number] = sentence_to_work_with
                    sentence_number += 1
                person_link_set.clear()
                org_link_set.clear()
                pop_link_set.clear()
                article = ' '.join(article_sentences)
                out.write(article + "\n")
                out.close()
persons.close()
orgs.close()
pops.close()
