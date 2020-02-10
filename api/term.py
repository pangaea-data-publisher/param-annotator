import requests
import urllib.parse
import json
from unidecode import unidecode
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
from nltk.tokenize import RegexpTokenizer
import re, unicodedata
import logging
from operator import itemgetter
from difflib import SequenceMatcher
from nltk.util import ngrams
import itertools
import string
import nltk.stem.snowball
logging.getLogger("Elasticsearch").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from collections import Counter
import math
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
class Term:
    global stemmer,tokenizer
    stemmer = SnowballStemmer("english")
    tokenizer = RegexpTokenizer(r'\w+')
    # replace pang_replace patterns  with ""
    pang_replace_onqual = ["aff\.", "cf\.?", "\scomplex$", "ex gr\.", "gr\.", "nov\.", "subgen\.", "gen\.?",
                    "ng\.", "g\.\ssp.", "sp\.", "spp\.", "indeterminata", "undifferentiated", "ind\.", "ssp\.",
                    "subsp\.", "sensu lato", "sensu stricto"]

    pang_replace_general =["\.?\-?group$", "\-?type$", "agg\."]
    global split_words
    split_words = ['the','and', 'aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'anti',
                   'around', 'as', 'at', 'before', 'behind','below', 'between', 'beneath', 'beside', 'besides', 'beyond', 'but', 'by', 'concerning',
                   'considering', 'despite', 'down', 'during', 'except', 'excepting', 'excluding', 'following', 'for',
                   'from', 'in','inside', 'into', 'like', 'minus', 'near', 'or','of', 'off', 'on', 'onto', 'opposite', 'out', 'outside',
                   'over', 'past', 'regarding', 'round', 'save', 'since', 'than','through', 'to', 'towards', 'under',
                   'underneath', 'unlike', 'until', 'up', 'upon', 'versus', 'via',
                   'with', 'within', 'without', 'targed with']
    pang_split_words = ['per', 'per unit', 'per unit mass','per unit area', 'per unit volume', 'per unit length',
                        'forma', 'plus', 'others','nm', 'unknown','targeted with', 'spp.']
    pang_split_incl = ['downward', 'upward', 'size','juvenile','particulate organic carbon','normalized','mixing ratio','ratio',
                       'mean','minimum', 'maximum', 'standard deviation','fraction','minerals']
    #12.08.2019
    #TO-DO split and/or exclude? rate, 'particulate', 'indicator', 'total'? -> Total organic carbon (TOC), TC

    #split words based on splitword_all

    splitword_all = split_words+pang_split_words +  pang_split_incl
    #exclude 'splitword_all_replace_only' after the split
    splitword_all_replace_only = pang_split_words + split_words

    # pang_qualifiers = ["aff\.", "cf\.", "ex gr\.", "gr\.", "nov\.", "subgen\.", "gen\.?",
    #                    "ng\.", "g\.\ssp.", "sp\.", "spp\.", "indeterminata", "undifferentiated", "ind\.", "ssp\.",
    #                    "subsp\.","sensu lato", "sensu stricto", "\-?group", "\-?type", "agg\.", "unit"]
    #
    # stop_words = ['with', 'given', 'aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among',
    #               'anti', 'around', 'as', 'at', 'before', 'behind',
    #               'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'but', 'by', 'concerning',
    #               'considering', 'despite', 'down', 'during', 'except', 'excepting', 'excluding', 'following', 'for',
    #               'from', 'in','inside', 'into', 'like', 'minus', 'near', 'of', 'off', 'on', 'onto', 'opposite', 'outside', 'over',
    #               'past', 'per', 'plus', 'regarding', 'round', 'save', 'since', 'than',
    #               'through', 'to', 'towards', 'under', 'underneath', 'unlike', 'until', 'up', 'upon', 'versus', 'via',
    #               'with', 'within', 'without']
    #
    # pang_stop_words = ["others", "error", "standard error", "male", "female", "total", "standard deviation", "mean",
    #                    "minimum","maximum", "daily minimum", "annual minumum", "daily mean", "annual mean", "normalized"]

    UCUM_SERVICE_QUANTITY = None
    ptn_pang_replace = None
    ptn_splitword_all=None
    elastic_host=None
    elastic_index=None
    elastic_doctype=None
    elastic_port=None
    elasticSearchInst = None
    primary_terminology = None
    secondary_terminologies = None
    tertiary_terminologies=None
    #chemistry_terminologies = None
    query_size_full = None
    query_size_shingle = None
    #min_match = None
    min_sim_value=None
    ptn_bracket = None
    ptn_digit = None
    elastic_shingle_tokenizer = "http://ws.pangaea.de/es/pangaea-terms/_analyze?analyzer=terms_folding&text="

    #def __init__(self, uservice,host,index,doctype,port,termi3,termi2,termi, size_full,size_shingle,minmatch, minsim):
    def __init__(self, uservice, host, index, doctype, port, termi3, termi2, termi, size_full, size_shingle, minsim):
        self.UCUM_SERVICE_QUANTITY = uservice
        self.elastic_host=host
        self.elastic_index = index
        self.elastic_doctype = doctype
        self.elastic_port = port
        self.initElasticSearch()
        self.primary_terminology = termi
        self.secondary_terminologies=termi2
        self.tertiary_terminologies=termi3
        #self.chemistry_terminologies=termichem
        self.query_size_full = size_full
        self.query_size_shingle = size_shingle
        #self.min_match = minmatch
        self.min_sim_value=minsim

        self.ptn_pang_replace_onqual = r'\b({})(?:\s|$)'.format('|'.join(self.pang_replace_onqual))
        self.pang_replace = self.pang_replace_onqual + self.pang_replace_general
        #self.pang_replace.sort()  # sorts normally by alphabetical order
        self.pang_replace.sort(key=lambda item: (-len(item), item))
        #self.pang_replace.sort(key=len, reverse=True)  # sorts by descending length
        self.ptn_pang_replace = r'\b({})(?:\s|$)'.format('|'.join(self.pang_replace))

        #self.splitword_all.sort()  # sorts normally by alphabetical order
        #sort by length of string followed by alphabetical order
        self.splitword_all.sort(key=lambda item: (-len(item), item))
        #print(self.splitword_all)
        #self.ptn_splitword_all = r'(?:\s|^)({})(?:\s|$)'.format('|'.join(self.splitword_all))
        #(?<!\S)(standard|of|total|sum..)(?!\S) will match and capture into Group 1 words in the group
        # when enclosed with whitespaces or at the string start/end.
        self.ptn_splitword_all = r'(?<!\S)({})(?!\S)'.format('|'.join(self.splitword_all))
        # .*? will match the string up to the FIRST character that matches )
        self.ptn_bracket = re.compile(r'\s\((.*?)\)(?=\s|$)')
        self.ptn_digit = r'(?:^)[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?(?:$)'
        self.ptn_subreplace = r'(?:\s|^)({})(?:\s)'.format('|'.join(split_words))

    def initElasticSearch(self):
        if not self.elasticSearchInst:
            logging.info('Initializing elastic search term index...')
            try:
                self.elasticSearchInst = Elasticsearch([self.elastic_host], port=self.elastic_port)
                logging.info("Elasticsearch is connected", self.elasticSearchInst.info())
            except Exception as ex:
                print("Error initElasticSearch:", ex)


    # def extractParamFragment_old(self,string_to_process):
    #     # str_list = [a.strip() for a in re.split(r'(\,\s)', string_to_process.strip()) if a]
    #     string_to_process = re.sub(r'\([a-zA-Z]+\s*\&\s*[a-zA-Z]+,?\s*\d+\)', '',
    #                                string_to_process)  # (Jennerjahn & Ittekkot, 1997)
    #     str_list = [a.strip() for a in re.split(r'\,\s|\sforma\s|\splus\s|\sper\s', string_to_process.strip()) if a]
    #     str_list = [x for x in str_list if x != ',']
    #     str_list = [x for x in str_list if len(str(x)) > 1]
    #     new_list = []
    #     for s in str_list:
    #         st = re.sub(self.pang_qualifiers_pattern, "", s)
    #         st = re.sub(self.stop_words_pattern, "", st)
    #         st = re.sub(r"\.+$", "", st)
    #         st = re.sub('\s+', ' ', st)  # remove multiple spaces
    #         if not re.match(r'^[_\W]+$', st):  # check if input contains word not (special) characters
    #             st = st.strip()
    #             if st.startswith(','):
    #                 st = st[1:].strip()
    #             if "sensu" in st:
    #                 st = st.split("sensu", 1)[0].strip()
    #             new_list.append(st)
    #     a_non_empty = [s for s in new_list if s]  # filter out the blank entries
    #     filtered_tokens = [w for w in a_non_empty if not w in self.pang_stop_words]
    #     return filtered_tokens

    def extractParamFragment(self,string_to_process):
        #extract chemical entities
        # chemical_frags =[]
        # chem = Document(string_to_process)
        # if chem.cems:
        #     for span in chem.cems:
        #         chemical_frags.append(span.text)
        #         string_to_process = string_to_process.replace(span.text,'')
        #     string_to_process = re.sub(' +', ' ', string_to_process)

        # exclude author,sensu...# (Jennerjahn & Ittekkot, 1997)
        string_to_process = re.sub(r'\([a-zA-Z]+\s*\&\s*[a-zA-Z]+,?\s*\d+\)$', '',
                                   string_to_process)
        if re.search(r'\b(sensu)\b', string_to_process):
            if not re.search(r'\slato\s|\sstricto\s', string_to_process):
                string_to_process = string_to_process.split("sensu", 1)[0].strip()

        #extract words in parenthesis
        # brac_match = self.ptn_bracket.search(string_to_process)
        # if brac_match:
        #     brac_word = brac_match.group(0).strip()
        #     string_to_process = string_to_process.replace(brac_word,'')

        # split by puctuation followed by a space
        #str_list = [a.strip() for a in re.split(r'(\,\s)', string_to_process.strip()) if a]
        #re.split(r'(:|;|,)\s'
        #str_list = [a.strip() for a in re.split(r'([^\w]\s)', string_to_process.strip()) if a]
        str_list = [a.strip() for a in re.split(r'(?:\:|;|,)\s(?![^()]*\))', string_to_process.strip()) if a]

        str_list_updated = []

        # if brac_match:
        #     str_list.append(brac_word.strip('()'))
        for i in str_list:
            splitted = [x.strip() for x in re.split(self.ptn_splitword_all, i)  if x.strip()]#........
            #splitted = list(filter(None, splitted)) # remove empty string
            str_list_updated.extend(splitted)
        str_list_updated = [w for w in str_list_updated if not w in self.splitword_all_replace_only]
        for idx, a in enumerate(str_list_updated):
            str_list_updated[idx]= re.sub(self.ptn_subreplace,'',a).strip()
        str_list_updated = [i for i in str_list_updated if len(i) > 1]  # remove single characters from list

        for idx, item in enumerate(str_list_updated):
            matched = self.ptn_bracket.search(item)
            #only extract word from thr brackets if the param does not contain ON qualifiers
            if matched and not re.search(self.ptn_pang_replace_onqual, string_to_process):
                found = matched.group(0).strip()  # (cf)
                #print('found',found)
                if len(found)> 4: # thisincludes brackets(2) and 2 char at max
                    str_list_updated[idx] = item.replace(found, '')
                    str_list_updated.append(found.strip('()'))

        new_list = []
        for s in str_list_updated:
            st = re.sub(self.ptn_pang_replace, "", s.strip())
            # st = re.sub(r'^(\W+)|(\W+)$', '', st) #strip special char beginning and end
            # strip all positive, negative, and/or decimals, e.g., Absorption coefficient, -1.23E+45 cm
            st = re.sub(self.ptn_digit, " ", st)
            st = re.sub('\s+', ' ', st).strip()
            new_list.append(st)

        #split if just one slash, e.g., Krypton-84/Argon-36
        for i in new_list:
            if i.count('/') == 1:
                new_list.extend(i.split('/'))
                new_list.remove(i)
        filtered_tokens = [s for s in new_list if s and len(s) > 1]  # filter out the blank entries
        return filtered_tokens

    def getUcumQuantity(self, uom):
        ucum_dict = {}
        try:
            #no need to do url encode of units
            q = self.UCUM_SERVICE_QUANTITY + urllib.parse.quote(uom)
            #q = self.UCUM_SERVICE_QUANTITY + uom
            resp = requests.get(q)
            #json_data = json.loads(resp.text)
            #encode to bytes, and then decode to text.
            json_data= json.loads(resp.text.encode('raw_unicode_escape').decode('utf8'))
            if (resp.status_code == requests.codes.ok):
                status = json_data['status']
                if (status == '201_QUANTITY_FOUND'):
                    ucum_dict['unit'] = json_data['input']
                    ucum_dict['ucum'] = json_data['ucum']
                    ucum_dict['fullname'] = json_data['fullname']
                    ucum_dict['quantity'] = json_data['qudt_quantity']
                    #ucum_dict['ucum_quantity'] = json_data['ucum_quantity']
                    l = []
                    #print(json_data['qudt_quantity'])
                    #for key, val in json_data['qudt_quantity'].items():
                        #l.append({"id": int(key), "name":val})
                    #ucum_dict['qudt_quantity'] =l
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            print(e)
        return ucum_dict

    #def executeTermQuery(self,t, isShingleMatch, isFuzzy, user_terminology,query_type):
    def executeTermQuery(self, t, user_terminology, query_type):
        size = self.query_size_full
        #convert the first letter of fragment to lowercase if first word of the fragment contains only alphabets

        if (query_type == "fullmatch"):
            q1 = Q({"multi_match": {"query": t, "fuzziness": 0, "fields":[ "name.fullmatch_exact^5", "name.fullmatch_folding" ]}})
        elif (query_type == "fuzzy_fullmatch"):
            q_a = Q({"multi_match": {"query": t, "fuzziness": 1, "prefix_length":1, "fields":[ "name.fullmatch_exact^5", "name.fullmatch_folding" ]}})
            q_b = Q({"multi_match": {"query": t, "fuzziness": "AUTO", "prefix_length":1,"fields":[ "name.fullmatch_exact^5", "name.fullmatch_folding" ]}})
            q1 = Q('bool', should=[q_a,q_b])
        else:
            size = self.query_size_shingle
            #if t.split()[0].isalpha():
                #t = re.sub('([a-zA-Z])', lambda x: x.groups()[0].lower(), t, 1)
            q1 = Q({"multi_match": {"query": t, "fuzziness": 0, "fields":[ "name.shinglematch_exact^5", "name.shinglematch_folding" ]}})

        qFilter = Q('terms', terminology_id=self.tertiary_terminologies)
        if user_terminology is not None:
            qShould1 = Q('constant_score', filter=Q('terms', terminology_id=user_terminology), boost=10)
            q = Q('bool', must=[q1], should=[qShould1], filter=[qFilter])
        else:
            qShould1 = Q('constant_score', filter=Q('terms',terminology_id=self.primary_terminology), boost=10)
            qShould2 = Q('constant_score', filter=Q('terms', terminology_id=self.secondary_terminologies), boost=5)
            q = Q('bool', must=[q1],should=[qShould1,qShould2], filter =[qFilter])
        s = Search(using=self.elasticSearchInst, index=self.elastic_index, doc_type=self.elastic_doctype).query(q)
        #s = s.filter("terms", terminology_id=self.all_terminologies)
        s = s.extra(size=size)
        response = s.execute()
        #print(s.to_dict())
        #print(response.success())
        list_res = []
        return_val = []
        if response:
            response = response.to_dict()
            #print("%d documents found" % response ['hits']['total'])
            for hit in response['hits']['hits']:
                dictres = {"id": int(hit['_id']), "name": hit['_source']['name'],"abbreviation": hit['_source']['abbreviation'],
                                     "score": hit['_score'],"terminology": hit['_source']['terminology']}
                if 'description_uri' in hit['_source']:
                    dictres['description_uri']=hit['_source']['description_uri']
                if 'topics' in hit['_source']:
                    dictres['topics'] = hit['_source']['topics']
                list_res.append(dictres)
            max_score = response['hits']['max_score']
            if list_res:
                if query_type == "shinglematch":
                    fragment_list = t.split()
                    #print('fragment_list',fragment_list)
                    t = t.lower()
                    list_resval = [d['name'].lower() for d in list_res]
                    # list_resid = [d['id'] for d in list_res]
                    # remove names that are included in another word
                    # ['water temperature','water','air temperature'] --> {'air temperature', 'water temperature'}
                    # list_res_low= self.substringSieve(list_res_low) do nto sieve!
                    # max_score_names = {d['name'].lower() for d in list_res if d['score']==response['hits']['max_score']}
                    # find a list of terms that match/form the most parts of a fragment
                    # options = list(set([d['name'].lower() for d in list_res]))
                    # print('options:',options)
                    # dict_grams = self.generateCombinations(list_res_low,len(fragment_list),max_score_names)
                    length_fragments = len(fragment_list)
                    #print('length_fragments: ',length_fragments)
                    dict_grams = self.generateCombinations(list_resval, length_fragments)
                    max_keys = self.get_max_cosine_sim(dict_grams.keys(), t,length_fragments)
                    # max_keys = self.fuzzy_process_extractBests(list(dict_grams.keys()), t)
                    # Attempts to account for partial string matches better. Calls ratio using the shortest
                    # string (length n) against all n-length substrings of the larger string and returns the highest score
                    # max_keys = self.partial_ratio(list(dict_grams.keys()),t)
                    # accounting for both how similar and different the two strings are
                    # max_keys = self.token_set_ratio(list(dict_grams.keys()), t)
                    #if max_keys:
                    final_terms = []
                    for m in max_keys:
                        value = dict_grams.get(m)
                        final_terms.extend(value)
                    list_res = [d for d in list_res if d['name'].lower() in final_terms]
                    #else:
                        #list_res = [d for d in list_res if d['score'] == max_score and len(d['name'].split())<=len(fragment_list)]
                        #print(list_res)
                else:
                    #TO-DO: return all five or just one
                    list_res = [d for d in list_res if d['score'] == max_score]

                k = {x['name'].lower() for x in list_res}
                for i in k:
                    sub_dict = [x for x in list_res if x['name'].lower() == i]
                    maxscore = max({d['score'] for d in sub_dict})
                    v = [d for d in sub_dict if d['score'] == maxscore]
                    return_val.extend(v)
                #remove redundant terms - only include highest scored terms
                # list_names = [d['name'] for d in list_res] #dont chnage to set
                # duplicates = {item for item, count in collections.Counter(map(str.lower, list_names)).items() if count > 1}
                # removes = []
                # for dup in duplicates:
                #     maxscore = max({d['score'] for d in list_res if d['name'].lower() == dup})
                #     remove_ids = {d['id'] for d in list_res if d['name'].lower() == dup and d['score'] < maxscore}
                #     removes.extend(list(remove_ids))
                # list_res = [d for d in list_res if d['id'] not in removes]
            #Remove any element from a list of strings that is a substring of another element
            #l = sorted(l.items(), key=lambda x: x[1], reverse=True)
        return return_val

    def substringSieve(self, string_list):
        return set(i for i in string_list
               if not any(i in s for s in string_list if i != s))

    #def generateCombinations(self,options,len_fragment, max_terms):
    def generateCombinations(self, options, len_fragment):
        dict_grams = {}
        #print('options',options)
        #for i in range(1, len(options) + 1):
        for i in range(1, len_fragment+1):
            for subset in itertools.combinations(options, i): #return i-length tuples in sorted order with no repeated elements.
                combined = ' '.join(subset) # e.g, subset -> ('carbon', 'magnetic flux')
                #a longer string can never be a substring of a shorter/equal length string
                if (len(combined.split()) <= len_fragment+1):
                    set_subset = set(subset)
                #if max_terms.intersection(set_subset):
                #if (len(combined.split()) <= len_fragment) and (len(max_terms.intersection(set_subset)) > 0):
                    dict_grams[combined] = set_subset # e.g., surface water temperature ('surface water', 'temperature')
                    # or surface water water ('surface water', 'water')
        #print(len(dict_grams))
        #print(dict_grams)
        return dict_grams

    # def preprocess_terms(self, text):
    #     q = self.elastic_shingle_tokenizer+text
    #     resp = requests.get(q)
    #     data = json.loads(resp.text.encode('raw_unicode_escape').decode('utf8'))
    #     words=None
    #     if (resp.status_code == requests.codes.ok):
    #         words = ' '.join(t['token'] for t in data['tokens'])
    #     return words

    # def text_to_vector(self,text):
    #     word = re.compile(r'\w+')
    #     words = word.findall(text)
    #     return Counter(words)

    def cosine_preprocess_elastic_to_vector(self,text):
        q = self.elastic_shingle_tokenizer + text
        resp = requests.get(q)
        #data = json.loads(resp.text.encode('raw_unicode_escape').decode('utf8'))
        data = json.loads(resp.text)
        words = None
        if (resp.status_code == requests.codes.ok):
            words = {t['token'] for t in data['tokens']}
        return Counter(words)

    def cosine_preprocess_elastic_to_string(self,text):
        q = self.elastic_shingle_tokenizer + text
        resp = requests.get(q)
        #data = json.loads(resp.text.encode('raw_unicode_escape').decode('utf8'))
        data = json.loads(resp.text)
        words = None
        return ' '.join(t['token'] for t in data['tokens'])

    def process_and_vectorize(self,text):
        # elastics term folding analyzer --> standard, extrafolding,folding, lowercase,english_stemmer
        # Tokenize
        #tokens = nltk.word_tokenize(text)
        #tokens = tokenizer.tokenize(text)
        #s.translate(None, string.punctuation)
        #tokens = re.split(r'\s|/', text)
        #words = {i.strip(string.punctuation) for i in tokens}
        mappings = {"alpha": "α", "beta": "β","gamma":"γ"}
        for key in mappings.keys():
            text = text.replace(key, mappings[key])
        text = unidecode(text)
        words = tokenizer.tokenize(text)
        #Param:  Highly hydroxy-interlayered vermiculite
        # if list :  ['hydroxy-interlayered vermiculite', 'vermiculite']
        # if set:  ['hydroxy-interlayered vermiculite']
        #words=[stemmer.stem(tkn) for tkn in words]
        words = {stemmer.stem(tkn) for tkn in words}
        #print(words)# 17α(H),21β(H)-30-norhopane -> ['17α', 'h', '21β', 'h', '30', 'norhopan']
        return Counter(words)

    def process_and_vectorize_string(self,text):
        text = unidecode(text)
        words = tokenizer.tokenize(text)
        words = {stemmer.stem(tkn) for tkn in words}
        return ' '.join(words)

    def get_cosine(self,vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys()) #set duplicates will beeliminated
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    #use a vector space block to determine similarity
    # similarity measure between two strings from the angular divergence within term based vector space.
    def get_max_cosine_sim(self, choices, query,length_fragments):
        similarities = {}
        final_matches = set()
        #query_vec = self.cosine_preprocess_elastic_to_vector(query)
        query_vec = self.process_and_vectorize(query)
        #print(query_vec)
        #len_query = len(query.split())
        #threshold_sim = (len_query-1)/len_query

        if length_fragments <=3:
            threshold_sim= 0.5
        else:
            threshold_sim = self.min_sim_value
        for c in choices:
            #sim = self.get_cosine(query_vec, self.cosine_preprocess_elastic_to_vector(c))
            sim = self.get_cosine(query_vec, self.process_and_vectorize(c))
            #print(sim, c)
            #if sim >= self.min_sim_value:
            if sim >= threshold_sim:
                similarities[c] = sim
        #sorted_x = sorted(similarities.items(), key=operator.itemgetter(1),reverse=True)
        #thresholds = {key: value for key, value in similarities.items() if value >= self.min_sim_value}

        if similarities:
            max_sim = max(similarities.values())
        #if max_sim >= self.min_sim_value:
        # getting all keys containing the `maximum`
            final_matches = [k for k, v in similarities.items() if v == max_sim]
        # else:
        #     final_matches_temp = {k for k, v in similarities.items() if v >= 0.5}
        #     final_matches = {k for k, v in final_matches_temp.items() if v == max(final_matches_temp.values())}
        return final_matches

    def remove_stopwords(self,words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words

    def remove_non_ascii(self,words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def stem_words(self,words):
        """Stem words in list of tokenized words"""
        #"Stemming refers to the process of reducing each word to its root or base"
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    def remove_punctuation(self, words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    def lemmatize_verbs(self,words):
        """Lemmatize verbs in list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    def generateNGram(self,options,len_fragment):
        dict_grams = {}
        for i in range(1, len(options) + 1):
            c = ngrams(options, i)
            for grams in c:
                #ignore any grams > length of fragment+1
                # +1 to allow additional one word
                #'water surface water temperature': ['water', 'surface water', 'temperature']}
                combined_grams = ' '.join(grams)
                if (len(combined_grams.split())<=len_fragment):#
                    dict_grams[combined_grams] = list(grams)
        return dict_grams

    def fuzzy_process_extractBests_old(self, choices, query):
        query_vec = self.preprocess_terms(query)
        # we have a list of options and we want to find the closest match(es)
        # dict_matches = dict(process.extract(query, choices_tokenized))
        # extractBests(query, choices, processor=default_processor, scorer=default_scorer, score_cutoff=0, limit=5):
        dict_matches = dict(process.extractBests(query_vec, choices, score_cutoff=50))
        max_matches = [k for k, v in dict_matches.items() if v == max(dict_matches.values())]
        #final_matches_idx = [choices.index(k) for k in max_matches]
        return max_matches

    def fuzzy_process_extractBests(self, choices, query):
        query_vec = self.process_and_vectorize_string(query)
        # we have a list of options and we want to find the closest match(es)
        choices_analyzed = []
        for c in choices:
            choices_analyzed.append(self.process_and_vectorize_string(c))
        #dict_matches = dict(process.extract(query, choices_tokenized))
        #extractBests(query, choices, processor=default_processor, scorer=default_scorer, score_cutoff=0, limit=5):
        #(query, score, key)<- results
        matches = process.extractBests(query_vec,choices_analyzed,score_cutoff=70)
        max_value = max(matches, key = itemgetter(1))[1]
        max_matches= {item[0] for item in matches if item[1] == max_value}
        #final_matches_idx = [choices_analyzed.index(k) for k in max_matches]
        #final_matches = [choices[i] for i in final_matches_idx]
        final_matches = {choices[choices_analyzed.index(k)] for k in max_matches}
        return final_matches

    def wratio(self, choices, query):
        query_vec = self.preprocess_terms(query)
        #https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings
        scores = {}
        # analyze both fragement and its combinations
        for value in choices:
            score = fuzz.WRatio(query_vec, self.preprocess_terms(value))
            scores[value] = score
        # sorted_x = sorted(scores.items(), key=operator.itemgetter(1))
        final_matches = [k for k, v in scores.items() if v == max(scores.values())]
        return final_matches

    def token_set_ratio(self,choices, query ):
        query_vec = self.cosine_preprocess_elastic_to_string(query)
        #query_vec = self.process_and_vectorize_string(query)
        #Attempts to rule out differences in the strings. Calls ratio on three particular substring sets and returns the max (code):
        #intersection-only and the intersection with remainder of string one
        #intersection-only and the intersection with remainder of string two
        #intersection with remainder of one and intersection with remainder of two
        # Notice that by splitting up the intersection and remainders of the two strings,
        # #we're accounting for both how similar and different the two strings are
        scores={}
        #analyze both fragement and its combinations
        for value in choices:
            #score = fuzz.token_set_ratio(query_vec, self.process_and_vectorize_string(value))
            score = fuzz.token_set_ratio(query_vec, self.cosine_preprocess_elastic_to_string(value))
            if score >= 70:
                scores[value] = score
        #sorted_x = sorted(scores.items(), key=operator.itemgetter(1))
        final_matches = {k for k, v in scores.items() if v == max(scores.values())}
        return final_matches

    def partial_ratio(self,choices, query ):
        #query_vec = self.preprocess_terms(query)
        query_vec = self.process_and_vectorize_string(query)
        #Attempts to rule out differences in the strings. Calls ratio on three particular substring sets and returns the max (code):
        #intersection-only and the intersection with remainder of string one
        #intersection-only and the intersection with remainder of string two
        #intersection with remainder of one and intersection with remainder of two
        # Notice that by splitting up the intersection and remainders of the two strings,
        # #we're accounting for both how similar and different the two strings are
        scores={}
        #analyze both fragement and its combinations
        for value in choices:
            score = fuzz.partial_ratio(query_vec, self.process_and_vectorize_string(value))
            if score >= 70:
                scores[value] = score
        final_matches = {k for k, v in scores.items() if v == max(scores.values())}
        return final_matches

    def sim_by_sequence(self,a,b):
        s = SequenceMatcher(None, a, b)
        return(s.ratio())

    def is_ci_stem_stopword_set_match(self, a, b, threshold=0.5):
        # Get default English stopwords and extend with punctuation
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend(string.punctuation)
        stopwords.append('')

        # Create tokenizer and stemmer
        tokenizer = nltk.tokenize.punkt.PunktWordTokenizer()
        stemmer = nltk.stem.snowball.SnowballStemmer('english')
        """Check if a and b are matches."""
        tokens_a = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(a) \
                    if token.lower().strip(string.punctuation) not in stopwords]
        tokens_b = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(b) \
                    if token.lower().strip(string.punctuation) not in stopwords]
        stems_a = [stemmer.stem(token) for token in tokens_a]
        stems_b = [stemmer.stem(token) for token in tokens_b]

        # Calculate Jaccard similarity
        ratio = len(set(stems_a).intersection(stems_b)) / float(len(set(stems_a).union(stems_b)))
        return (ratio >= threshold)