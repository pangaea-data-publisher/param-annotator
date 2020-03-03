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
from stop_words import get_stop_words
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
    split_words_special = ['aboard', 'across', 'along', 'amid', 'among', 'anti',
                           'around', 'behind', 'beneath', 'beside', 'besides', 'beyond', 'concerning',
                           'considering', 'despite', 'except', 'excepting', 'excluding', 'following', 'inside', 'like',
                           'minus', 'near', 'onto', 'opposite',
                           'outside', 'past', 'regarding', 'round', 'save', 'since', 'towards',
                           'underneath', 'unlike', 'upon', 'versus', 'via',
                           'within', 'without', 'targed with']
    stop_words = list(get_stop_words('en'))  # Have around 900 stopwords
    nltk_words = list(stopwords.words('english'))  # Have around 150 stopwords
    stop_words.extend(nltk_words)
    stop_words = [s for s in stop_words if len(s) != 1] #exclude 1 chat stop word from analysis (e.g., a)
    split_words = split_words_special+stop_words

    pang_split_words = ['per', 'per unit', 'per unit mass','per unit area', 'per unit volume', 'per unit length',
                        'forma', 'plus', 'others','nm', 'unknown','targeted with', 'spp.', 'given']
    pang_split_incl = ['downward', 'upward', 'size','juvenile','particulate organic carbon','normalized','mixing ratio','ratio',
                       'mean','minimum', 'maximum', 'standard deviation','fraction','minerals']
    #12.08.2019
    #TO-DO split and/or exclude? rate, 'particulate', 'indicator', 'total', activity? -> Total organic carbon (TOC), TC

    #split words based on splitword_all
    splitword_all = list(set(split_words+pang_split_words+pang_split_incl))
    #exclude 'splitword_all_replace_only' after the split
    splitword_all_replace_only = list(set(pang_split_words + split_words))

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
    elastic_min_should_match=None
    query_size_full = None
    query_size_shingle = None
    min_sim_value=None
    ptn_bracket = None
    ptn_digit = None
    prefix_length = 1
    field_boost = None
    min_length_frag = None
    primary_terminology_boost = None
    second_terminology_boost = None
    quantity_terminology_boost = None
    elasticurl_tokenizer_ids = None
    elasticurl_tokenizer_str = None

    #def __init__(self, uservice,host,index,doctype,port,termi3,termi2,termi, size_full,size_shingle,minmatch, minsim):
    def __init__(self, uservice, host, index, doctype, port, termi3, termi2, termi, size_full, size_shingle, minsim, plength,
                 minmatch, boost, fraglen, quanboost,priboost,secboost, elastokenids, elastokenstr):
        self.UCUM_SERVICE_QUANTITY = uservice
        self.elastic_host=host
        self.elastic_index = index
        self.elastic_doctype = doctype
        self.elastic_port = port
        self.initElasticSearch()
        self.primary_terminology = termi
        self.secondary_terminologies=termi2
        self.tertiary_terminologies=termi3
        self.query_size_full = size_full
        self.query_size_shingle = size_shingle
        self.elastic_min_should_match = minmatch
        self.min_sim_value=minsim
        self.prefix_length = plength
        self.field_boost=boost
        self.min_length_frag = fraglen
        self.quantity_terminology_boost= quanboost
        self.primary_terminology_boost = priboost
        self.second_terminology_boost = secboost
        self.elasticurl_tokenizer_ids= elastokenids
        self.elasticurl_tokenizer_str = elastokenstr

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
                logging.debug("Elasticsearch is connected", self.elasticSearchInst.info())
            except Exception as ex:
                logging.debug("Error initElasticSearch:", ex)

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
        # split by puctuation followed by a space
        str_list = [a.strip() for a in re.split(r'(?:\:|;|,)\s(?![^()]*\))', string_to_process.strip()) if a]

        str_list_updated = []
        # if brac_match:
        #     str_list.append(brac_word.strip('()'))
        for i in str_list:
            splitted = [x.strip() for x in re.split(self.ptn_splitword_all, i)  if x.strip()]
            str_list_updated.extend(splitted)
        str_list_updated = [w for w in str_list_updated if not w in self.splitword_all_replace_only]
        for idx, a in enumerate(str_list_updated):
            str_list_updated[idx]= re.sub(self.ptn_subreplace,'',a).strip()
        str_list_updated = [i for i in str_list_updated if len(i) > 1]  # remove fragment with a single character from list

        for idx, item in enumerate(str_list_updated):
            matched = self.ptn_bracket.search(item)
            #only extract word from thr brackets if the param does not contain ON qualifiers
            if matched and not re.search(self.ptn_pang_replace_onqual, string_to_process):
                found = matched.group(0).strip()  # (cf)
                if len(found)> 4: # this includes brackets(2) and 2 char at max
                    str_list_updated[idx] = item.replace(found, '')
                    str_list_updated.append(found.strip('()'))

        new_list = []
        for s in str_list_updated:
            st = re.sub(self.ptn_pang_replace, "", s.strip())
            # strip all positive, negative, and/or decimals, e.g., -1.23E+45
            st = re.sub(self.ptn_digit, " ", st)
            st = re.sub('\s+', ' ', st).strip()
            new_list.append(st)

        #split if just one slash, e.g., Krypton-84/Argon-36
        for i in new_list:
            if i.count('/') == 1:
                new_list.extend(i.split('/'))
                new_list.remove(i)
        #18-02-2020 filter out very short fragemnts
        filtered_tokens = [s for s in new_list if s and len(s) > self.min_length_frag]
        return filtered_tokens

    def getUcumQuantity(self, uom):
        ucum_dict = {}
        try:
            #no need to do url encode of units
            q = self.UCUM_SERVICE_QUANTITY + urllib.parse.quote(uom)
            resp = requests.get(q)
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
                    #l = []
                    #print(json_data['qudt_quantity'])
                    #for key, val in json_data['qudt_quantity'].items():
                        #l.append({"id": int(key), "name":val})
                    #ucum_dict['qudt_quantity'] =l
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            print(e)
        return ucum_dict

    def executeTermQuery(self, t, user_terminology, query_type):
        size = self.query_size_full
        if (query_type == "fullmatch"):
            q1 = Q({"multi_match": {"query": t, "fuzziness": 0, "fields":[ "name.fullmatch_exact^"+self.field_boost, "name.fullmatch_folding" ]}})
        elif (query_type == "fuzzy_fullmatch"):
            q_a = Q({"multi_match": {"query": t, "fuzziness": 1, "prefix_length":self.prefix_length, "fields":[ "name.fullmatch_exact^"+self.field_boost, "name.fullmatch_folding" ]}})
            q_b = Q({"multi_match": {"query": t, "fuzziness": "AUTO", "prefix_length":self.prefix_length,"fields":[ "name.fullmatch_exact^"+self.field_boost, "name.fullmatch_folding" ]}})
            q1 = Q('bool', should=[q_a,q_b])
        else:
            size = self.query_size_shingle
            q1 = Q({"multi_match": {"query": t, "fuzziness": 0, "fields":[ "name.shinglematch_exact^"+self.field_boost, "name.shinglematch_folding" ]}})

        qFilter = Q('terms', terminology_id=self.tertiary_terminologies)
        if user_terminology is not None:
            qShould1 = Q('constant_score', filter=Q('terms', terminology_id=user_terminology), boost=10)
            q = Q('bool', must=[q1], should=[qShould1], filter=[qFilter])
        else:
            qShould_q = Q('constant_score', filter=Q('term', terminology_id=13), boost=self.quantity_terminology_boost) #added 21-02-2020 boost by quantity
            qShould1 = Q('constant_score', filter=Q('terms',terminology_id=self.primary_terminology), boost=self.primary_terminology_boost)
            qShould2 = Q('constant_score', filter=Q('terms', terminology_id=self.secondary_terminologies), boost=self.second_terminology_boost)
            q = Q('bool', must=[q1],should=[qShould_q,qShould1,qShould2], filter =[qFilter])
        s = Search(using=self.elasticSearchInst, index=self.elastic_index, doc_type=self.elastic_doctype).query(q)
        #s = s.filter("terms", terminology_id=self.all_terminologies)
        s = s.extra(size=size)
        response = s.execute()
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

            if list_res:
                if query_type == "shinglematch":
                    fragment_vector = self.tokenize_string(t) #Counter({'temperature': 1, 'sea': 1, 'surface': 1})
                    #print('fragment_vector ',fragment_vector)
                    list_ids = [str(d['id']) for d in list_res]
                    tokenized_terms_dict = self.tokenize_by_ids(list_ids)
                    #print(tokenized_terms_dict)
                    list_ids_tuples = self.generateCombinationsByTermIds(list_ids, len(t.split()))
                    final_ids = self.compute_cosine_sim(tokenized_terms_dict, list_ids_tuples, fragment_vector)
                    final_ids = [int(i) for i in final_ids]
                    #remove the records not in final_ids
                    return_val = [d for d in list_res if d['id'] in final_ids]
                else:
                    #return_val = [d for d in list_res if d['score'] == max_score]
                    #27-02-2020 for full and fuzzy match return term with max score (for duplicate terms only)
                    list_names = [d['name'] for d in list_res]  # dont chnage to set
                    duplicates = {item for item, count in Counter(list_names).items() if count > 1}
                    remove_ids = []
                    for dup in duplicates:
                        mx = max({d['score'] for d in list_res if d['name'] == dup})
                        remove_ids.extend({d['id'] for d in list_res if d['name'] == dup and d['score'] < mx})
                    return_val = [d for d in list_res if d['id'] not in remove_ids]
        return return_val


    def tokenize_by_ids(self,list_ids):
        l= {}
        headers = {'Content-type': 'application/json'}
        data = json.dumps({'ids': list_ids,"parameters": { "fields": [ "name.tokenmatch_folding"], "term_statistics": False,
                                                "field_statistics": False, "offsets": False, "positions": False,"payloads": False}})
        resp = requests.post(url = self.elasticurl_tokenizer_ids, data = data, headers=headers)
        if (resp.status_code == requests.codes.ok):
            results = resp.json()
            for t in results['docs']:
                val_dict = t['term_vectors']['name.tokenmatch_folding']['terms']
                l[t['_id']] = list(val_dict.keys())
        return l

    def tokenize_string(self,text):
        q = self.elasticurl_tokenizer_str + text
        resp = requests.get(q)
        data = json.loads(resp.text)
        words = None
        if (resp.status_code == requests.codes.ok):
            words = {t['token'] for t in data['tokens']}
        return Counter(words)

    # def generateCombinations(self, options, len_fragment):
    #     dict_grams = {}
    #     for i in range(1, len_fragment + 1):
    #         # It return r-length tuples in sorted order with no repeated elements. For Example, combinations(‘ABCD’, 2) ==> [AB, AC, AD, BC, BD, CD].
    #         for subset in itertools.combinations(options, i):
    #             print(subset)
    #             combined = ' '.join(subset)  # convert tuple to string
    #             # print(subset,combined ) #('sea surface salinity', 'area temperature') sea surface salinity area temperature
    #             if (len(combined.split()) <= len_fragment + 1):  # allow buffer word
    #                 dict_grams[combined] = set(subset)
    #     return dict_grams

    def generateCombinationsByTermIds(self, list_ids, len_fragment):
        tuples_list= []
        for i in range(1, len_fragment+1):
            #It return r-length tuples in sorted order with no repeated elements. For Example, combinations(‘ABCD’, 2) ==> [AB, AC, AD, BC, BD, CD].
            for subset in itertools.combinations(list_ids, i):
                tuples_list.append(subset)
        return tuples_list

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

    def compute_cosine_sim(self, tokenized_dict, list_tuples, query_vec):
        #similarities = {}
        final_matches=set()
        for tuple in list_tuples:
            text =[]
            for t in tuple:
                text.extend(tokenized_dict.get(t))
            sim = self.get_cosine(query_vec, Counter(set(text)))
            #print(sim, text)
            if sim >= self.min_sim_value:
                #similarities[tuple] = sim
                final_matches.add(tuple)
        #final_matches = {k for k, v in similarities.items()}
        return set(sum(final_matches, ())) # transform a list of tuples into a flat list

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