import itertools
import json
import logging
import os

import regex as re
import string
import sys
import traceback
import urllib.parse
from difflib import SequenceMatcher
from operator import itemgetter
import nltk.stem.snowball
import requests
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
from nltk.tokenize import RegexpTokenizer
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from collections import Counter
import math
from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.stem.snowball import SnowballStemmer
import spacy


logging.getLogger("Elasticsearch").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)

class Term:
    global stemmer,tokenizer
    stemmer = SnowballStemmer("english")
    tokenizer = RegexpTokenizer(r'\w+')
    # replace pang_replace patterns  with ""
    pang_replace_onqual = ["forma","f\.", "div\.","var\.","aff\.", "cf\.?", "\scomplex$", "ex gr\.", "gr\.", "nov\.", "subgen\.", "gen\.?",
                    "ng\.", "g\.\ssp.", "sp\.", "spp\.", "indeterminata", "undifferentiated", "ind\.", "ssp\.",
                    "subsp\."]
    #, "sensu lato", "sensu stricto"]

    pang_replace_general =["\.?\-?group$", "\-?type$", "agg\."]
    global split_words
    split_words_special = ['aboard', 'across', 'along', 'amid', 'among', 'anti',
                           'around', 'behind', 'beneath', 'beside', 'besides', 'beyond', 'concerning',
                           'considering', 'despite', 'except', 'excepting', 'excluding', 'following', 'inside', 'like',
                           'minus', 'near', 'onto', 'opposite',
                           'outside', 'past', 'regarding', 'round', 'save', 'since', 'towards',
                           'underneath', 'unlike', 'upon', 'versus', 'via',
                           'within', 'without', 'targed with', 'due to']
    stop_words = list(get_stop_words('en'))  # Have around 900 stopwords
    nltk_words = list(stopwords.words('english'))  # Have around 150 stopwords
    stop_words.extend(nltk_words)
    stop_words = [s for s in stop_words if len(s) != 1] #exclude 1 chat stop word from analysis (e.g., a)
    split_words = split_words_special+stop_words
    #common words
    with open(os.path.join(os.path.dirname(__file__),"data" ,"google-10000-english.txt"), "r") as cwf:
        common_words = [cw.rstrip('\n') for cw in cwf.readlines()]

    pang_split_words = ['per', 'per unit', 'per unit mass','per unit area', 'per unit volume', 'per unit length',
                        'plus', 'others','nm', 'unknown','targeted with', 'spp.', 'given']
    pang_split_incl = ['downward', 'upward', 'size','juvenile','particulate organic carbon','normalized','mixing ratio','ratio',
                       'mean','minimum', 'maximum', 'standard deviation','fraction','minerals']
    pang_split_incl =[]

    #12.08.2019
    #TO-DO split and/or exclude? rate, 'particulate', 'indicator', 'total', activity? -> Total organic carbon (TOC), TC

    #split words based on splitword_all
    splitword_all = list(set(split_words+pang_split_words+pang_split_incl))
    #exclude 'splitword_all_replace_only' after the split
    splitword_all_replace_only = list(set(pang_split_words + split_words))

    UCUM_SERVICE_QUANTITY = None
    ptn_pang_replace = None
    ptn_splitword_all=None
    elastic_url=None
    elastic_index=None
    elastic_doctype=None
    elasticSearchInst = None
    elastic_min_should_match=None
    query_size_full = None
    query_size_shingle = None
    min_sim_value=None
    ptn_bracket = None
    ptn_digit = None
    prefix_length = 1
    field_boost = None
    min_length_frag = None
    elasticurl_tokenizer_ids = None
    elasticurl_tokenizer_str = None
    terminologies_dict=None
    size_shingle_return=None

    def __init__(self, uservice, esurl, index, doctype, size_full, size_shingle, size_shingle_rtn, minsim, plength,
                 minmatch, boost, fraglen, termsdict):
        self.UCUM_SERVICE_QUANTITY = uservice
        self.elastic_url = esurl
        self.elastic_index = index
        self.elastic_doctype = doctype
        self.initElasticSearch()
        self.query_size_full = size_full
        self.query_size_shingle = size_shingle
        self.size_shingle_return = size_shingle_rtn
        self.elastic_min_should_match = minmatch
        self.min_sim_value=minsim
        self.prefix_length = plength
        self.field_boost=boost
        self.min_length_frag = fraglen
        self.elasticurl_tokenizer_ids= "%s/%s/_mtermvectors" % (self.elastic_url, urllib.parse.quote(self.elastic_index))
        '''self.elasticurl_tokenizer_ids = "%s/%s/%s/_mtermvectors" % (
        self.elastic_url, urllib.parse.quote(self.elastic_index), urllib.parse.quote(self.elastic_doctype))'''
        self.elasticurl_tokenizer_ids = "%s/%s/%s/_mtermvectors" % (
        self.elastic_url, urllib.parse.quote(self.elastic_index), urllib.parse.quote(self.elastic_doctype))
        self.elasticurl_tokenizer_str = "%s/%s/_analyze?field=name.tokenmatch_folding&text=" % (self.elastic_url, urllib.parse.quote(self.elastic_index))
        self.terminologies_dict=termsdict

        self.ptn_pang_replace_onqual = r'\b({})(?:\s|$)'.format('|'.join(self.pang_replace_onqual))
        self.pang_replace = self.pang_replace_onqual + self.pang_replace_general
        #self.pang_replace.sort()  # sorts normally by alphabetical order
        self.pang_replace.sort(key=lambda item: (-len(item), item))
        #self.pang_replace.sort(key=len, reverse=True)  # sorts by descending length
        self.ptn_pang_replace = r'\b({})(?:\s|$|,)'.format('|'.join(self.pang_replace))
        #self.splitword_all.sort()  # sorts normally by alphabetical order
        #sort by length of string followed by alphabetical order
        self.splitword_all.sort(key=lambda item: (-len(item), item))
        #print(self.splitword_all)
        #self.ptn_splitword_all = r'(?:\s|^)({})(?:\s|$)'.format('|'.join(self.splitword_all))
        #(?<!\S)(standard|of|total|sum..)(?!\S) will match and capture into Group 1 words in the group
        # when enclosed with whitespaces or at the string start/end.
        self.ptn_splitword_all = r'(?<!\S)({})(?!\S)'.format('|'.join(self.splitword_all))
        # .*? will match the string up to the FIRST character that matches )
        self.ptn_bracket = re.compile(r'(?:\s|^)\((.*?)\)(?=\s|$)')
        self.ptn_digit = r'(?:^|\s)[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?(?:$)'
        self.ptn_subreplace = r'(?:\s|^)({})(?:\s)'.format('|'.join(split_words))
        self.lat_num_brackets = r'\((?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})\)\s+$'
        self.lat_num_unit = r'(?:\s|^)(\d+(?:\.\d*)?)((\s*\-\s*)(\d+(?:\.\d*)?))?\s*[A-Za-zÅµ]{1,3}(?=\s|$)' #kills also 14C which is not good
        self.nlp = spacy.load("en_core_web_sm")

    def initElasticSearch(self):
        if not self.elasticSearchInst:
            logging.info('Initializing elastic search term index...')
            try:
                self.elasticSearchInst = Elasticsearch([self.elastic_url])
                logging.debug("Elasticsearch is connected", self.elasticSearchInst.info())
            except Exception as ex:
                logging.debug("Error initElasticSearch:", ex)

    def is_constraint_candidate(self, string_to_process):
        is_constraint = False
        if isinstance(string_to_process, str):
            if str(' ') not in string_to_process:
                #string_to_process = [string_to_process.strip()]
                string_to_process = string_to_process.strip()
                try:
                    pos_tag = self.pos_tag_string(string_to_process)
                    #pos_tag = nltk.pos_tag(string_to_process)
                    if pos_tag:
                        if pos_tag[0][1] in ['VBN', 'ADJ','JJ','JJR','JJS','DT','RB','RBR','RBS','VBD']:
                            is_constraint = True
                except Exception as e:
                    print('NLTK Errror: ',e )
        return is_constraint


    def is_quality_candidate(self, string_to_process):
        is_quality = False
        standard_quantity = ['rate','ratio','mass','time','area','diameter','number','volume','height','level','range','weight','flux','age','content','index','pressure','factor','size']
        quantity_suffix_regex = r'(\w+\s)*\w+(ia|ty|ancy|ency|ance|ence|dom|ness|ship|tude|ion|ure|ment|osis|iasis|th)$'
        if re.match(quantity_suffix_regex, string_to_process):
            is_quality = True
        elif re.match('(\w+\s)*'+'('+'|'.join(standard_quantity)+')$', string_to_process) :
            is_quality = True
        return is_quality

    def is_taxon_candidate(self, string_to_process):
        is_taxon = False
        if string_to_process.count(' ') > 0:
            #check if words are common words
            cword = 0
            for word in string_to_process.split(' '):
                if word.lower() in self.common_words:
                    cword += 1
            ptn_pang_match = re.findall(self.ptn_pang_replace_onqual, string_to_process.strip())
            if re.fullmatch(r'[A-Z][a-z]{2,}(\s+\([A-Z][a-z]{2,}\))?(\s+[a-z]{2,})(\s+[a-z]{2,})?(\s+[a-z]{2,})?',string_to_process.strip()):
                is_taxon = True
            elif len(ptn_pang_match) > 0:
                is_taxon = True
            else:
                is_taxon = False
            if cword > 1:
                is_taxon = False
        return is_taxon

    def unpack_bracket_terms(self, term_list, level = 0, found_terms=[]):
        #Find all brackets with (\(([^()]|(?R))*\))
        #Check with findizter if it starts with space/start or ends with space/end
        if not isinstance(term_list, list):
            term_list = [term_list]
        for term in term_list:
            if term:
                term = term.strip()
                #print('TERM: ', term)
                # don't follow brackets which look like a taxon
                #if not re.match(r'^[A-Z][a-z]+\s\([A-Z][a-z]+\)\s[a-z]{2,}$',term):
                if not self.is_taxon_candidate(term):
                    if term:
                        rest_term = term
                        #Identify top level bracket, avoid to split brackets in whole words like 1-(2)-word-other(word)
                        if level==0:
                            if  ' ' in term:
                                #split using (nested) brackets as split term
                                bracket_matches = re.split(r'(\((?:[^()]|(?R))*\))', term)
                                #keep only those splits which were whole words
                                #create list of neighbour triples
                                bracket_splits = [(bracket_matches[i-1], l, bracket_matches[i+1]) for i,l in enumerate(bracket_matches) if i < len(bracket_matches)-1 and i > 0][::2]
                                #keep only those that have neighbors that start with a space on the adjacent side
                                rest_term = [bs[1] if bs[0].endswith(' ') or bs[2].startswith(' ') else ''.join(list(bs)) for bs in  bracket_splits]
                                rest_term =[t for t in rest_term if t]
                        #within the top level bracket split all nested brackets
                        elif term.startswith('(') or term.endswith(')'):
                            bracket_matches = re.finditer(r'\([^\(\)]*\)', term)
                            for bm in bracket_matches:
                                if bm:
                                    bm = bm[0]
                                    if level > 0:
                                        #remove inner brackets and continue with outer in recursive style
                                        #add split character to avoid ((a)-(b)) becomes a single term like a-b should be a|-|b which can be splitted later
                                        rest_term = rest_term.replace(bm,'|').strip()
                                        found_terms.append(bm.strip())
                                    else:
                                        rest_term = bm
                        #try to find nested brackets recursively
                        if rest_term and rest_term != term:
                            self.unpack_bracket_terms(rest_term, level+1, found_terms)
        return found_terms
    def pos_tag_string(self,string_to_process):
        pos_tagged_list =[]
        try:
            tagged_terms = self.nlp(string_to_process)
            for tt in tagged_terms:
                #participle adjectives
                if tt.dep_ == 'amod' and tt.tag_ == 'VBN':
                    tt.tag_ = 'JJV'
                pos_tagged_list.append((tt.text, tt.tag_))
        except:
            pass

    def get_compound_terms_fragments(self, text):
        compound_terms = []
        tagged_text = self.nlp(text)
        tags = []
        # participle adjectives
        for token in tagged_text:
            if token.dep_ == 'amod' and token.tag_ == 'VBN':
                token.tag_ = 'JJV'
            tags.append((token.text, token.tag_))
        for i in range(len(tags) - 3 + 1):
            if [t[1][:2] for t in tags[i: i + 3]] in [
                ['NN', 'NN', 'NN'], ['NN', 'JJ', 'NN'], ['JJ', 'NN', 'NN'],
                ['JJ', 'JJ', 'NN'], ['NN', 'IN', 'NN']
            ]:
                compound_terms.append(' '.join([t[0] for t in tags[i: i + 3]]))
        for i in range(len(tags) - 2 + 1):
            if [t[1][:2] for t in tags[i: i + 2]] in [
                ['JJ', 'NN'], ['NN', 'NN']
            ]:
                compound_terms.append(' '.join([t[0] for t in tags[i: i + 2]]))
        return compound_terms

    def extractParamFragmentNew(self, string_to_process):
        try:
            # exclude author,sensu...# (Jennerjahn & Ittekkot, 1997)
            string_to_process = re.sub(r'\([a-zA-Z]+\s*\&\s*[a-zA-Z]+,?\s*\d+\)$', '',
                                       string_to_process)
            fragment_list = []
            # split by delimiter but not terms in brackets
            str_list = [a.strip() for a in re.split(r'\s?[\+\-:;,\.]\s+(?![^()]*\))', string_to_process) if a]
            term_list = []
            print(1, str_list)
            for ti, term in enumerate(str_list):
                final_bracket_terms = []
                #unpack terms which are within brackets
                bracket_terms = self.unpack_bracket_terms(term, found_terms=[])
                if bracket_terms:
                    for bt in bracket_terms:
                        final_bracket_terms.extend(bt.split('|'))
                    split_regex = '|'.join([re.escape(r) for r in final_bracket_terms if r])
                    if split_regex:
                        split_terms = re.split('(' + split_regex + ')', term)
                        term_list.extend([t.strip('/').strip() for t in split_terms if t and len(t) > 1])
                else:
                    term_list.append(term.strip())
            # First round of fragment identification after bracket identification: split remaining terms by comma etc space and split words
            print(2, term_list)

            str_list = term_list
            for j, str_item in enumerate(str_list):
                ins_str_items = [a.strip() for a in re.split(r'\s?[\+\-:;,\.]\s+', str_item.strip()) if a.strip()]
                #ins_str_items = [a.strip() for a in re.split(r'[,\.;\-\+]\s+', str_item) if a and a.strip() != '']
                if ins_str_items:
                    term_list[j:j + 1] = ins_str_items
            fragment_list = []
            print(3, term_list)

            for i in term_list:
                splitted = [x.strip().strip('/') for x in re.split(self.ptn_splitword_all, i) if x.strip()]
                fragment_list.extend(splitted)
            print('3a', fragment_list)

            fragment_list = [w for w in fragment_list if not w in self.splitword_all_replace_only]
            print('3b', fragment_list)

            fragment_list = [re.sub(self.ptn_subreplace, '', w).strip() for w in fragment_list]
            print(4, fragment_list)

            # Second round of fragment identification, cleaning remove stop words and default words:
            new_fragment_list = fragment_list
            for fi, fr in enumerate(new_fragment_list):
                st = re.sub(self.ptn_pang_replace, "", fr.strip())
                #st = re.sub(r'\(\s*' + self.ptn_digit + r'\s*\)', " ", st)
                st = re.sub('\s+', ' ', st).strip()
                fragment_list[fi] = st

            # Third round or fragmentig: split slashes and plus signs but carefully
            for splitchar in ['/', '+']:
                new_fragment_list = []

                for fr in fragment_list:
                    # split fragments which look like a formula
                    if fr.count(splitchar) == 1:
                        new_fragment_list.extend(fr.split(splitchar))
                    else:
                        new_fragment_list.append(fr)
                fragment_list = new_fragment_list
            # strip remaining brackets
            fragment_list = [f.strip('()') for f in fragment_list]
            # remove numbers
            fragment_list = [re.sub(self.ptn_digit, '', f).strip() for f in fragment_list]
            # remove short strings
            fragment_list = [f for f in fragment_list if f and len(f) > self.min_length_frag]
            #remove duplicates
            fragment_list =sorted(set(fragment_list), key=fragment_list.index)
            #last step split long compounds to smaller compounds
            compound_fragment_list = []
            for frag in fragment_list:
                compound_fragment_list.append(frag)
                if frag.count(' ') > 1 and '(' not in frag and ')' not in frag:
                    for compound_fragement in self.get_compound_terms_fragments(frag):
                        if compound_fragement not in compound_fragment_list:
                            compound_fragment_list.append(compound_fragement)
            fragment_list = compound_fragment_list
            #classify
            final_fragment_list = []
            for s in fragment_list:
                shingle_on = False
                if self.is_quality_candidate(s) or self.is_taxon_candidate(s) :
                    shingle_on = True
                final_fragment_list.append({'fragment':s, 'shingle_active':shingle_on})
            return final_fragment_list
        except Exception as e:
            print('extractParamFragment ERROR: ',str(e))
            return []


    def extractParamFragment(self,string_to_process):
        #extract chemical entities
        # chemical_frags =[]
        # chem = Document(string_to_process)
        # if chem.cems:
        #     for span in chem.cems:
        #         chemical_frags.append(span.text)
        #         string_to_process = string_to_process.replace(span.text,'')
        #     string_to_process = re.sub(' +', ' ', string_to_process)

        token_type = None
        fragment_list = []
        # exclude author,sensu...# (Jennerjahn & Ittekkot, 1997)
        string_to_process = re.sub(r'\([a-zA-Z]+\s*\&\s*[a-zA-Z]+,?\s*\d+\)$', '',
                                   string_to_process)

        '''if re.search(r'\b(sensu)\b', string_to_process):
            if not re.search(r'\slato\s|\sstricto\s', string_to_process):
                string_to_process = string_to_process.split("sensu", 1)[0].strip()'''

        # handle brackets -> add a comma where apropriate to enable correct term splitting:
        # don't do this if the term could be a taxon to avoid to break subgenus names
        if not self.is_taxon_candidate(string_to_process.strip()):
            i = 0
            while '(' in string_to_process.strip() and i <= 10:
                bms = re.findall(r'(?:\s|^|\)|/)\(([^\(\)]*)\)', string_to_process.strip())
                for bm in bms:
                    if bm.isnumeric and len(bm) > 1:
                        string_to_process = string_to_process.strip().replace('(' + bm + ')', ' , ' + bm + ' , ')
                i += 1
        # split by puctuation followed by a space
        str_list = [a.strip() for a in re.split(r'\s?(?:\:|;|,|\s[\+-])\s(?![^()]*\))', string_to_process.strip()) if a]

        str_list_updated = []

        for i in str_list:
            splitted = [x.strip().strip('/') for x in re.split(self.ptn_splitword_all, i)  if x.strip()]
            str_list_updated.extend(splitted)
        # first round of fragment identification, split by stopwords and punctuation and remove these,
        # ignore single character fragments
        str_list_updated = [w for w in str_list_updated if not w in self.splitword_all_replace_only]
        for idx, a in enumerate(str_list_updated):
            str_list_updated[idx]= re.sub(self.ptn_subreplace,'',a).strip()
        str_list_updated = [i for i in str_list_updated if len(i) > 1]  # remove fragment with a single character from list
        # handle brackets -> add a comma where apropriate to enable correct term splitting:
        # don't do this if the term could be a taxon to avoid to break subgenus names
        if not self.is_taxon_candidate(string_to_process.strip()):
            i = 0
            while '(' in string_to_process.strip() and i <= 10:
                bms = re.findall(r'(?:\s|^|\)|/)\(([^\(\)]*)\)', string_to_process.strip())
                for bm in bms:
                    if bm.isnumeric and len(bm) > 1:
                        string_to_process = string_to_process.strip().replace('(' + bm + ')', ' , ' + bm + ' , ')
                i += 1
        #Second round of fragment identification, cleaning and setting up the dict:
        for ix, s in enumerate(str_list_updated):
            token_type = None
            if self.is_quality_candidate(s):
                token_type = 'quality'
            elif self.is_taxon_candidate(s):
                token_type = 'taxon'
            st = re.sub(self.ptn_pang_replace, "", s.strip())
            st = re.sub(r'\(\s*'+self.ptn_digit+r'\s*\)', " ", st)
            st = re.sub('\s+', ' ', st).strip()
            fragment_list.append({'token':st, 'raw_token': s, 'type': token_type})

        #Third round or fragmentig: split slashes and plus signs but carefully
        new_fragment_list =[]
        for i, fr in enumerate(fragment_list):
            # split and insert fragments by single slash
            if fr.get('token').count('/') == 1:
                slsplit = fr.get('token').split('/')
                new_fragment_list.extend([{'token':slsplit[0], 'raw_token':slsplit[0], 'type':None},
                                            {'token':slsplit[1], 'raw_token':slsplit[1], 'type':None}])
            else:
                new_fragment_list.append(fr)
        fragment_list = new_fragment_list

        new_fragment_list = []
        for i, fr in enumerate(fragment_list):
            # split and insert fragments by plus or slash -> full latin letter words only
            if re.fullmatch(r'^([A-Za-z]{2,}[\+/]?)+$', fr.get('token')):
                fr_split = re.split('[\+/]', fr.get('token'))
                if fr_split != [fr.get('token')]:
                    for tnp in fr_split:
                        if tnp:
                            new_fragment_list.append({'token':tnp, 'raw_token':tnp, 'type':None})
                else:
                    new_fragment_list.append(fr)
            else:
                new_fragment_list.append(fr)
        fragment_list = new_fragment_list
        #18-02-2020 filter out very short fragemnts
        filtered_tokens = [s for s in fragment_list if s.get('token') and len(s.get('token')) > self.min_length_frag]
        return filtered_tokens

    def getUcumQuantity(self, uom):
        ucum_dict = {}
        try:
            #no need to do url encode of units
            q = self.UCUM_SERVICE_QUANTITY + urllib.parse.quote(uom)
            resp = requests.get(q)
            json_data = json.loads(resp.text)
            #encode to bytes, and then decode to text.
            #json_data= json.loads(resp.text.encode('raw_unicode_escape').decode('utf8'))
            if (resp.status_code == requests.codes.ok):
                status = json_data['status']
                if (status == '201_QUANTITY_FOUND'):
                    ucum_dict['unit'] = uom
                    ucum_dict['ucum'] = json_data['ucum']
                    ucum_dict['fullname'] = json_data['fullname']
                    ucum_dict['quantity'] = json_data['qudt_quantity']
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            print(e)
        return ucum_dict

    def getIAdoptTermType(self, terminfo):
        topics = terminfo.get('search_terms')
        terminology = terminfo.get('terminology_id')
        #topics are actually 'search_terms'
        if isinstance(topics, str):
            topics =[topics]
        iadopt_type = 'ContextObject'
        if self.is_quality_candidate(topics[0]):
            iadopt_type = 'Property'
        elif self.is_constraint_candidate(topics[0]):
            iadopt_type = 'Constraint'
        if terminology in [13]: #PAN-Quantity
            iadopt_type = 'Property'
        elif 'Biological Classification' in topics:
            iadopt_type = 'ObjectOfInterest'

        #for pato entries
        if any(te.endswith('quality') for te in topics):
            if self.is_quality_candidate(topics[0]):
                iadopt_type = 'Property'
            elif len(topics) >1:
                if 'quality' in topics[1] or 'quantity' in topics[1]:
                    iadopt_type = 'Property'
                else:
                    iadopt_type = 'Constraint'
        return iadopt_type

    def executeTermQuery(self, t, user_terminology, query_type):
        size = self.query_size_full
        tparts = t.split(' ')
        if query_type == "fullmatch":
            q1 = Q({"multi_match": {"query": t, "fuzziness": 0, "fields":["name.fullmatch_exact^"+self.field_boost, "name.fullmatch_folding" ]}})
        elif (query_type == "fuzzy_fullmatch"):
            q_a = Q({"multi_match": {"query": t, "fuzziness": 1, "prefix_length":self.prefix_length, "fields":[ "name.fullmatch_exact^"+self.field_boost, "name.fullmatch_folding" ]}})
            q_b = Q({"multi_match": {"query": t, "fuzziness": "AUTO", "prefix_length":self.prefix_length,"fields":[ "name.fullmatch_exact^"+self.field_boost, "name.fullmatch_folding" ]}})
            q1 = Q('bool', should=[q_a,q_b])
        else:
            size = self.query_size_shingle
            q1 = Q({"multi_match": {"query": t, "fuzziness": 0, "fields":[ "name.shinglematch_exact^"+self.field_boost, "name.shinglematch_folding" ]}})

        qFilter = Q('terms', terminology_id=list(self.terminologies_dict.keys()))

        shoud_clause=[]
        if user_terminology is not None:
            # limit results to terminologies related to specific domain(s)
            qShould1 = Q('constant_score', filter=Q('terms', terminology_id=user_terminology), boost=20)
            q = Q('bool', must=[q1], should=[qShould1], filter=[qFilter])
        else:
            for k,v in self.terminologies_dict.items():
                shoud_clause.append(Q('constant_score', filter=Q('term', terminology_id=k),boost=v))
            q = Q('bool', must=[q1],should=shoud_clause, filter =[qFilter])
        #print(q.to_dict())
        try:
            s = Search(using=self.elasticSearchInst, index=self.elastic_index, doc_type=self.elastic_doctype).query(q)
            #s = s.extra(size=size)
            s = s[:size]
            response = s.execute()
        except Exception as e:
            logging.debug("Error ElasticSearch:", e)
            print('EL ERROR: ', e)

        list_res = []
        return_val = []
        if response:
            response = response.to_dict()
            #print("%d documents found" % response ['hits']['total'])
            for hit in response['hits']['hits']:
                token_match_similarity = 0
                word_similarity = self.get_word_similarity(str(hit['_source']['name']), str(t))
                try:
                    token_match_similarity = fuzz.ratio(str(hit['_source']['name']), str(t))
                except Exception as e:
                    print(e)
                someoverlap = False
                if any(nn in hit['_source']['name'].lower().split(' ') for nn in t.lower().split(' ')):
                    someoverlap = True
                if someoverlap:
                    dictres = {"id": int(hit['_id']), "name": hit['_source']['name'],"abbreviation": hit['_source']['abbreviation'],
                               "score": hit['_score'],"terminology": hit['_source']['terminology'],
                               "terminology_id": hit['_source']['terminology_id']}
                    if 'description_uri' in hit['_source']:
                        dictres['description_uri']=hit['_source']['description_uri']
                    if 'topics' in hit['_source']:
                        dictres['topics'] = hit['_source']['topics']
                    if 'search_terms' in hit['_source']:
                        termtype = self.getIAdoptTermType(hit['_source'])
                        dictres['iadopt_type'] = termtype
                    sim1 = token_match_similarity
                    sim2 = word_similarity * 100
                    dictres['similarity'] = (sim1+sim2)/2
                    list_res.append(dictres)
            if list_res:
                try:
                    if query_type == "shinglematch":
                        #2020-03-05 do not apply max score filter for shingle match
                        fragment_vector = self.tokenize_string(t) #Counter({'temperature': 1, 'sea': 1, 'surface': 1})
                        #print('fragment_vector ',fragment_vector)
                        list_ids = [str(d['id']) for d in list_res]
                        tokenized_terms_dict = self.tokenize_by_ids(list_ids)
                        #print(tokenized_terms_dict)
                        list_ids_tuples = self.generateCombinationsByTermIds(list_ids, len(t.split()))
                        final_ids = self.compute_cosine_sim(tokenized_terms_dict, list_ids_tuples, fragment_vector)
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
                except Exception as e:
                    print('Result list combi Error: ', e)
                    print(traceback.print_exc())
        return return_val


    def tokenize_by_ids(self,list_ids):
        l= {}
        headers = {'Content-type': 'application/json'}
        data = json.dumps({'ids': list_ids,"parameters": { "fields": [ "name.tokenmatch_folding"], "term_statistics": False,
                                                "field_statistics": False, "offsets": False, "positions": False,"payloads": False}})
        try:
            resp = requests.post(url = self.elasticurl_tokenizer_ids, data = data, headers=headers)
        except Exception as e:
            print('EL Tokenizer Error: ',e)
        if (resp.status_code == requests.codes.ok):
            results = resp.json()
            for t in results['docs']:
                val_dict = t['term_vectors']['name.tokenmatch_folding']['terms']
                l[t['_id']] = list(val_dict.keys())
        return l

    def tokenize_string(self,text):
        q = self.elasticurl_tokenizer_str + urllib.parse.quote(text)
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
        try:
            for i in range(1, len_fragment+1):
                #It return r-length tuples in sorted order with no repeated elements. For Example, combinations(‘ABCD’, 2) ==> [AB, AC, AD, BC, BD, CD].
                for subset in itertools.combinations(list_ids, i):
                    tuples_list.append(subset)
        except Exception as e:
            print('TermCombi Error:', e)
        return tuples_list

    def get_word_similarity(self, str1, str2):
        lst1 = str1.lower().split(' ')
        lst2 = str2.lower().split(' ')
        # calculate score for comparing lists of words
        c = sum(el in lst1 for el in lst2)
        if (len(lst1) == 0 or len(lst2) == 0):
            retval = 0.0
        else:
            retval = 0.5 * (c / len(lst1) + c / len(lst2))
        return round(retval,2)

    def get_cosine(self,vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys()) #set duplicates will beeliminated
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 0.0
        else:
            return round(float(numerator / denominator),1)

    def compute_cosine_sim(self, tokenized_dict, list_tuples, query_vec):
        final_matches=set()
        #temp={}
        for tuple in list_tuples:
            text =[]
            for t in tuple:
                text.extend(tokenized_dict.get(t))
            sim = self.get_cosine(query_vec, Counter(text))
            #print(sim, text)
            if sim >= self.min_sim_value:
                #similarities[tuple] = sim
                final_matches= final_matches.union(tuple)
                #temp[sim]=tuple
        #print(temp)
        #return final_matches
        return list(map(int, final_matches))

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