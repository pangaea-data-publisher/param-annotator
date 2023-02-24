from flask import request
import json
import logging
import termv1
import configparser as ConfigParser
import operator
import os
import re
from flask import Flask
import pandas as pd
import configparser
import argparse

# Create the application instance
app = Flask(__name__)

termInstance = None
dftopic = None

@app.route('/param-annotator/')
def home():
    return "Hello here's param-annotator! Please use me with the /api endpoint, e.g.: /param-annotator/api?name=Carbon,%20concentration&unit=mmol/l"

@app.route('/param-annotator/api', methods=['GET'])

def getTerm():
    unit=None
    paramName=None
    topic_ids = None
    shingle_active = None
    user_terminolgy = None
    if 'unit' in request.args:
        unit = request.args.get('unit').strip()
        logging.info('unit=%s', unit.encode('ascii', 'ignore'))
    if 'name' in request.args:
        paramName = request.args.get('name').strip()
        logging.info('paramName=%s', paramName.encode('ascii', 'ignore'))
    if 'topic' in request.args:
        topic_ids = request.args.get('topic').strip().split(',')
        logging.info('topic_ids=%s', topic_ids)
        # Get `class` list from request args
        #topics_id = request.args.getlist('topic')
        #topic_exists = dftopic.loc[dftopic.TopicId==topic_id,'TerminologyId']
        topic_exists = dftopic.loc[dftopic['TopicId'].isin(topic_ids), 'TerminologyId']
        if not topic_exists.empty:
            user_terminolgy = topic_exists.to_list()
    if 'shingle' in request.args:
        shingle_active = request.args.get('shingle')

    results = {}
    results['parameter'] =paramName
    ucum_jsn ={}
    term_jsn =[]
    # infer units
    if unit:
        ucum_jsn = termInstance.getUcumQuantity(unit)
    else:
        logging.debug('Extracting units from parameter')
        bracket = None
        if paramName.endswith(']'):
            bracket = '['
        #elif paramName.endswith(']'):
            #bracket = '['
        #elif paramName.endswith('}'):
            #bracket = '{'
        #else:
            #extract the last word from paramName; if all digit ignore them, e.g., Thallium 205
            #last_word_param = paramName.split()[-1]
            #if not re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?").fullmatch(last_word_param):
                #bracket = last_word_param
        if bracket:
            k = paramName.rfind(bracket)
            if k == -1:
                unit = None
            else:
                unit = paramName[k:]
                # remove unit from param name
                #if paramName.endswith(unit):
                # paramName= paramName[:-len(unit)]
                #my_regex = r"\s?\_?" + re.escape(unit) + r"\s?\_?"
                ucum_unit = unit
                #remove bracket if exists
                pattern_bracket = r"^(\[|\(|\{)"
                if re.match(pattern_bracket, unit):
                    ucum_unit = ucum_unit[1:-1]
                ucum_jsn = termInstance.getUcumQuantity(ucum_unit)
                if ucum_jsn:
                    my_regex = r"\s?\_?" + re.escape(unit) + r"\s?"
                    m = re.search(my_regex, paramName)
                    if m:
                        paramName = paramName[:-len(m.group(0))]
                        #paramName = paramName.replace(m.group(0),'')
                    logging.debug('Extracted units : %s',ucum_unit)
                    logging.debug('Final param after units extraction : %s', paramName)

    #get term - full, fuzzy and shing match for name
    list_fragments = termInstance.extractParamFragment(paramName)
    #logging.info("Param Fragments: " ,list_fragments)
    #logging.info("Param Fragments: {}".format(' '.join(str(e.encode('utf-8')) for (e) in list_fragments)))
    logging.debug(u"Param Fragments: %s" % [g['token'].encode('ascii', 'ignore') for g in list_fragments])
    #print('list_fragments ',list_fragments)
    if list_fragments:
        for fr in list_fragments:
            f = fr.get('token')
            tt = fr.get('type')
            startIndex = None
            endIndex = None
            match_type = None
            idscore_dict = None
            # full match w/o fuzzy
            if tt == 'taxon':
                idscore_dict = termInstance.executeTermQuery(f, user_terminolgy, 'taxon_fullmatch')
                match_type = 'full'
            elif tt == 'quality':
                idscore_dict = termInstance.executeTermQuery(f, user_terminolgy, 'quality_fullmatch')
                match_type = 'full'
            else:
                idscore_dict = termInstance.executeTermQuery(f, user_terminolgy, 'fullmatch')
                match_type = 'full'
            if not idscore_dict:
                # full match with fuzzy
                idscore_dict = termInstance.executeTermQuery(f, user_terminolgy, 'fuzzy_fullmatch')
                match_type = 'fuzzy'
            if not idscore_dict and shingle_active=="true":
                # shingle match
                match_type = 'shingle'
                idscore_dict = termInstance.executeTermQuery(f, user_terminolgy, 'shinglematch')
            results_dict = {}

            if f in paramName:
                startIndex = paramName.index(f)
                endIndex = startIndex + len(f)
            else:
                word_list = f.split()
                startIndex = paramName.index(word_list[0])
                endIndex = paramName.index(word_list[-1])+len(word_list[-1])

            results_dict['fragment'] = f
            results_dict['start_offset'] = startIndex
            results_dict['end_offset'] = endIndex
            #if idscore_dict: comment out on 27-02-2020

            results_dict['match_type'] = match_type
            results_dict['guessed_type'] = tt
            #else:
                #results_dict['match_type'] = None
            #results_dict['term'] = idscore_dict
            # sort terms dict by score
            results_dict['term'] = sorted(idscore_dict, key=lambda i: i['score'], reverse=True)
            if results_dict['term']:
                matched_term = results_dict['term'][0].get('name')
                if len(f) > len(matched_term):
                    results_dict['match_type'] = 'partial'
            term_jsn.append(results_dict)

    if ucum_jsn or term_jsn:
        term_jsn.sort(key=operator.itemgetter('start_offset'))
        results['dim_match'] = ucum_jsn
        results['text_match'] = term_jsn
        response = app.response_class(
        response=json.dumps(results),
        status=200,
        mimetype='application/json')
    else:
        response = app.response_class(status=404)
    return response

# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename='param-annotator.log', filemode="a+",
                        format="%(asctime)s %(levelname)s %(message)s")
    #root = logging.getLogger()
    path1 = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to termconf.ini config file")
    args = ap.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    ucum_service = config['INPUT']['ucum_service']
    elastic_url = config['INPUT']['elastic_url']
    elastic_index = config['INPUT']['elastic_index']
    elastic_doctype = config['INPUT']['elastic_doctype']
    #primary_terminologies = config.get("INPUT", "primary_terminology")
    #primary_terminologies = [int(x) for x in primary_terminologies.split(",")]
    query_size_full = int(config['INPUT']['query_size_full'])
    query_size_shingle = int(config['INPUT']['query_size_shingle'])
    query_size_shingle_return = int(config['INPUT']['query_size_shingle_return'])
    topic_mapping_file = path1 + "/"+config['INPUT']['topic_terminology_mapping_file']
    prefix_length = int(config['INPUT']['fuzzy_prefix_length'])
    param_annotator_host = config['INPUT']['service_host']
    param_annotator_port = int(config['INPUT']['service_port'])

    # added 05-03-2020 for dynamic assignment of terminolgies and their boost values
    terminologies_boost_temp = dict(config['TERMINOLOGY'])
    terminologies_boost_dict={}
    for key, value in terminologies_boost_temp.items():
        terminologies_boost_dict[int(key.split('_')[1])]=int(value)

    dftopic = pd.read_excel(topic_mapping_file, sheet_name=0,
                        index_col=None, na_values=['NA'], usecols="A,C",
                        header=0, converters={'TopicId': int, 'TerminologyId': int})
    dftopic = dftopic[dftopic.TerminologyId.notnull()]

    min_should_match = config['INPUT']['elastic_min_should_match']+"%"
    min_sim_value = float(config['INPUT']['min_sim_value'])
    match_field_boost = config.get("INPUT", "match_field_boost")
    min_length_frag = int(config['INPUT']['min_frag_length'])
    termInstance = termv1.Term(ucum_service, elastic_url, elastic_index, elastic_doctype, query_size_full,
                               query_size_shingle, query_size_shingle_return,min_sim_value, prefix_length,
                               min_should_match, match_field_boost,min_length_frag,terminologies_boost_dict)#

    app.run(host=param_annotator_host, port=param_annotator_port)

def rchop(thestring, ending):
  if thestring.endswith(ending):
    return thestring[:-len(ending)]
  return thestring