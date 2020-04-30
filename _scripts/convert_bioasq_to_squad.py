import json
import collections 
import requests
import time
from pathlib import Path
from bs4 import BeautifulSoup


class BioAsqToSquad2(object):

    def __init__(self, input_file, output_file):

        self.input_file = input_file
        self.output_file = output_file
        self.counters = collections.Counter(beginEndDiff=0,sectionIsTitle=0,instanceCount=0, numQuestions=0, badOffsets=0, otherExtractionFailures=0)
        self.answer_not_found = 0

        with open(self.input_file, "r") as reader:

            # deserialize into python object aka decoding
            self.input_data = json.load(reader)["questions"] 
           
    def transform_json(self):

        json_data = {} 

        # bioasq 8 contains all question in previous version + new questions + some questions that were removed
        json_data['version'] = 'BioASQ8b'

        # initialize data list
        json_data['data'] = [] 
        num_bioasq_questions = 0 

        for bioasq_question in self.input_data:
            
            num_bioasq_questions += 1 
            self.counters.update({'numQuestions':1})
            
            #if(num_bioasq_questions <=2400):
                #print("skipping question:",num_bioasq_questions)
                #continue
                  
            squad_data_instance = {}
            
            # we can later think of combining all list , factoid, yes no question types in title i.e. have one title for all questions of type factoid and so on
            squad_data_instance['title'] = bioasq_question['type']

            # only processes factorid questions 
            if squad_data_instance['title'] != 'factoid':
                continue
            else:
                print("processing factoid question num: ", num_bioasq_questions)
                print(self.counters)

            squad_data_instance['paragraphs'] = [] 
            
            bioasq_question_id = bioasq_question['id'] 
            
            bioasq_shared_question = bioasq_question['body']
            snippet_count_per_bioasq_question = 0 
            
            # we generate one entry per snippet 
            for bioasq_snippet in bioasq_question['snippets']:

                snippet_count_per_bioasq_question += 1 
                
                squad_paragraph_dict = {}
                squad_paragraph_dict['qas'] = [] 
                squad_qasi_dict = {}
                
                # one qas will map to one context ( unlike squad where mul qas per context )
                squad_qasi_dict['answers'] = []
                
                squad_answer_dict = {}

                document = bioasq_snippet['document']       
                bioasq_text = bioasq_snippet['text']
                bioasq_answer_start = bioasq_snippet['offsetInBeginSection']
                bioasq_begin_section = bioasq_snippet['beginSection']
                bioasq_end_section = bioasq_snippet['endSection']
                squad_answer_dict['text'] = bioasq_text
                squad_answer_dict['answer_start'] = bioasq_answer_start

                #bioasq_answer_end = bioasq_snippet['offsetInEndSection']
                
                if bioasq_begin_section != bioasq_end_section:
                    self.counters.update({'beginEndDiff':1})
                    print('begin and end section not the same: ', self.counters, bioasq_begin_section, bioasq_end_section )

                elif bioasq_begin_section != "abstract" and bioasq_begin_section != "sections.0" :

                    self.counters.update({'sectionIsTitle':1})
                    #print('begin section is not abstract: ', self.counters, "its: ", bioasq_begin_section)

                    # begin and end section is either abstract or sections.0 (abstract and sections.0 is the same thing) or title, it can't be anything else
                    if bioasq_begin_section != "title":
                        print("Yikes ths is not a title either !!!!!!! ")
                        print('begin section is not abstract or sections.0: its: ', bioasq_begin_section)

                    # move to next snippet as we only add entries that are abstracts
                    #print("skipping ", str(bioasq_question_id) + "_" + str(snippet_count_per_bioasq_question).zfill(3))
                    continue

                else:
                    squad_qasi_dict['question'] = bioasq_shared_question
                    squad_qasi_dict['id'] = str(bioasq_question_id) + "_" + str(snippet_count_per_bioasq_question).zfill(3)

                    # add answer dict to list 
                    squad_qasi_dict['answers'].append(squad_answer_dict)
                    squad_qasi_dict['is_impossible'] = False
                    
                squad_paragraph_dict['qas'].append(squad_qasi_dict)
                
                #squad_paragraph_dict['context'] = document 
                bioasq_context = self.getAbstractFromUrl(document)
                if(bioasq_context == ""):
                    print("unable to extract context from: ", document)
                    #print("extracted answer: ", bioasq_context)
                    #print("actual answer: ", bioasq_text)
                    #print("begin section: ", bioasq_begin_section)
                    self.counters.update({'otherExtractionFailures':1})
                    print("curr counters: ",self.counters) 
                    continue
                test_start = bioasq_answer_start
                test_end = test_start + len(bioasq_text) -1 
                answer = bioasq_context[test_start:test_end+1]

                # validate that the extracted answer (from url) matches the provided answer
                # if answers dont match (bad offsets) ignore this instance
                if answer!=bioasq_text:
                    print("incorrect offset provided for document url", document)
                    #print("extracted answer: ", answer)
                    #print("actual answer: ", bioasq_text)
                    #print("begin section: ", bioasq_begin_section)
                    self.counters.update({'badOffsets':1})
                    print("curr counters: ",self.counters)
                    continue

                squad_paragraph_dict['context'] = bioasq_context
                squad_data_instance['paragraphs'].append(squad_paragraph_dict)
                self.counters.update({'instanceCount':1})

            json_data['data'].append(squad_data_instance)
            if num_bioasq_questions % 300 == 0:
                new_file = str(self.output_file) + str(num_bioasq_questions)
                with open(new_file, 'w', encoding='utf-8') as writer:
                    json.dump(json_data, writer , sort_keys=False, ensure_ascii=False)
                print("curr counters: ", self.counters, "flushed json to file: ", new_file)

                time.sleep(5)
                json_data = {} 

                # bioasq 8 contains all question in previous version + new questions + some questions that were removed
                json_data['version'] = 'BioASQ8b'

                # initialize data list
                json_data['data'] = [] 
        #print(json.dumps(json_data, indent=4))

        # for the same question we will produce several context and answers (in squad2 same context had multiple question answers)
        with open(self.output_file, 'w', encoding='utf-8') as writer:
            json.dump(json_data, writer , sort_keys=False, ensure_ascii=False)
        print(self.counters, "for final file: ", self.output_file)

    def getAbstractFromUrl(self, abstractUrl):

        # add delay between page download requests
        time.sleep(0.7)
        try:
            res = requests.get(abstractUrl, timeout=3)
        except:
            self.counters.update({'requestTimeout':1})
            print("timed out for url: ", abstractUrl)
            return ""

        html_page = res.content
        soup = BeautifulSoup(html_page, 'lxml')
        if(not soup):
            #print("unable to extract html for doc: ", abstractUrl )
            return ""
        abstractDiv = soup.find("div", attrs={'class' : 'abstr'})
        if(not abstractDiv):
            #print("unable to find abstract for doc: ", abstractUrl )
            return ""
        abstract = abstractDiv.find("p").text
        h4 = soup.find("div", attrs={'class' : 'abstr'}).find("h4")

        # if the context/abstract contains prefix words like objective/introduction then this abstract uses
        # different format, adjust the extraction for this format
        if h4:
            #print(h4.text)
            text_full = soup.find("div", attrs={'class' : 'abstr'}).text
            abstract = text_full[8:]
        return abstract
            

if __name__ == '__main__':
   
    base_path = Path(__file__).parent
    #in_file_path = (base_path / "./bioasq/BioASQ-test8b/BioASQ-task8bPhaseB-testset1.json").resolve()
    #out_file_path = (base_path/ "./bioasq/BioASQ-test8b/BioASQ-task8bPhaseB-testset1_squad_format_only_factoid.json").resolve()

    in_file_path = (base_path / "./bioasq/BioASQ-training8b/training8b.json").resolve()
    out_file_path = (base_path/ "./bioasq/BioASQ-training8b/training8b_squad_format_factoid.json").resolve()
    cbs = BioAsqToSquad2(in_file_path,out_file_path)
    cbs.transform_json()



