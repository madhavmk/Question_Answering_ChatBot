# import statements
import wikipedia
import random
import re
from summa import summarizer
import operator
from rake_nltk import Rake
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from allennlp.predictors.predictor import Predictor

st = StanfordNERTagger('stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
                       'stanford-ner/stanford-ner.jar',
                       encoding='utf-8')


NLTKSTOPWORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                 "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                 "they", "them", "their", "theirs",  "themselves", "what", "which", "who", "whom", "this", "that",
                 "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                 "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
                 "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
                 "between", "into", "through", "during", "before", "after", "above", "below", "to", "from",
                 "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
                 "there", "when", "where", "why", "how", "all", "any",
                 "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
                 "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


PUNCIGN = "()"

LOCALINFO = {"you":'Data/About_Self',"yourself":'Data/About_Self',"You":'Data/About_Self',"Yourself":'Data/About_Self',
             "PESU":'Data/About_PESU',"PES University":'Data/About_PESU'}

DATAKEYS = LOCALINFO.keys()

r = Rake(stopwords=NLTKSTOPWORDS, punctuations=PUNCIGN)

predictor = Predictor.from_path("QAModels/allen_bidaf")


def replace_pronouns(text, noun):
    rep_pronouns = ["She", "she", "He", "he", "They", "they", "It", "it"]
    try:
        for rep in rep_pronouns:
            if text[0:len(rep)] == rep:
                text = noun + text[len(rep):]
                break
    except IndexError:
        pass
    return text


class Context:
    def __init__(self, topic, match):
        if match:
            try:
                self.page = wikipedia.page(topic)
            except wikipedia.exceptions.DisambiguationError as err:
                self.page = wikipedia.page(err.options[0])
        else:
            try:
                results = wikipedia.search(topic, results=100)
                self.page = wikipedia.page(results[random.randint(0, 100)])
            except wikipedia.exceptions.DisambiguationError as err:
                self.page = wikipedia.page(err.options[0])
        self.sections = self.page.sections
        self.summary = ""
        self.keywords = ""
        self.keywords_full = []
        self.sentences = []
        self.questions = []
        self.ktotal = []

    def get_section(self):
        print("Selected article:", self.page.title)
        print("Available sections:")
        print("0 : All")
        i = 1
        for section in self.sections:
            print(i, ":", section)
            i += 1
        print()
        choice = int(input("Selected section:"))
        if choice == 0:
            self.summary = summarizer.summarize(self.page.content, words=300)
            r.extract_keywords_from_text(self.page.content)
            self.ktotal = r.get_ranked_phrases_with_scores()
        else:
            self.summary = summarizer.summarize(self.page.section(self.sections[choice-1]), words=300)
            r.extract_keywords_from_text(self.page.section(self.sections[choice-1]))
            self.ktotal = r.get_ranked_phrases_with_scores()

    def gen_questions(self):

        r.extract_keywords_from_text(self.summary)
        self.keywords_full = r.get_ranked_phrases_with_scores()
        self.sentences = self.summary.split(".")
        # print("All keywords:",self.keywords_full)
        for s in self.sentences:
            r.extract_keywords_from_text(s)
            only_sentence_keywords = r.get_ranked_phrases_with_scores()
            # print("Only Sentence:", only_sentence_keywords)
            s.rstrip("\n")
            sentence_keywords = []
            for tup in self.keywords_full:
                i = list(tup)
                if ")" in i[1]:
                    split_string = i[1].split(")")
                    repstring = ""
                    for k in split_string:
                        repstring += k + "\)"
                    i[1] = repstring
                if "(" in i[1]:
                    split_string = i[1].split("(")
                    repstring = ""
                    for k in split_string:
                        repstring += "\(" + k
                    i[1] = repstring
                if re.search(i[1], s, flags=re.IGNORECASE):
                    sentence_keywords.append(i)
            for tup in only_sentence_keywords:
                sentence_keywords.append(list(tup))
            sentence_keywords.sort(key=operator.itemgetter(0), reverse=True)
            if len(sentence_keywords) != 0:
                if ")" in sentence_keywords[0][1]:
                    split_string = sentence_keywords[0][1].split(")")
                    repstring = ""
                    for k in split_string:
                        repstring += k+"\)"
                    sentence_keywords[0][1] = repstring
                if "(" in sentence_keywords[0][1]:
                    split_string = sentence_keywords[0][1].split("(")
                    repstring = ""
                    for k in split_string:
                        repstring += "\("+k
                    sentence_keywords[0][1] = repstring
                qtext = re.sub(sentence_keywords[0][1],
                               "_"*len(sentence_keywords[0][1]), s, flags=re.IGNORECASE)
                self.questions.append([qtext, sentence_keywords[0]])
        print("Generated Questions:")
        for q in self.questions:
            print(q[0])
            print(q[1])

    def gen_questions_df(self):
        r.extract_keywords_from_text(self.summary)
        self.keywords_full = r.get_ranked_phrases_with_scores()
        self.sentences = self.summary.split(".")
        # print("All keywords:",self.keywords_full)
        for s in self.sentences:
            r.extract_keywords_from_text(s)
            only_sentence_keywords = r.get_ranked_phrases_with_scores()
            # print("Only Sentence:", only_sentence_keywords)
            s.rstrip("\n")
            sentence_keywords=[]
            for tup in self.keywords_full:
                i = list(tup)
                if ")" in i[1]:
                    split_string = i[1].split(")")
                    repstring = ""
                    for k in split_string:
                        repstring += k+"\)"
                    i[1] = repstring
                if "(" in i[1]:
                    split_string = i[1].split("(")
                    repstring = ""
                    for k in split_string:
                        repstring += "\("+k
                    i[1] = repstring
                if re.search(i[1], s, flags=re.IGNORECASE):
                    sentence_keywords.append(i)
            for tup in only_sentence_keywords:
                sentence_keywords.append(list(tup))
            sentence_keywords.sort(key=operator.itemgetter(0), reverse=True)
            if len(sentence_keywords) != 0:
                qtext = re.sub(sentence_keywords[0][1],"_"*len(sentence_keywords[0][1]), s, flags=re.IGNORECASE)
                self.questions.append([qtext, sentence_keywords[0]])
        self.questions.append(["USING FKF", "****"])
        self.keywords_full.sort(key=operator.itemgetter(0), reverse=True)
        for s in self.sentences:
            for kf in self.keywords_full:
                if kf[1] in s:
                    qtext = re.sub(kf[1],"_"*len(kf[1]),s,flags=re.IGNORECASE)
                    self.questions.append([qtext, kf[1]])
                    self.keywords_full.remove(kf)

        print("Generated Questions:")
        for q in self.questions:
            print(q[0])
            print(q[1])

    def highvolume(self):
        print("500 word summary:")
        print(self.summary)
        print("Fulltext keywords:")
        for key in self.ktotal:
            if key[0]>4:
                print(key)


class QInput:
    def __init__(self,ip_txt):
        self.questext = ip_txt
        if self.questext[len(self.questext)-1]!="?":
            self.questext+="?"

        self.inbotdata = False
        self.dkey = ""
        for k in DATAKEYS:
            if k in self.questext:
                self.inbotdata = True
                self.dkey = k

        self.tokenized_text = word_tokenize(ip_txt)
        self.classified_text = st.tag(self.tokenized_text)
        self.people_names = []
        self.locations = []
        self.orgs = []
        self.others = []
        self.qkey = r.extract_keywords_from_text(ip_txt)
        prev_tag = ""
        current = ""
        for i in self.classified_text:
            if i[1] !=0:
                if i[1] == "PERSON":
                    if prev_tag != "PERSON":
                        prev_tag = "PERSON"
                        current = i[0]
                    else:
                        current += " "+i[0]
                else:
                    if prev_tag == "PERSON":
                        self.people_names.append(current)
                        current = ""
                    prev_tag = i[1]

        prev_tag = ""
        current = ""
        for i in self.classified_text:
            if i[1] != 0:
                if i[1] == "LOCATION":
                    if prev_tag != "LOCATION":
                        prev_tag = "LOCATION"
                        current = i[0]
                    else:
                        current += " "+i[0]
                else:
                    if prev_tag == "LOCATION":
                        self.locations.append(current)
                        current = ""
                    prev_tag = i[1]

        prev_tag = ""
        current = ""
        for i in self.classified_text:
            if i[1] != 0:
                if i[1] == "ORGANIZATION":
                    if prev_tag != "ORGANIZATION":
                        prev_tag = "ORGANIZATION"
                        current = i[0]
                    else:
                        current += " "+i[0]
                else:
                    if prev_tag == "ORGANIZATION":
                        self.orgs.append(current)
                        current = ""
                    prev_tag = i[1]
        self.pos_text = pos_tag(self.tokenized_text)
        self.backup_keys = []
        for i in self.pos_text:
            if i[1][:2] == "NN":
                self.backup_keys.append(i[0])


    def showner(self):
        print(self.pos_text)
        print("People found :", self.people_names)
        print("Locations found :", self.locations)
        print("Organizations found :", self.orgs)
        print("Backup keys :", self.backup_keys)

    def gen_searchstring(self):
        self.search = []
        self.search.extend(self.people_names)
        self.search.extend(self.orgs)
        for i in self.backup_keys:
            if i not in self.search and i not in self.locations:
                self.search.append(i)
        self.search.extend(self.locations)
        if self.inbotdata==True:
            f = open(LOCALINFO[self.dkey],"r")
            self.con_final = f.read()
        else:
            self.context = Context(self.search[0],1)
            self.con_final = self.context.page.summary
            self.con_summary = self.context.page.summary


    def guessans(self):
        print(predictor.predict(passage=self.con_final, question=self.questext)['best_span_str'])

'''
while True:
    ipq = input("Ask me a question \n")
    qobj = QInput(ipq)
    qobj.gen_searchstring()
    qobj.guessans()
'''