#importing packages
from os import listdir
from os.path import isfile, join
import os,sys,gc
import xml.sax
from nltk.corpus import stopwords 
from nltk import word_tokenize
import nltk
nltk.download("stopwords")
from nltk.stem import SnowballStemmer
import timeit,time
from collections import defaultdict
import re, string, unicodedata
import glob
from heapq import heapify, heappush, heappop
gc.disable()
stemmer = SnowballStemmer('english')
stop_words = frozenset(stopwords.words('english')) 
InvertedIndex = defaultdict(lambda:defaultdict(lambda:defaultdict(int)))
Stem_Words = {}
Inverted_Index_File_No = 1
index_word_count = 0
Doc_Limit = 30000

def remove_punctuation(data):
    Reg_Exp = re.compile(r'[.,;_()"/\']',re.DOTALL)
    data = Reg_Exp.sub('',data)
    return data
    
def remove_junk(data):
    Reg_Exp = re.compile(r"[~`!@#$%\-\^=\*+{\[}\]\|\\<>\?]",re.DOTALL)
    data = Reg_Exp.sub('',data)
    return data

def tokenize(data): 
    global stemmer
    global corpus_word_count
    tokens = data.split()
    corpus_word_count = corpus_word_count + len(tokens)
    words = []
    # remove all non-alphanumeric tokens
    for word in tokens:
        word = re.sub(r'[^\x00-\x7F]+','', word)
        if len(word) < 200 and word.isalnum() and word not in stop_words:
            if word in Stem_Words.keys():
                temp_word = Stem_Words[word]
            else:
                temp_word = stemmer.stem(word)
                Stem_Words[word] = temp_word
            if len(temp_word) > 2:
                words.append(temp_word)
    return words

def Insert_To_Inverted_Index(data, doc_id, tag):
    for word in data:
        word = re.sub(r'[^\x00-\x7F]+','', word)
        if len(word) > 2 and len(word) < 200:
            if(word in InvertedIndex):
                if(doc_id in InvertedIndex[word]):
                    if(tag in InvertedIndex[word][doc_id]):
                        InvertedIndex[word][doc_id][tag] += 1
                    else:
                        InvertedIndex[word][doc_id][tag] = 1
                else:
                    InvertedIndex[word][doc_id] = {tag:1}
            else:
                InvertedIndex[word] = dict({doc_id:{tag:1}})

def Data_Processing(data, docID, tag):
    global Inverted_Index_File_No
    global index_word_count
    global indexpath    
    data = data.lower()

    # Remove URL
    Reg_Exp = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',re.DOTALL)
    data = Reg_Exp.sub('',data)

    # Remove CSS
    Reg_Exp = re.compile(r'{\|(.*?)\|}',re.DOTALL)
    data = Reg_Exp.sub('',data)

    # Remove [[file
    Reg_Exp = re.compile(r'\[\[file:(.*?)\]\]',re.DOTALL)
    data = Reg_Exp.sub('',data)

    # Remove Cite 
    Reg_Exp = re.compile(r'{{v?cite(.*?)}}',re.DOTALL)
    data = Reg_Exp.sub('',data)

    # Remove Tag
    Reg_Exp = re.compile(r'<(.*?)>',re.DOTALL)
    data = Reg_Exp.sub('',data)

    if(tag == 'title'):
        # Remove junk
        Reg_Exp = re.compile(r"[~`!@#$%\-\^=\*+{\[}\]\|\\<>\?]",re.DOTALL)
        data = Reg_Exp.sub('',data)

        # Remove punctuation
        Reg_Exp = re.compile(r'[.,;_()"/\']',re.DOTALL)
        data = Reg_Exp.sub('',data)
        data = tokenize(data)
        Insert_To_Inverted_Index(data,docID,"t")
    elif(tag == 'text'):
        # Fetch categories
        categories = []
        categories = re.findall(r'\[\[category:(.*?)\]\]',data,flags=re.MULTILINE)

        # Fetch infobox
        infobox = []
        infobox = re.findall(r"{{infobox(.*?)}}",data,flags=re.DOTALL)
        
        # Fetch references
        references = []
        references = re.findall(r'== ?references ?==(.*?)==',data,flags=re.DOTALL)
        
        # Fetch external_links
        external_links = []
        Index = 0
        try:
            Index = text.index('==external links==')
            Index += 20
        except:
            pass

        if(Index):
            external_links = data[Index:]
            external_links = re.findall(r'\*\[(.*?)\]',external_links,flags=re.MULTILINE)

        if(Index):
            data = data[0:Index - 20]

        Reg_Exp = re.compile(r'{{infobox(.*?)}}',re.DOTALL)
        data = Reg_Exp.sub('',data)

        Reg_Exp = re.compile(r'== ?references ?==(.*?)==',re.DOTALL)
        data = Reg_Exp.sub('',data)

        Reg_Exp = re.compile(r'{{(.*?)}}',re.DOTALL)
        data = Reg_Exp.sub('',data)

        Reg_Exp = re.compile(r'[.,;_()"/\']',re.DOTALL)
        data = Reg_Exp.sub('',data)

        Reg_Exp = re.compile(r"[~`!@#$%\-\^=\*+{\[}\]\|\\<>\?]",re.DOTALL)
        data = Reg_Exp.sub('',data)

        #Tokenize, stem, remove stop words and non ascii chaarters

        data = tokenize(data)

        Insert_To_Inverted_Index(data,docID,"b")

        categories = ' '.join(categories)
        categories = remove_punctuation(categories)
        categories = remove_junk(categories)
        categories = tokenize(categories)
        Insert_To_Inverted_Index(categories,docID,"c")

        references = ' '.join(references)
        references = remove_punctuation(references)
        references = remove_junk(references)
        references = tokenize(references)
        Insert_To_Inverted_Index(references,docID,"r")

        external_links = ' '.join(external_links)
        external_links = remove_punctuation(external_links)
        external_links = remove_junk(external_links)
        external_links = tokenize(external_links)
        Insert_To_Inverted_Index(external_links,docID,"e")

        for infoList in infobox:
            infoboxList = []
            # print(type(infoList))
            infoboxList = re.findall(r'=(.*?)\|',infoList,re.DOTALL)
            infoboxList = ' '.join(infoboxList)
            infoboxList = remove_punctuation(infoboxList)
            infoboxList = remove_junk(infoboxList)
            infoboxList = tokenize(infoboxList)
            Insert_To_Inverted_Index(infoboxList,docID,"i")

        if docID % Doc_Limit == 0:
            f = open(indexpath + '/' + 'IndexChunk' + str(Inverted_Index_File_No) + '.txt' ,"w")
            for key,val in sorted(InvertedIndex.items()):
                key += "="
                for k,v in sorted(val.items()):
                    key += str(k) + ":"
                    for k1,v1 in v.items():
                        key = key + str(k1) + str(v1) + "#"
                    key = key[:-1]+","
                key = key[:-1]+"\n"
                index_word_count = index_word_count + 1
                f.write(key)
            f.close()
            InvertedIndex.clear()
            Stem_Words.clear()
            Inverted_Index_File_No += 1

def writeToPrimary():
    global indexFileCount
    offset = list()
    firstWord = True
    indexFileCount += 1
    fileName = folderToStore + "/" +"index" + str(indexFileCount) + ".txt"
    print(fileName)
    fp = open(fileName,"w")
    for i in sorted(invertedIndex):
        if firstWord:
            secondaryIndex[i] = indexFileCount
            firstWord = False
        toWrite = str(i) + "=" + invertedIndex[i] + "\n"
        fp.write(toWrite)

def writeToSecondary():
    fileName = folderToStore + "/" + "secondaryIndex.txt"
    fp = open(fileName,"w")
    for i in sorted(secondaryIndex):
        toWrite = str(i) + " " + str(secondaryIndex[i]) + "\n"
        fp.write(toWrite)                

class DataHandler( xml.sax.ContentHandler ):
    def __init__(self):
        self.DocId = 0
        self.CurrentData = ""
        self.title = ""
        self.id = ""
        self.text = ""
        self.id_flag = False

    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == "page":
            self.id_flag = True
            self.DocId += 1

    def endElement(self, tag):
        global f2
        if self.CurrentData == "title":
            f2.write(str(self.DocId)+"=="+self.title+"\n")
            Data_Processing(self.title,self.DocId,"title")
        elif self.CurrentData == "text":
            Data_Processing(self.text,self.DocId,"text")
        elif self.CurrentData == "id" and self.id_flag == True:
            self.id_flag = False

        self.CurrentData = ""
        self.id = ""
        self.text = ""
        self.title = ""

    def characters(self, content):
        if self.CurrentData == "title":
            self.title = self.title + content
        elif self.CurrentData == "text":
            self.text = self.text + content
        elif self.CurrentData == "id" and self.id_flag == True:
            self.id = self.id + content
            
if ( __name__ == "__main__"):
    # global index_word_count
    mypath = "./Phase2"
    allfiles = [mypath +'/'+ f for f in listdir(mypath) if isfile(join(mypath, f))]

    indexpath = "./testing"
    corpus_word_count = 0
    f2 = open('./testing/id-title.txt',"w")
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    start_time = time.time()
    Handler = DataHandler()
    parser.setContentHandler(Handler)

    for dumpfile in allfiles:
        print(dumpfile)
        parser.parse(dumpfile)
        f = open(indexpath + '/' + 'IndexChunk' + str(Inverted_Index_File_No) + '.txt' ,"w")
        for key,val in sorted(InvertedIndex.items()):
            stt = ""
            stt = key + "="
            for k,v in sorted(val.items()):
                stt += str(k) + ":"
                for k1,v1 in v.items():
                    stt = stt + str(k1) + str(v1) + "#"
                stt = stt[:-1]+","
            stt = stt[:-1]+"\n"
            index_word_count = index_word_count + 1
            f.write(stt)

    f.close()
    f2.close()
    mid_time = time.time()

    #MERGING PART

    folderToStore = "./result_testing"
    indexFileCount = 0
    secondaryIndex = dict()
    chunkSize = 1000000
    total_token_count = 0
    indexFiles = glob.glob("./testing/IndexChunk*")
    # primaryIndex = open("/content/drive/My Drive/IRE_Mini_Project/PHASE-1-UNZIPPED/primary_index.txt","w")
    completedFiles = [0] * len(indexFiles)
    filePointers = dict()
    currentRowOfFile = dict()
    percolator = list()
    words = dict()
    total = 0
    invertedIndex = defaultdict()

    fileCount = 0
    for i in range(len(indexFiles)):
        completedFiles[i] = 1
        try:
            filePointers[i] = open(indexFiles[i],"r")
            fileCount += 1
        except:
            print("Could Open Files: ",fileCount)
        currentRowOfFile[i] = filePointers[i].readline()
        words[i] = currentRowOfFile[i].strip().split("=")
        if words[i][0] not in percolator:
            heappush(percolator,words[i][0])

    while True:
        if completedFiles.count(0) == len(indexFiles):
            break
        else:
            total += 1
            word = heappop(percolator)
            for i in range(len(indexFiles)):
                if completedFiles[i] and words[i][0] == word:
                    if word in invertedIndex:
                        if len(words[i]) > 1:
                            invertedIndex[word] += "," + words[i][1]
                    else:
                        if len(words[i]) > 1:
                            invertedIndex[word] = words[i][1]

                    if total == chunkSize:
                        total_token_count += total
                        total = 0
                        writeToPrimary()
                        invertedIndex.clear()

                    currentRowOfFile[i] = filePointers[i].readline().strip()

                    if currentRowOfFile[i]:
                        words[i] = currentRowOfFile[i].split("=")
                        if words[i][0] not in percolator:
                            heappush(percolator,words[i][0])
                    else:
                        completedFiles[i] = 0
                        filePointers[i].close()
                # os.remove(indexFiles[i])

    writeToPrimary()
    writeToSecondary()
    end_time = time.time()
    gc.enable()
    print("Index Word Count : ",index_word_count)
    print("Total token Count : ",total_token_count)
    idc = float(mid_time-start_time)/60
    print("Index creation time : ",idc)
    mt = float(end_time-mid_time)/60
    print("Merging time : ",mt)
    total_time = idc+mt
    print("Total time : ",total_time)
