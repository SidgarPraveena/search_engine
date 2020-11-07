from bisect import bisect
from math import log10
from operator import itemgetter
import os,sys,gc,mmap
from nltk.corpus import stopwords 
from nltk import word_tokenize
import nltk
nltk.download("stopwords")
from nltk.stem import SnowballStemmer
import timeit,time
from collections import defaultdict
import re, string, unicodedata
gc.disable()
stemmer = SnowballStemmer('english')


def find_posting(path, start_flag, word):
    f = open(path,'r+b')
    # print(path)
    mf = mmap.mmap(f.fileno(), 0)
    mf.seek(0) # reset file cursor
    if start_flag:
        word = bytes(word + "=" ,'utf-8')
    else:
        word = bytes("\n" + word + "=" ,'utf-8')
    m = re.search(word, mf)
    if m == None:
        return ""
    else:
        mf.seek(m.start()+1)
        ans = mf.readline().decode('utf-8')
    mf.close()
    f.close()
    return ans
    
def process_input(query_words):
    search_words = []
    for word in query_words:
        word = word.lower().strip()
        if word not in stop_words:
            word = stemmer.stem(word)
        if word.isalnum() and len(word)>2 and word not in stop_words:
            search_words.append(word)
    return search_words   
    
def normal_query(query_words,K): 
    start = time.time()
    search_words = process_input(query_words)
    if len(search_words) == 0:
        exit()
    global_search = dict(list())
    for word in search_words:
        pos = bisect(top_word,word)
        start_flag = False
        if pos-1 >= 0 and top_word[pos-1] == word:
            start_flag = True
            if pos-1 != 0:
                pos -= 1
            if pos+1 == len(top_word) and top_word[pos] == word:
                pos += 1
        primary_file = "./result_testing_1" +"/" + "index" + str(pos) + ".txt"
        posting = find_posting(primary_file,start_flag,word)
        if posting == "":
            continue
        posting_list = re.split(",",re.split("=",posting)[1])    
        num_docs = len(posting_list)
        IDF = round((log10(total_docs/num_docs)),5) #keeping precision upto 5 decimal places
        for i in posting_list:
            docID, entry = re.split(":",i)[0],re.split(":",i)[1]
            if docID in global_search:
                global_search[docID].append(entry + "_" + str(IDF))
            else:
                global_search[docID] = [entry + "_" + str(IDF)]
    lengthFreq = dict(dict())
    regEx = re.compile(r'(\d+|\s+)')
    for k in global_search:
        weightedFreq = 0
        n = len(global_search[k])
        for x in global_search[k]:
            x,idf = re.split("_",x)[0] , re.split("_",x)[1]
            x = re.split("#",x)
            for y in x:
                lis = regEx.split(y)
                tagType, freq = lis[0], lis[1]
                if tagType == "t":
                    weightedFreq += int(freq)*1000
                elif tagType == "i" or tagType == "c" or tagType == "r" or tagType == "e":
                    weightedFreq += int(freq)*50
                elif tagType == "b":
                    weightedFreq += int(freq)
        if n in lengthFreq:
            lengthFreq[n][k] = float(log10(1+weightedFreq))*float(idf)
        else:
            lengthFreq[n] = {k : float(log10(1+weightedFreq))*float(idf)}
    count = 0
    flag = False
    result = []
    end = time.time()

    for k,v in sorted(lengthFreq.items(),reverse=True):
        for k1,v1 in sorted(v.items(),key = itemgetter(1),reverse=True):
            result.append(str(k1) + ", " + doc_title_map[k1])
            count += 1
            if count == K:
                flag = True
                break
        if flag:
            break                               
    return result,(end-start)        
    
def field_query(query_words,K):
    start = time.time()
    fieldDict = dict()
    search_words = []
    for word in query_words:
        tag, w = word.split(":")
        w = w.lower()
        if w not in stop_words:
            w = stemmer.stem(w)
        if w.isalnum() and len(w) > 2 and w not in stop_words:
            search_words.append(w)
            if w in fieldDict:
                fieldDict[w] += tag
            else:
                fieldDict[w] = tag            
    if len(search_words) == 0:
        exit()
    global_search = dict(list())
    for word in fieldDict: #changed from search_words to fieldDict, as words may repeat in query
        pos = bisect(top_word,word)
        start_flag = False
        if pos-1 >= 0 and top_word[pos-1] == word:
            start_flag = True
            if pos-1 != 0:
                pos -= 1
            if pos+1 == len(top_word) and top_word[pos] == word:
                pos += 1
        primary_file = "./result_testing_1" +"/" + "index" + str(pos) + ".txt"
        posting = find_posting(primary_file,start_flag,word)
        if posting == "":
            continue
        posting_list = re.split(",",re.split("=",posting)[1])    
        num_docs = len(posting_list)
        IDF = round((log10(total_docs/num_docs)),5) #keeping precision 5
        for i in posting_list:
            pls = re.split(":",i)
            if len(pls) > 1:
                docID, entry = pls[0],pls[1] 
                cnt = 0
                for ik in range(len(fieldDict[word])):
                    if fieldDict[word][ik] in entry:
                        cnt += 1
                if cnt >= 1:        
                    if docID in global_search:
                        global_search[docID].append(entry + "_" + str(IDF))
                    else:
                        global_search[docID] = [entry + "_" + str(IDF)]
    lengthFreq = dict(dict())
    regEx = re.compile(r'(\d+|\s+)')
    for k in global_search:
        unweightedFreq = 0
        n = len(global_search[k])
        #edited from here
        for wd in fieldDict:
            for ik in range(len(fieldDict[wd])):
                for x in global_search[k]:
                    x,idf = re.split("_",x)[0] , re.split("_",x)[1]
                    x = re.split("#",x)
                    for y in x:
                        lis = regEx.split(y)
                        tagType, freq = lis[0], lis[1]
                        if tagType == fieldDict[wd][ik]:
                            unweightedFreq += int(freq)*1000    #Just multiplied by 100 (no use)
                        else:
                            unweightedFreq += int(freq)*5    
        if n in lengthFreq:
            lengthFreq[n][k] = float(log10(1+unweightedFreq))*float(idf)
        else:
            lengthFreq[n] = {k : float(log10(1+unweightedFreq))*float(idf)}
    count = 0
    flag = False
    # K = 10
    result = []
    end = time.time()
    for k,v in sorted(lengthFreq.items(),reverse=True):
        for k1,v1 in sorted(v.items(),key=itemgetter(1),reverse=True):
            # print(doc_title_map[k1])
            result.append((k1) + ", " + doc_title_map[k1])
            count += 1
            if count == K:
                flag = True
                break
        if flag:
            break
    return result,(end-start)           


total_docs = 0
doc_title_map = dict()
doc_title_path = "./testing_1/id-title.txt"
top_word = []
top_word_path = "./result_testing_1/mysec.txt"
stop_words = frozenset(stopwords.words('english'))

f = open(top_word_path,"r")
top_word=[re.split(":",line)[0] for line in f]

f = open(doc_title_path,"r")
for line in f:
    if line.count('==') >= 2:
        docID = line.split('==')[0]
        titleMap = line[line.find('==')+2:]
    else:
        docID,titleMap = line.split('==')
    doc_title_map[docID] = titleMap
    total_docs += 1
    
    
path = "./queries.txt"
f = open(path,"r")
fw = open('queries_op.txt​','w')
for line in f:
	fw.write("\n")
	K = int(line.split(",")[0].strip())
	query = line.split(",")[1].strip()
	if ':' in query:
		tag_list = []
		for s in query.split():
			if ':' in s:
				tag_list.append(s.split(':')[0])
			ans = query.split(':')
		for l in range(1,len(ans)-1):
			ans[l]= ans[l][:-2]
		ans = ans[1:]
		for i in range(len(ans)):
			ans[i] = tag_list[i] + ':' + ans[i]
		finall = []
		# split strings containing space(multiple words) and associate tag 
		for i in range(len(ans)):
			if ' ' in ans[i]:
				tag = ans[i].split(':')[0]
				pl = (ans[i].split(':')[1]).split()
				for k in pl:
					finall.append(tag + ":" + k)
			else:
				finall.append(ans[i])
		res,ti = field_query(finall,int(K))
		#fw.write(res)
		#for p in res:
		#	fw.write(str(p))
		#fw.write("\n")
		fw.write(' '.join(res))
		fw.write(str(round(ti,4)) +", "+str(round(ti/K,4)) + '\n\n')
	else:
		res,ti = normal_query(query.split(),K)
		#fw.write(res)
		#for p in res:
		#	fw.write(str(p))
		#fw.write("\n")
		fw.write(' '.join(res))
		fw.write(str(round(ti,4)) +", "+str(round(ti/K,4)) + '\n\n')
fw.close()
