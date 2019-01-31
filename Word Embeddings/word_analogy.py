import os
import pickle
import numpy as np
import Queue as Q

model_path = './models/'
# loss_model = 'cross_entropy'
loss_model = 'nce'

def computeTop(word, dictionary, embeddings):
    w = embeddings[dictionary[word]]
    v1 = w.reshape(1, -1)
    # arr = np.zeros(embeddings.shape[0])
    pq = Q.PriorityQueue()    
    count = 0        
    for w in dictionary:    
        # print(w)

        v2  = embeddings[dictionary[w]]        
        t1 = np.matmul(v1, v2)
        t2 = np.linalg.norm(v1)*np.linalg.norm(v2)
        sim = np.divide(t1, t2)        
        # if("british" == w):
        #     print("british sim ", sim[0])
        pq.put((-1*sim[0], w))
        # if(count>20):
        #     # print(pq.get())
        #     pq.get()
        #     # count=count-1
        #     pq.put((sim[0], w))            
        # else:
        #     pq.put((sim[0], w))
        #     count = count+1
            
    print(word)
    print("/////////////////////////////////////////////")    
    # while not pq.empty():
    for i in range(0,20):
             t = pq.get()
             print(t[0], t[1])
    print("/////////////////////////////////////////////")

def getWordpair(arr):
    lst = arr.split(",")
    wrd_set=[]
    for i in range(0, len(lst)):
        tmp_lst = lst[i].split(":")                
        # print(tmp_lst)
        # if "\"" in tmp_lst[0]:
        #     str1 = tmp_lst[0][1:]
        # else:
        #     str1 = tmp_lst[0]

        # if "\"" in tmp_lst[1]:
        #     str2 = tmp_lst[1][:-1]
        # else:
        #     str2 = tmp_lst[1]
        str1 = tmp_lst[0].strip("\"")
        str2 = tmp_lst[1].strip("\"")
        wrd_set.append((str1,str2) )
    return wrd_set

def getVec(line, dictionary, embeddings):
    arr = line.split("||")
    train = getWordpair(arr[0])
    tmp = arr[1].strip("\n")
    test = getWordpair(tmp)
    
    diff_res = np.zeros((128,))
    count = 0
    # print(train)
    for t in train:        
        v1 = embeddings[dictionary[t[0]]]
        v2 = embeddings[dictionary[t[1]]]
        # print(- v1 + v2)
        diff_res = np.add(diff_res, (- v1 + v2))        
        count = count+1

    avg_diff = np.divide(diff_res, count)
    # print(avg_diff)
    tst_vec = np.zeros((avg_diff.shape[0],4))
    count=0
    for t in test:
        v1 = embeddings[dictionary[t[0]]]
        v2 = embeddings[dictionary[t[1]]]
        tst_vec[:, count] = (- v1 + v2)                
        count = count+1
    # 
    avg_diff = avg_diff.reshape(1, -1)
    dt_pr = np.matmul(avg_diff, tst_vec)    
    # print(avg)
    # print(dt_pr.shape)
    nrm_tst = np.linalg.norm(tst_vec, axis = 0, keepdims=True)
    nrm_tr = np.linalg.norm(avg_diff)
    nrm_vec = np.multiply(nrm_tr, nrm_tst)
    sim = np.divide(dt_pr, nrm_vec)
    # print(sim.shape)
    maxIndex = np.argmax(sim)
    minIndex = np.argmin(sim)

    lst = tmp.split(",")


    return " ".join(lst)+" "+lst[minIndex]+" "+lst[maxIndex]




model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'r'))
outF = open("myOutFile.txt", "w")

with open("./word_analogy_dev.txt") as fp:
    line = fp.readline()
    
    while line:
        ln = getVec(line, dictionary, embeddings)                
        # print(line)
        print >>outF, ln        
        line=fp.readline()

    
fp.close()
outF.close()
computeTop("first", dictionary, embeddings)
computeTop("american", dictionary, embeddings)
computeTop("would", dictionary, embeddings)


"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""
