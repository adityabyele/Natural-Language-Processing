import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    y = []

    # print emission_scores.shape
    # print trans_scores.shape
    # print start_scores.shape
    # print end_scores.shape

    ans = np.zeros((L,N))
    path = np.zeros((L,N))
    em = np.transpose(emission_scores)    
    trans_t = np.transpose(trans_scores)
    # start_scores = start_scores.reshape((start_scores.shape[0],1))
    # print(em[:,0].shape)
    # print(start_scores.shape)
    t1 = np.add(em[:,0], start_scores)
    ans[:,0] = t1
    # path[:,0] = np.argmax(t1, axis = )
    for i in range(1,N):
        tmp = np.add(np.transpose(ans[:,i-1]),trans_t)        
        ans[:,i] = em[:,i]+np.transpose(np.amax(tmp, axis = 1))        
        path[:,i] = np.argmax(tmp, axis = 1)
        

    # print path
    # print "//////////////////"
    # print ans
    ans[:,N-1] = np.add(ans[:,N-1], end_scores)    
    y.append(np.argmax(ans[:,N-1]))

    for i in range(N-1, 0,-1):        
        y.append(int(path[int(y[len(y)-1]),i]))
    y.reverse()
    # for i in xrange(N):
        # stupid sequence
        # y.append(i % L)
    # score set to 0
    score = np.amax(ans[:,N-1])
    # print N
    # print y 
    # print score

    return (score, y)
