import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            """                                                                                                                                 
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()
                
            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...
            
                
            2) Call forward_pass and get predictions
            
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam

            ...
            self.loss =
            
            ===================================================================
            """

            self.train_inputs = tf.placeholder(tf.int32, shape = (Config.batch_size, Config.n_Tokens))
            print "train ip", self.train_inputs.shape
            self.train_labels = tf.placeholder(tf.int32, shape = (Config.batch_size,parsing_system.numTransitions()))            
            print "train lb", self.train_labels.shape
            self.test_inputs = tf.placeholder(tf.int32, shape = (Config.n_Tokens,))
            print "test ip", self.test_inputs.shape
            # print tf.shape(self.train_inputs)
            embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            # embed = tf.reshape(self.train_inputs, shape =(Config.batch_size*Config.n_Tokens, ))             
            print "emb", embed.shape
            embed = tf.reshape(embed, shape = (Config.batch_size, Config.embedding_size*Config.n_Tokens))
            print "emb2", embed.shape
            weights_input = tf.Variable(tf.random_normal((Config.hidden_size, Config.embedding_size*Config.n_Tokens), 0, 0.1))
            print "w ip", weights_input.shape
            biases_input = tf.Variable(tf.zeros((Config.hidden_size, 1)))
            print "b ip", biases_input.shape            
            hidden_size2 = 200
            weights_input2 = tf.Variable(tf.random_normal((hidden_size2, Config.hidden_size), 0, 0.1))
            print "w ip2", weights_input2.shape
            biases_input2 = tf.Variable(tf.zeros((hidden_size2, 1)))
            print "b ip2", biases_input2.shape            
            weights_output = tf.Variable(tf.random_normal((parsing_system.numTransitions(), hidden_size2), 0, 0.1))
            print "w op", weights_output.shape
            # train_inputs()
            # print tf.shape(embed)
            # print embed.eval()      
            # 

            # embed = btch_sizex(lrge shp)                  
            # wts_ip = 200x(lrge shape)
            #
            # self.train_labels = 
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_input2, biases_input2, weights_output)
            print "pred", self.prediction.shape
            wt2_l2 = tf.nn.l2_loss(weights_input2)+tf.nn.l2_loss(biases_input2)
            l2 = wt2_l2+tf.nn.l2_loss(weights_input)+ tf.nn.l2_loss(weights_output)+tf.nn.l2_loss(biases_input)+tf.nn.l2_loss(self.embeddings)
            # cross = -1*tf.sum_reduce(tf.math.log(prediction))
            cross = tf.losses.softmax_cross_entropy(tf.nn.relu(self.train_labels), self.prediction)
            # pred = tf.log(self.prediction)
            # D = tf.reduce_sum(pred, axis = 1)
            # print "D", D.shape
            # pred = pred/D[...,tf.newaxis]
            # print "pred new", pred.shape
            # cross = tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.train_labels, logits = pred)
            # print cross
            # print Config.lam*l2
            self.loss = cross+(Config.lam*l2)

            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_input2, biases_input2, weights_output)

            # intializer
            self.init = tf.global_variables_initializer()   

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print "Initailized"

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)

            # print "start", start
            # print "end", end
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Train Finished."

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print "Starting to predict on test set"
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Saved the test results."
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass(self, embed, weights_input, biases_inpu,  weights_input2, biases_input2, weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """

        #########First Hidden Layer#######################
        wi = tf.transpose(weights_input)
        print "wi", wi.shape
        b = tf.transpose(biases_inpu)
        print "b", b.shape
        hidd_op = tf.math.add(tf.matmul(embed, wi), b)
        print "hidd_op", hidd_op.shape
        hidd_op = tf.pow(hidd_op, 3)


        ##############Second Hidden Layer#################
        wi2 = tf.transpose(weights_input2)
        print "wi2", wi2.shape
        b2 = tf.transpose(biases_input2)
        print "b", b2.shape
        hidd_op2 = tf.math.add(tf.matmul(hidd_op, wi2), b2)
        print "hidd_op", hidd_op2.shape
        hidd_op2 = tf.pow(hidd_op2, 3)

        #############Output Layer#########################
        wo = tf.transpose(weights_output)
        print "wo", wo.shape
        op = tf.matmul(hidd_op2, wo)
        print "op", op.shape
        # pred= tf.math.reduce_max(tf.nn.softmax(op, axis = 1), axis = 1)
        # pred =  tf.nn.softmax(op, axis = 1)
        return op



        # print tf.shape(embed)


def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """
    s=[]
    b=[]
    for i in range(0,3):
        s.append(c.getStack(i))
    for i in range(0,3):        
        s.append(c.getBuffer(i))
    # print s
    for i in range(0,2):
        s.append(c.getLeftChild(s[i],1))
        s.append(c.getRightChild(s[i],1))
        s.append(c.getLeftChild(s[i],2))        
        s.append(c.getRightChild(s[i],2))

    for i in [6,10]:
        s.append(c.getLeftChild(s[i],1))
        s.append(c.getRightChild(s[i+1],1))

    w=[]
    p=[]
    l=[]
    for i in range(0,len(s)):
        w.append(getWordID(c.getWord(s[i])))
        p.append(getPosID(c.getPOS(s[i])))
        if i>5:
            l.append(getLabelID(c.getLabel(s[i])))
    ans = w+p+l
    # print len(ans)
    return ans


def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            # print "sent", sents[i]
            c = parsing_system.initialConfiguration(sents[i])
            # print "afsdf", c
            parsing_system.isTerminal(c)
            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print i, label
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(list(labelDict.values())):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print parsing_system.rootLabel

    print "Generating Traning Examples"
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print "Done."

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)

