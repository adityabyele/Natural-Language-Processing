#!/bin/python

def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    pass

def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")
    if word.startswith("@"):
        ftrs.append("IS_HANDLENAME")
    if word.lower().startswith("http") or word.lower().endswith(".com") or word.lower().startswith("www"):
        ftrs.append("IS_URL")
    if word.lower().endswith("ly"):
        ftrs.append("IS_LY")
    if word.lower().endswith("er"):
        ftrs.append("IS_ER")
    if word.lower().endswith("ous"):
        ftrs.append("IS_OUS")    
    if word.lower().endswith("est"):
        ftrs.append("IS_EST")
    if word[0].isupper():        
        ftrs.append("IS_CAPITALIZED")
    if "!" in word:
        ftrs.append("IS_EXCL")
    if word.endswith("?"):
        ftrs.append("IS_Q")    
    if "," in word :
        ftrs.append("HAS_COMMA")
    if "," in word or "\"" in word or "?" or "!" or "'":
        ftrs.append("HAS_PUNC")    
    if "'" in word:
        ftrs.append("HAS_APOS")    
    if "\"" in word:
        ftrs.append("HAS_QUOTE")    
    if word.lower().endswith("able"):
        ftrs.append("IS_ABLE")
    if word.lower().endswith("ful"):
        ftrs.append("IS_FUL")
    if word.lower().startswith("en"):
        ftrs.append("IS_EN")
    if word.lower().startswith("im"):
        ftrs.append("IS_IM")
    if word.lower().startswith("anti"):
        ftrs.append("IS_ANTI")
    if word.lower().startswith("dis") or word.lower().startswith("un"):
        ftrs.append("IS_NEG")
    if word.lower() == "is":
        ftrs.append("IS_IS")
    if word.lower() == "not":
        ftrs.append("IS_NOT")
    if word.lower().startswith(":"):
        ftrs.append("HAS_COLON")
    if word.lower().startswith("rt"):
        ftrs.append("HAS_RT")
    if word.lower().endswith("ed"):        
        ftrs.append("HAS_ED")
    if "-" in word:
        ftrs.append("HAS_HYPHEN")
    if "#" in word:
        ftrs.append("HAS_HASH")
    if word.lower()=="fuck" or word.lower()=="fck" or ("#" in word and "!" in word) or word.lower()=="bastard" or word.lower()=="bstrd" or word.lower()=="dick" or word.lower()=="pussy" or word.lower()=="rascal":
        ftrs.append("IS_SLANG")
    if word.lower().startswith("re"):        
        ftrs.append("IS_RE")
    if word.lower().endswith("ing"):
        ftrs.append("IS_ING")
    if len(word)==1:
        ftrs.append("COUNT_1")
    if len(word)==2:
        ftrs.append("COUNT_2")
    if "." in word:        
        ftrs.append("HAS_FULLSTOP")
    if word.lower().startswith("&"):
        ftrs.append("IS_TAG")
    #-

    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "__main__":
    sents = [
    [ "http://urlify.com", "I", "omnious","lowest", "food","!" ]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)
