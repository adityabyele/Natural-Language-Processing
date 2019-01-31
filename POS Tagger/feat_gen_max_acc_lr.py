#crf-86.56
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
    # if word.lower().endswith("ous"):
    #     ftrs.append("IS_OUS")    
    if word.lower().endswith("est"):
        ftrs.append("IS_EST")
    if word[0].isupper():        
        ftrs.append("IS_CAPITALIZED")
    if "!" in word:
        ftrs.append("IS_EXCL")
    if word.endswith("?"):
        ftrs.append("IS_Q")    
    # if "," in word :
    #     ftrs.append("HAS_COMMA")
    # if "," in word or "\"" in word or "?" or "!" or "'":
    #     ftrs.append("HAS_PUNC")    
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
    if word.lower().startswith("dis") or word.lower().startswith("un") or word.lower().startswith("im") or word.lower().startswith("anti"):
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
    # if word.lower()=="fuck" or word.lower()=="fck" or ("#" in word and "!" in word) or word.lower()=="bastard" or word.lower()=="bstrd" or word.lower()=="dick" or word.lower()=="pussy" or word.lower()=="rascal":
    #     ftrs.append("HAS_SLANG")
    if word.lower().startswith("re"):        
        ftrs.append("IS_RE")
    if word.lower().endswith("ing"):
        ftrs.append("IS_ING")