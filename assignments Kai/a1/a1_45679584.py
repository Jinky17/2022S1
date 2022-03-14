import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
nltk.download('punkt')
nltk.download('gutenberg')
nltk.download('universal_tagset')
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
from itertools import chain

# Task 1 (2 marks)
def count_pos(document, pos):
    """Return the number of occurrences of words with a given part of speech. To find the part of speech, use 
    NLTK's "Universal" tag set. To find the words of the document, use NLTK's sent_tokenize and word_tokenize.
    >>> count_pos('austen-emma.txt', 'NOUN')
    31998
    >>> count_pos('austen-sense.txt', 'VERB')
    25074"""

    guten_sents = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(nltk.corpus.gutenberg.raw(document))]
    tagged_guten_sents = nltk.pos_tag_sents(guten_sents, tagset="universal")

    count = 0 
    for s in tagged_guten_sents:
        for w in s:
          x = w[1]
          if x == pos:
            count+=1
    return count    

count_pos('austen-emma.txt', 'NOUN')
count_pos('austen-sense.txt', 'VERB')

# Task 2 (2 marks)
def get_top_stem_bigrams(document, n):
    """Return the n most frequent bigrams of stems. Return the list sorted in descending order of frequency.
    The stems of words in different sentences cannot form a bigram. To stem a word, use NLTK's Porter stemmer.
    To find the words of the document, use NLTK's sent_tokenize and word_tokenize.
    >>> get_top_stem_bigrams('austen-emma.txt', 3)
    [(',', 'and'), ('.', "''"), (';', 'and')]
    >>> get_top_stem_bigrams('austen-sense.txt',4)
    [(',', 'and'), ('.', "''"), (';', 'and'), (',', "''")]
    """
    sent_tokens =[nltk.word_tokenize(s) for s in nltk.sent_tokenize(nltk.corpus.gutenberg.raw(document))]
    s = nltk.PorterStemmer()
    dic_stem=[[s.stem(d) for d in sents]for sents in sent_tokens]

    bigrams = []
    for s in dic_stem:
        bigrams += nltk.bigrams([p for p in s])

    c = collections.Counter(bigrams)
    return [b for b, f in c.most_common(n)]

get_top_stem_bigrams('austen-emma.txt', 3)
get_top_stem_bigrams('austen-sense.txt',4)

# Task 3 (2 marks)
def get_same_stem(document, word):
    """Return the list of words that have the same stem as the word given, and their frequencies. 
    To find the stem, use NLTK's Porter stemmer. To find the words of the document, use NLTK's 
    sent_tokenize and word_tokenize. The resulting list must be sorted alphabetically.

    
    >>> get_same_stem('austen-emma.txt','respect')[:5]
    [('Respect', 2), ('respect', 41), ('respectability', 1), ('respectable', 20), ('respectably', 1)]
    >>> get_same_stem('austen-sense.txt','respect')[:5]
    [('respect', 22), ('respectability', 1), ('respectable', 14), ('respectably', 1), ('respected', 3)]
    """
    words =nltk.word_tokenize(nltk.corpus.gutenberg.raw(document))
    stemmer = nltk.stem.PorterStemmer()
    word_stem = stemmer.stem(word)
    
    
    same_stem_list = []
    for w in words:
        if stemmer.stem(w) == word_stem and (w, words.count(w)) not in same_stem_list:
            same_stem_list.append((w, words.count(w)))
    return sorted(same_stem_list, key=lambda x: x[0])

get_same_stem('austen-emma.txt','respect')[:5]
get_same_stem('austen-sense.txt','respect')[:5]


# Task 4 (2 marks)
def most_frequent_after_pos(document, pos):
    """Return the most frequent word after a given part of speech, and its frequency. Do not consider words
    that occur in the next sentence after the given part of speech.
    To find the part of speech, use NLTK's "Universal" tagset.
    >>> most_frequent_after_pos('austen-emma.txt','VERB')
    [('not', 1932)]
    >>> most_frequent_after_pos('austen-sense.txt','NOUN')
    [(',', 5310)]
    """
    
    sents = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(nltk.corpus.gutenberg.raw(document))]
    tagged_sents = nltk.pos_tag_sents(sents, tagset="universal")

    
    filtered_pos = []
    for s in tagged_sents:
        bigrams = nltk.bigrams(s)
        filtered_pos += [w2 for (w1, p1), (w2, p2) in bigrams if p1 == pos]
    c = collections.Counter(filtered_pos)
    return(c.most_common(1))


most_frequent_after_pos('austen-emma.txt','VERB')
most_frequent_after_pos('austen-sense.txt','NOUN')


# Task 5 (2 marks)
def get_word_tfidf(text):
    """Return the tf.idf of the words given in the text. If a word does not have tf.idf information or is zero, 
    then do not return its tf.idf. The reference for computing tf.idf is the list of documents from the NLTK 
    Gutenberg corpus. To compute the tf.idf, use sklearn's TfidfVectorizer with the option to remove the English 
    stop words (stop_words='english'). The result must be a list of words sorted in alphabetical order, together 
    with their tf.idf.
    >>> get_word_tfidf('Emma is a respectable person')
    [('emma', 0.8310852062844262), ('person', 0.3245184217533661), ('respectable', 0.4516471784898886)]
    >>> get_word_tfidf('Brutus is a honourable person')
    [('brutus', 0.8405129362379974), ('honourable', 0.4310718596448824), ('person', 0.32819971943754456)]
    """

    sents = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(nltk.corpus.gutenberg.raw(documents))]

    documents = [gutenberg.raw('austen-emma.txt'), gutenberg.raw('austen-persuasion.txt'), gutenberg.raw('austen-sense.txt'), gutenberg.raw('bible-kjv.txt'), gutenberg.raw('blake-poems.txt'), gutenberg.raw('bryant-stories.txt'), gutenberg.raw('burgess-busterbrown.txt'), gutenberg.raw('carroll-alice.txt'), gutenberg.raw('chesterton-ball.txt'), gutenberg.raw('chesterton-brown.txt'), gutenberg.raw('chesterton-thursday.txt'), gutenberg.raw('edgeworth-parents.txt'), gutenberg.raw('melville-moby_dick.txt'), gutenberg.raw('melville-mrs_dalloway.txt'), gutenberg.raw('milton-paradise.txt'), gutenberg.raw('shakespeare-caesar.txt'), gutenberg.raw('shakespeare-hamlet.txt'), gutenberg.raw('shakespeare-macbeth.txt'), gutenberg.raw('whitman-leaves.txt')]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit(gutenberg.fileids())
    words = tfidf.transform([text])
    words_list = words.toarray()[0]
    words_tfidf = []
    for w in words_list:
        if w != 0:
            words_tfidf.append((tfidf.get_feature_names()[words_list.index(w)], w))
    return sorted(words_tfidf, key=lambda x: x[1], reverse=True)



# DO NOT MODIFY THE CODE BELOW
if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
