import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
nltk.download('punkt')
nltk.download('gutenberg')
from sklearn.feature_extraction.text import TfidfVectorizer
import collections

# Task 1 (1 mark)
import collections
def get_top_stems(document, n):
    """Return a list of the n most frequent stems of a Gutenberg document, sorted by 
    frequency in descending order. Don't forget to remove stop words before counting 
    the stems.
    >>> get_top_stems('austen-emma.txt', 10)
    [',', '.', '--', "''", ';', '``', 'mr.', '!', "'s", 'emma']
    >>> get_top_stems('austen-sense.txt', 7)
    [',', '.', "''", ';', '``', '--', 'elinor']
    """
    stopwords = nltk.corpus.stopwords.words('english')
    raw = nltk.corpus.gutenberg.raw(document)
    tokens = [w for s in nltk.sent_tokenize(raw) for w in nltk.word_tokenize(s)
              if w.lower() not in stopwords]
    s = nltk.PorterStemmer()
    stems = [s.stem(t) for t in tokens]
    c = collections.Counter(stems)
    #print(c.most_common(n))
    return [w for w, f in c.most_common(n)]

# Task 2 (1 mark)
def get_top_pos_bigrams(document, n):
    """Return the n most frequent bigrams of parts of speech. Return the list sorted in descending order of frequency.
    The parts of speech of words in different sentences cannot form a bigram. Use the universal pos tag set.
    >>> get_top_pos_bigrams('austen-emma.txt', 3)
    [('NOUN', '.'), ('PRON', 'VERB'), ('DET', 'NOUN')]
    """
    sent_tokens = [word_tokenize(s) for s in sent_tokenize(nltk.corpus.gutenberg.raw(document))]
    sent_pos = nltk.pos_tag_sents(sent_tokens, tagset='universal')
    bigrams = []
    for s in sent_pos:
        bigrams += nltk.bigrams([p for w, p in s])
    c = collections.Counter(bigrams)
    #print(c.most_common(n))
    return [b for b, f in c.most_common(n)]


# Task 3 (1 mark)
def get_pos_after(document, word):
    """Return the distribution of frequencies of the parts of speech occurring after a word. Return the result sorted by 
    frequency in descending order. Do not consider words that occur in different sentences. Use the 
    universal pos tag set.
    >>> get_pos_after('austen-emma.txt','the')
    [('NOUN', 3434), ('ADJ', 1148), ('ADV', 170), ('NUM', 61), ('VERB', 24), ('.', 7)]
    """
    sent_tokens = [word_tokenize(s) for s in sent_tokenize(nltk.corpus.gutenberg.raw(document))]
    sent_pos = nltk.pos_tag_sents(sent_tokens, tagset='universal')
    filtered_pos = []
    for s in sent_pos:
        bigrams = nltk.bigrams(s)
        filtered_pos += [p2 for (w1, p1), (w2, p2) in bigrams if w1==word]
    c = collections.Counter(filtered_pos)
    return c.most_common()

# Task 4 (1 mark)
def get_top_word_tfidf(document, n):
    """Return the list of n words with highest tf.idf. The reference for computing 
    tf.idf is the NLTK Gutenberg corpus. The list of words must be sorted by frequency
    in descending order.

    >>> get_top_word_tfidf('austen-emma.txt', 3)
    ['emma', 'mr', 'harriet']
    """
    tfidf = TfidfVectorizer(input='content',stop_words='english')
    data = [nltk.corpus.gutenberg.raw(f) for f in nltk.corpus.gutenberg.fileids()]
    tfidf.fit(data)
    result = tfidf.transform([nltk.corpus.gutenberg.raw(document)]).toarray()
    words = tfidf.get_feature_names()
    sorted_words = sorted(words, key=lambda x: result[0, words.index(x)], reverse=True)
    #print('thee:', result[0, words.index('thee')])
    #print('thel:', result[0, words.index('thel')])
    return sorted_words[:n]

# Task 5 (1 mark)
def get_top_sentence_tfidf(document, n):
    """Return the positions of the n sentences which have the largest average tf.idf. The list of sentences
    must be returned in the order of occurrence in the document. The reference for computing 
    tf.idf is the list of sentences from the NLTK Gutenberg corpus.
    >>> get_top_sentence_tfidf('austen-emma.txt', 3)
    [5668, 5670, 6819]
    """
    tfidf = TfidfVectorizer(input='content',stop_words='english')
    data = [s for f in nltk.corpus.gutenberg.fileids() for s in sent_tokenize(nltk.corpus.gutenberg.raw(f))]
    tfidf.fit(data)
    sentences = sent_tokenize(nltk.corpus.gutenberg.raw(document))
    tfidf_result = tfidf.transform(sentences).toarray()
    tfidf_avg = [np.mean(r) for r in tfidf_result]
    top_positions = sorted(range(len(sentences)), key=lambda x: tfidf_avg[x], reverse=True)[:n]
    return sorted(top_positions)


# DO NOT MODIFY THE CODE BELOW
if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
