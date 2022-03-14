import nltk
import unittest
import numpy as np

import a1


class TestBasic(unittest.TestCase):
    def test_task1(self):
        self.assertListEqual(a1.get_top_stems('austen-emma.txt', 10),
                             [',', '.', '--', "''", ';', '``', 'mr.', '!', "'s", 'emma'])
        self.assertListEqual(a1.get_top_stems('austen-sense.txt', 7),
                             [',', '.', "''", ';', '``', '--', 'elinor'])

    def test_task2(self):
        self.assertListEqual(a1.get_top_pos_bigrams('austen-emma.txt', 3),
                             [('NOUN', '.'), ('PRON', 'VERB'), ('DET', 'NOUN')])


    def test_task3(self):
        self.assertListEqual(a1.get_pos_after('austen-emma.txt','the'), 
                             [('NOUN', 3434), ('ADJ', 1148), ('ADV', 170), ('NUM', 61), ('VERB', 24), ('.', 7)])

    def test_task4(self):
        self.assertListEqual(a1.get_top_word_tfidf('austen-emma.txt', 3),
                             ['emma', 'mr', 'harriet'])

    def test_task5(self):
        self.assertListEqual(a1.get_top_sentence_tfidf('austen-emma.txt', 3),
                             [5668, 5670, 6819])

class TestAdvanced(unittest.TestCase):
    def test_task1(self):
        self.assertListEqual(a1.get_top_stems('blake-poems.txt', 11),
                             [',', '.', ';', ':', '!', '?', '``', 'littl', "'s", 'thee', 'love'])

    def test_task2(self):
        self.assertListEqual(a1.get_top_pos_bigrams('blake-poems.txt', 4),
                             [('NOUN', '.'), ('DET', 'NOUN'), ('NOUN', 'NOUN'), ('ADJ', 'NOUN')])

    def test_task3(self):
        self.assertListEqual(a1.get_pos_after('blake-poems.txt','my'), 
                             [('NOUN', 59), ('ADJ', 12), ('VERB', 1)])

    def test_task4(self):
        self.assertListEqual(a1.get_top_word_tfidf('blake-poems.txt', 2),
                             ['thee', 'thel'])

    def test_task5(self):
        self.assertListEqual(a1.get_top_sentence_tfidf('blake-poems.txt', 2),
                             [297, 334])


if __name__ == "__main__":
    unittest.main()
