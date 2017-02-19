import unittest



class TestStringMethods(unittest.TestCase):

    def test_punct_removes_punct(self):
        text = u'This, sentence.'
        text = remove_punctuation(text)
        self.assertEqual(text, u'This sentence')
