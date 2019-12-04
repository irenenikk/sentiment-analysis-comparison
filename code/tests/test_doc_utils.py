from unittest import TestCase
from doc_utils import doc_tokenize

class TestDocUtils(TestCase):

    def test_doc_tokenization(self):
        doc = "I am4355 a WEIRDLY €€€ written d.oc.u.ment"
        self.assertEqual(doc_tokenize(doc), ['i', 'am', 'a', 'weirdly', 'written', 'document'])
