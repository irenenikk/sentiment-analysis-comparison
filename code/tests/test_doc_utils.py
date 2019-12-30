from unittest import TestCase
from doc_utils import doc_tokenize

class TestDocUtils(TestCase):

    def test_doc_tokenization(self):
        doc = "I am a WEIRDLY €€€ written document."
        self.assertEqual(doc_tokenize(doc), ['I', 'am', 'a', 'WEIRDLY', '€€€', 'written', 'document', '.'])
