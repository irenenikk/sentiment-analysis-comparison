import re
import gensim
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)


imdb_data_folder = 'aclImdb'
imdb_sentiments = ['pos', 'neg']
subfolders = ['train', 'test']
review_list = []
review_id_list = []
review_grade_list = []
for sent in imdb_sentiments:
    for subf in subfolders:
        for review_file in os.listdir(os.path.join(imdb_data_folder, subf, sent)):
            idd = review_file.split('_')[0]
            review_id_list += [idd]
            grade = re.search('_(.*)\.txt', review_file).group(1)
            review_grade_list += [grade]
            f = open(os.path.join(imdb_data_folder, subf, sent, review_file), 'r+')
            review = f.read()
            review_list += [review]
reviews = pd.DataFrame(list(zip(review_list, review_id_list, review_grade_list)), columns=['review', 'id', 'grade'])


def doc_tokenize(doc):
    return [x.lower() for x in re.sub(r'[^a-zA-Z\s]', '', a).split()]


documents = [TaggedDocument(doc_tokenize(doc), [i]) for i, doc in enumerate(reviews['review'].values)]


vec_size = 100
window_size = 2
min_count = 4
model = Doc2Vec(documents, vector_size=vec_size, window=2, min_count=min_count, workers=4)


fname = get_tmpfile(f'doc2vec_{vec_size}_{window_size}_{min_count}')
model.save(fname)
model = Doc2Vec.load(fname)
