import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from doc_utils import get_doc2vec_data, doc_tokenize
import numpy as np
from matplotlib import cm as CM

def heat_plot_two_vectors(v1, t1, v2, t2):
    plt.figure(1, figsize=(10, 4))    
    plt.subplot(2, 1, 1)
    # a cheap trick to make the plot more readable
    im = plt.imshow(np.tile(v1,(10,1)), cmap='hot')
    plt.clim(vmin=-0.5, vmax=0.5)
    plt.colorbar(im)
    plt.title(t1)
    plt.subplot(2, 1, 2)
    im2 = plt.imshow(np.tile(v2,(10,1)), cmap='hot')
    plt.clim(vmin=-0.5, vmax=0.5)
    plt.colorbar(im2)
    plt.title(t2)
    plt.show()

def doc2vec_visualisation(X, Y, model, vocab, title=''):
    vocab_vectors = np.asarray([model[word] for word in vocab if word in model.wv.vocab])
    visualize_two_different_vectors(X1=X, Y1=Y, X2=vocab_vectors, Y2=vocab, title=title)

def visualise_individual_reviews(indices, imdb_reviews, doc2vec_model, title=''):
    revs = imdb_reviews['review'].values
    labels = imdb_reviews['sentiment'].values
    for i in indices:
        review = revs[i]
        sent = labels[i]
        print('Review')
        print(review)
        rev_X = get_doc2vec_data([review], doc2vec_model)
        vocab = [word for word in doc_tokenize(review)]
        doc2vec_visualisation(rev_X, [sent], doc2vec_model, vocab, title=title)

def visualize_two_different_vectors(X1, Y1, X2, Y2, use_pca=True, perplexity=10, pca_components=20, title=''):
    # project both datasets into the same space
    data = np.vstack((X1, X2))
    if use_pca:
    # lower dimensions first with PCA to reduce noise
        pca = PCA(n_components=pca_components)
        data = pca.fit_transform(data)
    x_tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=2000).fit_transform(data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_tsne[:len(X1),0],
                            y=x_tsne[:len(X1),1],
                            mode='markers+text',
                            text=Y1,
                            marker_color='rgba(66, 138, 245, 0.8)'
                            ))
    fig.add_trace(go.Scatter(x=x_tsne[len(X1):,0],
                            y=x_tsne[len(X1):,1],
                            mode='markers+text',
                            text=Y2,
                            marker_color='rgba(245, 66, 66, 0.9)'
                            ))        
    fig.update_traces(textposition='top center')
    fig.update_layout(title=title)
    fig.show()

def visualize_vectors(X, Y, texts, use_pca=True, pca_components=30, perplexity=20, n_iter=2000, title=''):
    # lower dimensions first with PCA to reduce noise
    if use_pca:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
    x_tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter).fit_transform(X)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_tsne[:,0],
                                y=x_tsne[:,1],
                                mode='markers+text',
                                marker_color=Y,
                                text=texts))
    fig.update_traces(textposition='top center')
    fig.update_layout(title=title)
    fig.show()

def plot_matrix(matrix, axislabels=[], title=''):
    im = plt.imshow(matrix, cmap='hot')
    plt.colorbar(im)
    plt.xticks(range(len(axislabels)), axislabels, rotation=90)
    plt.yticks(range(len(axislabels)), axislabels)
    plt.title(title)
    plt.show()