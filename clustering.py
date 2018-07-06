import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

def hierarchical_clustering(df, tsne_mapping, num_labels, plot=False, dt='exp'):
    clusters = {}
    scores = []
    years = df.index.get_level_values(level='year').unique()
    if type(num_labels) == int:
        num_labels = [num_labels for i in years]

    for idx,year in enumerate(years):
        #print(year)
        x = df.loc[year].values
        x_embedded = tsne_mapping.loc[year].values
        lk = linkage(x, 'ward')
        labels = fcluster(lk, num_labels[idx], 'maxclust')
        countries = df.loc[year].index.get_level_values(level='origin')

        features_means = df.loc[year].groupby(labels, axis=0).mean()
        clusters[year] = {i:[c for j,c in enumerate(countries) if labels[j]==i] for i in range(1,num_labels[idx]+1)}
        scores.append(silhouette_score(x, labels))

        if plot:
            #fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, gridspec_kw = {'width_ratios':[2, 1]}, figsize=(30,10))
            plt.figure(figsize=(25,18))
            plt.suptitle('Agrupamento Hierarquico - {:d}'.format(year), fontsize=38)

            ax1 = plt.subplot(211)
            ax1.set_title('Dendrograma', fontsize=25)
            ax1.set_xlabel('País Idx', fontsize=20)
            ax1.set_ylabel('Distância', fontsize=20)
            t = lk[-num_labels[idx]+1,2]
            dendrogram(lk, leaf_rotation=90., no_labels=True, ax=ax1, color_threshold=t)

            ax2 = plt.subplot(223)
            ax2.set_title('TSNE', fontsize=25)
            ax2.scatter(x_embedded[:,0],x_embedded[:,1], c=labels, s=40, cmap='tab10')

            ax3 = plt.subplot(224)
            ax3.set_xticks(range(21))
            ax3.set_title('Feature means', fontsize=25)
            ax3.set_xlabel('Features', fontsize=20)
            ax3.set_ylabel('Peso', fontsize=20)
            ax3.plot(features_means.T)

            plt.savefig('figures/hierarquico/h_{:s}_{:d}_{:d}.png'.format(dt, year, num_labels[idx]))
            #plt.show()


    return scores, clusters


def k_means(df, tsne_mapping, num_labels, plot=False, dt='exp'):
    clusters = {}
    scores = []
    years = df.index.get_level_values(level='year').unique()
    if type(num_labels) == int:
        num_labels = [num_labels for i in years]

    for idx,year in enumerate(years):
        x = df.loc[year].values
        x_embedded = tsne_mapping.loc[year].values
        model = KMeans(num_labels[idx], random_state=23, init='random', n_init=50).fit(x)
        labels = model.labels_
        features_means = pd.DataFrame(model.cluster_centers_, columns=df.loc[year].columns, index=range(1,num_labels[idx]+1))

        countries = df.loc[year].index.get_level_values(level='origin')
        clusters[year] = {i:[c for j,c in enumerate(countries) if labels[j]==i] for i in range(1,num_labels[idx]+1)}
        scores.append(silhouette_score(x, labels))

        if plot:
            #fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, gridspec_kw = {'width_ratios':[2, 1]}, figsize=(30,10))
            plt.figure(figsize=(20,7))
            plt.suptitle('k-means - {:d}'.format(year), fontsize=30)

            ax1 = plt.subplot(121)
            ax1.set_title('TSNE', fontsize=20)
            ax1.scatter(x_embedded[:,0],x_embedded[:,1], c=labels, s=40, cmap='tab10')

            ax2 = plt.subplot(122)
            ax2.set_xticks(range(21))
            ax2.set_title('Feature means', fontsize=20)
            ax2.set_xlabel('Features', fontsize=15)
            ax2.set_ylabel('Weigth', fontsize=15)
            ax2.plot(features_means.T)

            plt.savefig('figures/k_means/h_{:s}_{:d}_{:d}.png'.format(dt, year, num_labels[idx]))
            #plt.show()


    return scores, clusters
