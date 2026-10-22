import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

def preprocessing(df,features):
  X = df[features].copy()

  numerical = X.select_dtypes(include=['int64', 'float64']).columns
  categorical = X.select_dtypes(include=['object']).columns

  scaler = StandardScaler()
  X[numerical] = scaler.fit_transform(X[numerical])
  encoder = LabelEncoder()
  X[categorical] = X[categorical].apply(encoder.fit_transform)

  return X

def clustering(X, n_clusters,linkage):
  model = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage)
  model.fit(X)
  return model.labels_

def evaluate(X,labels):
  return silhouette_score(X, labels), davies_bouldin_score(X, labels)


def model_selection(x, model_type, n_clusters=None, eps=None, min_samples=None):
    if model_type == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=nclusters)
        model.fit(x)
        return model.labels
    elif model_type == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=minsamples)
        model.fit(x)
        return model.labels
    else:
        raise ValueError("Invalid model type. Choose either 'Agglomerative' or 'DBSCAN'.") 