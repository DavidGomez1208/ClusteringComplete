import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import preprocessing, clustering, evaluate

st.title('Clustering App')

file = st.file_uploader('upload your dataset',type=['csv'])

if file is not None:
    df = pd.read_csv(file)
else:
    st.stop()

st.subheader('Dataset Preview')
st.dataframe(df.head())

features = st.multiselect('Select the features for clustering',df.columns.to_list())

if len(features) < 2:
    st.warning('Select at least 2 features')
    st.stop()

n_cluster = st.slider('Number of clusters',2,10,3)
linkage = st.selectbox('Select linkage',['ward','single','complete'])

X = preprocessing(df,features)
labels = clustering(X,n_clusters=n_cluster,linkage=linkage)
df['cluster'] = labels
st.subheader('Clustered Dataset')
st.dataframe(df)
silhoutte, davis = evaluate(X,labels)
st.write('Silhouette',silhoutte)
st.write('Davis Index',davis)

if len(features) == 2:
    fig, ax = plt.subplots()
    ax.scatter(df[features[0]], df[features[1]], c=labels)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    st.pyplot(fig)
else:
    st.info('Select exactly 2 features for visualization. ')

model_type = st.selectbox("Select the model ", ["Agglomerative", "DBSCAN"])

if model_type == "Agglomerative":
    labels = clustering(x, n_clusters, linkage)
    df['Cluster'] = labels
    st.subheader("Clustering Results")
    st.dataframe(df)
    silhouette, davies_bouldin = evaluate(x, labels)
    st.write(f"Silhouette Score: {silhouette}")
    st.write(f"Davies-Bouldin Score: {davies_bouldin}")

elif model_type == "DBSCAN":
    eps = st.slider("Select epsilon value", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    min_samples = st.slider("Select minimum samples", min_value=1, max_value=10, value=5)
    labels = model_selection(x, model_type="DBSCAN", eps=eps, min_samples=min_samples)
    df['Cluster'] = labels
    st.subheader("Clustering Results")
    st.dataframe(df)
    silhouette, davies_bouldin = evaluate(x, labels)
    st.write(f"Silhouette Score: {silhouette}")
    st.write(f"Davies-Bouldin Score: {davies_bouldin}")

else:
    st.warning("Please select a valid model type.")