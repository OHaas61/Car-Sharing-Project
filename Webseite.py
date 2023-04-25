
# Datahandling libraries
import pandas as pd
import streamlit as st
import folium
import itertools


# ML libraries -> DBSCAN
from sklearn.cluster import DBSCAN
from streamlit_folium import st_folium, folium_static


# we now have a smaller data frame with the pickup-longitude aswell as the pickup-latitude
# because we don't need the other columns
# Website
# adding a Titel to the website
st.title('Spatial Clustering mit DBSCAN')
# adding a Title for the first "chapter"
st.header('Zusammenfassung der Webseite')
'Diese Website soll den räumlichen Clustering-Algorithmus DBSCAN etwas nachvollziehbarer machen. Die Theorie beginnt allerdings nicht bei null sondern geht von einem bestimmten Vorwissen in Sachen Algorithmen, Machine Learning und Datensätzen aus. Als Datengrundlage für das Clustering dient ein Datensatz der Firma «Uber». In diesem Datensatz ist jeweils Längen- und Breitengrad der Abholstellen von Uberkunden – hauptsächlich in New York – enthalten'
'Die Website ist in zwei Teile geteilt: Ein Theorieteil und ein Ergebnisteil:'
'-  Im Theorieteil werden Grundinformation über den DBSCAN-Algorithmus geliefert und erklärt wie der DBSCAN-Algorithmus beim Clustern (zusammenfassen von Datenpunkten) vorgeht.'
'-  Im Ergebnisteil werden die Ergebnisse des DBSCAN-Algorithmus visualisiert (welche Gruppen wurden aufgrund des Datensatzes gebildet?). Diese Ergebnisse werden auf einer interaktiven Karte zusammengefasst, um so nachvollziehbar zu machen, wie der DBSCAN-Algorithmus funktioniert.'

st.header('Was macht der DBSCAN-Algorithmus?')
#adding a subtitle
st.subheader('Allgemeine Informationen')
'-  Ein Clustering Algorithmus ist eine Form von unsupervised machine learning, bei dem ähnliche Datenpunkte in einem Datenset mit einem Algorithmus zusammengefasst werden. Dabei ist das Ziel, dass sich die Datenpunkte innerhalb eines Clusters sehr ähnlich sind. Die Unterschiede von Datenpunkten aus verschiedenen Clustern sollten wiederum sehr gross sein.'
'-  DBSCAN steht für Density-Based Spatial Clustering of Applications with Noise (Deutsch: Dichtebasierte räumliche Clusteranalyse mit Rauschen)'
'-  Der DBSCAN-Algorithmus ist ein also sogenannter räumlicher Clustering Algorithmus. Das bedeutet die Ähnlichkeit von Datenpunkten wird basierend auf ihrer räumlichen Nähe zueinander ermittelt. Diese Datenpunkte werden dann zu Clustern (Gruppen) zusammengefasst (Abbildung 1).'
# adding an image
st.image('pictures/Abbildung1.png')
# adding an image caption
st.caption('Abbildung 1')
st.subheader('Vorgehensweise des DBSCAN-Algorithmus')
'Beim DBSCAN-Algorithmus unterscheidet man grundsätzlich zwischen folgenden zwei Arten von Punkten:'
'-  Kernobjekte --> Datenpunkte die zu einem Cluster gehören'
'-  Noise --> Datenpunkte die keinem Cluster zugeordnet werden können'
'Der DBSCAN-Algorithmus hat mehrere Parameter, die zwei wichtigsten sind jedoch:'
'-  Epsilon --> Dieser Parameter legt den maximalen Abstand fest, die zwei Punkte voneinander haben dürfen, damit sie als verbundene Punkte gelten'
'-  min_samples --> Dieser Parameter legt ein Minimum an verbundenen Punkten fest, die ein Cluster enthalten muss'
'1.	Der DBSCAN-Algorithmus beginnt nun an einem zufälligen Datenpunkt und schaut, ob es in einem Umkreis von Epsilon einen weiteren Punkt gibt. Angenommen er findet zwei weitere Datenpunkte, dann führen diese dasselbe erneut durch, bis er von allen Punkten aus, keinen Punkt mehr in einem Umkreis von Epsilon finden kann. Somit ist ein erstes Cluster gefunden.'
'2.	Mit den noch nicht zugeordneten Datenpunkten wird nun das Selbe durchführt bis alle Cluster gefunden sind. Es bleiben unter Umständen noch Punkte übrig, die keinem Cluster zugeordnet werden können, weil ihre Gruppe von verbundenen Punkten weniger Punkte als min_samples enthält.'
st.image('pictures/Abbildung2.png')
st.caption('Abbildung 2: Datenpunkte (schwarz, links) auf welche der DBSCAN-Algorithmus angewendet wird. Es entstehen Cluster (farbig) und Noisepunkte (schwarz, rechts)')
st.header('Ergebnisse')
# loading data set
df = pd.read_csv('data/uber.csv')

# "cleaning" data set
del df['Unnamed: 0'] 
del df['key']
del df['fare_amount']
del df['pickup_datetime']
del df['dropoff_latitude']
del df['dropoff_longitude']
del df['passenger_count']

# defining a smaller Dataset for less computational effort
df_small = df[:20000]

# deleting points with latitude or longitude zero (we only want the points in New York)
df_small = df_small[(df_small['pickup_latitude'] != 0) | (df_small['pickup_longitude'] != 0)]

# creating the output of the DBSCAN algorithm
clusters = DBSCAN(eps = st.slider("Epsilon",0.0020,0.0100,0.0093,0.0001,format="%f"), min_samples = st.slider("min_samples",10,200,100)).fit(df_small)

# creating an empty list that stores every new cluster dataframe
clusters_list = []

# converting the output of the DBSCAN algorithm into a dataframe per cluster
labels = clusters.labels_ # storing the cluster-labels in a variable "labels"

# storing the number of clusters in a variable by creating a new set of labels and subtracting 1
# if there is noise (-1) contained in the labels
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# iterate over each cluster
for i in range(num_clusters):
    # select all the points that are in cluster i
    cluster_data = df_small[labels == i]

    # with an f-string, each iteration cluster_name changes so that the for-iteration returns
    # each cluster in a new dataframe named cluster_1, cluster_2 and so on
    cluster_name = f'cluster_{i+1}'

    # changing the name of the colums from e.g cluster_1 to pickup_longitude/latiude_cluster_1
    cluster_data.columns = [f'{col}_{cluster_name}' for col in cluster_data.columns]

    # adding the dataframe from each new iteration to the empty list clusters_list
    clusters_list.append(cluster_data)

# adding noise points to the cluster list, the last element of the cluster list is the noise
# data frame
noise = df_small[labels == -1]
noise.columns = [f'{col}_noise' for col in noise.columns]
clusters_list.append(noise)

print(num_clusters)


colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'lightgray']
color_iterator = itertools.cycle(colors)

m = folium.Map()
m.fit_bounds([(40.893895, -74.045570),(40.588878, -73.679881)])
for i in range(num_clusters-1):
    longitudestring = "pickup_longitude_cluster_" + str(i+1)  # create string with current number
    latitudestring = "pickup_latitude_cluster_" + str(i+1)  # create string with current number
    color = next(color_iterator)  # get the next color in the sequence
    for _, row in clusters_list[i].head(20).iterrows():
        folium.Marker(location=[row[latitudestring], row[longitudestring]], icon=folium.Icon(color=color)).add_to(m)
for i,row in clusters_list[num_clusters].head(20).iterrows():
    # add each row to the map
    folium.Marker(location=[row['pickup_latitude_noise'],row['pickup_longitude_noise']],icon=folium.Icon(color='black')).add_to(m)                             

# determines how big the map will be
st_folium(
    m,
    height=400,
    width=700,
)

'Mit den beiden Parametern Epsilon = 0.0093 und min_samples = 100 wurden durch den DBSCAN-Algorithmus vier Cluster generiert (rot, pink, grün, blau). Weiter gibt es einige noise-Punkte (schwarz), die keinem Cluster zugeordnet werden können. Bei den Clustern können dichte Punkteansammlungen festgestellt werden, während die noise-Punkte mehrheitlich einzeln zu finden sind.'
'Mit dem verändern der Parameter, könnte man nun auch noch einige dieser noise-Punkte zu Clustern hinzufügen z.B. durch Vergrösserung von Epsilon. Dabei besteht aber wiederum die Gefahr, dass die Anzahl der Cluster kleiner wird und wenige grosse Cluster generiert würden, was nicht sehr sinnvoll ist. Eine Möglichkeit mehr Cluster zu erhalten, wäre eine Verkleinerung des Parameters min_samples. Dabei besteht aber wiederum die Gefahr von sehr vielen, kleinen Cluster und dies ist wiederum auch nicht sinnvoll.'
'Schlussendlich ist es also nicht einfach die Parameter Epsilon und min_samples so zu optimieren, um eine optimale Clusterperformance zu erhalten (nicht wenige grosse Cluster, nicht viele kleine Cluster und sinnvolles zuordnen von noise-Punkten)'



