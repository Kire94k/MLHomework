from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt 


df = pd.read_csv('uge15/Uge15Data.csv')
train_data = pd.DataFrame(df)

X = train_data[["Age", "Income"]].values
#y = train_data["Income"].values

errors = []
for i in range(1,21):
    kmeans = KMeans(n_clusters=i).fit(X)
    errors.append(kmeans.inertia_)

plt.plot(range(1,21), errors)
plt.xlabel("No of clusters(k)")
plt.ylabel("Errors (Inertia)")
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.show()

finalresult = KMeans(n_clusters=5)
y = finalresult.fit_predict(X)

clusterfuck = finalresult.cluster_centers_
print(clusterfuck)

plt.scatter(X[:,0],X[:,1], c=y)
plt.scatter(finalresult.cluster_centers_[:,0], finalresult.cluster_centers_[:,1],c='black', marker='+')
plt.show()