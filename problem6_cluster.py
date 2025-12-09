import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, AgglomerativeClustering
from sklearn.decomposition import PCA

# Load encoded tag context tuples
with open("q6_contexts.json", "r") as f:
    X = np.array(json.load(f))

# Reduce to 2D for plotting
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)


def run_meanshift():
    ms = MeanShift()
    labels = ms.fit_predict(X)

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=10)
    plt.title("MeanShift Clusters")
    plt.savefig("meanshift.png", dpi=220)
    plt.close()

    print("MeanShift clusters:", len(set(labels)))


def run_agg(linkage, metric=None):
    # Ward does NOT allow metric parameter
    if linkage == "ward":
        agg = AgglomerativeClustering(linkage="ward")
        metric_label = "None"
    else:
        agg = AgglomerativeClustering(linkage=linkage, metric=metric)
        metric_label = metric

    labels = agg.fit_predict(X)

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=10)
    plt.title(f"{linkage} linkage ({metric_label})")
    plt.savefig(f"agg_{linkage}_{metric_label}.png", dpi=220)
    plt.close()

    print(f"agg_{linkage}_{metric_label}.png clusters:", len(set(labels)))


if __name__ == "__main__":
    print("Running clustering...\n")

    run_meanshift()
    run_agg("ward")
    run_agg("average", "manhattan")
    run_agg("average", "cosine")

    print("\nSaved all cluster plot images!")
