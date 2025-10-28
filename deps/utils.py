import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

def min_max_transformation_cvi(cvi):
    return (cvi - min(cvi))/(max(cvi) - min(cvi))

def find_optimal_number_of_clusters(max_n_clusters, path, training_data, seed, 
                                    cov_type=["spherical", "diag", "full", "tied"]):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(path)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    fig6, ax6 = plt.subplots()
    
    results = {}

    for r in cov_type:
        sil_score = []
        db_score = []
        ch_score = []
        bic_score = []

        for cl in tqdm(range(2, max_n_clusters), total=(max_n_clusters-3), desc=f"% investigated clusters for {r} covariance type"):
            #model = GMMFitter(n_clusters=cl, covariance_type=r, max_iter=100,
            #                  n_init=80, init_params='k-means++', random_state=0).fit(X=training_data)
            
            model = GaussianMixture(n_components=cl, covariance_type=r, random_state=seed, 
                                    init_params='k-means++', max_iter=1000, tol=0.00001, n_init=100
                                    ).fit(training_data)
            
            _labels = model.predict(training_data)
            
            cvi_scores = cvi(model, training_data, _labels)

            bic_score.append(cvi_scores["bic_score"])
            sil_score.append(cvi_scores["silhouette_score"])
            db_score.append(cvi_scores["davies_bouldin"])
            ch_score.append(cvi_scores["calinski_harabasz"])

        bic_score = np.asarray(bic_score) + np.abs(np.min(bic_score))
        sil_score = np.asarray(sil_score)
        db_score = np.asarray(db_score)
        ch_score = np.asarray(ch_score)

        transformed_sil = min_max_transformation_cvi(sil_score)
        transformed_bic = min_max_transformation_cvi(1/(bic_score + 1))
        transformed_db = min_max_transformation_cvi(1/db_score)
        transformed_ch = min_max_transformation_cvi(ch_score)

        ccvi = (transformed_sil + transformed_db + transformed_bic + transformed_ch)/4
        pcvi = np.power((transformed_sil * transformed_db * transformed_bic * transformed_ch), 1/4)
        
        results[r] = pcvi

        ax1.plot(list(range(2, max_n_clusters)), bic_score, "-*", linewidth=2, label=f"Cov. type: {r}")
        ax2.plot(list(range(2, max_n_clusters)), sil_score, "-*", linewidth=2, label=f"Cov. type: {r}")
        ax3.plot(list(range(2, max_n_clusters)), db_score, "-*", linewidth=2, label=f"Cov. type: {r}")
        ax4.plot(list(range(2, max_n_clusters)), ch_score, "-*", linewidth=2, label=f"Cov. type: {r}")
        ax5.plot(list(range(2, max_n_clusters)), ccvi, "-*", linewidth=2, label=f"Cov. type: {r}")
        ax6.plot(list(range(2, max_n_clusters)), pcvi, "-*", linewidth=2, label=f"Cov. type: {r}")

    ax1.set_xlabel("Number of Clusters", fontsize=14, fontstyle='normal', fontname="Arial")
    ax1.set_ylabel("BIC score", fontsize=14, fontstyle='normal', fontname="Arial")
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.legend(fontsize=14)
    fig1.savefig(os.path.join(path, "Number of Clusters (BIC).png"))
    fig1.savefig(os.path.join(path, "Number of Clusters (BIC).svg"))
    plt.close(fig1)

    ax2.set_xlabel("Number of Clusters", fontsize=14, fontstyle='normal', fontname="Arial")
    ax2.set_ylabel("Silhouette score", fontsize=14, fontstyle='normal', fontname="Arial")
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.legend(fontsize=14)
    fig2.savefig(os.path.join(path, "Number of Clusters (Silhouette).png"))
    fig2.savefig(os.path.join(path, "Number of Clusters (Silhouette).svg"))
    plt.close(fig2)

    ax3.set_xlabel("Number of Clusters", fontsize=14, fontstyle='normal', fontname="Arial")
    ax3.set_ylabel("Davies-Bouldin score", fontsize=14, fontstyle='normal', fontname="Arial")
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.legend(fontsize=14)
    fig3.savefig(os.path.join(path, "Number of Clusters (DB).png"))
    fig3.savefig(os.path.join(path, "Number of Clusters (DB).svg"))
    plt.close(fig3)

    ax4.set_xlabel("Number of Clusters", fontsize=14, fontstyle='normal', fontname="Arial")
    ax4.set_ylabel("Cal. Harabasz score", fontsize=14, fontstyle='normal', fontname="Arial")
    ax4.tick_params(axis='both', which='major', labelsize=14)
    ax4.legend(fontsize=14)
    fig4.savefig(os.path.join(path, "Number of Clusters (CH).png"))
    fig4.savefig(os.path.join(path, "Number of Clusters (CH).svg"))
    plt.close(fig4)

    ax5.set_xlabel("Number of Clusters", fontsize=14, fontstyle='normal', fontname="Arial")
    ax5.set_ylabel("Average CVI", fontsize=14, fontstyle='normal', fontname="Arial")
    ax5.tick_params(axis='both', which='major', labelsize=14)
    ax5.legend(fontsize=14)
    fig5.savefig(os.path.join(path, "Number of Clusters (ccvi).png"))
    fig5.savefig(os.path.join(path, "Number of Clusters (ccvi).svg"))
    plt.close(fig5)

    ax6.set_xlabel("Number of Clusters", fontsize=14, fontstyle='normal', fontname="Arial")
    ax6.set_ylabel("Product CVI", fontsize=14, fontstyle='normal', fontname="Arial")
    ax6.tick_params(axis='both', which='major', labelsize=14)
    ax6.legend(fontsize=14)
    fig6.savefig(os.path.join(path, "Number of Clusters (pcvi).png"))
    fig6.savefig(os.path.join(path, "Number of Clusters (pcvi).svg"))
    plt.show()
    plt.close(fig6)
    
    return results


def cvi(model, X, labels=None):
    # calculate individuals cluster validity index for the provided clusters
    return {"silhouette_score": silhouette_score(X, labels=labels),
            "davies_bouldin": davies_bouldin_score(X=X, labels=labels),
            "calinski_harabasz": calinski_harabasz_score(X=X, labels=labels),
            "bic_score": model.bic(X=X)}
    

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

