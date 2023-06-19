import pandas as pd
from sklearn.cluster import KMeans


def k_means_elbow_curve(X, random_state):
    """Computes and plots elbow curve with k-means algorithm.

    Args:
        X (pd.DataFrame): features
        random_state (int): randome state to fix

    Returns:
    """
    # Create an empty list to store the sum of squared distances
    sum_distance = []

    # Define the range of K values to try
    k_values = range(1, 11)
    import matplotlib.pyplot as plt

    # Calculate the sum of squared distances for each K value
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        kmeans.fit(X)
        sum_distance.append(kmeans.inertia_)

    # Plot the Elbow Curve
    plt.ion()  # allow interactive terminal
    plt.plot(k_values, sum_distance, "bx-")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Sum of Squared Distances")
    plt.title("Elbow Curve for K-means Clustering")
    plt.show()


def k_means_clustering(
    X, cols_numerical, cols_categorical, nb_clusters, random_state, p_value_threshold
):
    """Computes and plots elbow curve with k-means algorithm.

    Args:
        X (pd.DataFrame): features
        cols_numerical (list) : list of numerical columns from X
        cols_categorical (list) : list of categorical columns from X
        nb_clusters : number of clusters for keamns algo
        random_state (int): randome state to fix

    Returns:
        results_df_signif (pd.DataFrame): DataFrame contaning significant variables their mean or mode value, and the cluster nb
    """
    # Create an empty list to store the sum of squared distances
    kmeans = KMeans(n_clusters=nb_clusters, n_init="auto", random_state=random_state)
    kmeans.fit(X)

    # Get the cluster labels
    cluster_labels = kmeans.labels_
    cluster_labels_df = pd.DataFrame({"Cluster Labels": cluster_labels})

    # Concatenate the df with the cluster labels
    new_df = pd.concat([X, cluster_labels_df], axis=1)

    import scipy.stats as stats

    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=["Variable", "P-Value", "Cluster"])

    # Perform t-test and chi-square test for each cluster against the rest
    for cluster_of_interest in range(nb_clusters):
        rest_of_clusters = [c for c in range(nb_clusters) if c != cluster_of_interest]

        # Perform t-test for numerical variables
        for numerical_var in cols_numerical:
            t_stat, p_value = stats.ttest_ind(
                new_df[new_df["Cluster Labels"] == cluster_of_interest][numerical_var],
                pd.concat(
                    [
                        new_df[new_df["Cluster Labels"] == i][numerical_var]
                        for i in rest_of_clusters
                    ]
                ),
            )

            if p_value <= 0.05:
                mean_value = new_df[new_df["Cluster Labels"] == cluster_of_interest][
                    numerical_var
                ].mean()
                results_df = pd.concat(
                    [
                        results_df,
                        pd.DataFrame(
                            {
                                "Variable": [numerical_var],
                                "P-Value": [p_value],
                                "Cluster": [cluster_of_interest],
                                "Mean/Mode": [mean_value],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

        # Perform chi-square test for categorical variables
        for categorical_var in cols_categorical:
            observed = pd.crosstab(new_df[categorical_var], new_df["Cluster Labels"])

            chi2_stat, p_value, _, _ = stats.chi2_contingency(observed)

            if p_value <= 0.05:
                mode_value = (
                    new_df[new_df["Cluster Labels"] == cluster_of_interest][
                        categorical_var
                    ]
                    .mode()
                    .iloc[0]
                )
                results_df = results_df.append(
                    {
                        "Variable": categorical_var,
                        "P-Value": p_value,
                        "Cluster": cluster_of_interest,
                        "Mean/Mode": mode_value,
                    },
                    ignore_index=True,
                )

    results_df.reset_index(drop=True, inplace=True)
    results_df_signif = results_df[results_df["P-Value"] <= p_value_threshold]

    return results_df_signif
