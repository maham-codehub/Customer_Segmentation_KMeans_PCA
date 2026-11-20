"""
Customer Segmentation using K-Means and PCA
Hunarmand Punjab - Artificial Intelligence Final Project (Q3)

The included sample_transactions.csv is synthetic demonstration data.
For a real-world submission, replace it with a genuine transaction dataset
having the required columns described in README.md.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "sample_transactions.csv"  # Change to Online Retail.xlsx for the real UCI dataset
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
K_MIN = 2
K_MAX = 8
FINAL_K = 4  # Review elbow_plot.png and change this if another K is more suitable.


def load_data(file_path):
    """Load transaction-level customer data from CSV."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Place a transaction CSV in the data folder or update DATA_FILE in main.py."
        )
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(file_path)
    else:
        data = pd.read_csv(file_path)
    required = {"InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    print(f"Loaded {len(data):,} transaction rows.")
    return data


def clean_transactions(data):
    """Clean missing, invalid and infinite transaction values."""
    data = data.copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # CustomerID is essential for grouping transactions by customer.
    data.dropna(subset=["CustomerID", "InvoiceNo", "InvoiceDate"], inplace=True)
    data["CustomerID"] = data["CustomerID"].astype(str).str.replace(r"\.0$", "", regex=True)

    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["UnitPrice"] = pd.to_numeric(data["UnitPrice"], errors="coerce")
    data.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice"], inplace=True)

    # Remove cancelled invoices and non-positive quantities/prices.
    data["InvoiceNo"] = data["InvoiceNo"].astype(str)
    data = data[~data["InvoiceNo"].str.startswith("C")]
    data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]

    # Fill optional text fields rather than dropping useful transactions.
    if "Description" in data.columns:
        data["Description"] = data["Description"].fillna("Unknown product")

    data["TotalAmount"] = data["Quantity"] * data["UnitPrice"]
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=["TotalAmount"], inplace=True)
    print(f"Rows after cleaning: {len(data):,}")
    return data


def create_customer_features(data):
    """Convert transactions into one RFM-style row per customer."""
    reference_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)

    customer_data = data.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda dates: (reference_date - dates.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum"),
        TotalQuantity=("Quantity", "sum"),
        AverageOrderValue=("TotalAmount", "mean"),
    ).reset_index()

    feature_columns = [
        "Recency", "Frequency", "Monetary",
        "TotalQuantity", "AverageOrderValue"
    ]
    customer_data[feature_columns] = customer_data[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    for column in feature_columns:
        customer_data[column] = customer_data[column].fillna(
            customer_data[column].median()
        )

    customer_data = customer_data[
        np.isfinite(customer_data[feature_columns]).all(axis=1)
    ].copy()

    # K-Means works best with non-negative values; these engineered features
    # should be non-negative after transaction cleaning.
    return customer_data, feature_columns


def run_elbow_method(scaled_features):
    """Calculate and save inertia values for a range of K values."""
    k_values = list(range(K_MIN, K_MAX + 1))
    inertias = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        model.fit(scaled_features)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.title("Elbow Method for Selecting K")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.xticks(k_values)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "elbow_plot.png", dpi=300)
    plt.close()
    return k_values, inertias


def interpret_clusters(customer_data):
    """Create simple, data-driven business descriptions for each cluster."""
    summary = customer_data.groupby("Cluster").agg(
        Customers=("CustomerID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Avg_Order_Value=("AverageOrderValue", "mean"),
    ).round(2)

    overall_frequency = customer_data["Frequency"].median()
    overall_monetary = customer_data["Monetary"].median()
    overall_recency = customer_data["Recency"].median()

    descriptions = []
    for cluster_id, row in summary.iterrows():
        if row["Avg_Frequency"] >= overall_frequency and row["Avg_Monetary"] >= overall_monetary:
            label = "High-value loyal customers"
            meaning = "Purchase frequently and spend more; consider loyalty rewards and retention offers."
        elif row["Avg_Recency"] <= overall_recency and row["Avg_Frequency"] < overall_frequency:
            label = "Recent / new customers"
            meaning = "Purchased relatively recently but less often; encourage a second purchase."
        elif row["Avg_Recency"] > overall_recency and row["Avg_Frequency"] >= overall_frequency:
            label = "At-risk regular customers"
            meaning = "Have a purchase history but have not purchased recently; consider a re-engagement campaign."
        elif row["Avg_Monetary"] >= overall_monetary:
            label = "High-spending occasional customers"
            meaning = "Spend well when they buy but purchase less frequently; use targeted reminders."
        else:
            label = "Low-engagement customers"
            meaning = "Show lower purchasing activity; test low-cost offers before investing heavily."
        descriptions.append({
            "Cluster": cluster_id,
            "Suggested_Segment_Name": label,
            "Business_Interpretation": meaning
        })

    interpretation = pd.DataFrame(descriptions)
    summary = summary.reset_index().merge(interpretation, on="Cluster", how="left")
    return summary


def main():
    print("\\nCUSTOMER SEGMENTATION: K-MEANS + PCA")
    print("=" * 45)

    transactions = load_data(DATA_FILE)
    cleaned = clean_transactions(transactions)

    customer_data, feature_columns = create_customer_features(cleaned)
    if len(customer_data) < K_MAX:
        raise ValueError(
            f"Need at least {K_MAX} customers to test K={K_MAX}; "
            f"only {len(customer_data)} remain."
        )

    print(f"Customers available for clustering: {len(customer_data):,}")
    print("\\nCustomer feature preview:")
    print(customer_data.head())

    # Mandatory feature scaling before K-Means.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_data[feature_columns])

    # Elbow method: inspect the saved plot to choose a suitable K.
    k_values, inertias = run_elbow_method(scaled_features)
    print("\\nElbow results:")
    print(pd.DataFrame({"K": k_values, "Inertia": np.round(inertias, 2)}))
    print(f"Using FINAL_K={FINAL_K}. Review outputs/elbow_plot.png and adjust if needed.")

    kmeans = KMeans(n_clusters=FINAL_K, random_state=RANDOM_STATE, n_init=10)
    customer_data["Cluster"] = kmeans.fit_predict(scaled_features)

    if customer_data["Cluster"].nunique() > 1:
        score = silhouette_score(scaled_features, customer_data["Cluster"])
        print(f"Silhouette score (additional diagnostic): {score:.3f}")

    # PCA reduces the scaled feature space to two components for visualization.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(scaled_features)
    customer_data["PCA1"] = components[:, 0]
    customer_data["PCA2"] = components[:, 1]
    explained = pca.explained_variance_ratio_.sum()

    # Save customer-level results.
    customer_data.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)

    # 2D PCA scatter plot.
    plt.figure(figsize=(9, 6))
    for cluster_id in sorted(customer_data["Cluster"].unique()):
        subset = customer_data[customer_data["Cluster"] == cluster_id]
        plt.scatter(
            subset["PCA1"], subset["PCA2"],
            label=f"Cluster {cluster_id}", alpha=0.75, s=45
        )
    plt.title("Customer Segments Visualized with PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_clusters.png", dpi=300)
    plt.close()

    # Cluster size chart.
    counts = customer_data["Cluster"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Number of Customers in Each Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of customers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_sizes.png", dpi=300)
    plt.close()

    summary = interpret_clusters(customer_data)
    summary.to_csv(OUTPUT_DIR / "cluster_interpretation.csv", index=False)

    with open(OUTPUT_DIR / "project_summary.txt", "w", encoding="utf-8") as file:
        file.write("CUSTOMER SEGMENTATION PROJECT SUMMARY\\n")
        file.write("=" * 42 + "\\n")
        file.write(f"Transaction rows after cleaning: {len(cleaned)}\\n")
        file.write(f"Customers clustered: {len(customer_data)}\\n")
        file.write(f"Number of clusters (K): {FINAL_K}\\n")
        file.write(f"PCA variance explained by 2 components: {explained:.2%}\\n\\n")
        file.write("CLUSTER INTERPRETATION\\n")
        file.write(summary.to_string(index=False))
        file.write("\\n\\nNote: Suggested segment names are heuristic business interpretations. "
                   "Validate them against the actual dataset and business context.\\n")

    print("\\nCLUSTER INTERPRETATION:")
    print(summary.to_string(index=False))
    print(f"\\nPCA variance explained by two components: {explained:.2%}")
    print("\\nCompleted successfully. Check the outputs folder for CSV files and plots.")


if __name__ == "__main__":
    main()
