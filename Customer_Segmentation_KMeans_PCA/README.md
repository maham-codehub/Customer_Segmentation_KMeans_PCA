# Customer Segmentation Using K-Means + PCA
**Hunarmand Punjab — Artificial Intelligence Final Project (Question 3)**

## 1. Project objective
Group customers into meaningful segments based on their purchasing behaviour. The project uses transaction-level data, creates customer-level purchasing features, applies K-Means clustering, and uses PCA to visualize the clusters in two dimensions.

> **Dataset note:** `data/sample_transactions.csv` is a synthetic demonstration dataset created for learning and immediate testing. It is not real customer data. For a stronger real-world submission, use the official UCI Online Retail dataset (541,909 transaction rows; 8 fields; CC BY 4.0) and cite its source in your report. Download `Online Retail.xlsx` from https://archive.ics.uci.edu/dataset/352/online-retail, place it in `data/`, then change `DATA_FILE` in `src/main.py` to `PROJECT_DIR / "data" / "Online Retail.xlsx"`. The loader automatically reads `.xlsx` files using pandas.

## 2. Assignment requirements covered
| Hunarmand Punjab requirement | Where it is completed |
|---|---|
| Load customer transaction dataset | `src/main.py` — `load_data()` |
| Handle missing and infinite values | `clean_transactions()` and `create_customer_features()` |
| Feature scaling (mandatory) | `StandardScaler` in `main()` |
| Apply K-Means clustering | `KMeans` in `main()` |
| Elbow Method to select K | `run_elbow_method()`; output `elbow_plot.png` |
| Reduce dimensions using PCA | `PCA(n_components=2)` |
| Visualize clusters in 2D | `outputs/pca_clusters.png` |
| Explain business meaning of each cluster | `interpret_clusters()` and `cluster_interpretation.csv` |

## 3. Tools and libraries
- Python
- Pandas — loading, cleaning, grouping and preparing data
- NumPy — numerical operations and handling infinite values
- Matplotlib — charts and visualizations
- Scikit-learn — StandardScaler, KMeans, PCA and silhouette score

## 4. Project structure
```text
Customer_Segmentation_KMeans_PCA/
│
├── data/
│   └── sample_transactions.csv
│
├── src/
│   └── main.py
│
├── outputs/                 # generated when the script runs
│   ├── elbow_plot.png
│   ├── pca_clusters.png
│   ├── cluster_sizes.png
│   ├── customer_segments.csv
│   ├── cluster_interpretation.csv
│   └── project_summary.txt
│
├── requirements.txt
└── README.md
```

## 5. Run in Windows VS Code
Open the `Customer_Segmentation_KMeans_PCA` folder in VS Code.

Open Terminal → New Terminal and run:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\main.py
```

If activation works, run:

```powershell
python -m pip install -r requirements.txt
python src\main.py
```

## 6. Dataset format for a real transaction CSV
To use another transaction dataset, update `DATA_FILE` near the top of `src/main.py`. The CSV or Excel workbook must contain these columns:
- `InvoiceNo` — order/invoice identifier
- `Quantity` — quantity purchased
- `InvoiceDate` — date/time of purchase
- `UnitPrice` — price per item
- `CustomerID` — customer identifier

`Description`, `StockCode` and `Country` are optional. The included sample has all of these columns.

The script builds customer-level features:
- **Recency:** days since the customer's most recent purchase
- **Frequency:** number of distinct invoices/orders
- **Monetary:** total amount spent
- **TotalQuantity:** total units purchased
- **AverageOrderValue:** average transaction-line value (a simplified proxy)

For a real dataset, check its currency, date format, cancellations, returns and column meanings before interpreting results.

## 7. How the algorithms work
### K-Means
K-Means is an unsupervised machine-learning algorithm. It groups customers according to similarity in their scaled purchasing features. It assigns each customer to one cluster and updates cluster centres until the grouping stabilizes.

### Elbow Method
The script trains K-Means for K values from 2 to 8 and plots inertia. Inertia measures the within-cluster sum of squared distances. Look for the point where the reduction in inertia begins to slow (the “elbow”). The example script uses K=4 as an initial setting; inspect `elbow_plot.png` and change `FINAL_K` in `src/main.py` if the elbow suggests another value. Do not claim that K=4 is universally best.

### StandardScaler
Features such as Monetary and Recency use different numeric scales. StandardScaler transforms each feature to a comparable scale (mean 0 and standard deviation 1), preventing large-valued features from dominating distance calculations.

### PCA
Principal Component Analysis converts the scaled features into new components. The script keeps two components so the clusters can be plotted in 2D. PCA is used for visualization here; K-Means is trained on the full scaled feature set, not only the two PCA components.

## 8. Output files
- `elbow_plot.png` — helps select K.
- `pca_clusters.png` — each point represents a customer; colour represents its cluster.
- `cluster_sizes.png` — number of customers in each cluster.
- `customer_segments.csv` — customer features, cluster number and PCA coordinates.
- `cluster_interpretation.csv` — average cluster behaviour and suggested business meaning.
- `project_summary.txt` — short run summary.

## 9. Important interpretation note
Cluster numbers (0, 1, 2, 3) are identifiers, not rankings. Suggested segment names are rule-based starting interpretations. Review each cluster's average Recency, Frequency and Monetary values before finalizing business recommendations. Results depend on the dataset and chosen K.

## 10. Suggested report structure
1. Title page
2. Introduction
3. Problem statement
4. Objectives
5. Dataset description and source
6. Data cleaning and feature engineering
7. Methodology: StandardScaler, Elbow Method, K-Means and PCA
8. Results and visualizations
9. Cluster-wise business interpretation
10. Limitations
11. Conclusion
12. References

## 11. Limitations
- The bundled dataset is synthetic and only demonstrates the workflow.
- K-Means is sensitive to feature scaling, outliers and the selected K.
- PCA plots compress information; some separation may be lost in two dimensions.
- The segment labels are suggested interpretations, not ground-truth customer categories.
