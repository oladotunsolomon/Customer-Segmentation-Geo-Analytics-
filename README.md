# Customer-Segmentation-Geo-Analytics-
This project applies **customer segmentation and geographic analytics** to support a data-driven **Customer Appreciation Promotion** strategy. By integrating transactional, customer, and geographic data, the analysis identifies high-value customers, churn risks, and country-level opportunities to maximize marketing ROI.

---

## Executive Summary
A comprehensive segmentation model was developed using **RFM (Recency, Frequency, Monetary) analysis**, **hierarchical clustering**, and **geographic aggregation**.

Customers were segmented into four actionable groups. The largest segment, **Core / Promising Customers** (65.8%), represents moderately engaged customers with strong growth potential. **Champions / VIPs** (10.2%) are high-spending, frequent, and recent buyers, while a small elite group, **Elite VIPs** (0.3%), contributes a disproportionate share of total revenue. **At Risk / Churning Customers** (24.0%) show long periods of inactivity and require re-engagement efforts.

Geographically, the **United Kingdom dominates** the customer base, accounting for 90.4% of customers and 82.0% of total revenue. Despite this dominance, international markets such as the **Netherlands and Australia** exhibit the highest average customer spend, while **Norway, Belgium, and Germany** demonstrate the strongest customer engagement based on RFM scores.

---

## Key Findings

### Customer Segments
| Segment | Share | Description |
|------|------|-------------|
| Core / Promising Customers | 65.8% | Moderate engagement with growth potential |
| Champions / VIPs | 10.2% | High-spending, frequent, recent buyers |
| Elite VIPs | 0.3% | Extremely high-value micro-segment |
| At Risk / Churning | 24.0% | Inactive customers at high churn risk |

### Geographic Insights
- United Kingdom:  
  - 90.4% of customers  
  - 82.0% of total revenue  
- Highest average spend per customer:  
  - Netherlands, Australia  
- Highest engagement (RFM):  
  - Norway, Belgium, Germany  
- Top 8 countries generate **95% of total revenue**

---

## Data Understanding & Preparation

### Business Objective
Understand customer behavior and geographic distribution to design a targeted and effective customer appreciation promotion.

### Data Quality Issues Identified
- Missing Customer IDs (24.9% of transactions)
- Cancelled orders and invalid transaction values
- Duplicate customer records
- Inconsistent country naming

### Data Cleaning Actions
- Removed transactions without Customer IDs
- Filtered cancelled and invalid transactions
- Standardized country names (e.g., *EIRE → Ireland*)
- Added missing geographic coordinates
- Resolved duplicate customer records

### Final Dataset
- 397,884 transactions  
- 4,338 unique customers  

---

## RFM Analysis

RFM analysis was conducted using June 1, 2012, as the reference date for Recency.

### Summary Statistics
| Metric | Mean | Median | Max |
|------|------|--------|-----|
| Recency (days) | 266 | 100 | 547 |
| Frequency | 4.3 | 2 | 209 |
| Monetary (£) | £2,054 | £674 | £280,206 |

### Insight
Although **Champions** are not the largest group, they generate **64.75% of total revenue**, confirming that a small portion of customers drives business performance.

---

## Hierarchical Cluster Analysis

### Methodology
- Variables: Recency, Frequency, Monetary
- Standardized features
- Ward’s linkage with Euclidean distance
- Optimal solution: **4 clusters**

### Cluster Profiles
| Cluster | Name | Size | Avg Monetary (£) | Description |
|------|------|------|------------------|-------------|
| 1 | Elite VIPs | 15 (0.3%) | 111,916 | Extremely frequent and high-spending |
| 4 | Loyal Champions | 431 (9.9%) | 8,154 | Core high-value customers |
| 3 | Core Customers | 2,853 (65.8%) | 1,128 | Moderate engagement |
| 2 | At Risk / Lapsed | 1,039 (24.0%) | 483 | Long inactive |

---

## Geographic Analysis

- UK is the primary revenue and customer hub
- Revenue is highly concentrated geographically
- Several smaller countries demonstrate strong high-value or high-engagement customer profiles

---

## Geographically Weighted Regression (GWR)

The assignment specified a GWR analysis to examine geographic variation in sales drivers.  
However, the dataset did not contain required predictors (e.g., income, loyalty years), preventing implementation.

### Conceptual Value
GWR would allow:
- Country-specific modeling of sales drivers
- Improved marketing budget allocation
- Tailored geographic strategies

---

## Final Recommendations

### By Customer Segment
- **Champions & Elite VIPs:** Retention, loyalty rewards, exclusive offers
- **Core Customers:** Incentives to increase purchase frequency and value
- **At Risk Customers:** Targeted win-back and reactivation campaigns

### By Geography
- **United Kingdom:** Full segment-differentiated campaign
- **Germany & France:** Focused retention and growth efforts
- **Netherlands, Australia, Norway:** Pilot campaigns targeting high-spending customers

```python
# Customer Segmentation & Geo-Analytics
# Extracted from: Customer Segementation.ipynb


# =========================
# Cell 1
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

import warnings
warnings.filterwarnings('ignore')


# =========================
# Cell 2
# =========================
# ============================================================
# 1) Load data (adjust file paths as needed)
# ============================================================

# Replace these paths with your actual file locations
transactions_path = "transactions.csv"
customers_path = "customers.csv"
country_centroids_path = "country_centroids.csv"

transactions = pd.read_csv(transactions_path, encoding="ISO-8859-1")
customers = pd.read_csv(customers_path, encoding="ISO-8859-1")
countries = pd.read_csv(country_centroids_path, encoding="ISO-8859-1")

print(f"Transactions shape: {transactions.shape}")
print(f"Customers shape: {customers.shape}")
print(f"Country centroids shape: {countries.shape}")

print(f"\nAll rows:")
print(countries.head(20))


# =========================
# Cell 3
# =========================
# ============================================================
# 2) Data Quality Checks
# ============================================================

print("\nMissing CustomerID in transactions:", transactions['CustomerID'].isna().mean())

# Identify cancelled orders: InvoiceNo starts with 'C'
cancelled = transactions['InvoiceNo'].astype(str).str.startswith('C')
print("Cancelled transactions:", cancelled.sum())

# Negative quantities or non-positive prices
invalid_qty = transactions['Quantity'] <= 0
invalid_price = transactions['UnitPrice'] <= 0
print("Invalid qty count:", invalid_qty.sum())
print("Invalid price count:", invalid_price.sum())

# Check country inconsistencies
print("\nUnique countries in customers:", customers['Country'].nunique())
print("Unique countries in transactions:", transactions['Country'].nunique())
print("Unique countries in centroids:", countries['Country'].nunique())

print("\nCountries in transactions not in centroids:")
missing_countries = set(transactions['Country'].unique()) - set(countries['Country'].unique())
print(missing_countries)

print("\nCountries in customers not in centroids:")
missing_countries2 = set(customers['Country'].unique()) - set(countries['Country'].unique())
print(missing_countries2)


# =========================
# Cell 4
# =========================
# ============================================================
# 3) Data Cleaning
# ============================================================

# Drop missing CustomerID
transactions_clean = transactions.dropna(subset=['CustomerID'])

# Remove cancelled orders & invalid qty/price
transactions_clean = transactions_clean[~transactions_clean['InvoiceNo'].astype(str).str.startswith('C')]
transactions_clean = transactions_clean[(transactions_clean['Quantity'] > 0) & (transactions_clean['UnitPrice'] > 0)]

# Standardize country names
transactions_clean['Country'] = transactions_clean['Country'].replace({'EIRE': 'Ireland'})
customers['Country'] = customers['Country'].replace({'EIRE': 'Ireland'})

# Resolve duplicate CustomerIDs (keep first)
customers_clean = customers.drop_duplicates(subset=['CustomerID'], keep='first')

# Enrich missing geographic data
manual_additions = pd.DataFrame({
    "Country": ["Ireland", "European Community", "Unspecified"],
    "Latitude": [53.3498, 50.8503, 51.5074],
    "Longitude": [-6.2603, 4.3517, -0.1278]
})

countries_clean = pd.concat([countries, manual_additions], ignore_index=True)
countries_clean = countries_clean.drop_duplicates(subset=['Country'], keep='first')

print("Final transactions:", transactions_clean.shape)
print("Final customers:", customers_clean.shape)
print("Final centroids:", countries_clean.shape)


# =========================
# Cell 5
# =========================
# ============================================================
# 4) Build RFM Table
# ============================================================

transactions_clean['InvoiceDate'] = pd.to_datetime(transactions_clean['InvoiceDate'])

# Reference date (Recency anchor)
reference_date = pd.Timestamp("2012-06-01")

# Create TotalPrice
transactions_clean['TotalPrice'] = transactions_clean['Quantity'] * transactions_clean['UnitPrice']

rfm = transactions_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,   # Recency
    'InvoiceNo': 'nunique',                                     # Frequency
    'TotalPrice': 'sum'                                         # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print(rfm.describe())


# =========================
# Cell 6
# =========================
# ============================================================
# 5) RFM Scoring (1-5)
# ============================================================

rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])

rfm['RFM_Score'] = rfm['R_Score'].astype(int) + rfm['F_Score'].astype(int) + rfm['M_Score'].astype(int)
print(rfm.head())


# =========================
# Cell 7
# =========================
# ============================================================
# 6) Hierarchical Clustering (Ward + Euclidean)
# ============================================================

features = rfm[['Recency', 'Frequency', 'Monetary']]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Linkage matrix
Z = linkage(features_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='lastp', p=30)
plt.title("Dendrogram (Ward Linkage)")
plt.xlabel("Cluster size")
plt.ylabel("Distance")
plt.show()

# Choose number of clusters = 4
cluster_model = AgglomerativeClustering(n_clusters=4, linkage='ward')
rfm['Cluster'] = cluster_model.fit_predict(features_scaled)

rfm['Cluster'].value_counts()


# =========================
# Cell 8
# =========================
# ============================================================
# 7) Cluster Profiling
# ============================================================

cluster_profile = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
cluster_size = rfm['Cluster'].value_counts().sort_index()

print(cluster_profile)
print("\nCluster Sizes:\n", cluster_size)

# Optional: map cluster names manually after inspecting profile
cluster_names = {
    0: "Core Customers",
    1: "At Risk / Lapsed",
    2: "Loyal Champions",
    3: "Elite VIPs"
}

rfm['ClusterName'] = rfm['Cluster'].map(cluster_names)
print(rfm[['CustomerID','Cluster','ClusterName']].head())


# =========================
# Cell 9
# ============================
# 8) Country-Level Geographic Analysis
# ============================================================

# Merge customer country into RFM
rfm_geo = rfm.merge(customers_clean[['CustomerID','Country']], on='CustomerID', how='left')

# Merge in lat/long
rfm_geo = rfm_geo.merge(countries_clean, on='Country', how='left')

# Country summary
country_summary = rfm_geo.groupby('Country').agg(
    Customers=('CustomerID', 'count'),
    TotalRevenue=('Monetary', 'sum'),
    AvgRevenue=('Monetary', 'mean'),
    AvgRFM=('RFM_Score', 'mean')
).sort_values('TotalRevenue', ascending=False)

country_summary['RevenueShare'] = country_summary['TotalRevenue'] / country_summary['TotalRevenue'].sum()

print(country_summary.head(15))

