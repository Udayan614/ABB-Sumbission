#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install catboost')


# In[6]:


get_ipython().system('pip install xgboost')


# ===============================
# STEP 0: IMPORTS & CONFIGURATION
# ===============================

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

pd.set_option("display.max_columns", None)
sns.set(style="whitegrid")


# ===============================
# STEP 1: LOAD DATA
# ===============================

# In[2]:


train = pd.read_csv("train_av.csv")
test = pd.read_csv("test_av.csv")

train["source"] = "train"
test["source"] = "test"

data = pd.concat([train, test], ignore_index=True)

# Why

# Combine train & test for consistent preprocessing

# Still safe because target is untouched


# ===============================
# STEP 2: BASIC DATA UNDERSTANDING (EDA)
# ===============================

# In[3]:


data.info()
data.describe()

# Check:

# Missing values

# Data types

# Numerical ranges


# =============================== STEP 2.1: CATEGORICAL CARDINALITY CHECK ===============================

# In[4]:


print("\nCategorical Cardinality Check\n")

categorical_cols_check = [
    "Item_Identifier",
    "Outlet_Identifier",
    "Item_Type",
    "Outlet_Type",
    "Outlet_Size",
    "Outlet_Location_Type"
]

cardinality_df = pd.DataFrame({
    "Feature": categorical_cols_check,
    "Unique_Count": [data[col].nunique() for col in categorical_cols_check],
    "Total_Rows": len(data)
})

cardinality_df["Cardinality_%"] = (
    cardinality_df["Unique_Count"] / cardinality_df["Total_Rows"] * 100
).round(2)

cardinality_df = cardinality_df.sort_values(
    by="Unique_Count", ascending=False
)

print(cardinality_df)


# ===============================
# STEP 3: TARGET VARIABLE ANALYSIS
# ===============================

# In[5]:


plt.figure(figsize=(6,4))
sns.histplot(train["Item_Outlet_Sales"], bins=50)
plt.title("Raw Sales Distribution")
plt.show()

# Observation

# Highly right-skewed

# RMSE will be dominated by large values

# Log transformation (industry standard)


# In[6]:


train["log_sales"] = np.log1p(train["Item_Outlet_Sales"])


# In[7]:


train["log_sales"].head()


# =============================== STEP 3.1: FEATURE RELATIONSHIPS (EDA) ===============================

# In[8]:


print("\n Correlation Matrix (Numerical Features)\n")

# Select only numerical columns
numeric_cols = train.select_dtypes(include=[np.number]).columns

# Remove target raw if you want separate view
corr_matrix = train[numeric_cols].corr()

plt.figure(figsize=(10,7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix (Numerical Features)")
plt.show()


# --------------------------------------
# Boxplot: Sales vs Outlet_Type
# --------------------------------------

plt.figure(figsize=(8,5))
sns.boxplot(
    x="Outlet_Type",
    y="Item_Outlet_Sales",
    data=train
)
plt.xticks(rotation=45)
plt.title("Sales Distribution by Outlet Type")
plt.show()


# --------------------------------------
# Boxplot: Sales vs Item_Type
# --------------------------------------

plt.figure(figsize=(10,6))
sns.boxplot(
    x="Item_Type",
    y="Item_Outlet_Sales",
    data=train
)
plt.xticks(rotation=90)
plt.title("Sales Distribution by Item Type")
plt.show()


# ===============================
# STEP 4: MISSING VALUE TREATMENT
# ===============================

# In[9]:


#4.1 Item_Weight (product-specific)
item_weight_median = data.groupby("Item_Identifier")["Item_Weight"].median()

data["Item_Weight"] = data.apply(
    lambda x: item_weight_median[x["Item_Identifier"]]
    if pd.isnull(x["Item_Weight"]) else x["Item_Weight"],
    axis=1
)

data["Item_Weight"].fillna(data["Item_Weight"].median(), inplace=True)


# Why

# Weight is intrinsic to product

# Preserves physical meaning


# In[10]:


# 4.2 Outlet_Size (store attribute)

outlet_size_mode = data.groupby("Outlet_Type")["Outlet_Size"].agg(
    lambda x: x.mode()[0]
)

data["Outlet_Size"] = data.apply(
    lambda x: outlet_size_mode[x["Outlet_Type"]]
    if pd.isnull(x["Outlet_Size"]) else x["Outlet_Size"],
    axis=1
)


# In[11]:


#4.3 Item_Visibility = 0 (data error)
visibility_median = data.groupby("Item_Identifier")["Item_Visibility"].median()

data.loc[data["Item_Visibility"] == 0, "Item_Visibility"] = \
    data.loc[data["Item_Visibility"] == 0, "Item_Identifier"].map(visibility_median)


# ===============================
# STEP 5: FEATURE ENGINEERING
# ===============================

# In[12]:


#5.1 Product category

# First 2 characters of item code define category:

# FD → Food

# DR → Drinks

# NC → Non-Consumable
data["Item_Category"] = data["Item_Identifier"].str[:2]



#5.2 Outlet age

# Dataset year = 2013

# Converts year into age of outlet

# “Older outlets sell more”
# better than:

# “Lower year value sells more”

data["Outlet_Age"] = 2013 - data["Outlet_Establishment_Year"]

#5.3 Price features

#MRP is right-skewed

# Log:

# Compresses large values

# Makes relationship more linear

#log1p Handles zero safely:
    
data["Item_MRP_log"] = np.log1p(data["Item_MRP"])

# Divides MRP into quartiles

# Each bin has ~equal number of observations

# Captures non-linear pricing effects

# Some outlets may sell:

# Premium products better than low-priced ones

data["Item_MRP_bin"] = pd.qcut(
    data["Item_MRP"], q=4, labels=["Low", "Medium", "High", "Premium"]
)

#5.4 Visibility ratio
#Computes average visibility per product

#Visibility Ratio=Current Visibility/Typical Visibility of Item

avg_visibility = data.groupby("Item_Identifier")["Item_Visibility"].mean()
data["Item_Visibility_Ratio"] = data["Item_Visibility"] / data["Item_Identifier"].map(avg_visibility)


# ===============================
# STEP 6: ENCODING CATEGORICAL VARIABLES
# ===============================

# In[13]:


categorical_cols = [
    "Item_Identifier", "Item_Type", "Item_Category",
    "Outlet_Identifier", "Outlet_Size",
    "Outlet_Location_Type", "Outlet_Type",
    "Item_MRP_bin"
]

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    
    
# Why

# Label encoding works best for tree models

# Avoids one-hot explosion


# ===============================
# STEP 7: SPLIT BACK TRAIN & TEST
# ===============================

# In[14]:


train_final = data[data["source"] == "train"].copy()
test_final = data[data["source"] == "test"].copy()

train_final["log_sales"] = np.log1p(train_final["Item_Outlet_Sales"])


# ===============================
# STEP 8: FEATURE LIST (FINAL)
# ===============================

# In[15]:


features = [
    "Item_Weight",
    "Item_Visibility",
    "Item_Visibility_Ratio",
    "Item_MRP",
    "Item_MRP_log",
    "Item_MRP_bin",
    "Outlet_Age",
    "Item_Category",
    "Item_Type",
    "Outlet_Identifier",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type"
]


# ===============================
# STEP 9: K-FOLD WITH LEAKAGE-SAFE AGGREGATION
# ===============================

# In[16]:


N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_cat = np.zeros(len(train_final))

test_preds_cat = np.zeros(len(test_final))


# ===============================
# STEP 10: CROSS-VALIDATION LOOP
# ===============================

# In[17]:


for fold, (tr_idx, val_idx) in enumerate(kf.split(train_final)):
    print(f"\nFold {fold + 1}")

    tr = train_final.iloc[tr_idx].copy()
    val = train_final.iloc[val_idx].copy()

    # ---- Leakage-safe aggregation ----
    item_avg = tr.groupby("Item_Identifier")["Item_Outlet_Sales"].mean()
    outlet_avg = tr.groupby("Outlet_Identifier")["Item_Outlet_Sales"].mean()

    for df in [tr, val, test_final]:
        df["Item_Avg_Sales"] = df["Item_Identifier"].map(item_avg)
        df["Outlet_Avg_Sales"] = df["Outlet_Identifier"].map(outlet_avg)

        df["Item_Avg_Sales"].fillna(tr["Item_Avg_Sales"].median(), inplace=True)
        df["Outlet_Avg_Sales"].fillna(tr["Outlet_Avg_Sales"].median(), inplace=True)

    fold_features = features + ["Item_Avg_Sales", "Outlet_Avg_Sales"]

    X_tr, y_tr = tr[fold_features], tr["log_sales"]
    X_val, y_val = val[fold_features], val["log_sales"]

    # ---- CatBoost ----
    cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        #random_seed=42,
        verbose=False
    )

    cat.fit(X_tr, y_tr)
    oof_cat[val_idx] = cat.predict(X_val)
    test_preds_cat += cat.predict(test_final[fold_features]) / N_FOLDS


# In[18]:


##predicted valuses - log scale
test_preds_cat


# ===============================
# STEP 11: MODEL Evaluation Matrix (RMSE) 
# ===============================

# In[19]:


rmse_cat = mean_squared_error(train_final["log_sales"], oof_cat, squared=False)

print(f"CatBoost CV RMSE (log): {rmse_cat:.4f}")


# In[20]:


#Step 1: Back-transform predictions
oof_cat_raw = np.expm1(oof_cat)


#Step 2: Get true raw target
y_true_raw = train_final["Item_Outlet_Sales"].values

#Step 3: Compute RMSE on original scale
rmse_cat_raw = mean_squared_error(
    y_true_raw,
    oof_cat_raw,
    squared=False
)

print(f"CatBoost CV RMSE (raw): {rmse_cat_raw:,.2f}")


# ===============================
# STEP 12: FEATURE IMPORTANCE (CatBoost)
# ===============================

# In[21]:


feat_imp = pd.DataFrame({
    "feature": fold_features,
    "importance": cat.get_feature_importance()
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x="importance", y="feature", data=feat_imp.head(10))
plt.title("Top 10 Feature Importances (CatBoost)")
plt.show()


# ===============================
# STEP 13: FINAL PREDICTION & SUBMISSION
# ===============================

# In[22]:


final_preds = np.expm1(test_preds_cat)

submission = pd.DataFrame({
    "Item_Identifier": test["Item_Identifier"],
    "Outlet_Identifier": test["Outlet_Identifier"],
    "Item_Outlet_Sales": final_preds
})

submission.to_csv("final_submission.csv", index=False)


# In[ ]:




