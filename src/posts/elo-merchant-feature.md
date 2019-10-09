---
title: Detailed feature engineering of kaggle Elo merchant competition - Top 3% solution 🔥
date: '2019-09-02'
tags:
  - tabular-data
  - feature-engineering
  - deep-learning
  - data-science
  - blog
  - kaggle
---

- [Elo Merchant Competition](#heading-elo-merchant-competition)
    - [What files do we need?](#heading-what-files-do-we-need)
    - [What should we expect the data format to be?](#heading-what-should-we-expect-the-data-format-to-be)
    - [What are we predicting?](#heading-what-are-we-predicting)
    - [File descriptions](#heading-file-descriptions)
    - [Data fields](#heading-data-fields)
  - [Understanding the data](#heading-understanding-the-data)
- [Feature engineering](#heading-feature:-engineering)
  - [Feature: Add date features meta info](#heading-feature:-add-date-features-meta-info)
  - [Feature: Add derived date features](#heading-feature:-add-derived-date-features)
  - [Feature: Reverse engineering purchase amount column](#heading-feature:-reverse-engineering-purchase-amount-column)
  - [Feature: Aggregates grouped by month & card_id](#heading-feature:-aggregates-grouped-by-month-and-card_id)
  - [Feature: Time between successive transactions](#heading-feature:-time-between-successive-transactions)
  - [Feature: Holiday features](#heading-feature:-holiday-features)
  - [Feature: Latest & First 5 feature aggregates](#heading-feature:-latest-and-first-5-feature-aggregates)
  - [Feature: Merchants features](#heading-feature:-merchants-features)
  - [Aggregate: Aggregate by card_id](#heading-aggregate:-aggregate-by-card_id)
  - [Feature: Add exta interpreted columns on aggregates](#heading-feature:-add-exta-interpreted-columns-on-aggregates)
  - [Aggregate: Aggregate on categories](#heading-aggregate:-aggregate-on-categories)
  - [Aggregate: Aggregate on month](#heading-aggregate:-aggregate-on-month)
  - [Feature: Reverse engineering observed date aka reference date](#heading-feature:-reverse-engineering-observed-date-aka-reference-date)
  - [Aggregates: Merge train & test with new & old transactions history](#heading-aggregates:-merge-train-and-test-with-new-and-old-transactions-history)
  - [Feature: Adding features based on observed_date](#heading-feature:-adding-features-based-on-observed_date)
  - [Feature: Add features based on old & new transactions](#heading-feature:-add-features-based-on-old-and-new-transactions)
  - [Feature: Redo some date features with observed time](#heading-feature:-redo-some-date-features-with-observed-time)
  - [Feature: Mark the outliers](#heading-feature:-mark-the-outliers)
  - [Feature: Redo features based on new purchase amount](#heading-feature:-redo-features-based-on-new-purchase-amount)
- [Feature Selection](#heading-feature-selection)
- [Model training](#heading-model-training)
  - [Cross validation data set](#heading-cross-validation-data-set)
  - [LGBM model](#heading-lgbm-model)
  - [XGBM Model](#heading-xgbm-model)
- [Post processing](#heading-post-processing)
  - [Stacking the model predictions](#heading-stacking-the-model-predictions)
  - [Combining model without outliers data](#heading-combining-model-without-outliers-data)
- [Kaggle submission](#heading-kaggle-submission)


## Elo Merchant Competition

Elo is one of the largest payment brands in Brazil. It has built partnerships with merchants in order to offer promotions or discounts to cardholders. But do these promotions work for either the consumer or the merchant? Do customers enjoy their experience? Do merchants see repeat business? Personalization is key.

In this competition our challenge is to develop algorithms to identify and serve the most relevant opportunities to individuals, by uncovering signal in customer loyalty. Our input will help Elo reduce unwanted campaigns, to create the right experience for customers.

#### What files do we need?

All the data can be downloaded from [here](https://www.kaggle.com/c/elo-merchant-category-recommendation/data)
We will need, at a minimum, the train.csv and test.csv files. These contain the card_ids that we'll be using for training and prediction.

#### What should we expect the data format to be?

The data is formatted as follows:

`train.csv` and `test.csv` contain card_ids and information about the card itself - the first month the card was active, etc. `train.csv` also contains the target.

`historical_transactions.csv` and `new_merchant_transactions.csv` are designed to be joined with `train.csv`, `test.csv`, and `merchants.csv`. They contain information about transactions for each card, as described above.

`merchants.csv` can be joined with the transaction sets to provide additional merchant-level information.

#### What are we predicting?

We are predicting a loyalty score for each card_id represented in test.csv and sample_submission.csv.

#### File descriptions

`train.csv` - the training set

`test.csv` - the test set

`sample_submission.csv` - a sample submission file in the correct format - contains all card_ids we are expected to predict for.

`historical_transactions.csv` - up to 3 months' worth of historical transactions for each card_id

`merchants.csv` - additional information about all merchants / merchant_ids in the dataset.

`new_merchant_transactions.csv` - two months' worth of data for each card_id containing ALL purchases that card_id made at merchant_ids that were not visited in the historical data.

#### Data fields

Data field descriptions are provided in Data Dictionary.xlsx.

I downloaded the files to my local directory. Let's get started!

### Understanding the data

Though Elo mentioned that all data is simulated and fictitious, and is not real customer data, it was later hinted that data was highly transformed & modified using statistical techniques to make it hard for data leaks.

```python
PATH = '../data/elo/'
files = ['historical_transactions', 'new_merchant_transactions']
hist_trans, new_hist_trans = [pd.read_csv(f'{PATH}{c}.csv') for c in files]
```

Let's have a look at the transactions data available.

```python
hist_trans.head().T
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right; overflow:scroll;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>authorized_flag</th>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>card_id</th>
      <td>C_ID_4e6213e9bc</td>
      <td>C_ID_4e6213e9bc</td>
      <td>C_ID_4e6213e9bc</td>
      <td>C_ID_4e6213e9bc</td>
      <td>C_ID_4e6213e9bc</td>
    </tr>
    <tr>
      <th>city_id</th>
      <td>88</td>
      <td>88</td>
      <td>88</td>
      <td>88</td>
      <td>88</td>
    </tr>
    <tr>
      <th>category_1</th>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>installments</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>category_3</th>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>merchant_category_id</th>
      <td>80</td>
      <td>367</td>
      <td>80</td>
      <td>560</td>
      <td>80</td>
    </tr>
    <tr>
      <th>merchant_id</th>
      <td>M_ID_e020e9b302</td>
      <td>M_ID_86ec983688</td>
      <td>M_ID_979ed661fc</td>
      <td>M_ID_e6d5ae8ea6</td>
      <td>M_ID_e020e9b302</td>
    </tr>
    <tr>
      <th>month_lag</th>
      <td>-8</td>
      <td>-7</td>
      <td>-6</td>
      <td>-5</td>
      <td>-11</td>
    </tr>
    <tr>
      <th>purchase_amount</th>
      <td>-0.703331</td>
      <td>-0.733128</td>
      <td>-0.720386</td>
      <td>-0.735352</td>
      <td>-0.722865</td>
    </tr>
    <tr>
      <th>purchase_date</th>
      <td>2017-06-25 15:33:07</td>
      <td>2017-07-15 12:10:45</td>
      <td>2017-08-09 22:04:29</td>
      <td>2017-09-02 10:06:26</td>
      <td>2017-03-10 01:14:19</td>
    </tr>
    <tr>
      <th>category_2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>state_id</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>subsector_id</th>
      <td>37</td>
      <td>16</td>
      <td>37</td>
      <td>34</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>

```python
new_hist_trans.head().T
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>authorized_flag</th>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>card_id</th>
      <td>C_ID_415bb3a509</td>
      <td>C_ID_415bb3a509</td>
      <td>C_ID_415bb3a509</td>
      <td>C_ID_415bb3a509</td>
      <td>C_ID_ef55cf8d4b</td>
    </tr>
    <tr>
      <th>city_id</th>
      <td>107</td>
      <td>140</td>
      <td>330</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>category_1</th>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>installments</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>category_3</th>
      <td>B</td>
      <td>B</td>
      <td>B</td>
      <td>B</td>
      <td>B</td>
    </tr>
    <tr>
      <th>merchant_category_id</th>
      <td>307</td>
      <td>307</td>
      <td>507</td>
      <td>661</td>
      <td>166</td>
    </tr>
    <tr>
      <th>merchant_id</th>
      <td>M_ID_b0c793002c</td>
      <td>M_ID_88920c89e8</td>
      <td>M_ID_ad5237ef6b</td>
      <td>M_ID_9e84cda3b1</td>
      <td>M_ID_3c86fa3831</td>
    </tr>
    <tr>
      <th>month_lag</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>purchase_amount</th>
      <td>-0.557574</td>
      <td>-0.56958</td>
      <td>-0.551037</td>
      <td>-0.671925</td>
      <td>-0.659904</td>
    </tr>
    <tr>
      <th>purchase_date</th>
      <td>2018-03-11 14:57:36</td>
      <td>2018-03-19 18:53:37</td>
      <td>2018-04-26 14:08:44</td>
      <td>2018-03-07 09:43:21</td>
      <td>2018-03-22 21:07:53</td>
    </tr>
    <tr>
      <th>category_2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>state_id</th>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>subsector_id</th>
      <td>19</td>
      <td>19</td>
      <td>14</td>
      <td>8</td>
      <td>29</td>
    </tr>
  </tbody>
</table>
</div>

Let's see data dictionary to find out what each of these fields mean.

| train.csv          |                                                                                    |
| ------------------ | ---------------------------------------------------------------------------------- |
| card_id            | Unique card identifier                                                             |
| first_active_month | YYYY-MM', month of first purchase                                                  |
| feature_1          | Anonymized card categorical feature                                                |
| feature_2          | Anonymized card categorical feature                                                |
| feature_3          | Anonymized card categorical feature                                                |
| target             | Loyalty numerical score calculated 2 months after historical and evaluation period |

| historical_transactions.csv |                                                  |
| --------------------------- | ------------------------------------------------ |
| card_id                     | Card identifier                                  |
| month_lag                   | month lag to reference date                      |
| purchase_date               | Purchase date                                    |
| authorized_flag             | Y' if approved, 'N' if denied                    |
| category_3                  | anonymized category                              |
| installments                | number of installments of purchase               |
| category_1                  | anonymized category                              |
| merchant_category_id        | Merchant category identifier (anonymized )       |
| subsector_id                | Merchant category group identifier (anonymized ) |
| merchant_id                 | Merchant identifier (anonymized)                 |
| purchase_amount             | Normalized purchase amount                       |
| city_id                     | City identifier (anonymized )                    |
| state_id                    | State identifier (anonymized )                   |
| category_2                  | anonymized category                              |

| new_merchant_period.csv |                                                  |
| ----------------------- | ------------------------------------------------ |
| card_id                 | Card identifier                                  |
| month_lag               | month lag to reference date                      |
| purchase_date           | Purchase date                                    |
| authorized_flag         | Y' if approved, 'N' if denied                    |
| category_3              | anonymized category                              |
| installments            | number of installments of purchase               |
| category_1              | anonymized category                              |
| merchant_category_id    | Merchant category identifier (anonymized )       |
| subsector_id            | Merchant category group identifier (anonymized ) |
| merchant_id             | Merchant identifier (anonymized)                 |
| purchase_amount         | Normalized purchase amount                       |
| city_id                 | City identifier (anonymized )                    |
| state_id                | State identifier (anonymized )                   |
| category_2              | anonymized category                              |

| merchants.csv               |                                                                                                |
| --------------------------- | ---------------------------------------------------------------------------------------------- |
| merchant_id                 | Unique merchant identifier                                                                     |
| merchant_group_id           | Merchant group (anonymized )                                                                   |
| merchant_category_id        | Unique identifier for merchant category (anonymized )                                          |
| subsector_id                | Merchant category group (anonymized )                                                          |
| numerical_1                 | anonymized measure                                                                             |
| numerical_2                 | anonymized measure                                                                             |
| category_1                  | anonymized category                                                                            |
| most_recent_sales_range     | Range of revenue (monetary units) in last active month --> A > B > C > D > E                   |
| most_recent_purchases_range | Range of quantity of transactions in last active month --> A > B > C > D > E                   |
| avg_sales_lag3              | Monthly average of revenue in last 3 months divided by revenue in last active month            |
| avg_purchases_lag3          | Monthly average of transactions in last 3 months divided by transactions in last active month  |
| active_months_lag3          | Quantity of active months within last 3 months                                                 |
| avg_sales_lag6              | Monthly average of revenue in last 6 months divided by revenue in last active month            |
| avg_purchases_lag6          | Monthly average of transactions in last 6 months divided by transactions in last active month  |
| active_months_lag6          | Quantity of active months within last 6 months                                                 |
| avg_sales_lag12             | Monthly average of revenue in last 12 months divided by revenue in last active month           |
| avg_purchases_lag12         | Monthly average of transactions in last 12 months divided by transactions in last active month |
| active_months_lag12         | Quantity of active months within last 12 months                                                |
| category_4                  | anonymized category                                                                            |
| city_id                     | City identifier (anonymized )                                                                  |
| state_id                    | State identifier (anonymized )                                                                 |
| category_2                  | anonymized category                                                                            |

Let's start with transactions data. We will come back to merchants & cards data later. We have ~1.9m new & ~2.9m historical transactions data. This can be joined with cards & merchants data on `card_id` and `merchant_id` respectively. We will join all the tables after doing feature engineering on each of them separately.

```python
new_hist_trans.shape, hist_trans.shape
    ((1963031, 14), (29112361, 14))
```

Data preprocessing consists of:

1. Treatment of missing values
2. Mapping of categorical columns

Starting with treatment of missing values for each of the tables in the transactions data:

```python
hist_trans.isnull().sum()/len(new_hist_trans)
```

```text
    authorized_flag         0.000000
    card_id                 0.000000
    city_id                 0.000000
    category_1              0.000000
    installments            0.000000
    category_3              0.090757
    merchant_category_id    0.000000
    merchant_id             0.070544
    month_lag               0.000000
    purchase_amount         0.000000
    purchase_date           0.000000
    category_2              1.351412
    state_id                0.000000
    subsector_id            0.000000
    dtype: float64
```

```python
new_hist_trans.isnull().sum()/len(new_hist_trans)
```

```text
    authorized_flag         0.000000
    card_id                 0.000000
    city_id                 0.000000
    category_1              0.000000
    installments            0.000000
    category_3              0.028488
    merchant_category_id    0.000000
    merchant_id             0.013355
    month_lag               0.000000
    purchase_amount         0.000000
    purchase_date           0.000000
    category_2              0.056925
    state_id                0.000000
    subsector_id            0.000000
    dtype: float64
```

It's a standard practice to remove the rows which have missing data. Doing that had negative impact in the leaderboard score for me in the competition. So I decided to fill in the missing values selectively with mode for categorical columns & for installments with -1 & 999.

Now the second step of data preprocessing, mapping the categorical columns.

Let's map `category_1` & `category_3` as categorical columns with key & value.

```python
def fill_nas_for_transactions_df(df):
    # Fill nas for category_3 with mode
    df['category_2'].fillna(1.0,inplace=True)
    df['category_3'].fillna('A',inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    df['installments'].replace(-1, np.nan,inplace=True)
    df['installments'].replace(999, np.nan,inplace=True)
    df['category_1'] = df['category_1'].map({'Y':1, 'N':0})
    df['category_3'] = df['category_3'].map({'A':0, 'B':1, 'C':2})
    return df
```

We are also skipping another step which reduces the size of the dataframe by manually mapping the columns to lower data-type wherever possible without data loss.

```python
dfs = [hist_trans, new_hist_trans]

hist_trans, new_hist_trans = [fill_nas_for_transactions_df(df) for df in dfs]
```

## Feature engineering

Feature engineering consists of various logical steps. Feature engineering played a very big role for this competition in particular. I can classify broadly into following categories:

1. Interpreting dates and adding new features dependant on date & time like time since, first, latest, difference between occurances etc.
2. Statistical aggregates on numerical columns like avg, percentile, max, min, peak-to-peak (ptp) etc.
3. Statistical aggregates grouped by categorical columns.
4. Feature interactions amongst core features of date, transaction amounts etc
5. Feature interactions on the aggregated features of different dataframes like merchants, cards & transactions.
6. Feature interactions between new & old transactions.
7. Data mining for reverse engineering the normalized features like purchase_amount & observed_date.
8. Advanced aggregrates grouped by month & later aggregated turned out be futile. Nevertheless I'm including them here.

### Feature: Add date features meta info

fast.ai library provides excellent utility functions. We will be leveraging functions from fast.ai throughout for data manipulation & feature engineering starting with date field. `add_datepart` function reads a date and generates additional interpreted fields which are very useful for feature engineering.

```python
add_datepart(hist_trans, 'purchase_date', drop=False, time=True)
add_datepart(new_hist_trans, 'purchase_date', drop=False, time=True)
new_hist_trans.head().T
```

After adding the date features our dataframe looks like this:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>authorized_flag</th>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>card_id</th>
      <td>C_ID_415bb3a509</td>
      <td>C_ID_415bb3a509</td>
      <td>C_ID_415bb3a509</td>
      <td>C_ID_415bb3a509</td>
      <td>C_ID_ef55cf8d4b</td>
    </tr>
    <tr>
      <th>city_id</th>
      <td>107</td>
      <td>140</td>
      <td>330</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>category_1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>installments</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>category_3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>merchant_category_id</th>
      <td>307</td>
      <td>307</td>
      <td>507</td>
      <td>661</td>
      <td>166</td>
    </tr>
    <tr>
      <th>merchant_id</th>
      <td>M_ID_b0c793002c</td>
      <td>M_ID_88920c89e8</td>
      <td>M_ID_ad5237ef6b</td>
      <td>M_ID_9e84cda3b1</td>
      <td>M_ID_3c86fa3831</td>
    </tr>
    <tr>
      <th>month_lag</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>purchase_amount</th>
      <td>-0.557574</td>
      <td>-0.56958</td>
      <td>-0.551037</td>
      <td>-0.671925</td>
      <td>-0.659904</td>
    </tr>
    <tr>
      <th>purchase_date</th>
      <td>2018-03-11 14:57:36</td>
      <td>2018-03-19 18:53:37</td>
      <td>2018-04-26 14:08:44</td>
      <td>2018-03-07 09:43:21</td>
      <td>2018-03-22 21:07:53</td>
    </tr>
    <tr>
      <th>category_2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>state_id</th>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>subsector_id</th>
      <td>19</td>
      <td>19</td>
      <td>14</td>
      <td>8</td>
      <td>29</td>
    </tr>
    <tr>
      <th>purchase_Year</th>
      <td>2018</td>
      <td>2018</td>
      <td>2018</td>
      <td>2018</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>purchase_Month</th>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>purchase_Week</th>
      <td>10</td>
      <td>12</td>
      <td>17</td>
      <td>10</td>
      <td>12</td>
    </tr>
    <tr>
      <th>purchase_Day</th>
      <td>11</td>
      <td>19</td>
      <td>26</td>
      <td>7</td>
      <td>22</td>
    </tr>
    <tr>
      <th>purchase_Dayofweek</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>purchase_Dayofyear</th>
      <td>70</td>
      <td>78</td>
      <td>116</td>
      <td>66</td>
      <td>81</td>
    </tr>
    <tr>
      <th>purchase_Is_month_end</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>purchase_Is_month_start</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>purchase_Is_quarter_end</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>purchase_Is_quarter_start</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>purchase_Is_year_end</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>purchase_Is_year_start</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>purchase_Hour</th>
      <td>14</td>
      <td>18</td>
      <td>14</td>
      <td>9</td>
      <td>21</td>
    </tr>
    <tr>
      <th>purchase_Minute</th>
      <td>57</td>
      <td>53</td>
      <td>8</td>
      <td>43</td>
      <td>7</td>
    </tr>
    <tr>
      <th>purchase_Second</th>
      <td>36</td>
      <td>37</td>
      <td>44</td>
      <td>21</td>
      <td>53</td>
    </tr>
    <tr>
      <th>purchase_Elapsed</th>
      <td>1520780256</td>
      <td>1521485617</td>
      <td>1524751724</td>
      <td>1520415801</td>
      <td>1521752873</td>
    </tr>
  </tbody>
</table>
</div>

### Feature: Add derived date features

`add_datepart` already added columns like `'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'` to our dataframe. Let's add some more columns like if the transaction has been done on weekday or weekend, `month_diff` which says the months since the reference date. We will find out later that these columns turned out to be pretty strong features.

```python
sns.kdeplot(new_hist_trans['purchase_amount'])
# sns.kdeplot(new_hist_trans['purchase_amount'], bw=.2, label="bw: 0.2")
# sns.kdeplot(new_hist_trans['purchase_amount'])
plt.legend();
plt.xlim(-20,20)
sns.distplot(new_hist_trans['purchase_amount'], bins=20, kde=True, rug=False);
```

![png](/images/output_34_1.png)

We will add more features like if the purchases were made on weekend or weekday, if the transaction was authorized or not. We will also define `month_diff` as the months from the transaction has happened.

```python
def add_extra_cols(df):
    df['purchased_on_weekend'] = (df.purchase_Dayofweek >=5).astype(int)
    df['purchased_on_weekday'] = (df.purchase_Dayofweek <5).astype(int)
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
#     Trim the purchase_amount
#     df['purchase_amount'] = df['purchase_amount'].apply(lambda x: min(x, 0.8))
    return df
hist_trans, new_hist_trans = [add_extra_cols(df) for df in dfs]
hist_trans.shape, new_hist_trans.shape
```

```text
    ((29112361, 33), (1963031, 33))
```

The new dataframe after adding features are:

```python
new_hist_trans.head().T
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>authorized_flag</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>card_id</th>
      <td>C_ID_415bb3a509</td>
      <td>C_ID_415bb3a509</td>
      <td>C_ID_415bb3a509</td>
      <td>C_ID_415bb3a509</td>
      <td>C_ID_ef55cf8d4b</td>
    </tr>
    <tr>
      <th>city_id</th>
      <td>107</td>
      <td>140</td>
      <td>330</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>category_1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>installments</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>category_3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>merchant_category_id</th>
      <td>307</td>
      <td>307</td>
      <td>507</td>
      <td>661</td>
      <td>166</td>
    </tr>
    <tr>
      <th>merchant_id</th>
      <td>M_ID_b0c793002c</td>
      <td>M_ID_88920c89e8</td>
      <td>M_ID_ad5237ef6b</td>
      <td>M_ID_9e84cda3b1</td>
      <td>M_ID_3c86fa3831</td>
    </tr>
    <tr>
      <th>month_lag</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>purchase_amount</th>
      <td>-0.557574</td>
      <td>-0.56958</td>
      <td>-0.551037</td>
      <td>-0.671925</td>
      <td>-0.659904</td>
    </tr>
    <tr>
      <th>purchase_date</th>
      <td>2018-03-11 14:57:36</td>
      <td>2018-03-19 18:53:37</td>
      <td>2018-04-26 14:08:44</td>
      <td>2018-03-07 09:43:21</td>
      <td>2018-03-22 21:07:53</td>
    </tr>
    <tr>
      <th>category_2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>state_id</th>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>subsector_id</th>
      <td>19</td>
      <td>19</td>
      <td>14</td>
      <td>8</td>
      <td>29</td>
    </tr>
    <tr>
      <th>purchase_Year</th>
      <td>2018</td>
      <td>2018</td>
      <td>2018</td>
      <td>2018</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>purchase_Month</th>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>purchase_Week</th>
      <td>10</td>
      <td>12</td>
      <td>17</td>
      <td>10</td>
      <td>12</td>
    </tr>
    <tr>
      <th>purchase_Day</th>
      <td>11</td>
      <td>19</td>
      <td>26</td>
      <td>7</td>
      <td>22</td>
    </tr>
    <tr>
      <th>purchase_Dayofweek</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>purchase_Dayofyear</th>
      <td>70</td>
      <td>78</td>
      <td>116</td>
      <td>66</td>
      <td>81</td>
    </tr>
    <tr>
      <th>purchase_Is_month_end</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>purchase_Is_month_start</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>purchase_Is_quarter_end</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>purchase_Is_quarter_start</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>purchase_Is_year_end</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>purchase_Is_year_start</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>purchase_Hour</th>
      <td>14</td>
      <td>18</td>
      <td>14</td>
      <td>9</td>
      <td>21</td>
    </tr>
    <tr>
      <th>purchase_Minute</th>
      <td>57</td>
      <td>53</td>
      <td>8</td>
      <td>43</td>
      <td>7</td>
    </tr>
    <tr>
      <th>purchase_Second</th>
      <td>36</td>
      <td>37</td>
      <td>44</td>
      <td>21</td>
      <td>53</td>
    </tr>
    <tr>
      <th>purchase_Elapsed</th>
      <td>1520780256</td>
      <td>1521485617</td>
      <td>1524751724</td>
      <td>1520415801</td>
      <td>1521752873</td>
    </tr>
    <tr>
      <th>purchased_on_weekend</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>purchased_on_weekday</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>month_diff</th>
      <td>19</td>
      <td>18</td>
      <td>18</td>
      <td>19</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>

### Feature: Reverse engineering purchase amount column

The `purchase_amount` column is normalized and heavily transformed. Let's try to reverse engineer it to get the actual amount. One kaggler in the competition solved it by using a simple optimisation function to make the least value of purchase amount to be non-negative and tuning the regression co-efficients to round purchase_amount to 2 decimals. Let's see a different approach for achieving the same.

We start with basic assumption that the purchase amount was normalized using scaling & transforming.

```python
data = pd.concat((historical_transactions, new_transactions))
data.purchase_amount.mean()
-0.0007032266623490891
```

The mean is indeed zero. They have been normalized!! Purchase amounts can be never negative, so let's start it with being zero.

```python
data['purchase_amount_new'] = (data.purchase_amount - data.purchase_amount.min())
```

Let's sort them and look at the head:
| index | purchase_amount_new |
|------------|----------|
| 0 | 0 |
| 1 | 0.000015 |
| 2 | 0.00003 |
| 3 | 0.000045 |
| 4 | 0.00006 |
| 5 | 0.000075 |
| 6 | 0.00009 |
| 7 | 0.000105 |
| 8 | 0.00012 |
| 9 | 0.000135 |
Let's compute the successive diff between the purchase amounts:

```python
data['delta'] = data.purchase_amount_new.diff(1)
data.head(10)
```

| index | purchase_amount_new | delta    |
| ----- | ------------------- | -------- |
| 0     | 0                   | NaN      |
| 1     | 0.000015            | 0.000015 |
| 2     | 0.00003             | 0.000015 |
| 3     | 0.000045            | 0.000015 |
| 4     | 0.00006             | 0.000015 |
| 5     | 0.000075            | 0.000015 |
| 6     | 0.00009             | 0.000015 |
| 7     | 0.000105            | 0.000015 |
| 8     | 0.00012             | 0.000015 |
| 9     | 0.000135            | 0.000015 |

Hmm. All the deltas are looking same. That's an interesting find. Does is apply for all?

```python
data[data.delta > 2e-5].head()
```

| index | purchase_amount_new | delta   |
| ----- | ------------------- | ------- |
| 52623 | 0.790755            | 0.00003 |
| 54532 | 0.819456            | 0.00003 |
| 57407 | 0.862672            | 0.00003 |
| 60592 | 0.910547            | 0.00003 |
| 60757 | 0.913041            | 0.00003 |

They tail to 0.0003 after 2e-5. Let's get the mean of it.

```python
data.delta.mean()
1.5026511915168561e-05
```

This should be the least delta between two prices. Let's assume the least denomination difference between our prices to be 100 cents and each cent is approximately equal to 1.5026511915168561e-05. Dividing our purchase_amount_new with 100\*delta_mean should get approximate price.

```python
data['purchase_amount_new'] = data.purchase_amount_new / (100 * data.delta.mean())
```

Now lets' look at the most frequent values in our new purchase amount:

```python
data.purchase_amount_new.value_counts().head(10)

50.000000     735619
20.000004     640964
30.000003     547680
10.000005     444249
100.000001    418773
15.000001     379041
40.000002     271846
12.000004     233231
25.000000     232732
5.000003      208044
Name: purchase_amount_new, dtype: int64
```

Woohoo! They appear close. Let's round them to 2 decimals

```python
data['two_decimal_amount'] = np.round(data.purchase_amount_new, 2)
```

### Feature: Aggregates grouped by month & card_id

Now that we have our purchase amounts, lets' calculate `mean`, `sum`, `max`, `peak to peak` aggregates per month grouped by `card_id`.

```python
def aggregate_per_month(history):
    grouped = history.groupby(['card_id', 'month_diff'])['purchase_amount']

    agg_func = {
            'purchase_amount': ['count', 'sum', 'max', 'mean'],
            }

    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)

    final_group = intermediate_group.groupby('card_id').agg(['mean', 'sum', np.ptp, 'max'])
    final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
    final_group.reset_index(inplace=True)

    return final_group
```

### Feature: Time between successive transactions

Adding another new feature - time between successive transactions. The time taken by a customer between his transactions. For this we group our transactions by `card_id` and find the time diff between successive transactions.

```python
def time_diff(df):
    df['purchase_date_successive_diff'] = df.sort_values('purchase_date').groupby('card_id')['purchase_date'].diff().dt.total_seconds()
    df['purchase_date_successive_diff'].fillna(0, inplace=True)
    return df

hist_trans, new_hist_trans = [time_diff(df) for df in dfs]
```

Save the data frame to disk in feather format. This is our first milestone.

```python
hist_trans.to_feather('hist_trans_beta')
new_hist_trans.to_feather('new_hist_trans_beta')
# hist_trans = feather.read_dataframe('hist_trans_beta')
new_hist_trans = feather.read_dataframe('new_hist_trans_beta')
dfs = [hist_trans, new_hist_trans]
```

### Feature: Holiday features

Adding more features like holidays in Brazil. Typically on the holidays merchants see a spike in sales and have very good offers. We will also add other features like what was the EMI paid every month which is `purchase_amount` divided by `installments`. Another feature `duration` which is a feature interaction between `purchase_amount` and `month_diff` (months from reference date). We also have `amount_month_ratio` which is also a feature interaction `purchase_amount` between `month_diff` which signifies amount spent by a customer from the reference date. This one adds more weight to the spendings close to the reference date.

```python
def additional_feats(hist_df):
    hist_df['price'] = hist_df['purchase_amount'] / hist_df['installments']
    #Christmas : December 25 2017
    hist_df['Christmas_Day_2017']=(pd.to_datetime('2017-12-25')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Mothers Day: May 14 2017
    hist_df['Mothers_Day_2017']=(pd.to_datetime('2017-06-04')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #fathers day: August 13 2017
    hist_df['fathers_day_2017']=(pd.to_datetime('2017-08-13')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Childrens day: October 12 2017
    hist_df['Children_day_2017']=(pd.to_datetime('2017-10-12')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Valentine's Day : 12th June, 2017
    hist_df['Valentine_Day_2017']=(pd.to_datetime('2017-06-12')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Black Friday : 24th November 2017
    hist_df['Black_Friday_2017']=(pd.to_datetime('2017-11-24') - hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #2018
    #Mothers Day: May 13 2018
    hist_df['Mothers_Day_2018']=(pd.to_datetime('2018-05-13')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    hist_df['duration'] = hist_df['purchase_amount']*hist_df['month_diff']
    hist_df['amount_month_ratio'] = hist_df['purchase_amount']/hist_df['month_diff']
    return hist_df
```

After adding the additional features we copy them to existing dataframes to save space.

```python
hist_trans, new_hist_trans = [additional_feats(df) for df in dfs]
hist_trans.shape, new_hist_trans.shape
((29112361, 52), (1963031, 52))
```

### Feature: Latest & First 5 feature aggregates

As seen in previous additional feature `amount_month_ratio` which adds more weight to purchases close to reference date, let's add features like sum/max of last 5 purchases etc.

```python
def head_sum(x):
    return x.head().sum()

def head_max(x):
    return x.head().max()

def tail_sum(x):
    return x.tail().sum()

def tail_max(x):
    return x.tail().max()
```

We will sort all our transactions by purchase date chronologically.

```python
%time new_hist_trans = new_hist_trans.sort_values('purchase_date')


CPU times: user 864 ms, sys: 681 ms, total: 1.54 s
Wall time: 358 ms
```

```python
%time hist_trans = hist_trans.sort_values('purchase_date')


CPU times: user 4.94 s, sys: 3.11 s, total: 8.05 s
Wall time: 6.06 s
# hist_trans.head()
new_hist_trans = new_hist_trans.reset_index().drop('index', axis=1)
hist_trans.reset_index(inplace=True)
hist_trans.drop('index', axis=1, inplace=True)
hist_trans.head()
hist_trans.drop('level_0', axis=1, inplace=True)
```

### Feature: Merchants features

```python
merchants = pd.read_csv('data/elo/merchants.csv')
merchants.head().T
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>merchant_id</th>
      <td>M_ID_838061e48c</td>
      <td>M_ID_9339d880ad</td>
      <td>M_ID_e726bbae1e</td>
      <td>M_ID_a70e9c5f81</td>
      <td>M_ID_64456c37ce</td>
    </tr>
    <tr>
      <th>merchant_group_id</th>
      <td>8353</td>
      <td>3184</td>
      <td>447</td>
      <td>5026</td>
      <td>2228</td>
    </tr>
    <tr>
      <th>merchant_category_id</th>
      <td>792</td>
      <td>840</td>
      <td>690</td>
      <td>792</td>
      <td>222</td>
    </tr>
    <tr>
      <th>subsector_id</th>
      <td>9</td>
      <td>20</td>
      <td>1</td>
      <td>9</td>
      <td>21</td>
    </tr>
    <tr>
      <th>numerical_1</th>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
    </tr>
    <tr>
      <th>numerical_2</th>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
    </tr>
    <tr>
      <th>category_1</th>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>most_recent_sales_range</th>
      <td>E</td>
      <td>E</td>
      <td>E</td>
      <td>E</td>
      <td>E</td>
    </tr>
    <tr>
      <th>most_recent_purchases_range</th>
      <td>E</td>
      <td>E</td>
      <td>E</td>
      <td>E</td>
      <td>E</td>
    </tr>
    <tr>
      <th>avg_sales_lag3</th>
      <td>-0.4</td>
      <td>-0.72</td>
      <td>-82.13</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>avg_purchases_lag3</th>
      <td>9.66667</td>
      <td>1.75</td>
      <td>260</td>
      <td>1.66667</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>active_months_lag3</th>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>avg_sales_lag6</th>
      <td>-2.25</td>
      <td>-0.74</td>
      <td>-82.13</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>avg_purchases_lag6</th>
      <td>18.6667</td>
      <td>1.29167</td>
      <td>260</td>
      <td>4.66667</td>
      <td>0.361111</td>
    </tr>
    <tr>
      <th>active_months_lag6</th>
      <td>6</td>
      <td>6</td>
      <td>2</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>avg_sales_lag12</th>
      <td>-2.32</td>
      <td>-0.57</td>
      <td>-82.13</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>avg_purchases_lag12</th>
      <td>13.9167</td>
      <td>1.6875</td>
      <td>260</td>
      <td>3.83333</td>
      <td>0.347222</td>
    </tr>
    <tr>
      <th>active_months_lag12</th>
      <td>12</td>
      <td>12</td>
      <td>2</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>category_4</th>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>city_id</th>
      <td>242</td>
      <td>22</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>state_id</th>
      <td>9</td>
      <td>16</td>
      <td>5</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>category_2</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

Let's have a look at the dataset summary

```python
DataFrameSummary(merchants).summary().T
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>counts</th>
      <th>uniques</th>
      <th>missing</th>
      <th>missing_perc</th>
      <th>types</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>merchant_id</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>334696</td>
      <td>334633</td>
      <td>0</td>
      <td>0%</td>
      <td>categorical</td>
    </tr>
    <tr>
      <th>merchant_group_id</th>
      <td>334696</td>
      <td>31028.7</td>
      <td>31623</td>
      <td>1</td>
      <td>3612</td>
      <td>19900</td>
      <td>51707.2</td>
      <td>112586</td>
      <td>334696</td>
      <td>109391</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>merchant_category_id</th>
      <td>334696</td>
      <td>423.132</td>
      <td>252.898</td>
      <td>-1</td>
      <td>222</td>
      <td>373</td>
      <td>683</td>
      <td>891</td>
      <td>334696</td>
      <td>324</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>subsector_id</th>
      <td>334696</td>
      <td>25.1164</td>
      <td>9.80737</td>
      <td>-1</td>
      <td>19</td>
      <td>27</td>
      <td>33</td>
      <td>41</td>
      <td>334696</td>
      <td>41</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>numerical_1</th>
      <td>334696</td>
      <td>0.0114764</td>
      <td>1.09815</td>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
      <td>-0.0475558</td>
      <td>183.735</td>
      <td>334696</td>
      <td>954</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>numerical_2</th>
      <td>334696</td>
      <td>0.00810311</td>
      <td>1.0705</td>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
      <td>-0.0475558</td>
      <td>182.079</td>
      <td>334696</td>
      <td>947</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>category_1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>334696</td>
      <td>2</td>
      <td>0</td>
      <td>0%</td>
      <td>bool</td>
    </tr>
    <tr>
      <th>most_recent_sales_range</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>334696</td>
      <td>5</td>
      <td>0</td>
      <td>0%</td>
      <td>categorical</td>
    </tr>
    <tr>
      <th>most_recent_purchases_range</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>334696</td>
      <td>5</td>
      <td>0</td>
      <td>0%</td>
      <td>categorical</td>
    </tr>
    <tr>
      <th>avg_sales_lag3</th>
      <td>334683</td>
      <td>13.833</td>
      <td>2395.49</td>
      <td>-82.13</td>
      <td>0.88</td>
      <td>1</td>
      <td>1.16</td>
      <td>851845</td>
      <td>334683</td>
      <td>3372</td>
      <td>13</td>
      <td>0.00%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>avg_purchases_lag3</th>
      <td>334696</td>
      <td>inf</td>
      <td>NaN</td>
      <td>0.333495</td>
      <td>0.92365</td>
      <td>1.01667</td>
      <td>1.14652</td>
      <td>inf</td>
      <td>334696</td>
      <td>100003</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>active_months_lag3</th>
      <td>334696</td>
      <td>2.99411</td>
      <td>0.0952475</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>334696</td>
      <td>3</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>avg_sales_lag6</th>
      <td>334683</td>
      <td>21.6508</td>
      <td>3947.11</td>
      <td>-82.13</td>
      <td>0.85</td>
      <td>1.01</td>
      <td>1.23</td>
      <td>1.51396e+06</td>
      <td>334683</td>
      <td>4507</td>
      <td>13</td>
      <td>0.00%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>avg_purchases_lag6</th>
      <td>334696</td>
      <td>inf</td>
      <td>NaN</td>
      <td>0.167045</td>
      <td>0.902247</td>
      <td>1.02696</td>
      <td>1.21558</td>
      <td>inf</td>
      <td>334696</td>
      <td>135202</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>active_months_lag6</th>
      <td>334696</td>
      <td>5.9474</td>
      <td>0.394936</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>334696</td>
      <td>6</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>avg_sales_lag12</th>
      <td>334683</td>
      <td>25.2277</td>
      <td>5251.84</td>
      <td>-82.13</td>
      <td>0.85</td>
      <td>1.02</td>
      <td>1.29</td>
      <td>2.56741e+06</td>
      <td>334683</td>
      <td>5009</td>
      <td>13</td>
      <td>0.00%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>avg_purchases_lag12</th>
      <td>334696</td>
      <td>inf</td>
      <td>NaN</td>
      <td>0.0983295</td>
      <td>0.898333</td>
      <td>1.04336</td>
      <td>1.26648</td>
      <td>inf</td>
      <td>334696</td>
      <td>172917</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>active_months_lag12</th>
      <td>334696</td>
      <td>11.5993</td>
      <td>1.52014</td>
      <td>1</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>334696</td>
      <td>12</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>category_4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>334696</td>
      <td>2</td>
      <td>0</td>
      <td>0%</td>
      <td>bool</td>
    </tr>
    <tr>
      <th>city_id</th>
      <td>334696</td>
      <td>102.918</td>
      <td>107.091</td>
      <td>-1</td>
      <td>-1</td>
      <td>69</td>
      <td>182</td>
      <td>347</td>
      <td>334696</td>
      <td>271</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>state_id</th>
      <td>334696</td>
      <td>11.8609</td>
      <td>6.17689</td>
      <td>-1</td>
      <td>9</td>
      <td>9</td>
      <td>16</td>
      <td>24</td>
      <td>334696</td>
      <td>25</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>category_2</th>
      <td>322809</td>
      <td>2.38</td>
      <td>1.56266</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>322809</td>
      <td>5</td>
      <td>11887</td>
      <td>3.55%</td>
      <td>numeric</td>
    </tr>
  </tbody>
</table>
</div>

We see that we have 334696 rows but only 334633 merchants. Maybe some of them are duplicated. Let's drop duplicates and check again.

```python
merchant_details_cols = ['merchant_id', 'merchant_group_id', 'merchant_category_id', 'subsector_id', 'category_1',
                        'category_4', 'city_id', 'state_id', 'category_2']
merchant_details = merchants[merchant_details_cols]

# Delete duplicates
merchant_details = merchant_details.drop_duplicates()
```

```python
DataFrameSummary(merchant_details).summary().T
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>counts</th>
      <th>uniques</th>
      <th>missing</th>
      <th>missing_perc</th>
      <th>types</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>merchant_id</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>334645</td>
      <td>334633</td>
      <td>0</td>
      <td>0%</td>
      <td>categorical</td>
    </tr>
    <tr>
      <th>merchant_group_id</th>
      <td>334645</td>
      <td>31032.5</td>
      <td>31623.2</td>
      <td>1</td>
      <td>3625</td>
      <td>19908</td>
      <td>51716</td>
      <td>112586</td>
      <td>334645</td>
      <td>109391</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>merchant_category_id</th>
      <td>334645</td>
      <td>423.123</td>
      <td>252.905</td>
      <td>-1</td>
      <td>222</td>
      <td>373</td>
      <td>683</td>
      <td>891</td>
      <td>334645</td>
      <td>324</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>subsector_id</th>
      <td>334645</td>
      <td>25.1171</td>
      <td>9.80706</td>
      <td>-1</td>
      <td>19</td>
      <td>27</td>
      <td>33</td>
      <td>41</td>
      <td>334645</td>
      <td>41</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>category_1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>334645</td>
      <td>2</td>
      <td>0</td>
      <td>0%</td>
      <td>bool</td>
    </tr>
    <tr>
      <th>category_4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>334645</td>
      <td>2</td>
      <td>0</td>
      <td>0%</td>
      <td>bool</td>
    </tr>
    <tr>
      <th>city_id</th>
      <td>334645</td>
      <td>102.925</td>
      <td>107.093</td>
      <td>-1</td>
      <td>-1</td>
      <td>69</td>
      <td>182</td>
      <td>347</td>
      <td>334645</td>
      <td>271</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>state_id</th>
      <td>334645</td>
      <td>11.8616</td>
      <td>6.17629</td>
      <td>-1</td>
      <td>9</td>
      <td>9</td>
      <td>16</td>
      <td>24</td>
      <td>334645</td>
      <td>25</td>
      <td>0</td>
      <td>0%</td>
      <td>numeric</td>
    </tr>
    <tr>
      <th>category_2</th>
      <td>322778</td>
      <td>2.38005</td>
      <td>1.56268</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>322778</td>
      <td>5</td>
      <td>11867</td>
      <td>3.55%</td>
      <td>numeric</td>
    </tr>
  </tbody>
</table>
</div>

We still see that out of 334645 rows we have only 334633 unique merchant ids. `drop_duplicates` compares values of each and every column before marking them as duplicate. These merchants aren't duplicate after all, they may be sharing a different city, region or feature like franchises etc. Let's dig deeper.

```python
merchant_details.loc[merchant_details['merchant_id'].duplicated()]
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>merchant_id</th>
      <th>merchant_group_id</th>
      <th>merchant_category_id</th>
      <th>subsector_id</th>
      <th>category_1</th>
      <th>category_4</th>
      <th>city_id</th>
      <th>state_id</th>
      <th>category_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3393</th>
      <td>M_ID_bd49e37dda</td>
      <td>4170</td>
      <td>692</td>
      <td>21</td>
      <td>N</td>
      <td>N</td>
      <td>51</td>
      <td>16</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4182</th>
      <td>M_ID_ef233cff26</td>
      <td>28799</td>
      <td>560</td>
      <td>34</td>
      <td>N</td>
      <td>Y</td>
      <td>69</td>
      <td>9</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7585</th>
      <td>M_ID_dbbf07ebf0</td>
      <td>35</td>
      <td>278</td>
      <td>37</td>
      <td>N</td>
      <td>Y</td>
      <td>17</td>
      <td>22</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>29465</th>
      <td>M_ID_30340088f2</td>
      <td>35</td>
      <td>544</td>
      <td>29</td>
      <td>N</td>
      <td>Y</td>
      <td>69</td>
      <td>9</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>47804</th>
      <td>M_ID_645a6af169</td>
      <td>19140</td>
      <td>87</td>
      <td>27</td>
      <td>N</td>
      <td>N</td>
      <td>29</td>
      <td>15</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>166813</th>
      <td>M_ID_ebbdb42da6</td>
      <td>35</td>
      <td>383</td>
      <td>2</td>
      <td>Y</td>
      <td>Y</td>
      <td>-1</td>
      <td>-1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>210654</th>
      <td>M_ID_c2b9ac2ea4</td>
      <td>35</td>
      <td>554</td>
      <td>25</td>
      <td>Y</td>
      <td>Y</td>
      <td>-1</td>
      <td>-1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>221181</th>
      <td>M_ID_992a180b15</td>
      <td>8568</td>
      <td>554</td>
      <td>25</td>
      <td>N</td>
      <td>Y</td>
      <td>17</td>
      <td>22</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>228275</th>
      <td>M_ID_d123532c72</td>
      <td>49094</td>
      <td>385</td>
      <td>17</td>
      <td>Y</td>
      <td>Y</td>
      <td>-1</td>
      <td>-1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>330958</th>
      <td>M_ID_42697d5d44</td>
      <td>35</td>
      <td>690</td>
      <td>1</td>
      <td>N</td>
      <td>N</td>
      <td>271</td>
      <td>9</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>333904</th>
      <td>M_ID_6464db3b45</td>
      <td>35</td>
      <td>210</td>
      <td>35</td>
      <td>Y</td>
      <td>Y</td>
      <td>-1</td>
      <td>-1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>334071</th>
      <td>M_ID_1802942aaf</td>
      <td>72963</td>
      <td>302</td>
      <td>22</td>
      <td>N</td>
      <td>N</td>
      <td>96</td>
      <td>9</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

Let's fill in the missing values before digging further

```python
merchants['category_2'] = merchants['category_2'].fillna(0).astype(int)
merchants.loc[merchants['city_id'] == -1, 'city_id'] = 0
merchants.loc[merchants['state_id'] == -1, 'state_id'] = 0
```

We will create a unique vector string which is just concatenation of our need to be unique columns. We will not use `merchant_group_id` as it is not present in transactions table.

```python
merchant_address_id = merchants['merchant_id'].map(str)+ '_' + \
    merchants['merchant_category_id'].map(str) + '_'+ \
    merchants['subsector_id'].map(str) + '_'+ \
    merchants['city_id'].map(str) + '_'+ \
    merchants['state_id'].map(str) + '_'+ \
    merchants['category_2'].map(str)

merchants['merchant_address_id'] = merchant_address_id
```

```python
# merchants.drop(['merchant_id', 'merchant_category_id', 'subsector_id',
#                          'city_id', 'state_id', 'category_2'], axis=1, inplace=True)

merchants.head().T
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>merchant_id</th>
      <td>M_ID_838061e48c</td>
      <td>M_ID_9339d880ad</td>
      <td>M_ID_e726bbae1e</td>
      <td>M_ID_a70e9c5f81</td>
      <td>M_ID_64456c37ce</td>
    </tr>
    <tr>
      <th>merchant_group_id</th>
      <td>8353</td>
      <td>3184</td>
      <td>447</td>
      <td>5026</td>
      <td>2228</td>
    </tr>
    <tr>
      <th>merchant_category_id</th>
      <td>792</td>
      <td>840</td>
      <td>690</td>
      <td>792</td>
      <td>222</td>
    </tr>
    <tr>
      <th>subsector_id</th>
      <td>9</td>
      <td>20</td>
      <td>1</td>
      <td>9</td>
      <td>21</td>
    </tr>
    <tr>
      <th>numerical_1</th>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
    </tr>
    <tr>
      <th>numerical_2</th>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
      <td>-0.0574706</td>
    </tr>
    <tr>
      <th>category_1</th>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>most_recent_sales_range</th>
      <td>E</td>
      <td>E</td>
      <td>E</td>
      <td>E</td>
      <td>E</td>
    </tr>
    <tr>
      <th>most_recent_purchases_range</th>
      <td>E</td>
      <td>E</td>
      <td>E</td>
      <td>E</td>
      <td>E</td>
    </tr>
    <tr>
      <th>avg_sales_lag3</th>
      <td>-0.4</td>
      <td>-0.72</td>
      <td>-82.13</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>avg_purchases_lag3</th>
      <td>9.66667</td>
      <td>1.75</td>
      <td>260</td>
      <td>1.66667</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>active_months_lag3</th>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>avg_sales_lag6</th>
      <td>-2.25</td>
      <td>-0.74</td>
      <td>-82.13</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>avg_purchases_lag6</th>
      <td>18.6667</td>
      <td>1.29167</td>
      <td>260</td>
      <td>4.66667</td>
      <td>0.361111</td>
    </tr>
    <tr>
      <th>active_months_lag6</th>
      <td>6</td>
      <td>6</td>
      <td>2</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>avg_sales_lag12</th>
      <td>-2.32</td>
      <td>-0.57</td>
      <td>-82.13</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>avg_purchases_lag12</th>
      <td>13.9167</td>
      <td>1.6875</td>
      <td>260</td>
      <td>3.83333</td>
      <td>0.347222</td>
    </tr>
    <tr>
      <th>active_months_lag12</th>
      <td>12</td>
      <td>12</td>
      <td>2</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>category_4</th>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>city_id</th>
      <td>242</td>
      <td>22</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>state_id</th>
      <td>9</td>
      <td>16</td>
      <td>5</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>category_2</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>merchant_address_id</th>
      <td>M_ID_838061e48c_792_9_242_9_1.0</td>
      <td>M_ID_9339d880ad_840_20_22_16_1.0</td>
      <td>M_ID_e726bbae1e_690_1_-1_5_5.0</td>
      <td>M_ID_a70e9c5f81_792_9_-1_-1_nan</td>
      <td>M_ID_64456c37ce_222_21_-1_-1_nan</td>
    </tr>
  </tbody>
</table>
</div>

Now the merchants data! We have already pre-processed and added some features to the merchants dataframe. Let's load it in and merge with our transactions data.

```python
merchants = feather.read_dataframe('merchants_df')
new_hist_trans = new_hist_trans.merge(merchants, on='merchant_id', how='left')
hist_trans = hist_trans.merge(merchants, on='merchant_id', how='left')
hist_trans.shape, new_hist_trans.shape
hist_trans.to_feather('hist_trans_beta')
new_hist_trans.to_feather('new_hist_trans_beta')
hist_trans = feather.read_dataframe('hist_trans_beta')
new_hist_trans = feather.read_dataframe('new_hist_trans_beta')
```

After merging the merchants data to our transactions data, let's see the final list of columns.

```python
new_hist_trans.head().T
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>authorized_flag</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>card_id</th>
      <td>C_ID_7c628841cb</td>
      <td>C_ID_25d399500c</td>
      <td>C_ID_e3542c52f1</td>
      <td>C_ID_fabd47ca44</td>
      <td>C_ID_6f9a771d17</td>
    </tr>
    <tr>
      <th>city_id</th>
      <td>69</td>
      <td>19</td>
      <td>199</td>
      <td>69</td>
      <td>96</td>
    </tr>
    <tr>
      <th>category_1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>installments</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>category_3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>merchant_category_id</th>
      <td>80</td>
      <td>307</td>
      <td>307</td>
      <td>80</td>
      <td>178</td>
    </tr>
    <tr>
      <th>merchant_id</th>
      <td>M_ID_c03b62d83d</td>
      <td>M_ID_2445d76702</td>
      <td>M_ID_b16ae63c45</td>
      <td>M_ID_b35d0757d1</td>
      <td>M_ID_b6b9b8ed67</td>
    </tr>
    <tr>
      <th>month_lag</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>purchase_amount</th>
      <td>46.4</td>
      <td>174.01</td>
      <td>50</td>
      <td>5.5</td>
      <td>550</td>
    </tr>
    <tr>
      <th>purchase_date</th>
      <td>2017-03-01 03:24:51</td>
      <td>2017-03-01 11:01:06</td>
      <td>2017-03-01 11:27:39</td>
      <td>2017-03-01 11:55:11</td>
      <td>2017-03-01 12:37:26</td>
    </tr>
    <tr>
      <th>category_2</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>state_id</th>
      <td>9</td>
      <td>9</td>
      <td>14</td>
      <td>9</td>
      <td>24</td>
    </tr>
    <tr>
      <th>subsector_id</th>
      <td>37</td>
      <td>19</td>
      <td>19</td>
      <td>37</td>
      <td>29</td>
    </tr>
    <tr>
      <th>purchase_Year</th>
      <td>2017</td>
      <td>2017</td>
      <td>2017</td>
      <td>2017</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>purchase_Month</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>purchase_Week</th>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>purchase_Day</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>purchase_Dayofweek</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>purchase_Dayofyear</th>
      <td>60</td>
      <td>60</td>
      <td>60</td>
      <td>60</td>
      <td>60</td>
    </tr>
    <tr>
      <th>purchase_Is_month_end</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>purchase_Is_month_start</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>purchase_Is_quarter_end</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>purchase_Is_quarter_start</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>purchase_Is_year_end</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>purchase_Is_year_start</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>purchase_Hour</th>
      <td>3</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>12</td>
    </tr>
    <tr>
      <th>purchase_Minute</th>
      <td>24</td>
      <td>1</td>
      <td>27</td>
      <td>55</td>
      <td>37</td>
    </tr>
    <tr>
      <th>purchase_Second</th>
      <td>51</td>
      <td>6</td>
      <td>39</td>
      <td>11</td>
      <td>26</td>
    </tr>
    <tr>
      <th>purchase_Elapsed</th>
      <td>1488338691</td>
      <td>1488366066</td>
      <td>1488367659</td>
      <td>1488369311</td>
      <td>1488371846</td>
    </tr>
    <tr>
      <th>purchased_on_weekend</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>purchased_on_weekday</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>month_diff</th>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
    </tr>
    <tr>
      <th>purchase_date_successive_diff</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>price</th>
      <td>46.4</td>
      <td>174.01</td>
      <td>inf</td>
      <td>inf</td>
      <td>550</td>
    </tr>
    <tr>
      <th>Christmas_Day_2017</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mothers_Day_2017</th>
      <td>94</td>
      <td>94</td>
      <td>94</td>
      <td>94</td>
      <td>94</td>
    </tr>
    <tr>
      <th>fathers_day_2017</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Children_day_2017</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Valentine_Day_2017</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Black_Friday_2017</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mothers_Day_2018</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>duration</th>
      <td>1113.6</td>
      <td>4176.24</td>
      <td>1200</td>
      <td>132</td>
      <td>13200</td>
    </tr>
    <tr>
      <th>amount_month_ratio</th>
      <td>1.93333</td>
      <td>7.25042</td>
      <td>2.08333</td>
      <td>0.229167</td>
      <td>22.9167</td>
    </tr>
    <tr>
      <th>category_2_mean</th>
      <td>130.934</td>
      <td>130.934</td>
      <td>125.997</td>
      <td>130.934</td>
      <td>127.287</td>
    </tr>
    <tr>
      <th>category_2_min</th>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>category_2_max</th>
      <td>175626</td>
      <td>175626</td>
      <td>40000</td>
      <td>175626</td>
      <td>22610</td>
    </tr>
    <tr>
      <th>category_2_sum</th>
      <td>1.53191e+08</td>
      <td>1.53191e+08</td>
      <td>2.25018e+07</td>
      <td>1.53191e+08</td>
      <td>8.35804e+06</td>
    </tr>
    <tr>
      <th>category_3_mean</th>
      <td>93.4493</td>
      <td>93.4493</td>
      <td>102.429</td>
      <td>102.429</td>
      <td>93.4493</td>
    </tr>
    <tr>
      <th>category_3_min</th>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>category_3_max</th>
      <td>48713.6</td>
      <td>48713.6</td>
      <td>65056</td>
      <td>65056</td>
      <td>48713.6</td>
    </tr>
    <tr>
      <th>category_3_sum</th>
      <td>7.81402e+07</td>
      <td>7.81402e+07</td>
      <td>1.00192e+08</td>
      <td>1.00192e+08</td>
      <td>7.81402e+07</td>
    </tr>
    <tr>
      <th>category_4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>merchant_address_id</th>
      <td>M_ID_c03b62d83d_80_37_69_9_1</td>
      <td>M_ID_2445d76702_307_19_0_9_1</td>
      <td>M_ID_b16ae63c45_307_19_0_14_4</td>
      <td>M_ID_b35d0757d1_80_37_69_9_1</td>
      <td>M_ID_b6b9b8ed67_178_29_0_24_2</td>
    </tr>
    <tr>
      <th>numerical_range</th>
      <td>173</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>737</td>
    </tr>
    <tr>
      <th>merchant_rating</th>
      <td>10</td>
      <td>12</td>
      <td>16</td>
      <td>15</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>

You can see that the features we compute earlier for merchants like `merchant_address_id`, `numerical_range`. Other merchant features didn't turn out to be useful appended to our transactions data. Now our transactions data has 56 features in total. `numerical_range` is feature interaction between `numerical_1` & `numerical_2`.

```python
hist_trans.shape, new_hist_trans.shape

((29112368, 56), (1963031, 56))
```

### Aggregate: Aggregate by card_id

Let's replace the `0` in our transactions data with `0.0001` to avoid numerical inconsistencies.

```python
hist_trans.loc[hist_trans['purchase_amount'] == 0, 'purchase_amount'] = 0.0001
```

Now we aggregate on the transactions data grouped by `card_id` as we have to calculate the loyalty score for each `card_id`. The following code is pruned after some experimentation and removal of aggregates which didn't have so much feature importance. Most of the aggregates are self-explanatory. We have some special aggregates grouped by category & we calculate `purchase_amount` aggregates on the grouped by category data.

```python
def aggregate_hist_trans(df):
    unique_cols = ['subsector_id', 'merchant_id', 'merchant_category_id', 'merchant_address_id']

    col_seas = ['purchase_Month','purchase_Week', 'purchase_Dayofweek', 'purchase_Day', 'purchase_Hour',
               'merchant_rating']

    aggs = {}
    for c in unique_cols:
        aggs[c] = ['nunique']
    for c in col_seas:
        aggs[c] = ['nunique', 'mean', 'min', 'max']

    aggs['purchase_amount'] = [('sum', 'sum'), ('pct_75', lambda x: np.percentile(x, q = 75)),
                               ('pct_25', lambda x: np.percentile(x, q = 25)), ('mean', 'mean'),
                               ('median', 'median'), ('max', 'max'), ('min', 'min'), ('var', 'var'),
                               ('skew', 'skew'), ('head_sum', head_sum), ('head_max', head_max),
                               ('tail_sum', tail_sum), ('tail_max', tail_max), ('gmean', scipy.stats.gmean ),
                                ('hmean', scipy.stats.hmean)]
    aggs['installments'] = [('sum', 'sum'), ('pct_75', lambda x: np.percentile(x, q = 75)),
                               ('pct_25', lambda x: np.percentile(x, q = 25)), ('mean', 'mean'),
                               ('median', 'median'), ('max', 'max'), ('min', 'min'), ('var', 'var'),
                               ('skew', 'skew'), ('head_sum', head_sum), ('head_max', head_max),
                               ('tail_sum', tail_sum), ('tail_max', tail_max)]
    aggs['purchase_date'] = ['max','min']
    aggs['month_lag'] = ['max','min','mean','var','skew']
    aggs['month_diff'] = ['max','min','mean','var','skew']
    aggs['authorized_flag'] = ['mean']
    aggs['purchased_on_weekend'] = ['mean'] # overwrite
    aggs['purchase_Dayofweek'] = ['mean'] # overwrite
    aggs['purchase_Day'] = ['nunique', 'mean', 'min'] # overwrite
    aggs['category_1'] = ['mean']
    aggs['category_2'] = ['mean']
    aggs['category_3'] = ['mean']
    aggs['category_4'] = ['mean']
    aggs['numerical_range'] = ['mean', 'min', 'max', 'skew']
    aggs['card_id'] = ['size','count']
    aggs['price'] = ['sum','mean','max','min','var', 'skew']
    aggs['Christmas_Day_2017'] = ['mean']
    aggs['Mothers_Day_2017'] = ['mean']
    aggs['fathers_day_2017'] = ['mean']
    aggs['Children_day_2017'] = ['mean']
    aggs['Valentine_Day_2017'] = ['mean']
    aggs['Black_Friday_2017'] = ['mean']
    aggs['Mothers_Day_2018'] = ['mean']
    aggs['duration']=['mean','min','max','var','skew']
    aggs['amount_month_ratio']=['mean','min','max','var','skew']

    #exta
    aggs['purchase_date_successive_diff'] = ['mean']

#     aggs['purchase_date_successive_diff'] = ['mean', 'median', 'max', 'min', 'var', 'skew']
    for col in ['category_2','category_3']:
        df[col+'_mean'] = df.groupby([col])['purchase_amount'].transform('mean')
        df[col+'_min'] = df.groupby([col])['purchase_amount'].transform('min')
        df[col+'_max'] = df.groupby([col])['purchase_amount'].transform('max')
        df[col+'_sum'] = df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col+'_mean'] = ['mean']

    new_df = df.groupby(['card_id']).agg(aggs)
    new_df.columns = ['_'.join(col).strip() for col in new_df.columns.values]
    new_df.reset_index(inplace=True)
    other_df = (df.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))

    new_df = pd.merge(other_df, new_df, on='card_id', how='left')

    new_df['purchase_date_diff'] = (new_df['purchase_date_max'] - new_df['purchase_date_min']).dt.days
    new_df['purchase_date_average'] = new_df['purchase_date_diff']/new_df['card_id_size']
    new_df['purchase_date_uptonow'] = (datetime.datetime.today() - new_df['purchase_date_max']).dt.days
    new_df['purchase_date_uptomin'] = (datetime.datetime.today() - new_df['purchase_date_min']).dt.days
    return new_df
```

We are not calculating the same aggregates for both historical & new transactional data, as some of the features add little value and will be removed during feature selection.

```python
def aggregate_new_trans(df):
    unique_cols = ['subsector_id', 'merchant_id', 'merchant_category_id', 'merchant_address_id']

    col_seas = ['purchase_Month', 'purchase_Week', 'purchase_Dayofweek','purchase_Day', 'purchase_Hour',
                'merchant_rating']

    aggs = {}
    for c in unique_cols:
        aggs[c] = ['nunique']
    for c in col_seas:
        aggs[c] = ['nunique', 'mean', 'min', 'max']

    aggs['purchase_amount'] = [('sum', 'sum'), ('pct_75', lambda x: np.percentile(x, q = 75)),
                               ('pct_25', lambda x: np.percentile(x, q = 25)), ('mean', 'mean'),
                               ('median', 'median'), ('max', 'max'), ('min', 'min'), ('var', 'var'),
                               ('skew', 'skew'), ('head_sum', head_sum), ('head_max', head_max),
                               ('tail_sum', tail_sum), ('tail_max', tail_max), ('gmean', scipy.stats.gmean ),
                                ('hmean', scipy.stats.hmean)]
    aggs['installments'] = [('sum', 'sum'), ('pct_75', lambda x: np.percentile(x, q = 75)),
                               ('pct_25', lambda x: np.percentile(x, q = 25)), ('mean', 'mean'),
                               ('median', 'median'), ('max', 'max'), ('min', 'min'), ('var', 'var'),
                               ('skew', 'skew'), ('head_sum', head_sum), ('head_max', head_max),
                               ('tail_sum', tail_sum), ('tail_max', tail_max)]
    aggs['purchase_date'] = ['max','min']
    aggs['month_lag'] = ['max','min','mean','var','skew']
    aggs['month_diff'] = ['mean','var','skew']
    aggs['purchased_on_weekend'] = ['mean']
    aggs['purchase_Month'] = ['mean', 'min', 'max']
    aggs['purchase_Dayofweek'] = ['mean', 'min', 'max']
    aggs['category_1'] = ['mean']
    aggs['category_2'] = ['mean']
    aggs['category_3'] = ['mean']
    aggs['category_4'] = ['mean']
    aggs['numerical_range'] = ['mean', 'min', 'max', 'skew']
    aggs['card_id'] = ['size','count']
    aggs['price'] = ['mean','max','min','var']
    aggs['Christmas_Day_2017'] = ['mean']
    aggs['Children_day_2017'] = ['mean']
    aggs['Black_Friday_2017'] = ['mean']
    aggs['Mothers_Day_2018'] = ['mean']
    aggs['duration']=['mean','min','max','var','skew']
    aggs['amount_month_ratio']=['mean','min','max','var','skew']

    #extra
    aggs['purchase_date_successive_diff'] = ['mean']

#     aggs['purchase_date_successive_diff'] = ['mean', 'median', 'max', 'min', 'var', 'skew']
    for col in ['category_2','category_3']:
        df[col+'_mean'] = df.groupby([col])['purchase_amount'].transform('mean')
        df[col+'_min'] = df.groupby([col])['purchase_amount'].transform('min')
        df[col+'_max'] = df.groupby([col])['purchase_amount'].transform('max')
        df[col+'_sum'] = df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col+'_mean'] = ['mean']

    new_df = df.groupby(['card_id']).agg(aggs)
    new_df.columns = ['_'.join(col).strip() for col in new_df.columns.values]
    new_df.reset_index(inplace=True)
    other_df = (df.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))

    new_df = pd.merge(other_df, new_df, on='card_id', how='left')

    new_df['purchase_date_diff'] = (new_df['purchase_date_max'] - new_df['purchase_date_min']).dt.days
    new_df['purchase_date_average'] = new_df['purchase_date_diff']/new_df['card_id_size']
    new_df['purchase_date_uptonow'] = (datetime.datetime.today() - new_df['purchase_date_max']).dt.days
    new_df['purchase_date_uptomin'] = (datetime.datetime.today() - new_df['purchase_date_min']).dt.days
    return new_df
```

```python
%time hist_trans_agg = aggregate_hist_trans(hist_trans)

  CPU times: user 14min 28s, sys: 53.6 s, total: 15min 22s
  Wall time: 14min 13s

%time new_hist_trans_agg = aggregate_new_trans(new_hist_trans)

  CPU times: user 8min 44s, sys: 1.25 s, total: 8min 46s
  Wall time: 8min 16s

```

### Feature: Add exta interpreted columns on aggregates

Adding extra features on top of the aggregates. I know it's overwhelming but these are important. `card_id_size` is the number of transactions done by the card. Most of the features are self-explanatory.

```python
def add_extra_cols_on_agg(df):
    df['inverse_avg_transactions_per_day'] = df['purchase_date_diff']/df['card_id_size']
    df['repurchase_merchant_rate'] = df['transactions_count']/df['merchant_id_nunique']
    df['merchant_category_repurchase'] = df['merchant_category_id_nunique']/df['merchant_id_nunique']
    df['avg_spend_per_merchant'] = df['purchase_amount_sum']/df['merchant_id_nunique']
    df['avg_trans_per_merchant'] = df['transactions_count']/df['merchant_id_nunique']
    df['avg_spend_per_transaction'] = df['purchase_amount_sum']/df['transactions_count']
    return df
[hist_trans_agg, new_hist_trans_agg] = [add_extra_cols_on_agg(df) for df in [hist_trans_agg,
                                                                             new_hist_trans_agg]]
hist_trans_agg.to_feather('hist_trans_agg_beta')
new_hist_trans_agg.to_feather('new_hist_trans_agg_beta')
hist_trans_agg = feather.read_dataframe('hist_trans_agg_beta')
new_hist_trans_agg = feather.read_dataframe('new_hist_trans_agg_beta')
```

We now have 114, 108 features for old & new transactional aggregates data. We will adding some more features on top of this.

```python
hist_trans_agg.shape, new_hist_trans_agg.shape


    ((325540, 114), (290001, 108))

```

### Aggregate: Aggregate on categories

Some more aggregates on the categories. Pivot tables are a common approach when calculating aggregates grouped by more than one conditions. Here we are delving deep in to the finer categorical spends aggregates. Our `category_1` had a cardinality of 2 whereas `category_2` had a cardinality of 6. We will calculate aggregates grouped by each of those possible values for the category. We will later see that these aggregates don't add much feature importance, but this was a good exercise for me to flex my pandas skills :D

```python
def agg_on_cat(df, category, feature):
    temp_df = df.pivot_table(index='card_id', columns=category, aggfunc={feature: ['sum', 'mean']})
    cols = [category + '_{0[2]}_{0[0]}_{0[1]}'.format(col) for col in temp_df.columns.tolist()]
    temp_df.columns = cols
    return temp_df
def get_cat_agg(df):
    agg_df = agg_on_cat(df, 'category_1', 'purchase_amount')
    agg_df = pd.merge(agg_df, agg_on_cat(df, 'category_2', 'purchase_amount'), on='card_id', how='left')
    agg_df = pd.merge(agg_df, agg_on_cat(df, 'category_3', 'purchase_amount'), on='card_id', how='left')
    agg_df = pd.merge(agg_df, agg_on_cat(df, 'authorized_flag', 'purchase_amount'), on='card_id', how='left')
    return agg_df
%time hist_trans_agg_cat, new_hist_trans_agg_cat = [get_cat_agg(df) for df in [hist_trans, new_hist_trans]]
```

    CPU times: user 30.4 s, sys: 9.67 s, total: 40.1 s
    Wall time: 27.9 s

```python
hist_trans_agg_cat.shape, new_hist_trans_agg_cat.shape
```

    ((325540, 24), (290001, 22))

Let's have a look at our newly calculated aggregates

```python
new_hist_trans_agg_cat.head().T
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>card_id</th>
      <th>C_ID_00007093c1</th>
      <th>C_ID_0001238066</th>
      <th>C_ID_0001506ef0</th>
      <th>C_ID_0001793786</th>
      <th>C_ID_000183fdda</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>category_1_0_purchase_amount_mean</th>
      <td>-0.664262</td>
      <td>-0.564558</td>
      <td>-0.723677</td>
      <td>-0.007407</td>
      <td>-0.599162</td>
    </tr>
    <tr>
      <th>category_1_1_purchase_amount_mean</th>
      <td>NaN</td>
      <td>-0.650332</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>category_1_0_purchase_amount_sum</th>
      <td>-1.328524</td>
      <td>-13.549391</td>
      <td>-1.447354</td>
      <td>-0.229620</td>
      <td>-6.590778</td>
    </tr>
    <tr>
      <th>category_1_1_purchase_amount_sum</th>
      <td>NaN</td>
      <td>-1.300665</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>category_2_1.0_purchase_amount_mean</th>
      <td>-0.656749</td>
      <td>-0.580966</td>
      <td>NaN</td>
      <td>0.139747</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>category_2_2.0_purchase_amount_mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.344766</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>category_2_3.0_purchase_amount_mean</th>
      <td>-0.671775</td>
      <td>NaN</td>
      <td>-0.723677</td>
      <td>0.102887</td>
      <td>-0.599162</td>
    </tr>
    <tr>
      <th>category_2_4.0_purchase_amount_mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>category_2_5.0_purchase_amount_mean</th>
      <td>NaN</td>
      <td>-0.495945</td>
      <td>NaN</td>
      <td>-0.361628</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>category_2_1.0_purchase_amount_sum</th>
      <td>-0.656749</td>
      <td>-13.362220</td>
      <td>NaN</td>
      <td>2.375707</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>category_2_2.0_purchase_amount_sum</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2.758131</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>category_2_3.0_purchase_amount_sum</th>
      <td>-0.671775</td>
      <td>NaN</td>
      <td>-1.447354</td>
      <td>0.514433</td>
      <td>-6.590778</td>
    </tr>
    <tr>
      <th>category_2_4.0_purchase_amount_sum</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>category_2_5.0_purchase_amount_sum</th>
      <td>NaN</td>
      <td>-1.487835</td>
      <td>NaN</td>
      <td>-0.361628</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>category_3_0_purchase_amount_mean</th>
      <td>NaN</td>
      <td>-0.152008</td>
      <td>-0.723677</td>
      <td>-0.007407</td>
      <td>-0.107680</td>
    </tr>
    <tr>
      <th>category_3_1_purchase_amount_mean</th>
      <td>-0.664262</td>
      <td>-0.625781</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.696173</td>
    </tr>
    <tr>
      <th>category_3_2_purchase_amount_mean</th>
      <td>NaN</td>
      <td>-0.389160</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.576515</td>
    </tr>
    <tr>
      <th>category_3_0_purchase_amount_sum</th>
      <td>NaN</td>
      <td>-0.152008</td>
      <td>-1.447354</td>
      <td>-0.229620</td>
      <td>-0.107680</td>
    </tr>
    <tr>
      <th>category_3_1_purchase_amount_sum</th>
      <td>-1.328524</td>
      <td>-13.141406</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-4.177040</td>
    </tr>
    <tr>
      <th>category_3_2_purchase_amount_sum</th>
      <td>NaN</td>
      <td>-1.556641</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2.306059</td>
    </tr>
    <tr>
      <th>authorized_flag_1_purchase_amount_mean</th>
      <td>-0.664262</td>
      <td>-0.571156</td>
      <td>-0.723677</td>
      <td>-0.007407</td>
      <td>-0.599162</td>
    </tr>
    <tr>
      <th>authorized_flag_1_purchase_amount_sum</th>
      <td>-1.328524</td>
      <td>-14.850055</td>
      <td>-1.447354</td>
      <td>-0.229620</td>
      <td>-6.590778</td>
    </tr>
  </tbody>
</table>
</div>

```python
hist_trans_agg_cat.reset_index().to_feather('hist_trans_agg_cat')
new_hist_trans_agg_cat.reset_index().to_feather('new_hist_trans_agg_cat')
hist_trans_agg_cat = feather.read_dataframe('hist_trans_agg_cat')
new_hist_trans_agg_cat = feather.read_dataframe('new_hist_trans_agg_cat')
```

### Aggregate: Aggregate on month

More aggregates grouped by month, (`month_diff` is the months since the reference date, we want to add more weight to our recent transactions (i.e., new customers)) & calculate aggregates over the customer spending.

```python
def aggregate_per_month(history):
    grouped = history.groupby(['card_id', 'month_diff'])['purchase_amount']

    agg_func = {
            'purchase_amount': ['count', 'sum', 'max', 'mean'],
            }

    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_month_diff_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)

    return intermediate_group
# aggregate_per_month(new_hist_trans)
%time hist_trans_agg_month, new_hist_trans_agg_month = [aggregate_per_month(df) for df in [hist_trans, new_hist_trans]]

    CPU times: user 3min 36s, sys: 12.5 s, total: 3min 49s
    Wall time: 3min 29s
```

```python
hist_trans_agg_month.shape, new_hist_trans_agg_month.shape
    ((618851, 6), (431268, 6))
```

The new aggregates grouped by month & purchase amount for each card are:

```python
new_hist_trans_agg_month.head().T
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>card_id</th>
      <td>C_ID_00007093c1</td>
      <td>C_ID_0001238066</td>
      <td>C_ID_0001238066</td>
      <td>C_ID_0001506ef0</td>
      <td>C_ID_0001793786</td>
    </tr>
    <tr>
      <th>month_diff</th>
      <td>12</td>
      <td>11</td>
      <td>12</td>
      <td>12</td>
      <td>15</td>
    </tr>
    <tr>
      <th>purchase_amount_month_diff_count</th>
      <td>2</td>
      <td>5</td>
      <td>21</td>
      <td>2</td>
      <td>18</td>
    </tr>
    <tr>
      <th>purchase_amount_month_diff_sum</th>
      <td>110</td>
      <td>857.42</td>
      <td>2183.57</td>
      <td>30.92</td>
      <td>12340.8</td>
    </tr>
    <tr>
      <th>purchase_amount_month_diff_max</th>
      <td>60</td>
      <td>250</td>
      <td>444.94</td>
      <td>21</td>
      <td>2580</td>
    </tr>
    <tr>
      <th>purchase_amount_month_diff_mean</th>
      <td>55</td>
      <td>171.484</td>
      <td>103.98</td>
      <td>15.46</td>
      <td>685.599</td>
    </tr>
  </tbody>
</table>
</div>

### Feature: Reverse engineering observed date aka reference date

Let's load the train & test datasets to dataframes. 

```python
PATH = 'data/elo/'
train, test = [pd.read_csv(f'{PATH}{c}') for c in ['train.csv', 'test.csv']]
train.head().T
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>first_active_month</th>
      <td>2017-06</td>
      <td>2017-01</td>
      <td>2016-08</td>
      <td>2017-09</td>
      <td>2017-11</td>
    </tr>
    <tr>
      <th>card_id</th>
      <td>C_ID_92a2005557</td>
      <td>C_ID_3d0044924f</td>
      <td>C_ID_d639edf6cd</td>
      <td>C_ID_186d6a6901</td>
      <td>C_ID_cdbd2c0db2</td>
    </tr>
    <tr>
      <th>feature_1</th>
      <td>5</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>feature_2</th>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>feature_3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>target</th>
      <td>-0.820283</td>
      <td>0.392913</td>
      <td>0.688056</td>
      <td>0.142495</td>
      <td>-0.159749</td>
    </tr>
  </tbody>
</table>
</div>

```python
test.head().T
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>first_active_month</th>
      <td>2017-04</td>
      <td>2017-01</td>
      <td>2017-08</td>
      <td>2017-12</td>
      <td>2015-12</td>
    </tr>
    <tr>
      <th>card_id</th>
      <td>C_ID_0ab67a22ab</td>
      <td>C_ID_130fd0cbdd</td>
      <td>C_ID_b709037bc5</td>
      <td>C_ID_d27d835a9f</td>
      <td>C_ID_2b5e3df5c2</td>
    </tr>
    <tr>
      <th>feature_1</th>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>feature_2</th>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>feature_3</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
Most of the historical transactions tailed at month end & new transactions dataset started at month start. Let's start with an assumption that observed_date should be during the start fo the month.

We will calculate the latest spendings of the customer on historical transactions , & the earlies first month's spendings in the new transactions.

```python
last_hist_transaction = hist_trans.groupby('card_id').agg({'month_lag' : 'max', 'purchase_date' : 'max'}).reset_index()
last_hist_transaction.columns = ['card_id', 'hist_month_lag', 'hist_purchase_date']
first_new_transaction = new_hist_trans.groupby('card_id').agg({'month_lag' : 'min', 'purchase_date' : 'min'}).reset_index()
first_new_transaction.columns = ['card_id', 'new_month_lag', 'new_purchase_date']
last_hist_transaction['hist_purchase_date'] = pd.to_datetime(last_hist_transaction['hist_purchase_date'])
first_new_transaction['new_purchase_date'] = pd.to_datetime(first_new_transaction['new_purchase_date'])
last_hist_transaction.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>card_id</th>
      <th>hist_month_lag</th>
      <th>hist_purchase_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C_ID_00007093c1</td>
      <td>0</td>
      <td>2018-02-27 05:14:57</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C_ID_0001238066</td>
      <td>0</td>
      <td>2018-02-27 16:18:59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C_ID_0001506ef0</td>
      <td>0</td>
      <td>2018-02-17 12:33:56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C_ID_0001793786</td>
      <td>0</td>
      <td>2017-10-31 20:20:18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C_ID_000183fdda</td>
      <td>0</td>
      <td>2018-02-25 20:57:08</td>
    </tr>
  </tbody>
</table>
</div>

```python
first_new_transaction.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>card_id</th>
      <th>new_month_lag</th>
      <th>new_purchase_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C_ID_00007093c1</td>
      <td>2</td>
      <td>2018-04-03 11:13:35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C_ID_0001238066</td>
      <td>1</td>
      <td>2018-03-01 16:48:27</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C_ID_0001506ef0</td>
      <td>1</td>
      <td>2018-03-16 22:21:58</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C_ID_0001793786</td>
      <td>1</td>
      <td>2017-11-15 15:44:20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C_ID_000183fdda</td>
      <td>1</td>
      <td>2018-03-02 12:26:26</td>
    </tr>
  </tbody>
</table>
</div>

Our suspicions might be true. We already knew that month_lag is the number of months from the reference date. Let's add observation_date by offsetting the last transaction with month_lag for historical & new transactions.

```python
last_hist_transaction['observation_date'] = \
    last_hist_transaction.apply(lambda x: x['hist_purchase_date']  - pd.DateOffset(months=x['hist_month_lag']), axis=1)

first_new_transaction['observation_date'] = \
    first_new_transaction.apply(lambda x: x['new_purchase_date']  - pd.DateOffset(months=x['new_month_lag']-1), axis=1)
last_hist_transaction.head(5)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>card_id</th>
      <th>hist_month_lag</th>
      <th>hist_purchase_date</th>
      <th>observation_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C_ID_00007093c1</td>
      <td>0</td>
      <td>2018-02-27 05:14:57</td>
      <td>2018-02-27 05:14:57</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C_ID_0001238066</td>
      <td>0</td>
      <td>2018-02-27 16:18:59</td>
      <td>2018-02-27 16:18:59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C_ID_0001506ef0</td>
      <td>0</td>
      <td>2018-02-17 12:33:56</td>
      <td>2018-02-17 12:33:56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C_ID_0001793786</td>
      <td>0</td>
      <td>2017-10-31 20:20:18</td>
      <td>2017-10-31 20:20:18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C_ID_000183fdda</td>
      <td>0</td>
      <td>2018-02-25 20:57:08</td>
      <td>2018-02-25 20:57:08</td>
    </tr>
  </tbody>
</table>
</div>

```python
first_new_transaction.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>card_id</th>
      <th>new_month_lag</th>
      <th>new_purchase_date</th>
      <th>observation_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C_ID_00007093c1</td>
      <td>2</td>
      <td>2018-04-03 11:13:35</td>
      <td>2018-03-03 11:13:35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C_ID_0001238066</td>
      <td>1</td>
      <td>2018-03-01 16:48:27</td>
      <td>2018-03-01 16:48:27</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C_ID_0001506ef0</td>
      <td>1</td>
      <td>2018-03-16 22:21:58</td>
      <td>2018-03-16 22:21:58</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C_ID_0001793786</td>
      <td>1</td>
      <td>2017-11-15 15:44:20</td>
      <td>2017-11-15 15:44:20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C_ID_000183fdda</td>
      <td>1</td>
      <td>2018-03-02 12:26:26</td>
      <td>2018-03-02 12:26:26</td>
    </tr>
  </tbody>
</table>
</div>

Let's round off the `observation_date` to month and remove the time info.

```python
last_hist_transaction['observation_date'] = last_hist_transaction['observation_date'].dt.to_period('M').dt.to_timestamp() + pd.DateOffset(months=1)
first_new_transaction['observation_date'] = first_new_transaction['observation_date'].dt.to_period('M').dt.to_timestamp()
last_hist_transaction.head()
```

After rounding off our data looks like this:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>card_id</th>
      <th>hist_month_lag</th>
      <th>hist_purchase_date</th>
      <th>observation_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C_ID_00007093c1</td>
      <td>0</td>
      <td>2018-02-27 05:14:57</td>
      <td>2018-03-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C_ID_0001238066</td>
      <td>0</td>
      <td>2018-02-27 16:18:59</td>
      <td>2018-03-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C_ID_0001506ef0</td>
      <td>0</td>
      <td>2018-02-17 12:33:56</td>
      <td>2018-03-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C_ID_0001793786</td>
      <td>0</td>
      <td>2017-10-31 20:20:18</td>
      <td>2017-11-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C_ID_000183fdda</td>
      <td>0</td>
      <td>2018-02-25 20:57:08</td>
      <td>2018-03-01</td>
    </tr>
  </tbody>
</table>
</div>

```python
first_new_transaction.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>card_id</th>
      <th>new_month_lag</th>
      <th>new_purchase_date</th>
      <th>observation_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C_ID_00007093c1</td>
      <td>2</td>
      <td>2018-04-03 11:13:35</td>
      <td>2018-03-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C_ID_0001238066</td>
      <td>1</td>
      <td>2018-03-01 16:48:27</td>
      <td>2018-03-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C_ID_0001506ef0</td>
      <td>1</td>
      <td>2018-03-16 22:21:58</td>
      <td>2018-03-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C_ID_0001793786</td>
      <td>1</td>
      <td>2017-11-15 15:44:20</td>
      <td>2017-11-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C_ID_000183fdda</td>
      <td>1</td>
      <td>2018-03-02 12:26:26</td>
      <td>2018-03-01</td>
    </tr>
  </tbody>
</table>
</div>

Let's validate if our calculation is true by comparing the observed_date between historical & new transactions. This should be true for all card_ids.

```python
first_new_transaction.drop(['new_month_lag', 'new_purchase_date'], axis=1, inplace=True)
last_hist_transaction.drop(['hist_month_lag', 'hist_purchase_date'], axis=1, inplace=True)
validate = last_hist_transaction.merge(first_new_transaction, on = 'card_id')
all(validate['observation_date_x'] == validate['observation_date_y'])

    True

```

They indeed are same! We will redo some of our features based on observation_date later.

### Aggregates: Merge train & test with new & old transactions history

```python
hist_trans_agg.shape, hist_trans_agg_cat.shape, last_hist_transaction.shape, hist_trans_agg_month.shape

    ((325540, 114), (325540, 25), (325540, 2), (618851, 6))
```

Merging all our aggregates & features with train & test data frames

```python
def join_dfs(left, right, left_on, right_on=None, suffix='_old'):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, suffixes=("", suffix))
```

Our train & test datasets shape:

```python
train.shape, test.shape
    ((201917, 6), (123623, 5))

```

Let's go ahead and join all our aggregates on the train & test datasets.

```python
train_df = join_dfs(train, new_hist_trans_agg, left_on='card_id')
train_df = join_dfs(train_df, hist_trans_agg, left_on='card_id', suffix='_old')
# train_df = join_dfs(train_df, hist_trans_agg_month, left_on='card_id', suffix='_old')
train_df = join_dfs(train_df, hist_trans_agg_cat, left_on='card_id', suffix='_old')
train_df = join_dfs(train_df, last_hist_transaction, left_on='card_id', suffix='_old')
# train_df = join_dfs(train_df, new_hist_trans_agg_month, left_on='card_id')
train_df = join_dfs(train_df, new_hist_trans_agg_cat, left_on='card_id')
train_df = join_dfs(train_df, first_new_transaction, left_on='card_id')
test_df = join_dfs(test, new_hist_trans_agg, left_on='card_id')
test_df = join_dfs(test_df, first_new_transaction, left_on='card_id')
# test_df = join_dfs(test_df, new_hist_trans_agg_month, left_on='card_id')
test_df = join_dfs(test_df, new_hist_trans_agg_cat, left_on='card_id')
# test_df = join_dfs(test_df, hist_trans_agg_month, left_on='card_id', suffix='_old')
test_df = join_dfs(test_df, hist_trans_agg_cat, left_on='card_id', suffix='_old')
test_df = join_dfs(test_df, hist_trans_agg, left_on='card_id', suffix='_old')
test_df = join_dfs(test_df, last_hist_transaction, left_on='card_id', suffix='_old')
test_df.shape, train_df.shape

    ((123623, 273), (201917, 274))
```

### Feature: Adding features based on observed_date

Adding feature interactions between `feature_1, feature_2 & feature_3` of the card & the time elapsed since the observed date aka reference date.

```python
def add_days_feature_interaction(df):
    # to datetime
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['observation_date_old'] = pd.to_datetime(df['observation_date_old'])
    # datetime features
    df['quarter'] = df['first_active_month'].dt.quarter
#     df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
    df['observed_elapsed_time'] = (df['observation_date_old'] - df['first_active_month']).dt.days
    df['days_feature1'] = df['observed_elapsed_time'] * df['feature_1']
    df['days_feature2'] = df['observed_elapsed_time'] * df['feature_2']
    df['days_feature3'] = df['observed_elapsed_time'] * df['feature_3']

    df['days_feature1_ratio'] = df['feature_1'] / df['observed_elapsed_time']
    df['days_feature2_ratio'] = df['feature_2'] / df['observed_elapsed_time']
    df['days_feature3_ratio'] = df['feature_3'] / df['observed_elapsed_time']

    df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']
    df['feature_mean'] = df['feature_sum']/3
    df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
    df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
    df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)
    return df
train_df, test_df = [add_days_feature_interaction(d) for d in [train_df, test_df]]
```

### Feature: Add features based on old & new transactions

```python
train_df.shape, test_df.shape
    ((201917, 286), (123623, 285))
```

We now have around 285 features adding all our aggregates. Can we squeeze in more? Let's add some feature interactions with old & new transactions data. I also added new features like `CLV` (my own Customer Lifetime Value formula) which is basically feature interaction between number of transactions, age & sum of the transactions.

```python
for df in [train_df, test_df]:
    df['card_id_total'] = df['card_id_size']+df['card_id_size_old']
    df['card_id_count_total'] = df['card_id_count']+df['card_id_count_old']
    df['card_id_count_ratio'] = df['card_id_count'] / df['card_id_count_old']
    df['purchase_amount_total'] = df['purchase_amount_sum_old']+df['purchase_amount_sum']
    df['purchase_amount_total_mean'] = df['purchase_amount_mean']+df['purchase_amount_mean_old']
    df['purchase_amount_total_max'] = df['purchase_amount_max']+df['purchase_amount_max_old']
    df['purchase_amount_total_min'] = df['purchase_amount_min']+df['purchase_amount_min_old']
    df['purchase_amount_sum_ratio'] = df['purchase_amount_sum'] / df['purchase_amount_sum_old']
    df['hist_first_buy'] = (df['purchase_date_min_old'] - df['first_active_month']).dt.days
    df['new_first_buy'] = (df['purchase_date_min'] - df['first_active_month']).dt.days
    df['hist_last_buy'] = (df['purchase_date_max_old'] - df['first_active_month']).dt.days
    df['new_last_buy'] = (df['purchase_date_max'] - df['first_active_month']).dt.days
    df['avg_spend_per_transaction'] = df['purchase_amount_total']/df['card_id_total']
    df['purchased_before_issue'] = df['hist_first_buy'] < 0
    df['month_diff_mean_total'] = df['month_diff_mean']+df['month_diff_mean_old']
    df['month_diff_ratio'] = df['month_diff_mean']/df['month_diff_mean_old']
    df['month_lag_mean_total'] = df['month_lag_mean']+df['month_lag_mean_old']
    df['month_lag_max_total'] = df['month_lag_max']+df['month_lag_max_old']
    df['month_lag_min_total'] = df['month_lag_min']+df['month_lag_min_old']
    df['category_1_mean_total'] = df['category_1_mean']+df['category_1_mean_old']
    df['category_4_mean_total'] = df['category_4_mean']+df['category_4_mean_old']
    df['category_4_mean_ratio'] = df['category_4_mean']/df['category_4_mean_old']
    df['category_1_mean_ratio'] = df['category_1_mean']/df['category_1_mean_old']
    df['numerical_range_mean_total'] = df['numerical_range_mean']+df['numerical_range_mean_old']
    df['numerical_range_mean_ratio'] = df['numerical_range_mean']/df['numerical_range_mean_old']
    df['merchant_rating_mean_ratio'] = df['merchant_rating_mean']/df['merchant_rating_mean_old']
    df['installments_total'] = df['installments_sum']+df['installments_sum_old']
    df['installments_mean_total'] = df['installments_mean']+df['installments_mean_old']
    df['installments_max_total'] = df['installments_max']+df['installments_max_old']
    df['installments_ratio'] = df['installments_sum']/df['installments_sum_old']
    df['price_total'] = df['purchase_amount_total'] / df['installments_total']
    df['price_mean'] = df['purchase_amount_mean'] / df['installments_mean']
    df['price_max'] = df['purchase_amount_max'] / df['installments_max']
    df['duration_mean_total'] = df['duration_mean']+df['duration_mean_old']
    df['duration_min_total'] = df['duration_min']+df['duration_min_old']
    df['duration_max_total'] = df['duration_max']+df['duration_max_old']
    df['amount_month_ratio_mean_total']=df['amount_month_ratio_mean']+df['amount_month_ratio_mean_old']
    df['amount_month_ratio_min_total']=df['amount_month_ratio_min']+df['amount_month_ratio_min_old']
    df['amount_month_ratio_max_total']=df['amount_month_ratio_max']+df['amount_month_ratio_max_old']
    df['CLV'] = df['card_id_count'] * df['purchase_amount_sum'] / df['month_diff_mean']
    df['CLV_old'] = df['card_id_count_old'] * df['purchase_amount_sum_old'] / df['month_diff_mean_old']
    df['CLV_ratio'] = df['CLV'] / df['CLV_old']
    df['category_1_0_purchase_amount_mean_total'] = df['category_1_0_purchase_amount_mean'] + df['category_1_0_purchase_amount_mean_old']
    df['category_1_1_purchase_amount_mean_total'] = df['category_1_1_purchase_amount_mean'] + df['category_1_1_purchase_amount_mean_old']
    df['category_1_0_purchase_amount_sum_total'] = df['category_1_0_purchase_amount_sum'] + df['category_1_0_purchase_amount_sum_old']
    df['category_1_1_purchase_amount_sum_total'] = df['category_1_1_purchase_amount_sum'] + df['category_1_1_purchase_amount_sum_old']
    df['category_2_1.0_purchase_amount_mean_total'] = df['category_2_1.0_purchase_amount_mean'] + df['category_2_1.0_purchase_amount_mean_old']
    df['category_2_2.0_purchase_amount_mean_total'] = df['category_2_2.0_purchase_amount_mean'] + df['category_2_2.0_purchase_amount_mean_old']
    df['category_2_3.0_purchase_amount_mean_total'] = df['category_2_3.0_purchase_amount_mean'] + df['category_2_3.0_purchase_amount_mean_old']
    df['category_2_4.0_purchase_amount_mean_total'] = df['category_2_4.0_purchase_amount_mean'] + df['category_2_4.0_purchase_amount_mean_old']
    df['category_2_5.0_purchase_amount_mean_total'] = df['category_2_5.0_purchase_amount_mean'] + df['category_2_5.0_purchase_amount_mean_old']
    df['category_2_1.0_purchase_amount_sum_total'] = df['category_2_1.0_purchase_amount_sum'] + df['category_2_1.0_purchase_amount_sum_old']
    df['category_2_2.0_purchase_amount_sum_total'] = df['category_2_2.0_purchase_amount_sum'] + df['category_2_2.0_purchase_amount_sum_old']
    df['category_2_3.0_purchase_amount_sum_total'] = df['category_2_3.0_purchase_amount_sum'] + df['category_2_3.0_purchase_amount_sum_old']
    df['category_2_4.0_purchase_amount_sum_total'] = df['category_2_4.0_purchase_amount_sum'] + df['category_2_4.0_purchase_amount_sum_old']
    df['category_2_5.0_purchase_amount_sum_total'] = df['category_2_5.0_purchase_amount_sum'] + df['category_2_5.0_purchase_amount_sum_old']
    df['category_3_0_purchase_amount_mean_total'] = df['category_3_0_purchase_amount_mean'] + df['category_3_0_purchase_amount_mean_old']
    df['category_3_1_purchase_amount_mean_total'] = df['category_3_1_purchase_amount_mean'] + df['category_3_1_purchase_amount_mean_old']
    df['category_3_2_purchase_amount_mean_total'] = df['category_3_2_purchase_amount_mean'] + df['category_3_2_purchase_amount_mean_old']
    df['category_3_0_purchase_amount_sum_total'] = df['category_3_0_purchase_amount_sum'] + df['category_3_0_purchase_amount_sum_old']
    df['category_3_1_purchase_amount_sum_total'] = df['category_3_1_purchase_amount_sum'] + df['category_3_1_purchase_amount_sum_old']
    df['category_3_2_purchase_amount_sum_total'] = df['category_3_2_purchase_amount_sum'] + df['category_3_2_purchase_amount_sum_old']
    df['category_1_0_purchase_amount_mean_ratio']  = df['category_1_0_purchase_amount_mean'] / df['category_1_0_purchase_amount_mean_old']
    df['category_1_1_purchase_amount_mean_ratio']  = df['category_1_1_purchase_amount_mean'] / df['category_1_1_purchase_amount_mean_old']
    df['category_1_0_purchase_amount_sum_ratio']  = df['category_1_0_purchase_amount_sum'] / df['category_1_0_purchase_amount_sum_old']
    df['category_1_1_purchase_amount_sum_ratio']  = df['category_1_1_purchase_amount_sum'] / df['category_1_1_purchase_amount_sum_old']
    df['category_2_1.0_purchase_amount_mean_ratio']  = df['category_2_1.0_purchase_amount_mean'] / df['category_2_1.0_purchase_amount_mean_old']
    df['category_2_2.0_purchase_amount_mean_ratio']  = df['category_2_2.0_purchase_amount_mean'] / df['category_2_2.0_purchase_amount_mean_old']
    df['category_2_3.0_purchase_amount_mean_ratio']  = df['category_2_3.0_purchase_amount_mean'] / df['category_2_3.0_purchase_amount_mean_old']
    df['category_2_4.0_purchase_amount_mean_ratio']  = df['category_2_4.0_purchase_amount_mean'] / df['category_2_4.0_purchase_amount_mean_old']
    df['category_2_5.0_purchase_amount_mean_ratio']  = df['category_2_5.0_purchase_amount_mean'] / df['category_2_5.0_purchase_amount_mean_old']
    df['category_2_1.0_purchase_amount_sum_ratio']  = df['category_2_1.0_purchase_amount_sum'] / df['category_2_1.0_purchase_amount_sum_old']
    df['category_2_2.0_purchase_amount_sum_ratio']  = df['category_2_2.0_purchase_amount_sum'] / df['category_2_2.0_purchase_amount_sum_old']
    df['category_2_3.0_purchase_amount_sum_ratio']  = df['category_2_3.0_purchase_amount_sum'] / df['category_2_3.0_purchase_amount_sum_old']
    df['category_2_4.0_purchase_amount_sum_ratio']  = df['category_2_4.0_purchase_amount_sum'] / df['category_2_4.0_purchase_amount_sum_old']
    df['category_2_5.0_purchase_amount_sum_ratio']  = df['category_2_5.0_purchase_amount_sum'] / df['category_2_5.0_purchase_amount_sum_old']
    df['category_3_0_purchase_amount_mean_ratio']  = df['category_3_0_purchase_amount_mean'] / df['category_3_0_purchase_amount_mean_old']
    df['category_3_1_purchase_amount_mean_ratio']  = df['category_3_1_purchase_amount_mean'] / df['category_3_1_purchase_amount_mean_old']
    df['category_3_2_purchase_amount_mean_ratio']  = df['category_3_2_purchase_amount_mean'] / df['category_3_2_purchase_amount_mean_old']
    df['category_3_0_purchase_amount_sum_ratio']  = df['category_3_0_purchase_amount_sum'] / df['category_3_0_purchase_amount_sum_old']
    df['category_3_1_purchase_amount_sum_ratio']  = df['category_3_1_purchase_amount_sum'] / df['category_3_1_purchase_amount_sum_old']
    df['category_3_2_purchase_amount_sum_ratio']  = df['category_3_2_purchase_amount_sum'] / df['category_3_2_purchase_amount_sum_old']
    df['purchase_amount_sum_total'] = df['purchase_amount_sum'] + df['purchase_amount_sum_old']
    df['purchase_amount_sum_ratio'] = df['purchase_amount_sum'] / df['purchase_amount_sum_old']
    df['purchase_amount_pct_75_ratio'] = df['purchase_amount_pct_75'] / df['purchase_amount_pct_75_old']
    df['purchase_amount_pct_25_ratio'] = df['purchase_amount_pct_25'] / df['purchase_amount_pct_25_old']
    df['purchase_amount_inter_quartile'] = df['purchase_amount_pct_75'] - df['purchase_amount_pct_25']
    df['purchase_amount_inter_quartile_old'] = df['purchase_amount_pct_75_old'] - df['purchase_amount_pct_25_old']
    df['purchase_amount_inter_quartile_ratio'] = df['purchase_amount_inter_quartile'] / df['purchase_amount_inter_quartile_old']
    df['purchase_amount_mean_total'] = df['purchase_amount_mean'] + df['purchase_amount_mean_old']
    df['purchase_amount_mean_ratio'] = df['purchase_amount_mean'] / df['purchase_amount_mean_old']
    df['purchase_amount_median_ratio'] = df['purchase_amount_median'] / df['purchase_amount_median_old']
    df['purchase_amount_max_total'] = df['purchase_amount_max'] + df['purchase_amount_max_old']
    df['purchase_amount_min_total'] = df['purchase_amount_min'] + df['purchase_amount_min_old']
    df['purchase_amount_skew_ratio'] = df['purchase_amount_skew'] / df['purchase_amount_skew_old']
    df['purchase_amount_before_after_ratio'] = df['purchase_amount_head_sum'] / df['purchase_amount_tail_sum_old']
    df['purchase_amount_first_last_diff'] = df['purchase_amount_tail_sum'] - df['purchase_amount_head_max_old']
    df['purchase_amount_fi_last_old_total'] = df['purchase_amount_tail_sum'] -  df['purchase_amount_head_sum']
    df['purchase_amount_fi_last_new_total'] = df['purchase_amount_tail_sum_old'] -  df['purchase_amount_head_max_old']
    df['purchase_amount_gmean_total'] = df['purchase_amount_gmean'] + df['purchase_amount_gmean_old']
    df['purchase_amount_hmean_total'] = df['purchase_amount_hmean'] + df['purchase_amount_hmean_old']
    df['purchase_amount_gmean_ratio'] = df['purchase_amount_gmean'] / df['purchase_amount_gmean_old']
    df['purchase_amount_hmean_ratio'] = df['purchase_amount_hmean'] / df['purchase_amount_hmean_old']

```

### Feature: Redo some date features with observed time

Initially we calculated `purchase_date_uptonow` and `purchase_date_uptomin` as the age of the max and minimum transaction amount till date by the customer. This is a dwindling number as the result will change as time progresses. Let's calculate it against some fixed date. What about our `observation_date` aka reference date. This yielded some boost in the leaderboard personally for me.

```python
for df in [train_df, test_df]:
    df['purchase_date_uptonow'] = (df['observation_date_old'] - df['purchase_date_max']).dt.days
    df['purchase_date_uptomin'] = (df['observation_date_old'] - df['purchase_date_min']).dt.days
    df['purchase_date_uptonow_old'] = (df['observation_date_old'] - df['purchase_date_max_old']).dt.days
    df['purchase_date_uptomin_old'] = (df['observation_date_old'] - df['purchase_date_min_old']).dt.days
#     df.drop(['days_since_last_transaction', 'days_since_last_transaction_old'], inplace=True, axis=1)
train_df.shape, test_df.shape
    ((201917, 386), (123623, 384))

```

### Feature: Mark the outliers

Add an additional feature which flags the outliers. Let's plot the distribution of `target` using a histogram.
![image.png](attachment:image.png)

```python
train_df['outliers'] = 0
train_df.loc[train_df['target'] < -30, 'outliers'] = 1
train_df['outliers'].value_counts()

    0    199710
    1      2207
    Name: outliers, dtype: int64

```

We have around 2207 outliers in our dataset which is around 1.105% of the dataset. This is a very skewed distribution and the outliers penalise our metric RMSE heavily as they have huge variance with non-outliers. It's important to retain them in our dataset instead of deleting them. Later we will learn how to build models with and without outliers and optimise both models to improve our prediction accuracy.

### Feature: Redo features based on new purchase amount

```python

new_hist_trans_purchase_aggs = new_hist_trans.groupby('card_id').agg({
    'purchase_amount_new': [('sum', 'sum'), ('pct_75', lambda x: np.percentile(x, q = 75)),
                            ('pct_25', lambda x: np.percentile(x, q = 25)), ('mean', 'mean'),
                            ('median', 'median'), ('max', 'max'), ('min', 'min'), ('var', 'var'),
                            ('skew', 'skew'), ('head_sum', head_sum), ('head_max', head_max),
                            ('tail_sum', tail_sum), ('tail_max', tail_max), ('gmean', scipy.stats.gmean ),
                            ('hmean', scipy.stats.hmean)]
})
new_hist_trans_purchase_aggs.columns = [ '_'.join(c).strip() for c in new_hist_trans_purchase_aggs.columns.values]
hist_trans.loc[hist_trans['purchase_amount_new'] == 0, 'purchase_amount_new'] = 0.0001
hist_trans_purchase_aggs = hist_trans.groupby('card_id').agg({
        'purchase_amount_new': [('sum', 'sum'), ('pct_75', lambda x: np.percentile(x, q = 75)),
                                ('pct_25', lambda x: np.percentile(x, q = 25)), ('mean', 'mean'),
                                ('median', 'median'), ('max', 'max'), ('min', 'min'), ('var', 'var'),
                                ('skew', 'skew'), ('head_sum', head_sum), ('head_max', head_max),
                                ('tail_sum', tail_sum), ('tail_max', tail_max), ('gmean', scipy.stats.gmean ),
                                ('hmean', scipy.stats.hmean)]
    })
hist_trans_purchase_aggs.columns = [ '_'.join(c).strip() for c in hist_trans_purchase_aggs.columns.values]
hist_trans_purchase_aggs.reset_index().to_feather('hist_trans_purchase_aggs_alpha')
new_hist_trans_purchase_aggs.reset_index().to_feather('new_hist_trans_purchase_aggs_alpha')
train_df.shape, test_df.shape, hist_trans_purchase_aggs.shape, new_hist_trans_purchase_aggs.shape
    ((201917, 357), (123623, 355), (325540, 15), (290001, 15))
```

We now have 355 features so far. Let's export the dataframe to feather and store it to disk:

```python
train_df.to_feather('train_df_alpha')
test_df.to_feather('test_df_alpha')
```

## Feature Selection

We have 355 features on train & test datasets. Thats' a quite huge number. Also feeding all of them to the model made the LB score worse. This is because we are either adding too much noise the data or unnecessarily weighing wrong features.

Feature selection is a very important step in Data science. There exists lots of approaches for feature selection:

1. Prune features which tail at the feature importance ranking after certain percentile.

2. Remove features which are highly correlated by calculating VarianceThreshold.

3. Use hierarchical clustering techniques like constructing a Dendrogram to eliminate correlated features.

4. Boruta algorithm: Shuffle one column randomly and see if it improves the score.

5. Forward Feature Selection (FFS): Add features one after the other if they improve the RMSE score.

After trying all the above techniques I stuck with FFS as it was more reliable than others though it was taking lot of compute time & power. This is because of too little representation & very high penalty on RMSE by the outliers in this dataset.

What I've done:

1. Established a base line score by training a LGBM regressor on 3 base columns.

2. Added features one by one to the regressor to compute the score. Remove the feature if it detoriates the base score. If it improves, add the feature to my base columns and update the base score to the current score.

Here is our training function:

```python
def lgb_train_fn(train_df, target, trn_cols,  n_fold):
    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=4590)
    # initialise out of fold preds to 0s.
    oof = np.zeros(len(train_df))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df,train_df['outliers'].values)):
        trn_data = lgb.Dataset(train_df.iloc[trn_idx][trn_cols], label=target.iloc[trn_idx])
        val_data = lgb.Dataset(train_df.iloc[val_idx][trn_cols], label=target.iloc[val_idx])

        num_round = 10000
        clf = lgb.train(lgb_param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=0, early_stopping_rounds = 200)
        oof[val_idx] = clf.predict(train_df.iloc[val_idx][trn_cols], num_iteration=clf.best_iteration)

    print(np.sqrt(mean_squared_error(oof, target)), 'CV score')
    return np.sqrt(mean_squared_error(oof, target))
```

Here is our selection loop:

```python
for c in cols_to_add:
    lgb_cols = final_cols + [c]
    print(len(lgb_cols), 'lg_cols', c)
    score = lgb_train_fn(x, y, lgb_cols, 5)
    delta = base_score - score
    fe_d[c] = delta
    if delta > 0:
        base_score = score
        selected_cols.append(c)
        print('Selected cols', c)
        print('Selected col delta', delta)
        print(' score with col', score)
        np.save('selecte_cols_extra', selected_cols)
        final_cols = final_cols + [c]
```

After running the above pieces of code for about 1.5 hours we are left with 180 features out of our initial 355 features.

## Model training

### Cross validation data set

Cross-validation is a statistical method used to estimate the skill of machine learning models.

It is commonly used in applied machine learning to compare and select a model for a given predictive modeling problem because it is easy to understand, easy to implement, and results in skill estimates that generally have a lower bias than other methods.

We will be using Stratified k fold for sampling our data. We have seen previously that outliers are about ~1% of our dataset and they are heavily skewing the score. While training our model, it is important to keep the outlier distribution even so that we get the best predictions. Stratified sampling makes sure that data from all strata is represented proportionately.

[StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold) is a variation of k-fold which returns stratified folds: each set contains approximately the same percentage of samples of each target class as the complete set.

### LGBM model

I used the optimised & finetuned hyperparameter from one of the public kernels. We can arrive at the same numbers by using grid search or leveraging Bayesian optimisation techniques using [Hyperopt](https://hyperopt.github.io/hyperopt/).

```python
%%time
n_fold =5
param = {
        'task': 'train',
        'boosting': 'goss',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'subsample': 0.9855232997390695,
        'max_depth': 7,
        'top_rate': 0.9064148448434349,
        'num_leaves': 123,
        'min_child_weight': 41.9612869171337,
        'other_rate': 0.0721768246018207,
        'reg_alpha': 9.677537745007898,
        'colsample_bytree': 0.5665320670155495,
        'min_split_gain': 9.820197773625843,
        'reg_lambda': 8.2532317400459,
        'min_data_in_leaf': 21,
        'verbose': -1,
        'seed':int(2**n_fold),
        'bagging_seed':int(2**n_fold),
        'drop_seed':int(2**n_fold)
        }
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=4590)
oof = np.zeros(len(train_df2))
predictions = np.zeros(len(test_df2))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df2,train_df2['outliers'].values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(train_df2.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_df2.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    oof[val_idx] = clf.predict(train_df2.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = df_train_columns
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test_df2[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits

print(np.sqrt(mean_squared_error(oof, target)), 'CV score')
np.save('oof_lgbm', oof)
np.save('predictions_lgbm', predictions)
3.644042133990217 CV score
```

Let's store the predictions on test & train set to a pickle file for stacking & blending with other models later.

### XGBM Model

```python

xgb_params = {
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': True,
    'booster': 'gbtree',
    'n_jobs': 4,
    'n_estimators': 20000,
    'grow_policy': 'lossguide',
    'max_depth': 12,
    'seed': 538,
    'colsample_bylevel': 0.9,
    'colsample_bytree': 0.8,
    'gamma': 0.0001,
    'learning_rate': 0.006150886706231842,
    'max_bin': 128,
    'max_leaves': 47,
    'min_child_weight': 40,
    'reg_alpha': 10.0,
    'reg_lambda': 10.0,
    'silent': True,
    'eta': 0.005,
    'subsample': 0.9
}
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df2,train_df2['outliers'].values)):
    print('-')
    print("Fold {}".format(fold_ + 1))
    trn_data = xgb.DMatrix(data=train_df2.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])
    val_data = xgb.DMatrix(data=train_df2.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])
    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
    print("xgb " + str(fold_) + "-" * 50)
    num_round = 10000
    xgb_model = xgb.train(xgb_params, trn_data, num_round, watchlist, early_stopping_rounds=500, verbose_eval=1000)
    oof_xgb_3[val_idx] = xgb_model.predict(xgb.DMatrix(train_df2.iloc[val_idx][df_train_columns]), ntree_limit=xgb_model.best_ntree_limit+50)

    predictions_xgb_3 += xgb_model.predict(xgb.DMatrix(test_df2[df_train_columns]), ntree_limit=xgb_model.best_ntree_limit+50) / folds.n_splits

np.save('oof_xgb', oof_xgb_3)
np.save('predictions_xgb', predictions_xgb_3)
print(np.sqrt(mean_squared_error(target.values, oof_xgb_3)))

3.649844028552147
```

## Post processing

### Stacking the model predictions

There are various techniques for improving the score based on the predictions of multiple models. Simple average of models' predictions has significant boosts in the LB. Here we will see another such technique.

We will construct a meta model which takes training data predictions from various models as features. Our dependant variable will still remain `target` from the training data frame. We will feed this features to a simple Ridge regression model and make predictions by repeating the same on afore said models' test predictions.

```python
train_stack = np.vstack([oof_lgbm, oof_dl, oof_xgb]).transpose()
test_stack = np.vstack([predictions_lgbm, predictions_dl, predictions_xgb]).transpose()

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
oof_stacked = np.zeros(train_stack.shape[0])
predictions_stacked = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, train_df2['outliers'].values)):
    print("fold n°{}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

    clf = Ridge(alpha=1)
    clf.fit(trn_data, trn_y)

    oof_stacked[val_idx] = clf.predict(val_data)
    predictions_stacked += clf.predict(test_stack) / folds.n_splits

np.sqrt(mean_squared_error(target.values, oof_stacked))

3.641238686022574
```

Our meta model gave a huge boost of 0.03 on stacking the LGBM, XGBoost & Deep learning models. Lets' make a submission to kaggle at this point.

### Combining model without outliers data

Outliers are greatly skewing the predictions for other `card_id`. Maybe if the outliers didn't penalise our model using RMSE so much, we can make better predictions for the rest of the data. So how we know who are the outliers in our test data set? We will build a classifier to classify outliers!

So the total steps are:

1. Train a model without outliers - model A.
2. Train a model to classify outliers in the data - model B.
3. Use model B to classify outlier card_ids in the test data using a threshold.
4. Use model A to predict the target for non-outliers, use our previous predictions from older models for the outliers.

This way we aren't changing the predictions for the outlier data but we would be using better predictions from model A for the non-outliers.

The above approach gave me a boost of +0.003 on the LB.

The 1st prize winner had a simple trick built on top of the above approach. Instead of picking and blending the outliers by a threshold, he did a linear interpolation based on the probabilites from the classifier.

```python
train['final'] = train['binpredict'](-33.21928)+(1-train['binpredict'])train['no_outlier']
```

Obviously accuracy depends on the performance of our classifier. It was so simple and yet very effective for him. It gave him 0.015 boost in local cv compare with same feature on the LB. See more discussion [here](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82036#latest-489417).

## Kaggle submission

```python
submission_name = pd.to_datetime("today")
sub_df = pd.DataFrame({"card_id":test_df2["card_id"].values})
sub_df["target"] = predictions_stacked
sub_df.to_csv(f'dl_xgb_submission_stacked_lgb-{submission_name}.csv', index=False)
filename = f'dl_xgb_submission_stacked_lgb-{submission_name}.csv'
```

```shell
kaggle competitions submit elo-merchant-category-recommendation -f 'dl_xgb_submission_stacked_lgb-2019-02-26 22:45:24.597929.csv' -m "dl & xgb & lgbm stacked with 178 cols - 3.641238686022574"
```

Our submission gave us 3.61001 on private leaderboard and 3.68463 on public leaderboard.
