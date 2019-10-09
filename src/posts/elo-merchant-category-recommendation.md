---
title: Feature engineering of kaggle Elo merchant competition - Top 3% solution 🔥
date: '2019-09-02'
tags:
  - tabular-data
  - feature-engineering
  - deep-learning
  - data-science
  - blog
  - kaggle
---

Contents:
1. Data preprocessing
3. Feature engineering
4. New features for Elo merchant competition
5. Aggregations on the features
6. Feature engineering on the aggregates
7. Various models
    * LGBM
    * Random Forest
    * XG Boost
    * Deep learning model
    * Cat boost
8. Deep learning on categorical data
9. Feature selection/elimination via feature importance 
10. Post processing after model
11. Stacking, blending & ensembling 
12. Use ridge regression for model ensembling
13. First place winner’s solution & approach


## Elo Merchant Competition

 Elo is one of the largest payment brands in Brazil. It has built partnerships with merchants in order to offer promotions or discounts to cardholders. But do these promotions work for either the consumer or the merchant? Do customers enjoy their experience? Do merchants see repeat business? Personalization is key.

 In this competition our challenge is to develop algorithms to identify and serve the most relevant opportunities to individuals, by uncovering signal in customer loyalty. Our input will help Elo reduce unwanted campaigns, to create the right experience for customers.

 ### Understanding the data

 Though Elo mentioned that all data is simulated and fictitious, and is not real customer data, it was later hinted that data was highly transformed & modified using statistical techniques to make it hard for data leaks.


#### What files do we need?
All the data can be downloaded from [here] (https://www.kaggle.com/c/elo-merchant-category-recommendation/data)
We will need, at a minimum, the train.csv and test.csv files. These contain the card_ids that we'll be using for training and prediction.

The historical_transactions.csv and new_merchant_transactions.csv files contain information about each card's transactions. historical_transactions.csv contains up to 3 months' worth of transactions for every card at any of the provided merchant_ids. new_merchant_transactions.csv contains the transactions at new merchants (merchant_ids that this particular card_id has not yet visited) over a period of two months.

merchants.csv contains aggregate information for each merchant_id represented in the data set.

#### What should we expect the data format to be?
The data is formatted as follows:

train.csv and test.csv contain card_ids and information about the card itself - the first month the card was active, etc. train.csv also contains the target.

historical_transactions.csv and new_merchant_transactions.csv are designed to be joined with train.csv, test.csv, and merchants.csv. They contain information about transactions for each card, as described above.

merchants.csv can be joined with the transaction sets to provide additional merchant-level information.

#### What are we predicting?
We are predicting a loyalty score for each card_id represented in test.csv and sample_submission.csv.

#### File descriptions
train.csv - the training set
test.csv - the test set
sample_submission.csv - a sample submission file in the correct format - contains all card_ids we are expected to predict for.
historical_transactions.csv - up to 3 months' worth of historical transactions for each card_id
merchants.csv - additional information about all merchants / merchant_ids in the dataset.
new_merchant_transactions.csv - two months' worth of data for each card_id containing ALL purchases that card_id made at merchant_ids that were not visited in the historical data.

#### Data fields
Data field descriptions are provided in Data Dictionary.xlsx.

