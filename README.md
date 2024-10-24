## Introduction
Complex solution via a two-stage model for recommendation predictions. In the first stage (called the pre-models stage), we use an ensemble of ALS and BM25 models to select candidates and then rank them using a more complex model.

### Dataset

![alt text](imgs/image-1.png)

*d1* - dataset with data from weeks other than the last *n_weeks_for_catboost*(for pre-model training)\.
*d2* - dataset with data for the last *n_weeks_for_catboost* weeks, (for catboost training).

At the feature creation stage, we create target values for model training and additional features based on historical data and split the original dataset into three parts (for hypothesis testing and local testing and decision). The features include the following: 
1. The slope angle of the trend line (for each of the days of the week)
2. Signs based on the duration of the views(90 percentile, arithmetic mean, sum)

We focus on taking into account trends and seasonality, which allows our solution to minimise the problem of lack of historical data.

### Solution

![alt text](imgs/image.png)

Our solution uses implementations of algorithms from the implicit library, which are characterised by their speed and the ability to parallelise computations on a graphics accelerator.

Among other things, we should note the division of users into two groups based on the number of views: one of them will use prediction based on modelling, and the other - based on heuristics described in the ColdStart class.

The second step of the solution is to use Catboost gradient bousting to rank the candidates obtained from the pre-models. Candidates that the user has actually looked at are labelled label=1, otherwise 0. As attributes for catboost we use information about the user, the film and their interaction history. For each user, for X true candidates, we randomly sample X candidates that are only in the prediction of both ALS and BM25

Before recommending for all users, we re-train the pre-models on all data.

Training dataset: [Download](https://cloud.gs-labs.tv/s/iagLiCtlMDXmzKf)

### Install & Start
The whole solution is broken into scripts to modernise independent parts of the solution. The dataset must be unzipped before running.

```bash
git clone 
cd rec_system_2stepCatboost
pip install -r requirements.txt
```
```bash
python run_solution.py
```

The full solution in notebook with comments can be found [there](https://github.com/Gruz2520/rec_system_2stepCatboost/blob/main/Team%20Buns.ipynb). 

Solution by **Team Buns**

