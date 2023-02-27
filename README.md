_MBD @IE_

_Group 1_

<img width="300" style="float:center" 
     src="https://logos-world.net/wp-content/uploads/2022/06/BGG-Logo.png" />

# BCG Capstone Project - Documentation

## Abstract

This repository contains a **complete assesment** on **Client Co®'s** context, trends and predictions. The **analysis** covers different perspectives, in order to bring about a **concise** and **data-driven** set of solutions and recommendations to the end-client.

To achieve such task, the repository presents three concurrent analysis:

* `EDA, Customer Segmentations & Spend Propensity`: 
* `Sales Forescasting`: 
* `Product Analysis & Customer Lifetime Value`: 

> Note: The insights from each perspective are clearly detailed in both `.ipynb` and `.html` formats. The first is a standard in the field, whereas the second one eases a quick view to the reader.


## 1. Dependencies

- **Python 3.x**
- **Jupyter Notebook**: `pip install jupyter`
- **Pandas:** `pip install pandas`
- **Numpy:** `pip install numpy`
- **Matplotlib:** `pip install matplotlib.pyplot`
- **Plotly:** `pip install plotly`
- **Seaborn:** `pip install seaborn`
- **Plotnine:** `pip install plotnine`
- **ScikitPlot:** `pip install scikitplot`
- **Statsmodels:** `pip install statsmodels`
- **ScyPy:** `pip install scypy`
- **Lifetimes:** `pip install lifetimes`
- **Scikit Learn:** `pip install sklearn`
- **Prophet:** `pip install prophet`
- **Lime:** `pip install lime`
- **Abc Analysis:** `pip install abc_analysis`
- **Mlxtend:** `pip install mlxtend`
- **Itertools:** `pip install itertools`
- **Collections:** `pip install collections`
- **Networkx:** `pip install networkx`
- **Pickle:** `pip install pickle`


> Note: The required versions for each each are defined in `requirements.txt`



## 2. Quick Guide 

To follow along the project, the reader should tackle these analysis sequentially. This approach not only provides a broader understanding on the context, but is also consistent with the evolution and transformations on the data.

In this aspect:


1. `EDA, Customer Segmentation, Spend Propensity, Recommendations.ipynb`
2. `Sales_Forecast_9Weeks_Final.ipynb`
3. `Product_analysis_and_CLV.ipynb`



## 3. Code Documentation

### 3.1. EDA, Customer Segmentations & Spend Propensity | `EDA, Customer Segmentation, Spend Propensity, Recommendations.ipynb`

<img width="1000" style="float:center" 
     src="https://i.imgur.com/ldpdrIn.png" />

<a name="Footnote" >1</a>: _Schema of the `EDA, Customer Segmentation, Spend Propensity, Recommendations.ipynb` pipeline_

The following Notebook consists on the following:

1. Exploratory Data Analysis (EDA), which set the basis for customer segmentation and modeling.
2. Customer Segmentation using RFM technique.
3. Spend Propensity Model.

Those three pillars deliver core recommendations to the business, based on those data-driven analysis. 

### 3.2. Sales Forescasting | `Sales_Forecast_9Weeks_Final.ipynb`



### 3.3. Product Analysis & Customer Lifetime Value | `Product_analysis_and_CLV.ipynb`

The following Notebook contains the code used for Product Association Rules and Customer Lifetimes Value. From it, we highlight the following sections: 

1. Importing Libraries.
2. Load Data.
3. Refunds Analysis.
	3.1 Transactions with negative values are analyzed in this section in order to identify the 	most refunded products and the sales loss these represent for the business. 
4. Product Segmentation on all Products with ABC Analysis. 
	4.1 Data is aggregated and sorted by total sales in  a descending way. 
	4.2 This is to segment our products and see how long is the tail of products that are not 	contributing
5. Modeling Customer Lifetime Value
	5.1 In this section, the Lifetimes package is used for calculating the CLV of all returning customers. 
	5.2 We first use RFM features and predict the expected number of Transactions.
	5.3 Expected average monetary value it's predicted with a Gamma Gamma Model
	5.4 CLV calculations for all customers can be found at the end of this sections.
	5.5 As a final step, we merge the client_id, clv and customer segments for business 	purposes.
6. Product Analysis for Top100 Selling Products of 2019
	6.1 This part of the code is only used for filtering our data for considering only the top 	selling products and merging them with all transactions where these appear. 
7. Association Rules for Top 100 Products of 2019
	7.1 Once we take the data from 6.1, we use the Apriori model for mining the association 	rules.
8. Graph Analysis on Association Rules for Explainabilty. 
9. Additional Approach on Frequent Item Sets (Manual)
	9.1 Manually coded the combination of most frequent items without association metrics.
10. Data Saving (Only Run after all models have been executed)


## 4. Results and Recommendations


## 5. References

For more information on the techniques and methods used in these notebooks, please refer to the following references:

- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow
- Python Data Science Handbook
- Forecasting: Principles and Practice
