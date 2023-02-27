_MBD @IE_

_Group 1_

<img width="300" style="float:center" 
     src="https://logos-world.net/wp-content/uploads/2022/06/BGG-Logo.png" />

# BCG Capstone Project - Documentation

## Abstract

This repository contains a **complete assesment** on **Client Co®'s** context, trends and predictions. The **analysis** covers different perspectives, in order to bring about a **concise** and **data-driven** set of solutions and recommendations to the end-client.

To achieve such task, the repository presents three concurrent analysis:

* `EDA, Customer Segmentations & Spend Propensity`: shows the core segmentation of clients, according to the RFM Analysis. Showcases predictions for the following month spenders. Leverages `lime` package to bring explainability to the predictive model.

* `Product Analysis & Customer Lifetime Value`: covers Product Association Rules and Customer Lifetime Value. In more detail, tackles a refunds analysis, product segmentation, CLV modeling using RFM anda Gamma Gamma Model. The top 100 selling products of 2019 are analyzed using association rules with an Apriori model.


* `Sales Forecasting`: presents a prediction on the total sales of ClientCo over the next 9 weeks, using weekly sales to avoid daily fluctuations. The forecast leverages `Facebook Prophet`, with a two-level time series approach: forecasting total sales of ClientCo and for customer segments based on RFM. 


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
2. `Time Series Forecast - Total Sales 9 Weeks.ipynb`
3. `Product_analysis_and_CLV.ipynb`



## 3. Code Documentation

### 3.1. EDA, Customer Segmentations & Spend Propensity | `EDA, Customer Segmentation, Spend Propensity, Recommendations.ipynb`

<img width="1000" style="float:center" 
     src="https://i.imgur.com/lPDjAmj.png" />

<a name="Footnote" >1</a>: _Schema of the `EDA, Customer Segmentation, Spend Propensity, Recommendations.ipynb` pipeline_

The following Notebook covers:

1. Exploratory Data Analysis, the base of the customer's segmentation and modeling. In essence, the nuances of the data provided and deep analysis of it.
2. Customer Segmentation using RFM technique, segmenting clients into 6 segments based on the computed RFM score.
3. Spend Propensity Model, tested both Logistic Regression and RandomForestClassifier. The first one, intended towards bringing explainability to the model, and the swcond, being the most perfomant.
4. Based on the above, the notebook showcases & visualises certain recommendations for the business, giving answers to some business questions based on the above modeling and segmentation. In addition, a basic graph analysis between branches and customers was also performed.

Those three pillars deliver core recommendations to the business, based on those data-driven analysis. 


### 3.2. Product Analysis & Customer Lifetime Value | `Product_analysis_and_CLV.ipynb`


<img width="1000" style="float:center" 
     src="https://i.imgur.com/i4acMrR.png" />
<a name="Footnote" >2</a>: _Schema of the `Product_analysis_and_CLV.ipynb`_


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



### 3.3. Sales Forescasting | `Time Series Forecast - Total Sales 9 Weeks.ipynb`

<img width="1000" style="float:center" 
     src="https://i.imgur.com/iZU7Ig0.png" />
<a name="Footnote" >3</a>: _Schema of the `Sales_Forecast_9Weeks_Final.ipynb`` pipeline_

The following Notebook covers our approach on predicting the total sales of `ClientCo` in the next 9 weeks period (22 September 2019 - 22 November 2019). 

* We are using weekly sales to avoid daily fluctuation with the weekend sales being negligible.

* We are building a Time Series Forecast on two levels: 
  1. Forecast of total sales of `ClientCo`
  2. Forecast of the 6 different customer segments we have developed using RFM.

* For the forecast we are using `Facebook Prophet`, the open source software released by Facebook's core Data Science Team. We chose this model due to the following reasons:
  1. It performs well with time series that have strong seasonal effects and several seasons of historical data. 
  2. It is also robust to missing data and shifts in the trend
  3. It can handle outliers.
 
* For evaluating our forecast we are using `MAPE` and `MAE` score, which are popular metrics when evaluating forecasting models.




## 4. References

For more information on the techniques and methods used in these notebooks, please refer to the following:


[1] Harsh. "Association Analysis in Python: Frequent Item set Mining using Apriori algorithm in Python." Analytics Vidhya, 26 Sep. 2019, https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-association-rules/ 

[2] Logunova, I. (2022, June 9). A Beginner's Guide to Data Mining Techniques. Serokell. https://serokell.io/blog/guide-to-data-mining-techniques 

[3] Antonio, Meraldo. "Modeling Customer Lifetime Values with Lifetimes." Towards Data Science, 24 Mar. 2022, towardsdatascience.com/modeling-customer-lifetime-values-with-lifetimes-4d6df4b6c160 

[4] Onnen, Heiko. "Buy 'Til You Die: Predict Customer Lifetime Value in Python." Towards Data Science, 27 Nov. 2021, towardsdatascience.com/buy-til-you-die-predict-customer-lifetime-value-in-python-7bf87a28fc20 

[5] Fader, Peter S., Bruce G.S. Hardie, and Ka Lok Lee. "RFM and CLV: Using iso-value curves for customer base analysis." Journal of Marketing Research 50.4 (2013): 444-456 

[6] Raschka, Sebastian. "Mining Frequent Itemsets with Support and Confidence Using Python." Sebastian Raschka, 7 Mar. 2018, https://sebastianraschka.com/Articles/2014_python_apriori.html 

[7] Davidson-Pilon, C. (2020). Lifetimes: Easily estimate customer lifetime value. https://github.com/CamDavidsonPilon/lifetimes 

[8] Mathur, V. (2022, September 26). Association Rule Mining: Importance and Steps. Analytics Steps. https://www.analyticssteps.com/blogs/association-rule-mining-importance-and-steps

[9] Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow

[10] Python Data Science Handbook
