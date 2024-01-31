# COVID-Dynamics-Model-Comparison

## Introduction
Comparative study of the main techniques used for COVID modeling where the information available is infected curve. The objective is to identify those univariate techniques that produce the best results, analyzing whether the more complex models are really able to provide better predictions.

Since COVID-19 was declared a pandemic, the urgency to obtain accurate predictive methods to help institutions make decisions on measures to apply and the uncertainty surrounding the virus has facilitated the publication and application of different techniques. The motivation of this study is to compare them, in particular compartmental epidemiological models, linear regression models, ARIMA family models and recurrent neural networks. 

## Data 
COVID-19 cases in Spain reported by province daily (see [source](https://cnecovid.isciii.es/covid19/#documentación-y-datos)). 

## EDA and Data processing
[Exploratory Data Analysis](https://github.com/marialonsogar/COVID-Dynamics-Model-Comparison/blob/main/analysis/EDA-COVID-Spain.ipynb) was conducted to explore time series patterns (global and local trends, structural changes, seasonalities...), data inconsistencies, outliers, etc. 

![image](https://github.com/marialonsogar/COVID-Dynamics-Model-Comparison/assets/80226382/337ef2a9-99b5-4f54-a0bd-faea2edb477e)


[Data processing](https://github.com/marialonsogar/COVID-Dynamics-Model-Comparison/blob/main/analysis/EDA-COVID-Spain.ipynb):
- Aggregate data from province-level to national-level since the point of interest lies on a global level 
- Remove variables of hospitalized individuals and ICU inpatients (not relevant for this analysis) and rename columns for ease of analysis
- Add population (total population of Spain as constant) and recovery cases, required for SIR model study (population=susceptible)
- Smooth data by a mean average of 7 periods (days) to remove seasonal fluctuations caused by absence of data during weekends: the series exhibit seasonal fluctuations with period 7 (due to the lack of data communication from the communities on weekends)
- Outliers were identified duting summer and Christmas season, but they are inherent to the series
- Forecasting horizon set up to 14 days in the future

## Modeling
A set of models were fitted and evaluated (MAE, RMSE, MAPE, RMSLE) on different windows with time series cross validation (see [walk-forward schema](https://github.com/marialonsogar/COVID-Dynamics-Model-Comparison/blob/main/analysis/walk-forward-scheme.png) and [expanding walk-forward schema](https://github.com/marialonsogar/COVID-Dynamics-Model-Comparison/blob/main/analysis/walk-forward-expanding-scheme.png)). Implementation and mathematical details are well elaborated in each notebook:
- Epidemiological models ([SIR](https://github.com/marialonsogar/COVID-Dynamics-Model-Comparison/blob/main/modeling/SIR-COVID-Spain.ipynb), [SIS](https://github.com/marialonsogar/COVID-Dynamics-Model-Comparison/blob/main/modeling/SIS-COVID-Spain.ipynb))
- [Trend Extrapolation](https://github.com/marialonsogar/COVID-Dynamics-Model-Comparison/blob/main/modeling/trend-extrapolation-COVID-Spain.ipynb) with polynomial, exponential, logistic or Gompertz curves. 
- [Linear Regression](https://github.com/marialonsogar/COVID-Dynamics-Model-Comparison/blob/main/modeling/linear-regression-COVID-Spain.ipynb)
- [ARIMA](https://github.com/marialonsogar/COVID-Dynamics-Model-Comparison/blob/main/modeling/ARIMA-COVID-Spain.ipynb) (and SARIMA)
- [Recurrent Neural Networks (RNN)](https://github.com/marialonsogar/COVID-Dynamics-Model-Comparison/blob/main/modeling/RNN-COVID-Spain.ipynb)


## Results

[<img src="./data/analysis-data/comparison-all-metrics.png" width="700"/>](./data/analysis-data/comparison-all-metrics.png)

- All the metrics increase as the time horizon increases for all the models, which is reasonable, since the farther the future point is from the known observations, the greater the uncertainty.
- It can be seen that the best model for any metric is the ARIMA(2,1,5). The RNN considered is incapable of correctly capturing the dynamics of the virus, which is manifested by generating predictions that are insufficiently accurate. The SIS model and linear regression follow a similar evolution except for the RMSLE, when the linear regression model increases drastically from time horizon 8 onwards. This may be because the series studied does not verify the hypotheses of the SIS model and is unable to provide parameters with epidemiological significance. Consequently, the model has no epidemiological interpretation but becomes a mere regression adjustment.
- Finally, it should be recalled that none of the models studied verifies the initial hypotheses. Therefore, the results could be improved by studying another type of method.
- [More details](https://github.com/marialonsogar/COVID-Dynamics-Model-Comparison/blob/main/analysis/comparison.ipynb)
