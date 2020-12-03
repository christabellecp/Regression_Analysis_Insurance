# Regression Analysis on Health Insurance Cost
Christabelle Pabalan, Dashiell Brookhart, and Sicheng Zhou


## 1. Introduction
Regression analysis is a technique in statistics to investigate and model the relationship between variables (Douglas Montgomery, Peck, & Vinning, 2012). Multiple linear regression depends on modeling a relationship between the dependent (or response) variable and the independent (or predictor) variables. The purpose of this research is to analyze health insurance data to verify whether or not regression analysis models work effectively to predict health insurance charges. This study was also used to discover relevant factors that affect the cost of health insurance and investigate the extent in which these predictor variables are successful at prediction.

## 2. Problem Statement
We examined 5 main questions:
- What are the differences in significant predictors of health insurance cost for individuals who smoke and those who do not?  
- For a 45 year old individual with a BMI of 35 in the subset population of smokers, what is the expected cost of their health insurance? 
- For a 45 year old individual living in the northwest with 3 children in the subset population of nonsmokers, what is the expected cost of their health insurance? 
- How does the cost of health insurance change for an individual in the nonsmoker population when they have a child? 
- How does health insurance price change in respect to an increase in body mass index within the smoker population?  


### Dataset Description 
The dataset we analyzed is titled, “Medical Cost” from Kaggle. A link to the dataset is here. This dataset has 1,338 observations with seven different variables. Three of these variables are numerical (Age, BMI, and Charges) and the rest are categorical (Sex, Children, Smoker, and Region). 

### Methods and Measures
There are several fundamental model and data assumptions necessary to justify the use of linear regression models for the purpose of inference or prediction. These include: independence of predictor values, constant variance of errors, normality of the error distribution, independence of errors, and linearity between the response variable and any predictor variable. This section displays the methods and measures that were selected to optimize our regression model and correct issues with underlying model assumptions. 

**Sequential Analysis of Variance (ANOVA)**: The Sequential form of ANOVA is the calculation of the reduction in the error sum of squares (SSE) when one or more predictor variables are added to the model. It helps us by indicating the significance of a predictor variable given the predictors listed before it are already in the model.

**Partial Analysis of Variance (ANOVA):** The Partial form of ANOVA differs from the Sequential version as the order of the predictors does not matter. The Partial form calculates the significance of a predictor given all the rest of the predictors are already in the model.

**Variance Inflation Factor (VIF):** VIF measures how much the variance is inflated in the coefficient estimate caused by a predictor variable. We calculate the VIF score for each predictor variable and say that if the value is greater than or equal to 10, then that particular predictor variable is causing serious multicollinearity. 

**PhiK Correlation Heatmap:**  The PhiK measure is similar to the Pearson correlation coefficient in its interpretation but it works consistently between categorical, ordinal and interval variables. The correlation heatmap allows us at a glance to distinguish which variables in our dataset are correlated to each other. This helps us determine whether or not we may have a serious multicollinearity problem. In our report, variables that are highly correlated are shaded in a different color in the heatmap.

**Externally Studentized Residuals:**  Externally studentized residuals help identify points that negatively affect a regression model. For a specific point it is the residuals of the model without that observation included over the estimated standard deviation without that observation included. The rule of thumb to calculate possible influential points using this method is to collect all points whose absolute value of their externally studentized residual is greater than or equal to t𝛼/2, df=n-p-1. We use the intersection of points highlighted by both Cook’s Distance and Externally Studentized Residuals as our influential points

**Cook’s distances:**  Cook’s distance helps identify points that negatively affect a regression model. Cook’s distance is a combination of an observation’s leverage and residual value. The rule of thumb is that if the value of the Cook’s distance for an observation is greater than 4n (where n is the number of observations in our sample) then that observation may be an influential point. We use the intersection of points highlighted by both Cook’s Distance and Externally Studentized Residuals as our influential points. 

**Log Transformation on Response Variable:** Logarithmic transformation is a common method to transform a highly skewed variable into an approximately normal distribution. For regression purposes, it is also used to stabilize variance if the residuals consistently increases and to linearize a regression model with an increasing slope.

**Quantile-quantile plots (Q-Q):** Q-Q plots are a graphical technique that helps determine whether a model has a normal distribution. If the resulting plot is approximately linear on the diagonal of the plot, then it suggests that the error term of the model follows a normal distribution.

**Breusch-Pagan test:** The Breusch-Pagan test follows the idea that the variance of errors should not change given different predictor values. If the assumption of constant variance is violated, then the variance would change with the predictor values. This test helps us identify if our model has a significant heteroscedasticity problem. If the p-value of the Breush-Pagan test is less than or equal to 0.05, then we state that we have a significant heteroscedasticity problem.

**Omnibus and Jarque-Bera tests:** The Omnibus and Jarque-Bera tests are used to check for a violation of normality. They both test for this by calculating the skewness and kurtosis levels of a model. If the p-value is less than or equal to 0.05 for either test, then we say that the residuals of the model may not follow normality.

**Residuals vs. Predictor plot:** The residuals vs. predictors plots are used to test the assumption of linearity between the response variable and the predictor variables. If the plot shows a clear linear relationship then we say that the assumption of linearity holds for that particular predictor variable.

**Robust Regression:**  Robust regression was used as compromise between excluding the high influential points entirely and including them as an equal data point. This model is essentially a replacement for the least squares regression and gives a different weight to the outliers found in our dataset. We ran through a parameter selection process by comparing various robust linear model functions (which defines the weight given to outliers). The parameters used included Huber, Least Squares, Hampel, Andrew Wave, Trimmed Mean and RamsayE.

