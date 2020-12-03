import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
import scipy
from statsmodels.stats.diagnostic import het_breuschpagan
import phik
import scipy.stats as st

import statsmodels.regression.linear_model as lm_
import statsmodels.discrete.discrete_model as dm_
import statsmodels.regression.mixed_linear_model as mlm_
import statsmodels.genmod.generalized_linear_model as glm_
import statsmodels.robust.robust_linear_model as roblm_
import statsmodels.regression.quantile_regression as qr_
import statsmodels.duration.hazard_regression as hr_
import statsmodels.genmod.generalized_estimating_equations as gee_
import statsmodels.gam.generalized_additive_model as gam_

gls = lm_.GLS.from_formula
wls = lm_.WLS.from_formula
ols = lm_.OLS.from_formula
glsar = lm_.GLSAR.from_formula
mixedlm = mlm_.MixedLM.from_formula
glm = glm_.GLM.from_formula
rlm = roblm_.RLM.from_formula
mnlogit = dm_.MNLogit.from_formula
logit = dm_.Logit.from_formula
probit = dm_.Probit.from_formula
poisson = dm_.Poisson.from_formula
negativebinomial = dm_.NegativeBinomial.from_formula
quantreg = qr_.QuantReg.from_formula
phreg = hr_.PHReg.from_formula
ordinal_gee = gee_.OrdinalGEE.from_formula
nominal_gee = gee_.NominalGEE.from_formula
gee = gee_.GEE.from_formula
glmgam = gam_.GLMGam.from_formula

def set_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    
def plot_hist(data, title):
    plt.hist(data, color='steelblue',edgecolor='black', linewidth=.8)
    plt.title(f'{title}',fontname='Futura',fontsize=14)
    plt.xticks(fontname='Futura',fontsize=12)
    plt.yticks(fontname='Futura',fontsize=12)
    plt.show()

def sub_lists (l): 
    '''
    given a list l, return all sublists
    example: l = ['a', 'b', 'c']
    return: [[], ['a'], ['b'], ['c'], ['a', 'b'], ['a','c'], ['b','c'], ['a','b','c']]
    '''
    base = []   
    lists = [base] 
    for i in range(len(l)): 
        orig = lists[:] 
        new = l[i] 
        for j in range(len(lists)): 
            lists[j] = lists[j] + [new] 
        lists = orig + lists 
          
    return lists 

def Predictors_Residual(model, df, X_names, axes):
    '''
    plot residuals against all predictors
    '''
    axes = axes.flatten()
    i = 0
    for col in X_names:
        axes[i].scatter(df[col].values, model.resid)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("residuals")
        axes[i].set_title(col)
        i += 1

def BP_test(model):
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    labels = ['LM Statistic', 'LM-Test p-value']
    return dict(zip(labels, bp_test))

class LinearRegression:
    
    def __init__(self, y_name, X_names, df):
        self.y_name = y_name
        self.X_names = X_names
        self.df = df
        self.formula = f"{y_name} ~ {'+'.join(self.X_names)}"
        self.alpha = 0.05
    
    def fit_ols(self):
        
        self.model = smf.ols(self.formula, data=self.df).fit()
        return self.model
    
    def vif_score(self):
        y, X = dmatrices(self.formula, data=self.df, return_type='dataframe')
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif["features"] = X.columns
        return vif
    
    
    def fitted_residuals(self, ax):
        p1 = self.model.fittedvalues
        res1 = self.model.resid
        ax.scatter(p1,res1)
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residual")
        ax.set_title("Fitted Values vs. Residuals")

    def bp_test(self):
        return BP_test(self.model)
    
    def plot_influence(self, ax, criterion="cooks"):
        sm.graphics.influence_plot(self.model,ax=ax, criterion="cooks") 
        
    def drop_influential_points(self):
        '''
        drop influential points identified by both cook distance and external student residuals
        '''
        cook_index = self.influential_index_by_cook()
        cook_index = set(cook_index)
        ext_index = self.influential_index_by_stud()
        ext_index = set(ext_index)
        index = list(set.intersection(cook_index, ext_index))
        index.sort()
        
        return self.df.drop(index)
    
    def influential_index_by_cook(self):
        '''
        get index of influencial points using cook distance
        '''
        infl = self.model.get_influence()
        n=len(self.df)
        inflsum=infl.summary_frame()
        reg_cook=inflsum.cooks_d

        atyp_cook = np.abs(reg_cook) >= 4/n
        cook_list = reg_cook[atyp_cook].index
        
        return list(cook_list)
    
    def influential_index_by_stud(self):
        '''
        get index of influencial points using external student residuals
        '''
        n=len(self.df)
        infl = self.model.get_influence()
        p=self.model.df_model + 1
        seuil_stud = st.t.ppf(1-self.alpha/2,df=n-p-1)
        reg_studs=infl.resid_studentized_external
        atyp_stud = np.abs(reg_studs) > seuil_stud
        ext_list = self.df.index[atyp_stud]
        
        return list(ext_list)
    
    
NORMS = [
    sm.robust.norms.HuberT(),
    sm.robust.norms.LeastSquares(),
    sm.robust.norms.Hampel(),
    sm.robust.norms.AndrewWave(),
    sm.robust.norms.TrimmedMean(),
    sm.robust.norms.RamsayE()
]

NORM_NAMES = ['HuberT','LeastSquares','Hampel','AndrewWave', 'TrimmedMean','RamsayE']

class RobustLinearRegression(LinearRegression):
    def __init__(self, y_name, X_names, df):
        LinearRegression.__init__(self, y_name, X_names, df)
        
    def bp_test_each_norm(self):
        result = pd.DataFrame(columns=['Norm', 'LM Statistic', 'LM-Test p-value'])
        for name, norm in zip(NORM_NAMES, NORMS):
            bp_test = self.rlm_dwn(norm)
            bp_test['Norm'] = name
            result = result.append(bp_test, ignore_index=True)
        return result
        
    def rlm_dwn(self, norm):
        model_tst = rlm(self.formula,data=self.df, M=norm).fit()
        return BP_test(model_tst)
    
    def fitted_residuals_each_norm(self, ax):
        ax = ax.flatten()
        i = 0
        for name, norm in zip(NORM_NAMES, NORMS):
            model = rlm(self.formula,data=self.df, M=norm).fit()
            ax[i].scatter(model.fittedvalues, model.resid)
            ax[i].set_xlabel("Fitted Values")
            ax[i].set_ylabel("Residual")
            ax[i].set_title(name)
            i += 1
        

class ModelSelection:

    '''
    This is a class for automatic model selection in Linear Regression.

    Usage:

    model_selection = ModelSelection('WHAT_WE_PREDICT', ['PREDICTOR_0','PREDICTOR_1','PREDICTOR_2'], df=A_DATA_FRAME)
    model_selection.create_subset_models()
    summary = model_selection.summary()
    print(summary)

    Explanation: we feed the dataset and assign what we need to predict, and what's the predictors to ModelSelection,
        then the class will automatically calculate all sub-models, wrtie a table to report the results
    
    The summary table will contain: "Number of predictors", "Adj_R2", "Cp", "Predictors", we could add more.

    How to read the code:
    1. start with __init__
    2. core part is create_subset_models

    '''

    def __init__(self, y_name, X_names, df):

        '''
        work flow: 
        1. accept y_name, X_names, and df
        2. create a full model, save the mse_resid to the class for further use.
        3. create a report table
        4. create a subset models

        '''
        self.p = len(X_names) + 1
        
        self.full_model = self.create_full_model(y_name, X_names, df)
        self.mse_full = self.full_model.mse_resid
        
        self.table = pd.DataFrame(columns=['Number of predictors', 'Adj_R2', 'Cp', 'AIC', 'BIC', 'Predictors'])
        self.create_subset_models(y_name, X_names, df)
    
    def create_full_model(self, y_name, X_names, df):
        '''
        create a full model and return the model
        '''
        model = smf.ols(f"{y_name} ~ {'+'.join(X_names)}", data=df).fit()
        return model

    def create_subset_models(self, y_name, X_names, df):
        '''
        the CORE of this class, workflow:
        1. create the subset of predictors using sub_lists (l)
        2. for each subset of combinations of predictors, train a model.
           If it is a null model, it will call fit_null_model. Else, it will call fit_model
           both of them will return [n_preds, adj_R2, mallow_cp, predictors]
        3. append the list to report table

        '''
        sub_vars = sub_lists(X_names)
        df_columns = list(self.table.columns)
        for each_comb in sub_vars:
            
            result = self.fit_model(y_name, each_comb, df)
            self.table = self.table.append(dict(zip(df_columns, result)), ignore_index=True)
            
    def fit_model(self, y_name, X_sub, df):
        '''
        fit a sub-model, return:
        n_preds: number of predictors
        adj_R2: adjusted R square value
        mallow_cp: Mallow's Cp value
        predictors: predictors separated by comma
        '''
        
        n_preds = len(X_sub)
        
        if n_preds == 0:
            model = smf.ols(f"{y_name} ~ 1", data=df).fit()
        else:     
            model = smf.ols(f"{y_name} ~ {'+'.join(X_sub)}", data=df).fit()
        
        k = n_preds + 1
        adj_R2 = model.rsquared_adj
        n = len(df)
        sse = model.mse_resid * (n-k)
        mallow_cp = (sse/self.mse_full) - (n-2*k)
        aic = model.aic
        bic = model.bic
        predictors = ', '.join(X_sub)
        result = [n_preds, adj_R2, mallow_cp, aic, bic, predictors]
        
        return result

    def summary(self):
        '''
        return the summary table
        '''
        self.table = self.table.sort_values(by=['Cp','AIC','BIC'], ascending=[1,1,1])
        return self.table
    