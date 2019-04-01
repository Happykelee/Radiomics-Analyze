# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:10:56 2018

@author: Caizhengting

<small but necessary>

"""
import os
from numpy import *
from pandas import *
import matplotlib
import matplotlib.pyplot as plt

from math import log,pi
from itertools import combinations
from scipy.stats import f,norm
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import check_scoring
from sklearn.model_selection import StratifiedKFold

def plot_default(size=10):
    """
    Set the default parameters of plotting 
    设置做图时的默认参数

    Key parameters
    ----------
        size:10,Default
    """

    matplotlib.rcdefaults()
    params = {
    'figure.figsize': [size, size/1.5],
    'figure.dpi': 100,
    'font.weight': 'bold',
    'font.size': size,
    'axes.labelsize': size*1.2,
    'axes.titlesize': size*1.5,
    'axes.titlepad': size*1.5,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.linewidth': size/10,
    'lines.linewidth': size/8,
    'legend.loc': 'upper right',
    }
    matplotlib.rcParams.update(params)

def plot_unit(lim):
    """
    Get the unit of xaxis and yaxis

     Key parameters
    ----------
        lim:ax.get_xlim() or ax.get_ylim()
    """
    unit = [(i[1]-i[0])/0.75/72 for i in lim]
    return(unit)

def list_shaping(obs,pred):
    """
    Judge whether the lengths of observation-list and prediction-list are conformity or not, and change to the data-structure of Series or List
    判断观察值列表和预测值列表长度是否一致，并且取成series/list数据结构
    """
    if type(obs) == type(DataFrame()): obs = obs.iloc[:,0].sort_index()
    if type(pred) == type(DataFrame()): pred = pred.iloc[:,0].sort_index()
    if len(obs) != len(pred):
        raise ValueError('Error! The lengths of observation-list and prediction-list are inconformity!')
    return(obs,pred,len(obs))

class paras():
    def __init__(self,X='',y='',pred='',pred_proba='',indice='',coef='',model='',name='ML',fpath='./',size=10,
                X_test='',y_test='',ID_test='',pred_test='',pred_proba_test='',indice_test='',
                X_train='',y_train='',ID_train='',pred_train='',pred_proba_train='',indice_train=''):
        try:self.X = X.sort_index()
        except:self.X = X
        try:self.y = y.sort_index()
        except:self.y = y
        try:self.X_train = X_train.sort_index()
        except:self.X_train = X_train
        try:self.X_test = X_test.sort_index()
        except:self.X_test = X_test
        try:self.y_train = y_train.sort_index()
        except:self.y_train = y_train
        try:self.y_test = y_test.sort_index()
        except:self.y_test = y_test
        self.model = model
        self.name = name
        self.fpath = fpath
        if not os.path.exists(self.fpath):os.makedirs(self.fpath)
        self.size = size
        self.ID_test = ID_test
        self.ID_train = ID_train
        self.pred = pred
        self.pred_proba = pred_proba
        self.pred_train = pred_train
        self.pred_test = pred_test
        self.pred_proba_train = pred_proba_train
        self.pred_proba_test = pred_proba_test
        self.indice = indice
        self.indice_test = indice_test
        self.indice_train = indice_train
        self.coef = coef
        plot_default(size=self.size)

#--------------------

def ICC(table,typeof='single'):
    """
    Calculate the itraclass correlation coefficent(ICC) using two-way random model 
    使用双因素随机模型计数组内相关系数

    Key parameters
    ----------
        typeof:'single'--single measure(单个观察者的多次观察评分),Default
                'average'--average measure(多个观察者间的评分)

    Reference
    ---------
        1.‘应用Excel完成组内相关系数ICC的计算和评价’，中国卫生统计2008年6月第25卷第三期
        2.‘组内相关系数及其软件实现’，中国卫生统计2011年10月第28卷第五期
    """
    if type(table) == type(DataFrame()): table = array(table)
    n, k = shape(table)
    # n: the number of subjects
    # k: the number of obsever-times
    mean_r = table.mean(axis = 1)
    mean_c = table.mean(axis = 0)
    mean_all = table.mean().mean()

    l_all = square(table-mean_all).sum()
    l_r = square(mean_r-mean_all).sum()*k
    l_c = square(mean_c-mean_all).sum()*n
    l_e = l_all - l_r - l_c

    # degree of freedom
    v_r = n-1
    v_c = k-1
    v_e = v_r*v_c

    MSR = l_r/v_r
    MSC = l_c/v_c
    MSE = l_e/v_e

    if typeof == 'single':
        ICC = (MSR-MSE)/(MSR+(k-1)*MSE+k*(MSC-MSE)/n)
    elif typeof == 'average':
        ICC = (MSR-MSE)/(MSR+(MSC-MSE)/n)
    else:
        print('No such type!')
    return(ICC)

def Find_Optimal_Cutoff(obs,pred,typeof='Null'):
    """
    Find the optimal cutoff based on ROC curve
    基于ROC曲线寻找最佳的二分类分界点

    Key parameters
    ----------
    typeof:'Null'--0.5
        'Ratio'--the positive ratio
        'TF'--the minimum of absolute value of difference between sensitivity and specificity
        ’Youden'--Youden index:the maximum of sum of sensitivity and specificity,Default
    """
    obs,pred,_ = list_shaping(obs,pred)
    fpr, tpr, threshold = roc_curve(obs, pred, drop_intermediate = False)
    i = arange(len(tpr))
    roc = DataFrame({
        'tf':Series(tpr-(1-fpr), index=i),
        'Y':Series(tpr-fpr),
        'threshold':Series(threshold)})
    if typeof == 'Null':
        value = 0.5
    elif typeof == 'Ratio':
        value = sum(obs == 1)/len(obs)
    elif typeof == 'TF':
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
        value = roc_t['threshold'].values[0]
    elif typeof == 'Youden':
        roc_t = roc.iloc[roc.Y.argsort()[len(tpr)-1:len(tpr)]]
        value = roc_t['threshold'].values[0]
    else:
        raise ValueError('the values of "typeof" are "Null", "Ratio", "TF" and "Youden".')
    return(typeof,value)

def __compare(obs,pred):
    label = list(obs.drop_duplicates().sort_values()) # get the label of obs
    n = sum(obs == label[0]) # length of label[0]
    m = sum(obs == label[1]) # length of label[1]
    num0 = array(pred[obs == label[0]])
    num1 = array(pred[obs == label[1]])
    V10 = list()
    V01 = list()
    for i in range(m):
        V10.append((sum(num0 < num1[i]) + sum(num0 == num1[i])/2)/n)
    for j in range(n):
        V01.append((sum(num1 > num0[j]) + sum(num1 == num0[j])/2)/m)
    return(V10,V01,m,n)

def CI_ROC(obs,pred,typeof='Binomial',percent=95,n_bootstraps=1000):
    """
    Calculate confidence interval of AUC
    计算AUC的置信区间

    Key parameters
    ----------
        typeof: different type of CI calculation
            'Wald','Exact Binomial'(Default),'Modified Wald','Bootstrap'
        percent: range from 0 to 100, Default is 95
        n_bootstraps: the repeating times of bootstraps, only working when typeof = 'Bootstrap', Default is 1000

    Reference
    ----------
        1.DeLong.et al.Comparing the Areas Under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach.BIOMETRICS,44, 837-845,September 1988.
        2.https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    """
    if percent > 100 or percent < 0 :
        raise ValueError('out of range:0-100')
    percent = percent/100 # converted into range of 0-1
    z_upper = norm.ppf(1-(1-percent)/2)
    z_lower = norm.ppf(0+(1-percent)/2)

    obs,pred,N = list_shaping(obs,pred) # length of the list
    V10,V01,m,n = __compare(obs,pred)
    A = sum(V01)/n # A = AUC
    X = A*N
    S10 = (sum(multiply(V10,V10))-m*A*A)/(m-1)
    S01 = (sum(multiply(V01,V01))-n*A*A)/(n-1)
    SE = (S10/m+ S01/n) ** 0.5

    if typeof == 'Wald':
        CI_lower = A - SE * z_upper
        CI_upper = A + SE * z_lower
    elif typeof == 'Binomial':
        C1 = (N-X+1)/X
        F1 = f.ppf(0+(1-percent)/2,2*X,2*(N-X+1))
        CI_lower = 1/(1+C1/F1)
        C2 = (N-X)/(X+1)
        F2 = f.ppf(1-(1-percent)/2,2*(X+1),2*(N-X))
        CI_upper = 1/(1+C2/F2)
    elif typeof == 'Modified Wald':
        P = (X+2)/(N+4)
        W = 2*(P*(1-P)/(N+4)) ** 0.5
        CI_lower = A - W
        CI_upper = A + W
    elif typeof == 'Bootstrap':
        num0 = array(pred[obs == 0])
        num1 = array(pred[obs == 1])
        n_bootstraps = n_bootstraps
        bootstrapped_scores = []
        rng = random.RandomState(0)
        for i in range(n_bootstraps):
            indice0 = rng.randint(0, n, n)
            indice1 = rng.randint(0, m, m)
            prob = append(num0[indice0],num1[indice1])
            true = append(repeat(0,n),repeat(1,m))
            s = roc_auc_score(true, prob)
            bootstrapped_scores.append(s)
        SE = (sum((bootstrapped_scores-mean(bootstrapped_scores)) ** 2)/(n_bootstraps-1)) ** 0.5
        CI_lower = A - SE * z_lower
        CI_upper = A + SE * z_upper
        typeof = '-'.join([typeof,str(n_bootstraps)])
    else:
        raise ValueError('the values of "Wald","Exact Binomial"(Default),"Modified Wald","Bootstrap".')
    if CI_lower < 0: CI_lower = 0
    if CI_upper > 1: CI_upper = 1
    return(typeof,CI_lower,CI_upper)

def Delong_test(obs,pred_group):
    """
    Compare two ROC curves by Delong test.
    
    Reference
    ----------
        1.DeLong.et al.Comparing the Areas Under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach.BIOMETRICS,44, 837-845,September 1988.
    """
    if type(pred_group) != type(DataFrame()):
        raise ValueError('Please transfer the input data into the DataFrame structure!')
    obs,_,_ = list_shaping(obs,pred_group.iloc[:,0]) # renew the obs and get the length of obs
    
    index = DataFrame(index=['V10','V01','AUC'],columns=pred_group.columns)
    for i in pred_group:
        index[i]['V10'],index[i]['V01'],m,n = __compare(obs,pred_group[i])
        index[i]['AUC'] = sum(index[i]['V10'])/len(index[i]['V10'])

    Groups = combinations(pred_group,2) # randomly take two of possible-preds in group
    Delong = DataFrame(index=['Diff','SE','Zvalue','pvalue'])
    L = array([[1,-1]])
    for g in Groups:
        Diff = abs(index[g[0]]['AUC']-index[g[1]]['AUC'])
        Sa = cov(index[g[0]]['V10'],index[g[1]]['V10'])
        Sn = cov(index[g[0]]['V01'],index[g[1]]['V01'])
        S  = Sa/m+Sn/n
        SE = dot(L,S); SE = dot(SE,L.T) ** (0.5); SE=SE[0][0]
        Zvalue = Diff/SE 
        pvalue = norm.sf(Zvalue)*2
        name = '/'.join(g)
        Delong[name] = [Diff,SE,Zvalue,pvalue]
    return(Delong)
      
class Criterion():
    """
    Calculate Akaike Information criterion (AIC) and Bayesian Information creterion(BIC)
    计算AIC和BIC准则

    Key parameters
    ----------
        num_fea: the number of features
    """
    def __init__(self,obs,pred,num_fea):
        self.obs,self.pred,self.__num_smp = list_shaping(obs,pred)
        self.num_fea = num_fea
        self.__SSR = sum((self.obs - self.pred) ** 2) # SSR: SS-residual
    # the formulas and the names of methods follow the functions in R languages
    def AIC(self):
        try:
            return 2*self.num_fea+self.__num_smp*log(float(self.__SSR)/self.__num_smp)
        except:
            raise ValueError('The two inputted lists are same!')
    def extractAIC(self):
        try:
            return 2*(self.num_fea+1)+self.__num_smp*(1+log(2*pi)+log(float(self.__SSR)/self.__num_smp))
        except:
            raise ValueError('The two inputted lists are same!')
    def BIC(self):
        try:
            return log(self.__num_smp)*self.num_fea+self.__num_smp*log(float(self.__SSR)/self.__num_smp)
        except:
            raise ValueError('The two inputted lists are same!')
    def extractBIC(self):
        try:
            return log(self.__num_smp)*(self.num_fea+1)+self.__num_smp*(1+log(2*pi)+log(float(self.__SSR)/self.__num_smp))
        except:
            raise ValueError('The two inputted lists are same!')