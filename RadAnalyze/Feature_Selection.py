# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 11:11:00 2018

@author: Caizhengting
"""

import os
from numpy import *
from pandas import *
import matplotlib.pyplot as plt
from own_packages.SN import *
from own_packages.preprocess import *
from own_packages.Final_Model import *
from itertools import combinations,cycle
from collections import Counter
from copy import copy

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.linear_model import Lasso,lasso_path,LassoCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve,roc_auc_score

class LowVariance(paras):
    """
    Exclude the features with low variance, better utilized in the categorical. 
    """
    def __init__(self,X,name,threshold=0.1,fpath='./',size=10):
        super().__init__(X=X,name=name,fpath=fpath,size=size)
        self.threshold = threshold

    def selection(self,threshold=0.1):
        try:self.threshold = threshold = min(self.threshold,threshold)
        except:pass
        LV = VarianceThreshold(threshold=threshold).fit(self.X)
        LV_scores = DataFrame(LV.variances_,columns=['Variance'],index=self.X.columns)[LV.get_support()]
        LV_scores = LV_scores.sort_values('Variance',ascending=False) # get_support() get T or F of each feature
        LV_scores.index = LV_scores.index.rename('Features')
        X_LV = self.X[LV_scores.index]
        print("Categorical features reduced from {0} to {1} by LowVarianceThreshold".format(len(self.X.columns),len(X_LV.columns)))
        self.__X_LV = X_LV
        self.__LV_scores = LV_scores
        try:
            X_LV.to_csv(os.path.join(self.fpath,'_'.join([self.name,'X_LV.csv'])))
        except PermissionError:
            X_LV.to_csv(os.path.join(self.fpath,'_'.join([self.name,'X_LV(01).csv'])))
        try:
            LV_scores.to_csv(os.path.join(self.fpath,'_'.join([self.name,'X_LV_Var.csv'])))
        except PermissionError:
            LV_scores.to_csv(os.path.join(self.fpath,'_'.join([self.name,'X_LV_Var(01).csv'])))
        return(X_LV,LV_scores)

    def plot(self):
        filter_all = [i.split('_')[0] for i in self.X.columns]
        filter_sel = [i.split('_')[0] for i in self.__X_LV.columns]
        count_all = Counter(filter_all)
        count_sel = Counter(filter_sel)
        
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0.2)
        bars = 0.4
        ax.barh(arange(len(count_all))-bars/2,list(count_all.values()),bars,color ='b',label='the original features')
        ax.barh(arange(len(count_all))+bars/2,list(count_sel.values()),bars,color='lightblue',label='the selected features')
        ax.set_xlabel('Number')
        ax.set_ylabel('Filters')
        ax.set_ylim([-1,len(count_all)+0.6])
        ax.set_title('Features selected by VarianceThreshold')
        ax.set_yticks(arange(len(count_all)))
        ax.set_yticklabels(count_all.keys())
        ax.legend()

        for i in range(len(count_all)):
            value_all = list(count_all.values())[i]
            value_sel = list(count_sel.values())[i]
            ax.text(value_all+.5, i-bars/1.5, value_all)
            ax.text(value_sel+.5, i+bars/2.5, value_sel)
        fig.savefig(os.path.join(self.fpath,'_'.join([self.name,'X_LV.png'])))

class Kbest_ANOVA(paras):
    """
    Exclude the continuous features with ANOVA
    """
    def __init__(self,X,y,name,fpath='./',size=10):
       super().__init__(X=X,y=y,name=name,fpath=fpath,size=size)
    
    def selection(self):
        sel_KB = SelectKBest(f_classif,k = 'all').fit(self.X,self.y)
        sel_KB_scores = array([sel_KB.scores_,sel_KB.pvalues_]).T
        sel_KB_scores = DataFrame(sel_KB_scores,index = self.X.columns,columns=['F-statistic','p'])
        sel_KB_scores = sel_KB_scores.sort_values('F-statistic',ascending= False)
        sel_KB_scores = sel_KB_scores[sel_KB_scores.p< 0.05]
        sel_KB_scores.index = sel_KB_scores.index.rename('Features')
        X_KB = self.X[sel_KB_scores.index]
        print("Continuous features reduced from {0} to {1} by SelectKBest".format(len(self.X.columns),len(X_KB.columns)))
        self.__X_KB = X_KB
        self.__sel_KB_scores = sel_KB_scores       
        try:
            X_KB.to_csv(os.path.join(self.fpath,'_'.join([self.name,'X_KB.csv'])))
        except PermissionError:
            X_KB.to_csv(os.path.join(self.fpath,'_'.join([self.name,'X_KB(01).csv'])))
        try:
            sel_KB_scores.to_csv(os.path.join(self.fpath,'_'.join([self.name,'X_KB_stats.csv'])))
        except PermissionError:
            sel_KB_scores.to_csv(os.path.join(self.fpath,'_'.join([self.name,'X_KB_stats(01).csv'])))
        return(X_KB,sel_KB_scores)

    def plot(self):
        fig, ax = plt.subplots(figsize=(self.size, len(self.__sel_KB_scores)/35*self.size))
        fig.subplots_adjust(left=0.5,right=0.9,top=0.95,bottom=0.05)
        ax.barh(arange(len(self.__sel_KB_scores)),self.__sel_KB_scores['F-statistic'])
        ax.set_xlabel('F-value')
        ax.set_ylabel('Features')
        ax.set_title('F-statistic of features selected by ANOVA')
        ax.set_ylim([-1,len(self.__sel_KB_scores)])
        ax.set_yticks(arange(len(self.__sel_KB_scores)))
        ax.set_yticklabels(self.__sel_KB_scores.index)
        fig.savefig(os.path.join(self.fpath,'_'.join([self.name,'X_KB.png'])))

class LASSO(paras):
    """
    Exclude the features by LASSO

    steps
    ---------
        1.MSEPath
        2.LassoPath
        3.Lasso_origin
        ……
    """
    def __init__(self,X,y,name,fpath='./',size=10):
       super().__init__(X=X,y=y,name=name,fpath=fpath,size=size)

    def MSEPath(self,max_iter=10000,cv=10):
        sel_Lasso = LassoCV(cv=cv,max_iter=max_iter).fit(self.X,self.y)
        sel_log_LassoAlphas = -log10(sel_Lasso.alphas_)
        sel_log_LassoAlpha =  -log10(sel_Lasso.alpha_)
        print('Alpha:', sel_Lasso.alpha_)

        fig, ax = plt.subplots()
        ax.plot(sel_log_LassoAlphas, sel_Lasso.mse_path_, ':')
        ax.plot(sel_log_LassoAlphas, sel_Lasso.mse_path_.mean(axis=-1), 'k',label='Average across the folds', linewidth=self.size/5)
        ax.axvline(sel_log_LassoAlpha, linestyle='--', color='k',label='$\\alpha$: CV estimate')
        xunit,yunit = plot_unit([ax.get_xlim(),ax.get_ylim()])
        ax.text(sel_log_LassoAlpha-xunit, ax.get_ylim()[1]+yunit, round(sel_log_LassoAlpha,2))
        ax.legend(loc='best')
        ax.set_title('Mean Square Error Path')
        ax.set_xlabel('-log($\\alpha$)')
        ax.set_ylabel('Mean square error')
        fig.savefig(os.path.join(self.fpath,'_'.join([self.name,'X_Lasso_MSEPath.png'])))
        return(sel_Lasso.alpha_)

    def LassoPath(self,alpha,eps=1e-2): # eps the smaller it is the longer is the path
        sel_log_LassoAlpha =  -log10(alpha)
        alphas_lasso, coefs_lasso, _ = lasso_path(self.X, self.y, eps, fit_intercept=False)
        colors = cycle(['b', 'r', 'g', 'c', 'k'])
        neg_log_LassoAlphas = -log10(alphas_lasso)
        
        fig, ax = plt.subplots() 
        for coef_l, c in zip(coefs_lasso,colors):
            ax.plot(neg_log_LassoAlphas, coef_l, c=c)
        ax.axvline(sel_log_LassoAlpha, linestyle='--', color='k',label='$\\alpha$: CV estimate')
        xunit,yunit = plot_unit([ax.get_xlim(),ax.get_ylim()])
        ax.text(sel_log_LassoAlpha-xunit, ax.get_ylim()[1]+yunit, round(sel_log_LassoAlpha,2))
        ax.set_xlabel('-log($\\alpha$)')# or # ax.set_xlabel(r'-log($\alpha$)')
        ax.set_ylabel('Coefficients')
        ax.set_title('Lasso Path')
        fig.savefig(os.path.join(self.fpath,'_'.join([self.name,'X_LassoPath.png'])))

    def Lasso_origin(self,alpha):
        LassoModel = Lasso(alpha).fit(self.X,self.y)
        sel_FeaName = self.X.columns[LassoModel.coef_ != 0] # get the selected features names
        Lasso_Coefs = DataFrame(LassoModel.coef_[LassoModel.coef_ != 0],index = sel_FeaName,columns = ['Coefficients'])
        Lasso_Coefs = Lasso_Coefs.sort_values('Coefficients')
        Lasso_Coefs.index = Lasso_Coefs.index.rename('Features')
        X_Lasso = self.X[sel_FeaName]
        X_predict = DataFrame(LassoModel.predict(self.X),index = X_Lasso.index,columns = ['RadScore'])
        print("\nFeatures reduced from {0} to {1} by LASSO".format(len(self.X.columns),len(sel_FeaName)))

        concat([Lasso_Coefs,DataFrame(LassoModel.intercept_,index=['intercept(non-feature)'],columns=['Coefficients'])]).to_csv(
                os.path.join(self.fpath,'_'.join([self.name,'X_Lasso_Coefs.csv']))) # add the intercept and then export
        X_Lasso.to_csv(os.path.join(self.fpath,'_'.join([self.name,'X_Lasso_origin.csv'])))
        X_predict.to_csv(os.path.join(self.fpath,'_'.join([self.name,'X_Lasso_predict.csv'])))

        if len(Lasso_Coefs)/30*self.size <= self.size/1.5:
            ysize = self.size/1.5
        else:
            ysize = len(Lasso_Coefs)/30*self.size
        fig, ax = plt.subplots(figsize = (self.size,ysize))
        fig.subplots_adjust(left=0.55,right=0.9)
        ax.barh(arange(len(Lasso_Coefs)),Lasso_Coefs.iloc[:,0])
        ax.set_xlabel('Coefficients')
        ax.set_ylabel('Features')
        ax.set_yticks(arange(len(Lasso_Coefs)))
        ax.set_yticklabels(Lasso_Coefs.index)
        ax.set_title("Coefficients in the Lasso Model")
        fig.savefig(os.path.join(self.fpath,'_'.join([self.name,'X_Lasso_Coefs.png'])))
        print(Lasso_Coefs)
        return(X_Lasso,Lasso_Coefs)

class Further_Selection(paras):
    """
    Reduce the feature number further based the previous selection method to avoid overfitting

    Methods
    ----------
        1.Ranking: select the absolute values of coefficients ranking in the top
         * the number depended by 'max_FeaNum'
        2.RFE_CV
        3.Forward_CV
        4.Complete_CV
    """
    def __init__(self,X,y,name,model='',fpath='./',size=10):
       super().__init__(model=model,X=X,y=y,name=name,fpath=fpath,size=size)

    def __plot(self,scoring,scores,step,threshold,plotname):
        fig, ax = plt.subplots()
        ax.set_xlabel('Number of features selected')
        ax.set_ylabel('Cross validation score(average %s)' %scoring)
        xlim = []
        for i in range(len(scores)): # calculate the mean score by step
            if i*step+1 > len(self.X.columns):
                xlim.append(len(self.X.columns))
            else:
                xlim.append(i*step+1)
        ax.plot(xlim, scores,'-o')
        if max(scores) >= threshold:
            ax.axhline(threshold, linestyle='--', color='r')
            xunit,yunit = plot_unit([ax.get_xlim(),ax.get_ylim()])
            ax.text(ax.get_xlim()[0]+xunit, threshold+yunit, threshold)
        fig.savefig(os.path.join(self.fpath,'_'.join([self.name,plotname,'png'])))

    def Ranking(self,origin_Coefs,max_FeaNum=5):
        coef = origin_Coefs.abs().sort_values(origin_Coefs.columns[0],ascending = False)
        FeaNum = min(len(self.X.columns),max_FeaNum)
        coef_further = coef.iloc[:FeaNum,]
        X_further = self.X[coef_further.index]
        return(X_further,coef_further)

    def RFE_CV(self,n_splits=5,scoring='roc_auc',step=1,threshold=0.85):
        rfecv = RFECV(estimator=self.model, step=step, cv=StratifiedKFold(n_splits),scoring=scoring)
        rfecv.fit(self.X, self.y)
        for i in range(len(rfecv.grid_scores_)):
            if rfecv.grid_scores_[i] >= threshold:
                rfecv.estimator.fit(self.X,self.y)
                coef = DataFrame(rfecv.estimator.coef_,columns=self.X.columns,index=['Coefficients']).T
                coef = coef.abs().sort_values('Coefficients',ascending = False)
                X_further = self.X[coef.iloc[:1+step*i,].index]
                break
        try: X_further
        except:
            X_further = self.X[self.X.columns[rfecv.support_]]
        Further_Selection.__plot(self,scoring,rfecv.grid_scores_,step,threshold,'RFECV')
        return(X_further,rfecv)

    def Forward_CV(self,origin_Coefs,n_splits=5,scoring='roc_auc',step=1,threshold=0.85):
        coef = origin_Coefs.abs().sort_values(origin_Coefs.columns[0], ascending = False).index
        mean_scores = Series()
        for i in range(int(ceil((len(coef)+step-1)/step))): # calculate the mean score by step
            if i*step+1 > len(coef):
                feat_tmp = coef
            else:
                feat_tmp = coef[0:(i*step+1)]
            self_tmp = copy(self)
            self_tmp.X = self.X[feat_tmp]
            score_tmp,_,_= CV_learning(self_tmp,n_splits,scoring)
            mean_scores.loc[len(feat_tmp)] = mean(score_tmp)
        for i in mean_scores.index: # choose the optimal one
            if mean_scores.loc[i] >= threshold:
                X_further = self.X[coef[0:i]]
                break
        try: X_further
        except:
            X_further = self.X[coef[0:mean_scores.idxmax()]]
        Further_Selection.__plot(self,scoring,mean_scores,step,threshold,'ForwardCV')
        return(X_further,mean_scores.rename('_'.join(['mean',scoring])))
   
    def Complete_CV(self,n_splits=5,scoring='roc_auc'):
        coef = self.X.columns
        mean_scores = Series()
        for n in range(len(coef)):
            for i in combinations(coef,n+1):
                name_tmp = '/'.join(i)
                self_tmp = copy(self)
                self_tmp.X = self.X[array(i)]
                score_tmp,_,_= CV_learning(self_tmp,n_splits,scoring)
                mean_scores.loc[name_tmp] = mean(score_tmp)
        if scoring == 'aic' or scoring == 'bic':
            mean_scores = mean_scores.sort_values()
        else:
            mean_scores = mean_scores.sort_values(ascending=False)
        X_further = self.X[mean_scores.index[0].split('/')]
        return(X_further,mean_scores.rename('_'.join(['mean',scoring])))

class Filter_CV(paras):
    """
    Exclude the features by filters in the cross-validation
    """
    def __init__(self,filter,X,y,ID_test,ID_train,name,fpath='./',size=10):
        super().__init__(model=filter,X=X,y=y,ID_test=ID_test,ID_train=ID_train,name=name,fpath=fpath,size=size)

    def selection(self,n_splits=5):
        try: shape(self.ID_test) == shape(self.ID_train)
        except:
            self.ID_train,self.ID_test = get_CVID_CSV(self.y,n_splits=n_splits,name=self.name,fpath=self.fpath)      

        X_train_group = []
        y_train_group = []
        X_test_group = []
        y_test_group = []
        coef_group = []
        print('-----Start Filter_CV-----')
        for cv in range(len(self.ID_test.columns)):
            fpath_out = os.path.join(self.fpath,str(cv))
            if not os.path.exists(fpath_out):os.makedirs(fpath_out)
            id_test = self.ID_test.iloc[:,cv].dropna(axis = 0).apply(int) # get the ID
            id_train = self.ID_train.iloc[:,cv].dropna(axis = 0).apply(int)

            X_test = self.X.reindex(id_test) # split into training set and test set of X
            X_train = self.X.reindex(id_train)
            X_train.to_csv(os.path.join(fpath_out,'_'.join([self.name,'train',str(cv),'X.csv']))) #store
            X_test.to_csv(os.path.join(fpath_out,'_'.join([self.name,'test',str(cv),'X.csv'])))
            y_test = self.y.reindex(id_test) # split into training set and test set of y
            y_train = self.y.reindex(id_train)
            y_train.to_csv(os.path.join(fpath_out,'_'.join([self.name,'train',str(cv),'y.csv'])))
            y_test.to_csv(os.path.join(fpath_out,'_'.join([self.name,'test',str(cv),'y.csv'])))

            self_tmp = paras(X=X_train,y=y_train,fpath=fpath_out,name='_'.join([self.name,'train',str(cv)]))
            X_train_filter, coef = self.model.selection(self_tmp)
            self.model.plot(self_tmp)
            X_test_filter = X_test[X_train_filter.columns]
            X_test_filter.to_csv(os.path.join(fpath_out,'_'.join([self.name,'test',str(cv),'X_filter.csv'])))
            X_test_group.append(X_test_filter)
            X_train_group.append(X_train_filter)
            y_test_group.append(y_test)
            y_train_group.append(y_train)
            coef_group.append(coef)
            print('-----{0}-----'.format(cv))
        print('-----Finish!-----')

        output = paras(
            X_train=X_train_group,
            X_test=X_test_group,
            y_train=y_train_group,
            y_test=y_test_group,
            coef=coef_group)
        return(output)
    
class LASSO_CV(paras):
    """
    Exclude the features by LASSO in the cross-validation
    """
    def __init__(self,X,y,ID_test,ID_train,name,fpath='./',size=10):
        super().__init__(X=X,y=y,ID_test=ID_test,ID_train=ID_train,name=name,fpath=fpath,size=size)

    def selection(self,n_splits=5,inner_n_splits=5,max_iter=10000):
        try: shape(self.ID_test) == shape(self.ID_train)
        except:
            self.ID_train,self.ID_test = get_CVID_CSV(self.y,n_splits=n_splits,name=self.name,fpath=self.fpath)

        X_train_group = []
        y_train_group = []
        X_test_group = []
        y_test_group = []
        coef_group = []
        print('-----Start LASSO_CV-----')
        for cv in range(len(self.ID_test.columns)):
            fpath_out = os.path.join(self.fpath,str(cv))
            if not os.path.exists(fpath_out):os.makedirs(fpath_out)
            id_test = self.ID_test.iloc[:,cv].dropna(axis = 0).apply(int) # get the ID
            id_train = self.ID_train.iloc[:,cv].dropna(axis = 0).apply(int)

            X_test = self.X.reindex(id_test) # split into training set and test set of X
            X_train = self.X.reindex(id_train)
            scaler = StandardScaler().fit(X_train) # standardization
            X_train = DataFrame(scaler.transform(X_train),columns = X_train.columns,index=X_train.index)
            X_test = DataFrame(scaler.transform(X_test),columns = X_test.columns,index=X_test.index)
            X_train.to_csv(os.path.join(fpath_out,'_'.join([self.name,'train',str(cv),'X_Sd.csv']))) #store
            X_test.to_csv(os.path.join(fpath_out,'_'.join([self.name,'test',str(cv),'X_Sd.csv'])))
            y_test = self.y.reindex(id_test) # split into training set and test set of y
            y_train = self.y.reindex(id_train)
            y_train.to_csv(os.path.join(fpath_out,'_'.join([self.name,'train',str(cv),'y.csv'])))
            y_test.to_csv(os.path.join(fpath_out,'_'.join([self.name,'test',str(cv),'y.csv'])))

            self_tmp = paras(X=X_train,y=y_train,fpath=fpath_out,name='_'.join([self.name,'train',str(cv)]))
            alpha = LASSO.MSEPath(self_tmp,max_iter,inner_n_splits)
            LASSO.LassoPath(self_tmp,alpha)
            X_train_Lasso,coef = LASSO.Lasso_origin(self_tmp,alpha)
            X_test_Lasso = X_test[X_train_Lasso.columns]
            X_test_Lasso.to_csv(os.path.join(fpath_out,'_'.join([self.name,'test',str(cv),'X_Lasso_origin.csv'])))
            X_test_group.append(X_test_Lasso)
            X_train_group.append(X_train_Lasso)
            y_test_group.append(y_test)
            y_train_group.append(y_train)
            coef_group.append(coef)
            print('-----{0}-----'.format(cv))
        print('-----Finish!-----')

        output = paras(
            X_train=X_train_group,
            X_test=X_test_group,
            y_train=y_train_group,
            y_test=y_test_group,
            coef=coef_group)
        return(output)

class Further_Selection_CV(paras):
    """
    Reduce the feature number further based the previous selection method in the cross-validation
    """
    def __init__(self,X_train,X_test,y_train,y_test,coef,model,typeof,name,fpath='./',size=10):
        super().__init__(model=model,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,name=name,fpath=fpath,size=size)
        self.typeof=typeof
    
    def selection(self,max_FeaNum=5,n_splits=5,scoring='roc_auc',step=1,threshold=0.85):
        X_train_group = []
        y_train_group = []
        X_test_group = []
        y_test_group = []
        coef_group = []
        cv = 0
        print('-----Start Further_Selection_CV-----')
        for X_test,X_train,y_test,y_train,coef in zip(self.X_test,self.X_train,self.y_test,self.y_train,self.coef):
            fpath_out = os.path.join(self.fpath,str(cv))
            name = '_'.join([self.name,'train',str(cv)])
            self_tmp = paras(model=self.model,X=X_train,y=y_train,fpath=fpath_out,name=name)
            if self.typeof == Further_Selection.Ranking:
                X_train_fur,coef_fur = self.typeof(self_tmp,coef,max_FeaNum)
            elif self.typeof == Further_Selection.RFE_CV:
                X_train_fur,coef_fur = self.typeof(self_tmp,n_splits,scoring,step,threshold)
            elif self.typeof == Further_Selection.Forward_CV:
                X_train_fur,coef_fur = self.typeof(self_tmp,coef,n_splits,scoring,step,threshold)
            elif self.typeof == Further_Selection.Complete_CV:
                X_train_fur,coef_fur = self.typeof(self_tmp,n_splits,scoring)
            X_test_fur = X_test[X_train_fur.columns]
            X_train_fur.to_csv(os.path.join(fpath_out,'_'.join([self.name,'train',str(cv),'X_Further.csv'])))
            X_test_fur.to_csv(os.path.join(fpath_out,'_'.join([self.name,'test',str(cv),'X_Further.csv'])))

            X_test_group.append(X_test_fur)
            X_train_group.append(X_train_fur)
            y_test_group.append(y_test)
            y_train_group.append(y_train)
            coef_group.append(coef_fur)
            print('-----{0}-----'.format(cv))
            cv += 1
        print('-----Finish!-----')
        
        output = paras(
            X_train=X_train_group,
            X_test=X_test_group,
            y_train=y_train_group,
            y_test=y_test_group,
            coef=coef_group)
        return(output)