# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 07:00:13 2018

@author: marc ochsner
"""

from sklearn import datasets
iris = datasets.load_iris()

gnb = GaussianNB()

from sklearn.utils import validation
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from numpy import ravel

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

#from mpl_toolkits.axes_grid1 import AxesGrid


def format_pandas_fields(gl):
    new_col_names = []
    for i in gl.columns.values:
        s = i.replace(" ", "_")
        s = s.replace("#", "NUM")
        s = s.replace(".", "")
        new_col_names.append(s)
    gl.columns = new_col_names
    return gl


gl = pd.read_csv('../data/GL2017_norm_diff.csv',
                 converters={'Visiting Score': np.float64, 'Home Score': np.float64})
gl = format_pandas_fields(gl)

# =============================================================================
# Plot of all possible differences
# =============================================================================
#sns.pairplot(gl[['diff_scores','diff_hr','diff_hits','diff_doubles','diff_triples','diff_rbi','diff_putouts','diff_errors','diff_assist','diff_bats','bln_home_win']],hue='bln_home_win',size=1.5)

model = GaussianNB()

# INPUTS: 'date','visiting_team','home_team'

# Relative game data
gl = gl[['diff_scores', 'diff_hr', 'diff_hits', 'diff_doubles', 'diff_triples',
         'diff_rbi', 'diff_putouts', 'diff_errors', 'diff_assist', 'diff_bats', 'bln_home_win']]

# ALL
gl = gl[gl.columns.difference(
    ['diff_doubles', 'diff_triples', 'diff_hits', 'diff_putouts'])]

Xtrain, Xtest, ytrain, ytest = train_test_split(gl[gl.columns.difference(['bln_home_win', 'diff_scores'])],
                                                ravel(gl[['bln_home_win']]),
                                                random_state=1,
                                                test_size=.25)
#    print(listgl[['diff_scores']])
model.fit(Xtrain, ytrain)
print(model.score(Xtrain, ytrain))
y_model = model.predict(Xtest)

print(str(list(gl[gl.columns.difference(['bln_home_win', 'diff_scores'])])
          )+" -> "+str(accuracy_score(ytest, y_model)))
plt.show()
# Individual Analysis --- deprecate?
i = 0
for COL in list(gl):
        try:
            print(i)
            i = i+1
            Xtrain, Xtest, ytrain, ytest = train_test_split(gl[[COL]],
                                                            ravel(
                                                                gl[['diff_score']]),
                                                            random_state=1,
                                                            test_size=.25)
            model.fit(Xtrain, ytrain)
            y_model = model.predict(Xtest)

            print(COL+" -> "+str(accuracy_score(ytest, y_model)))
            mat = confusion_matrix(ytest, y_model)
            sns.heatmap(mat, square=True, annot=True, cbar=False)
#            grid.cbar_axes[0]
            plt.xlabel('pred val')
            plt.ylabel('true val')
            plt.title(COL)
        except:
            print("")


# =============================================================================
# # GaussianBN based on only Visitor Data
# print("just visits_______")
# visit = gl
# for col in list(visit):
#     print(col)
#     if "visit" not in col:
#         print(col)
#         if (col != 'diff_score'):
#             visit.drop(columns=[col],inplace=True)
# #visit.drop(columns=['visiting_team','visiting_team_game','visiting_league'],inplace=True)
# print("LIST OF VISITORS")
# print(list(visit))
# try:
#     Xtrain, Xtest, ytrain, ytest = train_test_split(visit[visit.columns.difference(['diff_score'])],\
#                                                ravel(visit[['diff_score']]),\
#                                                random_state=1,\
#                                                test_size = .75) #
#     model.fit(Xtrain, ytrain)
#     y_model = model.predict(Xtest)
#
#     print(str(list(visit[visit.columns.difference(['bln_home_win'])]))+" -> "+str(accuracy_score(ytest,y_model)))
# except:
#     print("err")
# =============================================================================

# =============================================================================
# # Confusion matrix
# from sklearn.metrics import confusion_matrix
# mat = confusion_matrix(ytest,y_model)
# plt.xlabel('pred val')
# plt.ylabel('true val')
# =============================================================================

#data = iris.data
#target = iris.target
#y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
#print(type(iris.target))
#print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))
## Number of mislabeled points out of a total 150 points : 6
#
#pf = pd.read_csv('../data/pf_2017.csv')
