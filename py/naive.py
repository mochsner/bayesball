from sklearn import datasets
iris = datasets.load_iris()

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
data = iris.data
target = iris.target
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print(type(iris.target))
print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))
# Number of mislabeled points out of a total 150 points : 6


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
def format_pandas_fields(gl):
    new_col_names = []
    for i in gl.columns.values:
        s = i.replace(" ","_")
        s = s.replace("#","NUM")
        s = s.replace(".","")
        new_col_names.append(s)
    gl.columns = new_col_names
    return gl

gl = pd.read_csv('../data/GL2017Modified.TXT',sep='\t',\
                 converters={'Visiting Score':np.float64,'Home Score':np.float64})
gl = format_pandas_fields(gl)

pf = pd.read_csv('../data/pf_2017.csv')
print(list(pf))
#gl.Visiting_Score = gl.Visiting_Score.fl
gl['Home_Win'] = (gl['Visiting_Score'] < gl['Home_Score']).astype(int)
gl['Home_Diff'] = (gl['Home_Score'] - gl['Visiting_Score']).astype(int)

parks = gl.Park_Code.unique()
teams = gl.Visiting_Team.unique()


# =============================================================================
# #conf_labels = gl.groupby(['Home_Team'])['Home_League']
team_home_win_prob = gl.groupby(['Home_Team'])
new df = pd.DataFrame(team_home_win_prob['Home_Win'].mean(),)
# #team_home_win_prob = team_home_win_prob.sort_values(axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
# #
# #display = pd.DataFrame(dict(x=teams,y=team_home_win_prob.to_frame,label=conf_labels))
# #
# #groups = display.groupby('label')
# #groups.plot(kind='bar')
# =============================================================================
#ax.legend()
#
plt.show()

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(),GaussianNB())
#model.fit()
model.fit()



#team_home_win_prob.plot(kind='bar')
#plt.show()
#team_home_win_prob = pd.concat(gl.groupby(['Visiting_Team']),gl.groupby(['Visiting_Team'])['Home_Win'].mean())

   
    
#sns.stripplot(x="team_home_win_prob", y="total_bill", data=team_home_win_prob);
#plt.xlabel('Teams')
#plt.ylabel('Probability')
#plt.title('Probability of a Home Win')
##plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#plt.axis([40, 160, 0, 0.03])
#plt.grid(True)
#plt.show()


#from sklearn.pipeline import make_pipeline
#mat = 