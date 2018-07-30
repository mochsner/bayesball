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
        s = str.lower(s)
        new_col_names.append(s)
    gl.columns = new_col_names
    return gl
def ret_pandas_fields(gl):
    new_col_names = []
    for i in gl.columns.values:
        s = i.replace("_","")
        s = s.replace("NUM","#")
        s = s.replace("",".")
        new_col_names.append(s)
    gl.columns = new_col_names
    return gl

gl = pd.read_csv('../data/GL2017Modified.TXT',sep='\t',\
                 converters={'Visiting Score':np.float64,'Home Score':np.float64})

gl = format_pandas_fields(gl)

gl['bln_home_win'] = (gl['visiting_score'] < gl['home_score']).astype(int)
gl['diff_scores'] = (gl['home_score'] - gl['visiting_score']).astype(int)
gl['diff_hr'] = (gl['home_homeruns'] - gl['visit_homeruns']).astype(int)
gl['diff_hits'] = (gl['home_hits'] - gl['visit_hits']).astype(int)
gl['diff_doubles'] = (gl['home_doubles'] - gl['visit_doubles']).astype(int)
gl['diff_triples'] = (gl['home_triples'] - gl['visit_triples']).astype(int)
gl['diff_rbi'] = (gl['home_rbi'] - gl['visit_rbi']).astype(int)
gl['diff_putouts'] =(gl['home_putouts'] - gl['visit_putouts']).astype(int)
gl['diff_errors'] =(gl['home_errors'] - gl['visit_errors']).astype(int)
gl['diff_assist'] =(gl['home_assist'] - gl['visit_assist']).astype(int)
gl['diff_bats'] =(gl['home_bats'] - gl['visit_bats']).astype(int)

gl.to_csv('../data/GL2017_norm_diff.csv')