import matplotlib.pyplot as plt|\label{line:importplt}|
import numpy as np|\label{line:importnp}|
...
import pandas as pd|\label{line:importpd}|

...

def ann_post( yv, disp = True, graph = True):
    """
    After ann_pre and shell command, ann_post can be used.
    """
    df_ann = pd.read_csv( 'ann_out.csv') |\label{line:read}|
    yv_ann = np.mat( df_ann['out'].tolist()).T |\label{line:mat}|
    
    r_sqr, RMSE = ann_show( yv, yv_ann, disp = disp, graph = graph) |\label{line:plotcall}|

    return r_sqr, RMSE

...

def regress_show( yEv, yEv_calc, disp = True, graph = True, plt_title = None):
...
    if len( np.shape(yEv_calc)) == 1: |\label{line:lencall}|
        yEv_calc = np.mat( yEv_calc).T
...
    plt.plot(yEv.tolist(), yEv_calc.tolist(), '.', ms = ms_sz) |\label{line:plot}|

...

ann_show = regress_show |\label{line:ann_show}|
