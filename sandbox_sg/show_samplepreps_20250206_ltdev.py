#%%
from pathlib import Path
import pandas as pd
import numpy as np
import math
import time

import plotly.express as px
import panel as pn

import naatos_module_tools.logreader as logreader

#%% Setup and Input
root = r'C:\Users\SimonGhionea\Global Health Labs, Inc\NAATOS Product Feasibility - General - Internal - Electronic Control Module\Beta design\SamplePrepTestData\by_exp'
rootpath = Path(root)

experiment_list = [x.name for x in rootpath.iterdir() if x.is_dir()];

experiments_to_plot = [
    #'20250206_sgdev_ghllifetime',
    '20250210_sgdev_ghllifetime',
]

#%% Load associated datafiles from the unit-logged run
dfraw = logreader.processRootFolder(rootpath,experiments_to_plot);
df_events = dfraw[ ~dfraw['Event'].isnull() ]

#%% filter desired runs
#df = dfraw[ dfraw['run']=='sample_01-09-25_093447' ]
#df = dfraw[ dfraw['run']=='sample_01-09-25_093447' ]
df = dfraw;

# ignore sample no longer valid
selmask = df['Event'].str.startswith('Sample is no longer valid due to timeout').fillna(False).infer_objects(copy=False);
df = df[~selmask]

# ignore any short one-liners where the device starts up fresh on battery
selmask = df['Event'].str.startswith('Sample preperation unit powered on.').fillna(False).infer_objects(copy=False);
df = df[~selmask]

# ignore any short one-liners where the device starts up fresh on battery
selmask = df['Event'].str.startswith('HALL sensor interrupted.').fillna(False).infer_objects(copy=False);
df = df[~selmask]


#%% PLT - Multiple Runs and Units, Run-time (lifetest) 2
dfplot = df;

# drop events
#dfplot = dfplot[dfplot['Event']!='Cycle 1 Started.'];
#dfplot = dfplot[dfplot['Event']!='Sample is no longer valid due to timeout.'];

import plotly.graph_objects as go
from plotly.subplots import make_subplots
tracenames  = [
    (('Temps','DegC'),['HeaterTemp','BatteryT']),
    #(('PWMs/Percents','%'),['HeaterPWM','MotorPWM','Battery']),
    (('PWMs/Percents','%'),['MotorPWM']),
    (('RPMs','RPM'),['MotorSpeed']),
    #(('SOC','%'),['Battery']),
    (('Volts','Volts'),['BatteryV']),
];

# make a tab pane per unit
#units = df['unit'].unique().tolist();
panel_tabs = [];
#for unit in units:
for (unit,dfgrp) in df.groupby('unit'):
    print('Working on unit',unit);
    fig = make_subplots(rows=len(tracenames), cols=1, start_cell="top-left",shared_xaxes=True, vertical_spacing=0.02);

    for row,((subplot_title,ylabel),traces) in enumerate(tracenames):
        figtmp = px.line(dfgrp,x='Time',y=traces,hover_data=['run']);
        figtmp.update_traces(legendgroup=row+1);
        for trace in figtmp.data:
            fig.add_trace(trace, row=row+1,col=1);
        fig.update_yaxes(title=ylabel,row=row+1,col=1)

    fig.update_layout(legend_tracegroupgap = 180);
    #fig.update_traces(connectgaps=False);
    fig.update_xaxes(showspikes=True,spikemode='across');
    fig.update_xaxes(row=len(tracenames),title='RTC Timestamp');
    fig.update_traces(xaxis="x{:}".format(len(tracenames)), connectgaps=True);
    nruns = dfgrp['run'].unique().shape[0];
    fig.update_layout(title='{:s} UNIT: {:s} #_OF_RUNS:{:d}'.format(dfgrp.iloc[0]['expname'],dfgrp.iloc[0]['unit'],nruns));

    # add tab with this figure to a Panel tabset
    fig.layout.autosize = True
    pn_fig = pn.pane.Plotly(fig);
    panel_tabs.append( ('{:} Plot'.format(unit) , pn_fig) );

#fig.show(renderer='browser');

# arrange using PANEL
# fig.layout.autosize = True
# pn_fig = pn.pane.Plotly(fig,);

# pn_config = pn.pane.Str(config_string);

# cfilename = config_file if type(config_file) is str else config_file.name
# pn_final = pn.Tabs( ('Plot',pn_fig) , (cfilename,pn_config) , sizing_mode='stretch_both', width_policy='max');
pn_final = pn.Tabs(*panel_tabs,sizing_mode='stretch_both', width_policy='max')
pn_final.save('c:\\TEMP\\NAATOS_SAMPLEPREP_{:s}_PANEL_PLOTLY_RTCtime.html'.format( time.strftime('%Y%m%dT%H%M') ));

#%% PLT - Multiple Runs and Units, Run-time (lifetest) 3
dfplot = df;


tracenames  = ['HeaterTemp','HeaterPWM','MotorSpeed','MotorPWM'];
tracerowcol = [(1,1),(2,1),(3,1),(4,1)];


# make a tab pane per unit
#units = df['unit'].unique().tolist();
panel_tabs = [];
#for unit in units:
for (unit,dfgrp) in dfplot.groupby('unit'):
    print('Working on unit',unit);
    fig = make_subplots(rows=len(tracenames), cols=1, start_cell="top-left",shared_xaxes=True, vertical_spacing=0.02);
    nrows = np.array(tracerowcol).max(0)[0];

    # tracenames, units, non-units
    df['tracelabel'] = df.apply(lambda x: '{:s}_{:s}'.format(x['unit'],x['run'].replace('sample_','')),axis=1);

    for tracename,(row,col) in zip(tracenames,tracerowcol):
        dfmelted = dfgrp.melt(id_vars=['runtime','tracelabel','run'],value_vars=[tracename],var_name='qty');
        figtmp = px.line(dfmelted,x='runtime',y='value',line_group='run',color='tracelabel')
        for trace in figtmp.data:
            fig.add_trace(trace, row=row,col=col);
        fig.update_yaxes(row=row,col=col,title=tracename);


    fig.update_xaxes(showspikes=True,spikemode='across')
    fig.update_xaxes(row=nrows,title='runtime (s)')

    fig.update_traces(showlegend=False);
    fig.update_traces(showlegend=True,row=1,col=1);

    fig.update_traces(xaxis="x{:d}".format(nrows))

    explist = dfgrp['expname'].unique().tolist();
    unitlist = dfgrp['unit'].unique().tolist();
    runlist = dfgrp['run'].unique().tolist();
    if(len(explist)==1):
        # single experiment
        fig.update_layout(title='Exp {:s} (nunits={:d} nruns={:d})'.format(explist[0],len(unitlist),len(runlist)))

    # add tab with this figure to a Panel tabset
    fig.layout.autosize = True
    pn_fig = pn.pane.Plotly(fig);
    panel_tabs.append( ('{:} Plot'.format(unit) , pn_fig) );

#fig.show(renderer='browser')

#arrange using panel
pn_final = pn.Tabs(*panel_tabs,sizing_mode='stretch_both', width_policy='max')
pn_final.save('c:\\TEMP\\NAATOS_SAMPLEPREP_{:s}_PANEL_PLOTLY_RUNtime.html'.format( time.strftime('%Y%m%dT%H%M') ));
