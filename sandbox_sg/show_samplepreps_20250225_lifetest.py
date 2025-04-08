#%%
"""

Generate sample-prep plots useful for
looking at the lifetest
2025-02

2 units, many runs per unit

"""
#%%
from pathlib import Path
import pandas as pd
import numpy as np
import math
import time

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import panel as pn

import naatos_module_tools.logreader as logreader

#%% Setup and Input
#root = r'C:\Users\SimonGhionea\Global Health Labs, Inc\NAATOS Product Feasibility - General - Internal - Electronic Control Module\Beta design\SamplePrepTestData\by_exp'
root = r'C:\Users\SimonGhionea\Global Health Labs, Inc\NAATOS Product Feasibility - General - Internal - Electronic Control Module\Beta design\SamplePrepLifeTest'
rootpath = Path(root)

experiment_list = [x.name for x in rootpath.iterdir() if x.is_dir()];

experiments_to_plot = [
    #'20250206_sgdev_ghllifetime',
    'exp20250210',
    'exp20250220',
    'exp20250221',
]

#%% Load associated datafiles from the unit-logged run
dfraw = logreader.processRootFolder(rootpath,experiments_to_plot);
df_events = dfraw[ ~dfraw['Event'].isnull() ]

#%% filter the data
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


#%% PLT - Multiple Runs and Units, RTC Time (lifetest)
dfplot = df;

tracenames  = [
    (('Temps','DegC'),['HeaterTemp','BatteryT']),
    #(('PWMs/Percents','%'),['HeaterPWM','MotorPWM','Battery']),
    (('PWMs/Percents','%'),['MotorPWM']),
    (('RPMs','RPM'),['MotorSpeed']),
    #(('SOC','%'),['Battery']),
    (('Volts','Volts'),['BatteryV']),
];

# make a tab pane per unit
panel_tabs = [];

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

pn_final = pn.Tabs(*panel_tabs,sizing_mode='stretch_both', width_policy='max')
pn_final.save('c:\\TEMP\\NAATOS_SAMPLEPREP_{:s}_PANEL_PLOTLY_RTCtime.html'.format( time.strftime('%Y%m%dT%H%M') ));

#%% PLT - Multiple Runs and Units, Run-time (lifetest), all runs overlayed
dfplot = df;

tracenames  = ['HeaterTemp','HeaterPWM','MotorSpeed','MotorPWM'];
tracerowcol = [(1,1),(2,1),(3,1),(4,1)];

# make a tab pane per unit
panel_tabs = [];
for (unit,dfgrp) in dfplot.groupby('unit'):
    print('Working on unit',unit);
    fig = make_subplots(rows=len(tracenames), cols=1, start_cell="top-left",shared_xaxes=True, vertical_spacing=0.02);
    nrows = np.array(tracerowcol).max(0)[0];

    # tracenames, units, non-units
    dfgrp['tracelabel'] = dfgrp.apply(lambda x: '{:s}_{:s}'.format(x['unit'],x['run'].replace('sample_','')),axis=1);

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

#arrange using panel
pn_final = pn.Tabs(*panel_tabs,sizing_mode='stretch_both', width_policy='max')
pn_final.save('c:\\TEMP\\NAATOS_SAMPLEPREP_{:s}_PANEL_PLOTLY_RUNtime.html'.format( time.strftime('%Y%m%dT%H%M') ));

#%% summarize the lifetest runs - pulling out only motor from cycles 3 and 4

dfbuildlist = [];
for unit,dfunits in df.groupby('unit'):
    unit_run_counter = 0;
    for run,dfrun in dfunits.groupby('run'):
        #print('{:s} run: {:s} shape: {:s}'.format(unit,run,str(dfrun.shape)))

        # total run time , should be at least 350 seconds for us to count it
        if((dfrun.Time.iloc[-1]-dfrun.Time.iloc[0]).total_seconds() > 350.0):
            # all good
            pass;
        else:
            print('{:s} {:s} shp={:s} skipping due to incomplete run'.format(unit,run,str(dfrun.shape)))
            continue;

        # find Cycle 3 and Cycle 4 boundaries, and pull subset (if applicable)
        dfrunevents = dfrun['Event'][~dfrun['Event'].isnull()];
        matchidx = dfrunevents[
                dfrunevents.isin(['Cycle 3 Started.','Cycle 4 Stopped.'])
            ].index;
        try:
            assert(matchidx.shape[0]==2);
        except AssertionError as e:
            print('{:s} {:s} shp={:s} skipping due to not finding cycle events'.format(unit,run,str(dfrun.shape)))
            continue
        idxstart = matchidx[0]+1;
        idxstop = matchidx[1]-1;

        # extract only the data from cycles 3 and 4
        dfrunsubset = dfrun.loc[idxstart:idxstop].copy();

        # assign a run_number incrementally
        unit_run_counter+=1;
        dfrunsubset['run_number'] = unit_run_counter;

        dfbuildlist.append(dfrunsubset);
dflifetest = pd.concat(dfbuildlist)

#%% PLT - Life test - boxplot
def mkfig_lifetestsummary_boxplot():
    fig = px.box(dflifetest,
                x='run_number',y='MotorPWM',color='unit',
                hover_data='run',
                boxmode='group',points='outliers'
            );
    fig.update_layout(title='SP Lifetest MotorPWM Boxplots for Cycle 3 and Cycle 4')
    return fig;
fig = mkfig_lifetestsummary_boxplot();
#fig.show(renderer='browser')

#%% PLT - Life test - only medians plot
def mkfig_lifetestsummary_medians():
    catorders = dflifetest['unit'].unique().tolist();
    # MOTORPWM plots
    # plot just the medians of cycle 3&4 as a scatter
    dfmedians = dflifetest.groupby(['run_number','run','unit'])['MotorPWM'] \
        .median().reset_index();
    figtmp = px.scatter(dfmedians,
                x='run_number',y='MotorPWM',color='unit',
                hover_data='run',category_orders={'unit':catorders},
                labels={'MotorPWM':'MotorPWM% median'}
            );
    figtmp.update_layout(title='SP Lifetest MotorPWM Medians for Cycle 3 and Cycle 4')
    return figtmp;
fig = mkfig_lifetestsummary_medians();
#fig.show(renderer='browser')


#%% PLT - Life test - 2 subplot medians and boxplot
def mkfig_2plt_medians_and_boxplot():
    fig = make_subplots(rows=2, cols=1,
            start_cell="top-left",shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('MotorPWM Medians in Cycle3&4','MotorPWM Boxplots in Cycle3&4'),
        );

    catorders = dflifetest['unit'].unique().tolist();
    # MOTORPWM plots
    # plot just the medians of cycle 3&4 as a scatter
    dfmedians = dflifetest.groupby(['run_number','run','unit'])['MotorPWM'] \
        .median().reset_index();
    figtmp = px.scatter(dfmedians,
                x='run_number',y='MotorPWM',color='unit',
                hover_data='run',category_orders={'unit':catorders}
            );
    for trace in figtmp.data:
        fig.add_trace(trace, row=1,col=1);

    # make boxplots of cycle 3&4
    figtmp = px.box(dflifetest,
                x='run_number',y='MotorPWM',color='unit',
                hover_data='run',category_orders={'unit':catorders},
                boxmode='group',points='outliers'
            );
    for trace in figtmp.data:
        fig.add_trace(trace, row=2,col=1);

    fig.update_layout(title='SP Lifetest MotorRPM for Cycle 3 and Cycle 4')
    #fig.update_layout(xaxis_type='category');
    #fig.update_xaxes(xaxis_type='category',row=2);
    fig.update_traces(xaxis="x{:d}".format(2))
    fig.update_xaxes(title='run_number',row=2)
    fig.update_yaxes(title='MotorPWM [%]');

    return fig;

fig = mkfig_2plt_medians_and_boxplot();
fig.show(renderer='browser')

#%% PLT - Life test - summary tabs
panel_tabs = [];

# add tab with this figure to a Panel tabset
fig = mkfig_lifetestsummary_medians();
fig.layout.autosize = True;
pn_fig = pn.pane.Plotly(fig);
panel_tabs.append( ('MotorPWM Medians for Cycle 3&4', pn_fig) );

# add tab with this figure to a Panel tabset
fig = mkfig_lifetestsummary_boxplot();
fig.layout.autosize = True;
pn_fig = pn.pane.Plotly(fig);
panel_tabs.append( ('MotorPWM Boxplots for Cycle 3&4', pn_fig) );


#fig.show(renderer='browser')

#arrange using panel
pn_final = pn.Tabs(*panel_tabs,sizing_mode='stretch_both', width_policy='max')
pn_final.save('c:\\TEMP\\NAATOS_SAMPLEPREP_{:s}_PANEL_PLOTLY_summary.html'.format( time.strftime('%Y%m%dT%H%M') ));


