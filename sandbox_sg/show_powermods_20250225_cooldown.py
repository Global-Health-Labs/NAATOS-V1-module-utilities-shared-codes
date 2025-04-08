#%%
from pathlib import Path
import pandas as pd
import numpy as np
import math
import time

import plotly.express as px
import panel as pn

import naatos_module_tools.logreader as logreader

#%% Setup and Enumerate
root = r'C:\Users\SimonGhionea\Global Health Labs, Inc\NAATOS Product Feasibility - General - Internal - Electronic Control Module\Beta design\PowermoduleTestData\by_exp'
rootpath = Path(root)

experiment_list = [x.name for x in rootpath.iterdir() if x.is_dir()];

experiments_to_plot = [
    #'20250128_sgdev_3.1',
    #'20250128_sgdev_3.1_b_abridge_pwmbugsearch',
    #'20250128_sgdev_3.1_c_abridge_diffpid',
    #'20250128_sgdev_3.1_d_2cycle_emulate',
    
    '20250225_sgdev_cooldown_test',
    '20250228_sgdev_cooldown_test',
]

#%% Load associated datafile from the unit-logged run
# logfile = logfilenames[0];
# df_in,df_events = logreader.scanALogfile(logfile)
dfraw = logreader.processRootFolder(rootpath,experiments_to_plot);
df_events = dfraw[ ~dfraw['Event'].isnull() & ~(dfraw['Event']==' ') ]

#%% Show run summary
runs_full = [];
for run, dfgrp in df_events.groupby('run'):
    if dfgrp[dfgrp['Event'].str.startswith('Cycle 3 Stopped.')].shape[0]<=0:
        continue;
    print(run,dfgrp.shape)
    print(dfgrp[['Time','Event']])
    runs_full.append( run )
    

#%% filter desired runs
#df = dfraw[ dfraw['run']=='sample_01-09-25_093447' ]
#df = dfraw[ dfraw['run']=='sample_02-25-25_134947' ]
df = dfraw[ dfraw['run'].isin(runs_full) ];
#df = dfraw;


# ignore sample no longer valid
selmask = df['Event'].str.startswith('Sample is no longer valid due to timeout').fillna(False).infer_objects(copy=False);
df = df[~selmask]

# ignore any short one-liners where the device starts up fresh on battery
selmask = df['Event'].str.startswith('Sample preperation unit powered on.').fillna(False).infer_objects(copy=False);
df = df[~selmask]

# ignore any short one-liners where the device starts up fresh on battery
selmask = df['Event'].str.startswith('HALL sensor interrupted.').fillna(False).infer_objects(copy=False);
df = df[~selmask]

# #plot heater temp and PWM, and motorspeed, and battery
# build_bokeh_plot(df, x_axis = 'runtime',
#                  y_scale_type = 'linear', xmin = 0, xmax = 600, ymin = 0, ymax = 150, 
#                 #title_text = os.path.split(root)[1],
#                 title_text = 'title_text',
#                 x_scale_type = 'linear', lines_points_both='points', manual_points = manual_points,
#                 save_plot = False, save_loc = root, save_name = savename, show = True);


#%% PLT - Multiple Runs and Units, Run time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = df.copy();

fig = make_subplots(rows=4, cols=1, start_cell="top-left",shared_xaxes=True, vertical_spacing=0.02);

#tracenames  = ['HeaterTemp','HeaterPWM','MotorSpeed','MotorPWM'];
tracenames  = ['ValveTemp','ValvePWM','AmpTemp','AmpPWM'];
tracerowcol = [(1,1),(2,1),(3,1),(4,1)];

nrows = np.array(tracerowcol).max(0)[0];

# tracenames, units, non-units
df['tracelabel'] = df.apply(lambda x: '{:s}_{:s}'.format(x['unit'],x['run'].replace('sample_','')),axis=1);

for tracename,(row,col) in zip(tracenames,tracerowcol):
    dfmelted = df.melt(id_vars=['runtime','tracelabel','run'],value_vars=[tracename],var_name='qty');
    figtmp = px.line(dfmelted,x='runtime',y='value',line_group='run',color='tracelabel')
    for trace in figtmp.data:
        fig.add_trace(trace, row=row,col=col);
    fig.update_yaxes(row=row,col=col,title=tracename);


fig.update_xaxes(showspikes=True,spikemode='across')
fig.update_xaxes(row=nrows,title='runtime (s)')

fig.update_traces(showlegend=False);
fig.update_traces(showlegend=True,row=1,col=1);

fig.update_traces(xaxis="x{:d}".format(nrows))

explist = df['expname'].unique().tolist();
unitlist = df['unit'].unique().tolist();
runlist = df['run'].unique().tolist();
if(len(explist)==1):
    # single experiment
    fig.update_layout(title='Exp {:s} (nunits={:d} nruns={:d})'.format(explist[0],len(unitlist),len(runlist)))

fig.show(renderer='browser')
#fig.write_html('c:\\TEMP\\NAATOS_SAMPLEPREP_{:s}_MULTIDEVRUM_PLOTLY.html'.format( time.strftime('%Y%m%dT%H%M') ));





#%% PLT - Multiple Runs and Units, Run-time, all runs overlayed
dfplot = df;

tracenames  = ['AmpTemp','AmpPWM','BatteryT','BatteryV'];
tracerowcol = [(1,1),(1,1),(1,1),(2,1)];

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
pn_final.save('c:\\TEMP\\NAATOS_POWERMOD_{:s}_PANEL_PLOTLY_RUNtime.html'.format( time.strftime('%Y%m%dT%H%M') ));






#%% PLT - Multiple Runs and Units, RTC-Time (lifetest)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = df.copy();

fig = make_subplots(rows=4, cols=1, start_cell="top-left",shared_xaxes=True, vertical_spacing=0.02);

tracenames  = ['HeaterTemp','HeaterPWM','MotorSpeed','MotorPWM'];
tracerowcol = [(1,1),(2,1),(3,1),(4,1)];

nrows = np.array(tracerowcol).max(0)[0];

# tracenames, units, non-units
#df['tracelabel'] = df.apply(lambda x: '{:s}_{:s}'.format(x['unit'],x['run'].replace('sample_','')),axis=1);

for tracename,(row,col) in zip(tracenames,tracerowcol):
    #dfmelted = df.melt(id_vars=['runtime','tracelabel','run'],value_vars=[tracename],var_name='qty');
    #figtmp = px.line(dfmelted,x='runtime',y='value',line_group='run',color='tracelabel')
    #figtmp = px.line(df,x='Time',y=tracename,hover_data=['unit','run','Event'])
    figtmp = px.line(df,x='Time',y=tracename)
    for trace in figtmp.data:
        fig.add_trace(trace, row=row,col=col);
    fig.update_yaxes(row=row,col=col,title=tracename);


fig.update_xaxes(showspikes=True,spikemode='across')
fig.update_xaxes(row=nrows,title='runtime (s)')

fig.update_traces(showlegend=False);
fig.update_traces(showlegend=True,row=1,col=1);

fig.update_traces(xaxis="x{:d}".format(nrows))

explist = df['expname'].unique().tolist();
unitlist = df['unit'].unique().tolist();
runlist = df['run'].unique().tolist();
if(len(explist)==1):
    # single experiment
    fig.update_layout(title='Exp {:s} (nunits={:d} nruns={:d})'.format(explist[0],len(unitlist),len(runlist)))

fig.show(renderer='browser')
#fig.write_html('c:\\TEMP\\NAATOS_SAMPLEPREP_{:s}_MULTIDEVRUM_PLOTLY.html'.format( time.strftime('%Y%m%dT%H%M') ));


#%% PLT - Multiple Runs and Units, RTC-Time
dfplot = df;

# drop events
#dfplot = dfplot[dfplot['Event']!='Cycle 1 Started.'];
#dfplot = dfplot[dfplot['Event']!='Sample is no longer valid due to timeout.'];

import plotly.graph_objects as go
from plotly.subplots import make_subplots
tracenames  = [
    (('Temps','DegC'),['ValveTemp','AmpTemp','BatteryT']),
    (('PWMs/Percents','%'),['ValvePWM','AmpPWM']),
    (('SOC','%'),['Batt']),
    (('Volts','Volts'),['BatteryV']),
];
fig = make_subplots(rows=len(tracenames), cols=1, start_cell="top-left",shared_xaxes=True, vertical_spacing=0.02);

for row,((subplot_title,ylabel),traces) in enumerate(tracenames):
    figtmp = px.line(dfplot,x='Time',y=traces,hover_data=['run','unit']);
    figtmp.update_traces(legendgroup=row+1);
    for trace in figtmp.data:
        fig.add_trace(trace, row=row+1,col=1);
    fig.update_yaxes(title=ylabel,row=row+1,col=1)

fig.update_layout(legend_tracegroupgap = 250);
#fig.update_traces(connectgaps=False);
fig.update_xaxes(showspikes=True,spikemode='across');
fig.update_xaxes(row=len(tracenames),title='RTC Timestamp');
fig.update_traces(xaxis="x{:}".format(len(tracenames)), connectgaps=False);
nruns = df['run'].unique().shape[0];
#fig.update_layout(hovermode="x unified");
fig.update_layout(title='{:s} UNIT: {:s} #_OF_RUNS:{:d}'.format(dfplot.iloc[0]['expname'],dfplot.iloc[0]['unit'],nruns));

fig.show(renderer='browser');


#%% PLT - 1 Run Overlaied, Run Time
dfplot = df;

# drop events
#dfplot = dfplot[dfplot['Event']!='Cycle 1 Started.'];
#dfplot = dfplot[dfplot['Event']!='Sample is no longer valid due to timeout.'];

import plotly.graph_objects as go
from plotly.subplots import make_subplots
tracenames  = [
    (('Temps','DegC'),['ValveTemp','AmpTemp','BatteryT']),
    (('PWMs/Percents','%'),['ValvePWM','AmpPWM']),
    (('SOC','%'),['Batt']),
    (('Volts','Volts'),['BatteryV']),
];
fig = make_subplots(rows=len(tracenames), cols=1, start_cell="top-left",shared_xaxes=True, vertical_spacing=0.02);

for row,((subplot_title,ylabel),traces) in enumerate(tracenames):
    figtmp = px.line(dfplot,x='runtime',y=traces,hover_data=['run','unit']);
    figtmp.update_traces(legendgroup=row+1);
    for trace in figtmp.data:
        fig.add_trace(trace, row=row+1,col=1);
    fig.update_yaxes(title=ylabel,row=row+1,col=1)

fig.update_layout(legend_tracegroupgap = 150);
#fig.update_traces(connectgaps=False);
fig.update_xaxes(showspikes=True,spikemode='across');
fig.update_xaxes(row=len(tracenames),title='Runtime');
fig.update_traces(xaxis="x{:}".format(len(tracenames)), connectgaps=False);
nruns = df['run'].unique().shape[0];
#fig.update_layout(hovermode="x unified");
fig.update_layout(title='{:s} UNIT: {:s} #_OF_RUNS:{:d}'.format(dfplot.iloc[0]['expname'],dfplot.iloc[0]['unit'],nruns));

fig.show(renderer='browser');
#fig.write_html('c:\\TEMP\\NAATOS_POWERMOD_{:s}_1RUN_RUNTIME_PLOTLY.html'.format( time.strftime('%Y%m%dT%H%M') ));


#%% PLT - 2025-03 - plot
dfplot = df;

# columns = run

import plotly.graph_objects as go
from plotly.subplots import make_subplots
dfplot['tracelabel'] = dfplot.apply(lambda x: '{:s}_{:s}'.format(x['unit'],x['run'].replace('sample_','')),axis=1);
tracenames  = [
    (('Temps','DegC'),['ValveTemp','AmpTemp','BatteryT']),
    (('PWMs/Pcnts','%'),['ValvePWM','AmpPWM']),
    (('SOC','%'),['Batt']),
    (('Volts','Volts'),['BatteryV']),
];
facetcol = 'tracelabel';
xvals = 'runtime';
nrows = len(tracenames);
ncols = len(dfplot[facetcol].unique().tolist());

fig = make_subplots(
    rows=len(tracenames),
    cols=ncols,
    start_cell="top-left",
    shared_xaxes='all', vertical_spacing=0.02, horizontal_spacing=0.02,
    shared_yaxes='rows'
);

for row,((subplot_title,ylabel),traces) in enumerate(tracenames):
    
    dfmelted = dfplot.melt(
        id_vars=['runtime','tracelabel','run','unit'],
        value_vars=traces,
        var_name='qty'
    );

    # use plotly-express to quickly generate these traces
    figtmp = px.line(
        dfmelted,
        x=xvals,
        y='value',
        hover_data=['run','unit'],
        color='qty',
        facet_col=facetcol,
        markers=True,
    );

    # put the plotly express traces into our greater figure we are assembling (with subplots)
    figtmp.update_traces(legendgroup=row+1,legendgrouptitle_text=subplot_title);
    for trace in figtmp.data:
        c = 1;
        if(trace.xaxis == 'x2'):
            c = 2;
        fig.add_trace(trace, row=row+1,col=c);
    if(row==0):
        # titles at the top of each column
        for cnt,annotation in enumerate(figtmp.layout.annotations):
            fig.add_annotation(
                text=annotation.text.split('=')[1],
                xref="x domain", yref="y domain",
                yanchor='bottom',
                valign='bottom',x=0.5, y=1.0,
                #font_size=20,
                font=dict(weight="bold",size=16),
                showarrow=False,
                row=1,col=cnt+1
            );
    if(subplot_title=='Temps'):
        # ambient temperature estimation.... take from first value with nonzero runtime
        first_data = dfplot[(dfplot['runtime']<2.0) & (dfplot['runtime']>0)].groupby(facetcol).first()
        for cnt,(grp,firstvals) in enumerate(first_data.iterrows()):
            fig.add_hline(y=firstvals['AmpTemp'],line_width=3, line_dash="dash", line_color="red",
                          showlegend=True,
                          legendgroup=row+1,
                          name='Amp Initial',
                          label_text='Amp initial value {:.1f}degC'.format(firstvals['AmpTemp']),
                          label_yanchor="top",
                          row=row+1,col=cnt+1
                          )
    fig.update_yaxes(title=ylabel,row=row+1,col=1)
    #break;

# force to share same x-axis
#fig.update_traces(xaxis="x{:}".format(len(tracenames)));

#fig.update_shapes(selector=dict(type="line"), xref="x2 domain")

explist = df['expname'].unique().tolist();
unitlist = df['unit'].unique().tolist();
runlist = df['run'].unique().tolist();
if(len(explist)==1):
    # single experiment
    fig.update_layout(title='Exp {:s} (nunits={:d} nruns={:d})'.format(explist[0],len(unitlist),len(runlist)))
#fig.update_layout(hovermode="y unified")
fig.update_xaxes(showspikes=True,spikemode='across');
fig.update_layout(legend=dict(groupclick="toggleitem"))

# fig.update_layout(
#     margin=dict(l=20, r=20, t=20, b=20),
# )

# set markers and line widths
fig.update_traces(
    marker=dict(size=4),
    line_width=1.5
)

# setup axes and labels labels
fig.update_xaxes(dtick=300,scaleanchor='x',scaleratio=1,constrain='domain');
fig.update_xaxes(title=xvals+' [seconds]',row=nrows);
fig.update_yaxes(nticks=10);    # more ticks on y-axes

# setup legends per row
for i, yaxis in enumerate(fig.select_yaxes(col=ncols), 1):
    legend_name = f"legend{i}"
    fig.update_layout({legend_name: dict(y=yaxis.domain[1], yanchor="top")}, showlegend=True)
    fig.update_traces(row=i, legend=legend_name)

# title
explist = dfgrp['expname'].unique().tolist();
unitlist = dfgrp['unit'].unique().tolist();
runlist = dfgrp['run'].unique().tolist();
fig.update_layout(title='Exps {:s} (nunits={:d} nruns={:d})'.format(str(explist),len(unitlist),len(runlist)))


fig.show(renderer='browser')
fig.write_html('c:\\TEMP\\NAATOS_POWERMOD_{:s}_2runs_sidebyside_RUNTIME_PLOTLY.html'.format( time.strftime('%Y%m%dT%H%M') ));

# %%
