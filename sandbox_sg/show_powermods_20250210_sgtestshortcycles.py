#%%
from pathlib import Path
import pandas as pd
import numpy as np
import math

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
    '20250210_sgdev_shortcycles_v31ghlrc4',
]

#%% Load associated datafile from the unit-logged run
# logfile = logfilenames[0];
# df_in,df_events = logreader.scanALogfile(logfile)
dfraw = logreader.processRootFolder(rootpath,experiments_to_plot);
df_events = dfraw[ ~dfraw['Event'].isnull() & ~(dfraw['Event']==' ') ]

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


#%% PLT - Multiple Runs and Units, RTC-Time (lifetest) 2
dfplot = df;

# drop events
#dfplot = dfplot[dfplot['Event']!='Cycle 1 Started.'];
#dfplot = dfplot[dfplot['Event']!='Sample is no longer valid due to timeout.'];

import plotly.graph_objects as go
from plotly.subplots import make_subplots
tracenames  = [
    (('Temps','DegC'),['HeaterTemp','BatteryT']),
    (('PWMs/Percents','%'),['HeaterPWM','MotorPWM','Battery']),
    (('RPMs','RPM'),['MotorSpeed']),
    #(('SOC','%'),['Battery']),
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

# # arrange using PANEL
# fig.layout.autosize = True
# pn_fig = pn.pane.Plotly(fig,);

# pn_config = pn.pane.Str(config_string);

# cfilename = config_file if type(config_file) is str else config_file.name
# pn_final = pn.Tabs( ('Plot',pn_fig) , (cfilename,pn_config) , sizing_mode='stretch_both', width_policy='max');
# #pn_final.save('c:\\TEMP\\NAATOS_SAMPLEPREP_{:s}_PANEL_PLOTLY.html'.format( time.strftime('%Y%m%dT%H%M') ));
