#%%
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import math
import time

import plotly.express as px
import panel as pn

import naatos_module_tools.logreader as logreader

#%% Setup and Enumerate
#root = r'C:\Users\SimonGhionea\Global Health Labs, Inc\NAATOS Product Feasibility - General - Internal - Electronic Control Module\Beta design\PowermoduleTestData\by_exp'
root = r'C:\Temp\NAATOS_MODULE_AUTODOWNLOADS'
rootpath = Path(root)

experiment_list = [x.name for x in rootpath.iterdir() if x.is_dir()];

experiments_to_plot = [
    #'20250128_sgdev_3.1',
    #'20250128_sgdev_3.1_b_abridge_pwmbugsearch',
    #'20250128_sgdev_3.1_c_abridge_diffpid',
    #'20250128_sgdev_3.1_d_2cycle_emulate',
    
    #'20250225_sgdev_cooldown_test',
    #'20250228_sgdev_cooldown_test',

    #'20250311_DX_download',
    
    '20250310_SimonGhionea_autodl',
    '20250311_SimonGhionea_autodl',
    '20250312_SimonGhionea_autodl',
    '20250313_SimonGhionea_autodl_PM_eelab_continuous',
]

#%% Load associated datafile from the unit-logged run
# logfile = logfilenames[0];
# df_in,df_events = logreader.scanALogfile(logfile)
dfraw = logreader.processRootFolder(rootpath,experiments_to_plot);
df_events = dfraw[ ~dfraw['Event'].isnull() & ~(dfraw['Event']==' ') ]

#%% Filter
df = dfraw;

#df = df[df['expname']=='20250311_SimonGhionea_autodl']
#df = df[df['unit']=='PM11'];
#df = df[df['run']=='sample_03-11-25_153808']
#df = df[df['run']=='sample_03-11-25_153400']

#%% Pull out by-cycle information

#dfbuildlist = [];
buildlist = [];
for unit,dfunits in df.groupby('unit'):
    #unit_run_counter = 0;
    for expname,dfexps in dfunits.groupby('expname'):
        for run,dfrun in dfexps.groupby('run'):
            print('unit:{:s} exp:{:} run:{:s} shape:{:s}'.format(unit,expname,run,str(dfrun.shape)))
            
            dfrunevents = dfrun[ ~dfrun['Event'].isnull() & ~(dfrun['Event']==' ') ]
            dfrunevents = dfrunevents[['Time','Event']]
            
            status = 'nostatus';
            text = '';

            # Cycle Analysis
            evts_beg = dfrunevents[dfrunevents['Event'].str.match(r'^Cycle \d+ Started')]
            evts_end = dfrunevents[dfrunevents['Event'].str.match(r'^Cycle \d+ Stopped')]
            evts_nocyc = dfrunevents.loc[dfrunevents.index.symmetric_difference(evts_beg.index.tolist()+evts_end.index.tolist())]
            
            if( evts_beg.shape[0]==0 ):
                text+='\tNot A Run: {:s}'.format(str(dfrunevents['Event'].to_list()));
                status = 'notrun';
                
                dfrunevents['expname'] = expname;
                dfrunevents['unit'] = unit;
                dfrunevents['run'] = run;
                dfrunevents['status'] = status;
                dfrunevents['text'] = text;
                dfrunevents['Cycle'] = 0;
                buildlist.append(dfrunevents);
            elif( evts_beg.shape[0]==evts_end.shape[0]):
                #print('here5')
                # all Start and Stopped are matched
                ncycles = evts_beg.shape[0];
                cyc_num_beg = evts_beg['Event'].str.extract(r'Cycle (\d+) ',expand=False).astype(np.uint8);
                cyc_num_beg.name='Cycle'
                cyc_num_end = evts_end['Event'].str.extract(r'Cycle (\d+) ',expand=False).astype(np.uint8);
                cyc_num_end.name='Cycle'
                earlyterm = any(evts_end['Event'].str.match(r'.*early\.'));
                if(cyc_num_beg.shape[0]>0):
                    fn_lmbda_split_keyvalue_strings = lambda x: dict(tuple([tuple(z.split('=')) for z in x]));

                    #cyc_data_beg = evts_beg['Event'].str.extract(r'Cycle \d.*\.+( .*)',expand=False).str.split(' ').apply(lambda x: x[1:]).apply(lambda x: dict(tuple([tuple(z.split('=')) for z in x]))).apply(pd.Series)
                    #cyc_data_end = evts_end['Event'].str.extract(r'Cycle \d.*\.+( .*)',expand=False).str.split(' ').apply(lambda x: x[1:]).apply(lambda x: dict(tuple([tuple(z.split('=')) for z in x]))).apply(pd.Series)
                    cyc_data_beg = evts_beg['Event'].str.extract(r'Cycle \d.*\.+( .*)',expand=False).str.split(' ');
                    if(not all(cyc_data_beg.isna())):
                        # there are fields after "Cycle N started." messages
                        cyc_data_beg = cyc_data_beg.apply(lambda x: x[1:]).apply(fn_lmbda_split_keyvalue_strings).apply(pd.Series)

                        evts_beg = pd.concat((evts_beg,cyc_num_beg,cyc_data_beg),axis='columns')
                        #print('here1');
                    else:
                        evts_beg = pd.concat((evts_beg,cyc_num_beg),axis='columns');
                        #print('here2');
                    evts_beg = evts_beg.rename(columns={'Time':'TimeBeg'})


                    cyc_data_end = evts_end['Event'].str.extract(r'Cycle \d.*\.+( .*)',expand=False).str.split(' ');
                    if(not all(cyc_data_end.isna())):
                        # there are fields after "Cycle N started." messages
                        cyc_data_end = cyc_data_end.apply(lambda x: x[1:]).apply(fn_lmbda_split_keyvalue_strings).apply(pd.Series)

                        evts_end = pd.concat((evts_end,cyc_num_end,cyc_data_end),axis='columns')
                        #print('here3')
                    else:
                        evts_end = pd.concat((evts_end,cyc_num_end),axis='columns');
                        #print('here4');
                    evts_end = evts_end.rename(columns={'Time':'TimeEnd'})

                    # evts_beg = pd.concat((evts_beg,cyc_num_beg,cyc_data_beg),axis='columns')
                    # evts_beg = evts_beg.rename(columns={'Time':'TimeBeg'})

                    # evts_end = pd.concat((evts_end,cyc_num_end,cyc_data_end),axis='columns')
                    # evts_end = evts_end.rename(columns={'Time':'TimeEnd'})
                    # #cyc_data_beg.set_index(cyc_num,inplace=True);
                    
                    #cyc_data_end.set_index(cyc_num,inplace=True);
                    cyc_data = pd.merge(evts_beg,evts_end,on='Cycle',suffixes=('Beg','End')).set_index('Cycle',drop=True);

                    # calculate cycle information
                    cyc_data['CCycleRTCSeconds'] = (cyc_data['TimeEnd']-cyc_data['TimeBeg']).apply(lambda x: x.total_seconds());
                    #cyc_data.loc[0,['EventEnd']] = 'norm'
                else:
                    print('\tOTHER')

                
                if(earlyterm):                   
                    # print('\tEarly abort in cycle {:d} @ {:.0f} s @ total runtime {:.0f} s:'.format(
                    #     cyc_data.index[-1],
                    #     cyc_data['CalcCycleRTCSeconds'].iloc[-1].item(),
                    #     cyc_data['CalcCycleRTCSeconds'].sum().item()
                    #     )
                    # )
                    text += '\tEarly abort in cycle {:d} @ {:.0f} s @ total runtime {:.0f} s:'.format(
                        cyc_data.index[-1],
                        cyc_data['CCycleRTCSeconds'].iloc[-1].item(),
                        cyc_data['CCycleRTCSeconds'].sum().item()
                    );
                    #print('\t',str(evts_nocyc['Event'].to_list()))
                    text += '\t{:s}'.format(str(evts_nocyc['Event'].to_list()));
                    normal_interruptions = [
                        'HALL sensor interrupted',
                        'Optical sensor interrupted',
                        'ButtonCycle cancled via button click',
                    ]
                    if(any(pd.concat([evts_nocyc['Event'].str.contains(s) for s in normal_interruptions]))):
                        status = 'run_ended_early_user';
                    else:
                        status = 'run_ended_early_other';
                    # if(evts_nocyc['Event'].str.contains('Optical sensor interrupted')|evts_nocyc['Event'].str.contains('Hall sensor interrupted')):
                    #     status = 'run_ended_early';
                else:
                    # print('\tCompletedRun @ total runtime {:.0f} s'.format(
                    #     cyc_data['CalcCycleRTCSeconds'].sum().item()
                    # ))
                    text += '\tCompletedRun @ total runtime {:.0f} s'.format(
                        cyc_data['CCycleRTCSeconds'].sum().item()
                    )
                    status = 'run_success';
                
                try:
                    if(not earlyterm):
                        if( not all(cyc_data_beg['runtime_s']==cyc_data_end['expected_sec']) ):
                            #print('\tMismatched runtimes');
                            text += '\n\tMismatched runtimes';
                            status = 'run_mismatched_runtimes';
                    else:
                        if( not all(cyc_data_beg['runtime_s'][0:-1]==cyc_data_end['expected_sec'][0:-1]) ):
                            #print('\tMismatched runtimes (early)');
                            text += '\n\tMismatched runtimes (early)';
                            status = 'run_mismatched_runtimes_early';
                except:
                    pass;
                
                # SUMMARIZE PER RUN
                if(not earlyterm):
                    pass;
                    # for (k1,v1),(k2,v2) in zip(evts_beg.iterrows(),evts_end.iterrows()):
                    #     #print( 'Cycle{:d} idx{:d} to idx{:d}'.format(v1['Cycle'],k1,k2) );
                        
                    #     # subset this cycle in the run
                    #     dfruncyc = dfrun.loc[k1:k2]

                    #     # last third
                    #     dfruncyclst3rd = dfruncyc.loc[k2-((k2-k1)//3):]

                    #     # calculate some summaries
                    #     #cyc_data.loc[v1['Cycle'],'C'] = cyc_data['CCycleRTCSeconds'].sum().item()
                    #     cyc_data.loc[v1['Cycle'],'CL3MedAmpTemp'] = dfruncyclst3rd['AmpTemp'].median()
                    #     cyc_data.loc[v1['Cycle'],'CL3MedValveTemp'] = dfruncyclst3rd['ValveTemp'].median()
                    #     cyc_data.loc[v1['Cycle'],'CL3MaxAmpPWM'] = dfruncyclst3rd['AmpPWM'].max()
                    #     cyc_data.loc[v1['Cycle'],'CL3MaxValvePWM'] = dfruncyclst3rd['ValvePWM'].max()
                    #     cyc_data.loc[v1['Cycle'],'CMaxAmpTemp'] = dfruncyc['AmpTemp'].max()
                    #     cyc_data.loc[v1['Cycle'],'CMaxValveTemp'] = dfruncyc['ValveTemp'].max()

                    #     pass
                for (k1,v1),(k2,v2) in zip(evts_beg.iterrows(),evts_end.iterrows()):
                    #print( 'Cycle{:d} idx{:d} to idx{:d}'.format(v1['Cycle'],k1,k2) );
                    
                    # subset this cycle in the run
                    dfruncyc = dfrun.loc[k1:k2]

                    # last third
                    dfruncyclst3rd = dfruncyc.loc[k2-((k2-k1)//3):]

                    # calculate some summaries
                    #cyc_data.loc[v1['Cycle'],'C'] = cyc_data['CCycleRTCSeconds'].sum().item()
                    #cyc_data.loc[v1['Cycle'],'C'] = cyc_data['CCycleRTCSeconds'].sum().item()
                    cyc_data.loc[v1['Cycle'],'CL3MedAmpTemp'] = dfruncyclst3rd['AmpTemp'].median()
                    cyc_data.loc[v1['Cycle'],'CL3MedValveTemp'] = dfruncyclst3rd['ValveTemp'].median()
                    cyc_data.loc[v1['Cycle'],'CL3MaxAmpPWM'] = dfruncyclst3rd['AmpPWM'].max()
                    cyc_data.loc[v1['Cycle'],'CL3MaxValvePWM'] = dfruncyclst3rd['ValvePWM'].max()
                    cyc_data.loc[v1['Cycle'],'CMaxAmpTemp'] = dfruncyc['AmpTemp'].max()
                    cyc_data.loc[v1['Cycle'],'CMaxValveTemp'] = dfruncyc['ValveTemp'].max()

                    pass
                #cyc_data['CRTCRuntime'] = (dfrun['Time'].iloc[[0,-1]]).diff().apply(lambda x: x.total_seconds()).iloc[-1].item();
                cyc_data['CRTCRuntime'] = ( cyc_data.iloc[-1]['TimeEnd'] - cyc_data.iloc[0]['TimeBeg'] ).total_seconds();

                cyc_data['expname'] = expname;
                cyc_data['unit'] = unit;
                cyc_data['run'] = run;
                cyc_data['status'] = status;
                cyc_data['text'] = text;
                buildlist.append(cyc_data.reset_index());
            else:
                text += '\tRun cycles non-sensical, nbegun={:d} nended={:d}'.format(evts_beg.shape[0],evts_end.shape[0]);
                status = 'nonsensical_cycles';

                dfrunevents['expname'] = expname;
                dfrunevents['unit'] = unit;
                dfrunevents['run'] = run;
                dfrunevents['status'] = status;
                dfrunevents['text'] = text;
                dfrunevents['Cycle'] = 0;
                buildlist.append(dfrunevents);
            # buildlist.append(dict(
            #     unit=unit,expname=expname,run=run,
            #     cyc_data=cyc_data
            # ));
            print(text)
            #break;


#dflifetest = pd.concat(dfbuildlist)
dfbuilt = pd.concat(buildlist,ignore_index=True)
dfbuilt['Cycle'] = dfbuilt['Cycle'].astype(np.uint8)
dfbuilt = dfbuilt.set_index(['unit','expname','run','Cycle'])
#dfbuilt = pd.concat(buildlist).reset_index()

#%% Show run summary
#for (unit,expname,run),dfgrp in dfbuilt.xs(1,level='Cycle').groupby(['unit','expname','run']):
for (unit,expname,run),dfgrp in dfbuilt.groupby(['unit','expname','run']):
    df1st = dfgrp.reset_index().iloc[0];
    print(unit,expname,run,df1st.shape)
    print(df1st['text'])

#%% Output Summary Table
import panel as pn
#mydf = dfbuilt.reorder_levels(['expname','unit','run','Cycle']);
# mydf = dfbuilt.reset_index().set_index(['expname','unit']);
# mydf = mydf[['run','Cycle','TimeBeg','TimeEnd','status','text']];
mydf = dfbuilt.reset_index().set_index(['expname','unit']);
#mydf = dfbuilt.reset_index()
#mydf = mydf[['run','Cycle','TimeBeg','status','text']];

#mydf.apply(lambda x: '<a href="file:///c:\\TEMP\\runtime_{:s}_{:s}_{:s}.html">link</a>'.format( x['exp'],'b','c' ),axis=1)
def mkrunlink(x):
    if((x['status'] == 'run_success') or (x['status'].find('run_ended_early')>=0)):
        return '<a href="file:///c:\\TEMP\\NAATOS_PM_RUN_runtime_{:s}_{:s}_{:s}.html" target="_blank">{:s}</a>'.format( x['expname'],x['unit'],x['run'],x['run'] );
    else:
        return x['run'];
    #return 'test';
ret = dfbuilt.reset_index().apply( lambda x: mkrunlink(x) ,axis=1);
mydf['run'] = ret.tolist();

# simplify Time field
ret = mydf.reset_index().apply( lambda x: x['TimeBeg'] if pd.isna(x['Time'])==True else x['Time'],axis=1);
mydf['Time'] = ret.tolist();

# Choose Columns
mydf = mydf[['run','status','Time','Cycle',*dfbuilt.columns[dfbuilt.columns.str.startswith('C')].tolist(),'text']];

# STYLING
def style_statustext(val):
    """
    Takes a scalar and returns a string with
    the css property
    """
    if(val=='run_success'):
        return "background-color: lightgreen";
    elif(val=='run_ended_early_user'):
        return "background-color: lightyellow";
    elif(val=='notrun'):
        return '';
    else:
        return "background-color: orchid";
mydf = mydf.sort_values(['run','Time']);
mydf = mydf.style.map(style_statustext,subset=pd.IndexSlice[:, ['status']]);

header_filters = {
    'texy': {'type': 'input', 'func': 'like', 'placeholder': 'Search text'},
}
tabulator_editors = {
    #'float': {'type': 'number', 'max': 10, 'step': 0.1},
    #'bool': {'type': 'tickCross', 'tristate': True, 'indeterminateValue': None},
    'status': {'type': 'list', 'valuesLookup': True},
    'text': {'type': 'input', 'func': 'like', 'placeholder': 'Search text'},
}
pn_summarytable = pn.widgets.Tabulator(
    value=mydf,
    #pagination='local', page_size=20,
    pagination='local',page_size=1000,
    #hierarchical=True,
    #groupby=['expname','unit'],
    #theme='midnight',
    aggregators={"origin": "mean", "yr": "mean"},
    formatters = {
        'run': dict(type='html')
    },
    #header_filters=header_filters,
    editors=tabulator_editors,header_filters=True,
    sizing_mode='stretch_both',
);
# pn_summarytable = pn.widgets.DataFrame(
#     value=mydf,
#     hierarchical=True,
#     sizing_mode='stretch_both', width_policy='max',autosize_mode='fit_viewport'
# );

pn_final = pn.Column(pn_summarytable);
pn_final.save(r'C:\TEMP\summaryinfo.html');

#%% Show run summary
runs_full = [];
for run, dfgrp in df_events.groupby('run'):
    if dfgrp[dfgrp['Event'].str.startswith('Cycle 3 Stopped')].shape[0]<=0:
        continue;
    print(run,dfgrp.shape)
    print(dfgrp[['Time','Event']])
    runs_full.append( run )
    

#%% filter desired runs
#df = dfraw[ dfraw['run']=='sample_01-09-25_093447' ]
#df = dfraw[ dfraw['run']=='sample_02-25-25_134947' ]
df = dfraw[ dfraw['run'].isin(runs_full) ];
#df = dfraw;


# # ignore sample no longer valid
# selmask = df['Event'].str.startswith('Sample is no longer valid due to timeout').fillna(False).infer_objects(copy=False);
# df = df[~selmask]

# # ignore any short one-liners where the device starts up fresh on battery
# selmask = df['Event'].str.startswith('Sample preperation unit powered on.').fillna(False).infer_objects(copy=False);
# df = df[~selmask]

# # ignore any short one-liners where the device starts up fresh on battery
# selmask = df['Event'].str.startswith('HALL sensor interrupted.').fillna(False).infer_objects(copy=False);
# df = df[~selmask]

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

#%% PLT - 2025-03-12 - plot
#dfplot = df;
dfplot = dfraw[dfraw['run']=='sample_03-06-25_133432'];

# columns = run
def mkplot(dfplot):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    dfplot['tracelabel'] = dfplot.apply(lambda x: '{:s}_{:s}'.format(x['unit'],x['run'].replace('sample_','')),axis=1);
    tracenames  = [
        (('Temps','DegC'),['ValveTemp','AmpTemp','BatteryT']),
        (('PWMs/Pcnts','%'),['ValvePWM','AmpPWM','Batt']),
        #(('SOC','%'),['Batt']),
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

    explist = dfplot['expname'].unique().tolist();
    unitlist = dfplot['unit'].unique().tolist();
    runlist = dfplot['run'].unique().tolist();
    if(len(explist)==1):
        # single experiment
        fig.update_layout(title='Exp {:s} (nunits={:d} nruns={:d})'.format(explist[0],len(unitlist),len(runlist)))

    if(len(explist)==1 and len(runlist)==1 and len(unitlist)==1):
        # we are plotting a single run

        # show cycle lines using dfbuilt summary data
        dfcyc = dfbuilt.xs(explist[0],level='expname').xs(unitlist[0],level='unit').xs(runlist[0],level='run')[['TimeEnd','EventEnd']];
        dfcyc['runtime_end'] = (dfcyc['TimeEnd']-dfplot.iloc[0]['Time']).apply(lambda x: x.total_seconds())
        for k,v in dfcyc.iterrows():
            fig.add_vline(x=v['runtime_end'],line_width=1, line_dash="dash", line_color="black",
                showlegend=False,
                #legendgroup=row+1,
                #name='Amp Initial',
                label_text=v['EventEnd'],
                #label_yanchor="top",
                #row=row+1,col=cnt+1
            )
        
        # handle other non-cycle events
        dfevt = dfplot[(~dfplot['Event'].str.startswith('Cycle')) & (dfplot['Event']!=' ')];

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
    #fig.update_xaxes(dtick=60,scaleanchor='x',scaleratio=1,constrain='domain');
    fig.update_xaxes(nticks=60,range=(0,dfplot.iloc[-1]['runtime']));    # more ticks on x-axes
    fig.update_xaxes(title=xvals+' [seconds]',row=nrows);
    fig.update_yaxes(nticks=20);    # more ticks on y-axes

    # setup legends per row
    for i, yaxis in enumerate(fig.select_yaxes(col=ncols), 1):
        legend_name = f"legend{i}"
        fig.update_layout({legend_name: dict(y=yaxis.domain[1], yanchor="top")}, showlegend=True)
        fig.update_traces(row=i, legend=legend_name)

    # title
    explist = dfplot['expname'].unique().tolist();
    unitlist = dfplot['unit'].unique().tolist();
    runlist = dfplot['run'].unique().tolist();
    fig.update_layout(title='Exps {:s} (nunits={:d} nruns={:d})'.format(str(explist),len(unitlist),len(runlist)))


    #fig.show(renderer='browser')
    #fig.write_html('c:\\TEMP\\NAATOS_PM_RUN_runtime_{:s}_{:s}_{:s}.html'.format( time.strftime('%Y%m%dT%H%M') ));
    fig.write_html('c:\\TEMP\\NAATOS_PM_RUN_runtime_{:s}_{:s}_{:s}.html'.format( explist[0], unitlist[0], runlist[0] ));

for (exp,unit,run),dfgrp in dfbuilt.groupby(['expname','unit','run']):
    str_status = dfgrp.iloc[0]['status'];
    if((str_status=='run_success') or (str_status.find('run_ended_early')>=0)):
        print(exp,unit,run);

        dfplot = dfraw.query("run==@run and unit==@unit and expname==@exp")
        mkplot(dfplot);
    #break;

# %%
