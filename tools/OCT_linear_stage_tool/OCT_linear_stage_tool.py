###
### Cell-based to test basic operations of Velmex VXM linear stage
### 2025/03/18
### Simon Ghionea
###
#%%
import velmex_vxm as vxm

%load_ext autoreload
%autoreload 2

#%%
XYZ = vxm.VXM()

#%% Set online mode (control through UART)
XYZ.setOnline();

#%% Home The Linear Stage
XYZ.setVelocity(axis=1,speed=1000);
XYZ.autohome(axis=1,stop='negative');
XYZ.null();

#%% Move Relative
XYZ.move_rel_mm(axis=1,rel_mm=10,wait=True,speed=6000);

#%% Move Absolute
XYZ.move_abs_mm(axis=1,abs_mm=250.0,wait=True,speed=600);

#%% Move Absolute
XYZ.move_abs_mm(axis=1,abs_mm=144.0,wait=True,speed=6000);

#%% Move Absolute
XYZ.move_abs_mm(axis=1,abs_mm=144.0,wait=True,speed=6000);

#%% Get position
print('Current position is at {:} mm'.format(XYZ.getPosition(axis=1,value='mm')));
print('Current position is at {:} steps'.format(XYZ.getPosition(axis=1,value='steps')));

#%% Set offline mode and close the port
XYZ.setOffline();
XYZ.close();