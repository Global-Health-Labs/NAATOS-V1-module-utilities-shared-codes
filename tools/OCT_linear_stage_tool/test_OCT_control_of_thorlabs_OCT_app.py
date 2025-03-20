###
### Cell-based file to test api of Thorlabs OCT application control
### 2025/03/20
### Simon Ghionea
###

# Start with examples on OCT system that were installed at:
# C:\Program Files\Thorlabs\SpectralRadar\Python\PyThorImageAutomation

# Per the Thorlabs readme, install the wheel file using PIP
# in your python 3.11 environment.
# Step by step
# 1. Ensure the OCT_naatos environment is activated ("conda activate OCT_naatos")
# 2. cd "C:\Program Files\Thorlabs\SpectralRadar\Python\PyThorImageAutomation"
# 3. pip install lib/pytia-5.8.0.0-py3-none-any.whl
# Now the example codes could run, and the pytia library utilized.

#%% getting started with  pytia_demos.py
#%%% imports
import pytia as tia
import sys
import os.path


#%%
tia.load_library()

#%%
tia.start_thor_image()

#%%
# now we will manually click "Power ON" button in the software

#%%
#study_name = 'GHL_OCT_TIA_python_demo';
study_name = time.strftime('GHL_pyapp_%Y%m%dT%H%M');
tia.switch_mode(tia.Mode.MODE_3D);
tia.set_study_name(study_name);

#%%
# set things up in the software, like SIZE, FOV, ETC

#%% save parameters
ini_file = os.path.join(tia.folders.get_export(), "TS_DS_Parameters.ini")
tia.save_dataset_parameters(ini_file)

#%% load previously saved parameters
ini_file = os.path.join(tia.folders.get_export(), "TS_DS_Parameters.ini")
tia.load_dataset_parameters(ini_file)

#%% Switch to 3D
tia.switch_mode(tia.Mode.MODE_3D);

#%% 3D mode record
print('Capturing 3D volume...');
tia.record()
dsname = tia.get_last_dataset_name();
print('Captured',dsname);

#%% Save/export using one-click mechanism
tia.one_click_export();

#%% GLOBAL SETTINGS
folder_oct_export = Path(r'C:\OCTExport')
folder_destination_root = Path(r'G:\Projects\RemediReader\NAATOS_OCT_TEMP\OCTExport')

#%% QUICK TEST FOR STEPPING AND GETTING FOV's
import math
import json
import time
from pathlib import Path

import shutil

#pos_leftmost = 100.0;
pos_leftmost_edge_center = 95.25;
pos_rightmost_edge_center = 178.0;
FOV_Y = 7.00; # should match what's in the software

NUM_FOVS = math.ceil((pos_rightmost_edge_center-pos_leftmost_edge_center)/FOV_Y)+1;



# set name based on TOD
tia.switch_mode(tia.Mode.MODE_3D);
study_name = time.strftime('GHL_pyapp_%Y%m%dT%H%M');
tia.set_study_name(study_name);

# save settings
ini_file = os.path.join(tia.folders.get_export(), "{:s}_TS_DS_Parameters.ini".format(study_name))
tia.save_dataset_parameters(ini_file)

# gather FOV's
datanames = [];
positions = [];
for fovnum in range(NUM_FOVS):
    pos = pos_leftmost_edge_center+(FOV_Y*fovnum);
    print('FOV: {:d} GOTO:{:.02f}mm'.format(fovnum,pos))

    # OCT switch to 3D mode
    tia.switch_mode(tia.Mode.MODE_3D);

    # move stage
    XYZ.move_abs_mm(axis=1,abs_mm=pos,wait=True,speed=6000);

    # OCT record data
    tia.record();
    
    # OCT export
    tia.one_click_export();

    datanames.append( tia.get_last_dataset_name() );
    positions.append( pos );

#%% write an info file
info = dict(
    pos_leftmost_edge_center=pos_leftmost_edge_center,
    pos_rightmost_edge_center=pos_rightmost_edge_center,
    FOV_Y=FOV_Y,
    NUM_FOVS=NUM_FOVS,
    study_name = study_name,
    datanames=datanames,
    positions=positions
)

# make destination folder
folder_study = (folder_destination_root/study_name);
folder_study.mkdir(parents=True,exist_ok=True)

# write info json file
with open(folder_study/'info.json',mode='wt+') as file:
    json.dump(info,file,indent=4, sort_keys=True);

# copy files from OCT export to our desired destination folder
# copy everything that starts with our studyname
for file in folder_oct_export.glob('{:s}*'.format(study_name)):
    print('Copying ',file.name)
    shutil.copy(file,folder_study/file.name)

#%% close program
tia.close_thor_image();