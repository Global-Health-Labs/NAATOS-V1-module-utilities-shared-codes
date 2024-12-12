"""
Sample Prep Device Handling

Tool To Operate In Bulk On A Bunch Of Devices
Setup for Windows, would need reworking for Linux

Operation:
#1. (if enabled) wait for NRFUTIL dfu (initiate from device 5 presses quickly)
#2. (if enabled) USB flash
#3. wait for MSC connection (initiate from device long-hold "E:")
#4. do file operations:
    * A. copy all files
    * B. copy pre-existing config file auto-replacing current time inside it
    * C. 

"""

#%%
import os
import time
import io
import pandas as pd
from pathlib import Path
import wmi
import subprocess
import shutil

import naatos_paths

import naatos_module_tools.usb_file_utils
import naatos_module_tools.nrf_dfu_utils
import naatos_module_tools.config_utils



#%% CONFIGS

# what does the NAATOS MSC device mount as, on your computer?
CFG_DRIVE_LETTER = 'F:';
CFG_DFU_COMPORT = 'COM4';

CFG_CONFIG_FILENAME = 'config_v2.99b.txt';
CFG_DATA_FILE_EXPNAME = '20241212_sgtest';

CFG_EN_DELETE_DEVICE_LOGS_ON_COPYALL = False;



CFG_EN_COPY_WITH_TIME_CONFIG = True;

CFG_TEMPLATE_CONFIG_FILE=r'configs\config_v2.7b.txt.120524.txt';



cfg_file_path = os.path.join(naatos_paths.CFG_PATH_TEAMS_BETA_SAMPLEPREPTESTDATA,CFG_TEMPLATE_CONFIG_FILE);


#%% 20241203 script
if __name__ == "__main__":
    # sanity check before run
    # does desired config file exist?
    assert(os.path.exists(cfg_file_path));

    # ~~~~~COPY ALL FROM DEVICE~~~~
    # wait for mount
    naatos_module_tools.usb_file_utils.waitForDrive(CFG_DRIVE_LETTER);

    # get unitid (from user or from the device)
    CFG_UNITID = naatos_module_tools.usb_file_utils.idDevice(CFG_DRIVE_LETTER);

    # copy all files (and optionally delete)
    if True:
        naatos_module_tools.usb_file_utils.copyAllFromDevice(
            CFG_DRIVE_LETTER,
            EXPERIMENT_NAME=CFG_DATA_FILE_EXPNAME,
            UNITID=CFG_UNITID,
            DELETE_LOGS_AFTER_COPYALL=CFG_EN_DELETE_DEVICE_LOGS_ON_COPYALL
        );


    # ~~~~~FIRMWARE UPDATE~~~~~
    # upload new firmware
    if True:
        print('~~ NOW, FIRMWARE UPDATE ~~~');
        print('Windows-removal of previous DFU com ports....')
        naatos_module_tools.nrf_dfu_utils.windowsDeleteNRFSDFUComPorts();
    
        naatos_module_tools.nrf_dfu_utils.run_nrfutil_batch_and_wait(CFG_DFU_COMPORT);

    # ~~~~~FILESYSTEM REINITIALIZATION~~~~~
    if True:
        # as of v2.7b, there is a way to reset/reformat the filesystem through a devie action
        print('~~ DO FS RESET (unplug USB, power off, hold down button while powering on) ~~~');
        input('Press enter when reset done (and to set config with time/date)')


    # ~~~~~CONFIG-FILE UPLOADING~~~~~
    # write new config to the device

    # wait for mount
    naatos_module_tools.usb_file_utils.waitForDrive(CFG_DRIVE_LETTER);

    # read desired template config (and modify with a few parameters)
    config_file_new = naatos_module_tools.config_utils.getConfigFileText(
        cfg_file_path,
        set_time=True,
        unitid=CFG_UNITID
    );

    print(config_file_new);

    # write config to the device
    naatos_module_tools.usb_file_utils.writeConfigFile(
        drive_letter=CFG_DRIVE_LETTER,
        config_filename=CFG_CONFIG_FILENAME,
        strconfig_new=config_file_new
    )

    # graceful-ish
    #windowsEjectDrive(CFG_DRIVE_LETTER);
    print('ALL DONE! Remember to ..... "Safely Remove Drive.....NRF 52 USB Demo...."');
