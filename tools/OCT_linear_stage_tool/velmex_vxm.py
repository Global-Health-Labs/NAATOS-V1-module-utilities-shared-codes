###
### Code to operate Velmex VXM Stepper Motor Controller
### 2025/03/18
### Simon Ghionea
###
from serial import Serial
import time

#some settings
port = 'COM28'
baud = 9600
maxWaitTime = 0.1 #this is in seconds and I will play around with it

#some constants/defaults
defaultVelocity = 2000
stepsTomm = 0.0050
mmToSteps = (1/0.0050)
mmToin = (1/25.4)
inTomm = 25.4


#Make a Class for the VXM Stage
class VXM(object):
    _lastAxisSpeed = [0.0,0.0,0.0,0.0];
    _gotcarrot = False;

    #initialize our stage
    def __init__(self, port=port, baud=baud, maxWaitTime=maxWaitTime):
        self._port = Serial(port=port,baudrate=baud,timeout=maxWaitTime)
        print('Initializing VXM on port: %s' %(port))
        #set default motor speeds: The manual says 2000 steps/second is good
        #self.setVelocity('X',defaultVelocity)
        #self.setVelocity('Y',defaultVelocity)
        #self.setVelocity('Z',defaultVelocity)

    #provide function for sending commands
    def _sendcmd(self, commandString, expectEcho=True, verbose=True):
        #add the return variable \r and convert to bytes
        self._gotcarrot = False;

        stringToSend = (commandString + '\r').encode()
        #flush the input buffer
        self._port.flushInput()
        self._port.write(stringToSend)
        if verbose:
            print('Sent Stage Command: %s' %(commandString))
        
        # ECHO FROM DEVICE, VERIFY
        if(expectEcho):
            response = self._getresp();
            if(response.strip()!=commandString.strip()):
                if(not response.strip()[-1]=='^'):      # set velocity will respond with ^ at end always
                    raise RuntimeError("Bad-echo from VMX")
                else:
                    self._gotcarrot = True;

    #Function for getting response from stage
    def _getresp(self, verbose=True):
        resp = self._port.readline().decode() #the decode makes it be a string not a byte string aka 'string' not b'string'
        if verbose:
            print(resp)
        #flush the output buffer
        self._port.flushOutput()    
        return resp  

    def _waitForCarrot(self,wait=False):
        # device will send a single "^" when move is completed
        # we'll check for it here
        # if wait=True, we'll wait
        readstr = '';
        retval = True;
        if(self._gotcarrot):
            return True;
        while True:
            if(self._port.in_waiting>0):
                readstr = self._port.read(self._port.in_waiting).decode();
                print('Got \"{:s}\"'.format(readstr));
                retval = True;
                break;
            else:
                retval = False;
            
            if(wait):
                time.sleep(0.1);
            else:
                break;
        return retval;

    def _waitReady(self, verbose=True):
        #I want this command to go into all moves and things that take time
        #When one sends V to the stage, it returns B if busy, R if ready, J if in jog/slew mode, b if Jog/Slewing
        #need to wait for it to return R
        status = 'S'        #calling it S for start
        while status != 'R':
            #if verbose:
                #print('Sending command V')
            self._sendcmd('V',expectEcho=False);
            #now get a response
            status=self._getresp().strip();
            if verbose:
                print('Got a response: %s' %(status))
            time.sleep(0.1) #sleep for a tenth of a second
    
    def setOnline(self):
        # E - enable online with echo on
        # F - enable online with echo off
        # 
        self._sendcmd('E');
    
    def setOffline(self):
        # Q - Quit On-Line mode (return to Local mode)
        self._sendcmd('Q');
    
    def setVelocity(self, axis=1, speed=defaultVelocity):
        #make sure someone is only trying to input velocity for one of the 3 axes
        assert(axis>=1 and axis<=4);
        assert(speed>=5 and speed<=6000);

        if(self._lastAxisSpeed[axis-1]!=speed):
            cmdstr = 'CS{:d}M{:d},R'.format(axis,speed);
            self._sendcmd(cmdstr);
            self._lastAxisSpeed[axis-1] = speed; # keep track of the axis speeds

    def move_rel_mm(self,axis=1,rel_mm=1,wait=True,speed=None):
        assert(axis>=1 and axis<=4);

        # speed in steps (not mm)
        if(speed is not None):
            self.setVelocity(axis=axis,speed=speed);
        
        #get the distance in mm to be in steps not mm
        
        dist_step = int(rel_mm*mmToSteps)
        
        #make the command to move the stage

        cmdstr = 'CI{:d}M{:d},R'.format(axis,dist_step);
        self._sendcmd(cmdstr);
        #now we have sent the command to move the stage back to zero
        #need to wait for the stage to finish getting to zero
        #self._waitReady();
        if(wait):
            self._waitForCarrot(wait=True);
    
    def move_abs_mm(self,axis,abs_mm,wait=True, speed=None):
        assert(axis>=1 and axis<=4);

        # speed in steps (not mm)
        if(speed is not None):
            self.setVelocity(axis=axis,speed=speed);
        
        #get the distance in mm to be in steps not mm
        
        absolute_destination_step = int(abs_mm*mmToSteps)
        
        #make the command to move the stage
        
        cmdstr = 'CIA{:d}M{:d},R'.format(axis,absolute_destination_step);
        self._sendcmd(cmdstr);
        #now we have sent the command to move the stage back to zero
        #need to wait for the stage to finish getting to zero
        #self._waitReady();
        if(wait):
            self._waitForCarrot(wait=True);
    
    def getPosition(self,axis=1,value='mm'):
        assert(axis>=1 and axis<=4);
         # X is first axis
         # T is last axis (4th)
        cmdstrings = ['X','Y','Z','T']; #see vmx manual
        self._sendcmd(cmdstrings[axis-1],expectEcho=False)       #gets X position
        resp = str(self._getresp(verbose=True))
        

        if(resp[0]==cmdstrings[axis-1]):
            resp = resp[1:];
        num = int(resp[1:]);
        if resp[0] == '-':          #see if second value is + or -
            num = num*(-1);      #if negative, make it a negative value
        if(value=='mm'):
            mm = float(stepsTomm*num);
            return mm;
        elif(value=='steps'):
            # steps
            return num;
        else:
            raise ValueError('got unknown type');
        #print(Xresp)

    def autohome(self,axis=1,stop='positive'):
        # sends autohome to positive or negative limit
        # Velmex commands:
        # ImM0 # home motor m to positive limit
        # ImM-0 # home motor m to negative limit
        assert(axis>=1 and axis<=4);
        assert(stop=='positive' or stop=='negative');
        if(stop=='positive'):
            cmdstr = 'CI{:d}M0,R'.format(axis);
        elif(stop=='negative'):
            cmdstr = 'CI{:d}M-0,R'.format(axis);
        self._sendcmd(cmdstr);
        #self.waitReady();
        # movement in progress
        while(not self._waitForCarrot()):
            steps = self.getPosition(axis,value='steps')
            print('At {:d} step count'.format(steps));
            time.sleep(0.1);
    
    def null(self):
        # nulls all axes
        self._sendcmd('N');
    
    def stop(self):
        # nulls all axes
        self._sendcmd('D');

    def close(self):
        self._port.close();
    
    def __del__(self):
        self.close();
