'''
QLASS DRIVER


'''


from .RS232Manager import RS232Manager
from imswitch.imcommon.model import initLogger
# from pyvisa.constants import BufferOperation #incompatible with pyvisa 1.10.1!!
from pyvisa.constants import VI_READ_BUF_DISCARD
import numpy as np
import time

class QLASSManager(RS232Manager):

    

    def __init__(self,rs232Info,name,verbose=True,autoUpdate=True,**_lowLevelManagers):
        super().__init__(rs232Info,name,**_lowLevelManagers)
        print('hey I got after QLASSManager super().init! wow!')
        self.verbose = verbose
        self.__logger = initLogger(self, instanceName=name)
        self.autoUpdate = autoUpdate
        self.ranges = {0: 2.77,
                1: 25.0,
                2: 47.72}
        self.reset()
        
        self._sequences = np.full((16,8,2),fill_value=-1,dtype=int)
        self._current_mode = 'constant'
        self._current = np.zeros((16,),dtype = int)
        self._range = None

        self.range = 2
        #_sequences is an array containing the various steps of all the sequences
        #that are outputed by the board . usage:
        #self._sequences[channel][position][0 for time, 1 for value]
        #its value is set by the set_sequence_element function!!! Access (read-only)
        #should be performed via the sequences property.


    def reset(self) -> None:
        """
        Reset the instrument to factory default settings.
        """
        self.write("$R")
        #Reset returns three lines: b'Reset...\n\r$M=>Menu\n\rReady\n\r'
        #To ensure buffer is empty after resetting, we do this:
        res = ''
        while res.strip() != 'Ready':
            res = self._rs232port.resource.read()

        self.flush_serial()
        self.range = 2

        if self.verbose:
            self.__logger.debug("Instrument reset.")


    def idn(self) -> str:
        """
        Query the instrument identification string.

        Returns
        -------
        res: str
            The identification string returned by the instrument.
        """
        try:
            # Using commands from the driver documentation:
            res = self.query("$V").strip()
            if self.verbose:
                self.__logger.debug(f"Identification: {res}.")
            return res
        except Exception:
            if self.verbose:
                self.__logger.error("Identification command not supported or no response.")
            return "Unknown device."
        

    def flush_serial(self) -> None:
        """
        Flushes read serial buffer.

        Notes
        -----
        No flush operation was included in the RS232Manager or the RS232Driver;
        this is why the resource is accessed directly.
        """
        self._rs232port.resource.flush(VI_READ_BUF_DISCARD)
        if self.verbose:
            self.__logger.debug('Read buffer flushed.')


    def update(self) -> str:
        '''
        Updates the values which are being outputed by the device.
        '''
        res = self.query('$U')
        self.flush_serial()

        if res.strip() == 'Done':
            if self.verbose:
                self.__logger.debug('DAC values updated.')
        else:
            self.__logger.error(f'DAC values update failed; response = {res}')
            raise RuntimeError(f'DAC values update failed; response = {res}')
        return res
    
    
    def read_range(self) -> int:
        """
        Current range currently employed (numerical code).
        (Corresponding codes: 0 -> 2.77mA, 1 -> 25mA, 2 -> 47.72mA)
        """
        self.flush_serial()
        res = self.query('$F')
        if 'FSR=' not in res.strip():
            self.__logger.error(f'Current range reading failed: response is {repr(res)}')
            raise RuntimeError(f'Current range reading failed: response is {repr(res)}')

        # Res is of format 'Calib.=0; FSR=N', therefore we split it
        # in two, then again around = and see which is the FSR
        try:
            parts = res.strip().split(';')
            for part in parts:
                key, value = part.strip().split('=')
                if key == "FSR":
                    fsr = int(value)
        except:
            self.__logger.error(f'Invalid current range reading response retrieved : response is {res.strip()}')
            raise RuntimeError(f'Invalid current range reading response retrieved : response is {res.strip()}')
        

        if fsr not in [0,1,2]:
            self.__logger.error(f'Invalid FSR code retrieved: is {fsr} (type: {type(res)}), should be 0, 1 or 2 (type: int)')
            raise RuntimeError(f'Invalid FSR code retrieved: is {fsr} (type: {type(res)}), should be 0, 1 or 2 (type: int)')
        else:
            if self.verbose:
                self.__logger.error(f'FSR = {fsr}.')
            self._range = fsr
            return fsr
        

    @property
    def current(self):
        """
        Current (DAC value) applied at each pin.
        Returns ONLY the DAC value; therefore, it does not account for different ranges.

        This property is READ ONLY. Use set_current_level to set the current at each pin.
        """
        out = self._current.copy()
        out.setflags(write=False)
        return out


    @property
    def range(self):
        """
        Retrieves current range (0, 1, or 2).
        """
        if self._range in [0,1,2]:
            return self._range
        else:
            try:
                read_r = self.read_range()
                if read_r in [0,1,2]:
                    self._range = read_r
                    return read_r
            except:
                self.__logger.error(f'Could not read range: stored value is {repr(self._range)}')
                raise RuntimeError(f'Could not read range: stored value is {repr(self._range)}')
    

    @range.setter
    def range(self,val: int):
        """
        Sets current range (numerical code). Usage:
        QLASSManager.range = N
        (Corresponding codes: 0 -> 2.77mA, 1 -> 25mA, 2 -> 47.72mA)

        Raises
        ------
            ValueError: if FSR to be set is not 0, 1, or 2
        """
        # Input sanity check
        if val not in [0,1,2]:
            self.__logger.error(f'Tried to set invalid FSR code : is {val} (type: {type(val)}), should be 0, 1 or 2 (type: int)')
            raise ValueError(f'Tried to set invalid FSR code : is {val} (type: {type(val)}), should be 0, 1 or 2 (type: int)')
        
        # Setting the FSR
        res = self.query(f"$F{val}")

        if 'FSR=' not in res.strip():
            self.__logger.error(f'Current range setting failed: sent $F{val} - response is {res.strip()}')
            raise RuntimeError(f'Current range setting failed: sent $F{val} - response is {res.strip()}')

        # Check FSR value returned by board
        try:
            parts = res.strip().split(';')
            for part in parts:
                key, value = part.strip().split('=')
                if key == "FSR":
                    fsr = int(value)
        except:
            self.__logger.error(f'Invalid current range response setting retrieved : sent $F{val} - response is {res.strip()}')
            raise RuntimeError(f'Invalid current range response setting retrieved : sent $F{val} - response is {res.strip()}')

        self.flush_serial()
        
        if fsr == val:
            self._range = val
            if self.verbose:
                self.__logger.debug(f'Current range set to {val} - now max current is {self.ranges[val]}mA.')
            return
        else:
            self.__logger.error(f'Current range setting to {val} failed - retrieved FSR value is {fsr}.')
            raise RuntimeError(f'Current range setting to {val} failed - retrieved FSR value is {fsr}.')
        
    
    @property
    def current_mode(self) -> str:
        '''
        Returns a string dictating the currently active mode:
        - 'constant' for static regime (default when starting up, after idling, or after calling $t0)
        - 'sequence' for dynamic regime (after calling $t1)
        '''
        return self._current_mode


    @current_mode.setter
    def current_mode(self,val: str) -> bool:
        '''
        Sets current mode to val.
        'timing' functions as alias of 'sequence', but will log a warning.
        'idle' will call the idle function, setting the _current_mode to 'constant' in the process.
        
        Raises
        ------
            ValueError: if provided value is not 'constant','sequence','timing', or 'idle'
            TypeError: if provided value is not a string
        '''
        valid_str = ['constant','idle','sequence','timing']
        if not isinstance(val,str):
            try:
                self.__logger.error(f'Provided current mode {val} must be a string, is {type(val)}.')
                raise TypeError(f'Provided current mode is not a string and not convertible to one, is {type(val)}.')
            except:
                self.__logger.error(f'Provided current mode {val} is not among valid current modes.')
                raise TypeError(f'Provided current mode is not a string and not convertible to one, is {type(val)}.')
        if val not in valid_str:
            self.__logger.error(f'Provided current mode {val} is not among valid current modes.')
            raise ValueError(f'Provided current mode {val} is not among valid current modes.')
        
        if val == 'constant':
            self.start_constant_mode()
        if val == 'sequence':
            self.start_sequence_mode()
        if val == 'idle':
            self.idle()
        if val == 'timing':
            self.start_sequence_mode()
            self.__logger.debug('Setting current_mode to sequences after current_mode = \'timing\' was called.')

        return
    

    def set_current_level(self,ch: int, val: int) -> str:
        '''
        Sets to the DAC for channel ch a value of val.

        Raises
        ------
            TypeError: if input values are not of type int
            ValueError: if 0<=ch<16 and 0<=val<65535 is False
        '''

        if (not isinstance(ch,int)) or (not isinstance(val,int)):
            self.__logger.error(f'Check types of input variables: ch -> {type(ch)}; val -> {type(val)} (must be both int)')
            raise TypeError(f'Check types of input variables: ch -> {type(ch)}; val -> {type(val)} (must be both int)')

        #Input sanity check
        if ch >= 16 or ch<0:
            self.__logger.error(f'Channel number {ch} is invalid (must be between 0 and 15).')
            raise ValueError(f'Channel number {ch} is invalid (must be between 0 and 15).')
        if val <0 or val >= 2**16:
            self.__logger.error(f'Cannot set {val} to DAC: level must be between 0 and 65535.')
            raise ValueError(f'Cannot set {val} to DAC: level must be between 0 and 65535.')
        
        res = self.query(f'$D{ch:02d},{val}')
        if res.strip() == 'Ready':
            self.__logger.error(f'set_current_level(ch={ch},val={val}) method failed being delivered to the board (DAC value = {val}).')
            raise RuntimeError(f'set_current_level(ch={ch},val={val}) method failed being delivered to the board (DAC value = {val}).')
        else:
            if self.verbose:
                self.__logger.debug(f'Current at channel {ch} set to DAC value = {val}.')
            self._current[ch] = val
        if self.autoUpdate:
            self.update()
        return res
    

    def set_zero(self) -> None:
        '''
        Sets manually the current to all channels to zero.
        '''
        was_verbose = self.verbose
        if was_verbose:
            self.__logger.debug('Setting all channels to zero.')
            self.verbose = False

        was_autoupdate = self.autoUpdate
        if was_autoupdate:
            self.autoUpdate = False

        for i in range(16):
            self.set_current_level(i,0)
        self.update()

        self.verbose = was_verbose
        self.autoUpdate = was_autoupdate
        return

    

    def start_constant_mode(self) -> None:
        '''
        Start application of the saved values to the DACs in a constant 
        manner.
        '''
        self.query('$t0')
        self.flush_serial()
        self._current_mode = 'constant'
        if self.verbose:
            self.__logger.debug('Started constant mode.')


    def start_sequence_mode(self) -> None:
        '''
        Start application of the saved sequences. 
        '''
        if -1 in self._sequences:
            self.__logger.warning('Uninitialized sequence elements detected.')
        self.query('$t1')
        self.flush_serial()
        self._current_mode = 'sequence'
        if self.verbose:
            self.__logger.debug('Started sequence mode.')
    

    def idle(self) -> None:
        '''
        Sets the board in an idle state: constant mode + zero current at DACs.
        '''
        self.start_constant_mode()
        self.set_zero()
        self._current_mode = 'constant'


    @property
    def sequences(self):
        '''
        Multidimensional array for indirect access to the _sequences array.
        Has shape (16,8,2):
        - 16 channels
        - 8 time steps
        - [0] for step duration, [1] for DAC value applied

        This property does not have a setter element, but its value must be modified by using the methods
        set_sequence_element, set_sequence_array, and set_all_sequences. Via direct access, this property
        is set to read only, and so will probably raise an error if tried to be modified directly.


        Example
        -------
        To obtain the time duration of all steps:
        sequences[:,:,0]
        To obtain the DAC values applied by pin 7:
        sequences[7,:,1]
        To see the first step duration of all channels:
        sequences[:,0,1]
        '''
        out = self._sequences.copy()
        out.setflags(write=False)
        return out
    

    def set_sequence_element(self,ch:int,pos:int,time:int,val:int):
        '''
        Sets a single sequence element for a given pin.
        For traceability, also modifies the array _sequences.

        Parameters
        ----------
        ch : int
            The channel whose sequence needs to be changed. Goes from 0 to 15.
        pos : int
            The position in the sequence to change. Goes from 0 to 7.
        time : int
            Duration of this sequence step, in steps of 500 us.
            Goes from 0 (500 us) to 999 (500 ms).
        val : int
            Value to apply to the DAC for this step.
            As all DAC values, goes from 0 to 65535.
        '''

        if self._current_mode == 'sequence':
            self.__logger.warning('Tried to modify sequences while in sequence mode. Idling.')
            self.idle()

        try:
            ch = int(ch)
            pos = int(pos)
            time = int(time)
            val = int(val)
        except ValueError:
            self.__logger.error('Invalid input values of set_sequence_element:\n'
                            f'ch = {ch}, pos = {pos}, time = {time}, val = {val}')
            raise ValueError('Invalid input values of set_sequence_element:\n'
                            f'ch = {ch}, pos = {pos}, time = {time}, val = {val}')
        except TypeError:
            self.__logger.error('Invalid input type of set_sequence_element:\n'
                            f'ch = {type(ch)}, pos = {type(pos)}, time = {type(time)}, val = {type(val)}')
            raise TypeError('Invalid input type of set_sequence_element:\n'
                            f'ch = {type(ch)}, pos = {type(pos)}, time = {type(time)}, val = {type(val)}')

        # input sanity check
        if ch < 0 or ch >= 16:
            self.__logger.error(f'ch = {ch} in set_sequence_element({ch},{pos},{time},{val}): must be between 0 and 15')
            raise ValueError(f'ch = {ch} in set_sequence_element({ch},{pos},{time},{val}): must be between 0 and 15')
        if pos < 0 or pos >= 8:
            self.__logger.error(f'pos = {pos} in set_sequence_element({ch},{pos},{time},{val}): must be between 0 and 15')
            raise ValueError(f'pos = {pos} in set_sequence_element({ch},{pos},{time},{val}): must be between 0 and 7')
        if time < 0 or time >= 256:
            self.__logger.error(f'time = {time} in set_sequence_element({ch},{pos},{time},{val}): must be between 0 and 15')
            raise ValueError(f'time = {time} in set_sequence_element({ch},{pos},{time},{val}): must be between 0 and 255')
        if val < 0 or val >= 65536:
            self.__logger.error(f'val = {val} in set_sequence_element({ch},{pos},{time},{val}): must be between 0 and 15')
            raise ValueError(f'val = {val} in set_sequence_element({ch},{pos},{time},{val}): must be between 0 and 65535')
        
        #send command to board
        res = self.query(f'$s{ch:02d},{pos:01d},{time:03d},{val}')
        exp_res = f'ch{ch:02d} [step{pos:01d}] for {time:03d} cycles = {val}'
        
        if res.strip() != exp_res:
            self._sequences[ch,pos,0] = -1
            self._sequences[ch,pos,1] = -1
            self.__logger.error(f'Failed sequence element writing: set_sequence_element({ch},{pos},{time},{val}) -> {res}')
            raise RuntimeError(f'Failed sequence element writing: set_sequence_element({ch},{pos},{time},{val}) -> {res}')
            

        #update sequences array
        self._sequences[ch,pos,0] = time
        self._sequences[ch,pos,1] = val

        if self.verbose:
            print(f' Element set for channel {ch}, position {pos}: val = {val} for {time} steps')
        return
    
    def set_sequence_array(self,ch,time_arr,val_arr):
        '''
        Sets the sequence for a whole channel.
        time_arr and val_arr must be 8 entries-long numpy arrays.

        Parameters
        ----------
        ch : int
            The channel whose sequence needs to be changed. Goes from 0 to 15.
        time_arr : List of int
            Duration of all sequence steps, in steps of 500 us.
            Goes from 0 (500 us) to 999 (500 ms).
            Must have shape (8,)
        val_arr : List of int
            Values to apply to the DAC for all steps.
            As all DAC values, goes from 0 to 65535.
            Must have shape (8,)
        '''
        # input sanity checks
        time_arr = np.squeeze(time_arr)
        val_arr = np.squeeze(val_arr)
        if time_arr.shape != (8,):
            raise ValueError(f' Invalid shape for time_arr: {time_arr.shape} (must be (8,))')
        if val_arr.shape != (8,):
            raise ValueError(f' Invalid shape for val_arr: {val_arr.shape} (must be (8,))')
        

        was_verbose = self.verbose
        if was_verbose:
            self.verbose = False
            self.__logger.debug(f'Setting sequence of channel {ch} in progress.')
        
        for i, (t, v) in enumerate(zip(time_arr, val_arr)):
            self.set_sequence_element(ch, i, t, v)
        
        self.verbose = was_verbose
        return
    
    def set_all_sequences(self,time_seqs,val_seqs):
        '''
        Sets the complete sequences to the memory of the board.
        time_seqs and val_seqs must be (16,8) matrices

        Parameters
        ----------
        time_arr : List of list of int
            Duration of all sequence steps, in steps of 500 us.
            Goes from 0 (500 us) to 999 (500 ms).
            Must have shape (16,8)
        val_arr : List of list of int
            Values to apply to the DACs for all steps.
            As all DAC values, goes from 0 to 65535.
            Must have shape (16,8)
        '''

        time_seqs = np.squeeze(time_seqs)
        val_seqs = np.squeeze(val_seqs)

        if time_seqs.shape != (16,8):
            self.__logger.error(f'Invalid shape for time_seqs: {time_seqs.shape} (must be (16,8))')
            raise ValueError(f'Invalid shape for time_seqs: {time_seqs.shape} (must be (16,8))')
        if val_seqs.shape != (16,8):
            self.__logger.error(f'Invalid shape for val_seqs: {val_seqs.shape} (must be (16,8))')
            raise ValueError(f'Invalid shape for val_seqs: {val_seqs.shape} (must be (16,8))')
        
        was_verbose = self.verbose
        if was_verbose:
            self.verbose = False
            self.__logger.debug(' Setting whole sequences matrix in progress.')

        for i in range(16):
            self.set_sequence_array(i,time_seqs[i],val_seqs[i])

        self.verbose = was_verbose
        return
    
    def initialize_sequences(self):
        '''
        Sets the values of all elements in the sequence to 0 mA for 500 ms.
        '''
        was_verbose = self.verbose
        if was_verbose:
            self.__logger.debug(' Initializing sequences to zero.')
            self.verbose = False

        try:
            self.set_all_sequences(time_seqs = np.full((16,8),fill_value=999,dtype=int), val_seqs = np.zeros((16,8),dtype=int))
        finally:
            self.verbose = was_verbose

        return


    