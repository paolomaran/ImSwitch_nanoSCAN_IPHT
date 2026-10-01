import requests
import json
import os

import numpy as np
import time
import threading
from datetime import datetime
import tifffile as tif

from imswitch.imcommon.model import dirtools, initLogger, APIExport, ostools
from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.framework import Signal, Thread, Worker, Mutex, Timer

from ..basecontrollers import LiveUpdatedController

import os
import time
import numpy as np
from pathlib import Path
import tifffile

import imswitch

from datetime import datetime

try:
    import mcsim
    ismcSIM=True
except:
    ismcSIM=False

if ismcSIM:
    try:
        import cupy as cp
        from mcsim.analysis import sim_reconstruction as sim
        isGPU = True
    except:
        print("GPU not available")
        import numpy as cp
        from mcsim.analysis import sim_reconstruction as sim
        isGPU = False
else:
    isGPU = False

try:
    import NanoImagingPack as nip
    isNIP = True
except:
    isNIP = False



try:
    from napari_sim_processor.processors.convSimProcessor import ConvSimProcessor
    from napari_sim_processor.processors.hexSimProcessor import HexSimProcessor
    isSIM = True

except:
    isSIM = False

try:
    # FIXME: This does not pass pytests!
    import torch
    isPytorch = True
except:
    isPytorch = False

isDEBUG = False

class SIMQLASSController(ImConWidgetController):
    """Linked to SIMWidget."""

    sigImageReceived = Signal(np.ndarray, str)
    sigSIMProcessorImageComputed = Signal(np.ndarray, str)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        self.IS_FASTAPISIM=False
        self.IS_HAMAMATSU=False
        # switch to detect if a recording is in progress
        self.isRecording = False
        self.isReconstruction = False

        # Laser flag
        self.LaserWL = 0

        self.simFrameVal = 0
        self.nsimFrameSyncVal = 2

        # Choose which laser will be recorded
        self.is488 = False
        self.is635 = True

        # we can switch between mcSIM and napari
        self.reconstructionMethod = "napari" # or "mcSIM"

        # save directory of the reconstructed frames
        self.simDir = os.path.join(dirtools.UserFileDirs.Root, 'imcontrol_sim')
        if not os.path.exists(self.simDir):
            os.makedirs(self.simDir)

        # load config file
        if self._setupInfo.sim is None:
            self._widget.replaceWithError('SIM is not configured in your setup file.')
            return

        # connect live update  https://github.com/napari/napari/issues/1110
        self.sigImageReceived.connect(self.displayImage)

        # select lasers
        allLaserNames = self._master.lasersManager.getAllDeviceNames()
        self.lasers = []
        for iDevice in allLaserNames:
            if iDevice.lower().find("laser")>=0 or iDevice.lower().find("led"):
                self.lasers.append(self._master.lasersManager[iDevice])
        if len(self.lasers) == 0:
            self._logger.error("No laser found")
            # add a dummy laser
            class dummyLaser():
                def __init__(self, name, power):
                    self.power = 0.0
                    self.setEnabled = lambda x: x
                    self.name = name
                    self.power = power
                def setPower(self,power):
                    self.power = power
                def setEnabled(self,enabled):
                    self.enabled = enabled
            self.lasers.append(dummyLaser("DummyLaser", 100))
        # select detectors
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.detector = self._master.detectorsManager[allDetectorNames[0]]
        if self.detector.model == "CameraPCO":
            # here we can use the buffer mode
            self.isPCO = True
        else:
            # here we need to capture frames consecutively
            self.isPCO = False
        print(f'Available camera params: {self.detector.parameters}')

        # select positioner
        self.positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self.positioner = self._master.positionersManager[self.positionerName]

        # setup the SIM processors
        sim_parameters = SIMParameters()
        self.SimProcessor = SIMProcessor(self, sim_parameters, wavelength=sim_parameters.wavelength)

        # Connect CommunicationChannel signals
        self.sigSIMProcessorImageComputed.connect(self.displayImage)

        #self.initFastAPISIM(self._master.simManager.fastAPISIMParams)
        self.initQLASSDriver()


        if imswitch.IS_HEADLESS:
            return


        self._widget.button_zero_all_DACs.clicked.connect(self.stopDACs)
        self._widget.button_start_test_cycle.clicked.connect(self.startThread_TestingPattern)
        self._widget.button_stop_test_cycle.clicked.connect(self.stopThread_TestingPattern)
        self._widget.button_start_pattern_calibration.clicked.connect(self.startCalibPattern)
                
        self._widget.checkbox_record_raw.stateChanged.connect(self.toggleRecording)
        self._widget.checkbox_record_reconstruction.stateChanged.connect(self.toggleRecordReconstruction)
        self._widget.sigPatternID.connect(self.patternIDChanged)
        self._widget.sigPatternMode.connect(self.patternModeChanged)
        #self._widget.checkbox_reconstruction.stateChanged.connect(self.toggleRecording)

        # read parameters from the widget
        self._widget.start_timelapse_button.clicked.connect(self.startTimelapse)
        self._widget.start_zstack_button.clicked.connect(self.startZstack)
        self._widget.start_button.clicked.connect(self.startSIM)
        self._widget.stop_button.clicked.connect(self.stopSIM)
        self._widget.openFolderButton.clicked.connect(self.openFolder)
        self.folder = self._widget.getRecFolder()

        self._current_pattern = 'none'
        self.SIMDriver.set_uniform_illumination()
        self._is3DSIM = False


    def stopDACs(self):
        self.driver.reset()
        self.SIMDriver.active_pattern = None

    def patternIDChanged(self,patternID):
        self._current_pattern = patternID
        self.SIMDriver.generate_patterns_testing(self._is3DSIM,self._current_pattern)

    def patternModeChanged(self,mode):
        self._is3DSIM = mode=='3DSIM'
        self.SIMDriver.generate_patterns_testing(self._is3DSIM,self._current_pattern)

    def toggleRecording(self):
        self.isRecording = not self.isRecording
        if not self.isRecording:
            self.isActive = False

    def toggleRecordReconstruction(self):
        self.isReconstruction = not self.isReconstruction
        if not self.isReconstruction:
            self.isActive = False

    def openFolder(self):
        """ Opens current folder in File Explorer. """
        folder = self._widget.getRecFolder()
        if not os.path.exists(folder):
            os.makedirs(folder)
        ostools.openFolderInOS(folder)


    def initQLASSDriver(self):
        self.driver = self._master.rs232sManager['QLASS']
        self.SIMDriver = SIMPatternGenerator(self.driver,self.lasers[0])
        self.driver.verbose=False


    def __del__(self):
        self.driver.reset()
        #self.imageComputationThread.quit()
        #self.imageComputationThread.wait()


    def displayImage(self, im, name="SIM Reconstruction"):
        """ Displays the image in the view. """
        self._widget.setImage(im, name=name)

    def saveParams(self):
        pass

    def loadParams(self):
        pass

    def startThread_TestingPattern(self):
        '''
        Start a thread in which the currently selected pattern is translated in its valid positions.
        Terminated via stopThread_TestingPatterns.
        '''
        patternID = self._widget.pattern_orientation_dropdown.currentText()
        val_is3D = self._widget.pattern_mode_dropdown.currentText()
        is3D = self._is3DSIM

        self.testingStopEvent = threading.Event()
        self.active = True
        self.testingPatternThread = threading.Thread(target=self.SIMDriver.generate_patterns_testing,args=(is3D,patternID,True,self.testingStopEvent))
        self.testingPatternThread.start()

    def stopThread_TestingPattern(self):
        '''
        Terminates thread TestingPattern.
        '''
        self.testingStopEvent.set()
        self.testingPatternThread.join()
        self.active = False
        

    def startCalibPattern(self):
        '''
        Start acquisition of calibration pattern
        '''
        if self.isPCO:    
            self._commChannel.sigStopLiveAcquisition.emit(True)
            self.detector.setParameter("trigger_source","External start")
            self.detector.setParameter("buffer_size",101)
            self.detector.flushBuffers()

        processor = self.SimProcessor

        try:
            exposureTime = self.detector.parameters['exposure'].value/1e3 # s^-1
        except:
            self._logger.warning('Camera failed to provide a valid exposure time.')
            exposureTime = 0.050
            
        self.active = True
 
        calibThread = threading.Thread(target=self.SIMDriver.generate_pattern_translations,args=(exposureTime,self.isPCO),daemon=True)
        calibThread.start()
        calibThread.join()
        
        self._logger.debug("Calib finished")
        self.active = False
        self.SIMStack = self.detector.getChunk(); self.detector.flushBuffers()
        if self.SIMStack is None:
            self._logger.error("No image received")
        self.sigImageReceived.emit(np.array(self.SIMStack),"SIMStack"+str(processor.wavelength))
        processor.setSIMStack(self.SIMStack)

        if self.isPCO:    
            self.detector.setParameter("trigger_source","Internal trigger")
            self.detector.setParameter("buffer_size",-1)
        self.detector.flushBuffers()
        
        if self.isRecording :
            uniqueID = np.random.randint(0,1000)
            date = datetime.now().strftime("%Y_%m_%d-%I-%M-%S_%p")
            processor.setDate(date)
            mFilenameStack = f"{date}_CALIB{self._current_pattern}_{uniqueID}.tif"
            threading.Thread(target=self.saveImageInBackground, args=(self.SIMStack, mFilenameStack,), daemon=True).start()
    

    def stopSIM(self):
        self.active = False
        self.simThread.join()
        self.lasers[0].setEnabled(False)
        if self.isPCO:
            self.detector.setParameter("trigger_source","Internal trigger")
            self.detector.setParameter("buffer_size",-1)
            self.detector.flushBuffers()
        self.SIMDriver.set_uniform_illumination()


    #TODO add way in GUI to pass the pattern mode to this func
    def startSIM(self):
        #  need to be in trigger mode
        # therefore, we need to stop the camera first and then set the trigger mode

        if self.isPCO:
            # prepare camera for buffer mode
            self._commChannel.sigStopLiveAcquisition.emit(True)
            self.detector.setParameter("trigger_source","External start")
            self.detector.setParameter("buffer_size",9)
            self.detector.flushBuffers()
        #self._commChannel.sigStartLiveAcquistion.emit(True)

        # start the background thread
        self.active = True
        sim_parameters = self.getSIMParametersFromGUI()
        #sim_parameters["reconstructionMethod"] = self.getReconstructionMethod()
        #sim_parameters["useGPU"] = self.getIsUseGPU()
        self.simThread = threading.Thread(target=self.performSIMExperimentQLASSThread, args=(sim_parameters,), daemon=True)
        self.simThread.start()


    #TODO add way in GUI to pass the pattern mode to this func
    def startTimelapse(self):
        
        timelapse_mode = self._widget.timelapse_pattern_mode_dropdown.currentText()
        buffer_sizes = {
            '2DSIM' : 9,
            '3DSIM_FULL' : 19,
            '3DSIM_HYBRID' : 11,
            'HiLo' : 4  #2 would be sufficient, but the min camera buffer is 4 
                        }
        
        if self.isPCO:    
            self._commChannel.sigStopLiveAcquisition.emit(True)
            self.detector.setParameter("trigger_source","External start")
            self.detector.setParameter("buffer_size",buffer_sizes[timelapse_mode])
            self.detector.flushBuffers()

        self.active = True
        sim_parameters = self.getSIMParametersFromGUI()

        timePeriod = float(self._widget.period_textedit.text())
        Nframes = int(self._widget.frames_textedit.text())
        self.oldTime = time.time()-timePeriod # to start the timelapse immediately
        iiter = 0
        # if it is nessary to put timelapse in background
        while iiter < Nframes:
            print(f'rechecking... t={time.time() - self.oldTime :.2f}',sep='',end='\r',flush=True)
            if time.time() - self.oldTime > timePeriod:
                # we HAVE to check that the simThread has terminated operation. If it has not, the QLASS board will get
                # mixed messages between different instances of performSIMTimelapseQLASSThread, resulting in unpredictable
                # behaviour and likely timeout errors.
                # this kind of makes it pointless to have the function run in a thread in the first place... oh well
                if hasattr(self,'simThread'): #in the very first iteration simThread is undefined
                    if self.simThread.is_alive():
                        self._logger.warning('Warning: simThread still executing. Waiting for end of execution before resuming.')
                        t1 = time.time()
                        self.simThread.join()
                        self._logger.warning(f'simThread took {time.time()-t1 :.3f} s longer than selected period. Resuming timelapse.')

                self.oldTime = time.time()
                self.simThread = threading.Thread(target=self.performSIMTimelapseQLASSThread, args=(sim_parameters,timelapse_mode), daemon=True)
                self.simThread.start()
                iiter += 1
            else:
                time.sleep(0.020)
        
        # in the last iteration, we need to wait for the simThread to finish before we can terminate the timelapse!
        # if we do not, the camera will be set to internal trigger for the last acquisition, resulting in 
        # asynchronously acquired frames, i.e. gibberish
        if self.simThread.is_alive():
            self.simThread.join()

        # if the previous fix works correctly, the following log will really be diplayed after the last frame is done being acquired
        self._logger.debug("Timelapse finished")
        self.active = False

        if self.isPCO:    
            self.detector.setParameter("trigger_source","Internal trigger")
            self.detector.setParameter("buffer_size",-1)
        self.detector.flushBuffers()


    #TODO add way in GUI to pass the pattern mode to this func
    def startZstack(self):
        
        zstack_mode = self._widget.zstack_pattern_mode_dropdown.currentText()
        
        buffer_sizes = {
            '2DSIM' : 9,
            '3DSIM_FULL' : 19,
            '3DSIM_HYBRID' : 11,
            'HiLo' : 4  #2 would be sufficient, but the min camera buffer is 4 
                        }
        
        if self.isPCO:
            self._commChannel.sigStopLiveAcquisition.emit(True)
            self.detector.setParameter("trigger_source","External start")
            self.detector.setParameter("buffer_size",buffer_sizes[zstack_mode])
            self.detector.flushBuffers()

        self.active = True
        sim_parameters = self.getSIMParametersFromGUI()
        zMin = float(self._widget.zmin_textedit.text())
        zMax = float(self._widget.zmax_textedit.text())
        zStep = int(self._widget.nsteps_textedit.text())
        zDis = int((zMax - zMin) / zStep)
        self._master.detectorsManager   #FIXME
        

        self.simThread = threading.Thread(target=self.performSIMZstackQLASSThread, args=(sim_parameters,zDis,zStep,zstack_mode), daemon=True)
        self.simThread.start()


    def updateDisplayImage(self, image):
        image = np.fliplr(image.transpose())
        self._widget.img.setImage(image, autoLevels=True, autoDownsample=False)
        self._widget.updateSIMDisplay(image)
        #self._logger.debug("Updated displayed image")


    def performSIMExperimentQLASSThread(self,sim_parameters,mode_SIM:str='2DSIM',trigCam:bool=True,trigLaser:bool=True):
        '''
        Using the QLASS Driver, iterate over all patterns to be displayed.
        '''

        if mode_SIM not in ['2DSIM','3DSIM_FULL','3DSIM_HYBRID']:
            self._logger.warning(f'Retrieved mode {mode_SIM} is invalid; switching to 2DSIM.')
            mode_SIM = '2DSIM'

        processor = self.SimProcessor

        # retreive Z-stack parameters
        if hasattr(self,'positioner'):
            zStackParameters = self._widget.getZStackParameters()
            zMin, zMax, zStep = zStackParameters[0], zStackParameters[1], zStackParameters[2] # if zStep < 0, it will not move in z
            tDebounce = 0.1 # debounce time between z-steps
            zPosInitially = self.positioner.getPosition()["Z"]
        else:
            self._logger.warning("No positioner found; Z-stack will not be performed.")
            zMin, zMax, zStep = 0, 0, 0
            tDebounce = 0
            zPosInitially = 0

        # retreive timelapse parameters
        timelapsedParameters = self._widget.getTimelapseParameters()
        timePeriod, Nframes = timelapsedParameters[0], timelapsedParameters[1] # if NFrames < 0, it will run indefinitely
        
        while self.active:

            # iterate over all z-positions
            if zStep > 0:
                allZPositions = np.arange(zMin, zMax, zStep)
            else:
                allZPositions = [0]

            try:
                exposureTime =  self.detector.parameters['exposure'].value/1e3 # s^-1 # s^-1
            except:
                exposureTime = 0.050

            if not self.active:
                if len(allZPositions)!=1:
                    self.positioner.move(value=zPosInitially, axis="Z", is_absolute=True, is_blocking=True)
                    time.sleep(tDebounce)
                break

            for zPos in allZPositions:
                # move to the next z-position
                if len(allZPositions)!=1 and hasattr(self,'positioner'):
                    self.positioner.move(value=zPos+zPosInitially, axis="Z", is_absolute=True, is_blocking=True)
                    time.sleep(tDebounce)

                if self.isPCO:
                    # display one round of SIM patterns for the right colour
                    self.SIMDriver.generate_patterns_acquisition(mode=mode_SIM,expTime=exposureTime,repNum=1,
                                                triggerCamera=trigCam,triggerLaser=trigLaser)

                    # download images from the camera
                    self.SIMStack = self.detector.getChunk(); self.detector.flushBuffers()
                    if self.SIMStack is None:
                        self._logger.error("No image received")
                        continue
                

                self.sigImageReceived.emit(np.array(self.SIMStack),"SIMStack"+str(processor.wavelength))
                processor.setSIMStack(self.SIMStack)
                processor.getWF(self.SIMStack)

                # activate recording in processor
                processor.setRecordingMode(self.isRecording)
                processor.setReconstructionMode(self.isReconstruction)
                processor.setWavelength(self.LaserWL,sim_parameters)


                # store the raw SIM stack
                if self.isRecording:
                    date = datetime.now().strftime("%Y_%m_%d-%I-%M-%S_%p")
                    processor.setDate(date)
                    mFilenameStack = f"{date}_SIM_Stack_{self.LaserWL}nm_{zPos+zPosInitially}mum.tif"
                    threading.Thread(target=self.saveImageInBackground, args=(self.SIMStack, mFilenameStack,), daemon=True).start()
                # self.detector.stopAcquisition()

                # process the frames and display
                processor.reconstructSIMStack()

                # reset the per-colour stack to add new frames in the next imaging series
                processor.clearStack()

            # move back to initial position
            if len(allZPositions)!=1 and hasattr(self,'positioner'):
                self.positioner.move(value=zPosInitially, axis="Z", is_absolute=True, is_blocking=True)
                time.sleep(tDebounce)


            # wait for the next rund
            time.sleep(timePeriod)
        self.SIMDriver.set_uniform_illumination()	


    def performSIMTimelapseQLASSThread(self,sim_parameters,mode_SIM:str='2DSIM',exposureTime:float=None,frameLabel:str=''):
        '''
        Do timelapse SIM via QLASS board.
        '''
        self.isReconstructing = False
        processor = self.SimProcessor
        #no need to manage lasers - they are triggered by the board itself.
        trigLaser = True
        trigCam = True

        if exposureTime is None:
            try:
                exposureTime = self.detector.parameters['exposure'].value/1e3 # s^-1 # s^-1

            except:
                self._logger.warning('Camera failed to provide a valid exposure time.')
                exposureTime = 0.050

        #this generates the pattern and triggers the camera.
        #After this function is done, the camera has in its buffer the acquired images.
        self.SIMDriver.generate_patterns_acquisition(mode=mode_SIM,expTime=exposureTime,repNum=1,
                                                     triggerCamera=trigCam,triggerLaser=trigLaser)

        #this gets us the images from the camera buffer
        #if mode_SIM == 'HiLo':
        #    frame0 = self.detector.getLatestFrame()
        #    frame1 = self.detector.getLatestFrame()
        self.SIMStack = self.detector.getChunk(); self.detector.flushBuffers()
        if self.SIMStack is None:
            self._logger.error("No image received")
        self.sigImageReceived.emit(np.array(self.SIMStack),"SIMStack"+str(processor.wavelength))
        processor.setSIMStack(self.SIMStack)

        # activate recording in processor
        processor.setRecordingMode(self.isRecording)
        processor.setReconstructionMode(self.isReconstruction)
        processor.setWavelength(self.LaserWL,sim_parameters)

        # store the raw SIM stack if recording is enabled
        if self.isRecording :
            uniqueID = np.random.randint(0,1000)
            date = datetime.now().strftime("%Y_%m_%d-%I-%M-%S_%p")
            processor.setDate(date)
            mFilenameStack = f"{date}_{mode_SIM}_Stack_{frameLabel}.tif"
            threading.Thread(target=self.saveImageInBackground, args=(self.SIMStack, mFilenameStack,), daemon=True).start()
        processor.clearStack()




    def performSIMZstackQLASSThread(self,sim_parameters,zDis,zStep,zmode_SIM:str='2DSIM'):
        mStep = 0
        acc = 0    #hardcoded acceleration
        mspeed = 1000   #hardcoded speed
        expTime = None

        while mStep < zStep:
            self.positioner.move(zDis,acceleration = acc, speed=mspeed)
            mStep += 1
            ZLabel = f'z{mStep*zDis:.1f}'
            
            retExpTime = self.performSIMTimelapseQLASSThread(sim_parameters,exposureTime = expTime,mode_SIM=zmode_SIM,frameLabel=ZLabel)
            if retExpTime is not None: 
                expTime = retExpTime
        self.active = False
        self.positioner.move(-zStep*zDis,acceleration = acc, speed=mspeed)
        if self.isPCO:
            self.detector.setParameter("trigger_source","Internal trigger")
            self.detector.setParameter("buffer_size",-1)
        self.detector.flushBuffers()
        self._logger.debug("Zstack finished")


    def saveImageInBackground(self, image, filename = None):
        if filename is None:
            date = datetime.now().strftime("%Y_%m_%d-%I-%M-%S_%p")
            filename = f"{date}_SIM_Stack.tif"
        try:
            self.folder = self._widget.getRecFolder()
            self.filename = os.path.join(self.folder,filename) #FIXME: Remove hardcoded path
            tif.imwrite(self.filename, image, dtype='uint16')
            self._logger.debug("Saving file: "+self.filename)
        except  Exception as e:
            self._logger.error(e)

    def getSIMParametersFromGUI(self):
        ''' retrieve parameters from the GUI '''
        sim_parameters = SIMParameters()

        # parse textedit fields
        sim_parameters.pixelsize = np.float32(self._widget.pixelsize_textedit.text())
        sim_parameters.NA = np.float32(self._widget.NA_textedit.text())
        sim_parameters.n = np.float32(self._widget.n_textedit.text())
        sim_parameters.alpha = np.float32(self._widget.alpha_textedit.text())
        sim_parameters.beta = np.float32(self._widget.beta_textedit.text())
        sim_parameters.eta = np.float32(self._widget.eta_textedit.text())
        sim_parameters.wavelength = np.float32(self._widget.wavelength_textedit.text())
        sim_parameters.magnification = np.float32(self._widget.magnification_textedit.text())
        sim_parameters.path = self._widget.path_edit.text()
        return sim_parameters

    def getReconstructionMethod(self):
        return self._widget.SIMReconstructorList.currentText()

    def getIsUseGPU(self):
        return self._widget.useGPUCheckbox.isChecked()


class SIMParameters(object):
    wavelength = 0.67
    NA = 1.4
    n = 1.52
    magnification = 90
    pixelsize = 6.5
    eta = 0.65
    alpha = 0.5
    beta = 0.98#
    path = 'C:\\Users\\admin\\Desktop\\Timelapse\\'

'''#####################################
            SIM PROCESSOR
#####################################'''

class SIMProcessor(object):

    def __init__(self, parent, simParameters, wavelength=635):
        '''
        setup parameters
        '''
        #current parameters is setting for 60x objective 635nm illumination
        self.parent = parent
        self.mFile = "/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/MicronController/PYTHON/NAPARI-SIM-PROCESSOR/DATA/SIMdata_2019-11-05_15-21-42.tiff" #FIXME: ???
        self.phases_number = 3
        self.angles_number = 3
        self.NA = simParameters.NA
        self.n = simParameters.n
        self.wavelength = wavelength
        self.pixelsize = simParameters.pixelsize
        self.dz= 0.55
        self.alpha = 0.5
        self.beta = 0.98
        self.w = 0.2
        self.eta = simParameters.eta
        self.group = 30
        self.use_phases = True
        self.find_carrier = True
        self.isCalibrated = False
        self.use_gpu = isPytorch
        self.stack = []

        # processing parameters
        self.isRecording = False
        self.allPatterns = []
        self.isReconstructing = False

        # initialize logger
        self._logger = initLogger(self, tryInheritParent=False)

        # switch for the different reconstruction algorithms
        self.reconstructionMethod = "napari"

        # set model
        #h = HexSimProcessor(); #
        if isSIM:
            self.h = ConvSimProcessor()
            self.k_shape = (3,1)
        else:
            self._logger.error("Please install napari sim! pip install napari-sim-processor")

        # setup
        self.h.debug = False
        self.setReconstructor()
        self.kx_input = np.zeros(self.k_shape, dtype=np.single)
        self.ky_input = np.zeros(self.k_shape, dtype=np.single)
        self.p_input = np.zeros(self.k_shape, dtype=np.single)
        self.ampl_input = np.zeros(self.k_shape, dtype=np.single)

        # set up the GPU for mcSIM
        if isGPU:
            # GPU memory usage
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            memory_start = mempool.used_bytes()

    #TODO change
    def loadPattern(self, path=None, filetype="bmp"):
        # sort filenames numerically
        import glob
        import cv2

        if path is None:
            path = sim_parameters["patternPath"]
        allPatternPaths = sorted(glob.glob(os.path.join(path, "*."+filetype)))
        self.allPatterns = []
        for iPatternPath in allPatternPaths:
            mImage = cv2.imread(iPatternPath)
            mImage = cv2.cvtColor(mImage, cv2.COLOR_BGR2GRAY)
            self.allPatterns.append(mImage)
        return self.allPatterns

    def getPattern(self, iPattern):
        # return ith sim pattern
        return self.allPatterns[iPattern]

    def setParameters(self, sim_parameters):
        # uses parameters from GUI
        self.pixelsize= sim_parameters.pixelsize
        self.NA= sim_parameters.NA
        self.n= sim_parameters.n
        self.reconstructionMethod = "napari" # sim_parameters["reconstructionMethod"]
        #self.use_gpu = False #sim_parameters["useGPU"]
        self.eta =  sim_parameters.eta
        self.magnification = sim_parameters.magnification

    def setReconstructionMethod(self, method):
        self.reconstructionMethod = method

    def setReconstructor(self):
        '''
        Sets the attributes of the Processor
        Executed frequently, upon update of several settings
        '''

        self.h.usePhases = self.use_phases
        self.h.magnification = 90
        self.h.NA = self.NA
        self.h.n = self.n
        #self.h.wavelength = self.wavelength
        #self.h.wavelength = 0.52
        self.h.pixelsize = self.pixelsize
        self.h.alpha = self.alpha
        self.h.beta = self.beta
        self.h.w = self.w
        self.h.eta = self.eta
        if not self.find_carrier:
            self.h.kx = self.kx_input
            self.h.ky = self.ky_input

    def getWF(self, mStack):
        # display the BF image
        bfFrame = np.sum(np.array(mStack[-3:]), 0)
        self.parent.sigSIMProcessorImageComputed.emit(bfFrame, "Widefield SUM")

    def setSIMStack(self, stack):
        self.stack = stack

    def getSIMStack(self):
        return np.array(self.stack)

    def clearStack(self):
        self.stack=[]

    def get_current_stack_for_calibration(self,data):
        self._logger.error("get_current_stack_for_calibration not implemented yet")
        '''
        Returns the 4D raw image (angles,phases,y,x) stack at the z value selected in the viewer

        if(0):
            data = np.expand_dims(np.expand_dims(data, 0), 0)
            dshape = data.shape # TODO: Hardcoded ...data.shape
            zidx = 0
            delta = group // 2
            remainer = group % 2
            zmin = max(zidx-delta,0)
            zmax = min(zidx+delta+remainer,dshape[2])
            new_delta = zmax-zmin
            data = data[...,zmin:zmax,:,:]
            phases_angles = phases_number*angles_number
            rdata = data.reshape(phases_angles, new_delta, dshape[-2],dshape[-1])
            cal_stack = np.swapaxes(rdata, 0, 1).reshape((phases_angles * new_delta, dshape[-2],dshape[-1]))
        '''
        return data


    def calibrate(self, imRaw):
        '''
        calibration
        '''
        #self._logger.debug("Starting to calibrate the stack")
        if self.reconstructionMethod == "napari":
            #imRaw = get_current_stack_for_calibration(mImages)
            if type(imRaw) is list:
                imRaw = np.array(imRaw)
            if self.use_gpu:
                self.h.calibrate_pytorch(imRaw, self.find_carrier)
            else:
                #self.h.calibrate(imRaw, self.find_carrier)
                self.h.calibrate(imRaw)
            self.isCalibrated = True
            if self.find_carrier: # store the value found
                self.kx_input = self.h.kx
                self.ky_input = self.h.ky
                self.p_input = self.h.p
                self.ampl_input = self.h.ampl
            self._logger.debug("Done calibrating the stack")
        elif self.reconstructionMethod == "mcsim":
            """
            test running SIM reconstruction at full speed on GPU
            """

            # ############################
            # for the first image, estimate the SIM parameters
            # this step is slow, can take ~1-2 minutes
            # ############################
            self._logger.debug("running initial reconstruction with full parameter estimation")

            # first we need to reshape the stack to become 3x3xNxxNy
            imRawMCSIM = np.stack((imRaw[0:3,],imRaw[3:6,],imRaw[6:,]),0)
            imgset = sim.SimImageSet({"pixel_size": self.pixelsize ,
                                    "na": self.NA,
                                    "wavelength": self.wavelength*1e-3},
                                    imRawMCSIM,
                                    otf=None,
                                    wiener_parameter=0.3,
                                    frq_estimation_mode="band-correlation",
                                    # frq_guess=frqs_gt, # todo: can add frequency guesses for more reliable fitting
                                    phase_estimation_mode="wicker-iterative",
                                    # TODO: what about 3DSIM?
                                    phases_guess=np.array([[0, 2*np.pi / 3, 4 * np.pi / 3],
                                                            [0, 2*np.pi / 3, 4 * np.pi / 3],
                                                            [0, 2*np.pi / 3, 4 * np.pi / 3]]),
                                    combine_bands_mode="fairSIM",
                                    fmax_exclude_band0=0.4,
                                    normalize_histograms=False,
                                    background=100,
                                    gain=2,
                                    use_gpu=self.use_gpu)

            # this included parameter estimation
            imgset.reconstruct()
            # extract estimated parameters
            self.mcSIMfrqs = imgset.frqs
            self.mcSIMphases = imgset.phases - np.expand_dims(imgset.phase_corrections, axis=1)
            self.mcSIMmod_depths = imgset.mod_depths
            self.mcSIMotf = imgset.otf

            # clear GPU memory
            imgset.delete()

    def getIsCalibrated(self):
        return self.isCalibrated


    def reconstructSIMStack(self):
        '''
        reconstruct the image stack asychronously
        '''
        # TODO: Perhaps we should work with quees?
        # reconstruct and save the stack in background to not block the main thread
        if not self.isReconstructing:  # not
            self.isReconstructing=True
            mStackCopy = np.array(self.stack.copy())
            self.mReconstructionThread = threading.Thread(target=self.reconstructSIMStackBackground, args=(mStackCopy,), daemon=True)
            self.mReconstructionThread.start()

    def setRecordingMode(self, isRecording):
        self.isRecording = isRecording

    def setReconstructionMode(self, isReconstruction):
        self.isReconstruction = isReconstruction

    def setDate(self, date):
        self.date = date

    def setWavelength(self, wavelength, sim_parameters):
        self.LaserWL = wavelength
        self.h.wavelength = sim_parameters.wavelength

    def reconstructSIMStackBackground(self, mStack):
        '''
        reconstruct the image stack asychronously
        the stack is a list of 9 images (3 angles, 3 phases)
        '''
        # compute image
        # initialize the model

        self._logger.debug("Processing frames")
        if not self.getIsCalibrated():
            self.setReconstructor()
            self.calibrate(mStack)
        SIMReconstruction = self.reconstruct(mStack)

        # save images eventually
        if self.isRecording:
            def saveImageInBackground(image, filename = None):
                try:
                    self.folder = SIMParameters.path
                    self.filename = os.path.join(self.folder,filename) #FIXME: Remove hardcoded path
                    tif.imwrite(self.filename, image)
                    self._logger.debug("Saving file: "+self.filename)
                except  Exception as e:
                    self._logger.error(e)
            mFilenameRecon = f"{self.date}_SIM_Reconstruction_{self.LaserWL}nm.tif"
            threading.Thread(target=saveImageInBackground, args=(SIMReconstruction, mFilenameRecon,)).start()

        self.parent.sigSIMProcessorImageComputed.emit(np.array(SIMReconstruction), "SIM Reconstruction")
        self.isReconstructing = False


    def reconstruct(self, currentImage):
        '''
        reconstruction
        '''
        if self.reconstructionMethod == "napari":
            # we use the napari reconstruction method
            self._logger.debug("reconstructing the stack with napari")
            assert self.isCalibrated, 'SIM processor not calibrated, unable to perform SIM reconstruction'

            dshape= np.shape(currentImage)
            phases_angles = self.phases_number*self.angles_number
            rdata = currentImage[:phases_angles, :, :].reshape(phases_angles, dshape[-2],dshape[-1])
            #this raises a mem allocation error...
            if self.use_gpu:
                imageSIM = self.h.reconstruct_pytorch(rdata.astype(np.float32)) #TODO:this is left after conversion from torch
            else:
                imageSIM = self.h.reconstruct_rfftw(rdata)

            return imageSIM

        elif self.reconstructionMethod == "mcSIM":
            """
            test running SIM reconstruction at full speed on GPU
            """

            '''
            # load images
            root_dir = os.path.join(Path(__file__).resolve().parent, 'data')
            fname_data = os.path.join(root_dir, "synthetic_microtubules_512.tif")
            imgs = tifffile.imread(fname_data)
            '''
            self._logger.debug("reconstructing the stack with mcsim")

            imgset_next = sim.SimImageSet({"pixel_size": self.dxy,
                                        "na": self.na,
                                        "wavelength": self.wavelength},
                                        currentImage,
                                        otf=self.mcSIMotf,
                                        wiener_parameter=0.3,
                                        frq_estimation_mode="fixed",
                                        frq_guess=self.mcSIMfrqs,
                                        phase_estimation_mode="fixed",
                                        phases_guess=self.mcSIMphases,
                                        combine_bands_mode="fairSIM",
                                        mod_depths_guess=self.mcSIMmod_depths,
                                        use_fixed_mod_depths=True,
                                        fmax_exclude_band0=0.4,
                                        normalize_histograms=False,
                                        background=100,
                                        gain=2,
                                        use_gpu=True,
                                        print_to_terminal=False)

            imgset_next.reconstruct()
            imageSIM = imgset_next.sim_sr.compute()
            return imageSIM

    def simSimulator(self, Nx=512, Ny=512, Nrot=3, Nphi=3):
        Isample = np.zeros((Nx,Ny))
        Isample[np.random.random(Isample.shape)>0.999]=1

        allImages = []
        for iRot in range(Nrot):
            for iPhi in range(Nphi):
                IGrating = 1+np.sin(((iRot/Nrot)*nip.xx((Nx,Ny))+(Nrot-iRot)/Nrot*nip.yy((Nx,Ny)))*np.pi/2+np.pi*iPhi/Nphi)
                allImages.append(nip.gaussf(IGrating*Isample,3))

        allImages=np.array(allImages)
        allImages-=np.min(allImages)
        allImages/=np.max(allImages)
        return allImages


'''#####################################
          SIM PATTERN GENERATOR
      QLASS-based pattern generator.
#####################################'''

class SIMPatternGenerator():
    '''
    SIM pattern generator via QLASS board. 
    Contains:
        -constants (all caps) which are used to define the parameters needed to generate 
         and manipiulate patterns;
        -pattern definitions as arrays, containing their respective sequences, pins, and 
         dwell times;
        -a function for testing pattern translations;
        -a function for generating patterns for aacquisition.
    All other operations (resetting, setting to zero, low-level manual operation) should
    be delegated to the QLASSManager directly, which is accessible also from this object
    as SIMPatternGenerator.drv or from the controller as SIMQLASSController.driver.
    '''

    PIN_DEPH_0 = 11
    PIN_DEPH_41 = 10
    PIN_DEPH_52 = 9
    PIN_DEPH_63 = 8
    PIN_CAM = 12
    VAL_CAM_ON = 50000 
    VAL_CAM_OFF = 0
    PIN_LASER = 13
    VAL_LASER_ON = 50000
    VAL_LASER_OFF = 0

    SEQ_41_2D = [(0,11916),(1,39317),(2,32895),(3,0),    (4,15871),(5,48747),(6,0),    (7,51153)]
    SEQ_52_2D = [(0,40788),(1,39366),(2,33102),(3,0),    (4,40254),(5,40254),(6,50682),(7,0)    ]
    SEQ_63_2D = [(0,40992),(1, 9425),(2,28744),(3,61643),(4,27767),(5,38953),(6,0),    (7,0)    ]
    
    DEPH_41_2D = [    0, 31575, 44231]
    DEPH_52_2D = [    0, 30709, 42961]
    DEPH_63_2D = [    0, 30580, 43389]


    SEQ_41_3D = [(0,13438),(1,40253),(2,23717),(3,0),    (4,15367),(5,49005),(6,0),    (7,51249)]
    SEQ_52_3D = [(0,62459),(1,37006),(2,49507),(3,0),    (4,40588),(5,42548),(6,50097),(7,0)    ]
    SEQ_63_3D = [(0,33014),(1,52267),(2,52353),(3,49815),(4,29177),(5,38426),(6,0),    (7,0)    ]
    
    SEQ_UNIFORM = [(0,12559),(1,37283),(2,51028),(3,0),(4,23178),(5,34435),(6,0),(7,0)]

    #first element for each couple is for PIN_DEPH_0, second element is for PIN_DEPH_[couple]
    DEPH_41_3D = [(0,0),(38709, 21812),(1026,  30339),(38088, 36968),(52951, 42246),(37154, 46210),(52465, 51157)]
    DEPH_52_3D = [(0,0),(20484, 24099),(28609, 33658),(34876, 41031),(40016, 47078)                              ]
    DEPH_63_3D = [(0,0),(29577, 21304),(18296, 29965),(32737, 35263),(23245, 40702),(35754, 44902),(46116, 49104)]
    #DEPH_63_3D = [[0,0],[29104, 21300],[18017, 29810],[33232, 35660],[23120, 40515],[35988, 44995],[27632, 49197]]
    
    pattern2D_41 = {
        'id'  : '41',
        'is3D': False,
        'seq' : SEQ_41_2D,
        'deph': DEPH_41_2D,
        'delay_rotation'   : 0.250,
        'delay_translation': 0.025,
        'deph_pin' : PIN_DEPH_41
    }
    pattern3D_41 = {
        'id'  : '41',
        'is3D': True,
        'seq' : SEQ_41_3D,
        'deph': DEPH_41_3D,
        'delay_rotation'   : 0.100,
        'delay_translation': 0.050,
        'deph_pin' : PIN_DEPH_41
    }
    pattern2D_52 = {
        'id'  : '52',
        'is3D': False,
        'seq' : SEQ_52_2D,
        'deph': DEPH_52_2D,
        'delay_rotation'   : 0.025,
        'delay_translation': 0.025,
        'deph_pin' : PIN_DEPH_52
    }
    pattern3D_52 = {
        'id'  : '52',
        'is3D': True,
        'seq' : SEQ_52_3D,
        'deph': DEPH_52_3D,
        'delay_rotation'   : 0.300,
        'delay_translation': 0.050,
        'deph_pin' : PIN_DEPH_52
    }
    pattern2D_63 = {
        'id'  : '63',
        'is3D': False,
        'seq' : SEQ_63_2D,
        'deph': DEPH_63_2D,
        'delay_rotation'   : 0.025,
        'delay_translation': 0.025,
        'deph_pin' : PIN_DEPH_63
    }
    pattern3D_63 = {
        'id'  : '63',
        'is3D': True,
        'seq' : SEQ_63_3D,
        'deph': DEPH_63_3D,
        'delay_rotation'   : 0.200,
        'delay_translation': 0.200,
        'deph_pin' : PIN_DEPH_63
    }
    
    pattern_uniform = {
        'id'  : 'uniform',
        'is3D': False,
        'seq' : SEQ_UNIFORM,
        'deph' : [0],
        'delay_rotation'   : 0.050,
        'delay_translation': 0.025,
        'deph_pin' : 11 #must be something valid, but no DAC voltage !=0 is ever applied here
    }
    pattern_structured_HiLo =  {
        'id'  : '63',
        'is3D': False,
        'seq' : SEQ_63_2D,
        'deph': [0],
        'delay_rotation'   : 0.050,
        'delay_translation': 0.025,
        'deph_pin' : PIN_DEPH_63
    }
    

    def __init__(self,driver,laser):
        self.drv = driver
        self.laser = laser
        self.active_pattern = None
        self._logger = initLogger(self)
        
    
    def set_uniform_illumination(self):
        '''
        Sets the QLASS board to uniform illumination, i.e. no SIM pattern. Useful for avoiding photobleaching.
        '''
        pattern = self.pattern_uniform
        for (pin,seq_dacval) in pattern['seq']:
            self.drv.set_current_level(pin,seq_dacval)
        self.drv.update()
        self.active_pattern = pattern


    def generate_patterns_acquisition(self,mode,expTime,repNum=1,triggerCamera=False,triggerLaser=False):
        '''
        Displays patterns for acquisition, i.e. at "full speed" and with camera and laser triggering.
        '''

        if mode == '2DSIM':
            patterns = [self.pattern2D_41,self.pattern2D_63,self.pattern2D_52]
        elif mode == '3DSIM_FULL':
            patterns = [self.pattern3D_41,self.pattern3D_52,self.pattern3D_63]
        elif mode == '3DSIM_HYBRID':
            patterns = [self.pattern2D_41,self.pattern3D_52,self.pattern2D_63]
        elif mode == 'HiLo':
            patterns = [self.pattern_uniform,self.pattern_structured_HiLo,self.pattern_uniform,self.pattern_structured_HiLo]
        else: 
            self._logger.error(f'Invalid mode retrieved: {mode}')
            raise ValueError(f'Invalid mode retrieved: {mode}')
        
        self.drv.autoUpdate = False
        #self.drv.set_zero()
        self._logger.debug(f'Acquiring with exposure time = {expTime}')

        if triggerLaser:
            self.laser.setEnabled(False)

        for rep in range(repNum):
            for pattern in patterns:
                #apply pattern   
                if not self.active_pattern == pattern:
                    for (pin,seq_dacval) in pattern['seq']:
                        self.drv.set_current_level(pin,seq_dacval)
                    self.drv.update()   #this takes roughly 30 ms overall
                self.active_pattern = pattern
                time.sleep(pattern['delay_rotation'])

                id = pattern['id']
                self._logger.debug(f'Starting pattern {id}')

                #start applying patterns
                deph_dacvals = pattern['deph']
                totalSteps = len(deph_dacvals)
                for imgIdx in range(totalSteps):
                    #first step does not need further delay
                    if imgIdx != 0:
                        time.sleep(pattern['delay_translation'])

                    #trigger camera and lasers
                    if triggerLaser:
                        self.laser.setEnabled(True)
                    if triggerCamera:
                        self.drv.set_current_level(self.PIN_CAM,self.VAL_CAM_ON)
                        self.drv.update()
                        
                    t0 = time.time()

                    #camera trigger line set to OFF. 
                    # TODO remove update if trigger mode is level trigger instead of edge trigger
                    if triggerCamera:
                        self.drv.set_current_level(self.PIN_CAM,self.VAL_CAM_OFF)
                        self.drv.update()


                    if imgIdx != totalSteps-1:
                        if pattern['is3D']:
                            self.drv.set_current_level(self.PIN_DEPH_0,deph_dacvals[imgIdx+1][0])
                            self.drv.set_current_level(pattern['deph_pin'],deph_dacvals[imgIdx+1][1])
                        else:
                            self.drv.set_current_level(pattern['deph_pin'],deph_dacvals[imgIdx+1])
                    else: #if this was the last step, set DAC values to 0 instead.
                        if pattern['is3D']:
                            self.drv.set_current_level(self.PIN_DEPH_0,0)
                            self.drv.set_current_level(pattern['deph_pin'],0)
                        else:
                            self.drv.set_current_level(pattern['deph_pin'],0)

                    #wait until the time to update has arrived
                    while time.time()-t0 < expTime:
                        time.sleep(0.001)
                    
                    if triggerLaser:
                        self.laser.setEnabled(False)

                    self.drv.update()
                    #TODO it would be smart here to actually log the dac value changes for
                    #the next pattern here. That would save roughly 100 ms per full reconstructed image.
        self.set_uniform_illumination() #avoid photobleaching the last pattern into the sample
     
                                 
    def generate_pattern_translations(self,expTime,triggerCamera:bool=False):
        '''
        Sets a pattern and acquires images for equally spaced values of the dephasing resistor.
        '''
        pattern = self.active_pattern
        if pattern == self.pattern_uniform:
            self._logger.warning('Cannot perform calibration on uniform pattern. Returning.')
            return
        elif pattern['is3D'] == True:
            self._logger.warning('Calibration of 3D patterns not yet implemented. Returning.')
            return            
        
        pinDAC = pattern['deph_pin']
        dacvals = np.linspace(0,55000,101,dtype=int)
        
        for val in dacvals:
            self.drv.set_current_level(pinDAC,int(val))
            self.drv.update()
            print(f'Calibration status: {val/max(dacvals)*100:.2f}%',flush=True,end='\r',sep='')
            time.sleep(pattern['delay_translation'])
            
            if triggerCamera:
                self.drv.set_current_level(self.PIN_CAM,self.VAL_CAM_ON)
                self.drv.update()
                self.drv.set_current_level(self.PIN_CAM,self.VAL_CAM_OFF)
                self.drv.update()
            
            time.sleep(expTime)
            
        self.drv.set_current_level(pinDAC,0)
        self.drv.update()
                          

    def generate_patterns_testing(self,is3D,patternID,cycleThroughPhases: bool=False,stopEvent = None ):
        '''
        Generates patterns slowly, for testing in alignment etc.
        If cycleThroughPhases is True, it is advisable to call this function in a separate thread.
        '''
        #self.drv.set_zero()
        if patternID == 'none':
            self.set_uniform_illumination()	
            return
        elif patternID == '41':
            pattern = self.pattern3D_41 if is3D else self.pattern2D_41
        elif patternID == '52':
            pattern = self.pattern3D_52 if is3D else self.pattern2D_52
        elif patternID == '63':
            pattern = self.pattern3D_63 if is3D else self.pattern2D_63
        else:
            self._logger.error(f'Invalid pattern ID retrieved: {patternID}')
            raise ValueError(f'Invalid pattern ID retrieved: {patternID}')

        self.drv.autoUpdate = False
        for pin,dacval in pattern['seq']:
            self.drv.set_current_level(pin,dacval)
        self.drv.update()
        self.active_pattern = pattern


        # if we want to cycle through phases, we show each pattern position in a loop
        # changing pattern translation 1/second. This cycle is repeated until the stop 
        # button is pressed or until the pattern is changed from the UI.
        if cycleThroughPhases:
            t0 = time.time()
            if stopEvent is None:
                self._logger.error(f'Tried to start testing cycle without a defined stop event; aborting test cycle.')
                return
            while not stopEvent.is_set(): #interrupt via UI
                for elem in pattern['deph']:
                    if is3D:
                        self.drv.set_current_level(self.PIN_DEPH_0,elem[0])
                        self.drv.set_current_level(pattern['deph_pin'],elem[1])
                    else:
                        self.drv.set_current_level(pattern["deph_pin"],elem)
                    self.drv.update()
                    time.sleep(1)
            if is3D:
                self.drv.set_current_level(self.PIN_DEPH_0,0)
                self.drv.set_current_level(pattern['deph_pin'],0)
            else:
                self.drv.set_current_level(pattern["deph_pin"],0)
            self.drv.update()
            self._logger.debug(f'Testing cycle concluded after {time.time()-t0:.1f} seconds')

        return
    

'''#####################################
      Unused functions and classes.
#####################################'''

class SIMClient:
    '''
    FastAPI client that is used to communicate with DMD. 
    Unused in QLASS implementation.
    '''

    def __init__(self, URL, PORT):
        self.base_url = f"http://{URL}:{PORT}"
        self.commands = {
            "start": "/start_viewer/",
            "single_run": "/start_viewer_single_loop/",
            "pattern_compeleted": "/wait_for_viewer_completion/",
            "pause_time": "/set_wait_time/",
            "stop_loop": "/stop_viewer/",
            "pattern_wl": "/change_wavelength/",
            "display_pattern": "/display_pattern/",
        }
        self.iseq = 60
        self.itime = 120
        self.laser_power = (400, 250)

    def get_request(self, url, timeout=0.3):
        try:
            response = requests.get(url, timeout=timeout)
            return response.json()
        except Exception as e:
            print(e)
            return -1

    def start_viewer(self):
        url = self.base_url + self.commands["start_viewer"]
        return self.get_request(url)

    def start_viewer_single_loop(self, number_of_runs, timeout=2):
        url = f"{self.base_url}{self.commands['single_run']}{number_of_runs}"
        return self.get_request(url, timeout=timeout)

    def wait_for_viewer_completion(self):
        url = self.base_url + self.commands["pattern_compeleted"]
        self.get_request(url)

    def set_pause(self, period):
        url = f"{self.base_url}{self.commands['pause_time']}{period}"
        self.get_request(url)

    def stop_loop(self):
        url = self.base_url + self.commands["stop_loop"]
        self.get_request(url)

    def set_wavelength(self, wavelength):
        url = f"{self.base_url}{self.commands['pattern_wl']}{wavelength}"
        self.get_request(url)

    def display_pattern(self, iPattern):
        url = f"{self.base_url}{self.commands['display_pattern']}{iPattern}"
        self.get_request(url)


class OldFuncs:
    '''
    Unused functions. These are here for reference only, but are not expected to work.
    '''
    
    #@APIExport(runOnUIThread=True)
    def performSIMExperimentThread(self, sim_parameters):
        """
        Iterate over all SIM patterns, display them and acquire images
        """
        self.patternID = 0
        self.isReconstructing = False
        nColour = 2 #[488, 635]
        dic_wl = [488, 635]

        # retreive Z-stack parameters
        zStackParameters = self._widget.getZStackParameters()
        zMin, zMax, zStep = zStackParameters[0], zStackParameters[1], zStackParameters[2] # if zStep < 0, it will not move in z
        tDebounce = 0.1 # debounce time between z-steps

        # retreive timelapse parameters
        timelapsedParameters = self._widget.getTimelapseParameters()
        timePeriod, Nframes = timelapsedParameters[0], timelapsedParameters[1] # if NFrames < 0, it will run indefinitely

        # get current z-position
        zPosInitially = self.positioner.getPosition()["Z"]

        # run the experiment indefinitely
        while self.active:

            # iterate over all z-positions
            if zStep > 0:
                allZPositions = np.arange(zMin, zMax, zStep)
            else:
                allZPositions = [0]

            for iColour in range(nColour):
                # toggle laser
                if not self.active:
                    if len(allZPositions)!=1:
                        self.positioner.move(value=zPosInitially, axis="Z", is_absolute=True, is_blocking=True)
                        time.sleep(tDebounce)
                    break

                if iColour == 0 and self.is488 and self.lasers[iColour].power>0.0:
                    # enable laser 1
                    self.lasers[0].setEnabled(True)
                    self.lasers[1].setEnabled(False)
                    self._logger.debug("Switching to pattern"+self.lasers[0].name)
                    processor = self.SimProcessorLaser1
                    processor.setParameters(sim_parameters)
                    self.LaserWL = processor.wavelength
                    # set the pattern-path for laser wl 1
                elif iColour == 1 and self.is635 and self.lasers[iColour].power>0.0:
                    # enable laser 2
                    self.lasers[0].setEnabled(False)
                    self.lasers[1].setEnabled(True)
                    self._logger.debug("Switching to pattern"+self.lasers[1].name)
                    processor = self.SimProcessorLaser2
                    processor.setParameters(sim_parameters)
                    self.LaserWL = processor.wavelength
                    # set the pattern-path for laser wl 1
                else:
                    time.sleep(.1) # reduce CPU load
                    continue

                # select the pattern for the current colour
                self.SIMClient.set_wavelength(dic_wl[iColour])

                for zPos in allZPositions:
                    # move to the next z-position
                    if len(allZPositions)!=1:
                        self.positioner.move(value=zPos+zPosInitially, axis="Z", is_absolute=True, is_blocking=True)
                        time.sleep(tDebounce)

                    if self.isPCO:
                        # display one round of SIM patterns for the right colour
                        self.SIMClient.start_viewer_single_loop(1)

                        # ensure lasers are off to avoid photo damage
                        self.lasers[0].setEnabled(False)
                        self.lasers[1].setEnabled(False)

                        # download images from the camera
                        self.SIMStack = self.detector.getChunk(); self.detector.flushBuffers()
                        if self.SIMStack is None:
                            self._logger.error("No image received")
                            continue
                    else:
                        # we need to capture images and display patterns one-by-one
                        self.SIMStack = []
                        try:
                            mExposureTime = self.detector.getParameter("exposure")/1e6 # s^-1
                        except:
                            mExposureTime = 0.1
                        for iPattern in range(9):
                            self.SIMClient.display_pattern(iPattern)
                            time.sleep(mExposureTime) # make sure we take the next newest frame to avoid motion blur from the pattern change

                            # Todo: Need to ensure thatwe have the right pattern displayed and the buffer is free - this heavily depends on the exposure time..
                            mFrame = None
                            lastFrameNumber = -1
                            timeoutFrameRequest = 3 # seconds
                            cTime = time.time()
                            frameRequestNumber = 0
                            while(1):
                                # something went wrong while capturing the frame
                                if time.time()-cTime> timeoutFrameRequest:
                                    break
                                mFrame, currentFrameNumber = self.detector.getLatestFrame(returnFrameNumber=True)
                                if currentFrameNumber <= lastFrameNumber:
                                    time.sleep(0.05)
                                    continue  
                                frameRequestNumber += 1
                                if frameRequestNumber > self.nsimFrameSyncVal:
                                    print(f"Frame number used for stack: {currentFrameNumber}") 
                                    break
                                lastFrameNumber = currentFrameNumber
                                
                                #mFrame = self.detector.getLatestFrame() # get the next frame after the pattern has been updated
                            self.SIMStack.append(mFrame)
                        if self.SIMStack is None:
                            self._logger.error("No image received")
                            continue

                    self.sigImageReceived.emit(np.array(self.SIMStack),"SIMStack"+str(processor.wavelength))
                    processor.setSIMStack(self.SIMStack)
                    processor.getWF(self.SIMStack)

                    # activate recording in processor
                    processor.setRecordingMode(self.isRecording)
                    processor.setReconstructionMode(self.isReconstruction)
                    processor.setWavelength(self.LaserWL,sim_parameters)


                    # store the raw SIM stack
                    if self.isRecording and self.lasers[iColour].power>0.0:
                        date = datetime.now().strftime("%Y_%m_%d-%I-%M-%S_%p")
                        processor.setDate(date)
                        mFilenameStack = f"{date}_SIM_Stack_{self.LaserWL}nm_{zPos+zPosInitially}mum.tif"
                        threading.Thread(target=self.saveImageInBackground, args=(self.SIMStack, mFilenameStack,), daemon=True).start()
                    # self.detector.stopAcquisition()
                    # We will collect N*M images and process them with the SIM processor

                    # process the frames and display
                    processor.reconstructSIMStack()

                    # reset the per-colour stack to add new frames in the next imaging series
                    processor.clearStack()

                # move back to initial position
                if len(allZPositions)!=1:
                    self.positioner.move(value=zPosInitially, axis="Z", is_absolute=True, is_blocking=True)
                    time.sleep(tDebounce)


            # wait for the next rund
            time.sleep(timePeriod)


    def performSIMTimelapseThread(self, sim_parameters):
        """
        Do timelapse SIM
        Q: should it have a separate thread?
        """
        self.isReconstructing = False
        nColour = 2 #[488, 635]
        dic_wl = [488, 635]
        for iColour in range(nColour):
            # toggle laser
            if not self.active:
                break

            if iColour == 0 and self.is488 and self.lasers[iColour].power>0.0:
                # enable laser 1
                self.lasers[0].setEnabled(True)
                self.lasers[1].setEnabled(False)
                self._logger.debug("Switching to pattern"+self.lasers[0].name)
                processor = self.SimProcessorLaser1
                processor.setParameters(sim_parameters)
                self.LaserWL = processor.wavelength
                # set the pattern-path for laser wl 1
            elif iColour == 1 and self.is635 and self.lasers[iColour].power>0.0:
                # enable laser 2
                self.lasers[0].setEnabled(False)
                self.lasers[1].setEnabled(True)
                self._logger.debug("Switching to pattern"+self.lasers[1].name)
                processor = self.SimProcessorLaser2
                processor.setParameters(sim_parameters)
                self.LaserWL = processor.wavelength
                # set the pattern-path for laser wl 1
            else:
                continue


            # select the pattern for the current colour
            self.SIMClient.set_wavelength(dic_wl[iColour])

            # display one round of SIM patterns for the right colour
            self.SIMClient.start_viewer_single_loop(1)

            # ensure lasers are off to avoid photo damage
            self.lasers[0].setEnabled(False)
            self.lasers[1].setEnabled(False)

            # download images from the camera
            self.SIMStack = self.detector.getChunk(); self.detector.flushBuffers()
            if self.SIMStack is None:
                self._logger.error("No image received")
                continue
            self.sigImageReceived.emit(np.array(self.SIMStack),"SIMStack"+str(processor.wavelength))
            processor.setSIMStack(self.SIMStack)


            # activate recording in processor
            processor.setRecordingMode(self.isRecording)
            processor.setReconstructionMode(self.isReconstruction)
            processor.setWavelength(self.LaserWL,sim_parameters)

            # store the raw SIM stack
            if self.isRecording and self.lasers[iColour].power>0.0:
                uniqueID = np.random.randint(0,1000)
                date = datetime.now().strftime("%Y_%m_%d-%I-%M-%S_%p")
                processor.setDate(date)
                mFilenameStack = f"{date}_SIM_Stack_{self.LaserWL}nm_{uniqueID}.tif"
                threading.Thread(target=self.saveImageInBackground, args=(self.SIMStack, mFilenameStack,), daemon=True).start()
            # self.detector.stopAcquisition()
            # We will collect N*M images and process them with the SIM processor

            # process the frames and display
            #processor.reconstructSIMStack()

            # reset the per-colour stack to add new frames in the next imaging series
            processor.clearStack()
   

    def performSIMZstackThread(self,sim_parameters,zDis,zStep):
        mStep = 0
        acc = 0    #hardcoded acceleration
        mspeed = 1000   #hardcoded speed
        while mStep < zStep:
            self.positioner.move(zDis,acceleration = acc, speed=mspeed)
            mStep += 1
            self.performSIMTimelapseThread(sim_parameters)
            time.sleep(0.1)
        self.active = False
        self.lasers[0].setEnabled(False)
        self.lasers[1].setEnabled(False)
        if self.isPCO:
            self.detector.setParameter("trigger_source","Internal trigger")
            self.detector.setParameter("buffer_size",-1)
        self.detector.flushBuffers()
        self._logger.debug("Zstack finished")

    # #@APIExport(runOnUIThread=True)
    # def sim_getSnapAPI(self, mystack):
    #     mystack.append(self.detector.getLatestFrame())
    #     #print(np.shape(mystack))

    # def toggle488Laser(self):
    #     self.is488 = not self.is488
    #     if self.is488:
    #         self._widget.is488LaserButton.setText("488 on")
    #     else:
    #         self._widget.is488LaserButton.setText("488 off")

    # def toggle635Laser(self):
    #     self.is635 = not self.is635
    #     if self.is635:
    #         self._widget.is635LaserButton.setText("635 on")
    #     else:
    #         self._widget.is635LaserButton.setText("635 off")

    # def patternIDChanged(self, patternID):
    #     wl = self.getpatternWavelength()
    #     if wl == 'Laser 488nm':
    #         laserTag = 0
    #     elif wl == 'Laser 635nm':
    #         laserTag = 1
    #     else:
    #         laserTag = 0
    #         self._logger.error("The laser wavelenth is not implemented")
    #     self.simPatternByID(patternID,laserTag)

    #     #@APIExport(runOnUIThread=True)
    # def simPatternByID(self, patternID: int, wavelengthID: int):
    #     try:
    #         patternID = int(patternID)
    #         wavelengthID = int(wavelengthID)
    #         self.SIMClient.set_wavelength(wavelengthID)
    #         self.SIMClient.display_pattern(patternID)
    #         return wavelengthID
    #     except Exception as e:
    #         self._logger.error(e)

    # def initFastAPISIM(self, params):
    #     self.fastAPISIMParams = params
    #     self.IS_FASTAPISIM = True

    #     # Usage example
    #     host = self.fastAPISIMParams["host"]
    #     port = self.fastAPISIMParams["port"]
    #     tWaitSequence = self.fastAPISIMParams["tWaitSquence"]

    #     if tWaitSequence is None:
    #         tWaitSequence = 0.1
    #     if host is None:
    #         host = "169.254.165.4"
    #     if port is None:
    #         port = 8000

    #     self.SIMClient = SIMClient(URL=host, PORT=port)
    #     self.SIMClient.set_pause(tWaitSequence)

    # def initHamamatsuSLM(self):
    #     self.hamamatsuslm = nip.HAMAMATSU_SLM() # FIXME: Add parameters
    #     # def __init__(self, dll_path = None, OVERDRIVE = None, use_corr_pattern = None, wavelength = None, corr_pattern_path = None):
    #     allPatterns = self._master.simManager.allPatterns[self.patternID]
    #     for im_number, im in enumerate(allPatterns):
    #         self.hamamatsuslm.send_dat(im, im_number)

    
    # def toggleSIMDisplay(self, enabled=True):
    #     self._widget.setSIMDisplayVisible(enabled)

    # def monitorChanged(self, monitor):
    #     self._widget.setSIMDisplayMonitor(monitor)

    def getpatternWavelength(self):
        return 660

    # def displayMask(self, image):
    #     self._widget.updateSIMDisplay(image)

    # def setIlluPatternByID(self, iRot, iPhi):
    #     self.detector.setIlluPatternByID(iRot, iPhi)




# Copyright (C) 2020-2023 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
