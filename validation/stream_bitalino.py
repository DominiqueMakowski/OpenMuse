# -*- coding: utf-8 -*-
"""
.. module:: bitalino
   :synopsis: BITalino API

*Created on Fri Jun 20 2014*

*Last Modified on Thur Jun 25 2015*
"""
__author__ = "Pedro Gonçalves & Carlos Azevedo"
__credits__ = [
    "Carlos Azevedo",
    "Pedro Gonçalves",
    "Hugo Silva",
    "Takuma Hashimoto",
    "Rui Freixo",
    "Margarida Reis",
]
__license__ = "GPL"
__version__ = "v3"
__email__ = "bitalino@plux.info"


import math
import platform
import re
import select
import socket
import struct
import sys
import time

import numpy
import serial


def find():
    """
    :returns: list of (tuples) with name and MAC address of each device found

    Searches for bluetooth devices nearby.
    """
    if platform.system() == "Windows" or platform.system() == "Linux":
        try:
            import bluetooth
        except Exception as e:
            raise Exception(ExceptionCode.IMPORT_FAILED + str(e))
        nearby_devices = bluetooth.discover_devices(lookup_names=True)
        return nearby_devices
    else:
        raise Exception(ExceptionCode.INVALID_PLATFORM)


class ExceptionCode:
    INVALID_ADDRESS = "The specified address is invalid."
    INVALID_PLATFORM = "This platform does not support bluetooth connection."
    CONTACTING_DEVICE = "The computer lost communication with the device."
    DEVICE_NOT_IDLE = "The device is not idle."
    DEVICE_NOT_IN_ACQUISITION = "The device is not in acquisition mode."
    INVALID_PARAMETER = "Invalid parameter."
    INVALID_VERSION = "Only available for Bitalino 2.0."
    IMPORT_FAILED = "Please connect using the Virtual COM Port or confirm that PyBluez is installed; bluetooth wrapper failed to import with error: "


class BITalino(object):
    """
    :param macAddress: MAC address or serial port for the bluetooth device
    :type macAddress: str
    :param timeout: maximum amount of time (seconds) elapsed while waiting for the device to respond
    :type timeout: int, float or None
    :raises Exception: invalid MAC address or serial port
    :raises Exception: invalid timeout value

    Connects to the bluetooth device with the MAC address or serial port provided.

    Possible values for parameter *macAddress*:

    * MAC address: e.g. ``00:0a:95:9d:68:16``
    * Serial port - device name: depending on the operating system. e.g. ``COM3`` on Windows; ``/dev/tty.bitalino-DevB`` on Mac OS X; ``/dev/ttyUSB0`` on GNU/Linux.
    * IP address and port - server: e.g. ``192.168.4.1:8001``

    Possible values for *timeout*:

    ===============  ================================================================
    Value            Result
    ===============  ================================================================
    None             Wait forever
    X                Wait X seconds for a response and raises a connection Exception
    ===============  ================================================================
    """

    def __init__(self, macAddress, timeout=None):
        regCompiled = re.compile("^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$")
        checkMatch = re.match(regCompiled, macAddress)
        self.isPython2 = True if sys.version_info[0] == 2 else False
        self.blocking = True if timeout is None else False
        if not self.blocking:
            try:
                self.timeout = float(timeout)
            except Exception:
                raise Exception(ExceptionCode.INVALID_PARAMETER)
        if checkMatch:
            if platform.system() == "Windows" or platform.system() == "Linux":
                try:
                    import bluetooth
                except Exception as e:
                    raise Exception(ExceptionCode.IMPORT_FAILED + str(e))
                self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                self.socket.connect((macAddress, 1))
                self.wifi = False
                self.serial = False
            else:
                raise Exception(ExceptionCode.INVALID_PLATFORM)
        elif (macAddress[0:3] == "COM" and platform.system() == "Windows") or (
            macAddress[0:5] == "/dev/" and platform.system() != "Windows"
        ):
            self.socket = serial.Serial(macAddress, 115200)
            self.wifi = False
            self.serial = True
        elif macAddress.count(":") == 1:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((macAddress.split(":")[0], int(macAddress.split(":")[1])))
            self.wifi = True
            self.serial = False
        else:
            raise Exception(ExceptionCode.INVALID_ADDRESS)
        self.started = False
        self.macAddress = macAddress
        split_string = "_v"
        split_string_old = "V"
        version = self.version()
        if split_string in version:
            version_nbr = float(version.split(split_string)[1][:3])
        else:
            version_nbr = float(version.split(split_string_old)[1][:3])
        self.isBitalino2 = True if version_nbr >= 4.2 else False
        self.isBitalino52 = True if version_nbr >= 5.2 else False

    def start(self, SamplingRate=1000, analogChannels=[0, 1, 2, 3, 4, 5]):
        """
        :param SamplingRate: sampling frequency (Hz)
        :type SamplingRate: int
        :param analogChannels: channels to be acquired
        :type analogChannels: array, tuple or list of int
        :raises Exception: device already in acquisition (not IDLE)
        :raises Exception: sampling rate not valid
        :raises Exception: list of analog channels not valid

        Sets the sampling rate and starts acquisition in the analog channels set.
        Setting the sampling rate and starting the acquisition implies the use of the method :meth:`send`.

        Possible values for parameter *SamplingRate*:

        * 1
        * 10
        * 100
        * 1000

        Possible values, types, configurations and examples for parameter *analogChannels*:

        ===============  ====================================
        Values           0, 1, 2, 3, 4, 5
        Types            list ``[]``, tuple ``()``, array ``[[]]``
        Configurations   Any number of channels, identified by their value
        Examples         ``[0, 3, 4]``, ``(1, 2, 3, 5)``
        ===============  ====================================

        .. note:: To obtain the samples, use the method :meth:`read`.
        """
        if self.started is False:
            if int(SamplingRate) not in [1, 10, 100, 1000]:
                raise Exception(ExceptionCode.INVALID_PARAMETER)

            # CommandSRate: <Fs>  0  0  0  0  1  1
            if int(SamplingRate) == 1000:
                commandSRate = 3
            elif int(SamplingRate) == 100:
                commandSRate = 2
            elif int(SamplingRate) == 10:
                commandSRate = 1
            elif int(SamplingRate) == 1:
                commandSRate = 0

            if isinstance(analogChannels, list):
                analogChannels = analogChannels
            elif isinstance(analogChannels, tuple):
                analogChannels = list(analogChannels)
            elif isinstance(analogChannels, numpy.ndarray):
                analogChannels = analogChannels.astype("int").tolist()
            else:
                raise Exception(ExceptionCode.INVALID_PARAMETER)

            analogChannels = list(set(analogChannels))

            if (
                len(analogChannels) == 0
                or len(analogChannels) > 6
                or any([item not in range(6) or type(item) != int for item in analogChannels])
            ):
                raise Exception(ExceptionCode.INVALID_PARAMETER)

            self.send((commandSRate << 6) | 0x03)

            # CommandStart: A6 A5 A4 A3 A2 A1 0  1
            commandStart = 1
            for i in analogChannels:
                commandStart = commandStart | 1 << (2 + i)

            self.send(commandStart)
            self.started = True
            self.analogChannels = analogChannels
        else:
            raise Exception(ExceptionCode.DEVICE_NOT_IDLE)

    def stop(self):
        """
        :raises Exception: device not in acquisition (IDLE)

        Stops the acquisition. Stoping the acquisition implies the use of the method :meth:`send`.
        """
        if self.started:
            self.send(0)
        else:
            if self.isBitalino2:
                # Command: 1  1  1  1  1  1  1  1 - Go to idle mode from all modes.
                self.send(255)
            else:
                raise Exception(ExceptionCode.DEVICE_NOT_IN_ACQUISITION)
        self.started = False

    def close(self):
        """
        Closes the bluetooth or serial port socket.
        """
        if self.wifi:
            self.socket.settimeout(1.0)  # force a timeout on TCP/IP sockets
            try:
                self.receive(1024)  # receive any pending data
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
            except socket.timeout:
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
        else:
            self.socket.close()

    def send(self, data):
        """
        Sends a command to the BITalino device.
        """
        time.sleep(0.1)
        if self.serial:
            if self.isPython2:
                self.socket.write(chr(data))
            else:
                self.socket.write(bytes([data]))
        else:
            if self.isPython2:
                self.socket.send(chr(data))
            else:
                self.socket.send(bytes([data]))

    def battery(self, value=0):
        """
        :param value: threshold value
        :type value: int
        :raises Exception: device in acquisition (not IDLE)
        :raises Exception: threshold value is invalid

        Sets the battery threshold for the BITalino device. Setting the battery threshold implies the use of the method :meth:`send`.

        Possible values for parameter *value*:

        ===============  =======  =====================
        Range            *value*  Corresponding threshold (Volts)
        ===============  =======  =====================
        Minimum *value*  0        3.4 Volts
        Maximum *value*  63       3.8 Volts
        ===============  =======  =====================
        """
        if self.started is False:
            if 0 <= int(value) <= 63:
                # CommandBattery: <bat   threshold> 0  0
                commandBattery = int(value) << 2
                self.send(commandBattery)
            else:
                raise Exception(ExceptionCode.INVALID_PARAMETER)
        else:
            raise Exception(ExceptionCode.DEVICE_NOT_IDLE)

    def pwm(self, pwmOutput=100):
        """
        :param pwmOutput: value for the pwm output
        :type pwmOutput: int
        :raises Exception: invalid pwm output value
        :raises Exception: device is not a BITalino 2.0

        Sets the pwm output for the BITalino 2.0 device. Implies the use of the method :meth:`send`.

        Possible values for parameter *pwmOutput*: 0 - 255.
        """
        if self.isBitalino2:
            if 0 <= int(pwmOutput) <= 255:
                self.send(163)
                self.send(pwmOutput)
            else:
                raise Exception(ExceptionCode.INVALID_PARAMETER)
        else:
            raise Exception(ExceptionCode.INVALID_VERSION)

    def state(self):
        """
        :returns: dictionary with the state of all channels
        :raises Exception: device is not a BITalino version 2.0
        :raises Exception: device in acquisition (not IDLE)
        :raises Exception: lost communication with the device when data is corrupted

        Returns the state of all analog and digital channels. Reading channel State from BITalino implies the use of the method :meth:`send` and :meth:`receive`.
        The returned dictionary structure contains the following key-value pairs:

        =================  ================================ ============== =====================
        Key                Value                            Type           Examples
        =================  ================================ ============== =====================
        analogChannels     Value of all analog channels     Array of int   [A1 A2 A3 A4 A5 A6]
        battery            Value of the battery channel     int
        batteryThreshold   Value of the battery threshold   int            :meth:`battery`
        digitalChannels    Value of all digital channels    Array of int   [I1 I2 O1 O2]
        =================  ================================ ============== =====================
        """
        if self.isBitalino2:
            if self.started is False:
                # CommandState: 0  0  0  0  1  0  1  1
                # Response: <A1 (2 bytes: 0..1023)> <A2 (2 bytes: 0..1023)> <A3 (2 bytes: 0..1023)>
                #           <A4 (2 bytes: 0..1023)> <A5 (2 bytes: 0..1023)> <A6 (2 bytes: 0..1023)>
                #           <ABAT (2 bytes: 0..1023)>
                #           <Battery threshold (1 byte: 0..63)>
                #           <Digital ports + CRC (1 byte: I1 I2 O1 O2 <CRC 4-bit>)>
                self.send(11)
                if self.isBitalino52:
                    number_bytes = 17
                else:
                    number_bytes = 16
                Data = self.receive(number_bytes)
                decodedData = list(struct.unpack(number_bytes * "B ", Data))
                crc = decodedData[-1] & 0x0F
                decodedData[-1] = decodedData[-1] & 0xF0
                x = 0
                for i in range(number_bytes):
                    for bit in range(7, -1, -1):
                        x = x << 1
                        if x & 0x10:
                            x = x ^ 0x03
                        x = x ^ ((decodedData[i] >> bit) & 0x01)
                if crc == x & 0x0F:
                    digitalPorts = []
                    digitalPorts.append(decodedData[-1] >> 7 & 0x01)
                    digitalPorts.append(decodedData[-1] >> 6 & 0x01)
                    digitalPorts.append(decodedData[-1] >> 5 & 0x01)
                    digitalPorts.append(decodedData[-1] >> 4 & 0x01)
                    offset = 0
                    if self.isBitalino52:
                        offset = -1
                    batteryThreshold = decodedData[-2 + offset]
                    battery = decodedData[-3 + offset] << 8 | decodedData[-4 + offset]
                    A6 = decodedData[-5 + offset] << 8 | decodedData[-6 + offset]
                    A5 = decodedData[-7 + offset] << 8 | decodedData[-8 + offset]
                    A4 = decodedData[-9 + offset] << 8 | decodedData[-10 + offset]
                    A3 = decodedData[-11 + offset] << 8 | decodedData[-12 + offset]
                    A2 = decodedData[-13 + offset] << 8 | decodedData[-14 + offset]
                    A1 = decodedData[-15 + offset] << 8 | decodedData[-16 + offset]
                    acquiredData = {}
                    acquiredData["analogChannels"] = [A1, A2, A3, A4, A5, A6]
                    acquiredData["battery"] = battery
                    acquiredData["batteryThreshold"] = batteryThreshold
                    acquiredData["digitalChannels"] = digitalPorts
                    return acquiredData
                else:
                    raise Exception(ExceptionCode.CONTACTING_DEVICE)
            else:
                raise Exception(ExceptionCode.DEVICE_NOT_IDLE)
        else:
            raise Exception(ExceptionCode.INVALID_VERSION)

    def trigger(self, digitalArray=None):
        """
        :param digitalArray: array which acts on digital outputs according to the value: 0 or 1
        :type digitalArray: array, tuple or list of int
        :raises Exception: list of digital channel output is not valid
        :raises Exception: device not in acquisition (IDLE) (for BITalino 1.0)

        Acts on digital output channels of the BITalino device. Triggering these digital outputs implies the use of the method :meth:`send`.
        Digital Outputs can be set on IDLE or while in acquisition for BITalino 2.0.

        Each position of the array *digitalArray* corresponds to a digital output, in ascending order. Possible values, types, configurations and examples for parameter *digitalArray*:

        ===============  ============================================== ==============================================
        Meta             BITalino 1.0                                   BITalino 2.0
        ===============  ============================================== ==============================================
        Values           0 or 1                                         0 or 1
        Types            list ``[]``, tuple ``()``, array ``[[]]``      list ``[]``, tuple ``()``, array ``[[]]``
        Configurations   4 values, one for each digital channel output  2 values, one for each digital channel output
        Examples         ``[1, 0, 1, 0]``                               ``[1, 0]``
        ===============  ============================================== ==============================================
        """
        arraySize = 2 if self.isBitalino2 else 4
        if not self.isBitalino2 and not self.started:
            raise Exception(ExceptionCode.DEVICE_NOT_IN_ACQUISITION)
        else:
            digitalArray = [0 for i in range(arraySize)] if digitalArray is None else digitalArray
            if isinstance(digitalArray, list):
                digitalArray = digitalArray
            elif isinstance(digitalArray, tuple):
                digitalArray = list(digitalArray)
            elif isinstance(digitalArray, numpy.ndarray):
                digitalArray = digitalArray.astype("int").tolist()
            else:
                raise Exception(ExceptionCode.INVALID_PARAMETER)

            pValues = [0, 1]
            if len(digitalArray) != arraySize or any(
                [item not in pValues or type(item) != int for item in digitalArray]
            ):
                raise Exception(ExceptionCode.INVALID_PARAMETER)

            if self.isBitalino2:
                # CommandDigital: 1  0  1  1  O2 O1 1  1 - Set digital outputs
                data = 179
            else:
                # CommandDigital: 1  0  O4  O3  O2 O1 1  1 - Set digital outputs
                data = 3

            for i, j in enumerate(digitalArray):
                data = data | j << (2 + i)
            self.send(data)

    def read(self, nSamples=100):
        """
        :param nSamples: number of samples to acquire
        :type nSamples: int
        :returns: array with the acquired data
        :raises Exception: device not in acquisition (in IDLE)
        :raises Exception: lost communication with the device when data is corrupted

        Acquires `nSamples` from BITalino. Reading samples from BITalino implies the use of the method :meth:`receive`.

        Requiring a low number of samples (e.g. ``nSamples = 1``) may be computationally expensive; it is recommended to acquire batches of samples (e.g. ``nSamples = 100``).

        The data acquired is organized in a matrix whose lines correspond to samples and the columns are as follows:

        * Sequence Number
        * 4 Digital Channels (always present)
        * 1-6 Analog Channels (as defined in the :meth:`start` method)

        Example matrix for ``analogChannels = [0, 1, 3]`` used in :meth:`start` method:

        ==================  ========= ========= ========= ========= ======== ======== ========
        Sequence Number*    Digital 0 Digital 1 Digital 2 Digital 3 Analog 0 Analog 1 Analog 3
        ==================  ========= ========= ========= ========= ======== ======== ========
        0
        1
        (...)
        15
        0
        1
        (...)
        ==================  ========= ========= ========= ========= ======== ======== ========

        .. note:: *The sequence number overflows at 15
        """
        if self.started:
            nChannels = len(self.analogChannels)

            if nChannels <= 4:
                number_bytes = int(math.ceil((12.0 + 10.0 * nChannels) / 8.0))
            else:
                number_bytes = int(math.ceil((52.0 + 6.0 * (nChannels - 4)) / 8.0))

            dataAcquired = numpy.zeros((nSamples, 5 + nChannels), dtype=int)
            for sample in range(nSamples):
                Data = self.receive(number_bytes)
                decodedData = list(struct.unpack(number_bytes * "B ", Data))
                crc = decodedData[-1] & 0x0F
                decodedData[-1] = decodedData[-1] & 0xF0
                x = 0
                for i in range(number_bytes):
                    for bit in range(7, -1, -1):
                        x = x << 1
                        if x & 0x10:
                            x = x ^ 0x03
                        x = x ^ ((decodedData[i] >> bit) & 0x01)
                if crc == x & 0x0F:
                    dataAcquired[sample, 0] = decodedData[-1] >> 4
                    dataAcquired[sample, 1] = decodedData[-2] >> 7 & 0x01
                    dataAcquired[sample, 2] = decodedData[-2] >> 6 & 0x01
                    dataAcquired[sample, 3] = decodedData[-2] >> 5 & 0x01
                    dataAcquired[sample, 4] = decodedData[-2] >> 4 & 0x01
                    if nChannels > 0:
                        dataAcquired[sample, 5] = ((decodedData[-2] & 0x0F) << 6) | (
                            decodedData[-3] >> 2
                        )
                    if nChannels > 1:
                        dataAcquired[sample, 6] = ((decodedData[-3] & 0x03) << 8) | decodedData[-4]
                    if nChannels > 2:
                        dataAcquired[sample, 7] = (decodedData[-5] << 2) | (decodedData[-6] >> 6)
                    if nChannels > 3:
                        dataAcquired[sample, 8] = ((decodedData[-6] & 0x3F) << 4) | (
                            decodedData[-7] >> 4
                        )
                    if nChannels > 4:
                        dataAcquired[sample, 9] = ((decodedData[-7] & 0x0F) << 2) | (
                            decodedData[-8] >> 6
                        )
                    if nChannels > 5:
                        dataAcquired[sample, 10] = decodedData[-8] & 0x3F
                else:
                    raise Exception(ExceptionCode.CONTACTING_DEVICE)
            return dataAcquired
        else:
            raise Exception(ExceptionCode.DEVICE_NOT_IN_ACQUISITION)

    def version(self):
        """
        :returns: str with the version of BITalino
        :raises Exception: device in acquisition (not IDLE)

        Retrieves the BITalino version. Retrieving the version implies the use of the methods :meth:`send` and :meth:`receive`.
        """
        if self.started is False:
            # CommandVersion: 0  0  0  0  0  1  1  1
            self.send(7)
            version_str = ""
            while True:
                if self.isPython2:
                    version_str += self.receive(1)
                else:
                    version_str += self.receive(1).decode("utf-8")
                if version_str[-1] == "\n" and "BITalino" in version_str:
                    break
            return version_str[version_str.index("BITalino") : -1]
        else:
            raise Exception(ExceptionCode.DEVICE_NOT_IDLE)

    def receive(self, nbytes):
        """
        :param nbytes: number of bytes to retrieve
        :type nbytes: int
        :return: string packed binary data
        :raises Exception: lost communication with the device when timeout is reached

        Retrieves `nbytes` from the BITalino device and returns it as a string pack with length of `nbytes`. The timeout is defined on instantiation.
        """
        if self.isPython2:
            data = ""
        else:
            data = b""
        if self.serial:
            while len(data) < nbytes:
                if not self.blocking:
                    initTime = time.time()
                    while self.socket.inWaiting() < 1:
                        finTime = time.time()
                        if (finTime - initTime) > self.timeout:
                            raise Exception(ExceptionCode.CONTACTING_DEVICE)
                data += self.socket.read(1)
        else:
            while len(data) < nbytes:
                if not self.blocking:
                    ready = select.select([self.socket], [], [], self.timeout)
                    if ready[0]:
                        pass
                    else:
                        raise Exception(ExceptionCode.CONTACTING_DEVICE)
                data += self.socket.recv(1)
        return data


if __name__ == "__main__":
    macAddress = "00:00:00:00:00:00"

    running_time = 5

    batteryThreshold = 30
    acqChannels = [0, 1, 2, 3, 4, 5]
    samplingRate = 1000
    nSamples = 10
    digitalOutput = [1, 1]

    # Connect to BITalino
    device = BITalino(macAddress)

    # Set battery threshold
    device.battery(batteryThreshold)

    # Read BITalino version
    print(device.version())

    # Start Acquisition
    device.start(samplingRate, acqChannels)

    start = time.time()
    end = time.time()
    while (end - start) < running_time:
        # Read samples
        print(device.read(nSamples))
        end = time.time()

    # Turn BITalino led on
    device.trigger(digitalOutput)

    # Stop acquisition
    device.stop()

    # Close connection
    device.close()
