# -*- coding: utf-8 -*-
"""
Minimal RINEX observation reader adapted for:
- RINEX 2.xx observation files (including files converted from CRINEX/Hatanaka)
- basic RINEX 3.xx observation files

Main goal:
- read observations epoch by epoch
- store them in convenient dictionaries / pandas DataFrames
- be simple enough to adapt for cycle slip detection workflows

Notes
-----
1) For RINEX 2, observation types are global in the header.
2) For RINEX 3, observation types are stored per constellation.
3) Carrier phases (L1, L2, etc.) are kept in their original units from file
   (typically cycles). No conversion to meters is performed here.
4) LLI and signal strength fields are not stored in this version.
"""

import datetime
import math
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


class rinexReader:
    def __init__(self, path=False):
        if path:
            self.path = path if isinstance(path, list) else [path]
        else:
            self.path = []

        self.period = [1, 0]
        self.obs: Dict[datetime.datetime, Dict[str, Dict[str, float]]] = {}
        self.obsSvid: Dict[str, Dict[datetime.datetime, Dict[str, float]]] = {}
        self.systems: List[str] = []
        self.upreftime = 0
        self.out = pd.DataFrame()
        self.f = False

        self.fileName = ""
        self.version = None
        self.fullversion = None
        self.fileStart = None
        self.fileEnd = None
        self.readConst = ["G"]
        self.comb = False
        self.oldEpoch = False
        self.timelist: List[datetime.datetime] = []

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def addFiles(self, path):
        """Add one or more RINEX file paths."""
        if not isinstance(path, list):
            path = [path]
        self.path.extend(path)

    def checkEpoch(self, epoch) -> bool:
        """Check whether epoch is inside requested interval."""
        if self.startTime <= epoch <= self.endTime:
            if self.oldEpoch:
                delta = (epoch - self.oldEpoch).total_seconds()
                if delta > self.period[0]:
                    print(
                        f'Gap in RINEX data from {self.fileName} @ '
                        f'{epoch.strftime("%H:%M:%S")}',
                        flush=True,
                    )
            return True
        return False

    def openRnxFile(self, path):
        """Open RINEX file and initialize header reading."""
        try:
            f = open(path, "r", encoding="utf-8", errors="ignore")
        except Exception:
            print("Path not found:", path)
            sys.exit()

        self.fileName = path.split("/")[-1].split("\\")[-1]

        line = f.readline()
        if "RINEX VERSION / TYPE" not in line:
            print("File error: No RINEX version detected in line 1")
            sys.exit()

        self.fullversion = float(line[:10])
        self.version = int(line[:10].strip()[0])

        if self.version == 3:
            self.readRnx3Header(f)
        elif self.version == 2:
            self.readRnx2Header(f)
        else:
            print("Error: Only RINEX version 2 or 3 is supported.")
            sys.exit()

        return f

    def readFile(
        self,
        readConst=["G"],
        sigTypes=["L1", "L2", "C1", "P2"],
        startTime=False,
        endTime=False,
    ):
        """
        Generic entry point to read a RINEX observation file.

        Parameters
        ----------
        readConst : list[str]
            Constellations to read, e.g. ["G"], ["G", "R"].
        sigTypes : list[str]
            Observation types to keep. For RINEX 2 examples: ["L1","L2","C1","P2"]
            For RINEX 3 examples: ["L1C","L2W","C1C"].
            Use [""] to keep all.
        startTime, endTime : datetime.datetime or False
            Optional time interval.
        """
        self.readConst = readConst if isinstance(readConst, list) else [readConst]
        self.comb = False if sigTypes == [""] else sigTypes

        if not self.path:
            print("No filepaths added...")
            return

        self.obs = {}
        self.obsSvid = {}
        self.timelist = []
        self.systems = []
        self.oldEpoch = False

        self.f = self.openRnxFile(self.path[0])

        self.startTime = startTime if startTime else self.fileStart
        self.endTime = endTime if endTime else self.fileEnd

        if self.f:
            if self.version == 3:
                self.readRnx3File(self.f)
            elif self.version == 2:
                self.readRnx2File(self.f)
            else:
                print("Only RINEX version 2 or 3 is supported!")

            try:
                self.f.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # RINEX 2
    # ------------------------------------------------------------------
    def readRnx2Header(self, f):
        """Read RINEX 2 observation header."""
        self.obsTypes = {}
        self.obsIdx = {}
        self.obsUse = {}
        self.obsMult = {}

        all_obs_types = []
        marker_name = None

        for line in f:
            if "END OF HEADER" in line:
                break

            elif "# / TYPES OF OBSERV" in line:
                nObs = int(line[0:6])
                all_obs_types.extend(line[6:60].split())

                while len(all_obs_types) < nObs:
                    line = f.readline()
                    all_obs_types.extend(line[6:60].split())

            elif "TIME OF FIRST OBS" in line:
                inp = line[:43].split()
                year = int(inp[0])
                month = int(inp[1])
                day = int(inp[2])
                hour = int(inp[3])
                minute = int(inp[4])
                sec = float(inp[5])
                sec_int = int(sec)
                micro = int(round((sec - sec_int) * 1e6))
                self.fileStart = datetime.datetime(year, month, day, hour, minute, sec_int, micro)

            elif "TIME OF LAST OBS" in line:
                inp = line[:43].split()
                year = int(inp[0])
                month = int(inp[1])
                day = int(inp[2])
                hour = int(inp[3])
                minute = int(inp[4])
                sec = float(inp[5])
                sec_int = int(sec)
                micro = int(round((sec - sec_int) * 1e6))
                self.fileEnd = datetime.datetime(year, month, day, hour, minute, sec_int, micro)

            elif "INTERVAL" in line:
                try:
                    self.period = [int(float(line[:10].strip())), 0]
                except Exception:
                    self.period = [1, 0]

            elif "APPROX POSITION XYZ" in line:
                try:
                    self.approxPos = np.array([float(x) for x in line[:60].split()])
                except Exception:
                    pass

            elif "ANTENNA: DELTA H/E/N" in line:
                try:
                    self.deltaHen = np.array([float(x) for x in line[:60].split()])
                except Exception:
                    pass

            elif "MARKER NAME" in line:
                marker_name = line[:60].strip()

            elif "ANT # / TYPE" in line:
                antType = line[:60].split()
                if len(antType) > 0:
                    self.antSerial = antType[0]
                if len(antType) > 1:
                    self.antType = antType[1]
                if len(antType) > 2:
                    self.antRadome = antType[2]

        if marker_name is not None:
            self.markerName = marker_name

        if self.fileStart is None:
            raise ValueError("TIME OF FIRST OBS not found in RINEX 2 header.")
        if self.fileEnd is None:
            self.fileEnd = self.fileStart + datetime.timedelta(days=1)

        if not self.readConst:
            self.readConst = ["G"]

        for const in self.readConst:
            if const not in self.systems:
                self.systems.append(const)
            self.obsTypes[const] = all_obs_types.copy()

            if self.comb:
                self.obsUse[const] = []
                self.obsIdx[const] = []
                self.obsMult[const] = []
                for idx, obscode in enumerate(self.obsTypes[const]):
                    if obscode in self.comb:
                        self.obsIdx[const].append(idx)
                        self.obsUse[const].append(obscode)
                        self.obsMult[const].append(1.0)
            else:
                self.obsUse[const] = self.obsTypes[const].copy()
                self.obsIdx[const] = list(range(len(self.obsTypes[const])))
                self.obsMult[const] = [1.0] * len(self.obsTypes[const])

            self.obsMult[const] = np.array(self.obsMult[const], dtype=float)

            

    def _rnx2_to_datetime(self, yy, mm, dd, hh, mi, sec):
        yy = int(yy)
        if yy >= 80:
            year = 1900 + yy
        else:
            year = 2000 + yy

        sec_float = float(sec)
        sec_int = int(sec_float)
        micro = int(round((sec_float - sec_int) * 1e6))
        if micro >= 1000000:
            sec_int += 1
            micro -= 1000000

        return datetime.datetime(year, int(mm), int(dd), int(hh), int(mi), sec_int, micro)

    def readRnx2EpochHeader(self, first_line, f):
        """Parse one RINEX 2 epoch header line."""
        yy = first_line[0:3].strip()
        mm = first_line[3:6].strip()
        dd = first_line[6:9].strip()
        hh = first_line[9:12].strip()
        mi = first_line[12:15].strip()
        sec = first_line[15:26].strip()
        flag = int(first_line[28:29].strip() or 0)
        nSat = int(first_line[29:32].strip() or 0)

        epoch = self._rnx2_to_datetime(yy, mm, dd, hh, mi, sec)

        sat_list = []
        sat_field = first_line[32:68]
        for i in range(0, len(sat_field), 3):
            s = sat_field[i:i + 3].strip()
            if s:
                sat_list.append(s)

        while len(sat_list) < nSat:
            cont = f.readline()
            cont_field = cont[32:68] if len(cont) >= 68 else cont
            for i in range(0, len(cont_field), 3):
                s = cont_field[i:i + 3].strip()
                if s:
                    sat_list.append(s)

        sat_list = sat_list[:nSat]

        fixed_sat_list = []
        for s in sat_list:
            if len(s) == 2 and s[0].isdigit():
                fixed_sat_list.append("G" + s)
            else:
                fixed_sat_list.append(s)

        return epoch, flag, nSat, fixed_sat_list

    def readRnx2ObsBlock(self, f, nObs):
        """Read one satellite observation block in RINEX 2."""
        nLines = int(math.ceil(nObs / 5.0))
        raw = ""
        for _ in range(nLines):
            line = f.readline()
            raw += line.rstrip("\n").ljust(80)

        vals = []
        for i in range(nObs):
            field = raw[i * 16:(i + 1) * 16]
            val_str = field[:14].strip()  # last two chars are LLI / signal strength
            if val_str == "":
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(val_str.replace("D", "E")))
                except Exception:
                    vals.append(np.nan)
        return vals

    def readRnx2File(self, f):
        """Read RINEX 2 observation file."""
        print(f"Reading RINEX{self.version} observations from {self.fileName}")
        self.readError = False

        if len(self.readConst) == 0:
            raise ValueError("readConst is empty")

        template_const = self.readConst[0]
        nObsTotal = len(self.obsTypes[template_const])

        while True:
            line = f.readline()
            if not line:
                break
            if line.strip() == "":
                continue

            try:
                epoch, flag, nSat, sat_list = self.readRnx2EpochHeader(line, f)
            except Exception:
                continue

            if flag not in [0, 1]:
                # event record: skip listed sat blocks to keep pointer aligned
                for _ in range(nSat):
                    _ = self.readRnx2ObsBlock(f, nObsTotal)
                continue

            if self.checkEpoch(epoch):
                self.timelist.append(epoch)
                if epoch not in self.obs:
                    self.obs[epoch] = {}

                for svid in sat_list:
                    const = svid[0]
                    obs_values = self.readRnx2ObsBlock(f, nObsTotal)

                    if const not in self.readConst:
                        continue

                    if svid not in self.obsSvid:
                        self.obsSvid[svid] = {}
                    self.obsSvid[svid][epoch] = {}

                    selected_vals = []
                    for idx in self.obsIdx[const]:
                        selected_vals.append(obs_values[idx] if idx < len(obs_values) else np.nan)

                    obs_dict = {obs_code: val for obs_code, val in zip(self.obsUse[const], selected_vals)}
                    self.obs[epoch][svid] = obs_dict
                    self.obsSvid[svid][epoch] = obs_dict.copy()

                self.oldEpoch = epoch
            else:
                for _ in sat_list:
                    _ = self.readRnx2ObsBlock(f, nObsTotal)

        f.close()

    # ------------------------------------------------------------------
    # RINEX 3 (basic support)
    # ------------------------------------------------------------------
    def readRnx3Header(self, f):
        """Read basic RINEX 3 observation header."""
        self.obsTypes = {}
        self.obsIdx = {}
        self.obsUse = {}
        self.obsMult = {}

        for line in f:
            if "END OF HEADER" in line:
                break

            elif 'SYS / # / OBS TYPES' in line[60:]:
                const = line[0]
                nObs = int(line[3:6])
                if const not in self.systems:
                    self.systems.append(const)
                self.obsTypes[const] = line[6:60].split()

                n = nObs - len(self.obsTypes[const])
                while n > 0:
                    line = f.readline()
                    self.obsTypes[const] += line[6:60].split()
                    n = nObs - len(self.obsTypes[const])

                if const in self.readConst:
                    if self.comb:
                        self.obsUse[const] = []
                        self.obsIdx[const] = []
                        self.obsMult[const] = []
                        for idx, obscode in enumerate(self.obsTypes[const]):
                            if obscode in self.comb:
                                self.obsIdx[const].append(idx)
                                self.obsUse[const].append(obscode)
                                self.obsMult[const].append(1.0)
                    else:
                        self.obsUse[const] = self.obsTypes[const].copy()
                        self.obsIdx[const] = list(range(len(self.obsTypes[const])))
                        self.obsMult[const] = [1.0] * len(self.obsTypes[const])

                    self.obsMult[const] = np.array(self.obsMult[const], dtype=float)

            elif 'TIME OF FIRST OBS' in line[60:]:
                inp = line[:60].split()
                self.fileStart = datetime.datetime(
                    int(inp[0]), int(inp[1]), int(inp[2]),
                    int(inp[3]), int(inp[4]), int(float(inp[5]))
                )

            elif 'TIME OF LAST OBS' in line[60:]:
                inp = line[:60].split()
                self.fileEnd = datetime.datetime(
                    int(inp[0]), int(inp[1]), int(inp[2]),
                    int(inp[3]), int(inp[4]), int(float(inp[5]))
                )

            elif 'INTERVAL' in line[60:]:
                try:
                    self.period = [int(float(line[:60].strip().split()[0])), 0]
                except Exception:
                    self.period = [1, 0]

            elif 'APPROX POSITION XYZ' in line[60:]:
                try:
                    self.approxPos = np.array([float(x) for x in line[:60].split()])
                except Exception:
                    pass

            elif 'ANTENNA: DELTA H/E/N' in line[60:]:
                try:
                    self.deltaHen = np.array([float(x) for x in line[:60].split()])
                except Exception:
                    pass

            elif 'ANT # / TYPE' in line[60:]:
                antType = [x for x in line[:60].split()]
                try:
                    self.antSerial = antType[0]
                    self.antType = antType[1]
                    if len(antType) > 2:
                        self.antRadome = antType[2]
                except Exception:
                    pass

        if self.fileStart is None:
            raise ValueError("TIME OF FIRST OBS not found in RINEX 3 header.")
        if self.fileEnd is None:
            self.fileEnd = self.fileStart

    def readRnx3File(self, f):
        """Read basic RINEX 3 observation file."""
        print(f'Reading RINEX{self.version} observations from {self.fileName}')
        self.readError = False

        for line in f:
            if not line:
                break
            if line[0] != '>':
                continue

            h = line[1:].split()
            year, month, day, hour, minute = map(int, h[0:5])
            sec_float = float(h[5])
            sec_int = int(sec_float)
            micro = int(round((sec_float - sec_int) * 1e6))
            epoch = datetime.datetime(year, month, day, hour, minute, sec_int, micro)
            flag = int(h[6])
            nSat = int(h[7])

            if flag not in [0, 1]:
                for _ in range(nSat):
                    _ = f.readline()
                continue

            if self.checkEpoch(epoch):
                self.timelist.append(epoch)
                if epoch not in self.obs:
                    self.obs[epoch] = {}

                for _ in range(nSat):
                    line = f.readline()
                    if not line:
                        break
                    svid = line[:3]
                    const = svid[0]

                    if const not in self.readConst:
                        continue

                    if svid not in self.obsSvid:
                        self.obsSvid[svid] = {}
                    self.obsSvid[svid][epoch] = {}

                    data = []
                    ll = len(self.obsTypes[const]) * 16 + 3
                    for idx, i in enumerate(range(3, ll, 16)):
                        if idx in self.obsIdx[const]:
                            try:
                                data.append(float(line[i:i + 16].split()[0]))
                            except Exception:
                                data.append(np.nan)

                    obs_dict = {z: y for z, y in zip(self.obsUse[const], np.array(data))}
                    self.obs[epoch][svid] = obs_dict
                    self.obsSvid[svid][epoch] = obs_dict.copy()

                self.oldEpoch = epoch
            else:
                for _ in range(nSat):
                    _ = f.readline()

        f.close()

    # ------------------------------------------------------------------
    # Data extraction helpers
    # ------------------------------------------------------------------
    def get_epoch_data(self, reftime, consts=['G'], oTypes=['C1'], svidx=[]):
        """Return data frame for one epoch."""
        if not (self.startTime <= reftime <= self.endTime):
            print(f"Given time is not within RINEX file from {self.startTime} to {self.endTime}")
            return pd.DataFrame()

        if reftime not in self.obs:
            return pd.DataFrame()

        obs = pd.DataFrame.from_dict(self.obs[reftime], orient='index')
        obs = obs.iloc[[num for num, idx in enumerate(obs.index) if idx[0] in consts], :]
        obs = obs.iloc[:, [num for num, col in enumerate(obs.columns) if col in oTypes]]

        if any(svidx):
            obs = obs.iloc[[num for num, idx in enumerate(obs.index) if idx in svidx], :]

        if len(obs) > 0:
            testnans = np.sum(np.isnan(obs.values), axis=0) == len(obs)
            if np.any(testnans):
                obs = obs.loc[:, ~(testnans)]
            obs.sort_index(axis=1, inplace=True)
            obs.sort_index(inplace=True)

        return obs

    def get_obs_data(self, oTypes='C1', svidx=[]):
        """Return a time-indexed DataFrame for all satellites for selected obs types."""
        obsDict = {}

        if np.any(svidx):
            for sv in self.obsSvid:
                if sv in svidx:
                    try:
                        obs = pd.DataFrame.from_dict(self.obsSvid[sv], orient='index')
                        obsDict[sv] = obs.loc[:, oTypes]
                    except Exception:
                        pass
        else:
            for sv in self.obsSvid:
                try:
                    obs = pd.DataFrame.from_dict(self.obsSvid[sv], orient='index')
                    obsDict[sv] = obs.loc[:, oTypes]
                except Exception:
                    pass

        if len(obsDict) == 0:
            return pd.DataFrame()

        obs = pd.DataFrame.from_dict(obsDict, orient='index').T
        obs.sort_index(axis=1, inplace=True)
        return obs

    def get_svid_data(self, svid, oTypes=[]):
        """Return DataFrame for one satellite."""
        if svid not in self.obsSvid:
            print("Given satellite was not found in RINEX data")
            return pd.DataFrame()

        obs = pd.DataFrame.from_dict(self.obsSvid[svid], orient='index')

        if any(oTypes):
            obs = obs.iloc[:, [num for num, col in enumerate(obs.columns) if col in oTypes]]

        if len(obs) > 0:
            testnans = np.sum(np.isnan(obs.values), axis=0) == len(obs)
            if np.any(testnans):
                obs = obs.loc[:, ~(testnans)]
            obs.sort_index(axis=1, inplace=True)
            obs.sort_index(inplace=True)

        return obs


if __name__ == "__main__":
    # Example usage for your converted file:
     rnx = rinexReader("data/nuuk1320.24o")
     rnx.readFile(readConst=["G"], sigTypes=["L1", "L2", "C1", "P2"])
     print("Start:", rnx.fileStart)
     print("End:", rnx.fileEnd)
     print("Epochs:", len(rnx.obs))
     print("Satellites:", len(rnx.obsSvid))
     print(rnx.get_svid_data("G05", ["L1", "L2"]).head())

