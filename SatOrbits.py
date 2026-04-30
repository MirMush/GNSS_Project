# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 22:07:18 2026

@author: sorla
"""

import numpy as np
import pandas as pd
import datetime as dt
import scipy.interpolate as interp

clight=2.99792458e8 #speed of light (meters/sec)

ohmE = 7.2921151467e-5; # WGS84 earth rotation rate, rad/s

kepler_const = {'G':[3.986005e14, 7.2921151467e-5],
                'R':[3.9860044e14, 7.292115e-5],
                'E':[3.986004418e14, 7.2921151467e-5],
                'C':[3.986004418e14, 7.292115e-5],
                'J':[3.986005e14, 7.2921151467e-5]}

class sp3Orbits:
    
    def __init__(self, paths=[]):
        # NOTE THAT THE CURRENT SCRIPT ONLY ALLOWS FOR READING OF ONE SP3 FILE!
        
        self.satpos = pd.DataFrame()
        self.sp3data = {} # pd.DataFrame()
        self.nSats = 0
        self.prns = []
        self.broadcast = False
        if not isinstance(paths, list):
            self.paths = [paths]
        else:
            self.paths = paths
        self.ohmE = 7.2921151467e-5; # WGS84 earth rotation rate, rad/s
        
        if self.paths:
            for path in self.paths:
                self.readSp3(path)
                self.makeInterpolator()
        
    def readSp3Header(self, file):
        # WHAT MUST BE READ FROM THE HEADER?
        self.prns = []
        
        for idx, line in enumerate(file):
            data = line.split()
            # print(line)
            # Read nSats and sat PRNS
            if line[0] == '#':
                if idx == 0:
                    self.filestart = dt.datetime(int(line[3:7]), int(line[8:10]), int(line[11:13]), int(line[14:16]), int(float(line[20:31])))
                    self.noepochs = int(line[32:39])
                    self.refsys = line[46:51]
                if idx ==1:
                    self.epochint = int(float(data[3]))
            
            if data[0] == '+':
                if not self.nSats:
                    self.nSats = int(data[1])
                
                self.prns.extend([line[i:i+3] for i in range(9,60,3)])
            
            self.prns = self.prns[:self.nSats]
            
            # Definition of end of header??
            if idx == 21:
                # Do something
                break
        
        self.fileend = self.filestart + dt.timedelta(seconds = (self.noepochs-1)*self.epochint)
        self.totalsec = self.epochint*self.noepochs
        
        for sv in self.prns:
            self.sp3data[sv] = {}
    
    def readSp3(self, path):
        # Note that only one sp3 file can be read currently!
        
        if path not in self.paths:
            self.paths.append(path)
        
        with open(path) as file:
            
            filename = path.split("/")
            print(f"Reading orbit parameters from: {filename[-1]}")
            self.readSp3Header(file)
            
            for line in file:
                
                data = line.split()
                # print(data)
                
                if data[0] == "*":
                    epoch = dt.datetime(int(data[1]), int(data[2]), int(data[3]), int(data[4]), int(data[5]), int(float(data[6])))
                        
                    for satno in range(self.nSats):
                        line = file.readline()
                        data = line.split()
                        satprn = data[0][1:]
                        self.sp3data[satprn][epoch] = [np.round(float(x),6)*1e3 for x in data[1:5]]
                        self.sp3data[satprn][epoch][3] *= 1e-9
                            
                elif data[0] == "EOF":
                    break
        
        for key in self.sp3data:
            self.sp3data[key] = pd.DataFrame.from_dict(self.sp3data[key])
    
    def makeInterpolator(self):
        self.interpolator = {}
        for key in self.sp3data:
            self.interpolator[key] = interp.CubicSpline(np.arange(0,self.totalsec,self.epochint), self.sp3data[key], axis=1)
    
    def getSvPos(self, refepoch, tau=pd.Series(), const=['G']):
        """
        Get data from self.sp3data and interpolate it somehow (Spline? Quadratic?)
        tau should be a dataseries
        refepoch should be a single datetime value
        tau is a pandas dataseries that must contain traveltimes, with 
        satellite PRNS as indexes, e.g. G01, G10, E14, etc.
        if usetau is set to False, then tau must be a dataseries containing the 
        PRNS to be found
        """
        
        satpos = {}
        
        if any(tau):
            # Remove all nans from tau, if it hasn't already happened
            # tau = tau[~pd.isna(tau)]
            # tau = tau[tau != 0]
            
            rot = -self.ohmE*tau
            
            deltasec = (refepoch - self.filestart).total_seconds()
            tidx = deltasec - tau
            
            for key in tau.index:
                # Define rotational matrix
                try:
                    rotmat = np.array([[np.cos(rot[key]), -np.sin(rot[key]), 0, 0],
                                       [np.sin(rot[key]), np.cos(rot[key]), 0, 0],
                                       [0, 0, 1, 0], 
                                       [0, 0, 0, 1]])

                    # Find position and rotate to account for rotation of the Earth
                    spdata = rotmat @ self.interpolator[key](tidx[key])

                    if spdata[3] < 0.99:
                        satpos[key] = spdata
                    
                except:
                    # print(f"{key} not found in SP3 file")
                    pass
        else:
            deltasec = (refepoch - self.filestart).total_seconds()
            for key in self.interpolator:
                if key[0] in const:
                    satpos[key] = self.interpolator[key](deltasec)
    
        satpos = pd.DataFrame.from_dict(satpos, orient='index', columns=['X','Y','Z','dt'])
        
        return satpos
    
#%% Broadcast orbits

# NOTE THAT CONSTELLATIONS C, R, J, S and I are not fully implemented
# BeiDou GEO satellites: C01-05, C59-62 
# TODO: Account for correct BeiDou and GLONASS time to get correct orbits
# TODO: Implement BeiDou GEO satellites
# TODO: Remember different omega_e and mu for BeiDou and GLONASS

_ephindx3 =  { 'G': ['epoch', 'clk bias', 'clk drift', 'clk drift rate',
                    'IODE', 'crs', 'Delta n', 'M0',
                    'cuc', 'e', 'cus', 'sqrtA',
                    'toe', 'cic', 'omega0', 'cis', 
                    'i0', 'crc', 'omega', 'omega dot',
                    'IDOT', 'codes on L2', 'week #', 'L2P flag',
                    'sv acc', 'sv health', 'tgd', 'IODC', 
                    'ttrx', 'fit int', 'spare', 'spare'],
             'E': ['epoch', 'clk bias', 'clk drift', 'clk drift rate',
                   'IODE', 'crs', 'Delta n', 'M0',
                   'cuc', 'e', 'cus', 'sqrtA',
                   'toe', 'cic', 'omega0', 'cis', 
                   'i0', 'crc', 'omega', 'omega dot',
                   'IDOT', 'Data sources', 'week #', 'spare',
                   'sv acc', 'sv health', 'BGD E5a/E1', 'BGD E5b/E1', 
                   'ttrx', 'fit int', 'spare', 'spare'],
             'R': ['epoch', 'clk bias', 'sv freq bias', 'toe',
                   'Sat x', 'Vel x', 'Accel x', 'health',
                   'Sat y', 'Vel y', 'Accel y', 'freq #',
                   'Sat z', 'Vel z', 'Accel z', 'age of oper',
                   'Status flag', 'L1/L2 dgroup delay', 'URAI', 'Health flag'],
                   # 'Status flag', 'L1/L2 gd', 'URAI', 'Health flags'],
             'C': ['epoch', 'clk bias', 'clk drift', 'clk drift rate',
                   'AODE', 'crs', 'Delta n', 'M0',
                   'cuc', 'e', 'cus', 'sqrtA',
                   'toe', 'cic', 'omega0', 'cis', 
                   'i0', 'crc', 'omega', 'omega dot',
                   'IDOT', 'spare', 'week #', 'spare',
                   'sv acc', 'sv health', 'TGD1 B1/B3', 'TGD1 B2/B3', 
                   'ttrx', 'AODC', 'spare', 'spare'],
             'J': ['epoch', 'clk bias', 'clk drift', 'clk drift rate',
                    'IODE', 'crs', 'Delta n', 'M0',
                    'cuc', 'e', 'cus', 'sqrtA',
                    'toe', 'cic', 'omega0', 'cis', 
                    'i0', 'crc', 'omega', 'omega dot',
                    'IDOT', 'codes on L2', 'week #', 'L2P flag',
                    'sv acc', 'sv health', 'tgd', 'IODC', 
                    'ttrx', 'fit int', 'spare', 'spare'],
             'S': ['epoch', 'clk bias', 'sv freq bias', 'toe',
                   'Sat x', 'Vel x', 'Accel x', 'health',
                   'Sat y', 'Vel y', 'Accel y', 'freq #',
                   'Sat z', 'Vel z', 'Accel z', 'age of oper'],
             'I': ['epoch', 'clk bias', 'clk drift', 'clk drift rate',
                    'IODE', 'crs', 'Delta n', 'M0',
                    'cuc', 'e', 'cus', 'sqrtA',
                    'toe', 'cic', 'omega0', 'cis', 
                    'i0', 'crc', 'omega', 'omega dot',
                    'IDOT', 'Blank', 'IRN week #', 'Blank',
                    'URA', 'sv health', 'tgd', 'Blank', 
                    'ttrx', 'Blank', 'Blank', 'Blank'],
             }

_ephindx305 =  { 'G': ['epoch', 'clk bias', 'clk drift', 'clk drift rate',
                    'IODE', 'crs', 'Delta n', 'M0',
                    'cuc', 'e', 'cus', 'sqrtA',
                    'toe', 'cic', 'omega0', 'cis', 
                    'i0', 'crc', 'omega', 'omega dot',
                    'IDOT', 'codes on L2', 'week #', 'L2P flag',
                    'sv acc', 'sv health', 'tgd', 'IODC', 
                    'ttrx', 'fit int', 'spare', 'spare'],
             'E': ['epoch', 'clk bias', 'clk drift', 'clk drift rate',
                   'IODE', 'crs', 'Delta n', 'M0',
                   'cuc', 'e', 'cus', 'sqrtA',
                   'toe', 'cic', 'omega0', 'cis', 
                   'i0', 'crc', 'omega', 'omega dot',
                   'IDOT', 'Data sources', 'week #', 'spare',
                   'sv acc', 'sv health', 'BGD E5a/E1', 'BGD E5b/E1', 
                   'ttrx', 'fit int', 'spare', 'spare'],
             'R': ['epoch', 'clk bias', 'sv freq bias', 'toe',
                   'Sat x', 'Vel x', 'Accel x', 'health',
                   'Sat y', 'Vel y', 'Accel y', 'freq #',
                   'Sat z', 'Vel z', 'Accel z', 'age of oper',
                   'Status flag', 'L1/L2 dgroup delay', 'URAI', 'Health flag'],
                   # 'Status flag', 'L1/L2 gd', 'URAI', 'Health flags'],
             'C': ['epoch', 'clk bias', 'clk drift', 'clk drift rate',
                   'AODE', 'crs', 'Delta n', 'M0',
                   'cuc', 'e', 'cus', 'sqrtA',
                   'toe', 'cic', 'omega0', 'cis', 
                   'i0', 'crc', 'omega', 'omega dot',
                   'IDOT', 'spare', 'week #', 'spare',
                   'sv acc', 'sv health', 'TGD1 B1/B3', 'TGD1 B2/B3', 
                   'ttrx', 'AODC', 'spare', 'spare'],
             'J': ['epoch', 'clk bias', 'clk drift', 'clk drift rate',
                    'IODE', 'crs', 'Delta n', 'M0',
                    'cuc', 'e', 'cus', 'sqrtA',
                    'toe', 'cic', 'omega0', 'cis', 
                    'i0', 'crc', 'omega', 'omega dot',
                    'IDOT', 'codes on L2', 'week #', 'L2P flag',
                    'sv acc', 'sv health', 'tgd', 'IODC', 
                    'ttrx', 'fit int', 'spare', 'spare'],
             'S': ['epoch', 'clk bias', 'sv freq bias', 'toe',
                   'Sat x', 'Vel x', 'Accel x', 'health',
                   'Sat y', 'Vel y', 'Accel y', 'freq #',
                   'Sat z', 'Vel z', 'Accel z', 'age of oper'],
             'I': ['epoch', 'clk bias', 'clk drift', 'clk drift rate',
                    'IODE', 'crs', 'Delta n', 'M0',
                    'cuc', 'e', 'cus', 'sqrtA',
                    'toe', 'cic', 'omega0', 'cis', 
                    'i0', 'crc', 'omega', 'omega dot',
                    'IDOT', 'Blank', 'IRN week #', 'Blank',
                    'URA', 'sv health', 'tgd', 'Blank', 
                    'ttrx', 'Blank', 'Blank', 'Blank'],
             }


class NavReader(object):
    
    def __init__(self, filepath, systems=['G','R','E','C']):
        self.f = open(filepath)
        self.filepath = filepath
        self.ephdict = {}
        self.systems = systems
        
        self.ohmE = 7.2921151467e-5; # WGS84 earth rotation rate, rad/s
        
        # How often ephemeris is updated (at least on Spetentrio receivers)
        self.updates = {}
        self.updates['G'] = 7200
        self.updates['R'] = 900
        self.updates['E'] = 600
        self.updates['C'] = 3600
        
        self.readheader()
        self.readNavData()
        self.curEpoch = dt.datetime(1970, 1, 1, 0, 0, 0)
        self.toeAgeDict = {}
        self.eph = {}
        
        self.rottime = 0
        self.calctime = 0
        self.ektime = 0
        self.initTime = 0
        
        self.broadcast = True
        
        # self.resetPos()
        
        self.geosv = ['C'+str(i+1).zfill(2) for i in range(5)]
        self.geosv.extend(['C'+str(i) for i in range(59,63)])
        
        self.nongeosv = ['C'+str(i).zfill(2) for i in range(6,59)]
        

    def readheader(self):
        
        self.header = {}
        
        # TODO: Read ionospheric parameters
        # TODO: Read leap seconds
        
        for line in self.f:
            if "RINEX VERSION / TYPE" in line:
                self.version = float(line.split()[0])
                if self.version < 3 or self.version >= 4:
                    print("Only RINEX3 navigation files are permitted")
                elif self.version >= 3.05:
                    self.ephidx = _ephindx305
                else:
                    self.ephidx = _ephindx3
            if "END OF HEADER" in line:
                break
    
    def readNavData(self):
        
        print(f"Reading broadcast navigation data from {self.filepath.split('/')[-1]}")
        
        data = []
        const='G'
        svid='G01'
        tid = dt.datetime.now()
        strtid = tid.strftime("%m/%d/%Y, %H:%M:%S")
        
        for line in self.f:
            
            if line[0] != " ":
                
                if any(data):
                    self.ephdict[const][svid+" "+strtid] = pd.Series(data, index = self.ephidx[const])
                
                const = line[0]
                svid = line[:3]
                # data = []
                
                if const not in self.ephdict:
                    self.ephdict[const] = {}
                
                tc = [int(t) for t in line[4:23].split()]
                tid = dt.datetime(tc[0], tc[1], tc[2], tc[3], tc[4], tc[5])
                strtid = tid.strftime("%m/%d/%Y, %H:%M:%S")
                data = [tid]
                
                # print(const, svid, strtid)
                
                for i in range(23,80,19):
                    try:
                        data.extend([float(line[i:i+19])])
                    except:
                        data.extend([0.0])
                
            else:   
                for i in range(4,80,19):
                    try:
                        data.extend([float(line[i:i+19])])
                    except:
                        data.extend([0.0])

        self.ephdict[const][svid+" "+strtid] = pd.Series(data, index = self.ephidx[const])
        
        # self.eph = {}
        for const in self.ephdict:
            self.ephdict[const] = pd.DataFrame.from_dict(self.ephdict[const], orient="index")
        

    def getSvPos(self,epoch,tauin=[],ephdict={},constlist=['G'],max_age=7200):
        """
        Determine and return GNSS satellite positions (from broadcast eph). \n
        
        GLONASS ORBITS NOT IMPLEMENTED
        BeiDou Geostationary orbits not considered properly yet

        """
        
        satpos = []
        # if not constlist:
        #     constlist = set([c[0] for c in tauin.index])
        
        if not ephdict:
            ephdict = self.ephdict
        
        for const in constlist:
            
            if epoch!=self.curEpoch:

                e0 = dt.datetime(epoch.year, epoch.month, epoch.day, 0, 0, 0)
                e0 = e0 + dt.timedelta(days = -e0.weekday()-1)
                
                if const == 'R':
                    print('GLONASS NOT IMPLEMENTED YET!')
                    return 0
    
                secofweek = (epoch - e0).total_seconds()
                toeageSec = secofweek - ephdict[const].loc[:,'toe']
                
                # max_age = 7200 Per constellation??
                # # How often ephemeris is updated (at least on Spetentrio receivers)
                # self.updates = {}
                # self.updates['G'] = 7200
                # self.updates['R'] = 900
                # self.updates['E'] = 600
                # self.updates['C'] = 3600
                
                #Check for week changeover
                mask1 = toeageSec > 302400
                mask2 = toeageSec < -302400
                toeageSec[mask1] -= 604800
                toeageSec[mask2] += 604800
                
                toeageSec = toeageSec[toeageSec < max_age]
                toeageSec = toeageSec[toeageSec > -max_age]
                
                # Check for (and remove) duplicate satellites:
                toeageSec = toeageSec.sort_values()
                sorttoeD = toeageSec.copy()
                sorttoeD.index = [s[:3] for s in sorttoeD.index]
        
                svidxKeep = sorttoeD.index.duplicated(keep='first')
                toeageSec = toeageSec[~svidxKeep]
                toeageSec = toeageSec.sort_index()
                
                svidx = toeageSec.index
                self.eph[const] = ephdict[const].loc[svidx,:]
                self.eph[const].index = [s[:3] for s in self.eph[const].index]
                toeageSec.index = [s[:3] for s in toeageSec.index]
                
                self.toeAgeDict[const] = toeageSec
            
            
            if any(tauin):
                tau = tauin.loc[tauin.index.str.startswith(const)]
                
                # Make sure we have both observations and satellite positions for a SV            
                toeageSec = self.toeAgeDict[const] - tau
                toeageSec = toeageSec[~toeageSec.isna()]
                eph = self.eph[const].loc[toeageSec.index, :]
                tau = tau.loc[toeageSec.index]
            else:
                toeageSec = self.toeAgeDict[const]
                eph = self.eph[const].loc[toeageSec.index, :]
                tau = []
            
            """Orbit calculations start here!"""
            A = eph.loc[:,'sqrtA']**2 #semi-major axis (A) [meters]
            n0=np.sqrt(kepler_const[const][0]/(A**3)) #Mean motion [rad/sec]
            
            Mk = eph.loc[:,'M0'] + (n0+eph.loc[:,'Delta n'])*toeageSec #Mean anomaly [rad]
            
            #Initialize E_k computations
            Ek=Mk; #Initial guess of Eccentric Anomaly
            Ekold=Ek+1
        
            #Iterative computations to find Eccentric Anomaly [rad]
            while (np.abs(Ek-Ekold) > 1e-12).any():
                Ekold=Ek
                Ek=Mk+eph.loc[:,'e']*np.sin(Ek)
            
            #Relativistic clock correction
            F = 2*np.sqrt(kepler_const[const][0])/(clight**2)
            dtr=F*eph.loc[:,'e']*eph.loc[:,'sqrtA']*np.sin(Ek) 
            # print(dtr)
            
            #True Anomaly [rad]
            vk=np.arctan2(np.sqrt(1-eph.loc[:,'e']**2)*np.sin(Ek),np.cos(Ek)-eph.loc[:,'e'])
            
            #Argument of Latitude [rad]
            Phik=vk+eph.loc[:,'omega'] 
            
            #Corrections with Second Harmonic Pertubations
            #Corrected Argument of Latitude [rad]
            uk=Phik + eph.loc[:,'cuc']*np.cos(2*Phik) + eph.loc[:,'cus']*np.sin(2*Phik)
            #Corrected radius [m]
            rk=A*(1-eph.loc[:,'e']*np.cos(Ek)) + eph.loc[:,'crc']*np.cos(2*Phik) + eph.loc[:,'crs']*np.sin(2*Phik)
            #Corrected inclination [rad]
            ik=eph.loc[:,'i0'] + eph.loc[:,'IDOT']*toeageSec + eph.loc[:,'cic']*np.cos(2*Phik) + eph.loc[:,'cis']*np.sin(2*Phik)
            
    
            #Corrected Longitude of Ascending Node
            lambdak=eph.loc[:,'omega0'] + (eph.loc[:,'omega dot']-kepler_const[const][1])*toeageSec - kepler_const[const][1]*eph.loc[:,'toe']
            
            #Positions in orbital plane - what I've previously called 'q-coords'
            xq=rk*np.cos(uk)
            yq=rk*np.sin(uk)
            
            xk=xq*np.cos(lambdak) - yq*np.cos(ik)*np.sin(lambdak)
            yk=xq*np.sin(lambdak) + yq*np.cos(ik)*np.cos(lambdak)
            zk=yq*np.sin(ik)
            
            # Satellite clock error
            # Group delay not properly implemented yet!
            # gd = (f1**2/fs**2)*eph.loc[:,'tgd'])
            gd = 0
            dts = eph.loc[:,'clk bias']+eph.loc[:,'clk drift']*toeageSec+eph.loc[:,'clk drift rate']*toeageSec**2-dtr-gd
            
            satloc = pd.DataFrame([xk, yk, zk, dts]).T
            
            if any(tau):
                rot = -self.ohmE*tau
                for key in tau.index:
                    # Define rotational matrix
                    rotmat = np.array([[np.cos(rot[key]), -np.sin(rot[key]), 0, 0],
                                       [np.sin(rot[key]), np.cos(rot[key]), 0, 0],
                                       [0, 0, 1, 0], 
                                       [0, 0, 0, 1]])
                    # Find position and rotate to account for rotation of the Earth
                    satloc.loc[key,:] = rotmat @ satloc.loc[key,:]
            
            satpos.append(satloc)
            
        satpos = pd.concat(satpos)
        satpos.columns = ['X','Y','Z','dt']
        self.curEpoch = epoch
        
        return satpos