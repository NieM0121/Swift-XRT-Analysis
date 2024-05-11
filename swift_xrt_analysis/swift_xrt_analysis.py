#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   swift_xrt_analysis.py
@Time    :   2024/05/11 14:13:27
@Author  :   Meng-Nie
@Version :   1.0
@Email   :   niemeng@mail.ynu.edu.cn
@Desc    :   None
'''

import os
import re
import subprocess as sbp
import shutil
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import numpy as np
import requests
import astropy.units as u
from astropy.io import fits
from collections import Counter
from bs4 import BeautifulSoup
from astropy.coordinates import SkyCoord
from scipy.optimize import curve_fit
from scipy.stats import linregress
#Define environment
heasoft_env=os.path.join(os.environ["HEADAS"],'headas-init.sh')
os.environ['HEADASPROMPT']='/dev/null'


class CheckEnv:
        """
        Check whether HEASoft are set up, usually the environment name is named 'HEADAS'.
        Attributes:
                path: The PATH of environments.
        Args:
                env: Environments name.
        """
        def __init__(self, env):
                if env not in os.environ:
                        raise ValueError(f"Environment variable {env} is not set up.")
                print(f'The {env} variable environment has been set up.')
                
                self._path = os.environ[env]

        @property
        def path(self):
                return self._path
        
class CheckCaldb:
        """
        To check that the CALDB is correctly installed.
        """
        def __init__(self):
                sbp.run(['bash','-c', f'source {heasoft_env} && \
                        caldbinfo infomode=INST mission=SWIFT instrument=XRT'],shell=False)
                
class OrganizeFiles:
        """
        Create the appropriate directory structure for the target file.
        Args:
                target_path: Downloaded and decompressed SWIFT/XRT data path. You can change the target name instead of 'download', 
                             but make sure the directory structure is as follows:
                                First level directory:  download (or you rename)
                                Second leval directory: obsid data files
        Notes:  If you have already executed this command, there is no need to execute this command in the next steps,
                otherwise the file format will be disrupted.
        """
        def __init__(self,target_path):
                files=os.listdir(target_path)
                obsids=[]
                for file in files:
                        obsids.append(file)
                        sbp.run(['mkdir',
                                target_path+'/'+f'{file}'+'_result'],shell=False)
                sbp.run(['mkdir',
                        target_path+'/data',
                        target_path+'/centroid',
                        target_path+'/lc',
                        target_path+'/spec',
                        target_path+'/img',
                        target_path+'/region',
                        target_path+'/log',
                        target_path+'/pup',
                        target_path+'/gti',
                        target_path+'/expmap',
                        target_path+'/corrfactor'],shell=False)
                
                for mode in ['pc','wt']:
                        sbp.run(['mkdir',
                                target_path+'/image/'+f'{mode}',
                                target_path+'/region/'+f'{mode}',
                                target_path+'/centroid/'+f'{mode}',
                                target_path+'/corrfactor/'+f'{mode}',
                                target_path+'/evt/'+f'{mode}',
                                target_path+'/expmap/'+f'{mode}',
                                target_path+'/gti/'+f'{mode}',
                                target_path+'/lc/'+f'{mode}',
                                target_path+'/spec/'+f'{mode}',
                                target_path+'/img/'+f'{mode}',
                                target_path+'/pup/'+f'{mode}'],shell=False)
                
                sbp.run(['mkdir',
                        target_path+'/log/cpslog',
                        target_path+'/log/mkimg',
                        target_path+'/evt/source',
                        target_path+'/evt/bg',
                        target_path+'/corrfactor/srawinstr'],shell=False)
                
                for i in obsids:
                        sbp.run(['mv',
                                target_path+'/'+f'{i}',
                                target_path+'/data'],shell=False)

class EnhancedPosition:
        """
        Confirm the enhanced position of the target source.
        Args:
                target_id: The ID of the target source. Generally, it is the first 8 digits of the observation ID.
        """
        def __init__(self,target_id):
                url = f"https://www.swift.ac.uk/xrt_positions/{target_id}/"
                response = requests.get(url)
                content = response.text
                soup = BeautifulSoup(content, "html.parser")
                ra_element = soup.find("th", string="RA (J2000):").find_next_sibling("td")
                dec_element= soup.find("th", string="Dec (J2000):").find_next_sibling("td")
                self.ra_text = ra_element.get_text(strip=True)
                self.dec_text = dec_element.get_text(strip=True)
                self.ra_value_d = self.ra_text.split("(")[-1].split(")")[0]
                self.dec_value_d = self.dec_text.split("(")[-1].split(")")[0]
                self.ra_value_h = self.ra_text.split("(")[0]
                self.dec_value_h = self.dec_text.split("(")[0]

        def save(self,target_path):
                pos=f"RA(degrees) = {self.ra_value_d}\n\
                        Dec(degrees) = {self.dec_value_d}\n\
                        \n\
                        RA(hh mm ss.s) = {self.ra_value_h}\n\
                        Dec(dd mm ss.s) = {self.dec_value_h}"
                enhancedposfile=f'{target_path}/enhanced_pos.txt'
                with open(enhancedposfile,"w") as tmp:
                        lines = [line.lstrip() for line in pos.split('\n')]
                        new_pos = '\n'.join(lines)
                        tmp.write(new_pos)
                
        def value(self):
                print(f'RA(J2000): {self.ra_text} \nDec(J2000): {self.dec_text}')
                
#Preparationphase                       
class GenerateList:
        """
        Get a list of required files.
        """
        def __init__(self,target_path):
                obsIDlist=f'{target_path}/obsID.list'
                PCvignetEXPOlist=f'{target_path}/pc_vignet_expo.list'
                WTvignetEXPOlist=f'{target_path}/wt_vignet_expo.list'
                PCevtlist=f'{target_path}/pc_evt.list'
                WTevtlist=f'{target_path}/wt_evt.list'

                obs=[]
                for i in os.listdir(f'{target_path}/data'):
                        if os.path.isdir(f'{target_path}/data/{i}'):
                                obs.append(i)
                obs=sorted(obs)
                with open(obsIDlist,"w") as tmp:
                        for i in obs:
                                tmp.write(f'{i}\n')
                with open(PCvignetEXPOlist,"w") as tmp:
                        for i in obs:
                                if os.path.exists(f'{target_path}/data/{i}/xrt/event/sw{i}xpcw3po_vignet_ex.img.gz'):
                                                tmp.write(f'{target_path}/data/{i}/xrt/event/sw{i}xpcw3po_vignet_ex.img.gz\n')
                with open(WTvignetEXPOlist,"w") as tmp:
                        for i in obs:
                                if os.path.exists(f'{target_path}/data/{i}/xrt/event/sw{i}xwtw2po_vignet_ex.img.gz'):
                                                tmp.write(f'{target_path}/data/{i}/xrt/event/sw{i}xwtw2po_vignet_ex.img.gz\n')
                with open(PCevtlist,"w") as tmp:
                        for i in obs:
                                if os.path.exists(f'{target_path}/data/{i}/xrt/event/sw{i}xpcw3po_cl.evt.gz'):
                                                tmp.write(f'{target_path}/data/{i}/xrt/event/sw{i}xpcw3po_cl.evt.gz\n')
                with open(WTevtlist,"w") as tmp:
                        for i in obs:
                                if os.path.exists(f'{target_path}/data/{i}/xrt/event/sw{i}xwtw2po_cl.evt.gz'):
                                                tmp.write(f'{target_path}/data/{i}/xrt/event/sw{i}xwtw2po_cl.evt.gz\n')

class getradectrig:
        """
        Get coordinates and trigger time from original file.

        """
        def __init__(self,target_path):
                PCevtlist=open(f'{target_path}/pc_evt.list',"r").readlines()
                workevt=PCevtlist[0][:PCevtlist[0].index("\n")]
                data=fits.open(workevt)
                header=data[0].header
                self._trigtime=header['TRIGTIME']
                self._ra=header['RA_OBJ']
                self._dec=header['DEC_OBJ']
        @property
        def value(self):
                print(f'RA:{self._ra} Dec:{self._dec} T0:{self._trigtime}')
        @property
        def trigtime(self):
                return self._trigtime
        @property
        def ra(self):
                return self._ra
        @property
        def dec(self):
                return self._dec
class mkimg:
        """
        Create image files.
        """
        def __init__(self,inputevt,output,log):
                sbp.run(['bash','-c', f'source {heasoft_env} && \
                        extractor adjustgti=yes gti=GTI copyall=yes clobber=yes \
                        filename={inputevt} eventsout=NONE imgfile={output} fitsbinlc=NONE phafile=NONE regionfile=NONE timefile=NONE xcolf=X ycolf=Y tcol=TIME > {log}'],shell=False,stdout=sbp.PIPE)

class segmenevt:
        """
        Split event files based on oorbits.
        """
        def __init__(self,evtfile,outfile,time):
                xco=f"xsel\n\
                        no\n\
                        read eve {evtfile}\n\
                        ./\n\
                        yes\n\
                        filter time scc\n\
                        {time}\n\
                        x\n\
                        extract eve copyall=yes\n\
                        save eve {outfile}\n\
                        y\n\
                        y\n\
                        quit\n\
                        no"
                xcofile=f'segmenevt.xco'
                with open(xcofile,"w") as tmp:
                        lines = [line.lstrip() for line in xco.split('\n')]
                        new_xco = '\n'.join(lines)
                        tmp.write(new_xco)
                sbp.run(['bash','-c', f'source {heasoft_env} && \
                        xselect @{xcofile}'],shell=False,stdout=sbp.PIPE)
                os.remove(xcofile)

class xrtcentroid:
        """
        Use XRTCENTROID to get the centroid.
        """
        def __init__(self,infile,outfile,outdir,ra,dec,radius):
                sbp.run(['bash','-c', f'source {heasoft_env} && \
                        xrtcentroid infile={infile} clobber=yes \
                        outfile={outfile} outdir={outdir} calcpos=yes interactive=no boxra={ra} boxdec={dec} boxradius={radius}'],shell=False,stdout=sbp.PIPE)


class sky2xy:
        """
        Sky coordinates are converted to physical coordinates of the image.
        """
        def __init__(self,infile,xsky,ysky,log):
                sbp.run(['bash','-c', f'source {heasoft_env} && sky2xy \
                         infile={infile} xsky={xsky} ysky={ysky} > {log}'])
                
class xy2sky:
        """
        Image physical coordinates are converted to sky coordinates.
        """
        def __init__(self,infile,xpix,ypix,log):
                sbp.run(['bash','-c', f'source {heasoft_env} && xy2sky \
                         infile={infile} xpix={xpix} ypix={ypix} > {log}'])

class sourcedetect:
        """
        Detect source.
        """
        def __init__(self,img,outfile):
                xcm=f"chat 0\n\
                        read {img}\n\
                        cpd /xtk\n\
                        disp\n\
                        detect/bright/back_box_size=32/prob_limit=4e-4/snr=2.5/source_box_size=4/filedet=\"{outfile}\"\n\
                        quit"
                xcmfile='detect.xcm'
                with open(xcmfile,"w") as tmp:
                        lines = [line.lstrip() for line in xcm.split('\n')]
                        new_xcm = '\n'.join(lines)
                        tmp.write(new_xcm)
                sbp.run(['bash','-c', f'source {heasoft_env} && \
                        ximage @{xcmfile}'],shell=False,stdout=sbp.PIPE)
                os.remove(xcmfile)

class centroid:
        """
        Use XIMAGE to get the centroid.
        """
        def __init__(self,img,x,y,log):
                xcm=f"read {img}\n\
                        centroid/xpix={x}/ypix={y}/boxradius=2\n\
                        exit"
                xcmfile='centroid.xcm'
                with open(xcmfile,"w") as tmp:
                        lines = [line.lstrip() for line in xcm.split('\n')]
                        new_xcm = '\n'.join(lines)
                        tmp.write(new_xcm)
                sbp.run(['bash','-c', f'source {heasoft_env} && \
                        ximage @{xcmfile} > {log}'],shell=False,stdout=sbp.PIPE)
                os.remove(xcmfile)

class SourceIgnore:
        """
        The detection source needs to ignore the area.
        Args:
                target_path: Target file path.
        """
        def __init__(self,target_path):
                PCevtlist=open(f'{target_path}/pc_evt.list',"r").readlines()
                detect_result=open(f'{target_path}/source.list',"w")
                badbg=open(f'{target_path}/badbg.reg',"w")
                badbg.write('fk5\n')
                workevt=PCevtlist[0][:PCevtlist[0].index("\n")]
                output='tmp.img'
                mkimg(workevt,f'{target_path}/{output}',f'{target_path}/tmp.log')
                detfile='detsource.det'
                sourcedetect(f'{target_path}/{output}',f'{target_path}/{detfile}')
                content=[]
                with open(detfile,"r") as tmp:
                        for line in tmp:
                                content.append(line.strip())
                RA=float(getradectrig(target_path).ra)
                DEC=float(getradectrig(target_path).dec)
                for line in content:
                        match_ra= re.search(r'(\d{2}) (\d{2}) (\d{2}\.\d+)', line)
                        match_dec= re.search(r'\+(\d{2}) (\d{2}) (\d{2}\.\d+)', line)
                        if match_ra is not None and match_dec is not None:
                                x=float(line[21:30])
                                y=float(line[31:40])
                                radius=float(line[77:84])
                                xy2sky(f'{target_path}/{output}',x,y,'xy2sky.log')
                                log=open('xy2sky.log',"r").readlines()
                                ra_fk5 = float(log[1][29:48])
                                dec_fk5 = float(log[1][52:])
                                distance=np.sqrt((ra_fk5-RA)**2+(dec_fk5-DEC)**2)*60*60
                                detect_result.write(f'({ra_fk5},{dec_fk5}),Distance={distance}\",SNR={float(line[94:])}\n')
                                print(f'Source at ({ra_fk5},{dec_fk5}) is {distance}\" away.')
                                if distance>20:
                                        badbg.write(f'-circle({ra_fk5},{dec_fk5},{radius}\")\n')
                                if distance<=20:
                                        getRA=ra_fk5
                                        getDEC=dec_fk5
                                os.remove('xy2sky.log')
                detect_result.close()
                badbg.close()
                os.remove(f'{target_path}/{output}')
                os.remove(f'{target_path}/tmp.log')
                worksource=f'{target_path}/working_radec.txt'
                with open(worksource,"w") as tmp:
                        tmp.write(f'{getRA}\n')
                        tmp.write(f'{getDEC}\n')
                print(f'Working RA:{getRA} DEC:{getDEC}')

class orbit:
        """
        Make orbit file.
        Args:
                evtfile: Input events file.
                outfile: Output file.
        """
        def __init__(self,evtfile,outfile):
                data=fits.open(evtfile)
                gti=data[2].data
                output=open(f'{outfile}',"w")
                print(f'There are {len(gti)} orbits')
                for i in range(len(gti)):
                        print(f'orbit{i}:{gti[i][0]} - {gti[i][1]}')
                        output.write(f'{gti[i][0]},{gti[i][1]}\n')
                output.close()

class mkexpomap:
        """
        Generate exposure maps for XRT Photon Counting and Windowed Timing modes event files.
        Args:
                evtfile: Name of the input cleaned event FITS file.
                attfile: Name of the input Attitude FITS file.
                hdfile: Name of the input Housekeeping Header Packets FITS file.
                outdir: Directory for the output files.
                stemout: Stem for the output files. If stemout=DEFAULT, the standard naming convention will be used.
        """
        def __init__(self,evtfile,attfile,hdfile,outdir,stemout):
                sbp.run(['bash','-c', f'source {heasoft_env} && xrtexpomap \
                        infile={evtfile} \
                        attfile={attfile} \
                        hdfile={hdfile} \
                        outdir={outdir} \
                        stemout={stemout} clobber=yes'],shell=False,stdout=sbp.PIPE)

class srccounts:
        """
        """
        def __init__(self,infile,regionfile,log):
                sbp.run(['bash','-c', f'source {heasoft_env} && \
                        extractor adjustgti=yes gti=GTI copyall=yes clobber=yes \
                        filename={infile} eventsout=NONE imgfile=None fitsbinlc=NONE phafile=NONE regionfile={regionfile} timefile=NONE xcolf=X ycolf=Y tcol=TIME > {log}'],shell=False,stdout=sbp.PIPE)


class extractevt:
        """
        """
        def __init__(slef,infile,outfile,regionfile):
                sbp.run(['bash','-c', f'source {heasoft_env} && \
                        extractor adjustgti=yes gti=GTI copyall=yes clobber=yes \
                        filename={infile} eventsout={outfile} imgfile=None fitsbinlc=NONE phafile=NONE regionfile={regionfile} timefile=NONE xcolf=X ycolf=Y tcol=TIME'],shell=False,stdout=sbp.PIPE)
class extractpha:
        """
        Extract spectrum.
        Args:
                infile: Input events file.
                outfile: Output events file.
                regionfile: Extract region file.
        """
        def __init__(slef,infile,outfile,regionfile):
                sbp.run(['bash','-c', f'source {heasoft_env} && \
                        extractor adjustgti=yes gti=GTI copyall=yes clobber=yes \
                        filename={infile} eventsout=None imgfile=None fitsbinlc=NONE phafile={outfile} regionfile={regionfile} timefile=NONE xcolf=X ycolf=Y xcolh=X ycolh=Y ecol=PI tcol=TIME'],shell=False,stdout=sbp.PIPE)

class preparation:
        """
        Some operations in the preparation stage.
        Args:
                target_path: Target file path.
        """
        def __init__(self,target_path):
                obslist=open(f'{target_path}/obsID.list',"r").readlines()
                coor=open(f'{target_path}/working_radec.txt',"r").readlines()
                ra=float(coor[0])
                dec=float(coor[1])
                for i in obslist:
                        obs=i[:i.index("\n")]
                        pcevt=f'{target_path}/data/{obs}/xrt/event/sw{obs}xpcw3po_cl.evt.gz'
                        wtevt=f'{target_path}/data/{obs}/xrt/event/sw{obs}xwtw2po_cl.evt.gz'
                        if os.path.exists(pcevt):
                                print(f'PC mode: {obs}')
                                orbit(pcevt,f'{target_path}/gti/pc/{obs}_pc.gti')
                                gtifile=open(f'{target_path}/gti/pc/{obs}_pc.gti',"r").readlines()
                                for j in range(len(gtifile)):
                                        time=gtifile[j][:gtifile[j].index("\n")]
                                        segmenevt(pcevt,f'{target_path}/evt/pc/{obs}_pc_orb{j}.evt',time)
                                        mkimg(f'{target_path}/evt/pc/{obs}_pc_orb{j}.evt',f'{target_path}/img/pc/{obs}_pc_orb{j}.img',f'{target_path}/log/mkimg/{obs}_pc_orb{j}_mkimg.log')
                                        log=open(f'{target_path}/log/mkimg/{obs}_pc_orb{j}_mkimg.log',"r").read()
                                        match = re.search(r"Grand Total\s+Good\s+Bad.*\n\s+\d+\s+(\d+)", log)
                                        good_value=int(match.group(1))
                                        print(good_value)
                                        if good_value >= 20:
                                                sky2xy(f'{target_path}/img/pc/{obs}_pc_orb{j}.img',ra,dec,'sky2xy.log')
                                                log_xy=open('sky2xy.log',"r").readlines()
                                                x=float(log_xy[1][27:50])
                                                y=float(log_xy[1][52:])
                                                mkexpomap(f'{target_path}/evt/pc/{obs}_pc_orb{j}.evt',f'{target_path}/data/{obs}/auxil/sw{obs}pat.fits.gz',f'{target_path}/data/{obs}/xrt/hk/sw{obs}xhd.hk.gz',f'{target_path}/expmap/pc',f'{obs}_pc_orb{j}')
                                                centroid(f'{target_path}/img/pc/{obs}_pc_orb{j}.img',x,y,f'{target_path}/centroid/pc/{obs}_pc_orb{j}_centroid.log')
                                                os.remove('sky2xy.log')
                        if os.path.exists(wtevt):
                                print(f'WT mode: {obs}')
                                orbit(wtevt,f'{target_path}/gti/wt/{obs}_wt.gti')
                                gtifile=open(f'{target_path}/gti/wt/{obs}_wt.gti',"r").readlines()
                                for j in range(len(gtifile)):
                                        time=gtifile[j][:gtifile[j].index("\n")]
                                        segmenevt(wtevt,f'{target_path}/evt/wt/{obs}_wt_orb{j}.evt',time)
                                        mkimg(f'{target_path}/evt/wt/{obs}_wt_orb{j}.evt',f'{target_path}/img/wt/{obs}_wt_orb{j}.img',f'{target_path}/log/mkimg/{obs}_wt_orb{j}_mkimg.log')
                                        log=open(f'{target_path}/log/mkimg/{obs}_wt_orb{j}_mkimg.log',"r").read()
                                        match = re.search(r"Grand Total\s+Good\s+Bad.*\n\s+\d+\s+(\d+)", log)
                                        good_value=int(match.group(1))
                                        print(good_value)
                                        if good_value >= 20:
                                                sky2xy(f'{target_path}/img/wt/{obs}_wt_orb{j}.img',ra,dec,'sky2xy.log')
                                                log_xy=open('sky2xy.log',"r").readlines()
                                                x=float(log_xy[1][27:50])
                                                y=float(log_xy[1][52:])
                                                mkexpomap(f'{target_path}/evt/wt/{obs}_wt_orb{j}.evt',f'{target_path}/data/{obs}/auxil/sw{obs}pat.fits.gz',f'{target_path}/data/{obs}/xrt/hk/sw{obs}xhd.hk.gz',f'{target_path}/expmap/wt',f'{obs}_wt_orb{j}')
                                                centroid(f'{target_path}/img/wt/{obs}_wt_orb{j}.img',x,y,f'{target_path}/centroid/wt/{obs}_wt_orb{j}_centroid.log')
                                                os.remove('sky2xy.log')

class FilterProcEvt:
        """
        Filter event files that need to be processed.
        Args:
                target_path: Target file path.
        """
        def __init__(self,target_path):
                pcevtlist=os.listdir(f'{target_path}/evt/pc/')
                wtevtlist=os.listdir(f'{target_path}/evt/wt/')
                proevt=open(f'{target_path}/procevt.list',"w")
                pupevt=open(f'{target_path}/pupevt.list',"w")
                coor=open(f'{target_path}/working_radec.txt',"r").readlines()
                workra=float(coor[0])
                workdec=float(coor[1])
                for i in pcevtlist:
                        evt=i[:i.index(".")]
                        if os.path.exists(f'{target_path}/centroid/pc/{evt}_centroid.log'):
                                centroid=open(f'{target_path}/centroid/pc/{evt}_centroid.log',"r").read()
                                coor_match = re.search(r'RA/Dec \(2000\) =\s*(\d+\.\d+)\s+(\d+\.\d+)', centroid)
                                ra=float(coor_match.group(1))
                                dec=float(coor_match.group(2))
                                with open(f'{target_path}/tmp.reg',"w") as tmp:
                                        tmp.write(f'fk5;circle({ra},{dec},70.8)')
                                srccounts(f'{target_path}/evt/pc/{i}',f'{target_path}/tmp.reg','srccounts.log')
                                log=open('srccounts.log',"r").read()
                                match = re.search(r"Grand Total\s+Good\s+Bad.*\n\s+\d+\s+(\d+)", log)
                                good_value=int(match.group(1))
                                matchtime= re.search(r'in\s+([0-9.]+)\s+seconds', log)
                                time=float(matchtime.group(1))
                                cps=good_value/time
                                distance=np.sqrt((ra-workra)**2+(dec-workdec)**2)*60*60
                                print(f'PC mode {evt} centroid distance is {distance}\" away and Source Counts: {good_value}.')
                                if distance<=20 and good_value>=20:
                                        print(f'Add {evt} to procevt.')
                                        proevt.write(f'{evt}\n')
                                        shutil.copy('srccounts.log',f'{target_path}/log/cpslog/{evt}_cps.log')
                                        if cps > 0.5:
                                                pupevt.write(f'{evt}\n')
                                                print(f'PC mode {evt} has {cps} counts/s (>0.5) exists Pile-up !')
                                os.remove(f'{target_path}/tmp.reg')
                                os.remove('srccounts.log')
                for i in wtevtlist:
                        evt=i[:i.index(".")]
                        if os.path.exists(f'{target_path}/centroid/wt/{evt}_centroid.log'):
                                centroid=open(f'{target_path}/centroid/wt/{evt}_centroid.log',"r").read()
                                coor_match = re.search(r'RA/Dec \(2000\) =\s*(\d+\.\d+)\s+(\d+\.\d+)', centroid)
                                ra=float(coor_match.group(1))
                                dec=float(coor_match.group(2))
                                with open(f'{target_path}/tmp.reg',"w") as tmp:
                                        tmp.write(f'fk5;circle({ra},{dec},70.8)')
                                srccounts(f'{target_path}/evt/wt/{i}',f'{target_path}/tmp.reg','srccounts.log')
                                log=open('srccounts.log',"r").read()
                                match = re.search(r"Grand Total\s+Good\s+Bad.*\n\s+\d+\s+(\d+)", log)
                                good_value=int(match.group(1))
                                matchtime= re.search(r'in\s+([0-9.]+)\s+seconds', log)
                                time=float(matchtime.group(1))
                                cps=good_value/time
                                distance=np.sqrt((ra-workra)**2+(dec-workdec)**2)*60*60
                                print(f'WT mode {evt} centroid distance is {distance}\" away and Source Counts: {good_value}.')
                                if distance<=12 and good_value>=20:
                                        print(f'Add {evt} to procevt.')
                                        proevt.write(f'{evt}\n')
                                        shutil.copy('srccounts.log',f'{target_path}/log/cpslog/{evt}_cps.log')
                                        if cps > 100:
                                                pupevt.write(f'{evt}\n')
                                                print(f'WT mode {evt} has {cps} counts/s (>100) exists Pile-up !')
                                os.remove(f'{target_path}/tmp.reg')
                                os.remove('srccounts.log')
                proevt.close()
                pupevt.close()

class DetermineRegionPC:
        """
        Determine the pile-up range of target source for PC mode.You need to move the psf.cod file in the model folder to your Target directory.
        Args:
                target_path: Target file path.
        """
        def __init__(self,target_path):
                pupevtlist=open(f'{target_path}/pupevt.list',"r").readlines()
                for i in range(len(pupevtlist)):
                        if 'pc' in pupevtlist[i]:
                                event=pupevtlist[i][:pupevtlist[i].index('\n')]
                                centroid=open(f'{target_path}/centroid/pc/{event}_centroid.log',"r").read()
                                coord_match = re.search(r'X/Ypix\s*=\s*(\d+\.\d+)\s+(\d+\.\d+)', centroid)
                                X=float(coord_match.group(1))
                                Y=float(coord_match.group(2))
                                log=open(f'{target_path}/log/cpslog/{event}_cps.log',"r").read()
                                match = re.search(r"Grand Total\s+Good\s+Bad.*\n\s+\d+\s+(\d+)", log)
                                good_value=int(match.group(1))
                                matchtime= re.search(r'in\s+([0-9.]+)\s+seconds', log)
                                ctime=float(matchtime.group(1))
                                cps=good_value/ctime
                                if cps < 1.5:
                                        wing=15
                                if cps >= 1.5:
                                        wing=22
                                xcm=f"cpd /null\n\
                                        read {target_path}/img/pc/{event}.img\n\
                                        back\n\
                                        set qdp_script \"{event}_script.txt\"\n\
                                        open $qdp_script \"w\"\n\
                                        set fp [open $qdp_script \"w\"]\n\
                                        puts $fp \"wdata {target_path}/pup/pc/{event}_psf.qdp\"\n\
                                        puts $fp \"r x {wing}\"\n\
                                        puts $fp \"col off 1 2 3 4 6\"\n\
                                        puts $fp \"model psf.cod\"\n\
                                        puts $fp \"\"\n\
                                        puts $fp \"fit I 1000000\"\n\
                                        puts $fp \"wm {target_path}/pup/pc/{event}_model.dat\"\n\
                                        puts $fp \"quit\"\n\
                                        close $fp\n\
                                        set status [catch {{psf/xpix={X}/ypix={Y}/radius=2}} errMsg]\n\
                                        if {{ $status != 0 }} {{exit 1}}\n\
                                        exit"
                                xcmfile=f'{target_path}/{event}_puppc.xcm'
                                with open(xcmfile,"w") as tmp:
                                        lines = [line.lstrip() for line in xcm.split('\n')]
                                        new_xcm = '\n'.join(lines)
                                        tmp.write(new_xcm)
                                sbp.Popen(['gnome-terminal','--','bash','-c',f'source {heasoft_env} && \
                                        ximage @{xcmfile}'])
                                
class PSF:
        """
        PSF fit.
        Args:
                model: model file.
                qdp: qdp file
                outfile: output file
        """
        def __init__(self,model,qdp,outfile):
                N=0.0807
                sigma=7.422
                rc=3.726
                beta=1.305
                imgpsf=open(f'{qdp}',"r").readlines()
                modelfile=open(f'{model}',"r").readlines()
                P1=float(modelfile[1][0:15])

                rlist=[]
                ylist=[]
                dylist=[]
                for line in imgpsf[3:]:
                        splitted_line = line.strip().split()
                        r = round(float(splitted_line[0]),8)
                        y=round(float(splitted_line[7]),8)
                        err=round(float(splitted_line[8]),8)
                        rlist.append(r)
                        ylist.append(y)
                        dylist.append(err)

                fitPSF=N*P1*np.exp(-np.array(rlist)**2/(2*sigma**2))+P1*(1+(np.array(rlist)/rc)**2)**(-beta)
                diff=fitPSF-ylist
                sdiff=(fitPSF-ylist)/dylist
                table=pd.DataFrame({'r':rlist,
                    'imgPSF':ylist,
                    'imgPSF_ERR':dylist,
                    'fitPSF':fitPSF,
                    'fitPSF_DIFF':diff,
                    'SDIFF':sdiff})
                table.to_csv(f'{outfile}',index=False)

class exclude:
        """
        Exclude region select.
        Args:
                target_path: Target file path.
        """
        def __init__(self,target_path):
                pileupfile=open(f'{target_path}/pupevt.list',"r").readlines()
                for i in pileupfile:
                        eventname=i[:i.index("\n")]
                        if "pc" in eventname:
                                if not os.path.exists(f'{target_path}/pup/pc/{eventname}_model.dat') or not os.path.exists(f'{target_path}/pup/pc/{eventname}_psf.qdp'):
                                        with open(f'{target_path}/pup/pc/{eventname}_exclude.txt',"w") as tmp:
                                                tmp.write('23.6\"')
                                if os.path.exists(f'{target_path}/pup/pc/{eventname}_model.dat') and os.path.exists(f'{target_path}/pup/pc/{eventname}_psf.qdp'):
                                        model=f'{target_path}/pup/pc/{eventname}_model.dat'
                                        psf=f'{target_path}/pup/pc/{eventname}_psf.qdp'
                                        PSF(model,psf,f'{target_path}/pup/pc/{eventname}_fitresult.txt')
                                        with open(f'{target_path}/pup/pc/{eventname}_exclude.txt',"w"):
                                                pass
                        
class RadiusSelection:
        """
        Select the radius of the region based on the source's rate (count per seconds).
        Args:
                target_path: Target file path.
                eventname: Event name.
        """
        def __new__(self,cpspath):
                log=open(f'{cpspath}',"r").read()
                match = re.search(r"Grand Total\s+Good\s+Bad.*\n\s+\d+\s+(\d+)", log)
                good_value=int(match.group(1))
                matchtime= re.search(r'in\s+([0-9.]+)\s+seconds', log)
                time=float(matchtime.group(1))
                cps=good_value/time
                if cps > 0.5:
                        return 30
                if cps > 0.1 and cps <=0.5:
                        return 25
                if cps > 0.05 and cps <=0.1:
                        return 20
                if cps > 0.01 and cps <=0.05:
                        return 15
                if cps > 0.005 and cps <=0.01:
                        return 12
                if cps > 0.001 and cps <=0.005:
                        return 9
                if cps > 0.0005 and cps <=0.001:
                        return 7
                if cps <=0.0005:
                        return 5




class hk:
        """
        Get parameter values in HK file.
        Args:   
                target_path: Target file path.
                obsid: The observation ID you need to open the image.
                parameter: Parameters to be viewed.
        """
        def __new__(self,infile,parameter):
                hk_data,hk_headers=fits.getdata(f'{infile}',header=True)
                return hk_headers[parameter]


class nH:
        """
        Determine the NH value of the target source.
        Args:
                target_path: Target file path.
        """
        def __new__(self,ra,dec):
                result=sbp.check_output(['bash','-c', f'source {heasoft_env} && \
                        nh Equinox=2000 RA={float(ra)} DEC={float(dec)}'])
                match_nh = re.search(r'Weighted average nH \(cm\*\*-2\)  (\d+\.\d+E[+-]\d+)', result.decode())
                nh=float(match_nh.group(1))
                return nh


class rmfinfo:
        """
        Search the RMF file used by the target source.
        Args:
        """
        def __new__(self,evtfile):
                data,data_header=fits.getdata(evtfile,header=True)
                date_obs=data_header['DATE-OBS']
                date=date_obs[:date_obs.index('T')]
                time=date_obs[date_obs.index('T')+1:]
                if 'pc' in evtfile:
                        result=sbp.check_output(['bash','-c', f'source {heasoft_env} && \
                                quzcif mission=SWIFT instrument=XRT codename=matrix detector=- filter=- date={date} time={time} \
                                expr=datamode.eq.photon.and.grade.eq.G0:12.and.XRTVSUB.eq.6'])
                        rmffile=result.decode("utf-8")
                        return rmffile[:rmffile.index(' ')]
                if 'wt' in evtfile:
                        result=sbp.check_output(['bash','-c', f'source {heasoft_env} && \
                                quzcif mission=SWIFT instrument=XRT codename=matrix detector=- filter=- date={date} time={time} \
                                expr=datamode.eq.windowed.and.grade.eq.G0:2.and.XRTVSUB.eq.6'])
                        rmffile=result.decode("utf-8")
                        return rmffile[:rmffile.index(' ')]

class DetermineRegionWT_PotonIndex:
        """
        Determine the pile-up range of target source for WT mode.
        Args:
                target_path: Target file path.
        """
        def __init__(self,target_path):
                os.environ['HEADASPROMPT']='/dev/null'
                eventlistfile=open(f'{target_path}/pupevt.list',"r")
                eventlist=eventlistfile.readlines()
                
                for i in range(len(eventlist)):
                        if 'wt' in eventlist[i]:
                                eventname=eventlist[i][:eventlist[i].index('\n')]
                                print(eventname)
                                outradius=RadiusSelection(f'{target_path}/log/cpslog/{eventname}_cps.log')
                                evtfile=f'{target_path}/evt/wt/{eventname}.evt'
                                obsid=eventname[:eventname.index('_')]
                                rmffile=rmfinfo(evtfile)
                                sbp.run(['mkdir',f'{target_path}/pup/wt/{eventname}'])
                                sbp.run(['mkdir',f'{target_path}/pup/wt/{eventname}/photonindex/'])
                                pixel=[]
                                index=[]
                                centroid=open(f'{target_path}/centroid/wt/{eventname}_centroid.log',"r").read()
                                coor_match = re.search(r'RA/Dec \(2000\) =\s*(\d+\.\d+)\s+(\d+\.\d+)', centroid)
                                ra=float(coor_match.group(1))
                                dec=float(coor_match.group(2))
                                nh=nH(ra,dec)
                                for j in range(25):
                                        num='{:02d}'.format(j)
                                        pixel.append(j)
                                        #make energy spectrum
                                        region=f'{target_path}/pup/wt/{eventname}/photonindex/region{num}.reg'
                                        with open(region,'w') as tmp:
                                                tmp.write(f'fk5\nannulus({ra},{dec},{round(j*2.36,2)}",{outradius*2.36}")')
                                        xco_spec=f'xsel\n\
                                                no\n\
                                                read eve {evtfile}\n\
                                                ./\n\
                                                yes\n\
                                                filter region {region}\n\
                                                extract spec\n\
                                                save spec {target_path}/pup/wt/{eventname}/photonindex/{num}.pha\n\
                                                yes\n\
                                                quit\n\
                                                no'
                                        xcofile_spec=f'{target_path}/pup/wt/{eventname}_pileup_{num}.xco'
                                        with open(xcofile_spec,"w") as tmp:
                                                lines = [line.lstrip() for line in xco_spec.split('\n')]
                                                new_xco_spec = '\n'.join(lines)
                                                tmp.write(new_xco_spec)
                                        sbp.run(['bash','-c', f'source {heasoft_env} && \
                                                xselect @{xcofile_spec}'],shell=False)
                                        os.remove(xcofile_spec)
                                        #make exposure map
                                        sbp.run(['bash','-c', f'source {heasoft_env} && \
                                                xrtexpomap infile={evtfile} \
                                                attfile={target_path}/data/{obsid}/auxil/sw{obsid}pat.fits.gz \
                                                hdfile={target_path}/data/{obsid}/xrt/hk/sw{obsid}xhd.hk.gz \
                                                outdir={target_path}/pup/wt/{eventname}/photonindex/ \
                                                stemout={num} clobber=yes'],shell=False)
                                        #make arf
                                        sbp.run(['bash','-c', f'source {heasoft_env} && \
                                                xrtmkarf expofile={target_path}/pup/wt/{eventname}/photonindex/{num}_ex.img \
                                                phafile={target_path}/pup/wt/{eventname}/photonindex/{num}.pha \
                                                srcx=-1 srcy=-1 \
                                                outfile={target_path}/pup/wt/{eventname}/photonindex/{num}.arf psfflag=yes clobber=yes'],shell=False)
                                        #fit data
                                        xcm_xspec=f'statistic cstat\n\
                                                data {target_path}/pup/wt/{eventname}/photonindex/{num}.pha\n\
                                                response {rmffile}\n\
                                                arf {target_path}/pup/wt/{eventname}/photonindex/{num}.arf\n\
                                                ignore **-0.3 10.0-**\n\
                                                model cflux*TBabs*powerlaw & 0.3 & 10.0 &  & {nh/1e22} -1 &  & 1 -1\n\
                                                fit 1000\n\
                                                save model {target_path}/pup/wt/{eventname}/photonindex/{num}_model.xcm\n\
                                                y\n\
                                                quit\n\
                                                y'
                                        xcmfile_xspec=f'{target_path}/pup/wt/{eventname}_xspec_{num}.xcm'
                                        with open(xcmfile_xspec,"w") as tmp:
                                                lines = [line.lstrip() for line in xcm_xspec.split('\n')]
                                                new_xcm_xspec = '\n'.join(lines)
                                                tmp.write(new_xcm_xspec)
                                        sbp.run(['bash','-c', f'source {heasoft_env} && \
                                                xspec {xcmfile_xspec}'],shell=False)
                                        os.remove(xcmfile_xspec)
                                        
                                        content=[]
                                        with open(f'{target_path}/pup/wt/{eventname}/photonindex/{num}_model.xcm','r') as tmp:
                                                for k in tmp:
                                                        content.append(k)
                                        index.append(float(content[11][0:22]))
                                pifile=f'{target_path}/pup/wt/{eventname}/photonindex/PI.csv'
                                table=pd.DataFrame({'PIXEL':pixel,'INDEX':index})
                                table.to_csv(pifile,index=False)
                                #plot
                                fig,axes = plt.subplots(figsize=(8,6))
                                axes.minorticks_on()
                                axes.tick_params(axis='both',which='major',direction='in',width=1,length=5)
                                axes.tick_params(axis='both',which='minor',direction='in',width=1,length=3)
                                axes.xaxis.set_minor_locator(MultipleLocator(1))
                                axes.xaxis.set_major_locator(MultipleLocator(5))
                                axes.plot(pixel,index,'ko')
                                #axes.axhline(np.mean(index),color='k',linestyle='--',label='Mean')
                                axes.set(ylim=(np.floor(np.min(index)),np.ceil(np.max(index))),title=f'Event:{eventname}',xlabel='Pixel', ylabel='Photon Index')
                                axes.legend(frameon=False)
                                fig.tight_layout()
                                fig.savefig(f'{target_path}/pup/wt/{eventname}/photonindex/pixel-index.png')

class DetermineRegionWT_Grade:
        """
        Determine the pile-up range of target source for WT mode.
        Args:
                target_path: Target file path.
        """
        def __init__(self,target_path):
                os.environ['HEADASPROMPT']='/dev/null'
                eventlistfile=open(f'{target_path}/pupevt.list',"r")
                eventlist=eventlistfile.readlines()
                for i in range(len(eventlist)):
                        if 'wt' in eventlist[i]:
                                eventname=eventlist[i][:eventlist[i].index('\n')]
                                print(eventname)
                                obsid=eventname[:eventname.index('_')]
                                outradius=RadiusSelection(f'{target_path}/log/cpslog/{eventname}_cps.log')
                                evtfile=f'{target_path}/evt/wt/{eventname}.evt'
                                sbp.run(['mkdir',f'{target_path}/pup/wt/{eventname}'])
                                sbp.run(['mkdir',f'{target_path}/pup/wt/{eventname}/grade/'])
                                pixel=[]
                                frac=[]
                                centroid=open(f'{target_path}/centroid/wt/{eventname}_centroid.log',"r").read()
                                coor_match = re.search(r'RA/Dec \(2000\) =\s*(\d+\.\d+)\s+(\d+\.\d+)', centroid)
                                ra=float(coor_match.group(1))
                                dec=float(coor_match.group(2))
                                for j in range(13):
                                        num='{:02d}'.format(j)
                                        pixel.append(j*2)
                                        #make events
                                        region=f'{target_path}/pup/wt/{eventname}/grade/region{num}.reg'
                                        with open(region,'w') as tmp:
                                                tmp.write(f'fk5\nannulus({ra},{dec},{round(j*2*2.36,2)}",{outradius*2.36}")')
                                        xco_spec=f'xsel\n\
                                                no\n\
                                                read eve {evtfile}\n\
                                                ./\n\
                                                yes\n\
                                                filter region {region}\n\
                                                extract eve copyall=yes\n\
                                                save eve {target_path}/pup/wt/{eventname}/grade/{num}.evt\n\
                                                yes\n\
                                                no\n\
                                                quit\n\
                                                no'
                                        xcofile_spec=f'{target_path}/pup/wt/{eventname}_pileup_{num}.xco'
                                        with open(xcofile_spec,"w") as tmp:
                                                lines = [line.lstrip() for line in xco_spec.split('\n')]
                                                new_xco_spec = '\n'.join(lines)
                                                tmp.write(new_xco_spec)
                                        sbp.run(['bash','-c', f'source {heasoft_env} && \
                                                xselect @{xcofile_spec}'],shell=False)
                                        os.remove(xcofile_spec)
                                        #make grade file for each evt
                                        gradefile=f'{target_path}/pup/wt/{eventname}/grade/{num}_grade.csv'
                                        eve=fits.open(f'{target_path}/pup/wt/{eventname}/grade/{num}.evt')
                                        data_table=pd.DataFrame({'GRADE':np.array(eve[1].data.field('GRADE'))})
                                        data_table.to_csv(gradefile)
                                        #calculate fraction of grade 0 to grade 0-2
                                        grade_data=pd.read_csv(gradefile)
                                        grade_sort=Counter(grade_data['GRADE'])
                                        frac.append(grade_sort[0] / (grade_sort[0] + grade_sort[1]+ grade_sort[2] ))
                                fracfile=f'{target_path}/pup/wt/{eventname}/grade/frac.csv'
                                frac_data=pd.DataFrame({'PIXEL':pixel,'FRACTION':frac})
                                frac_data.to_csv(fracfile,index=False)
                                #plot
                                fig,axes = plt.subplots(figsize=(8,6))
                                axes.minorticks_on()
                                axes.tick_params(axis='both',which='major',direction='in',width=1,length=5)
                                axes.tick_params(axis='both',which='minor',direction='in',width=1,length=3)
                                axes.xaxis.set_minor_locator(MultipleLocator(1))
                                axes.xaxis.set_major_locator(MultipleLocator(5))
                                axes.plot(pixel,frac,'k--')
                                axes.scatter(pixel,frac,s=50,color='k')
                                axes.set(title=f'Event:{eventname}',xlabel='Pixel', ylabel='Fraction (g$_0$ to g$_{0-2}$)')
                                fig.tight_layout()
                                fig.savefig(f'{target_path}/pup/wt/{eventname}/grade/pixel-frac.png')
        
class MakeRegion:
        """
        Make region files for the events that need to be processed.
        Args:
                target_path: Target file path.
        """
        def __init__(self,target_path):
                processeventlist=open(f'{target_path}/procevt.list',"r").readlines()
                pileupfile=open(f'{target_path}/pupevt.list').read()
                pcevtsrc=open(f'{target_path}/pcevtsrc.list',"w")
                wtevtsrc=open(f'{target_path}/wtevtsrc.list',"w")
                pcevtbg=open(f'{target_path}/pcevtbg.list',"w")
                wtevtbg=open(f'{target_path}/wtevtbg.list',"w")
                for i in processeventlist:
                        eventname=i[:i.index("\n")]
                        outradius=RadiusSelection(f'{target_path}/log/cpslog/{eventname}_cps.log')
                        if "pc" in eventname:
                                roll=hk(f'{target_path}/evt/pc/{eventname}.evt','PA_PNT')
                                centroid=open(f'{target_path}/centroid/pc/{eventname}_centroid.log',"r").read()
                                coord_match = re.search(r'X/Ypix\s*=\s*(\d+\.\d+)\s+(\d+\.\d+)', centroid)
                                X=float(coord_match.group(1))
                                Y=float(coord_match.group(2))
                        if "wt" in eventname:
                                roll=hk(f'{target_path}/evt/wt/{eventname}.evt','PA_PNT')
                                centroid=open(f'{target_path}/centroid/wt/{eventname}_centroid.log',"r").read()
                                coord_match = re.search(r'X/Ypix\s*=\s*(\d+\.\d+)\s+(\d+\.\d+)', centroid)
                                X=float(coord_match.group(1))
                                Y=float(coord_match.group(2))
                        if not eventname in pileupfile:
                                #make region for pc mode
                                if "pc" in eventname:
                                        region_src=f'{target_path}/region/pc/{eventname}_src.reg'
                                        region_back=f'{target_path}/region/pc/{eventname}_back.reg'
                                        with open(region_src,'w') as tmp:
                                                tmp.write(f'circle({X},{Y},{outradius})')
                                        with open(region_back,'w') as tmp:
                                                tmp.write(f'annulus({X},{Y},60,110)')
                                #make region for wt mode
                                if "wt" in eventname:
                                        region_src=f'{target_path}/region/wt/{eventname}_src.reg'
                                        region_back=f'{target_path}/region/wt/{eventname}_back.reg'
                                        with open(region_src,'w') as tmp:
                                                tmp.write(f'box({X},{Y},5,{outradius*2},{roll})')
                                        with open(region_back,'w') as tmp:
                                                tmp.write(f'box({X},{Y},600,600,{roll})-box({X},{Y},600,120,{roll})')
                        #For pileup events                
                        if eventname in pileupfile:
                                if "pc" in eventname:
                                        #make region
                                        with open(f'{target_path}/pup/pc/{eventname}_exclude.txt',"r") as tmp:
                                                content=tmp.readlines()
                                                excl=float(content[0])
                                        region_src=f'{target_path}/region/pc/{eventname}_src.reg'
                                        region_back=f'{target_path}/region/pc/{eventname}_back.reg'
                                        with open(region_src,'w') as tmp:
                                                tmp.write(f'annulus({X},{Y},{excl/2.36},{outradius})')
                                        with open(region_back,'w') as tmp:
                                                tmp.write(f'annulus({X},{Y},60,110)')
                                #make region for wt mode
                                if "wt" in eventname:
                                        #make region
                                        with open(f'{target_path}/pup/wt/{eventname}_exclude.txt',"r") as tmp:
                                                content=tmp.readlines()
                                                excl=float(content[0])
                                        region_src=f'{target_path}/region/wt/{eventname}_src.reg'
                                        region_back=f'{target_path}/region/wt/{eventname}_back.reg'
                                        with open(region_src,'w') as tmp:
                                                tmp.write(f'box({X},{Y},5,{outradius*2},{roll})-box({X},{Y},5,{excl},{roll})')
                                        with open(region_back,'w') as tmp:
                                                tmp.write(f'box({X},{Y},600,600,{roll})-box({X},{Y},600,120,{roll})')
                        #extract source and back
                        if "pc" in eventname:
                                extractevt(f'{target_path}/evt/pc/{eventname}.evt',f'{target_path}/evt/source/pc/{eventname}_src.evt',region_src)
                                extractevt(f'{target_path}/evt/pc/{eventname}.evt',f'{target_path}/evt/bg/pc/{eventname}_bg.evt',region_back)
                                extractpha(f'{target_path}/evt/pc/{eventname}.evt',f'{target_path}/spec/pc/{eventname}_src.pi',region_src)
                                extractpha(f'{target_path}/evt/pc/{eventname}.evt',f'{target_path}/spec/pc/{eventname}_bg.pi',region_back)
                                pcevtsrc.write(f'{target_path}/evt/source/pc/{eventname}_src.evt\n')
                                pcevtbg.write(f'{target_path}/evt/bg/pc/{eventname}_bg.evt\n')
                        if "wt" in eventname:
                                extractevt(f'{target_path}/evt/wt/{eventname}.evt',f'{target_path}/evt/source/wt/{eventname}_src.evt',region_src)
                                extractevt(f'{target_path}/evt/wt/{eventname}.evt',f'{target_path}/evt/bg/wt/{eventname}_bg.evt',region_back)
                                extractpha(f'{target_path}/evt/wt/{eventname}.evt',f'{target_path}/spec/wt/{eventname}_src.pi',region_src)
                                extractpha(f'{target_path}/evt/wt/{eventname}.evt',f'{target_path}/spec/wt/{eventname}_bg.pi',region_back)
                                wtevtsrc.write(f'{target_path}/evt/source/wt/{eventname}_src.evt\n')
                                wtevtbg.write(f'{target_path}/evt/bg/wt/{eventname}_bg.evt\n')

class mkspec:
        """
        Use XSELECT to create spectrum.
        Args:
                evtfile: Input event file.
                srcregion: Region of source.
                srcout: Spectrum output file of the source.
                bgregion: Region of background.
                bgout: Spectrum output file of the background.
        """
        def __init__(self,evtfile,srcregion,srcout,bgregion,bgout):
                xco_spec=f'xsel\n\
                        no\n\
                        read eve {evtfile}\n\
                        ./\n\
                        yes\n\
                        filter region {srcregion}\n\
                        extract spec\n\
                        save spec {srcout}\n\
                        yes\n\
                        clear region\n\
                        filter region {bgregion}\n\
                        extract spec\n\
                        save spec {bgout}\n\
                        yes\n\
                        quit\n\
                        no'
                xcofile_spec='mklc.xco'
                with open(xcofile_spec,"w") as tmp:
                        lines = [line.lstrip() for line in xco_spec.split('\n')]
                        new_xco_spec = '\n'.join(lines)
                        tmp.write(new_xco_spec)
                sbp.run(['bash','-c', f'source {heasoft_env} && \
                        xselect @{xcofile_spec}'],shell=False,stdout=sbp.PIPE)
                os.remove(xcofile_spec)  
              
class mkarf:
        """
        Generates an  ARF file for an input PHA  file.
        Args:
                expofile: Name of the input exposure file.
                phafile: Name of the input PHA FITS file.
                outfile: Name of the output ARF FITS file.
        """
        def __init__(self,expofile,phafile,outfile):
                sbp.run(['bash','-c', f'source {heasoft_env} && xrtmkarf \
                        expofile={expofile} \
                        phafile={phafile} \
                        srcx=-1 srcy=-1 \
                        outfile={outfile} \
                        psfflag=yes clobber=yes'],shell=False,stdout=sbp.PIPE)
class grppha:
        """
        Manipulates OGIP standard PHA FITS file.
        Args:
                srcpha: Input PHA FITS file.
                outfile: Output PHA FITS file.
                bgpha: Background PHA FITS file.
                arffile: ARF file.
                rmffile: RMF file.
        """
        def __init__(self,srcpha,outfile,bgpha,arffile,rmffile):
                sbp.run(['bash','-c', f'source {heasoft_env} && grppha \
                        infile={srcpha} \
                        outfile=!{outfile} \
                        comm="bad 0-29" \
                        tempc="exit"'],shell=False,stdout=sbp.PIPE)
                sbp.run(['bash','-c', f'source {heasoft_env} && grppha\
                        infile={outfile} \
                        outfile=!{outfile} \
                        comm="group min 1" \
                        tempc="exit"'],shell=False,stdout=sbp.PIPE)
                sbp.run(['bash','-c', f'source {heasoft_env} && grppha \
                        infile={outfile} \
                        outfile=!{outfile} \
                        comm="chkey backfile {bgpha}" \
                        tempc="exit"'],shell=False,stdout=sbp.PIPE)
                sbp.run(['bash','-c', f'source {heasoft_env} && grppha \
                        infile={outfile} \
                        outfile=!{outfile} \
                        comm="chkey ancrfile {arffile}" \
                        tempc="exit"'],shell=False,stdout=sbp.PIPE)
                sbp.run(['bash','-c', f'source {heasoft_env} && grppha \
                        infile={outfile} \
                        outfile=!{outfile} \
                        comm="chkey respfile {rmffile}" \
                        tempc="exit"'],shell=False,stdout=sbp.PIPE)
class arfmerge:
        """
        Add input ARF files with specified weights.
        Args:
                arflist: List of ARF files.
                outfile: The name of the ARF file to be created.
        """
        def __init__(self,arflist,outfile):
                sbp.run(['bash','-c', f'source {heasoft_env} && addarf @{arflist} out_ARF={outfile} clobber=yes'],shell=False,stdout=sbp.PIPE)


class evtmerge:
        """
        Merge event files.
        Args:
                evtlist: List of event files.
                eventsout: Output event file.
        """
        def __init__(self,evtlist,eventsout):
                sbp.run(['bash','-c', f'source {heasoft_env} &&  \
                        extractor adjustgti=yes gti=GTI clobber=yes copyall=yes filename=@{evtlist} eventsout={eventsout} imgfile=NONE phafile=NONE regionfile=NONE timefile=NONE fitsbinlc=NONE xcolf=X ycolf=Y tcol=TIME'],shell=False,stdout=sbp.PIPE)

class backscal:
        """
        BACKSCAL is defined as the ratio of the detector area from which the source is extracted to the total detector area.
        Args:
                phafile: Input of pha file.
        """
        def __new__(self,phafile):
                data=fits.open(phafile)
                data_header=data[1].header
                back=float(data_header['BACKSCAL'])
                return back

class MakeSpecturm:
        """
        Make spectrum files for the events that need to be processed.
        Args:
                target_path: Target file path.
        """
        def __init__(self,target_path):

                pcevtsrclist=f'{target_path}/pcevtsrc.list'
                wtevtsrclist=f'{target_path}/wtevtsrc.list'
                pcevtbglist=f'{target_path}/pcevtbg.list'
                wtevtbglist=f'{target_path}/wtevtbg.list'
                evtmerge(pcevtsrclist,f'{target_path}/evt/source/pcsource.evt')
                evtmerge(wtevtsrclist,f'{target_path}/evt/source/wtsource.evt')
                evtmerge(pcevtbglist,f'{target_path}/evt/bg/pcbg.evt')
                evtmerge(wtevtbglist,f'{target_path}/evt/bg/wtbg.evt')
                
                extractpha(f'{target_path}/evt/source/pcsource.evt',f'{target_path}/spec/merge/pcsource.pi','None')
                extractpha(f'{target_path}/evt/source/wtsource.evt',f'{target_path}/spec/merge/wtsource.pi','None')
                extractpha(f'{target_path}/evt/bg/pcbg.evt',f'{target_path}/spec/merge/pcbg.pi','None')
                extractpha(f'{target_path}/evt/bg/wtbg.evt',f'{target_path}/spec/merge/wtbg.pi','None')

                pc_counts=[]
                wt_counts=[]
                processeventlist=open(f'{target_path}/procevt.list',"r").readlines()
                for i in processeventlist:
                        eventname=i[:i.index("\n")]
                        log=open(f'{target_path}/log/cpslog/{eventname}_cps.log',"r").read()
                        match = re.search(r"Grand Total\s+Good\s+Bad.*\n\s+\d+\s+(\d+)", log)
                        good_value=int(match.group(1))
                        if "pc" in eventname:
                                expofile=f'{target_path}/expmap/pc/{eventname}_ex.img'
                                phafile=f'{target_path}/spec/pc/{eventname}_src.pi'
                                outfile=f'{target_path}/spec/pc/{eventname}_src.arf'
                                pc_counts.append(good_value)
                        if "wt" in eventname:
                                expofile=f'{target_path}/expmap/wt/{eventname}_ex.img'
                                phafile=f'{target_path}/spec/wt/{eventname}_src.pi'
                                outfile=f'{target_path}/spec/wt/{eventname}_src.arf'
                                wt_counts.append(good_value)
                        #mkarf(expofile,phafile,outfile)
                pcarflist=open(f'{target_path}/pc_arf.list',"w")
                wtarflist=open(f'{target_path}/wt_arf.list',"w")
                pcbackscal=[]
                wtbackscal=[]
                for i in processeventlist:
                        eventname=i[:i.index("\n")]
                        log=open(f'{target_path}/log/cpslog/{eventname}_cps.log',"r").read()
                        match = re.search(r"Grand Total\s+Good\s+Bad.*\n\s+\d+\s+(\d+)", log)
                        good_value=int(match.group(1))
                        if "pc" in eventname:
                                pcarflist.write(f'{target_path}/spec/pc/{eventname}_src.arf {good_value/sum(pc_counts)}\n')
                                pcbackscal.append(backscal(f'{target_path}/spec/pc/{eventname}_src.pi') * float(good_value/sum(pc_counts)))
                        if "wt" in eventname:
                                wtarflist.write(f'{target_path}/spec/wt/{eventname}_src.arf {good_value/sum(wt_counts)}\n')
                                wtbackscal.append(backscal(f'{target_path}/spec/wt/{eventname}_src.pi') * float(good_value/sum(pc_counts)))
                pctotal=fits.open(f'{target_path}/spec/merge/pcsource.pi')
                pcheader=pctotal[1].header
                pcheader['BACKSCAL']=sum(pcbackscal)
                pctotal.writeto(f'{target_path}/spec/merge/pcsource.pi',overwrite=True)
                wttotal=fits.open(f'{target_path}/spec/merge/wtsource.pi')
                wtheader=wttotal[1].header
                wtheader['BACKSCAL']=sum(wtbackscal)
                wttotal.writeto(f'{target_path}/spec/merge/wtsource.pi',overwrite=True)
                arfmerge(f'{target_path}/pc_arf.list',f'{target_path}/spec/merge/pcsource.arf')
                arfmerge(f'{target_path}/wt_arf.list',f'{target_path}/spec/merge/wtsource.arf')
                shutil.copy(rmfinfo(f'{target_path}/spec/merge/pcsource.pi'),f'{target_path}/spec/merge/pcsource.rmf')
                shutil.copy(rmfinfo(f'{target_path}/spec/merge/wtsource.pi'),f'{target_path}/spec/merge/wtsource.rmf')
                
                grppha(f'{target_path}/spec/merge/pcsource.pi',f'{target_path}/spec/merge/pc.pi',f'{target_path}/spec/merge/pcbg.pi',f'{target_path}/spec/merge/pcsource.arf',f'{target_path}/spec/merge/pcsource.rmf')
                grppha(f'{target_path}/spec/merge/wtsource.pi',f'{target_path}/spec/merge/wt.pi',f'{target_path}/spec/merge/wtbg.pi',f'{target_path}/spec/merge/wtsource.arf',f'{target_path}/spec/merge/wtsource.rmf')


class mklc:
        """
        Use XSELECT to create light curve.
        Args:
                evtfile: Input event file.
                srcregion: Region of source.
                srcout: Light curve output file of the source.
                bgregion: Region of background.
                bgout: Light curve output file of the background.
                erange: The energy range of the light curve (the numbers are in terms of channels, each of which is 10 eV)
                binsize: The bin size of light curve (s).
        """
        def __init__(self,evtfile,srcregion,srcout,bgregion,bgout,erange,binsize):
                xco_lc=f'xsel\n\
                        no\n\
                        read eve {evtfile}\n\
                        ./\n\
                        yes\n\
                        filter region {srcregion}\n\
                        filter pha_cutoff {erange}\n\
                        set binsize {binsize}\n\
                        extract curve\n\
                        save curve {srcout}\n\
                        yes\n\
                        clear region\n\
                        filter region {bgregion}\n\
                        extract curve\n\
                        save curve {bgout}\n\
                        yes\n\
                        quit\n\
                        no'
                xcofile_lc='mklc.xco'
                with open(xcofile_lc,"w") as tmp:
                        lines = [line.lstrip() for line in xco_lc.split('\n')]
                        new_xco_lc = '\n'.join(lines)
                        tmp.write(new_xco_lc)
                sbp.run(['bash','-c', f'source {heasoft_env} && \
                        xselect @{xcofile_lc}'],shell=False,stdout=sbp.PIPE)
                os.remove(xcofile_lc)                

class lccorr:
        """
        Correct the light curve.
        Args:
                lcfile: Name of the input Light Curve FITS file or NONE to read region from regionfile.
                outfile: Name of the Corrected Light Curve.
                corrfile: Correction factor output file.
                attfile: Input Attitude FITS file.
                outinstrfile: Output Instrument Map File.
                infile: Input Event FITS file.
                hdfile: Input Housekeeping Header Packets FITS file.
        """
        def __init__(self,regionfile,lcfile,outfile,corrfile,attfile,outinstrfile,infile,hdfile):
                os.environ['HEADASPROMPT']='/dev/null'
                sbp.run(['bash','-c', f'source {heasoft_env} && \
                        xrtlccorr clobber=yes pcnframe=4 wtnframe=10 regionfile={regionfile} \
                        lcfile={lcfile}\
                        outfile={outfile} \
                        corrfile={corrfile} \
                        attfile={attfile} \
                        outinstrfile={outinstrfile} \
                        infile={infile}\
                        hdfile={hdfile}'],shell=False,stdout=sbp.PIPE)                

class lcsub:
        """
        Merging of source and background light curves.
        Args:
                srcregion: Region of source.
                bgregion: Region of background.
                srclc: Light curve of source.
                bglc: Light curve of background.
                outlc: Output the combined light curve.
        """
        def __init__(self,srcregion,bgregion,srclc,bglc,outlc):
                region_src=srcregion
                region_back=bgregion
                with open(region_src,"r") as tmp:
                        src_data=tmp.read()
                        match_data_src= re.findall(r'[-+]?\d*\.\d+|\d+', src_data)
                        if "circle" in src_data:
                                src_radius=float(match_data_src[-1])
                                src_area=src_radius**2
                                print(f'circle: {src_radius}')
                        if "annulus" in src_data:
                                src_radius_inner=float(match_data_src[-2])
                                src_radius_outer=float(match_data_src[-1])
                                src_area=(src_radius_outer**2-src_radius_inner**2)
                                print(f'annulus: {src_radius_inner,src_radius_outer}')
                        if "box" in src_data and "-" not in src_data:
                                src_width=float(match_data_src[-3])
                                src_length=float(match_data_src[-2])
                                src_area=src_width*src_length
                                print(f'box: {src_width,src_length}')
                        if "box" in src_data and "-" in src_data:
                                src_inner_width=float(match_data_src[-3])
                                src_inner_length=float(match_data_src[-2])
                                src_outer_width=float(match_data_src[2])
                                src_outer_length=float(match_data_src[3])
                                src_area=src_outer_width*src_outer_length-src_inner_width*src_inner_length
                                print(f'box-box: {src_inner_width,src_inner_length,src_outer_width,src_outer_length}')
                with open(region_back,"r") as tmp:
                        back_data=tmp.read()
                        match_data_back= re.findall(r'[-+]?\d*\.\d+|\d+', back_data)
                        if "circle" in back_data:
                                back_radius=float(match_data_back[-1])
                                back_area=back_radius**2
                                print(f'circle: {back_radius}')
                        if "annulus" in back_data:
                                back_radius_inner=float(match_data_back[-2])
                                back_radius_outer=float(match_data_back[-1])
                                back_area=(back_radius_outer**2-back_radius_inner**2)
                                print(f'annulus: {back_radius_inner,back_radius_outer}')
                        if "box" in back_data and "-" not in back_data:
                                back_width=float(match_data_back[-3])
                                back_length=float(match_data_back[-2])
                                back_area=back_width*back_length
                                print(f'box: {back_width,back_length}')
                        if "box" in back_data and "-" in back_data:
                                back_inner_width=float(match_data_back[-3])
                                back_inner_length=float(match_data_back[-2])
                                back_outer_width=float(match_data_back[2])
                                back_outer_length=float(match_data_back[3])
                                back_area=back_outer_width*back_outer_length-back_inner_width*back_inner_length
                                print(f'box-box: {back_inner_width,back_inner_length,back_outer_width,back_outer_length}')
                        scaling=src_area/back_area
                        print(f'Source area: {src_area}')
                        print(f'Background area: {back_area}')
                        print(f'Scaling: {scaling}\n')
                        sbp.run(['bash','-c', f'source {heasoft_env} && \
                                lcmath err_mode=2 infile={srclc} addsubr = no\
                                bgfile={bglc} outfile={outlc} multi=1 multb={scaling}'],shell=False,stdout=sbp.PIPE)


class LightCurve:
        """
        Make light curve files for the events that need to be processed.
        Args:
                target_path: Target file path.
        """
        def __init__(self,target_path):
                self.target_path=target_path
                processeventlist=open(f'{target_path}/procevt.list',"r")
                eventlist=processeventlist.readlines()
                self.eventlist=eventlist

        def make(self,pcbinsize='2.51',wtbinsize='0.5'):
                os.environ['HEADASPROMPT']='/dev/null'
                for erange in ["0.3-10","0.3-1.5","1.51-10"]:
                        if not os.path.exists(f'{self.target_path}/lc/{erange}'):
                                sbp.run(['mkdir',f'{self.target_path}/lc/{erange}'])
                        lower, upper = map(float, erange.split('-'))
                        elower=int(lower*100)
                        eupper=int(upper*100)
                        for i in self.eventlist:
                                eventname=i[:i.index("\n")]
                                obsid=eventname[:eventname.index("_")]
                                if "pc" in eventname:
                                        evtfile=f'{self.target_path}/evt/pc/{eventname}.evt'
                                        region_src=f'{self.target_path}/region/pc/{eventname}_src.reg'
                                        region_back=f'{self.target_path}/region/pc/{eventname}_back.reg'
                                        binsize=pcbinsize
                                if "wt" in eventname:
                                        evtfile=f'{self.target_path}/evt/wt/{eventname}.evt'
                                        region_src=f'{self.target_path}/region/wt/{eventname}_src.reg'
                                        region_back=f'{self.target_path}/region/wt/{eventname}_back.reg'
                                        binsize=wtbinsize
                                srcout=f'{self.target_path}/lc/{erange}/{eventname}_src.lc'
                                bgout=f'{self.target_path}/lc/{erange}/{eventname}_back.lc'
                                #make light curve for PC and WT in energy range of 0.3-10 keV, 0.3-1.5keV, 1.51-10keV
                                mklc(evtfile,region_src,srcout,region_back,bgout,f'{elower} {eupper}',binsize)
                                #make correct light curve for source
                                lcfile=f'{self.target_path}/lc/{erange}/{eventname}_src.lc'
                                outfile=f'{self.target_path}/lc/{erange}/{eventname}_src_corr.lc'
                                corrfile=f'{self.target_path}/lc/{erange}/{eventname}_src_corrfactor.fits'
                                attfile=f'{self.target_path}/data/{obsid}/auxil/sw{obsid}pat.fits.gz'
                                outinstrfile=f'{self.target_path}/lc/{erange}/{eventname}_srawinstr.img'
                                hdfile=f'{self.target_path}/data/{obsid}/xrt/hk/sw{obsid}xhd.hk.gz'
                                lccorr('none',lcfile,outfile,corrfile,attfile,outinstrfile,evtfile,hdfile)
                                #make sub of source and background
                                srclc=f'{self.target_path}/lc/{erange}/{eventname}_src_corr.lc'
                                bglc=f'{self.target_path}/lc/{erange}/{eventname}_back.lc'
                                outlc_corr=f'{self.target_path}/lc/{erange}/{eventname}_sub_corr.lc'
                                print(f'Energy Range: {erange} Event: {eventname}')
                                lcsub(region_src,region_back,srclc,bglc,outlc_corr)
        
        def data(self,pcbinsize='2.51',wtbinsize='0.5'):
                self.pcsrc_data_list={}
                self.pcbg_data_list={}
                self.wtsrc_data_list={}
                self.wtbg_data_list={}
                self.pc_combine={}
                self.wt_combine={}
                self.pc_bg_combine={}
                self.wt_bg_combine={}
                self.pcbinsize=pcbinsize
                self.wtbinsize=wtbinsize
                for erange in ["0.3-10","0.3-1.5","1.51-10"]:
                        self.pcsrc_data_list[f'{erange}']={}
                        self.pcbg_data_list[f'{erange}']={}
                        self.wtsrc_data_list[f'{erange}']={}
                        self.wtbg_data_list[f'{erange}']={}
                        #combine source data
                        self.pc_combine[f'{erange}']={}
                        self.wt_combine[f'{erange}']={}
                        self.pc_combine[f'{erange}']['time']=[]
                        self.pc_combine[f'{erange}']['rate']=[]
                        self.pc_combine[f'{erange}']['error']=[]
                        self.pc_combine[f'{erange}']['fracexp']=[]
                        self.wt_combine[f'{erange}']['time']=[]
                        self.wt_combine[f'{erange}']['rate']=[]
                        self.wt_combine[f'{erange}']['error']=[]
                        self.wt_combine[f'{erange}']['fracexp']=[]
                        #combine background data
                        self.pc_bg_combine[f'{erange}']={}
                        self.wt_bg_combine[f'{erange}']={}
                        self.pc_bg_combine[f'{erange}']['time']=[]
                        self.pc_bg_combine[f'{erange}']['rate']=[]
                        self.pc_bg_combine[f'{erange}']['error']=[]
                        self.pc_bg_combine[f'{erange}']['fracexp']=[]
                        self.wt_bg_combine[f'{erange}']['time']=[]
                        self.wt_bg_combine[f'{erange}']['rate']=[]
                        self.wt_bg_combine[f'{erange}']['error']=[]
                        self.wt_bg_combine[f'{erange}']['fracexp']=[]
                        for i in range(len(self.eventlist)):
                                eventname=self.eventlist[i][:self.eventlist[i].index("\n")]
                                lc_src_data,lc_src_header=fits.getdata(f'{self.target_path}/lc/{erange}/{eventname}_sub_corr.lc',header=True)
                                lc_bg_data,lc_bg_header=fits.getdata(f'{self.target_path}/lc/{erange}/{eventname}_back.lc',header=True)
                                #extract lc data and header
                                self.trigtime=687014249.005
                                if 'pc' in eventname:
                                        self.pcsrc_data_list[f'{erange}'][f'{eventname}']={}
                                        self.pcbg_data_list[f'{erange}'][f'{eventname}']={}
                                        self.pcsrc_data_list[f'{erange}'][f'{eventname}']['data']=lc_src_data
                                        self.pcbg_data_list[f'{erange}'][f'{eventname}']['data']=lc_bg_data
                                        self.pcsrc_data_list[f'{erange}'][f'{eventname}']['header']=lc_src_header
                                        self.pcbg_data_list[f'{erange}'][f'{eventname}']['header']=lc_bg_header
                                        self.pc_combine[f'{erange}']['time'].extend(self.pcsrc_data_list[f'{erange}'][f'{eventname}']['data']['TIME']+self.pcsrc_data_list[f'{erange}'][f'{eventname}']['header']['TSTART']-self.trigtime)
                                        self.pc_combine[f'{erange}']['rate'].extend(self.pcsrc_data_list[f'{erange}'][f'{eventname}']['data']['RATE'])
                                        self.pc_combine[f'{erange}']['error'].extend(self.pcsrc_data_list[f'{erange}'][f'{eventname}']['data']['ERROR'])
                                        self.pc_combine[f'{erange}']['fracexp'].extend(self.pcsrc_data_list[f'{erange}'][f'{eventname}']['data']['FRACEXP'])
                                        
                                        self.pc_bg_combine[f'{erange}']['time'].extend(self.pcbg_data_list[f'{erange}'][f'{eventname}']['data']['TIME']+self.pcbg_data_list[f'{erange}'][f'{eventname}']['header']['TSTART']-self.trigtime)
                                        self.pc_bg_combine[f'{erange}']['rate'].extend(self.pcbg_data_list[f'{erange}'][f'{eventname}']['data']['RATE'])
                                        self.pc_bg_combine[f'{erange}']['error'].extend(self.pcbg_data_list[f'{erange}'][f'{eventname}']['data']['ERROR'])
                                        self.pc_bg_combine[f'{erange}']['fracexp'].extend(self.pcbg_data_list[f'{erange}'][f'{eventname}']['data']['FRACEXP'])
                                if 'wt' in eventname:
                                        self.wtsrc_data_list[f'{erange}'][f'{eventname}']={}
                                        self.wtbg_data_list[f'{erange}'][f'{eventname}']={}
                                        self.wtsrc_data_list[f'{erange}'][f'{eventname}']['data']=lc_src_data
                                        self.wtbg_data_list[f'{erange}'][f'{eventname}']['data']=lc_bg_data
                                        self.wtsrc_data_list[f'{erange}'][f'{eventname}']['header']=lc_src_header
                                        self.wtbg_data_list[f'{erange}'][f'{eventname}']['header']=lc_bg_header
                                        self.wt_combine[f'{erange}']['time'].extend(self.wtsrc_data_list[f'{erange}'][f'{eventname}']['data']['TIME']+self.wtsrc_data_list[f'{erange}'][f'{eventname}']['header']['TSTART']-self.trigtime)
                                        self.wt_combine[f'{erange}']['rate'].extend(self.wtsrc_data_list[f'{erange}'][f'{eventname}']['data']['RATE'])
                                        self.wt_combine[f'{erange}']['error'].extend(self.wtsrc_data_list[f'{erange}'][f'{eventname}']['data']['ERROR'])
                                        self.wt_combine[f'{erange}']['fracexp'].extend(self.wtsrc_data_list[f'{erange}'][f'{eventname}']['data']['FRACEXP'])
                                        
                                        self.wt_bg_combine[f'{erange}']['time'].extend(self.wtbg_data_list[f'{erange}'][f'{eventname}']['data']['TIME']+self.wtbg_data_list[f'{erange}'][f'{eventname}']['header']['TSTART']-self.trigtime)
                                        self.wt_bg_combine[f'{erange}']['rate'].extend(self.wtbg_data_list[f'{erange}'][f'{eventname}']['data']['RATE'])
                                        self.wt_bg_combine[f'{erange}']['error'].extend(self.wtbg_data_list[f'{erange}'][f'{eventname}']['data']['ERROR'])
                                        self.wt_bg_combine[f'{erange}']['fracexp'].extend(self.wtbg_data_list[f'{erange}'][f'{eventname}']['data']['FRACEXP'])

        @property
        def save_data(self):
                self.data()
                for erange in ["0.3-10","0.3-1.5","1.51-10"]:
                        sbp.run(['mkdir',f'{self.target_path}/lc/{erange}/merge/'])
                        pcsrc_combine_table=pd.DataFrame({'TIME':self.pc_combine[f'{erange}']['time'],
                                                          'RATE':self.pc_combine[f'{erange}']['rate'],
                                                          'ERROR':self.pc_combine[f'{erange}']['error'],
                                                          'FRACEXP':self.pc_combine[f'{erange}']['fracexp']})
                        wtsrc_combine_table=pd.DataFrame({'TIME':self.wt_combine[f'{erange}']['time'],
                                                          'RATE':self.wt_combine[f'{erange}']['rate'],
                                                          'ERROR':self.wt_combine[f'{erange}']['error'],
                                                          'FRACEXP':self.wt_combine[f'{erange}']['fracexp']})
                        pcbg_combine_table=pd.DataFrame({'TIME':self.pc_bg_combine[f'{erange}']['time'],
                                                          'RATE':self.pc_bg_combine[f'{erange}']['rate'],
                                                          'ERROR':self.pc_bg_combine[f'{erange}']['error'],
                                                          'FRACEXP':self.pc_bg_combine[f'{erange}']['fracexp']})
                        wtbg_combine_table=pd.DataFrame({'TIME':self.wt_bg_combine[f'{erange}']['time'],
                                                          'RATE':self.wt_bg_combine[f'{erange}']['rate'],
                                                          'ERROR':self.wt_bg_combine[f'{erange}']['error'],
                                                          'FRACEXP':self.wt_bg_combine[f'{erange}']['fracexp']})
                        pcsrc_combine_table.to_csv(f'{self.target_path}/lc/{erange}/merge/pc_src.csv',index=False)
                        wtsrc_combine_table.to_csv(f'{self.target_path}/lc/{erange}/merge/wt_src.csv',index=False)
                        pcbg_combine_table.to_csv(f'{self.target_path}/lc/{erange}/merge/pc_bg.csv',index=False)
                        wtbg_combine_table.to_csv(f'{self.target_path}/lc/{erange}/merge/wt_bg.csv',index=False)

        @property
        def save_lcplot(self):
                self.data()
                for erange in ["0.3-10","0.3-1.5","1.51-10"]:
                        for i in range(len(self.eventlist)):
                                eventname=self.eventlist[i][:self.eventlist[i].index("\n")]
                                if 'pc' in eventname:
                                        tstart=self.pcsrc_data_list[f'{erange}'][f'{eventname}']['header']['TSTART']
                                        xerr=float(self.pcbinsize)/2
                                        time=self.pcsrc_data_list[f'{erange}'][f'{eventname}']['data']['TIME']+tstart-self.trigtime
                                        rate=self.pcsrc_data_list[f'{erange}'][f'{eventname}']['data']['RATE']
                                        yerr=self.pcsrc_data_list[f'{erange}'][f'{eventname}']['data']['ERROR']
                                if 'wt' in eventname:
                                        tstart=self.wtsrc_data_list[f'{erange}'][f'{eventname}']['header']['TSTART']
                                        xerr=float(self.wtbinsize)/2
                                        time=self.wtsrc_data_list[f'{erange}'][f'{eventname}']['data']['TIME']+tstart-self.trigtime                                                                                                           
                                        rate=self.wtsrc_data_list[f'{erange}'][f'{eventname}']['data']['RATE']
                                        yerr=self.wtsrc_data_list[f'{erange}'][f'{eventname}']['data']['ERROR']
                                fig,axes = plt.subplots(figsize=(12,5))
                                axes.minorticks_on()
                                axes.tick_params(axis='both',which='major',direction='in',width=1,length=5)
                                axes.tick_params(axis='both',which='minor',direction='in',width=1,length=3)
                                axes.xaxis.set_minor_locator(MultipleLocator(10))
                                axes.xaxis.set_major_locator(MultipleLocator(100))
                                axes.errorbar(np.array(time)+xerr,rate,xerr=xerr,yerr=yerr,fmt='k+')
                                axes.set_title(f'Event: {eventname}\nEnergy range: {erange} keV\nTime binsize: PC:  {self.pcbinsize} s  WT: {self.wtbinsize} s\nSwift MET = {self.trigtime}',fontsize=14)
                                axes.set_ylabel(f'Count Rate ({erange})(/s)',fontsize=14)
                                axes.set_xlabel('Time since BAT trigger (s)',fontsize=14)
                                fig.tight_layout()
                                fig.savefig(f'{self.target_path}/lc/{erange}/image/{eventname}_lc.png')
                                plt.close()
        
        @property
        def plot_combine(self):
                self.data()
                for erange in ["0.3-10","0.3-1.5","1.51-10"]:
                        fig,axes = plt.subplots(figsize=(12,8))
                        axes.minorticks_on()
                        axes.tick_params(axis='both',which='major',direction='in',width=1,length=5)
                        axes.tick_params(axis='both',which='minor',direction='in',width=1,length=3)
                        axes.errorbar(np.array(self.wt_combine[f'{erange}']['time'])+float(self.wtbinsize)/2,self.wt_combine[f'{erange}']['rate'],xerr=float(self.wtbinsize)/2,yerr=self.wt_combine[f'{erange}']['error'],fmt='b+',label='WT source')
                        axes.errorbar(np.array(self.pc_combine[f'{erange}']['time'])+float(self.pcbinsize)/2,self.pc_combine[f'{erange}']['rate'],xerr=float(self.pcbinsize)/2,yerr=self.pc_combine[f'{erange}']['error'],fmt='r+',label='PC source')
                        axes.set(xscale='log',yscale='log')
                        axes.set_title(f'Combination of PC data and WT data\nEnergy range: {erange} keV\nTime binsize: PC: {self.pcbinsize} s  WT: {self.wtbinsize} s\nSwift MET = {self.trigtime}',fontsize=14)
                        axes.set_ylabel(f'Count Rate ({erange} keV)(/s)',fontsize=14)
                        axes.set_xlabel('Time since BAT trigger (s)',fontsize=14)
                        axes.legend(frameon=False)
                        fig.tight_layout()
                        fig.savefig(f'{self.target_path}/lc/{erange}/image/combine_lc.png')
        
        @property
        def plot_detail(self):
                self.data()
                for erange in ["0.3-10","0.3-1.5","1.51-10"]:
                        fig,(axe1,axe2) = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios': [2, 1]},figsize=(12,8))
                        axe1.minorticks_on()
                        axe1.tick_params(axis='both',which='major',direction='in',width=1,length=5)
                        axe1.tick_params(axis='both',which='minor',direction='in',width=1,length=3)
                        axe1.errorbar(np.array(self.wt_combine[f'{erange}']['time'])+float(self.wtbinsize)/2,self.wt_combine[f'{erange}']['rate'],xerr=float(self.wtbinsize)/2,yerr=self.wt_combine[f'{erange}']['error'],fmt='b+',label='WT source')
                        axe1.errorbar(np.array(self.pc_combine[f'{erange}']['time'])+float(self.pcbinsize)/2,self.pc_combine[f'{erange}']['rate'],xerr=float(self.pcbinsize)/2,yerr=self.pc_combine[f'{erange}']['error'],fmt='r+',label='PC source')
                        axe1.errorbar(np.array(self.wt_bg_combine[f'{erange}']['time'])+float(self.wtbinsize)/2,self.wt_bg_combine[f'{erange}']['rate'],xerr=float(self.wtbinsize)/2,yerr=self.wt_bg_combine[f'{erange}']['error'],fmt='+',color='purple',label='WT background')
                        axe1.errorbar(np.array(self.pc_bg_combine[f'{erange}']['time'])+float(self.pcbinsize)/2,self.pc_bg_combine[f'{erange}']['rate'],xerr=float(self.pcbinsize)/2,yerr=self.pc_bg_combine[f'{erange}']['error'],fmt='g+',label='PC background')
                        axe1.set(xscale='log',yscale='log')
                        axe1.legend(frameon=False)
                        axe1.set_title(f'Combination of PC data and WT data\nEnergy range: {erange} keV\nTime binsize: PC: {self.pcbinsize} s  WT: {self.wtbinsize} s\nSwift MET = {self.trigtime}',fontsize=14)
                        axe1.set_ylabel(f'Count Rate ({erange} keV)(/s)',fontsize=14)
                        axe2.tick_params(axis='both',which='major',direction='in',width=1,length=5)
                        axe2.tick_params(axis='both',which='minor',direction='in',width=1,length=3)
                        axe2.tick_params(axis='x', which='both', bottom=True, top=True)
                        axe2.scatter(np.array(self.wt_combine[f'{erange}']['time'])+float(self.wtbinsize)/2,self.wt_combine[f'{erange}']['fracexp'],marker='.',color='b',label='WT source')
                        axe2.scatter(np.array(self.pc_combine[f'{erange}']['time'])+float(self.pcbinsize)/2,self.pc_combine[f'{erange}']['fracexp'],marker='.',color='r',label='PC source')
                        axe2.set_xlabel('Time since BAT trigger (s)',fontsize=14)
                        axe2.set_ylabel('Frac Exp',fontsize=14)
                        axe2.legend(frameon=False)
                        fig.tight_layout()
                        fig.subplots_adjust(hspace=0)
                        fig.savefig(f'{self.target_path}/lc/{erange}/image/detail_lc.png')

        @property
        def plot_hr(self):
                self.data()
                fig,(axe1,axe2,axe3) = plt.subplots(3,1,sharex=True,gridspec_kw={'height_ratios': [1 , 1 , 1]},figsize=(12,8))
                wt_hr_data=np.array(self.wt_combine['1.51-10']['rate'])/np.array(self.wt_combine['0.3-1.5']['rate'])
                pc_hr_data=np.array(self.pc_combine['1.51-10']['rate'])/np.array(self.pc_combine['0.3-1.5']['rate'])
                wt_hr_err=np.array(self.wt_combine['1.51-10']['error'])/np.array(self.wt_combine['0.3-1.5']['error'])
                pc_hr_err=np.array(self.pc_combine['1.51-10']['error'])/np.array(self.pc_combine['0.3-1.5']['error'])
                axe1.tick_params(axis='both',which='major',direction='in',width=1,length=5)
                axe1.tick_params(axis='both',which='minor',direction='in',width=1,length=3)
                axe1.errorbar(np.array(self.wt_combine['1.51-10']['time'])+float(self.wtbinsize)/2,self.wt_combine['1.51-10']['rate'],xerr=float(self.wtbinsize)/2,yerr=self.wt_combine['1.51-10']['error'],fmt='b+',label='WT source')
                axe1.errorbar(np.array(self.pc_combine['1.51-10']['time'])+float(self.pcbinsize)/2,self.pc_combine['1.51-10']['rate'],xerr=float(self.pcbinsize)/2,yerr=self.pc_combine['1.51-10']['error'],fmt='r+',label='PC source')
                axe1.set(xscale='log',yscale='log')
                axe1.legend(frameon=False)
                axe1.set_title('Swift/XRT data HR',fontsize=14)
                axe1.set_ylabel('1.51-10 keV c/s',fontsize=14)
                axe2.tick_params(axis='both',which='major',direction='in',width=1,length=5)
                axe2.tick_params(axis='both',which='minor',direction='in',width=1,length=3)
                axe2.errorbar(np.array(self.wt_combine['0.3-1.5']['time'])+float(self.wtbinsize)/2,self.wt_combine['0.3-1.5']['rate'],xerr=float(self.wtbinsize)/2,yerr=self.wt_combine['0.3-1.5']['error'],fmt='b+')
                axe2.errorbar(np.array(self.pc_combine['0.3-1.5']['time'])+float(self.pcbinsize)/2,self.pc_combine['0.3-1.5']['rate'],xerr=float(self.pcbinsize)/2,yerr=self.pc_combine['0.3-1.5']['error'],fmt='r+')
                axe2.set(xscale='log',yscale='log')
                axe2.set_ylabel('0.3-1.51 keV c/s',fontsize=14)
                axe3.tick_params(axis='both',which='major',direction='in',width=1,length=5)
                axe3.tick_params(axis='both',which='minor',direction='in',width=1,length=3)
                axe3.tick_params(axis='x', which='both', bottom=True, top=True)
                axe3.errorbar(np.array(self.wt_combine['0.3-1.5']['time'])+float(self.wtbinsize)/2,wt_hr_data,xerr=float(self.wtbinsize)/2,yerr=wt_hr_err,fmt='b+')
                axe3.errorbar(np.array(self.pc_combine['0.3-1.5']['time'])+float(self.pcbinsize)/2,pc_hr_data,xerr=float(self.pcbinsize)/2,yerr=pc_hr_err,fmt='r+')
                axe3.set(xscale='log',yscale='log')
                axe3.set_ylabel('Hard/Soft',fontsize=14)
                axe3.set_xlabel('Time since BAT trigger (s)',fontsize=14)
                fig.tight_layout()
                fig.subplots_adjust(hspace=0)
                fig.savefig(f'{self.target_path}/lc/HR.png')
        
        def rebin(self, wtbincounts='30', pcbincounts='20', binfactor='1.5', ratefactor='10',snr='1.5',sigma='3'):
                self.data()
                for erange in ["0.3-10"]:
                        if not os.path.exists(f'{self.target_path}/lc/rebin'):
                                sbp.run(['mkdir',f'{self.target_path}/lc/rebin'])
                        if not os.path.exists(f'{self.target_path}/lc/rebin/{erange}'):
                                sbp.run(['mkdir',f'{self.target_path}/lc/rebin/{erange}'])
                        for i in range(len(self.eventlist)):
                                eventname=self.eventlist[i][:self.eventlist[i].index("\n")]
                                for mode in ["pc","wt"]:
                                        if mode in eventname:
                                                if mode == 'pc':
                                                        srcdata=self.pcsrc_data_list[f'{erange}'][f'{eventname}']['data']
                                                        srcheader=self.pcsrc_data_list[f'{erange}'][f'{eventname}']['header']
                                                        lcdata,lcheader=fits.getdata(f'{self.target_path}/lc/{erange}/{eventname}_src.lc',header=True)
                                                        bgdata,bgheader=fits.getdata(f'{self.target_path}/lc/{erange}/{eventname}_back.lc',header=True)
                                                        bincounts=float(pcbincounts)
                                                if mode == 'wt':
                                                        srcdata=self.wtsrc_data_list[f'{erange}'][f'{eventname}']['data']
                                                        srcheader=self.wtsrc_data_list[f'{erange}'][f'{eventname}']['header']
                                                        lcdata,lcheader=fits.getdata(f'{self.target_path}/lc/{erange}/{eventname}_src.lc',header=True)
                                                        bgdata,bgheader=fits.getdata(f'{self.target_path}/lc/{erange}/{eventname}_back.lc',header=True)
                                                        bincounts=float(wtbincounts)
                                #counts=CR*(TSTOP-TSTART) counts_err=CR_ERR*(TSTOP-TSTART)
                                srcstand=pd.DataFrame({'TSTART':np.array(srcdata['TIME'][0:-1])+srcheader['TSTART']-self.trigtime,
                                                        'TSTOP':np.array(srcdata['TIME'][1:])+srcheader['TSTART']-self.trigtime,
                                                        'COUNTS':(np.array(srcdata['TIME'][1:])-np.array(srcdata['TIME'][0:-1]))*np.array(srcdata['RATE'][0:-1]),
                                                        'COUNTS_ERR':(np.array(srcdata['TIME'][1:])-np.array(srcdata['TIME'][0:-1]))*np.array(srcdata['ERROR'][0:-1]),
                                                        'FRACEXP':np.array(srcdata['FRACEXP'][0:-1])})
                                srcstand.to_csv(f'{self.target_path}/lc/rebin/raw/{eventname}_raw.csv')

                                rawstand=pd.DataFrame({'TSTART':np.array(lcdata['TIME'][0:-1])+lcheader['TSTART']-self.trigtime,
                                                        'TSTOP':np.array(lcdata['TIME'][1:])+lcheader['TSTART']-self.trigtime,
                                                        'COUNTS':(np.array(lcdata['TIME'][1:])-np.array(lcdata['TIME'][0:-1]))*np.array(lcdata['RATE'][0:-1]),
                                                        'COUNTS_ERR':(np.array(lcdata['TIME'][1:])-np.array(lcdata['TIME'][0:-1]))*np.array(lcdata['ERROR'][0:-1]),
                                                        'FRACEXP':np.array(lcdata['FRACEXP'][0:-1])})
                                
                                bgstand=pd.DataFrame({'TSTART':np.array(bgdata['TIME'][0:-1])+bgheader['TSTART']-self.trigtime,
                                                        'TSTOP':np.array(bgdata['TIME'][1:])+bgheader['TSTART']-self.trigtime,
                                                        'COUNTS':(np.array(bgdata['TIME'][1:])-np.array(bgdata['TIME'][0:-1]))*np.array(bgdata['RATE'][0:-1]),
                                                        'COUNTS_ERR':(np.array(bgdata['TIME'][1:])-np.array(bgdata['TIME'][0:-1]))*np.array(bgdata['ERROR'][0:-1]),
                                                        'FRACEXP':np.array(bgdata['FRACEXP'][0:-1])})
                                
                                #Initial parameters
                                X=0 #minnum counts in the bin on rate is 1 counts/s
                                X_raw=0
                                X_bg=0
                                binfactor=float(binfactor) #bin multiplication factor
                                ratefactor=float(ratefactor) #rate multiplication factor
                                snr=float(snr)
                                sigma=float(sigma)
                                #Data production
                                subscript_pos=[]
                                rebin_counts=[]
                                rebin_err=[]
                                rebin_tstop=[]
                                rebin_tstart=[srcstand['TSTART'][0]]
                                rebin_fracexp=[]
                                #tem parameters
                                src_counts=srcstand['COUNTS']
                                raw_counts=rawstand['COUNTS']
                                bg_counts=bgstand['COUNTS']
                                src_err=[]
                                bg_err=[]
                                expotime=0
                                timelist=[]
                                for j in range(len(src_counts)-1):
                                        X_raw+=raw_counts[j]
                                        X+=src_counts[j]
                                        X_bg+=bg_counts[j]
                                        expotime+=srcstand['FRACEXP'][j]*(srcstand['TSTOP'][j]-srcstand['TSTART'][j])
                                        timelist.append(srcstand['TSTART'][j])
                                        src_err.append(srcstand['COUNTS_ERR'][j])
                                        bg_err.append(bgstand['COUNTS_ERR'][j])
                                        if bincounts / binfactor < 15:
                                                if X_raw >= 15:
                                                        bintime=srcstand['TSTOP'][j]-timelist[0]
                                                        rate=X/bintime
                                                        counts_err=np.sqrt(np.sum(np.array(src_err)**2))
                                                        bg_counts_err=np.sqrt(np.sum(np.array(bg_err)**2))
                                                        rate_err=counts_err/bintime
                                                        snr_value=rate / (rate_err/2)
                                                        sigma_value = (X_raw-X_bg)/(bg_counts_err*expotime/bintime)
                                                        if rate<1:
                                                                rebin_fracexp.append(expotime/bintime)
                                                                subscript_pos.append(j)
                                                                rebin_tstop.append(srcstand['TSTOP'][j])
                                                                rebin_counts.append(X)
                                                                rebin_err.append(rate_err)
                                                                X_raw=0
                                                                X=0
                                                                X_bg=0
                                                                expotime=0
                                                                timelist=[]
                                                                src_err=[]
                                                                bg_err=[]
                                                                continue
                                        if X_raw >= bincounts:
                                                bintime=srcstand['TSTOP'][j]-timelist[0]
                                                rate=X/bintime
                                                counts_err=np.sqrt(np.sum(np.array(src_err)**2))
                                                bg_counts_err=np.sqrt(np.sum(np.array(bg_err)**2))
                                                rate_err=counts_err/bintime
                                                snr_value=rate / (rate_err/2)
                                                sigma_value = (X_raw-X_bg)/(bg_counts_err*expotime/bintime)
                                                if rate>=1 and rate < 1*ratefactor:
                                                        rebin_fracexp.append(expotime/bintime)
                                                        subscript_pos.append(j)
                                                        rebin_tstop.append(srcstand['TSTOP'][j])
                                                        rebin_counts.append(X)
                                                        rebin_err.append(rate_err)
                                                        X_raw=0
                                                        X=0
                                                        X_bg=0
                                                        expotime=0
                                                        timelist=[]
                                                        src_err=[]
                                                        bg_err=[]
                                                        continue
                                        if X_raw >= bincounts*binfactor:
                                                bintime=srcstand['TSTOP'][j]-timelist[0]
                                                rate=X/bintime
                                                counts_err=np.sqrt(np.sum(np.array(src_err)**2))
                                                bg_counts_err=np.sqrt(np.sum(np.array(bg_err)**2))
                                                rate_err=counts_err/bintime
                                                snr_value=rate / (rate_err/2)
                                                sigma_value = (X_raw-X_bg)/(bg_counts_err*expotime/bintime)
                                                if rate>=1*ratefactor and rate < 1*ratefactor**2:
                                                        rebin_fracexp.append(expotime/bintime)
                                                        subscript_pos.append(j)
                                                        rebin_tstop.append(srcstand['TSTOP'][j])
                                                        rebin_counts.append(X)
                                                        rebin_err.append(rate_err)
                                                        X_raw=0
                                                        X=0
                                                        X_bg=0
                                                        expotime=0
                                                        timelist=[]
                                                        src_err=[]
                                                        bg_err=[]
                                                        continue
                                        if X_raw >= bincounts*binfactor**2:
                                                bintime=srcstand['TSTOP'][j]-timelist[0]
                                                rate=X/bintime
                                                counts_err=np.sqrt(np.sum(np.array(src_err)**2))
                                                bg_counts_err=np.sqrt(np.sum(np.array(bg_err)**2))
                                                rate_err=counts_err/bintime
                                                snr_value=rate / (rate_err/2)
                                                sigma_value = (X_raw-X_bg)/(bg_counts_err*expotime/bintime)
                                                if rate>=1*ratefactor**2:
                                                        rebin_fracexp.append(expotime/bintime)
                                                        subscript_pos.append(j)
                                                        rebin_tstop.append(srcstand['TSTOP'][j])
                                                        rebin_counts.append(X)
                                                        rebin_err.append(rate_err)
                                                        X_raw=0
                                                        X=0
                                                        X_bg=0
                                                        expotime=0
                                                        timelist=[]
                                                        src_err=[]
                                                        bg_err=[]
                                                        continue

                                if len(subscript_pos) == 0 :
                                        rebin_counts.append(sum(src_counts))
                                        rebin_tstop.append(srcstand['TSTOP'][len(srcstand)-1])
                                        exerr=srcstand['COUNTS_ERR']
                                        counts_err=np.sqrt(np.sum(np.array(exerr)**2))
                                        rebin_err.append(counts_err/(rebin_tstop[-1]-rebin_tstart[-1]))
                                        rebin_fracexp.append(sum(srcstand['FRACEXP']*(srcstand['TSTOP']-srcstand['TSTART']))/(rebin_tstop[-1]-rebin_tstart[-1]))
                                if len(subscript_pos) != 0 :
                                        for i in subscript_pos:
                                                rebin_tstart.append(srcstand['TSTART'][i+1])
                                        if subscript_pos[-1] != len(src_counts)-1:
                                                if sum(src_counts[subscript_pos[-1]+1:]) >= 15:
                                                        rebin_counts.append( sum(src_counts[subscript_pos[-1]+1:]) )
                                                        rebin_tstop.append(srcstand['TSTOP'][len(srcstand)-1])
                                                        exerr=srcstand['COUNTS_ERR'][subscript_pos[-1]+1:]
                                                        counts_err=np.sqrt(np.sum(np.array(exerr)**2))
                                                        rebin_err.append(counts_err/(rebin_tstop[-1]-rebin_tstart[-1]))
                                                        rebin_fracexp.append(sum(srcstand['FRACEXP'][subscript_pos[-1]+1:]*(srcstand['TSTOP'][subscript_pos[-1]+1:]-srcstand['TSTART'][subscript_pos[-1]+1:]))/(rebin_tstop[-1]-rebin_tstart[-1]))
                                                else:
                                                        if len(subscript_pos) != 1:
                                                                rebin_counts[-1]=rebin_counts[-1]+sum(src_counts[subscript_pos[-1]+1:])
                                                                rebin_tstart.pop()
                                                                rebin_tstop[-1]=np.array(srcstand['TSTOP'])[len(src_counts)-1]
                                                                exerr=srcstand['COUNTS_ERR'][subscript_pos[-2]+1:]
                                                                counts_err=np.sqrt(np.sum(np.array(exerr)**2))
                                                                rebin_err[-1]=counts_err/(rebin_tstop[-1]-rebin_tstart[-1])
                                                                rebin_fracexp[-1]=sum(srcstand['FRACEXP'][subscript_pos[-2]+1:]*(srcstand['TSTOP'][subscript_pos[-2]+1:]-srcstand['TSTART'][subscript_pos[-2]+1:]))/(rebin_tstop[-1]-rebin_tstart[-1])
                                                        if len(subscript_pos) == 1:
                                                                rebin_counts=sum(src_counts)
                                                                rebin_tstart.pop()
                                                                rebin_tstop=[srcstand['TSTOP'][len(srcstand)-1]]
                                                                exerr=srcstand['COUNTS_ERR']
                                                                counts_err=np.sqrt(np.sum(np.array(exerr)**2))
                                                                rebin_err=[counts_err/(rebin_tstop[-1]-rebin_tstart[-1])]
                                                                rebin_fracexp=sum(srcstand['FRACEXP']*(srcstand['TSTOP']-srcstand['TSTART']))/(rebin_tstop[-1]-rebin_tstart[-1])
                                #Save Rebin Data
                                rebintable=pd.DataFrame({'TIME':rebin_tstart+(np.array(rebin_tstop)-np.array(rebin_tstart))/2,
                                                         'TIME_NEG':[-x for x in (np.array(rebin_tstop)-np.array(rebin_tstart))/2],
                                                         'TIME_POS':(np.array(rebin_tstop)-np.array(rebin_tstart))/2,
                                                         'RATE':np.array(rebin_counts)/(np.array(rebin_tstop)-np.array(rebin_tstart)),
                                                         'RATE_NEG':[-x for x in np.array(rebin_err)/2],
                                                         'RATE_POS':np.array(rebin_err)/2,
                                                         'FRACEXP':rebin_fracexp})
                                rebintable.to_csv(f'{self.target_path}/lc/rebin/{erange}/{eventname}_rebin.csv',index=False)
        
        @property
        def rebin_data(self):
                self.pc_rebin_data={}
                self.wt_rebin_data={}
                self.pc_rebin_combine={}
                self.wt_rebin_combine={}
                for erange in ["0.3-10"]:
                        self.pc_rebin_data[f'{erange}']={}
                        self.wt_rebin_data[f'{erange}']={}
                        #combine source data for PC
                        self.pc_rebin_combine[f'{erange}']={}
                        self.pc_rebin_combine[f'{erange}']['TIME']=[]
                        self.pc_rebin_combine[f'{erange}']['TIME_NEG']=[]
                        self.pc_rebin_combine[f'{erange}']['TIME_POS']=[]
                        self.pc_rebin_combine[f'{erange}']['RATE']=[]
                        self.pc_rebin_combine[f'{erange}']['RATE_NEG']=[]
                        self.pc_rebin_combine[f'{erange}']['RATE_POS']=[]
                        self.pc_rebin_combine[f'{erange}']['FRACEXP']=[]
                        #combine source data for WT
                        self.wt_rebin_combine[f'{erange}']={}
                        self.wt_rebin_combine[f'{erange}']['TIME']=[]
                        self.wt_rebin_combine[f'{erange}']['TIME_NEG']=[]
                        self.wt_rebin_combine[f'{erange}']['TIME_POS']=[]
                        self.wt_rebin_combine[f'{erange}']['RATE']=[]
                        self.wt_rebin_combine[f'{erange}']['RATE_NEG']=[]
                        self.wt_rebin_combine[f'{erange}']['RATE_POS']=[]
                        self.wt_rebin_combine[f'{erange}']['FRACEXP']=[]
                        for i in range(len(self.eventlist)):
                                eventname=self.eventlist[i][:self.eventlist[i].index("\n")]
                                data=pd.read_csv(f'{self.target_path}/lc/rebin/{erange}/{eventname}_rebin.csv')
                                #extract lc data and header
                                if 'pc' in eventname:
                                        self.pc_rebin_data[f'{erange}'][f'{eventname}']={}
                                        self.pc_rebin_data[f'{erange}'][f'{eventname}']['data']=data
                                        self.pc_rebin_combine[f'{erange}']['TIME'].extend(data['TIME'])
                                        self.pc_rebin_combine[f'{erange}']['TIME_NEG'].extend(data['TIME_NEG'])
                                        self.pc_rebin_combine[f'{erange}']['TIME_POS'].extend(data['TIME_POS'])
                                        self.pc_rebin_combine[f'{erange}']['RATE'].extend(data['RATE'])
                                        self.pc_rebin_combine[f'{erange}']['RATE_NEG'].extend(data['RATE_NEG'])
                                        self.pc_rebin_combine[f'{erange}']['RATE_POS'].extend(data['RATE_POS'])
                                        self.pc_rebin_combine[f'{erange}']['FRACEXP'].extend(data['FRACEXP'])
                                if 'wt' in eventname:
                                        self.wt_rebin_data[f'{erange}'][f'{eventname}']={}
                                        self.wt_rebin_data[f'{erange}'][f'{eventname}']['data']=data
                                        self.wt_rebin_combine[f'{erange}']['TIME'].extend(data['TIME'])
                                        self.wt_rebin_combine[f'{erange}']['TIME_NEG'].extend(data['TIME_NEG'])
                                        self.wt_rebin_combine[f'{erange}']['TIME_POS'].extend(data['TIME_POS'])
                                        self.wt_rebin_combine[f'{erange}']['RATE'].extend(data['RATE'])
                                        self.wt_rebin_combine[f'{erange}']['RATE_NEG'].extend(data['RATE_NEG'])
                                        self.wt_rebin_combine[f'{erange}']['RATE_POS'].extend(data['RATE_POS'])
                                        self.wt_rebin_combine[f'{erange}']['FRACEXP'].extend(data['FRACEXP'])
        
        @property
        def rebin_combine(self):
                self.rebin_data
                self.data()
                for erange in ["0.3-10"]:
                        fig,axes = plt.subplots(figsize=(12,8))
                        axes.minorticks_on()
                        axes.tick_params(axis='both',which='major',direction='in',width=1,length=5)
                        axes.tick_params(axis='both',which='minor',direction='in',width=1,length=3)
                        pc_xerr=[abs(np.array(self.pc_rebin_combine[f'{erange}']['TIME_NEG'])),self.pc_rebin_combine[f'{erange}']['TIME_POS']]
                        wt_xerr=[abs(np.array(self.wt_rebin_combine[f'{erange}']['TIME_NEG'])),self.wt_rebin_combine[f'{erange}']['TIME_POS']]
                        pc_yerr=[abs(np.array(self.pc_rebin_combine[f'{erange}']['RATE_NEG'])),self.pc_rebin_combine[f'{erange}']['RATE_POS']]
                        wt_yerr=[abs(np.array(self.wt_rebin_combine[f'{erange}']['RATE_NEG'])),self.wt_rebin_combine[f'{erange}']['RATE_POS']]
                        axes.errorbar(self.wt_rebin_combine[f'{erange}']['TIME'],self.wt_rebin_combine[f'{erange}']['RATE'],xerr=wt_xerr,yerr=wt_yerr,fmt='b+',label='WT source')
                        axes.errorbar(self.pc_rebin_combine[f'{erange}']['TIME'],self.pc_rebin_combine[f'{erange}']['RATE'],xerr=pc_xerr,yerr=pc_yerr,fmt='r+',label='PC source')
                        axes.set(xscale='log',yscale='log')
                        axes.set_title(f'Combination of PC data and WT data of GRB 221009A\nEnergy range: {erange} keV\nSwift MET = {self.trigtime}\nWT Counts/bin:30 PC Counts/bin: 20\nWT min binsize: 0.5 s PC min binsize: 2.51 s\nRate factor: 10  Bin factor : 1.5',fontsize=14)
                        axes.set_ylabel(f'Count Rate ({erange} keV)(/s)',fontsize=14)
                        axes.set_xlabel('Time since BAT trigger (s)',fontsize=14)
                        axes.legend(frameon=False)
                        fig.tight_layout()
                        fig.savefig(f'{self.target_path}/lc/rebin/{erange}/rebin_combine_lc.png')
        
        def rebin_fit(self,breaktime):
                self.rebin_data
                erange="0.3-10"
                allrate=np.hstack((np.array(self.wt_rebin_combine[f'{erange}']['RATE']),np.array(self.pc_rebin_combine[f'{erange}']['RATE'])))
                alltime=np.hstack((np.array(self.wt_rebin_combine[f'{erange}']['TIME']),np.array(self.pc_rebin_combine[f'{erange}']['TIME'])))
                breaktime=[]
                breakrate=[]
                aftertime=[]
                afterrate=[]
                log_time=np.log(alltime)
                log_rate=np.log(allrate)
                for i in range(len(alltime)):
                        if alltime[i] >= np.e**breaktime:
                                breaktime.append(alltime[i])
                                breakrate.append(allrate[i])
                        else:
                                aftertime.append(alltime[i])
                                afterrate.append(allrate[i])
                
                log_aftertime=np.log(aftertime)
                log_afterrate=np.log(afterrate)
                log_breaktime=np.log(breaktime)
                log_breakrate=np.log(breakrate)
                slope1, intercept1, r_value1, p_value1, std_err1 = linregress(log_aftertime, log_afterrate)
                slope2, intercept2, r_value2, p_value2, std_err2 = linregress(log_breaktime, log_breakrate)
                fig,axes = plt.subplots(figsize=(12,8))
                axes.minorticks_on()
                axes.tick_params(axis='both',which='major',direction='in',width=1,length=5)
                axes.tick_params(axis='both',which='minor',direction='in',width=1,length=3)
                axes.plot(log_time,log_rate,'^',color='lightsteelblue')
                axes.plot(np.linspace(log_time[0],np.log(breaktime),1000), slope1 * np.linspace(log_time[0],np.log(breaktime),1000) + intercept1, 'k--', label=f"Fitted Line (slope: {slope1:.2f})")
                axes.plot(np.linspace(np.log(breaktime),log_time[-1],1000), slope2 * np.linspace(np.log(breaktime),log_time[-1],1000) + intercept2, 'r--',label=f"Fitted Line (slope: {slope2:.2f})")
                axes.set_title(f'Fit Light Curve',fontsize=14)
                axes.set_ylabel(f'Ln(Rate)',fontsize=14)
                axes.set_xlabel('Ln(Time)',fontsize=14)
                axes.legend(frameon=False)
                fig.tight_layout()
                fig.savefig(f'{self.target_path}/lcfit.png')





                        








                        




        


                

                
                



