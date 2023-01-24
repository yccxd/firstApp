#! /usr/bin/env python3

# hiihihi

##### image reduction pipeline version 221023
# Input file structure
#   LOT20XXXXXX/flat/		--> containing flat field and dark frame for flat reduction only
#              /bias-dark/	--> containing bias and dark frame
#              /BLKS/		--> all the fits file in this folder will be ignored
#
# Setup environment variables
basepath = ''		# relative path of the images
raw_path = basepath + 'RAW/'	# path includes RAW data
procpath = basepath + 'proc/'	# path for the calibrated output file
#
# Commands
#   >>> python3 <input_path> [-pdark] [-pflat] [-showls]
#     input_path can be a folder (e.g. LOT20210123) or an individual fits file
#     -pdark and -pflat are optional arguments which can assign the date of the dark/flat frame for this calibration task
#     -showls is another optional argument that only output the exposing parameters from fits header but do nothing about image reduction
#
# Output file
#   all the calibrated fits file will renamed as *_d.fit (with dark reduction), *_f.fit (with flat calibration), or *_df.fit for both
#   master dark/flat will also appeared in the output folder (the first digit in the filename indicating the image bin size)
#   proc.log is the log file recording all the image reduction process
#
# Known issue and todo list
#   program interpt when flatDlist do not fully covered the dark frame necessarily for the flat field reduction, try making another interpolation?
#
#####


import os, sys
import numpy as np
from glob import glob
from astropy.io import fits
from datetime import datetime
from operator import itemgetter
from scipy.interpolate import interp1d



def arr2fit(arr_in, out_path, hdr='', arrfmt='float32'):
  # arrfmt='float32' to capable for MaxIm DL
  #        'int32' to capable for astromatrica
  arr_in = np.array(arr_in, dtype=arrfmt)
  hdun = fits.PrimaryHDU(data=arr_in, header=hdr) if hdr != '' else fits.PrimaryHDU(data=arr_in)
  hdul = fits.HDUList([hdun])
  hdul.writeto(out_path, overwrite=True)



class imred():
  def __init__(self, rootpath, Fidx=0, show_imglist=''):
    self.Fidx = Fidx	# index number for the fits file (default=0)
    self.isidv = False	# identifier for batch (daily) process or individual image
    self.log_str = ''	# string for output logfile
    self.DinterpBx = {}	# container for dark interpolation (dict)

    # batch (daily) process
    targ_list, dark_list, flat_list, flatDlist = [], [], [], []
    if type(rootpath) != list and rootpath[:5] == 'LOT20' and rootpath[3:].isdigit() == True:
      self.LOTID = rootpath
      print ('\n> [%s] Making image list...\n' %(str(datetime.now())[:19]))
      for root, dirs, files in os.walk(raw_path+rootpath):
        for file in files:
          fpath, fp = os.path.join(root, file), root.split('/')
          if 'BLKS' not in fp and fpath.split('.')[-1] in ['fits', 'fit', 'fts']:
            imghdr = fits.getheader(fpath, ext=self.Fidx)
            imginf = [fpath, float(imghdr['JD']), imghdr['IMAGETYP'], float(imghdr['EXPTIME']), int(imghdr['XBINNING'])]
            if 'flat' in fp:
              if imghdr['IMAGETYP'] in ['DARK', 'dark', 'BIAS', 'bias']:
                flatDlist.append(imginf+[''])
              else:
                flat_list.append(imginf+[imghdr['FILTER']])
              continue
            if 'bias-dark' in fp:
              if imghdr['IMAGETYP'] in ['DARK', 'dark', 'BIAS', 'bias']:
                dark_list.append(imginf+[''])
                continue
            targ_list.append(imginf+[imghdr['FILTER']])

    # individual proc
    if type(rootpath) == list:
      self.isidv = True
      for f in rootpath[0].split('/'):
        if f[:5] == 'LOT20' and len(f) == 11:
          self.LOTID = f
          break
      for fpath in rootpath:
        imghdr = fits.open(fpath)[self.Fidx].header
        imginf = [fpath, float(imghdr['JD']), imghdr['IMAGETYP'], float(imghdr['EXPTIME']), int(imghdr['XBINNING'])]
        targ_list.append(imginf+[imghdr['FILTER']])
    self.epath = procpath+self.LOTID
    if os.path.exists(self.epath) == False:
      os.makedirs(self.epath)

    # sort image lists
    self.dark_list = np.array(sorted(dark_list, key = lambda x:x[1]))
    self.flat_list = np.array(sorted(flat_list, key = lambda x:x[1]))
    self.flatDlist = np.array(sorted(flatDlist, key = lambda x:x[1]))
    self.targ_list = np.array(sorted(targ_list, key = lambda x:x[1]))

    # print image list
    if show_imglist in ['tree', 'dark']:
      print ('>> Dark frame:')
      for i in self.dark_list:
        print ('   %s  %s  %.10f  %s %6.1f  %d  %s' %(self.LOTID, os.path.basename(i[0]), float(i[1]), i[2], float(i[3]), int(i[4]), i[5]))
    if show_imglist in ['tree', 'flat']:
      print ('\n>> Flat frame:')
      for i in self.flat_list:
        print ('   %s  %s  %.10f  %s %6.1f  %d  %s' %(self.LOTID, os.path.basename(i[0]), float(i[1]), i[2], float(i[3]), int(i[4]), i[5]))
    if show_imglist in ['tree', 'flatD']:
      print ('\n>> Dark frame for flat:')
      for i in self.flatDlist:
        print ('   %s  %s  %.10f  %s %6.1f  %d  %s' %(self.LOTID, os.path.basename(i[0]), float(i[1]), i[2], float(i[3]), int(i[4]), i[5]))
    if show_imglist in ['tree', 'targ']:
      print ('\n>> Targets:')
      for i in self.targ_list:
        print ('   %s  %s  %.10f  %s %6.1f  %d  %s' %(self.LOTID, os.path.basename(i[0]), float(i[1]), i[2], float(i[3]), int(i[4]), i[5]))
    if show_imglist != '':
      sys.exit()


  def print_inf(self, str_in, to_log=True):
    # handling on log/inf print out
    print (str_in)
    if to_log == True:
      self.log_str += str_in+'\n'


  def darkc(self, Dlist, savefit=True, savefit_suf=''):
    # dark combination, need to import the dark list to be combine
    # return dict containing master dark ordered by bin size
    # D_dict[bs][expt] (bs/expt = str)
    self.print_inf('> [%s] Creating master dark...' %(str(datetime.now())[:19]))
    D_dict, exp_list = {}, np.unique(Dlist[:,3:5], axis=0)
    for expt, bs in exp_list:
      if bs not in D_dict.keys():
        D_dict[bs] = {}
      Darr, Dpath = [], 'dark_%s_%03d.fit' %(bs, float(expt))
      self.print_inf('  %s' %(Dpath))
      for D in Dlist[(Dlist[:,3]==expt) & (Dlist[:,4]==bs)]:
        self.print_inf('     -- %s' %(D[0]))
        Darr.append(fits.open(D[0])[self.Fidx].data)
      Darr = np.median(np.array(Darr), axis=0)
      D_dict[bs][float(expt)] = Darr
      if savefit == True:
        arr2fit(Darr, self.epath+'/'+savefit_suf+Dpath)
    return D_dict


  def dark_interp_init(self, bs, D_dict):
    # interpolation of dark frame, need to import a D_dict
    if bs in self.DinterpBx.keys():
      return    # skip this step if interpolation data already exist
    exp_list = list(D_dict[bs].keys())
    Darr = itemgetter(*exp_list)(D_dict[bs])
    self.DinterpBx[bs] = interp1d(np.array(exp_list,dtype=float), Darr, axis=0, bounds_error=False, fill_value='extrapolate')


  def flatc(self, Pdark='', Pflat=''):
    # flat combination, no input pars required
    # return dict containing master flat ordered by bin size
    # F_dict[bs][filt] (bs/expt = str)
    self.F_dict = {}
    if len(self.flat_list) == 0:
      if Pflat == '':
        if input('Flat field not exist.\nContinue? (Y/n)') in ['Y','y']:
          print ('ok')
        else:
          sys.exit()
      # load flat field from the given path
      else:
        for file in glob(os.path.join(procpath+Pflat, 'flat_*.fit')):
          fileb = os.path.basename(file)
          bs, filt = fileb.split('_')[1], fileb[7:-4]
#          print (file, bs, filt)     # to be check!!!
          if bs not in self.F_dict.keys():
            self.F_dict[bs] = {}
          self.F_dict[bs][filt] = fits.open(file)[0].data

    # load (or create) dark frame for flat subtraction
    if len(self.flat_list) != 0:
      # loading dark frames
      if Pdark == '':
        flatD = self.D4targ if len(self.flatDlist) == 0 else self.darkc(self.flatDlist, savefit=False)
      if Pdark != '':
        flatD = {}
        for file in glob(os.path.join(procpath+Pdark, 'dark_*.fit')):
          fileb = os.path.basename(file)
          bs, expt = fileb.split('.')[0].split('_')[1:3]
          if bs not in flatD.keys():
            flatD[bs] = {}
          flatD[bs][expt] = fits.open(file)[0].data

      # flat reduction and combination
      self.print_inf('\n> [%s] Making master flat...' %(str(datetime.now())[:19]))
      exp_list = np.unique(self.flat_list[:,4:6], axis=0)
      for bs, filt in exp_list:
        if bs not in self.F_dict.keys():
          self.F_dict[bs] = {}
        Farr, Fpath = [], 'flat_%s_%s.fit' %(bs, filt)
        self.print_inf('  %s' %(Fpath))
        for flat in self.flat_list[(self.flat_list[:,4]==bs) & (self.flat_list[:,5]==filt)]:
          expt = float(flat[3])
          self.print_inf('     -- %s' %(flat[0]))
          # retrieving dark frame for flat field correction
          if bs in flatD.keys() and expt in flatD[bs].keys():
            Dframe = flatD[bs][expt]
          else:
            self.dark_interp_init(bs, flatD)
            Dframe = self.DinterpBx[bs](float(flat[3]))
          flat_D = fits.open(flat[0])[self.Fidx].data-Dframe
          flat_D /= np.median(flat_D)
          Farr.append(flat_D)
        Farr = np.median(np.array(Farr), axis=0)
        self.F_dict[bs][filt] = Farr
        arr2fit(Farr, '%s/flat_%s_%s.fit' %(self.epath, bs, filt))
      self.DinterpBx = {}


  def targred(self, Pdark='', Pflat='', afmt='float32'):
    # preparing dark frame
    if Pdark == '':
      if len(self.dark_list) != 0:
        self.D4targ = self.darkc(self.dark_list)
      else:
        if input('Dark frame not exist.\nContinue? (y/N)') in ['Y','y']:
          self.D4targ = {}
          print ('ok')
        else:
          sys.exit()
    else:
      self.D4targ = {}
      for file in glob(os.path.join(procpath+Pdark, 'dark_*.fit')):
        fileb = os.path.basename(file)
        bs, expt = fileb.split('.')[0].split('_')[1:3]
        if bs not in self.D4targ.keys():
          self.D4targ[bs] = {}
        self.D4targ[bs][float(expt)] = fits.open(file)[0].data

    # preparing flat field
    self.flatc(Pdark=Pdark, Pflat=Pflat)
    if len(self.targ_list) == 0:
      self.print_inf('> [%s] Target list is empty...' %(str(datetime.now())[:19]))
      return

    # image reduction
    self.print_inf('\n> [%s] Calibrating the scientific images...' %(str(datetime.now())[:19]))
    if os.path.exists(self.epath+'/targets') == False:
      os.makedirs(self.epath+'/targets')
    for targ in self.targ_list:
      self.print_inf('  %s' %(targ[0]))
      expt, bs, filt = targ[3:6]
      targ_data = fits.open(targ[0])
      targ_arr, hdr = np.array(targ_data[self.Fidx].data, dtype=float), targ_data[self.Fidx].header
      exp3d = '000' if float(targ[3]) <= 1. else '%03d' %(float(expt))
      Dbase = self.epath if Pdark == '' else Pdark
      Dpath = '%s/dark_%s_%s.fit' %(Dbase, bs, exp3d)

      # skip image if satisfying below conditions
      if ('darkpath' not in hdr)*('flatpath' not in hdr) == 0:
        self.print_inf('     xx Image already calibrated, skip this image')
        continue
      if bs not in self.D4targ.keys():
        self.print_inf('     xx Invalid CCD bin size, skip this image')
        continue

      # dark correction
      if bs in self.D4targ.keys() and float(expt) in self.D4targ[bs].keys():
        Dframe = self.D4targ[bs][float(expt)]
        hdr.append(('darkpath', Dpath, 'path for dark subtraction'))
        self.print_inf('     -- %s' %(Dpath))
      else:
        self.print_inf('      dark not exist, using interpolated dark')
        self.dark_interp_init(bs, self.D4targ)
        Dframe = self.DinterpBx[bs](float(expt))
        hdr.append(('darkpath', '*'+Dpath, 'path for dark subtraction'))
        self.print_inf('     -* %s' %(Dpath))
      targ_arr -= Dframe

      # flat correction
      Fbase = 'flat_%d_%s.fit' %(int(targ[4]), targ[5])
      Fpath = '%s/%s' %(self.epath, Fbase) if Pflat == '' else '%s/%s' %(Pflat, Fbase)
      savepath = self.epath+'/targets/'
      if bs in self.F_dict.keys() and filt in self.F_dict[bs].keys():
        Fframe = self.F_dict[bs][filt]
        hdr.append(('flatpath', Fpath, 'path for flat reduction'))
        targ_arr /= Fframe
        self.print_inf('     -- %s' %(Fpath))
        arr2fit(targ_arr, savepath+os.path.basename(targ[0]).replace('.%s' %(targ[0].split('.')[-1]),'_df.fit'), hdr, arrfmt=afmt)
      else:
        self.print_inf("     xx flat '%s' not exist" %(targ[5]))
        arr2fit(targ_arr, savepath+os.path.basename(targ[0]).replace('.%s' %(targ[0].split('.')[-1]),'_d.fit'), hdr, arrfmt=afmt)
    print
    return



  def savelog(self):
    if self.isidv == False:
      s2f_op = open(self.epath+'/proc.log','w')
    if self.isidv == True:
      s2f_op = open(self.epath+'/proc.log','a')
    s2f_op.write(self.log_str)
    s2f_op.close()





if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="Image reduction pipeline for LOT imaging ver. 211120")
  parser.add_argument('rootpath', type=str, nargs='*', help="LOT20XXXXXX, or individual fits file")
  parser.add_argument('-pdark',  type=str, default='', help="assign dark path for individual reduction (LOT20XXXXXX)")
  parser.add_argument('-pflat',  type=str, default='', help="assign flat path for individual reduction (LOT20XXXXXX)")
  parser.add_argument('-showls', type=str, default='', choices=['', 'tree', 'dark', 'flat', 'flatD', 'targ'], help="print the header infomation")
  args = parser.parse_args()

  # identify input type and initialize imred
  if len(args.rootpath) == 1 and args.rootpath[0][:5] == 'LOT20':
    calfits = imred(args.rootpath[0], show_imglist=args.showls)
  else:
    calfits = imred(args.rootpath, show_imglist=args.showls)

  # make master dark/flat only
  if len(calfits.targ_list) == 0:
    if len(calfits.dark_list) != 0:
      calfits.darkc(calfits.dark_list)
    if len(calfits.flat_list) != 0:
      D4flat = args.rootpath[0] if args.pdark == '' else args.pdark
      calfits.flatc(Pdark=D4flat)
  else:
    calfits.targred(Pdark=args.pdark, Pflat=args.pflat)
  calfits.savelog()


