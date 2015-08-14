#!/usr/bin/env python
import os, sys, argparse
import gcpparser
import features
import distancecalculator
import igmparser
import numpy as np
import libgpstime
import read_sol_file
import timeit
import datetime
import imagematcher

from osgeo import gdal

RIGHT = 1
LEFT = 2
CENTER =0
BANDLIST = [1, 2]

def gpssec(year, month, day, hour, minute, second):
   """
   converts header info to a rough gps second

   :param year:
   :param month:
   :param day:
   :param hour:
   :param minute:
   :param second:
   :return gpsseconds:
   """
   isoweekday = datetime.date(year, month, day).isoweekday()
   dayseconds = 86400

   secs = (dayseconds * isoweekday) + (3600 * hour) + (60 * minute) + second

   return secs

def altFind(hdrfile, navfile):
   """
   Function altFind

   takes a level one header file and open navfile object,
   returns an average altitude for the flightline

   :param hdrfile:
   :param navfile:
   :return altitude:
   """
   for line in hdrfile:
      #grab times from the header file
      if "GPS Start Time" in line:
         start = line[27:]
      if "GPS Stop Time" in line:
         end = line[26:]
      if "acquisition" in line:
         day = line[37:]

   #identify the start and stop points of the scanline

   day, month, year = day.split('-')

   hour, minute, second = start.split(':')
   second = int(second[:2].replace('.',''))

   gpsstart = gpssec(int(year), int(month), int(day), int(hour), int(minute), int(second))
   hour, minute, second = end.split(':')
   second = int(second[:2].replace('.', ''))

   gpsstop = gpssec(int(year), int(month), int(day), int(hour), int(minute), int(second))

   #grabs the relevant entries from a nav file
   trimmed_data=navfile[np.where(navfile['time'] > gpsstart)]
   trimmed_data=trimmed_data[np.where(trimmed_data['time'] < gpsstop)]

   #generate the average altitude
   altitude = np.mean(trimmed_data['alt'])
   return altitude

def autoBoresight(flightlinefolder, gcpfolder, gcpcsv, igmfolder, navfile, output, hdrfolder, lev1mapfolder):
   """
   Function autoBoresight

   Main function for boresighting, takes a queryflightline folder, igm folder nav file and level 1 header folder
   returns averaged adjustments across all flightlines in the queryflightline folder

   optionally takes gcp location info and a folder of gcp images,
   however this is currently not tested or implemented

   :param flightlinefolder:
   :param gcpfolder:
   :param gcpcsv:
   :param igmfolder:
   :param navfile:
   :param output:
   :param hdrfolder:
   :return  pitch, roll, heading:
   """
   #general set up operations
   start_time = timeit.default_timer()
   igmfiles = os.listdir(igmfolder)
   hdrfiles = os.listdir(hdrfolder)
   lev1maps = os.listdir(lev1mapfolder)
   navfile = read_sol_file.readSol(navfile)
   adjust=[]
   for flightline in [x for x in os.listdir(flightlinefolder) if 'hdr' not in x and "aux" not in x]:
      #we need to establish the altitude of our primary flightline
      baseigmfile = [x for x in igmfiles if flightline[:5] in x and 'osng' in x and 'igm' in x and 'hdr' not in x and "aux" not in x][0]
      baseflightline = flightline
      baseflightline = (flightlinefolder + '/' + flightline)
      baseflightlineheaderfile = open(hdrfolder + '/' + [hdrfile for hdrfile in hdrfiles if flightline[:5] in hdrfile and 'hdr' in hdrfile and "mask" not in hdrfile ][0])
      baseflightlinealtitude = altFind(baseflightlineheaderfile, navfile)
      baselev1map = lev1mapfolder + '/' + [x for x in lev1maps if flightline[:5] in x and 'bil' in x and 'hdr' not in x and "aux" not in x][0]
      igmarray = igmparser.bilReader(igmfolder + '/' + baseigmfile)

      scanlineadjustments = []
      totalpoints = 0
      for queryflightline in [x for x in os.listdir(flightlinefolder) if 'hdr' not in x and "aux" not in x]:
         #need to test if they have the same filename otherwise it would be bad
         if queryflightline not in baseflightline:
            print "%s being compared to %s" % (queryflightline, baseflightline)
            #first test for same altitude
            queryflightlineheaderfile = open(hdrfolder + '/' + [hdrfile for hdrfile in hdrfiles if queryflightline[:5] in hdrfile and 'hdr' in hdrfile and "mask" not in hdrfile][0])
            queryflightlinealtitude = altFind(queryflightlineheaderfile, navfile)


            if (queryflightlinealtitude >= baseflightlinealtitude - 100) and (queryflightlinealtitude <= baseflightlinealtitude + 100):
               print "altitudes matched at %s %s" % (queryflightlinealtitude, baseflightlinealtitude)
               #then test for overlap
               queryigmfile = [x for x in igmfiles if queryflightline[:5] in x and 'osng' in x and 'igm' in x and 'hdr' not in x and "aux" not in x][0]
               queryigmarray = igmparser.bilReader(igmfolder + '/' + queryigmfile)
               querylev1map = lev1mapfolder + '/' + [x for x in lev1maps if queryflightline[:5] in x and 'bil' in x and 'hdr' not in x and "aux" not in x][0]
               queryflightline = flightlinefolder + '/' + queryflightline
               gdalqueryflightline = gdal.Open(queryflightline)
               gdalbaseflightline = gdal.Open(baseflightline)
               baseflightlinegeotrans = gdalqueryflightline.GetGeoTransform()
               queryflightlinegeotrans = gdalqueryflightline.GetGeoTransform()
               baseflightlinebounds = [baseflightlinegeotrans[0],
                                      baseflightlinegeotrans[3],
                                      baseflightlinegeotrans[0] + (baseflightlinegeotrans[1] * gdalbaseflightline.RasterXSize),
                                      baseflightlinegeotrans[3] + (baseflightlinegeotrans[5] * gdalbaseflightline.RasterYSize)]
               querybounds = [baseflightlinegeotrans[0],
                             queryflightlinegeotrans[3],
                             queryflightlinegeotrans[0] + (queryflightlinegeotrans[1] * gdalqueryflightline.RasterXSize),
                             queryflightlinegeotrans[3] + (queryflightlinegeotrans[5] * gdalqueryflightline.RasterYSize)]

               overlap = [max(baseflightlinebounds[0], querybounds[0]),
                          min(baseflightlinebounds[1], querybounds[1]),
                          min(baseflightlinebounds[2], querybounds[2]),
                          max(baseflightlinebounds[3], querybounds[3])]

               if (overlap[2] < overlap[0]) or (overlap[1] < overlap[3]):
                  #if there is no overlap
                  overlap = None

               #if there isn't an overlap then we should ignore these flightlines
               if overlap != None:
                  print "overlap confirmed between %s and %s region is:" % (queryflightline, baseflightline)
                  for band in BANDLIST:
                     no_matches=False
                     try:
                         trainkeys, querykeys, trext, qext, good = imagematcher.matcher(baseflightline,
                                                                                        queryflightline,
                                                                                        "flann",
                                                                                        None,
                                                                                        band=band)
                     except ValueError:
                         no_matches=True
                     except TypeError:
                         no_matches=True
                     if not no_matches:
                         pointcombos = imagematcher.matches_to_hyperspectral_geopoints(baselev1map,
                                                                                      querylev1map,
                                                                                      igmfolder + '/' + baseigmfile,
                                                                                      igmfolder + '/' + queryigmfile,
                                                                                      good,
                                                                                      trainkeys,
                                                                                      querykeys,
                                                                                      trext,
                                                                                      qext)
                         try:
                             if len(pointcombos[0]) > 0:
                                pitch = []
                                roll = []
                                heading = []
                                for pointgroup in pointcombos:
                                   if pointgroup[5] is RIGHT:
                                      for comparisonpoint in pointcombos:
                                         if comparisonpoint[5] is LEFT:
                                            intersectpoint = distancecalculator.intersect(pointgroup[0],
                                                                                          comparisonpoint[0],
                                                                                          pointgroup[1],
                                                                                          comparisonpoint[1])
                                            intersectpoint = [intersectpoint[0] + pointgroup[2][0],
                                                             intersectpoint[1] + pointgroup[2][1],
                                                             pointgroup[2][2]]
                                            if (intersectpoint[0] < 800000) and (intersectpoint[1] < 800000):
                                                if (intersectpoint > -100000) and (intersectpoint > -100000):
                                                    try:
                                                        tempheading = distancecalculator.headingAngle(intersectpoint,
                                                                                                      pointgroup[0],
                                                                                                      pointgroup[1])
                                                        heading.append(tempheading)
                                                    except ValueError, e:
                                                        print e
                                            else:
                                                heading.append(0)
                                   if pointgroup[5] is LEFT:
                                      for comparisonpoint in pointcombos:
                                         if comparisonpoint[5] is RIGHT:
                                            intersectpoint = distancecalculator.intersect(pointgroup[0],
                                                                                          comparisonpoint[0],
                                                                                          pointgroup[1],
                                                                                          comparisonpoint[1])
                                            intersectpoint = [intersectpoint[0] + pointgroup[2][0],
                                                             intersectpoint[1] + pointgroup[2][1],
                                                             pointgroup[2][2]]
                                            if (intersectpoint[0] < 800000) and (intersectpoint[1] < 800000):
                                                if (intersectpoint > -100000) and (intersectpoint > -100000):
                                                    try:
                                                        tempheading = distancecalculator.headingAngle(intersectpoint,
                                                                                                      pointgroup[0],
                                                                                                      pointgroup[1])
                                                        heading.append(tempheading)
                                                    except ValueError, e:
                                                        print e
                                            else:
                                                heading.append(0)

                                for pointgroup in pointcombos:
                                   temppitch, temproll = distancecalculator.pitchRollAdjust(pointgroup[2],pointgroup[1],pointgroup[0],2000)
                                   pitch.append(temppitch)
                                   roll.append(temproll)
                                adjust.append([[baseflightline, queryflightline], pitch, roll, heading])
                         except IndexError:
                             continue
                     else:
                         print "no matches between %s and %s" % (queryflightline, baseflightline)
               else:
                  print "no overlap between %s and %s" % (queryflightline, baseflightline)
            else:
               print "%s and %s flown at different altitudes (%s, %s), skipping to avoid result skew" % (baseflightline, queryflightline, baseflightlinealtitude, queryflightlinealtitude)
         else:
            continue

      p = 0
      r = 0
      h = 0

      if len(scanlineadjustments) != 0:
         for adjustment in scanlineadjustments:

            p = np.float64(p + adjustment[0])
            r = np.float64(r + adjustment[1])
            h = np.float64(h + adjustment[2])

         length = len(scanlineadjustments)
         p = np.float64(p / length)
         r = np.float64(r / length)
         h = np.float64(h / length)

         adjust.append([p, r, h])
      else:
         continue

   p = 0
   r = 0
   h = 0
   print "Total queryflightline adjustments:"
   adjust = np.array(adjust)

   baseheading = np.array(adjust[:, 3])
   basepitch = np.array(adjust[:, 1])
   baseroll = np.array(adjust[:, 2])
   print "heading"
   print np.mean(np.ravel(baseheading))
   print "roll"
   print np.mean(np.ravel(baseroll))
   print "pitch"
   print np.mean(np.ravel(basepitch))

   print "per flightline avgs"
   print "pitch"
   for adjustment in adjust:
       print "heading"
       print np.mean(adjustment[:, 3])

       print "roll"
       print np.mean(adjustment[:, 2])

       print "pitch"
       print np.mean(adjustment[:, 1])
   # for flightline in adjust:
   #    p = p + reduce(lambda x, y: x + y, flightline[0]) / len(flightline[0])
   #    r = r +
   # length = len(adjust)
   # if length > 1:
   #    p = p / length
   #    r = r / length
   #    h = h / length
   # print "pitch"
   # print p / 2
   # print "roll"
   # print r / 2
   # print "heading"
   # print h / 2
   # print "Calculated on %s points from %s flightlines" % (totalpoints, len(os.listdir(flightlinefolder)))
   # print "Took %s seconds" % (timeit.default_timer() - start_time)
   # return p, r, h

if __name__=='__main__':
   #Get the input arguments
   parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   parser.add_argument('--gcps',
                       '-g',
                       help='Input gcp file to read',
                       default=None,
                       metavar="<csvfile>")
   parser.add_argument('--gcpimages',
                       '-a',
                       help='gcp image plates for location identification',
                       default=None,
                       metavar="<folder>")
   parser.add_argument('--igmfolder',
                       '-i',
                       help='project igm file folder',
                       default="",
                       metavar="<folder>")
   parser.add_argument('--navfile',
                       '-n',
                       help='project nav file (sol/sbet)',
                       default="",
                       metavar="<sol/sbet>")
   parser.add_argument('--bils',
                       '-b',
                       help='mapped bil folder',
                       default="",
                       metavar="<folder>")
   parser.add_argument('--output',
                       '-o',
                       help='Output TXT file to write',
                       default="",
                       metavar="<txtfile>")
   parser.add_argument('--lev1',
                       '-l',
                       help='level 1 folder with headers',
                       default="",
                       metavar="<folder>")
   parser.add_argument('--lev1maps',
                       '-m',
                       help='level 1 map folder with headers',
                       default="",
                       metavar="<folder>")
   commandline=parser.parse_args()

   if os.path.exists(commandline.igmfolder):
      igmlist = os.path.abspath(commandline.igmfolder)
   else:
      print "igm folder required, use -i or --igmfolder"
      exit(0)

   if os.path.exists(commandline.bils):
      billist = os.path.abspath(commandline.bils)
   else:
      print "bil folder required, use -b or --bils"
      exit(0)

   if os.path.exists(commandline.navfile):
      navfile = os.path.abspath(commandline.navfile)
   else:
      print "nav file required, use -n or --navfile"
      exit(0)

   if os.path.exists(commandline.lev1):
      hdrfolder = os.path.abspath(commandline.lev1)
   else:
      print "level 1 folder required, use -l or --lev1"
      exit(0)

   if os.path.exists(commandline.lev1maps):
      lev1mapfolder = os.path.abspath(commandline.lev1maps)
   else:
      print "level 1 map folder required, use -m or --lev1maps"
      exit(0)

   if commandline.gcps or commandline.gcpimages:
      if not commandline.gcps or not commandline.gcpimages:
         print "gcp csv or gcp images folder not present"
         exit(0)
      else:
         gcpimagesfolder = commandline.gcpimages
         gcpcsv = commandline.gcps
   else:
      boresight = autoBoresight(billist,
                                None,
                                None,
                                igmlist,
                                navfile,
                                None,
                                hdrfolder,
                                lev1mapfolder)