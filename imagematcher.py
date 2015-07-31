import gdal
import cv2
import numpy
import distancecalculator
import os

def scanlinepixelgrabber(xidx, yidx, rowcolmap):
    scanline = rowcolmap[0][yidx][xidx]
    pixel = rowcolmap[1][yidx][xidx]
    if (scanline == 4294967295 or pixel == 4294967295):
        scanline = rowcolmap[0][yidx - 1][xidx - 1]
        pixel = rowcolmap[1][yidx - 1][xidx - 1]

    if (scanline == 4294967295 or pixel == 4294967295):
        scanline = rowcolmap[0][yidx + 1][xidx + 1]
        pixel = rowcolmap[1][yidx + 1][xidx + 1]

    if (scanline == 4294967295 or pixel == 4294967295):
        scanline = rowcolmap[0][yidx][xidx - 1]
        pixel = rowcolmap[1][yidx][xidx - 1]

    if (scanline == 4294967295 or pixel == 4294967295):
        scanline = rowcolmap[0][yidx][xidx + 1]
        pixel = rowcolmap[1][yidx][xidx + 1]

    if (scanline == 4294967295):
        scanline = None
        print "im setting it to no"
    return scanline, pixel


def gdal_intersect_reader(line1, line2):
    tbg = line1.GetGeoTransform()
    r1 = [tbg[0], tbg[3], tbg[0] + (tbg[1] * line1.RasterXSize), tbg[3] + (tbg[5] * line1.RasterYSize)]
    qbg = line2.GetGeoTransform()
    r2 = [qbg[0], qbg[3], qbg[0] + (qbg[1] * line2.RasterXSize), qbg[3] + (qbg[5] * line2.RasterYSize)]
    intersection = [max(r1[0], r2[0]), min(r1[1], r2[1]), min(r1[2], r1[2]), max(r1[3], r2[3])]
    left1 = int(round((intersection[0]-r1[0])/tbg[1])) # difference divided by pixel dimension
    top1 = int(round((intersection[1]-r1[1])/tbg[5]))
    col1 = int(round((intersection[2]-r1[0])/tbg[1])) - left1 # difference minus offset left
    row1 = int(round((intersection[3]-r1[1])/tbg[5])) - top1
    left2 = int(round((intersection[0]-r2[0])/qbg[1])) # difference divided by pixel dimension
    top2 = int(round((intersection[1]-r2[1])/qbg[5]))
    col2 = int(round((intersection[2]-r2[0])/qbg[1])) - left2 # difference minus new left offset
    row2 = int(round((intersection[3]-r2[1])/qbg[5])) - top2
    return [left1, top1, col1, row1], [left2, top2, col2, row2]

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    out = numpy.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    # Place the first image to the left
    out[:rows1,:cols1,:] = numpy.dstack([img1, img1, img1])
    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = numpy.dstack([img2, img2, img2])
    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
    return out

def centre_pixel_builder(scanline, igmarray):
    centre = igmarray.shape[2] / 2
    x = igmarray[0][scanline][centre]
    y = igmarray[1][scanline][centre]
    z = igmarray[2][scanline][centre]
    return [x, y, z]

def pixel_to_coords(xidx, yidx, igmarray):
    x = igmarray[0][yidx][xidx]
    y = igmarray[1][yidx][xidx]
    z = igmarray[2][yidx][xidx]
    return [x, y, z]

def matcher(image1, image2, match_alg="flann", output_location=None, band=1):
    FLANN_INDEX_LSH = 6
    flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    gdaltrainbil = gdal.Open(image1)
    gdalquerybil = gdal.Open(image2)
    trext, qext = gdal_intersect_reader(gdaltrainbil, gdalquerybil)
    gdaltrainband = numpy.array(gdaltrainbil.GetRasterBand(band).ReadAsArray(trext[0],trext[1],trext[2],trext[3]), dtype="uint16")
    gdalqueryband = numpy.array(gdalquerybil.GetRasterBand(band).ReadAsArray(qext[0],qext[1],qext[2],qext[3]), dtype="uint16")
    gdaltrainbandnormalised = gdaltrainband - gdaltrainband.min()
    gdaltrainband8bittemp = numpy.array(gdaltrainbandnormalised / ((numpy.max(gdaltrainbandnormalised)+1) / 255), dtype="uint8")
    gdalquerybandnormalised = gdalqueryband - gdalqueryband.min()
    gdalqueryband8bittemp = numpy.array(gdalquerybandnormalised / ((numpy.max(gdalquerybandnormalised)+1) / 255), dtype="uint8")
    gdaltrainband8bit = gdaltrainband8bittemp.copy()
    gdalqueryband8bit = gdalqueryband8bittemp.copy()
    gdalqueryband_mask = numpy.ma.masked_values(gdalqueryband8bit, 0)
    gdaltrainband_mask = numpy.ma.masked_values(gdaltrainband8bit, 0)
    orb = cv2.ORB(nfeatures=10000)
    trainkeys, traindescs = orb.detectAndCompute(gdaltrainband8bit, None)
    querykeys, querydescs = orb.detectAndCompute(gdalqueryband8bit, None)
    matches = matcher.knnMatch(traindescs, querydescs, k=2)
    good = []
    try:
        for m,n in matches:
            if m.distance < 0.6*n.distance:
                good.append(m)
    except ValueError:
        raise ValueError, "no matches found between images"
    if output_location != None:
        matchedimg = drawMatches(gdaltrainband8bit, trainkeys, gdalqueryband8bit, querykeys, good)
        cv2.imwrite(output_location + os.path.basename(image1) + os.path.basename(image2) + "_matched_points_" + match_alg, matchedimg)
    return trainkeys, querykeys, trext, qext, good

def matches_to_hyperspectral_geopoints(image1_lev1map, image2_lev1map, trainigmfile, queryigmfile, matches, trainkeys, querykeys, trext, qext):
    gdalquerylev1map = gdal.Open(image2_lev1map)
    gdaltrainlev1map = gdal.Open(image1_lev1map)
    gdaltrainlev1mapbil = numpy.array(gdaltrainlev1map.ReadAsArray(trext[0],trext[1],trext[2],trext[3]), dtype="uint32")
    gdalquerylev1mapbil = numpy.array(gdalquerylev1map.ReadAsArray(qext[0],qext[1],qext[2],qext[3]), dtype="uint32")
    trainigmarray = gdal.Open(trainigmfile).ReadAsArray()
    queryigmarray = gdal.Open(queryigmfile).ReadAsArray()
    pointcombos = []
    for num, match in enumerate(matches):
        trainscanline, trainpixel = scanlinepixelgrabber(trainkeys[match.queryIdx].pt[0],
                                                         trainkeys[match.queryIdx].pt[1],
                                                         gdaltrainlev1mapbil)
        queryscanline, querypixel = scanlinepixelgrabber(querykeys[match.trainIdx].pt[0],
                                                         querykeys[match.trainIdx].pt[1],
                                                         gdalquerylev1mapbil)
        if not queryscanline is None and not trainscanline is None:
            pointonscanline = pixel_to_coords(trainpixel, trainscanline, trainigmarray)
            pointoffscanline = pixel_to_coords(querypixel, queryscanline, queryigmarray)
            centrepixel = centre_pixel_builder(trainscanline, trainigmarray)
            if trainpixel >= trainigmarray.shape[2] / 2 + 50:
                position = 1
            elif trainpixel <= trainigmarray.shape[2] / 2 - 50:
                position = 2
            elif trainpixel < trainigmarray.shape[2] / 2 + 50 and trainpixel > trainigmarray.shape[2] / 2 - 50:
                position = 0
            centredpointonscanline = [pointonscanline[0] - centrepixel[0],pointonscanline[1] - centrepixel[1],pointonscanline[2] - centrepixel[2]]
            centredpointoffscanline = [pointoffscanline[0] - centrepixel[0],pointoffscanline[1] - centrepixel[1],pointoffscanline[2] - centrepixel[2]]
            pointcombos.append([pointonscanline, pointoffscanline, centrepixel, centredpointonscanline, centredpointoffscanline, position])
    pointcombos = numpy.array(pointcombos)
    return pointcombos

def main():
    image1 = "/data/visuyan2/scratch/arsf/2015/flight_data/arsf_internal/GB15_00-2014_096_Little_Riss/processing/hyperspectral/flightlines/georeferencing/mapped/f96063b_p_sct0.01_mapped_osng.bil"
    image2 = "/data/visuyan2/scratch/arsf/2015/flight_data/arsf_internal/GB15_00-2014_096_Little_Riss/processing/hyperspectral/flightlines/georeferencing/mapped/f96053b_p_sct0.97_mapped_osng.bil"
    match_alg="flann"
    trainkeys, querykeys, trext, qext, good, matchedimg = matcher(image1, image2, match_alg)
    output = "/users/rsg/stgo/image_outputs"
    image1_lev1map = "/data/visuyan2/scratch/arsf/2015/flight_data/arsf_internal/GB15_00-2014_096_Little_Riss/processing/hyperspectral/flightlines/georeferencing/rowcolmaps/f9606_rowcolmap.bil"
    image2_lev1map = "/data/visuyan2/scratch/arsf/2015/flight_data/arsf_internal/GB15_00-2014_096_Little_Riss/processing/hyperspectral/flightlines/georeferencing/rowcolmaps/f9605_rowcolmap.bil"
    trainigmfile = "/data/visuyan2/scratch/arsf/2015/flight_data/arsf_internal/GB15_00-2014_096_Little_Riss/processing/hyperspectral/flightlines/georeferencing/igm/f96063b_p_sct0.01_osng.igm"
    queryigmfile = "/data/visuyan2/scratch/arsf/2015/flight_data/arsf_internal/GB15_00-2014_096_Little_Riss/processing/hyperspectral/flightlines/georeferencing/igm/f96053b_p_sct0.97_osng.igm"
    pointcombos = matches_to_hyperspectral_geopoints(image1_lev1map,
                                                     image2_lev1map,
                                                     trainigmfile,
                                                     queryigmfile,
                                                     good,
                                                     trainkeys,
                                                     querykeys,
                                                     trext,
                                                     qext)
    print "points done"
    pitch = []
    roll = []
    for pointgroup in pointcombos:
        temppitch, temproll = distancecalculator.pitchRollAdjust(pointgroup[2],pointgroup[1],pointgroup[0],2000)
        pitch.append(temppitch)
        roll.append(temproll)
    return pitch, roll