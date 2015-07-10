import gdal
import cv2
import numpy


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

def pixel_to_coords(x, y, igmarray):
    x = igmarray[0][y][x]
    y = igmarray[1][y][x]
    z = igmarray[2][y][x]
    return [x, y, z]



FLANN_INDEX_LSH = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
               table_number = 6, # 12
               key_size = 12,     # 20
               multi_probe_level = 1) #2

matcher = cv2.FlannBasedMatcher(flann_params, {})
image1 = "/data/visuyan2/scratch/arsf/2015/flight_data/arsf_internal/GB15_00-2014_096_Little_Riss/processing/hyperspectral/flightlines/georeferencing/mapped/f96063b_p_sct0.01_mapped_osng.bil"

image2 = "/data/visuyan2/scratch/arsf/2015/flight_data/arsf_internal/GB15_00-2014_096_Little_Riss/processing/hyperspectral/flightlines/georeferencing/mapped/f96053b_p_sct0.97_mapped_osng.bil"

gdaltrainbil = gdal.Open(image1)
gdalquerybil = gdal.Open(image2)

band = 1
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
for m,n in matches:
    if m.distance < 0.6*n.distance:
        good.append(m)

matchedimg = drawMatches(gdaltrainband8bit, trainkeys, gdalqueryband8bit, querykeys, good)

image1_lev1map = "/data/visuyan2/scratch/arsf/2015/flight_data/arsf_internal/GB15_00-2014_096_Little_Riss/processing/hyperspectral/flightlines/georeferencing/rowcolmaps/f9606_rowcolmap.bil"
image2_lev1map = "/data/visuyan2/scratch/arsf/2015/flight_data/arsf_internal/GB15_00-2014_096_Little_Riss/processing/hyperspectral/flightlines/georeferencing/rowcolmaps/f9605_rowcolmap.bil"

gdalquerylev1map = gdal.Open(image2_lev1map)
gdaltrainlev1map = gdal.Open(image1_lev1map)

gdaltrainlev1mapbil = numpy.array(gdaltrainlev1map.ReadAsArray(trext[0],trext[1],trext[2],trext[3]), dtype="uint32")
gdalquerylev1mapbil = numpy.array(gdalquerylev1map.ReadAsArray(qext[0],qext[1],qext[2],qext[3]), dtype="uint32")
trainigmfile = "/data/visuyan2/scratch/arsf/2015/flight_data/arsf_internal/GB15_00-2014_096_Little_Riss/processing/hyperspectral/flightlines/georeferencing/igm/f96063b_p_sct0.01.igm"
queryigmfile = "/data/visuyan2/scratch/arsf/2015/flight_data/arsf_internal/GB15_00-2014_096_Little_Riss/processing/hyperspectral/flightlines/georeferencing/igm/f96053b_p_sct0.97.igm"
trainigmarray = gdal.Open(trainigmfile).ReadAsArray()
queryigmarray = gdal.Open(queryigmfile).ReadAsArray()

pointcombos = []

for match in good:
    trainscanline = gdaltrainlev1mapbil[0][trainkeys[match.trainIdx].pt[1]][trainkeys[match.trainIdx].pt[0]]
    trainpixel = gdaltrainlev1mapbil[1][trainkeys[match.trainIdx].pt[1]][trainkeys[match.trainIdx].pt[0]]
    queryscanline = gdalquerylev1mapbil[0][trainkeys[match.trainIdx].pt[1]][trainkeys[match.trainIdx].pt[0]]
    querypixel = gdalquerylev1mapbil[1][trainkeys[match.trainIdx].pt[1]][trainkeys[match.trainIdx].pt[0]]
    print trainscanline, trainpixel
    pointonscanline = pixel_to_coords(trainscanline, trainpixel, trainigmarray)
    pointoffscanline = pixel_to_coords(queryscanline, querypixel, queryigmarray)
    centrepixel = centre_pixel_builder(trainscanline, trainigmarray)
    centredpointonscanline = [pointonscanline[0] - centrepixel[0],pointonscanline[1] - centrepixel[1],pointonscanline[2] - centrepixel[2]]
    centredpointoffscanline = [pointoffscanline[0] - centrepixel[0],pointoffscanline[1] - centrepixel[1],pointoffscanline[2] - centrepixel[2]]
    pointcombos.append([pointonscanline, pointoffscanline, centrepixel, centredpointonscanline, centredpointoffscanline])

pointcombos = numpy.array(pointcombos)

print pointcombos



gdalquerylev1mapbil = numpy.array(gdalquerylev1map.ReadAsArray(), dtype="uint16")
cv2.imshow('Matched Features', matchedimg)
cv2.waitKey(0)
cv2.destroyAllWindows()