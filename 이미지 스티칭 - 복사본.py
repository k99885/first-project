import cv2
import numpy as np
import cv2 as cv

FLANN_INDEX_LSH = 6


def anorm2(a):
    return (a * a).sum(-1)


def anorm(a):
    return np.sqrt(anorm2(a))


def matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2):
    flann_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2

    matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)  # 2

    matches = []
    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.9:#0.79
            matches.append((m[0].trainIdx, m[0].queryIdx))

    if len(matches) >= 4:

        keyPoints1 = np.float32([keyPoints1[i] for (_, i) in matches])
        keyPoints2 = np.float32([keyPoints2[i] for (i, _) in matches])

        H, status = cv.findHomography(keyPoints1, keyPoints2, cv.RANSAC, 4.0)

        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
    else:
        H, status = None, None
        print(' matches found, not enough for homography estimation' )

    return matches, H, status


def drawMatches_V(image1, image2, keyPoints1, keyPoints2, matches, status):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    img_matching_result = np.zeros((h1+ h2, max(w1,w2), 3), dtype="uint8")#img_matching_result = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")

    img_matching_result[0:h2, 0:w2] = image2
    img_matching_result[h2:, 0:w1] = image1

    #img_matching_result[0:h2, 0:w2] = image2
    #img_matching_result[0:h1:, w2:] = image1

    for ((trainIdx, queryIdx), s) in zip(matches, status):

        if s == 1:
            keyPoint2 = (int(keyPoints2[trainIdx][0]), int(keyPoints2[trainIdx][1]))
            keyPoint1 = (int(keyPoints1[queryIdx][0]) , int(keyPoints1[queryIdx][1])+h2)
            cv.line(img_matching_result, keyPoint1, keyPoint2, (0, 255, 0), 2)

    return img_matching_result

def drawMatches_H(image1, image2, keyPoints1, keyPoints2, matches, status):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    img_matching_result = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")

    img_matching_result[0:h2, 0:w2] = image2
    img_matching_result[0:h1:, w2:] = image1

    for ((trainIdx, queryIdx), s) in zip(matches, status):

        if s == 1:
            keyPoint2 = (int(keyPoints2[trainIdx][0]), int(keyPoints2[trainIdx][1]))
            keyPoint1 = (int(keyPoints1[queryIdx][0]) + w2, int(keyPoints1[queryIdx][1]))
            cv.line(img_matching_result, keyPoint1, keyPoint2, (0, 255, 0), 1)

    return img_matching_result

def main_V(image1,image2):
    file_name1 = 'images/%d.dng' % image1
    file_name2 = 'images/%d.dng' % image2
    img1 = cv.imread(file_name1)#변형해서 추가할 이미지
    img2 = cv.imread(file_name2)#기준이미지

    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    detector = cv2.AKAZE_create()#cv.BRISK_create()
    keyPoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keyPoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    #print('img1 - %d features, img2 - %d features' % (len(keyPoints1), len(keyPoints2)))

    keyPoints1 = np.float32([keypoint.pt for keypoint in keyPoints1])
    keyPoints2 = np.float32([keypoint.pt for keypoint in keyPoints2])

    matches, H, status = matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2)

    img_matching_result = drawMatches_V(img1, img2, keyPoints1, keyPoints2, matches, status)

    result = cv.warpPerspective(img1, H, (img1.shape[1],img1.shape[0] + img2.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    return result, img_matching_result
    #result = cv.warpPerspective(img1, H,(img1.shape[1] + img2.shape[1], img1.shape[0]))
    #result[0:img2.shape[0], 0:img2.shape[1]] = img2
    #
    # cv2.namedWindow('result',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('result',1900,980)
    #
    # cv2.namedWindow('matching result',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('matching result',1900,980)
    #
    # cv.imshow('result', result)
    # cv.imshow('matching result', img_matching_result)
    # cv2.imwrite('result.jpg', result)
    # cv.waitKey()

def main_H(image1,image2):
    file_name1 = '%d.jpg' % image1
    file_name2 = '%d.jpg' % image2

    img1 = cv.imread(file_name1)#변형해서 추가할 이미지
    img2 = cv.imread(file_name2)#기준이미지

    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    detector = cv.BRISK_create()
    keyPoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keyPoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    print('img1 - %d features, img2 - %d features' % (len(keyPoints1), len(keyPoints2)))

    keyPoints1 = np.float32([keypoint.pt for keypoint in keyPoints1])
    keyPoints2 = np.float32([keypoint.pt for keypoint in keyPoints2])

    matches, H, status = matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2)

    img_matching_result = drawMatches_H(img1, img2, keyPoints1, keyPoints2, matches, status)

    result = cv.warpPerspective(img1, H,(img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    return  result,img_matching_result
    # cv2.namedWindow('result',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('result',1900,980)
    #
    # cv2.namedWindow('matching result',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('matching result',1900,980)
    #
    # cv.imshow('result', result)
    # cv.imshow('matching result', img_matching_result)
    # cv2.imwrite('result.jpg', result)
    # cv.waitKey()


result1,img_matching_result1=main_V(32,31)
# cv2.namedWindow('result1',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('result1',1900,980)

cv2.namedWindow('matching result1',cv2.WINDOW_NORMAL)
cv2.resizeWindow('matching result1',500,1000)

#cv.imshow('result1', result1)
cv.imshow('matching result1', img_matching_result1)
cv2.imwrite('1.jpg', result1)

result2,img_matching_result2=main_V(33,34)
cv2.namedWindow('matching result2',cv2.WINDOW_NORMAL)
cv2.resizeWindow('matching result2',500,1000)
cv.imshow('matching result2', img_matching_result2)
cv2.imwrite('2.jpg', result2)

result3,img_matching_result3=main_H(2,1)
cv2.imwrite('result3.jpg', result3)
cv.waitKey()
cv.destroyAllWindows()