import cv2
import numpy as np

MAX_FEATURES = 500
corners = None

def Align(refImg, alignImg):

    # 将图像上传到GPU
    gpuRefImg = cv2.cuda_GpuMat()
    gpuAlignImg = cv2.cuda_GpuMat()
    gpuRefImg.upload(refImg)
    gpuAlignImg.upload(alignImg)
    
    # 将图像转换为灰度图像
    gpuRefImgGray = cv2.cuda.cvtColor(gpuRefImg, cv2.COLOR_BGR2GRAY)
    gpuAlignImgGray = cv2.cuda.cvtColor(gpuAlignImg, cv2.COLOR_BGR2GRAY)
    
    # ORB特征点检测
    orb = cv2.cuda_ORB.create(MAX_FEATURES)
    gpuRefKp, gpuRefDes = orb.detectAndComputeAsync(gpuRefImgGray, None)
    gpuAlignKp, gpuAlignDes = orb.detectAndComputeAsync(gpuAlignImgGray, None)
    refKp = orb.convert(gpuRefKp)
    alignKp = orb.convert(gpuAlignKp)
    
    # 特征点匹配
    bf = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(gpuRefDes, gpuAlignDes)
    
    # 排序筛选匹配点
    matches = sorted(matches, key=lambda x: x.distance)
    goodMatches = matches[:int(len(matches) * 0.15)]
    
    # 绘制匹配结果
    matchedImg = cv2.drawMatches(refImg, refKp, alignImg, alignKp, goodMatches, None)
    # cv2.imshow('matchedImg', matchedImg)
    # cv2.waitKey(0)
    
    # 计算变换矩阵
    refPts = np.float32([refKp[m.queryIdx].pt for m in goodMatches])
    alignPts = np.float32([alignKp[m.trainIdx].pt for m in goodMatches])
    M, mask = cv2.findHomography(alignPts, refPts, cv2.RANSAC)
    
    # 进行透视变换
    CalcCorners(alignImg, M)
    alignedImg = cv2.warpPerspective(alignImg, M, (max(corners[2][0][0],corners[3][0][0]), refImg.shape[0]))
    
    return matchedImg, alignedImg

# 计算透视变换后的四个角坐标
def CalcCorners(img, H):
    h,w = img.shape[:2]
    pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,H)
    global corners
    corners = np.int32(dst)

# 优化连接处的拼接
def OptimizeSeam(refImg, alignedImg, stitchedImg):
    st = min(corners[0][0][0], corners[1][0][0])
    width = refImg.shape[1] - st
    rows = stitchedImg.shape[0]
    cols = refImg.shape[1]
    for y in range(rows):
        for x in range(st, cols):
            if alignedImg[y, x, 0] == 0 and alignedImg[y, x, 1] == 0 and alignedImg[y, x, 2] == 0:
                alpha = 1
            else:
                alpha = (width - x + st) / width
            stitchedImg[y, x, 0] = alpha * refImg[y, x, 0] + (1 - alpha) * alignedImg[y, x, 0]
            stitchedImg[y, x, 1] = alpha * refImg[y, x, 1] + (1 - alpha) * alignedImg[y, x, 1]
            stitchedImg[y, x, 2] = alpha * refImg[y, x, 2] + (1 - alpha) * alignedImg[y, x, 2]
    return stitchedImg
            

# 自我实现的拼接函数
def MyStitch(refImg, alignedImg, alignImg):
    dst_width = alignedImg.shape[1]
    dst_height = refImg.shape[0]
    dst = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
    dst[0:dst_height, 0:dst_width] = alignedImg
    dst[0:dst_height, 0:refImg.shape[1]] = refImg
    ShowImg(dst)
    dst1 = OptimizeSeam(refImg, alignedImg, dst)
    return dst, dst1

# 拼接配准后的图像和基准图像
def Stitch(refImg, alignImg):
    
    # 创建拼接器
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    # 拼接
    status, pano = stitcher.stitch([refImg, alignImg])
    # 黑边处理
    if status == cv2.Stitcher_OK:
        # 全景图轮廓提取
        stitched = cv2.copyMakeBorder(pano, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
 
        # 轮廓最小正矩形
        mask = np.zeros(thresh.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(cnts[0])  # 取出list中的轮廓二值图，类型为numpy.ndarray
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
 
        # 腐蚀处理，直到minRect的像素值都为0
        minRect = mask.copy()
        sub = mask.copy()
        while cv2.countNonZero(sub) > 0:
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)
 
        # 提取minRect轮廓并裁剪
        cnts = cv2.findContours(minRect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        (x, y, w, h) = cv2.boundingRect(cnts[0])
        stitched = stitched[y:y + h, x:x + w]
 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # ShowImg(stitched)
    else:
        print('图像匹配的特征点不足')
    return pano, stitched

def ShowResizedImg(img, rate, imgName='res'):
    cv2.imshow('res', cv2.resize(img,None,fx=rate,fy=rate,interpolation=cv2.INTER_CUBIC))
    cv2.waitKey(0)

def ShowImg(img, imgName='res'):
    cv2.imshow('res', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    # load
    refImg = cv2.imread('img/2.jpg')
    alignImg = cv2.imread('img/1.jpg')

    # # 显示原始图像
    # cv2.imshow('refImg', refImg)
    # cv2.imshow('alignImg', alignImg)
    cv2.waitKey(0)

    matchedImg, alignedImg = Align(refImg, alignImg)
    # ShowResizedImg(matchedImg, 0.5)
    # cv2.imshow('refImg', refImg)
    # ShowImg(alignedImg)
    stitchedImg, stitchedImg1 = MyStitch(refImg, alignedImg, alignImg)
    cv2.imwrite('img/stitchedImg.jpg', stitchedImg)
    cv2.imwrite('img/stitchedImg1.jpg', stitchedImg1)
    # stitchedImg = Stitch(refImg, alignImg)
    # cv2.imshow("拼接图像", stitchedImg)
    # cv2.waitKey(0)