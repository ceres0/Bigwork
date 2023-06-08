import cv2
import numpy as np

global MAX_FEATURES
MAX_FEATURES = 500

def align(refImg, alignImg):

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
    
    # 对齐
    alignedImg = cv2.warpPerspective(alignImg, M, (refImg.shape[1], refImg.shape[0]))
    return matchedImg, alignedImg

if __name__ == '__main__':
    # load
    refImg = cv2.imread('./ref.jpg')
    alignImg = cv2.imread('./align.jpg')

    # # 显示原始图像
    # cv2.imshow('refImg', refImg)
    # cv2.imshow('alignImg', alignImg)
    # cv2.waitKey(0)

    matced_img, alignedImg = align(refImg, alignImg)