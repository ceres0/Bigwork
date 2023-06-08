import sys
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget,QFileDialog,QGraphicsScene,QMessageBox
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import Qt
from Ui_test import Ui_MainWindow   #导入你写的界面类
from Ui_alignDialog import Ui_Dialog as alignDialog  #导入窗体类
from align import Align, MyStitch
import cv2

class MyMainWindow(QMainWindow,Ui_MainWindow): #这里也要记得改
    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        
class AlignDialog(QWidget,alignDialog):
    def __init__(self,parent =None):
        super(AlignDialog,self).__init__(parent)
        self.setupUi(self)
        self.buttonUploadRefImg.clicked.connect(lambda:OpenImage(self, self.refImgView, 'ref'))
        self.buttonUploadAlignImg.clicked.connect(lambda:OpenImage(self, self.alignImgView, 'align'))
        self.buttonAlign.clicked.connect(lambda:AlignImage(self, self.alignedImgView))
        self.buttonStitch.clicked.connect(lambda:StitchImage(self, self.stitchedImgView))
        self.buttonDownloadAlignedImg.clicked.connect(lambda:SaveImage(self, 'aligned'))
        self.buttonDownloadStitchedImg.clicked.connect(lambda:SaveImage(self, 'stitched'))
        self.img = {}
        
def OpenImage(self, imgView, imgG):  # 选择本地图片上传
    imgName, imgType = QFileDialog.getOpenFileName(self, "上传图片", "", "*.jpg;;*.png;;All Files(*)")    # 弹出一个文件选择框，第一个返回值imgName记录选中的文件路径+文件名，第二个返回值imgType记录文件的类型
    if imgName == "":
        return
    img = QPixmap(imgName).scaled(imgView.width()-2, imgView.height()-2)  # 通过文件路径获取图片文件，并设置图片长宽为显示控件大小
    scene = QGraphicsScene()  # 创建一个图片元素
    scene.addPixmap(img)  # 将图片元素添加到图片场景中
    imgView.setScene(scene)  # 将图片场景添加到图片视图中
    imgView.show()  # 显示图片视图
    self.img[imgG] = cv2.imread(imgName)
    
def SaveImage(self, imgG):  # 保存图片
    if imgG not in self.img or self.img[imgG] == '':
        QMessageBox.warning(self, "警告", "未生成{:}图片".format('配准' if imgG == 'aligned' else '拼接'))
        return
    img = self.img[imgG]
    imgName, imgType = QFileDialog.getSaveFileName(self, "保存图片", "", "*.jpg;;*.png;;All Files(*)")    # 弹出一个文件选择框，第一个返回值imgName记录选中的文件路径+文件名，第二个返回值imgType记录文件的类型
    if imgName == "":
        return
    cv2.imwrite(imgName, img)
    QMessageBox.information(self, "提示", "保存成功")
    
def AlignImage(self, imgView):
    if 'ref' not in self.img  or 'align' not in self.img :
        QMessageBox.warning(self, "警告", "未上传{:}".format('基准图片' if 'ref' not in self.img else '待配准图片'))
        return
    _, self.img['aligned'] = Align(self.img['ref'], self.img['align'])
    # cv2.imshow('alignedImg', self.img['aligned'])
    
    img = cv2.cvtColor(self.img['aligned'], cv2.COLOR_BGR2RGB) # 转换图像通道
    rows, cols, channels = img.shape # 获取图像形状
    bytesPerLine = channels * cols # 每行的字节数
    QImg = QImage(img.data, cols, rows, bytesPerLine, QImage.Format_RGB888) # 依据图像数据构造QImage
    img = QPixmap.fromImage(QImg).scaled(
            imgView.width()-2, imgView.height()-2) # 将QImage转换为QPixmap
    scene = QGraphicsScene()  # 创建一个图片元素
    scene.addPixmap(img)  # 将图片元素添加到图片场景中
    imgView.setScene(scene)  # 将图片场景添加到图片视图中
    imgView.show()  # 显示图片视图
    
def StitchImage(self, imgView):
    if 'ref' not in self.img  or 'aligned' not in self.img :
        QMessageBox.warning(self, "警告", "未完成图像配准")
        return
    _, self.img['stitched'] = MyStitch(self.img['ref'], self.img['aligned'])
    
    img = cv2.cvtColor(self.img['stitched'], cv2.COLOR_BGR2RGB) # 转换图像通道
    rows, cols, channels = img.shape # 获取图像形状
    bytesPerLine = channels * cols # 每行的字节数
    QImg = QImage(img.data, cols, rows, bytesPerLine, QImage.Format_RGB888) # 依据图像数据构造QImage
    img = QPixmap.fromImage(QImg).scaled(
            imgView.width()-2, imgView.height()-2) # 将QImage转换为QPixmap
    scene = QGraphicsScene()  # 创建一个图片元素
    scene.addPixmap(img)  # 将图片元素添加到图片场景中
    imgView.setScene(scene)  # 将图片场景添加到图片视图中
    imgView.show()  # 显示图片视图
    
    pass
    
                
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    alignDialog = AlignDialog()
    refImg = None
    alignImg = None
    myWin.show()
    myWin.buttonOpenAlignDialog.clicked.connect(alignDialog.show)
    
    sys.exit(app.exec_())  