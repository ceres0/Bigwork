import sys
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget,QFileDialog,QGraphicsScene
from PyQt5 import QtGui
from Ui_test import Ui_MainWindow   #导入你写的界面类
from Ui_alignDialog import Ui_Dialog as alignDialog  #导入窗体类
 
 
class MyMainWindow(QMainWindow,Ui_MainWindow): #这里也要记得改
    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        
class AlignDialog(QWidget,alignDialog):
    def __init__(self,parent =None):
        super(AlignDialog,self).__init__(parent)
        self.setupUi(self)
        
def openImage(self, imgView):  # 选择本地图片上传
    imgName, imgType = QFileDialog.getOpenFileName(self, "上传图片", "", "*.jpg;;*.png;;All Files(*)")    # 弹出一个文件选择框，第一个返回值imgName记录选中的文件路径+文件名，第二个返回值imgType记录文件的类型
    Img = QtGui.QPixmap(imgName).scaled(imgView.width()-2, imgView.height()-2)  # 通过文件路径获取图片文件，并设置图片长宽为显示控件大小
    scene = QGraphicsScene()  # 创建一个图片元素
    scene.addPixmap(Img)  # 将图片元素添加到图片场景中
    imgView.setScene(scene)  # 将图片场景添加到图片视图中
    imgView.show()  # 显示图片视图
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    alignDialog = AlignDialog()
    myWin.show()
    myWin.buttonOpenAlignDialog.clicked.connect(alignDialog.show)
    alignDialog.buttonUploadRefImg.clicked.connect(lambda:openImage(alignDialog, alignDialog.refImgView))
    alignDialog.buttonUploadAlignImg.clicked.connect(lambda:openImage(alignDialog, alignDialog.alignImgView))
    sys.exit(app.exec_())  