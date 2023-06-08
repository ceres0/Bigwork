import sys
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget
from Ui_test import Ui_MainWindow  #导入你写的界面类
 
 
class MyMainWindow(QMainWindow,Ui_MainWindow): #这里也要记得改
    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())  