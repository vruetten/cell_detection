#/usr/bin/python
import pyqtgraph as pg
from PyQt5 import QtCore
import numpy as np
import os, sys

class MyImageViewer(pg.ImageView):
    def __init__(self,image, title = 'frame', pos = (0,0)):
        super(MyImageViewer, self).__init__()
        self.title = title
        self.pos = pos
        self.image = image
        self.initUI()
    
    def initUI(self):   
        self.setWindowTitle(self.title)
        self.move(self.pos[0], self.pos[1])
        self.resize(1000,800)
        self.setImage(self.image)
        self.show()

    def keyPressEvent(self, e):
        key_ = e.key()
        if key_ == QtCore.Qt.Key_Escape:
            self.close()
        elif key_ == QtCore.Qt.Key_Q:
            self.close()