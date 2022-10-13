# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'display_window.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(542, 707)
        self.label_message = QtWidgets.QLabel(Dialog)
        self.label_message.setGeometry(QtCore.QRect(30, 640, 393, 51))
        self.label_message.setMaximumSize(QtCore.QSize(16777215, 150))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_message.setFont(font)
        self.label_message.setWordWrap(True)
        self.label_message.setObjectName("label_message")
        self.label_image = QtWidgets.QLabel(Dialog)
        self.label_image.setEnabled(True)
        self.label_image.setGeometry(QtCore.QRect(10, 10, 512, 512))
        self.label_image.setFrameShape(QtWidgets.QFrame.Box)
        self.label_image.setAlignment(QtCore.Qt.AlignCenter)
        self.label_image.setObjectName("label_image")
        self.spinBox_clip_limit = QtWidgets.QSpinBox(Dialog)
        self.spinBox_clip_limit.setGeometry(QtCore.QRect(20, 560, 111, 21))
        self.spinBox_clip_limit.setMinimum(1)
        self.spinBox_clip_limit.setMaximum(9999)
        self.spinBox_clip_limit.setProperty("value", 10)
        self.spinBox_clip_limit.setObjectName("spinBox_clip_limit")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 540, 55, 16))
        self.label.setObjectName("label")
        self.spinBox_tile_grid_size = QtWidgets.QSpinBox(Dialog)
        self.spinBox_tile_grid_size.setGeometry(QtCore.QRect(160, 560, 111, 21))
        self.spinBox_tile_grid_size.setMinimum(1)
        self.spinBox_tile_grid_size.setMaximum(9999)
        self.spinBox_tile_grid_size.setProperty("value", 16)
        self.spinBox_tile_grid_size.setObjectName("spinBox_tile_grid_size")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(160, 540, 81, 16))
        self.label_2.setObjectName("label_2")
        self.pushButton_apply_clahe = QtWidgets.QPushButton(Dialog)
        self.pushButton_apply_clahe.setGeometry(QtCore.QRect(282, 537, 131, 71))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_apply_clahe.setFont(font)
        self.pushButton_apply_clahe.setObjectName("pushButton_apply_clahe")
        self.pushButton_restore = QtWidgets.QPushButton(Dialog)
        self.pushButton_restore.setGeometry(QtCore.QRect(420, 550, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_restore.setFont(font)
        self.pushButton_restore.setObjectName("pushButton_restore")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_message.setText(_translate("Dialog", "TextLabel"))
        self.label_image.setText(_translate("Dialog", "TextLabel"))
        self.label.setText(_translate("Dialog", "clip limit"))
        self.label_2.setText(_translate("Dialog", "tile grid size"))
        self.pushButton_apply_clahe.setText(_translate("Dialog", "Apply CLAHE"))
        self.pushButton_restore.setText(_translate("Dialog", "restore"))

