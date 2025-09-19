from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        MainWindow.setStyleSheet("background-color: rgb(85, 85, 85);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.SwitchView_1 = QtWidgets.QLabel(self.centralwidget)
        self.SwitchView_1.setGeometry(QtCore.QRect(210, 14, 841, 491))
        self.SwitchView_1.setStyleSheet("color: rgb(156, 156, 156);\n"
"font-weight: bold;\n"
"border: 2px solid black; \n"
"border-radius: 10px;")
        self.SwitchView_1.setText("")
        self.SwitchView_1.setObjectName("SwitchView_1")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(10, 354, 171, 161))
        self.frame_2.setStyleSheet("QFrame {\n"
"    background-color: #4d4d4d;\n"
"    border-radius: 20px;\n"
"}")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.frame_11 = QtWidgets.QFrame(self.frame_2)
        self.frame_11.setGeometry(QtCore.QRect(10, 10, 151, 141))
        self.frame_11.setStyleSheet("QFrame {\n"
"    background-color: rgb(56, 56, 56);\n"
"    border-radius: 20px;\n"
"}")
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.label_62 = QtWidgets.QLabel(self.frame_11)
        self.label_62.setGeometry(QtCore.QRect(0, 10, 151, 121))
        self.label_62.setStyleSheet("image: url(\"assets/compas.png\");\n"
"border-radius: 5px;")
        self.label_62.setText("")
        self.label_62.setObjectName("label_62")
        self.DroneBottomView = QtWidgets.QLabel(self.frame_11)
        self.DroneBottomView.setGeometry(QtCore.QRect(35, 26, 81, 81))
        self.DroneBottomView.setStyleSheet(" image: url(\"assets/Drone 3.png\");\n"
"background: transparent;")
        self.DroneBottomView.setText("")
        self.DroneBottomView.setObjectName("DroneBottomView")
        self.frame_10 = QtWidgets.QFrame(self.centralwidget)
        self.frame_10.setGeometry(QtCore.QRect(10, 14, 171, 161))
        self.frame_10.setStyleSheet("QFrame {\n"
"    background-color: #4d4d4d;\n"
"    border-radius: 20px;\n"
"}")
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.frame_20 = QtWidgets.QFrame(self.frame_10)
        self.frame_20.setGeometry(QtCore.QRect(10, 10, 151, 141))
        self.frame_20.setStyleSheet("QFrame {\n"
"    background-color: rgb(56, 56, 56);\n"
"    border-radius: 20px;\n"
"}")
        self.frame_20.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_20.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_20.setObjectName("frame_20")
        self.label_60 = QtWidgets.QLabel(self.frame_20)
        self.label_60.setGeometry(QtCore.QRect(0, 10, 151, 121))
        self.label_60.setStyleSheet("image: url(\"assets/compas.png\");\n"
"border-radius: 5px;")
        self.label_60.setText("")
        self.label_60.setObjectName("label_60")
        self.DroneTopView = QtWidgets.QLabel(self.frame_20)
        self.DroneTopView.setGeometry(QtCore.QRect(46, 46, 61, 51))
        self.DroneTopView.setStyleSheet(" image: url(\"assets/Drone 2.png\");\n"
"background: transparent;")
        self.DroneTopView.setText("")
        self.DroneTopView.setObjectName("DroneTopView")
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(10, 184, 171, 161))
        self.frame_3.setStyleSheet("QFrame {\n"
"    background-color: #4d4d4d;\n"
"    border-radius: 20px;\n"
"}")
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.frame_19 = QtWidgets.QFrame(self.frame_3)
        self.frame_19.setGeometry(QtCore.QRect(10, 10, 151, 141))
        self.frame_19.setStyleSheet("QFrame {\n"
"    background-color: rgb(56, 56, 56);\n"
"    border-radius: 20px;\n"
"}")
        self.frame_19.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_19.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_19.setObjectName("frame_19")
        self.label_61 = QtWidgets.QLabel(self.frame_19)
        self.label_61.setGeometry(QtCore.QRect(0, 10, 151, 121))
        self.label_61.setStyleSheet("image: url(\"assets/compas.png\");\n"
"border-radius: 5px;")
        self.label_61.setText("")
        self.label_61.setObjectName("label_61")
        self.DroneSideView = QtWidgets.QLabel(self.frame_19)
        self.DroneSideView.setGeometry(QtCore.QRect(36, 33, 81, 71))
        self.DroneSideView.setStyleSheet(" image: url(\"assets/drone-display.png\");\n"
"background: transparent;")
        self.DroneSideView.setText("")
        self.DroneSideView.setObjectName("DroneSideView")
        self.frame_22 = QtWidgets.QFrame(self.centralwidget)
        self.frame_22.setGeometry(QtCore.QRect(1070, 830, 841, 151))
        self.frame_22.setStyleSheet("border-radius: 10px;\n"
"border: 1px solid black;\n"
"background-color: #4d4d4d;")
        self.frame_22.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_22.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_22.setObjectName("frame_22")
        self.label_45 = QtWidgets.QLabel(self.frame_22)
        self.label_45.setGeometry(QtCore.QRect(10, 5, 331, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_45.setFont(font)
        self.label_45.setStyleSheet("color: rgb(156, 156, 156);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_45.setObjectName("label_45")
        self.mcDisplayData = QtWidgets.QTableWidget(self.frame_22)
        self.mcDisplayData.setGeometry(QtCore.QRect(10, 30, 821, 111))
        font = QtGui.QFont()
        font.setPointSize(-1)
        self.mcDisplayData.setFont(font)
        self.mcDisplayData.setStyleSheet("QTableWidget {\n"
"    background-color: #1e1e1e;        /* Warna dasar tabel */\n"
"    alternate-background-color: #2a2a2a;\n"
"    gridline-color: #3c3c3c;          /* Garis antar sel */\n"
"    border: 1px solid #3c3c3c;\n"
"    border-radius: 6px;\n"
"    font-size: 14px;\n"
"    color: #f0f0f0;                   /* Warna teks */\n"
"    selection-background-color: #e63946;  /* Warna highlight (merah) */\n"
"    selection-color: white;\n"
"}\n"
"\n"
"QHeaderView::section {\n"
"    background-color: #2f2f2f;   /* Header gelap */\n"
"    color: #ffffff;\n"
"    padding: 6px;\n"
"    font-weight: bold;\n"
"    border: 1px solid #3c3c3c;\n"
"    border-left: none;\n"
"}\n"
"\n"
"QTableCornerButton::section {\n"
"    background-color: #2f2f2f;\n"
"    border: 1px solid #3c3c3c;\n"
"}\n"
"\n"
"/* Scrollbar vertikal */\n"
"QScrollBar:vertical {\n"
"    border: none;\n"
"    background: #2b2b2b;\n"
"    width: 10px;\n"
"    margin: 0px;\n"
"    border-radius: 4px;\n"
"}\n"
"QScrollBar::handle:vertical {\n"
"    background: #444;\n"
"    min-height: 20px;\n"
"    border-radius: 4px;\n"
"}\n"
"QScrollBar::handle:vertical:hover {\n"
"    background: #666;\n"
"}\n"
"\n"
"/* Scrollbar horizontal */\n"
"QScrollBar:horizontal {\n"
"    border: none;\n"
"    background: #2b2b2b;\n"
"    height: 10px;\n"
"    margin: 0px;\n"
"    border-radius: 4px;\n"
"}\n"
"QScrollBar::handle:horizontal {\n"
"    background: #444;\n"
"    min-width: 20px;\n"
"    border-radius: 4px;\n"
"}\n"
"QScrollBar::handle:horizontal:hover {\n"
"    background: #666;\n"
"}\n"
"")
        self.mcDisplayData.setObjectName("mcDisplayData")
        self.mcDisplayData.setColumnCount(6)
        self.mcDisplayData.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.mcDisplayData.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.mcDisplayData.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.mcDisplayData.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.mcDisplayData.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.mcDisplayData.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.mcDisplayData.setHorizontalHeaderItem(5, item)
        self.frame_17 = QtWidgets.QFrame(self.centralwidget)
        self.frame_17.setGeometry(QtCore.QRect(1070, 14, 841, 185))
        self.frame_17.setStyleSheet("QFrame {\n"
"    background-color: #4d4d4d;\n"
"    border-radius: 20px;\n"
"}")
        self.frame_17.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_17.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_17.setObjectName("frame_17")
        self.CommandConnect = QtWidgets.QPushButton(self.frame_17)
        self.CommandConnect.setGeometry(QtCore.QRect(10, 20, 141, 41))
        self.CommandConnect.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.CommandConnect.setStyleSheet("QPushButton {\n"
"    background-color: rgb(65, 65, 65);\n"
"    border: 2px solid black;\n"
"    border-radius: 8px;\n"
"    color: white;\n"
"    font-weight: bold;\n"
"    padding: 6px 12px;\n"
"    text-align: right;\n"
"}\n"
"\n"
"/* Hover effect */\n"
"QPushButton:hover {\n"
"    background-color: rgb(90, 90, 90);   /* lebih terang */\n"
"    border: 2px solid rgb(120, 120, 120);\n"
"    color: #e0e0e0;                      /* teks sedikit lebih terang */\n"
"}\n"
"\n"
"/* Saat ditekan */\n"
"QPushButton:pressed {\n"
"    background-color: rgb(40, 40, 40);   /* lebih gelap */\n"
"    border: 2px solid rgb(100, 100, 100);\n"
"    color: #cccccc;\n"
"}\n"
"")
        self.CommandConnect.setObjectName("CommandConnect")
        self.CommandOnline = QtWidgets.QLabel(self.frame_17)
        self.CommandOnline.setGeometry(QtCore.QRect(20, 30, 31, 21))
        self.CommandOnline.setStyleSheet("background-color: rgb(0, 255, 0);\n"
"border: 0px gray;\n"
"border-radius: 10px;")
        self.CommandOnline.setText("")
        self.CommandOnline.setObjectName("CommandOnline")
        self.label = QtWidgets.QLabel(self.frame_17)
        self.label.setGeometry(QtCore.QRect(150, 10, 201, 61))
        self.label.setStyleSheet(" image: url(\"assets/LOGO R BG-012.png\");")
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.frame_17)
        self.label_2.setGeometry(QtCore.QRect(270, 10, 341, 191))
        self.label_2.setStyleSheet(" image: url(\"assets/drone-display.png\");")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_11 = QtWidgets.QLabel(self.frame_17)
        self.label_11.setGeometry(QtCore.QRect(10, 80, 61, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.frame_17)
        self.label_12.setGeometry(QtCore.QRect(10, 110, 61, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.frame_17)
        self.label_13.setGeometry(QtCore.QRect(10, 140, 161, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_13.setFont(font)
        self.label_13.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.frame_17)
        self.label_14.setGeometry(QtCore.QRect(590, 20, 231, 151))
        font = QtGui.QFont()
        font.setPointSize(30)
        font.setBold(True)
        font.setWeight(75)
        self.label_14.setFont(font)
        self.label_14.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_14.setWordWrap(True)
        self.label_14.setObjectName("label_14")
        self.CommandStatus = QtWidgets.QLabel(self.frame_17)
        self.CommandStatus.setGeometry(QtCore.QRect(80, 80, 141, 21))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.CommandStatus.setFont(font)
        self.CommandStatus.setStyleSheet("color: rgb(209, 207, 207);\n"
"border: 0px gray; ")
        self.CommandStatus.setObjectName("CommandStatus")
        self.CommandMode = QtWidgets.QLabel(self.frame_17)
        self.CommandMode.setGeometry(QtCore.QRect(80, 110, 121, 21))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.CommandMode.setFont(font)
        self.CommandMode.setStyleSheet("color: rgb(209, 207, 207);\n"
"border: 0px gray; ")
        self.CommandMode.setObjectName("CommandMode")
        self.CommandControl = QtWidgets.QLabel(self.frame_17)
        self.CommandControl.setGeometry(QtCore.QRect(80, 140, 111, 21))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.CommandControl.setFont(font)
        self.CommandControl.setStyleSheet("color: rgb(209, 207, 207);\n"
"border: 0px gray; ")
        self.CommandControl.setObjectName("CommandControl")
        self.btAutonomousEmergency = QtWidgets.QPushButton(self.frame_17)
        self.btAutonomousEmergency.setGeometry(QtCore.QRect(760, 10, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btAutonomousEmergency.setFont(font)
        self.btAutonomousEmergency.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btAutonomousEmergency.setStyleSheet("QPushButton {\n"
"    border: none;\n"
"    border-radius: 8px;\n"
"    image: url(\"assets/emergency.png\");\n"
"    background-repeat: no-repeat;\n"
"    background-position: center;\n"
"    background-color: transparent; /* default transparan */\n"
"    min-width: 20px;\n"
"    min-height: 20px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    image: url(\"assets/emergency.png\");\n"
"    background-repeat: no-repeat;\n"
"    background-position: center;\n"
"    background-color: rgba(0, 0, 0, 50); /* efek gelap hover */\n"
"    border-radius: 8px;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"     image: url(\"assets/emergency.png\");\n"
"    background-repeat: no-repeat;\n"
"    background-position: center;\n"
"    background-color: rgba(0, 0, 0, 100); /* lebih gelap ketika ditekan */\n"
"    border-radius: 8px;\n"
"}\n"
"")
        self.btAutonomousEmergency.setText("")
        self.btAutonomousEmergency.setObjectName("btAutonomousEmergency")
        self.label_2.raise_()
        self.label.raise_()
        self.label_11.raise_()
        self.label_12.raise_()
        self.label_13.raise_()
        self.label_14.raise_()
        self.CommandStatus.raise_()
        self.CommandMode.raise_()
        self.CommandControl.raise_()
        self.btAutonomousEmergency.raise_()
        self.CommandConnect.raise_()
        self.CommandOnline.raise_()
        self.frame_8 = QtWidgets.QFrame(self.centralwidget)
        self.frame_8.setGeometry(QtCore.QRect(210, 514, 841, 471))
        self.frame_8.setStyleSheet("QFrame {\n"
"    background-color: #4d4d4d;\n"
"    border-radius: 20px;\n"
"}")
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.SwitchView_2 = QtWidgets.QLabel(self.frame_8)
        self.SwitchView_2.setGeometry(QtCore.QRect(10, 10, 451, 421))
        self.SwitchView_2.setStyleSheet("color: rgb(156, 156, 156);\n"
"font-weight: bold;\n"
"border: 2px solid black; \n"
"border-radius: 10px;")
        self.SwitchView_2.setText("")
        self.SwitchView_2.setObjectName("SwitchView_2")
        self.frame_12 = QtWidgets.QFrame(self.frame_8)
        self.frame_12.setGeometry(QtCore.QRect(670, 170, 161, 131))
        self.frame_12.setStyleSheet("QFrame {\n"
"    background-color: rgb(50, 50, 50);\n"
"    border-radius: 20px;\n"
"}")
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.label_25 = QtWidgets.QLabel(self.frame_12)
        self.label_25.setGeometry(QtCore.QRect(10, 10, 61, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_25.setFont(font)
        self.label_25.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_25.setWordWrap(True)
        self.label_25.setIndent(3)
        self.label_25.setOpenExternalLinks(False)
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(self.frame_12)
        self.label_26.setGeometry(QtCore.QRect(20, 60, 31, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_26.setFont(font)
        self.label_26.setStyleSheet("color: white;\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_26.setObjectName("label_26")
        self.label_27 = QtWidgets.QLabel(self.frame_12)
        self.label_27.setGeometry(QtCore.QRect(20, 80, 31, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_27.setFont(font)
        self.label_27.setStyleSheet("color: white;\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_27.setObjectName("label_27")
        self.DronePositionX = QtWidgets.QLabel(self.frame_12)
        self.DronePositionX.setGeometry(QtCore.QRect(50, 60, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.DronePositionX.setFont(font)
        self.DronePositionX.setStyleSheet("color: white;\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.DronePositionX.setObjectName("DronePositionX")
        self.DronePositionY = QtWidgets.QLabel(self.frame_12)
        self.DronePositionY.setGeometry(QtCore.QRect(50, 80, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.DronePositionY.setFont(font)
        self.DronePositionY.setStyleSheet("color: white;\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.DronePositionY.setObjectName("DronePositionY")
        self.frame_13 = QtWidgets.QFrame(self.frame_8)
        self.frame_13.setGeometry(QtCore.QRect(480, 170, 181, 81))
        self.frame_13.setStyleSheet("QFrame {\n"
"    background-color: rgb(50, 50, 50);\n"
"    border-radius: 20px;\n"
"}")
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.label_19 = QtWidgets.QLabel(self.frame_13)
        self.label_19.setGeometry(QtCore.QRect(20, 10, 61, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_19.setFont(font)
        self.label_19.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_19.setObjectName("label_19")
        self.DroneMode = QtWidgets.QLabel(self.frame_13)
        self.DroneMode.setGeometry(QtCore.QRect(20, 40, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.DroneMode.setFont(font)
        self.DroneMode.setStyleSheet("color: white;\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.DroneMode.setObjectName("DroneMode")
        self.frame_16 = QtWidgets.QFrame(self.frame_8)
        self.frame_16.setGeometry(QtCore.QRect(480, 70, 351, 91))
        self.frame_16.setStyleSheet("QFrame {\n"
"    background-color: rgb(50, 50, 50);\n"
"    border-radius: 20px;\n"
"}")
        self.frame_16.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_16.setObjectName("frame_16")
        self.label_18 = QtWidgets.QLabel(self.frame_16)
        self.label_18.setGeometry(QtCore.QRect(20, 10, 61, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_18.setFont(font)
        self.label_18.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_18.setObjectName("label_18")
        self.DroneBattery = QtWidgets.QProgressBar(self.frame_16)
        self.DroneBattery.setGeometry(QtCore.QRect(20, 40, 311, 23))
        self.DroneBattery.setCursor(QtGui.QCursor(QtCore.Qt.UpArrowCursor))
        self.DroneBattery.setStyleSheet("QProgressBar {\n"
"    border: 2px solid #555;\n"
"    border-radius: 10px;   /* sisi luar bar rounded */\n"
"    text-align: center;\n"
"    font-weight: bold;\n"
"    background-color: #2b2b2b;\n"
"    color: white;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"    background-color: #3ddb55;  /* hijau isi progress */\n"
"    border-radius: 9px;        /* sisi isi juga rounded */\n"
"}\n"
"")
        self.DroneBattery.setProperty("value", 75)
        self.DroneBattery.setAlignment(QtCore.Qt.AlignCenter)
        self.DroneBattery.setTextVisible(True)
        self.DroneBattery.setOrientation(QtCore.Qt.Horizontal)
        self.DroneBattery.setInvertedAppearance(False)
        self.DroneBattery.setTextDirection(QtWidgets.QProgressBar.TopToBottom)
        self.DroneBattery.setObjectName("DroneBattery")
        self.frame_14 = QtWidgets.QFrame(self.frame_8)
        self.frame_14.setGeometry(QtCore.QRect(670, 310, 161, 121))
        self.frame_14.setStyleSheet("QFrame {\n"
"    background-color: rgb(50, 50, 50);\n"
"    border-radius: 20px;\n"
"}")
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.DroneSpeedX = QtWidgets.QLabel(self.frame_14)
        self.DroneSpeedX.setGeometry(QtCore.QRect(50, 40, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.DroneSpeedX.setFont(font)
        self.DroneSpeedX.setStyleSheet("color: white;\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.DroneSpeedX.setObjectName("DroneSpeedX")
        self.label_31 = QtWidgets.QLabel(self.frame_14)
        self.label_31.setGeometry(QtCore.QRect(10, 10, 61, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_31.setFont(font)
        self.label_31.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_31.setWordWrap(True)
        self.label_31.setIndent(3)
        self.label_31.setOpenExternalLinks(False)
        self.label_31.setObjectName("label_31")
        self.label_32 = QtWidgets.QLabel(self.frame_14)
        self.label_32.setGeometry(QtCore.QRect(20, 60, 31, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_32.setFont(font)
        self.label_32.setStyleSheet("color: white;\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_32.setObjectName("label_32")
        self.label_33 = QtWidgets.QLabel(self.frame_14)
        self.label_33.setGeometry(QtCore.QRect(20, 40, 31, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_33.setFont(font)
        self.label_33.setStyleSheet("color: white;\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_33.setObjectName("label_33")
        self.DroneSpeedY = QtWidgets.QLabel(self.frame_14)
        self.DroneSpeedY.setGeometry(QtCore.QRect(50, 60, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.DroneSpeedY.setFont(font)
        self.DroneSpeedY.setStyleSheet("color: white;\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.DroneSpeedY.setObjectName("DroneSpeedY")
        self.label_35 = QtWidgets.QLabel(self.frame_14)
        self.label_35.setGeometry(QtCore.QRect(20, 80, 31, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_35.setFont(font)
        self.label_35.setStyleSheet("color: white;\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_35.setObjectName("label_35")
        self.DroneSpeedZ = QtWidgets.QLabel(self.frame_14)
        self.DroneSpeedZ.setGeometry(QtCore.QRect(50, 80, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.DroneSpeedZ.setFont(font)
        self.DroneSpeedZ.setStyleSheet("color: white;\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.DroneSpeedZ.setObjectName("DroneSpeedZ")
        self.frame_15 = QtWidgets.QFrame(self.frame_8)
        self.frame_15.setGeometry(QtCore.QRect(480, 260, 181, 81))
        self.frame_15.setStyleSheet("QFrame {\n"
"    background-color: rgb(50, 50, 50);\n"
"    border-radius: 20px;\n"
"}")
        self.frame_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_15.setObjectName("frame_15")
        self.label_21 = QtWidgets.QLabel(self.frame_15)
        self.label_21.setGeometry(QtCore.QRect(20, 10, 61, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_21.setFont(font)
        self.label_21.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_21.setObjectName("label_21")
        self.DroneHeight = QtWidgets.QLabel(self.frame_15)
        self.DroneHeight.setGeometry(QtCore.QRect(20, 40, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.DroneHeight.setFont(font)
        self.DroneHeight.setStyleSheet("color: white;\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.DroneHeight.setObjectName("DroneHeight")
        self.frame_18 = QtWidgets.QFrame(self.frame_8)
        self.frame_18.setGeometry(QtCore.QRect(480, 350, 181, 81))
        self.frame_18.setStyleSheet("QFrame {\n"
"    background-color: rgb(50, 50, 50);\n"
"    border-radius: 20px;\n"
"}")
        self.frame_18.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_18.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_18.setObjectName("frame_18")
        self.label_23 = QtWidgets.QLabel(self.frame_18)
        self.label_23.setGeometry(QtCore.QRect(20, 10, 91, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_23.setFont(font)
        self.label_23.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_23.setObjectName("label_23")
        self.DroneFlightTime = QtWidgets.QLabel(self.frame_18)
        self.DroneFlightTime.setGeometry(QtCore.QRect(20, 40, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.DroneFlightTime.setFont(font)
        self.DroneFlightTime.setStyleSheet("color: white;\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.DroneFlightTime.setObjectName("DroneFlightTime")
        self.label_9 = QtWidgets.QLabel(self.frame_8)
        self.label_9.setGeometry(QtCore.QRect(480, -20, 351, 111))
        self.label_9.setStyleSheet(" image: url(\"assets/LOGO R BG-012.png\");")
        self.label_9.setText("")
        self.label_9.setObjectName("label_9")
        self.DroneSwitch = QtWidgets.QPushButton(self.frame_8)
        self.DroneSwitch.setGeometry(QtCore.QRect(10, 430, 451, 31))
        self.DroneSwitch.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.DroneSwitch.setStyleSheet("QPushButton {\n"
"    background-color: rgb(65, 65, 65);\n"
"    border: 2px solid black;\n"
"    border-radius: 8px;\n"
"    color: white;\n"
"    font-weight: bold;\n"
"    padding: 6px 12px;\n"
"    text-align: center;\n"
"}\n"
"\n"
"/* Hover effect */\n"
"QPushButton:hover {\n"
"    background-color: rgb(90, 90, 90);   \n"
"    border: 2px solid rgb(120, 120, 120);\n"
"    color: #e0e0e0;                    \n"
"}\n"
"\n"
"/* Saat ditekan */\n"
"QPushButton:pressed {\n"
"    background-color: rgb(40, 40, 40);   \n"
"    border: 2px solid rgb(100, 100, 100);\n"
"    color: #cccccc;\n"
"}\n"
"")
        self.DroneSwitch.setObjectName("DroneSwitch")
        self.label_9.raise_()
        self.SwitchView_2.raise_()
        self.frame_12.raise_()
        self.frame_13.raise_()
        self.frame_16.raise_()
        self.frame_14.raise_()
        self.frame_15.raise_()
        self.frame_18.raise_()
        self.DroneSwitch.raise_()
        self.frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_4.setGeometry(QtCore.QRect(10, 524, 171, 461))
        self.frame_4.setStyleSheet("QFrame {\n"
"    border-radius: 20px;\n"
"background-color: #4d4d4d;\n"
"}")
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.DroneAltitude = QtWidgets.QSlider(self.frame_4)
        self.DroneAltitude.setGeometry(QtCore.QRect(10, 60, 151, 391))
        self.DroneAltitude.setStyleSheet("QSlider {\n"
"   background-color: #4d4d4d;\n"
"    border-radius: 10px;\n"
"    height: 300px;\n"
"    image: url(\"assets/altitude.png\");\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
"    background: transparent;\n"
"    width: 6px;\n"
"    margin: 0 20px;\n"
"}\n"
"\n"
"QSlider::sub-page:vertical, \n"
"QSlider::add-page:vertical {\n"
"    background: transparent;\n"
"}\n"
"\n"
"/* Handle kotak orange */\n"
"QSlider::handle:vertical {\n"
"    background: #ff9933;   \n"
"    border: 2px solid #cc6600;\n"
"    height: 28px;\n"
"    width: 136px;           \n"
"    border-radius: 6px;    \n"
"    margin: 30px -20px;\n"
"    image-position: center;\n"
"}\n"
"\n"
"QSlider::handle:vertical:hover {\n"
"    background: #ffb366;\n"
"    border: 2px solid #ff6600;\n"
"}\n"
"\n"
"/* Garis meteran */\n"
"QSlider::tick:vertical {\n"
"    background: rgb(83, 83, 83);\n"
"    background-color: rgb(83, 83, 83);\n"
"    width: 2px;\n"
"    height: 12px;\n"
"    margin: 2px 0;\n"
"}\n"
"")
        self.DroneAltitude.setMaximum(300)
        self.DroneAltitude.setOrientation(QtCore.Qt.Vertical)
        self.DroneAltitude.setObjectName("DroneAltitude")
        self.label_37 = QtWidgets.QLabel(self.frame_4)
        self.label_37.setGeometry(QtCore.QRect(50, 20, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.label_37.setFont(font)
        self.label_37.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_37.setWordWrap(True)
        self.label_37.setIndent(3)
        self.label_37.setOpenExternalLinks(False)
        self.label_37.setObjectName("label_37")
        self.imgDetector = QtWidgets.QLabel(self.centralwidget)
        self.imgDetector.setGeometry(QtCore.QRect(1070, 210, 421, 241))
        self.imgDetector.setStyleSheet("\n"
"    border: 1px solid black;\n"
"    border-radius: 10px;")
        self.imgDetector.setText("")
        self.imgDetector.setObjectName("imgDetector")
        self.imgCapture = QtWidgets.QLabel(self.centralwidget)
        self.imgCapture.setGeometry(QtCore.QRect(1500, 210, 411, 241))
        self.imgCapture.setStyleSheet("\n"
"    border: 1px solid black;\n"
"    border-radius: 10px;")
        self.imgCapture.setText("")
        self.imgCapture.setObjectName("imgCapture")
        self.frame_30 = QtWidgets.QFrame(self.centralwidget)
        self.frame_30.setGeometry(QtCore.QRect(1070, 460, 841, 361))
        self.frame_30.setStyleSheet("border-radius: 10px;\n"
"border: 1px solid black;\n"
"background-color: #4d4d4d;")
        self.frame_30.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_30.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_30.setObjectName("frame_30")
        self.label_16 = QtWidgets.QLabel(self.frame_30)
        self.label_16.setGeometry(QtCore.QRect(50, 10, 231, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_16.setFont(font)
        self.label_16.setStyleSheet("color: rgb(156, 156, 156);\n"
"font-weight: bold;\n"
"border: 0px solid gray;")
        self.label_16.setObjectName("label_16")
        self.btQrcodeOnline_3 = QtWidgets.QLabel(self.frame_30)
        self.btQrcodeOnline_3.setGeometry(QtCore.QRect(10, 10, 31, 21))
        self.btQrcodeOnline_3.setStyleSheet("background-color: rgb(0, 255, 0);\n"
"border: 0px gray;")
        self.btQrcodeOnline_3.setText("")
        self.btQrcodeOnline_3.setObjectName("btQrcodeOnline_3")
        self.frame_5 = QtWidgets.QFrame(self.frame_30)
        self.frame_5.setGeometry(QtCore.QRect(590, 40, 241, 311))
        self.frame_5.setStyleSheet("background-color: rgb(56, 56, 56);")
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.label_46 = QtWidgets.QLabel(self.frame_5)
        self.label_46.setGeometry(QtCore.QRect(10, 10, 221, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_46.setFont(font)
        self.label_46.setStyleSheet("color: rgb(156, 156, 156);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_46.setObjectName("label_46")
        self.tbDebugging = QtWidgets.QTextBrowser(self.frame_5)
        self.tbDebugging.setGeometry(QtCore.QRect(10, 40, 221, 251))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        self.tbDebugging.setFont(font)
        self.tbDebugging.setStyleSheet("color: white;\n"
"border: 0px;")
        self.tbDebugging.setObjectName("tbDebugging")
        self.frame_6 = QtWidgets.QFrame(self.frame_30)
        self.frame_6.setGeometry(QtCore.QRect(10, 40, 571, 311))
        self.frame_6.setStyleSheet("background-color: rgb(56, 56, 56);")
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.frameSingleCommand = QtWidgets.QFrame(self.frame_6)
        self.frameSingleCommand.setGeometry(QtCore.QRect(10, 20, 551, 81))
        self.frameSingleCommand.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameSingleCommand.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameSingleCommand.setObjectName("frameSingleCommand")
        self.scHover = QtWidgets.QPushButton(self.frameSingleCommand)
        self.scHover.setGeometry(QtCore.QRect(340, 10, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setBold(True)
        font.setWeight(75)
        self.scHover.setFont(font)
        self.scHover.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.scHover.setStyleSheet("QPushButton {\n"
"    background-color: rgb(65, 65, 65);\n"
"    border: 2px solid black;\n"
"    border-radius: 8px;\n"
"    color: white;\n"
"    font-weight: bold;\n"
"    padding: 6px 12px;\n"
"    text-align: center;\n"
"}\n"
"\n"
"/* Hover effect */\n"
"QPushButton:hover {\n"
"    background-color: rgb(90, 90, 90);   /* lebih terang */\n"
"    border: 2px solid rgb(120, 120, 120);\n"
"    color: #e0e0e0;                      /* teks sedikit lebih terang */\n"
"}\n"
"\n"
"/* Saat ditekan */\n"
"QPushButton:pressed {\n"
"    background-color: rgb(40, 40, 40);   /* lebih gelap */\n"
"    border: 2px solid rgb(100, 100, 100);\n"
"    color: #cccccc;\n"
"}\n"
"")
        self.scHover.setObjectName("scHover")
        self.scSendGoto = QtWidgets.QPushButton(self.frameSingleCommand)
        self.scSendGoto.setGeometry(QtCore.QRect(340, 45, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setBold(True)
        font.setWeight(75)
        self.scSendGoto.setFont(font)
        self.scSendGoto.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.scSendGoto.setStyleSheet("QPushButton {\n"
"    background-color: rgb(65, 65, 65);\n"
"    border: 2px solid black;\n"
"    border-radius: 8px;\n"
"    color: white;\n"
"    font-weight: bold;\n"
"    padding: 6px 12px;\n"
"    text-align: center;\n"
"}\n"
"\n"
"/* Hover effect */\n"
"QPushButton:hover {\n"
"    background-color: rgb(90, 90, 90);   /* lebih terang */\n"
"    border: 2px solid rgb(120, 120, 120);\n"
"    color: #e0e0e0;                      /* teks sedikit lebih terang */\n"
"}\n"
"\n"
"/* Saat ditekan */\n"
"QPushButton:pressed {\n"
"    background-color: rgb(40, 40, 40);   /* lebih gelap */\n"
"    border: 2px solid rgb(100, 100, 100);\n"
"    color: #cccccc;\n"
"}\n"
"")
        self.scSendGoto.setObjectName("scSendGoto")
        self.scEdit = QtWidgets.QPushButton(self.frameSingleCommand)
        self.scEdit.setGeometry(QtCore.QRect(230, 10, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setBold(True)
        font.setWeight(75)
        self.scEdit.setFont(font)
        self.scEdit.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.scEdit.setStyleSheet("QPushButton {\n"
"    background-color: rgb(65, 65, 65);\n"
"    border: 2px solid black;\n"
"    border-radius: 10%;\n"
"    color: white;\n"
"    font-weight: bold;\n"
"    padding: 6px 12px;\n"
"    text-align: center;\n"
"}\n"
"\n"
"/* Hover effect */\n"
"QPushButton:hover {\n"
"    background-color: rgb(90, 90, 90);   /* lebih terang */\n"
"    border: 2px solid rgb(120, 120, 120);\n"
"    color: #e0e0e0;                      /* teks sedikit lebih terang */\n"
"}\n"
"\n"
"/* Saat ditekan */\n"
"QPushButton:pressed {\n"
"    background-color: rgb(40, 40, 40);   /* lebih gelap */\n"
"    border: 2px solid rgb(100, 100, 100);\n"
"    color: #cccccc;\n"
"}\n"
"")
        self.scEdit.setObjectName("scEdit")
        self.scLanding = QtWidgets.QCheckBox(self.frameSingleCommand)
        self.scLanding.setGeometry(QtCore.QRect(440, 50, 101, 21))
        self.scLanding.setStyleSheet("QCheckBox {\n"
"    spacing: 8px;\n"
"    font-size: 10px;\n"
"    color: #f0f0f0;\n"
"    font-weight: bold;\n"
"    border: none;\n"
"}\n"
"\n"
"QCheckBox::indicator {\n"
"    width: 18px;\n"
"    height: 18px;\n"
"    border-radius: 4px;\n"
"    border: 2px solid #4CAF50;\n"
"    background-color: #2b2b2b;\n"
"}\n"
"\n"
"QCheckBox::indicator:hover {\n"
"    border: 2px solid #66BB6A;\n"
"    background-color: #333333;\n"
"}\n"
"\n"
"QCheckBox::indicator:checked {\n"
"    background-color: #4CAF50;\n"
"    border: 2px solid #66BB6A;\n"
"}\n"
"\n"
"QCheckBox::indicator:disabled {\n"
"    background-color: #555555;\n"
"    border: 2px solid #777777;\n"
"}\n"
"")
        self.scLanding.setObjectName("scLanding")
        self.scYawEnable = QtWidgets.QCheckBox(self.frameSingleCommand)
        self.scYawEnable.setGeometry(QtCore.QRect(440, 20, 101, 21))
        self.scYawEnable.setStyleSheet("QCheckBox {\n"
"    spacing: 8px;\n"
"    font-size: 10px;\n"
"    color: #f0f0f0;\n"
"    font-weight: bold;\n"
"    border: none;\n"
"}\n"
"\n"
"QCheckBox::indicator {\n"
"    width: 18px;\n"
"    height: 18px;\n"
"    border-radius: 4px;\n"
"    border: 2px solid #4CAF50;\n"
"    background-color: #2b2b2b;\n"
"}\n"
"\n"
"QCheckBox::indicator:hover {\n"
"    border: 2px solid #66BB6A;\n"
"    background-color: #333333;\n"
"}\n"
"\n"
"QCheckBox::indicator:checked {\n"
"    background-color: #4CAF50;\n"
"    border: 2px solid #66BB6A;\n"
"}\n"
"\n"
"QCheckBox::indicator:disabled {\n"
"    background-color: #555555;\n"
"    border: 2px solid #777777;\n"
"}\n"
"")
        self.scYawEnable.setObjectName("scYawEnable")
        self.label_15 = QtWidgets.QLabel(self.frameSingleCommand)
        self.label_15.setGeometry(QtCore.QRect(15, 20, 51, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_15.setFont(font)
        self.label_15.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_15.setObjectName("label_15")
        self.scPositionX = QtWidgets.QLabel(self.frameSingleCommand)
        self.scPositionX.setGeometry(QtCore.QRect(100, 20, 51, 21))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.scPositionX.setFont(font)
        self.scPositionX.setStyleSheet("color: rgb(209, 207, 207);\n"
"border: 0px gray; ")
        self.scPositionX.setObjectName("scPositionX")
        self.label_17 = QtWidgets.QLabel(self.frameSingleCommand)
        self.label_17.setGeometry(QtCore.QRect(80, 20, 16, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_17.setFont(font)
        self.label_17.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_17.setObjectName("label_17")
        self.scPositionY = QtWidgets.QLabel(self.frameSingleCommand)
        self.scPositionY.setGeometry(QtCore.QRect(180, 20, 51, 21))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.scPositionY.setFont(font)
        self.scPositionY.setStyleSheet("color: rgb(209, 207, 207);\n"
"border: 0px gray; ")
        self.scPositionY.setObjectName("scPositionY")
        self.label_20 = QtWidgets.QLabel(self.frameSingleCommand)
        self.label_20.setGeometry(QtCore.QRect(160, 20, 16, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_20.setFont(font)
        self.label_20.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_20.setObjectName("label_20")
        self.label_22 = QtWidgets.QLabel(self.frameSingleCommand)
        self.label_22.setGeometry(QtCore.QRect(15, 50, 71, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_22.setFont(font)
        self.label_22.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_22.setObjectName("label_22")
        self.scOrientation = QtWidgets.QLabel(self.frameSingleCommand)
        self.scOrientation.setGeometry(QtCore.QRect(100, 50, 71, 21))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.scOrientation.setFont(font)
        self.scOrientation.setStyleSheet("color: rgb(209, 207, 207);\n"
"border: 0px gray; ")
        self.scOrientation.setObjectName("scOrientation")
        self.scClearMarker = QtWidgets.QPushButton(self.frameSingleCommand)
        self.scClearMarker.setGeometry(QtCore.QRect(230, 45, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setBold(True)
        font.setWeight(75)
        self.scClearMarker.setFont(font)
        self.scClearMarker.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.scClearMarker.setStyleSheet("QPushButton {\n"
"    background-color: rgb(65, 65, 65);\n"
"    border: 2px solid black;\n"
"    border-radius: 8px;\n"
"    color: white;\n"
"    font-weight: bold;\n"
"    padding: 6px 12px;\n"
"    text-align: center;\n"
"}\n"
"\n"
"/* Hover effect */\n"
"QPushButton:hover {\n"
"    background-color: rgb(90, 90, 90);   /* lebih terang */\n"
"    border: 2px solid rgb(120, 120, 120);\n"
"    color: #e0e0e0;                      /* teks sedikit lebih terang */\n"
"}\n"
"\n"
"/* Saat ditekan */\n"
"QPushButton:pressed {\n"
"    background-color: rgb(40, 40, 40);   /* lebih gelap */\n"
"    border: 2px solid rgb(100, 100, 100);\n"
"    color: #cccccc;\n"
"}\n"
"")
        self.scClearMarker.setObjectName("scClearMarker")
        self.frameMultipleCommand = QtWidgets.QFrame(self.frame_6)
        self.frameMultipleCommand.setGeometry(QtCore.QRect(10, 120, 551, 181))
        self.frameMultipleCommand.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameMultipleCommand.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameMultipleCommand.setObjectName("frameMultipleCommand")
        self.mcViewControls = QtWidgets.QCheckBox(self.frameMultipleCommand)
        self.mcViewControls.setGeometry(QtCore.QRect(15, 90, 161, 21))
        font = QtGui.QFont()
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.mcViewControls.setFont(font)
        self.mcViewControls.setStyleSheet("QCheckBox {\n"
"    spacing: 8px;\n"
"    font-size: 11px;\n"
"    color: #f0f0f0;\n"
"    font-weight: bold;\n"
"    border: none;\n"
"}\n"
"\n"
"QCheckBox::indicator {\n"
"    width: 18px;\n"
"    height: 18px;\n"
"    border-radius: 4px;\n"
"    border: 2px solid #4CAF50;\n"
"    background-color: #2b2b2b;\n"
"}\n"
"\n"
"QCheckBox::indicator:hover {\n"
"    border: 2px solid #66BB6A;\n"
"    background-color: #333333;\n"
"}\n"
"\n"
"QCheckBox::indicator:checked {\n"
"    background-color: #4CAF50;\n"
"    border: 2px solid #66BB6A;\n"
"}\n"
"\n"
"QCheckBox::indicator:disabled {\n"
"    background-color: #555555;\n"
"    border: 2px solid #777777;\n"
"}\n"
"")
        self.mcViewControls.setObjectName("mcViewControls")
        self.label_28 = QtWidgets.QLabel(self.frameMultipleCommand)
        self.label_28.setGeometry(QtCore.QRect(15, 66, 131, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_28.setFont(font)
        self.label_28.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_28.setObjectName("label_28")
        self.mcSaveMaps = QtWidgets.QPushButton(self.frameMultipleCommand)
        self.mcSaveMaps.setGeometry(QtCore.QRect(190, 105, 186, 31))
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setBold(True)
        font.setWeight(75)
        self.mcSaveMaps.setFont(font)
        self.mcSaveMaps.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.mcSaveMaps.setStyleSheet("QPushButton {\n"
"    background-color: rgb(65, 65, 65);\n"
"    border: 2px solid black;\n"
"    border-radius: 8px;\n"
"    color: white;\n"
"    font-weight: bold;\n"
"    padding: 6px 12px;\n"
"    text-align: center;\n"
"}\n"
"\n"
"/* Hover effect */\n"
"QPushButton:hover {\n"
"    background-color: rgb(90, 90, 90);   /* lebih terang */\n"
"    border: 2px solid rgb(120, 120, 120);\n"
"    color: #e0e0e0;                      /* teks sedikit lebih terang */\n"
"}\n"
"\n"
"/* Saat ditekan */\n"
"QPushButton:pressed {\n"
"    background-color: rgb(40, 40, 40);   /* lebih gelap */\n"
"    border: 2px solid rgb(100, 100, 100);\n"
"    color: #cccccc;\n"
"}\n"
"")
        self.mcSaveMaps.setObjectName("mcSaveMaps")
        self.label_29 = QtWidgets.QLabel(self.frameMultipleCommand)
        self.label_29.setGeometry(QtCore.QRect(15, 16, 131, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_29.setFont(font)
        self.label_29.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_29.setObjectName("label_29")
        self.mcEditMode = QtWidgets.QCheckBox(self.frameMultipleCommand)
        self.mcEditMode.setGeometry(QtCore.QRect(15, 40, 161, 21))
        font = QtGui.QFont()
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.mcEditMode.setFont(font)
        self.mcEditMode.setStyleSheet("QCheckBox {\n"
"    spacing: 8px;\n"
"    font-size: 11px;\n"
"    color: #f0f0f0;\n"
"    font-weight: bold;\n"
"    border: none;\n"
"}\n"
"\n"
"QCheckBox::indicator {\n"
"    width: 18px;\n"
"    height: 18px;\n"
"    border-radius: 4px;\n"
"    border: 2px solid #4CAF50;\n"
"    background-color: #2b2b2b;\n"
"}\n"
"\n"
"QCheckBox::indicator:hover {\n"
"    border: 2px solid #66BB6A;\n"
"    background-color: #333333;\n"
"}\n"
"\n"
"QCheckBox::indicator:checked {\n"
"    background-color: #4CAF50;\n"
"    border: 2px solid #66BB6A;\n"
"}\n"
"\n"
"QCheckBox::indicator:disabled {\n"
"    background-color: #555555;\n"
"    border: 2px solid #777777;\n"
"}\n"
"")
        self.mcEditMode.setObjectName("mcEditMode")
        self.mcSendCommand = QtWidgets.QPushButton(self.frameMultipleCommand)
        self.mcSendCommand.setGeometry(QtCore.QRect(190, 140, 186, 31))
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setBold(True)
        font.setWeight(75)
        self.mcSendCommand.setFont(font)
        self.mcSendCommand.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.mcSendCommand.setStyleSheet("QPushButton {\n"
"    background-color: rgb(65, 65, 65);\n"
"    border: 2px solid black;\n"
"    border-radius: 8px;\n"
"    color: white;\n"
"    font-weight: bold;\n"
"    padding: 6px 12px;\n"
"    text-align: center;\n"
"}\n"
"\n"
"/* Hover effect */\n"
"QPushButton:hover {\n"
"    background-color: rgb(90, 90, 90);   /* lebih terang */\n"
"    border: 2px solid rgb(120, 120, 120);\n"
"    color: #e0e0e0;                      /* teks sedikit lebih terang */\n"
"}\n"
"\n"
"/* Saat ditekan */\n"
"QPushButton:pressed {\n"
"    background-color: rgb(40, 40, 40);   /* lebih gelap */\n"
"    border: 2px solid rgb(100, 100, 100);\n"
"    color: #cccccc;\n"
"}\n"
"")
        self.mcSendCommand.setObjectName("mcSendCommand")
        self.label_30 = QtWidgets.QLabel(self.frameMultipleCommand)
        self.label_30.setGeometry(QtCore.QRect(15, 116, 131, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_30.setFont(font)
        self.label_30.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_30.setObjectName("label_30")
        self.mcHeightFiltering = QtWidgets.QDoubleSpinBox(self.frameMultipleCommand)
        self.mcHeightFiltering.setGeometry(QtCore.QRect(15, 140, 131, 31))
        self.mcHeightFiltering.setStyleSheet("QDoubleSpinBox {\n"
"    background-color: #2b2b2b;       /* warna dasar gelap */\n"
"    border: 1px solid #4CAF50;       /* border hijau */\n"
"    border-radius: 6px;              /* sudut membulat */\n"
"    padding: 4px 28px 4px 8px;       /* spasi dalam, kanan untuk tombol */\n"
"    color: #f0f0f0;                  /* warna teks */\n"
"    font-size: 13px;\n"
"    selection-background-color: #4CAF50; /* highlight saat memilih angka */\n"
"}\n"
"\n"
"/* Hover */\n"
"QDoubleSpinBox:hover {\n"
"    border: 1px solid #66FF99;\n"
"}\n"
"\n"
"/* Fokus (saat aktif) */\n"
"QDoubleSpinBox:focus {\n"
"    border: 1px solid #00FFCC;\n"
"    background-color: #333333;\n"
"}\n"
"\n"
"/* Tombol panah */\n"
"QDoubleSpinBox::up-button, \n"
"QDoubleSpinBox::down-button {\n"
"    background-color: #3c3c3c;\n"
"    border: none;\n"
"    width: 20px;\n"
"    border-radius: 4px;\n"
"}\n"
"\n"
"/* Hover tombol panah */\n"
"QDoubleSpinBox::up-button:hover, \n"
"QDoubleSpinBox::down-button:hover {\n"
"    background-color: #4CAF50;\n"
"}\n"
"\n"
"/* biarkan arrow default bawaan Qt */\n"
"QDoubleSpinBox::up-arrow, \n"
"QDoubleSpinBox::down-arrow {\n"
"    width: 10px;\n"
"    height: 10px;\n"
"}\n"
"")
        self.mcHeightFiltering.setObjectName("mcHeightFiltering")
        self.mcDialOrientation = QtWidgets.QDial(self.frameMultipleCommand)
        self.mcDialOrientation.setGeometry(QtCore.QRect(395, 25, 151, 161))
        self.mcDialOrientation.setStyleSheet("")
        self.mcDialOrientation.setWrapping(True)
        self.mcDialOrientation.setNotchTarget(3.7)
        self.mcDialOrientation.setObjectName("mcDialOrientation")
        self.mcOrientation = QtWidgets.QLabel(self.frameMultipleCommand)
        self.mcOrientation.setGeometry(QtCore.QRect(485, 10, 61, 21))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.mcOrientation.setFont(font)
        self.mcOrientation.setStyleSheet("color: rgb(209, 207, 207);\n"
"border: 0px gray; ")
        self.mcOrientation.setObjectName("mcOrientation")
        self.label_36 = QtWidgets.QLabel(self.frameMultipleCommand)
        self.label_36.setGeometry(QtCore.QRect(410, 10, 71, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_36.setFont(font)
        self.label_36.setStyleSheet("color: rgb(209, 207, 207);\n"
"font-weight: bold;\n"
"border: 0px gray; ")
        self.label_36.setObjectName("label_36")
        self.mcDialOnline = QtWidgets.QLabel(self.frameMultipleCommand)
        self.mcDialOnline.setGeometry(QtCore.QRect(510, 40, 10, 10))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.mcDialOnline.setFont(font)
        self.mcDialOnline.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.mcDialOnline.setStyleSheet("QLabel {\n"
"    background-color: rgb(0, 255, 0);\n"
"    border: none;\n"
"    min-width: 10px;\n"
"    min-height: 10px;\n"
"    max-width: 10px;   /* biar fix, tidak melebar */\n"
"    max-height: 10px;  /* biar fix */\n"
"    border-radius: 5px;  /* setengah dari 25px ≈ 12px */\n"
"}\n"
"")
        self.mcDialOnline.setText("")
        self.mcDialOnline.setObjectName("mcDialOnline")
        self.mcHover = QtWidgets.QPushButton(self.frameMultipleCommand)
        self.mcHover.setGeometry(QtCore.QRect(190, 70, 186, 31))
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setBold(True)
        font.setWeight(75)
        self.mcHover.setFont(font)
        self.mcHover.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.mcHover.setStyleSheet("QPushButton {\n"
"    background-color: rgb(65, 65, 65);\n"
"    border: 2px solid black;\n"
"    border-radius: 8px;\n"
"    color: white;\n"
"    font-weight: bold;\n"
"    padding: 6px 12px;\n"
"    text-align: center;\n"
"}\n"
"\n"
"/* Hover effect */\n"
"QPushButton:hover {\n"
"    background-color: rgb(90, 90, 90);   /* lebih terang */\n"
"    border: 2px solid rgb(120, 120, 120);\n"
"    color: #e0e0e0;                      /* teks sedikit lebih terang */\n"
"}\n"
"\n"
"/* Saat ditekan */\n"
"QPushButton:pressed {\n"
"    background-color: rgb(40, 40, 40);   /* lebih gelap */\n"
"    border: 2px solid rgb(100, 100, 100);\n"
"    color: #cccccc;\n"
"}\n"
"")
        self.mcHover.setObjectName("mcHover")
        self.mcHome = QtWidgets.QPushButton(self.frameMultipleCommand)
        self.mcHome.setGeometry(QtCore.QRect(190, 35, 186, 31))
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setBold(True)
        font.setWeight(75)
        self.mcHome.setFont(font)
        self.mcHome.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.mcHome.setStyleSheet("QPushButton {\n"
"    background-color: rgb(65, 65, 65);\n"
"    border: 2px solid black;\n"
"    border-radius: 8px;\n"
"    color: white;\n"
"    font-weight: bold;\n"
"    padding: 6px 12px;\n"
"    text-align: center;\n"
"}\n"
"\n"
"/* Hover effect */\n"
"QPushButton:hover {\n"
"    background-color: rgb(90, 90, 90);   /* lebih terang */\n"
"    border: 2px solid rgb(120, 120, 120);\n"
"    color: #e0e0e0;                      /* teks sedikit lebih terang */\n"
"}\n"
"\n"
"/* Saat ditekan */\n"
"QPushButton:pressed {\n"
"    background-color: rgb(40, 40, 40);   /* lebih gelap */\n"
"    border: 2px solid rgb(100, 100, 100);\n"
"    color: #cccccc;\n"
"}\n"
"")
        self.mcHome.setObjectName("mcHome")
        self.CommandControl_2 = QtWidgets.QLabel(self.frame_6)
        self.CommandControl_2.setGeometry(QtCore.QRect(20, 10, 101, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.CommandControl_2.setFont(font)
        self.CommandControl_2.setStyleSheet("color: rgb(148, 148, 148);\n"
"border: 0px gray; ")
        self.CommandControl_2.setAlignment(QtCore.Qt.AlignCenter)
        self.CommandControl_2.setObjectName("CommandControl_2")
        self.CommandControl_3 = QtWidgets.QLabel(self.frame_6)
        self.CommandControl_3.setGeometry(QtCore.QRect(20, 110, 111, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.CommandControl_3.setFont(font)
        self.CommandControl_3.setStyleSheet("color: rgb(148, 148, 148);\n"
"border: 0px gray; ")
        self.CommandControl_3.setAlignment(QtCore.Qt.AlignCenter)
        self.CommandControl_3.setObjectName("CommandControl_3")
        self.frame_6.raise_()
        self.label_16.raise_()
        self.btQrcodeOnline_3.raise_()
        self.frame_5.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_45.setText(_translate("MainWindow", "AUTONOMOUS DISPLAYING DATA"))
        item = self.mcDisplayData.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Position"))
        item = self.mcDisplayData.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Orientation"))
        item = self.mcDisplayData.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Edit"))
        item = self.mcDisplayData.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Yaw Enable"))
        item = self.mcDisplayData.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "Landing"))
        item = self.mcDisplayData.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "Action"))
        self.CommandConnect.setText(_translate("MainWindow", "CONNECT"))
        self.label_11.setText(_translate("MainWindow", "TCP Connection :"))
        self.label_12.setText(_translate("MainWindow", "Websocket      :"))
        self.label_13.setText(_translate("MainWindow", "Control        :"))
        self.label_14.setText(_translate("MainWindow", "DRONE COMMAND CENTER"))
        self.CommandStatus.setText(_translate("MainWindow", "Disconnected"))
        self.CommandMode.setText(_translate("MainWindow", "Initial"))
        self.CommandControl.setText(_translate("MainWindow", "None"))
        self.label_25.setText(_translate("MainWindow", "GLOBAL POSITION"))
        self.label_26.setText(_translate("MainWindow", "X ="))
        self.label_27.setText(_translate("MainWindow", "Y ="))
        self.DronePositionX.setText(_translate("MainWindow", "[0.00] m"))
        self.DronePositionY.setText(_translate("MainWindow", "[0.00] m"))
        self.label_19.setText(_translate("MainWindow", "MODE"))
        self.DroneMode.setText(_translate("MainWindow", "Auto"))
        self.label_18.setText(_translate("MainWindow", "Battery"))
        self.DroneSpeedX.setText(_translate("MainWindow", "[0.00] m"))
        self.label_31.setText(_translate("MainWindow", "SPEED"))
        self.label_32.setText(_translate("MainWindow", "Y ="))
        self.label_33.setText(_translate("MainWindow", "X ="))
        self.DroneSpeedY.setText(_translate("MainWindow", "[0.00] m"))
        self.label_35.setText(_translate("MainWindow", "Z ="))
        self.DroneSpeedZ.setText(_translate("MainWindow", "[0.00] m"))
        self.label_21.setText(_translate("MainWindow", "HEIGHT"))
        self.DroneHeight.setText(_translate("MainWindow", "0.47 meter"))
        self.label_23.setText(_translate("MainWindow", "FLIGHT TIME"))
        self.DroneFlightTime.setText(_translate("MainWindow", "01:44"))
        self.DroneSwitch.setText(_translate("MainWindow", "Switch"))
        self.label_37.setText(_translate("MainWindow", "ALTITUDE"))
        self.label_16.setText(_translate("MainWindow", "DRONE CONTROLLER"))
        self.label_46.setText(_translate("MainWindow", "DEBUGGING CONSOLE"))
        self.tbDebugging.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Consolas\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.scHover.setText(_translate("MainWindow", "Hover"))
        self.scSendGoto.setText(_translate("MainWindow", "Send Goto"))
        self.scEdit.setText(_translate("MainWindow", "Edit"))
        self.scLanding.setText(_translate("MainWindow", "Landing"))
        self.scYawEnable.setText(_translate("MainWindow", "Yaw Enable"))
        self.label_15.setText(_translate("MainWindow", "Position :"))
        self.scPositionX.setText(_translate("MainWindow", "0.00"))
        self.label_17.setText(_translate("MainWindow", "X :"))
        self.scPositionY.setText(_translate("MainWindow", "0.00"))
        self.label_20.setText(_translate("MainWindow", "Y :"))
        self.label_22.setText(_translate("MainWindow", "Orientation :"))
        self.scOrientation.setText(_translate("MainWindow", "0.0000 rad"))
        self.scClearMarker.setText(_translate("MainWindow", "Clear Marker"))
        self.mcViewControls.setText(_translate("MainWindow", "Lock to Top-Down View"))
        self.label_28.setText(_translate("MainWindow", "View Controls"))
        self.mcSaveMaps.setText(_translate("MainWindow", "Save Maps"))
        self.label_29.setText(_translate("MainWindow", "Edit Mode"))
        self.mcEditMode.setText(_translate("MainWindow", "Edit Mode"))
        self.mcSendCommand.setText(_translate("MainWindow", "Start Autonomous"))
        self.label_30.setText(_translate("MainWindow", "Height Filtering (m) :"))
        self.mcOrientation.setText(_translate("MainWindow", "0.0000 rad"))
        self.label_36.setText(_translate("MainWindow", "Orientation :"))
        self.mcHover.setText(_translate("MainWindow", "Hover"))
        self.mcHome.setText(_translate("MainWindow", "Home"))
        self.CommandControl_2.setText(_translate("MainWindow", "Single Command"))
        self.CommandControl_3.setText(_translate("MainWindow", "Multiple Command"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
