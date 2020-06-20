"-*- coding: utf-8 -*-"
# Main.py

import os
import time
from datetime import datetime
from tkinter import *
import numpy as np
import DetectChars
import DetectPlates
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication,QWidget,QLabel,QVBoxLayout, QFileDialog, QMessageBox
from PIL import Image, ImageOps

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

def main():
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()

    if blnKNNTrainingSuccessful == False:
        print("\nerror: KNN traning was not successful\n")
        return

    app = QApplication(sys.argv)
    #fileName = QFileDialog.getOpenFileName(None,"Dosya Aç","C:/Users/Sevki/Desktop/Car Foto/VideoGoruntuleri", "Image File (*.jpeg *.png *.jpg)")
    fileName = QFileDialog.getOpenFileName(None,"Dosya Aç","C:/Users/Sevki/PycharmProjects/Plate1/Plate 1.3", "Image File (*.jpeg *.png *.jpg)")

    txt = str(fileName[0])

    imgOriginalScene = cv2.imread(txt)


    if imgOriginalScene is None:
        print("\nerror: Dosya fotoğrafı okumadı. \n\n")
        os.system("pause")
        return

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

    #font = cv2.FONT_HERSHEY_TRIPLEX
    #cv2.putText(imgOriginalScene, 'Plaka bulundu, yukleniyor...', (10,30), 0, 1, (0,0,255), 2, cv2.LINE_AA)
    #cv2.imshow("Plaka Tanimlama Sistemi", imgOriginalScene)
    #cv2.waitKey(2000)

    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Information)
    msgBox.setWindowTitle("Plaka Tanıma Sistemi")
    msgBox.setText("Plaka bulundu, yükleniyor...")
    msgBox.setStandardButtons(QMessageBox.Ok)
    msgBox.exec_()

    time.sleep(1)

    if len(listOfPossiblePlates) == 0:
        print("\n Herhangi bir plaka tespit edilemedi. \n")
    else:
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)


        global licPlate
        licPlate = listOfPossiblePlates[0]

        if len(licPlate.strChars) == 0:
            print("\n Karakter bulunamadı. \n\n")
            return

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)

        print("\n Görüntüde okunan plaka = " + licPlate.strChars + "\n")
        print("----------------------------------------")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)
        arayuz(imgOriginalScene, licPlate)

    cv2.imwrite("imgOriginalScene.png", imgOriginalScene)
    cv2.waitKey(0)
    return

#############################INTERFACE############################
def arayuz(imgOriginalScene,licPlate):
    app = QApplication(sys.argv)
    pencere = QWidget()

    label = QLabel(" Araç Görüntüsü ")
    label.setFont(QFont("Arial", 14, QFont.Bold))

    imgOriginalScene = cv2.resize(imgOriginalScene, (800,500), interpolation=cv2.INTER_AREA)

    resim = QLabel()
    a = imgOriginalScene
    height, width, channel = a.shape
    bytesPerLine = 3 * width
    qImg = QImage(imgOriginalScene, width, height, bytesPerLine, QImage.Format_RGB888)
    resim.setPixmap(QPixmap(qImg))

    resim2 = QLabel()
    a = licPlate.imgPlate
    height, width, channel = a.shape
    bytesPerLine = 3 * width
    qImg = QImage(licPlate.imgPlate, width, height, bytesPerLine, QImage.Format_RGB888)
    resim2.setPixmap(QPixmap(qImg))

    resim3 = QLabel()
    b = licPlate.imgThresh
    height, width, = b.shape
    bytesPerLine = 3 * width
    qpImg = QImage(licPlate.imgThresh, width, height, bytesPerLine, QImage.Format_RGB444)
    resim3.setPixmap(QPixmap(qpImg))

    plakaKoduYaz = QLabel("Plaka : " + licPlate.strChars)
    plakaKoduYaz.setFont(QFont("Arial", 14, QFont.Bold))

    t=0
    try:
        for i in (0,6,1):
            if (licPlate.strChars[i]) == 'I':
                t = t + 13.43253
    except:
        return

########################## 7 Character for Turkish License Plate Error Rate ##########################
    if len(licPlate.strChars)==7:
        try:
            if int(licPlate.strChars[0]) == int:
                return
        except:
            t = t + 14.28
        try:
            if int(licPlate.strChars[1]) == int:
                return
        except:
            t = t + 14.28

        try:
            if str(licPlate.strChars[2]) == str:
                return
        except:
            t = t + 14.28
        try:
            if int(licPlate.strChars[5]) == int:
                return
        except:
            t = t + 14.28
        try:
            if int(licPlate.strChars[6]) == int:
                return
        except:
            t = t + 14.28
        k = 100 - t- 14.28 + 10
        kstr = str(k)
########################## 8 Character for Turkish License Plate Error Rate ##########################
    if len(licPlate.strChars) == 8:
        try:
            if int(licPlate.strChars[0]) == int:
                return
        except:
            t = t + 12.5
        try:
            if int(licPlate.strChars[1]) == int:
                return
        except:
            t = t + 12.5

        try:
            if str(licPlate.strChars[2]) == str:
                return
        except:
            t = t + 12.5
        try:
            if str(licPlate.strChars[3]) == str:
                return
        except:
            t = t + 12.5
        try:
            if int(licPlate.strChars[5]) == int:
                return
        except:
            t = t + 12.5
        try:
            if int(licPlate.strChars[6]) == int:
                return
        except:
            t = t + 12.5
        try:
            if int(licPlate.strChars[7]) == int:
                return
        except:
            t = t + 12.5
        k = 100 - t - 12.5 + 10
        kstr = str(k)

    try:
        hataOranı = QLabel("Tanıma Oranı : % " + kstr  )
        hataOranı.setFont(QFont("Arial", 14, QFont.Bold))
        uyari = QLabel("*( TR kodlu plakalar için %85 ve üzeri tahminler güvenilirdir.)")
        uyari.setFont(QFont("Arial", 8 ))
    except:
        return

    now = QDate.currentDate()
    nowTime = QTime.currentTime()
    okunanGunveSaat = QLabel("Gün ve Saat : " + now.toString(Qt.ISODate) + "  " + nowTime.toString())

    okunanGunveSaat.setFont(QFont("Arial", 14, QFont.Bold))

    v_box = QVBoxLayout()
    v_box.addWidget(label)
    v_box.addStretch()
    v_box.addWidget(resim)
    v_box.addWidget(resim2)
    v_box.addWidget(resim3)
    v_box.addWidget(plakaKoduYaz)
    v_box.addWidget(hataOranı)
    v_box.addWidget(uyari)
    v_box.addWidget(okunanGunveSaat)

    pencere.setGeometry(50, 50, 1000, 1000)
    pencere.setLayout(v_box)
    pencere.setWindowTitle("Plaka Tanımlama Sistemi")
    pencere.showMaximized()
    pencere.show()
    sys.exit(app.exec_())

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # Kırmızı Çizgi
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             # Metnin yazılacağı alan merkezi
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          #Metnin yazılacağı sol alt ksıım
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # Jane font seçimi
    fltFontScale = float(plateHeight) / 30.0                    # plaka alanı yükseliğinde font boyu
    intFontThickness = int(round(fltFontScale * 1.5))           # yazı tipi

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)

            # unpack roatated rect into center point, width and height, and angle
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # Merkezi tam sayıya çevirme
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         # Metin alanının yatayı plaka ile aynıdır.

    if intPlateCenterY < (sceneHeight * 0.75):
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # Plakanın int olarak bölgenin altına yazılması
    else:                                                                                       # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # Karakteri plaka bölgesinin yukarısına üstüne yazılması

    textSizeWidth, textSizeHeight = textSize

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           # metin alanının sol alt kökenini hesaplar
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          # metin alanı merkezini, genişliğini ve yüksekliğini temel alarak


    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)

###################################################################################################
if __name__ == "__main__":
    main()
