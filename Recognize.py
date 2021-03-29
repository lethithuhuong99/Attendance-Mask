import datetime
import os
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('MyTrainingModel.h5')
threshold=0.90

def preprocessing(img):
    img=img.astype("uint8")
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img = img/255
    return img

def get_className(classNo):
	if classNo==0:
		return "Mask"
	elif classNo==1:
		return "No Mask"

#-------------------------
def recognize_attendence():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel"+os.sep+"Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails"+os.sep+"StudentDetails.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time','Mask']
    attendance = pd.DataFrame(columns=col_names)

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, im = cam.read()
        im = cv2.flip(im,1)
        k = cv2.waitKey(125)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5,minSize = (int(minW), int(minH)),flags = cv2.CASCADE_SCALE_IMAGE)
        for(x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            crop_img = im[y:y + h, x:x + h]
            img = cv2.resize(crop_img, (32, 32))
            img = preprocessing(img)
            img = img.reshape(1, 32, 32, 1)
            prediction = model.predict(img)
            classIndex = model.predict_classes(img)
            probabilityValue = np.amax(prediction)

            if probabilityValue > threshold:
                if classIndex == 0:
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(im, (x, y - 40), (x + w, y), (0, 255, 0), -2)
                    cv2.putText(im, str(get_className(classIndex)), (x, y - 10), font, 0.75,
                                (255, 255, 255), 1,
                                cv2.LINE_AA)
                    print("Mask")
                elif classIndex == 1:
                    cv2.rectangle(im, (x, y), (x + w, y + h), (50, 50, 255), 2)
                    cv2.rectangle(im, (x, y - 40), (x + w, y), (50, 50, 255), -2)
                    cv2.putText(im, str(get_className(classIndex)), (x, y - 10), font, 0.75,
                                (255, 255, 255), 1,
                                cv2.LINE_AA)
                    print("No Mask")
            if (100-conf) > 50:
                # lấy tên và id
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                aa = df.loc[df['Id'] == Id]['Name'].values
                confstr = "  {0}%".format(round(100 - conf))
                tt = str(Id)+"-"+aa

                #xử lý điểm danh, lưu vào file
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = str(aa)[2:-2]
                mask = str(get_className(classIndex))
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp,mask]

                # hiển thị điểm danh thành công
                tt = tt + " [Pass]"
                cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (0, 255, 0), 2)

                # hiển thị tên người điểm danh
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font,1, (0, 255, 0),1 )
                print("Da diem danh")

            else:
                print("CHua diem danh")
                # không lấy tên và id
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                Id = '  Unknown  '
                tt = str(Id)
                confstr = "  {0}%".format(round(100 - conf))

                # điểm danh khong thành công
                cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (0, 0, 255), 2)

                # hiển thị unknown
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)

            tt = str(tt)[2:-2]

        attendance = attendance.sort_values(['Id', 'Mask'], ascending=[True,True])
        cv2.imshow('Attendance', im)

        if (cv2.waitKey(1) == ord('q')) :
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "Attendance"+os.sep+"Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
    attendance.to_csv(fileName, index=False)
    print("Attendance Successful")
    cam.release()
    cv2.destroyAllWindows()