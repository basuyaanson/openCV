import imp
from symbol import varargslist
import cv2

"""
人臉辨識實時檢測
"""

#加載訓練數據文件
recogizer = cv2.face.LBPHFaceRecognizer_create()
recogizer1 = cv2.face.LBPHFaceRecognizer_create()
recogizer2 = cv2.face.LBPHFaceRecognizer_create()

#加載訓練數據
recogizer.read(r"C:\Users\user\Desktop\python\trainer\trainer.yml")
recogizer1.read(r"C:\Users\user\Desktop\python\trainer\trainer1.yml")
recogizer2.read(r"C:\Users\user\Desktop\python\trainer\trainer2.yml")

facecascde = cv2.CascadeClassifier('face_detect.xml')#導入分類器

#鏡頭
cap = cv2.VideoCapture(0)

while(True):
    #獲取鏡頭畫面
    ret, frame = cap.read()
    faces = facecascde.detectMultiScale(frame,2,3) #辨識人臉
    img = frame
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #轉為灰圖
    for(x,y,w,h) in faces:    
         ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
         ids, confidence1 = recogizer1.predict(gray[y:y + h, x:x + w])
         ids, confidence2 = recogizer2.predict(gray[y:y + h, x:x + w])
         if confidence < 50 :
             #當評分小於50,代表檢測的人的可信,將名子打印到方框上
             img = cv2.rectangle(img,(x,y),(x+w, y+h),(0,225,0),2) #檢測方框
             cv2.putText(img,str("anson"), (x+10, y-10), cv2.FONT_HERSHEY_DUPLEX,0.75,(0,225,0),1)
         elif confidence1 < 60 :
             #當評分小於60,代表檢測的人的可信,將名子打印到方框上
             img = cv2.rectangle(img,(x,y),(x+w, y+h),(0,225,0),2) #檢測方框
             cv2.putText(img,str("Mads"), (x+10, y-10), cv2.FONT_HERSHEY_DUPLEX,0.75,(0,225,0),1) 
         elif confidence2 < 80 :
             #當評分小於80,代表檢測的人的可信,將名子打印到方框上
             img = cv2.rectangle(img,(x,y),(x+w, y+h),(0,225,0),2) #檢測方框
             cv2.putText(img,str("tsuneta daiki"), (x+10, y-10), cv2.FONT_HERSHEY_DUPLEX,0.75,(0,225,0),1)        
         else:  
             img = cv2.rectangle(img,(x,y),(x+w, y+h),(0,0,255),2) #檢測方框
             cv2.putText(img,str("unknown"), (x+10, y-10), cv2.FONT_HERSHEY_DUPLEX,0.75,(0,0,255),1) 

         
        
    cv2.imshow('frame2',img) #實時顯示畫面
    if cv2.waitKey(5) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()

 
 
