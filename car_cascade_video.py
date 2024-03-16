import cv2

cap=cv2.VideoCapture("D:\opencv_udemy/11_car_cascade\car.mp4")
car_cascade=cv2.CascadeClassifier("D:\opencv_udemy/11_car_cascade\car_cascade.xml")

while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(640,480))

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cars=car_cascade.detectMultiScale(gray,1.1,1)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

    cv2.imshow("Video",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()