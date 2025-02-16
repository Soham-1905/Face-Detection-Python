import cv2

def generate_dataset(img, id, img_id):
    cv2.imwrite("C:\\Users\\naiks\\OneDrive\\Desktop\\HaarCascade\\data//user."+ str(id) +"."+ str(img_id)+".jpg", img)
      
def draw_boundary(img, classifier, scaleFactor, minNeighbor, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbor)
    coords = []
    for (x, y, w, h ) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2 )
        id,_ = clf.predict(gray_img[y : y+h, x : x+w])
        if id==1:
            cv2.putText(img, "Soham", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX , 0.8, color, 1, cv2.LINE_AA)
        elif id == 2:
            cv2.putText(img, "Maithili", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX , 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords   

    
def recognize(img, clf, faceCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade , 1.1, 10, color["blue"], "Face", clf)
    return img

def detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade, img_id):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    
    coords = draw_boundary(img, faceCascade , 1.1, 10, color['blue'], "Face")
    
    if len(coords)==4:
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        user_id = 1
        generate_dataset(roi_img , user_id, img_id)
        #coords = draw_boundary(roi_img , eyesCascade, 1.1 , 14, color['red'], "Eyes")
        #coords = draw_boundary(roi_img , noseCascade , 1.1 , 5, color['green'], "Nose")
        #coords = draw_boundary(roi_img , mouthCascade , 1.1 , 20, color['white'], "Mouth")
        
    
    return img


faceCascade = cv2.CascadeClassifier("C:\\Users\\naiks\\OneDrive\\Desktop\\HaarCascade\\haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier("C:\\Users\\naiks\\OneDrive\\Desktop\\HaarCascade\\haarcascade_eye.xml")
noseCascade = cv2.CascadeClassifier("C:\\Users\\naiks\\OneDrive\\Desktop\\HaarCascade\\nose.xml")
mouthCascade = cv2.CascadeClassifier("C:\\Users\\naiks\\OneDrive\\Desktop\\HaarCascade\\haarcascade_mcs_mouth.xml")

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("C:\\Users\\naiks\\OneDrive\\Desktop\\HaarCascade\\classifier.yml")

video_capture = cv2.VideoCapture(0)
img_id = 0
while True:
    _, img = video_capture.read()
    #img = detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade,img_id)
    img = recognize(img, clf, faceCascade)
    cv2.imshow("Face Detection", img)
    img_id +=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()