def detect(filename,cascade_file="lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)        
    
    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
               # detector options
        scaleFactor = 1.1, #scaleFactor
        minNeighbors = 5, #minNeighbors
        minSize = (48,48)
                      )
    
    for i,(x,y,w,h) in enumerate(faces):
        face = image[y: y+h, x:x+w,:]  # 裁切區域的 x 與 y 座標（左上角） x = 100 y = 100
                                     # 裁切區域的長度與寬度 w = 250 h = 150
                                     # 裁切圖片 crop_img = img[y:y+h, x:x+w]
        face = cv2.resize(face,(96,96))
        save_filename = '%s.jpg' % (os.path.basename(filename).split('.')[0])   
                                    #os.path.basename('c:\\test.csv')  'test.csv'
        cv2.imwrite("faces/"+save_filename,face)
        
if __name__ == '__main__':
    if os.path.exists('faces') is False:
        os.makedirs('faces')
    file_list = glob('imgs/*.jpg')
    for filename in file_list:
        detect(filename)
