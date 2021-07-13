"""
Main module to infer emotions in real time using a webcam.

This module uses a simple HOG face detection model, 
which crops the detected face and feed it to a FER model.
The output is computed in real time and visualized in the video.
"""

import sys
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from utils.data import *
from utils.model import *
from utils.train import *
from utils.infer import *
from utils.plot import *
from models.vggnet import *
from models.resnet_narrow_nocp import *
plt.style.use('ggplot')

# from mtcnn import MTCNN
# import cv2
# detector = MTCNN()

def infer_frame(model, frame):
    x = torch.tensor(np.asarray(frame))/255 
    x.unsqueeze_(0)
    x.unsqueeze_(0)
    _, y_p, y = infer(model, x, device=device)
    return y_p, y


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vgg = VGGNet().to(device)
load("./pretrained/VGG4.0_6_0.691.tar", vgg, device=device)
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cascPath = "./pretrained/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    # faces = detector.detect_faces(frame)

    # Draw a rectangle around the faces
    if len(faces) == 1:
        # box = faces[0]['box']
        # x, y, w, h = box[0], box[1], box[2], box[3]

        # frame_pil = Image.fromarray(frame[y:y+h, x:x+h, 0])
        # frame_pil.thumbnail((48, 48), Image.ANTIALIAS)
        # y_p, y_c = infer_frame(vgg, frame_pil)

        # cv2.rectangle(frame, (x, y), (x+h, y+h), (0, 255, 0), 2)
        # cv2.putText(frame, f"{class_names[y_c.item()]:<8} {y_p.item()*100:.0f}%", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 
        #             fontScale=0.5, color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)

        for (x, y, w, h) in faces:
            frame_pil = Image.fromarray(frame[y:y+h, x:x+w, 0])
            frame_pil.thumbnail((48, 48), Image.ANTIALIAS)
            y_p, y_c = infer_frame(vgg, frame_pil)

            cv2.rectangle(frame, (x, y), (x+h, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_names[y_c.item()]:<8} {y_p.item()*100:.0f}%", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.5, color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # cv2.putText(frame, f"{class_names[y_c.item()]} {y_p.item()*100:.0f}", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 
            #             fontScale=0.75, color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()