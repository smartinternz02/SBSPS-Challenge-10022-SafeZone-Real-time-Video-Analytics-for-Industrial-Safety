# SBSPS-Challenge-10022-SafeZone-Real-time-Video-Analytics-for-Industrial-Safety
SafeZone: Real-time Video Analytics for Industrial Safety
INTRODUCTION
OVERVIEW:
Industrial Safety is essential for promoting worker safety ,preventing accidents, ensuring compliance, and enhancing overall operational efficiency in industrial environments .By leveraging advanced video analytics technologies, it offers aproactive and data-driven approach to industrial safety management. So here the problem is to develop a real-time video analytics tool, that enhances industrial safety by detecting and preventing potential hazards and unsafe situations in industrial environments. 
PURPOSE:
The business impacts of implementing real-time video analytics for occupational safety include a stronger safety culture, reduced workplace accidents and downtime, improved compliance, cost savings, operational efficiency, positive corporate image, competitive advantage, data-driven decision-making, talent engagement and collaborative opportunities with safety partners. 
LITERATURE SURVEY
Existing problem:
The existing problem Industrial workers they working time they not wearing the (PPE). So there accidents will be occur.
Proposed solution:
The solution using Python ,Artificial Learning ,Deep Learning , Object Detection , Yolo &Flask. To create a Web Application so, we can monitor live video feeds and provide real-time by using Yolov8 Object Detection .To Detect the Personal Protective Equipment (PPE) for Industry Worker. enhancing industrial safety and preventing accidents in industrial environments. This technology improves workplace safety, productivity and reputation, which benefits both the company and its employees.
FlowChart:

EXPERIMENTAL INVESTIGATIONS:
The implementation of real-time video analysis of occupational safety promises to significantly improve occupational safety in industrial environments. Using state-of-the-art technology, potential safety risks can be detected and quickly addressed, resulting in a safer and healthier workforce. This technology promotes a culture of safety, empowers employees and promotes a positive corporate image for companies that invest in the well-being of their employees. In addition, the positive social impact extends beyond the workplace to influence industry practices and benefit local communities. By prioritizing occupational safety, companies advance the goals of sustainability and create a safer, more engaging and sustainable work environment for everyone.
Result:
Web Page Image:-
 
Video Button:-
 

Live Web Button:
 




Video Uploading Output:
 
Live Web Output:
 



Our Model Train Confusion Matrix:
 

Our Model Train Loss Graph
 



Validate Our Model:
 

Our Model Detected Image:
 

ADVANTAGES :
Enhanced Industrial Safety: The application utilizes advanced object detection techniques to identify whether industrial workers are wearing proper Personal Protective Equipment (PPE) such as helmets, gloves, safety vests, etc. This significantly reduces the risk of accidents and injuries in industrial settings, promoting a safer work environment.
Real-time Monitoring: The real-time nature of the solution allows supervisors and safety personnel to monitor workers' adherence to safety regulations as it happens. This quick response capability enables timely interventions if any worker is not wearing the required PPE.
Accident Prevention: By ensuring that workers are equipped with the necessary protective gear, the solution helps prevent accidents that could result from inadequate safety measures. This leads to a reduction in workplace incidents and their associated costs.Customizable and Scalable: The web application built using Flask can be easily customized to meet specific industry needs and scaled up to monitor multiple locations and camera feeds simultaneously.
DISADVANTAGES :
Complex Implementation: Developing and fine-tuning an accurate object detection system requires expertise in deep learning and machine learning. Implementing such a system can be complex and time-consuming, requiring specialized knowledge and resources.
Dependency on Cameras: The system relies on cameras to capture live video feeds. If cameras malfunction or experience technical issues, it could disrupt the entire monitoring process.
APPLICATIONS:
Industrial environments stand to benefit greatly from the deployment of real-time video analysis for occupational safety. A safer and healthier workforce can be achieved by immediately identifying possible safety issues using cutting-edge technologies. This technology supports a culture of safety, empowers staff, and enhances the reputation of businesses that care about the welfare of their workers. The beneficial social impact also influences business practises and helps local communities outside of the workplace. Companies can promote sustainability objectives and make the workplace safer, more enjoyable, and sustainable for all employees by putting a high priority on occupational safety.

CONCLUSION:
The objective is to leverage Deep Learning techniques to monitor live video feeds and provide real-time hazard detection, prompt alerts, and interventions, enhancing industrial safety and preventing accidents in industrial environments.

APPENDIX:
SOURCE CODE
Flaskapp.py
from flask import Flask, render_template, Response,jsonify,request,session

#FlaskForm--> it is required to receive input from the user
# Whether uploading a video file  to our object detection model

from flask_wtf import FlaskForm


from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
import os


# Required to run the YOLOv8 model
import cv2

# YOLO_Video is the python file which contains the code for our object detection model
#Video Detection is the Function which performs Object Detection on Input Video
from YOLO_Video import video_detection
app = Flask(_name_)

app.config['SECRET_KEY'] = 'Thanush'
app.config['UPLOAD_FOLDER'] = 'static/files'


#Use FlaskForm to get input video file  from user
class UploadFileForm(FlaskForm):
    #We store the uploaded video file path in the FileField in the variable file
    #We have added validators to make sure the user inputs the video in the valid format  and user does upload the
    #video when prompted to do so
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Run")


def generate_frames(path_x = ''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

@app.route('/', methods=['GET','POST'])
@app.route('/home', methods=['GET','POST'])
def home():
    session.clear()
    return render_template('indexpage.html')
# Rendering the Webcam Rage
#Now lets make a Webcam page for the application
#Use 'app.route()' method, to render the Webcam page at "/webcam"
@app.route("/webcam", methods=['GET','POST'])

def webcam():
    session.clear()
    return render_template('liveweb.html')
@app.route('/FrontPage', methods=['GET','POST'])
def front():
    # Upload File Form: Create an instance for the Upload File Form
    form = UploadFileForm()
    if form.validate_on_submit():
        # Our uploaded video file path is saved here
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(_file_)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(_file_)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('video_detect.html', form=form)
@app.route('/video')
def video():
    #return Response(generate_frames(path_x='static/files/bikes.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames(path_x = session.get('video_path', None)),mimetype='multipart/x-mixed-replace; boundary=frame')

# To display the Output Video on Webcam page
@app.route('/webapp')
def webapp():
    #return Response(generate_frames(path_x = session.get('video_path', None),conf_=round(float(session.get('conf_', None))/100,2)),mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if _name_ == "_main_":
    app.run(debug=True)

YOLO_Video.py
from ultralytics import YOLO
import cv2
import math

def video_detection(path_x):
    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))

    model=YOLO("YOLO_Weights/best.pt")
    classNames = ['Protective Helmet', 'Shield', 'Jacket', 'Dust Mask', 'Eye Wear', 'Glove', 'Protective Boots']
    while True:
        success, img = cap.read()
        results=model(img,stream=True)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                print(x1,y1,x2,y2)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                if class_name == 'Dust Mask':
                    color=(0, 204, 255)
                elif class_name == "Glove":
                    color = (222, 82, 175)
                elif class_name == "Protective Helmet":
                    color = (0, 149, 255)
                else:
                    color = (85,45,255)
                if conf>0.5:
                    cv2.rectangle(img, (x1,y1), (x2,y2), color,3)
                    cv2.rectangle(img, (x1,y1), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

        yield img
        #out.write(img)
        #cv2.imshow("image", img)
        #if cv2.waitKey(1) & 0xFF==ord('1'):
            #break
    #out.release()
cv2.destroyAllWindows()
liveweb.html (LIVE WEB PAGE )
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport"
          content="width=device-width, user-scalable=no, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Video</title>
    <style>
        body{
            color:white;
            margin:0px;
            padding:0px;
        }
        header.feature-box.top{
        background-color:black;
        height:100px;
        margin:0px;
        padding:20px;
        text-align:center;
        }
        header.feature-box.second{
        background-color:blue;
        height:50px;
        text-align:center;
        margin-top:-25px;
        }
        .features{
          background-color:black;
          width: 900px;
          height: 700px;
          border-radius: 35px;
          object-fit: contain;
          margin:20px;
        }
        section.col-sm{
          background-color:black;
          width: 1000px;
          height: 650px;
          border-radius: 35px;
          object-fit: contain;
          margin:40px;
        }img{
          width: 900px;
          height: 600px;
          border-radius: 35px;
          object-fit: contain;
          margin:40px;
        }.new{
        color:black;
        margin:0px;
        padding:10px;
        background-color:green;
        margin:0px;
        margin-top:-10px;
        }
    </style>
</head>
<body>
<header class="feature-box top">
    <h1><strong>Object Detection using YOLOv8</strong></h1>
</header>
<header class="feature-box second">
     <h1><strong>Output Video</strong></h1>
</header> <section class="col-sm " >
          <img src="{{ url_for('webapp') }}"   alt="Upload video">
</section>
</body></html>


