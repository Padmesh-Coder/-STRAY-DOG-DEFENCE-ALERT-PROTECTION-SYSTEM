import cv2
from ultralytics import YOLO
import dlib
import threading
from playsound import playsound
from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()

# Twilio configuration
account_sid = 'Twilio_SID_id1'
auth_token = os.environ['AUTH_KEY']
client = Client(account_sid, auth_token)

# Twilio configuration 2
Account_sid = 'Twilio_SID_id1'
Auth_token = os.environ['AUTH_KEY_2']
msg = Client(Account_sid, Auth_token)

def send_sms_alert():
    message = client.messages.create(
        body="Alert: Aggressive dog detected!",
        from_='+18562813205',
        to='+919600063232'
    )
    print(f"SMS sent: {message.sid}")

def send_whatsapp_alert():
    message = msg.messages.create(
        body="Alert: Aggressive dog detected!",
        from_='whatsapp:+14155238886',
        to='whatsapp:+919600063232'
    )
    print(f"WhatsApp message sent: {message.sid}")

# Load the YOLOv8 model
model_yolo = YOLO(r"/Users/rafaelzieganpalg/Projects/Hema Mam/New_YOLO/best.pt")

# Load the dog head detector
pathDet = '/Users/rafaelzieganpalg/Projects/Hema Mam/New_YOLO/dogHeadDetector.dat'
detector = dlib.cnn_face_detection_model_v1(pathDet)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Reduce the resolution of the video capture for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

def play_sound():
    playsound('/Users/rafaelzieganpalg/Projects/Hema Mam/New_YOLO/Water_Bro.mp3')

last_detection = None
cooldown_time = 2  # Cooldown time in seconds to prevent rapid repeat of sound
frame_count = 0
skip_frames = 5  # YOLO will run every 5th frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    small_frame = cv2.resize(frame, (320, 240))

    detections = detector(small_frame, upsample_num_times=0)

    # Process detections from dog head detector
    for detection in detections:
        confidence = detection.confidence  # Confidence score of the dog head detector

        # Proceed only if confidence is greater than 0.5
        if confidence > 0.7:
            # Run YOLO every 5th frame
            if frame_count % skip_frames == 0:
                results = model_yolo(frame, device="mps" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")

                highest_confidence = 0
                best_box = None
                best_class_id = None

                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()

                    for i in range(len(boxes)):
                        yolo_confidence = confidences[i]
                        if yolo_confidence > highest_confidence:
                            highest_confidence = yolo_confidence
                            best_box = boxes[i]
                            best_class_id = int(class_ids[i])

                if highest_confidence > 0.5 and best_box is not None and best_class_id is not None:
                    x1, y1, x2, y2 = [int(coord) for coord in best_box]
                    label = model_yolo.names[best_class_id]
                    confidence_text = f'{label} {highest_confidence:.2f}'

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, confidence_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    if label.lower() == 'aggressive':
                        if last_detection is None or (cv2.getTickCount() - last_detection) / cv2.getTickFrequency() > cooldown_time:
                            threading.Thread(target=play_sound, daemon=True).start()
                            threading.Thread(target=send_sms_alert, daemon=True).start()
                            threading.Thread(target=send_whatsapp_alert, daemon=True).start()
                            last_detection = cv2.getTickCount()

    cv2.imshow('YOLOv8 Live Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()