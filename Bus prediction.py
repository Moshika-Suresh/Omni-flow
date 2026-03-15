import cv2
from ultralytics import YOLO
import time
import sys
import logging
import os

# disable YOLO logs
logging.getLogger("ultralytics").setLevel(logging.ERROR)
os.environ["YOLO_VERBOSE"] = "False"

TOTAL_SEATS = 34

stops = [
    "Thiruverkadu",
    "Nerkundram",
    "Rohini",
    "Chathiram",
    "Koyambedu",
    "Arumbakkam",
    "Vadapalani",
    "Ashok Nagar",
    "Saidapet",
    "Teynampet",
    "T Nagar"
]

avg_passengers = {
    "Chathiram": 5,
    "Koyambedu": 8,
    "Arumbakkam": 6
}

bus_number = sys.argv[1]
user_stop = sys.argv[2].title()

if bus_number != "72":
    print("Currently system supports only Bus 72")
    sys.exit()

if user_stop not in stops:
    print("Invalid stop")
    sys.exit()

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

print("Camera started")
print("Detecting passengers for 10 seconds...")

start_time = time.time()
max_people = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # silence YOLO output
    results = model(frame, verbose=False)

    people_count = 0

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])

            if cls == 0:
                people_count += 1

                x1,y1,x2,y2 = map(int,box.xyxy[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    if people_count > max_people:
        max_people = people_count

    cv2.putText(frame,
                f"Passengers Detected: {people_count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2)

    cv2.imshow("Bus Passenger Detection",frame)

    if time.time() - start_time > 10:
        break

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

people_count = max_people

current_stop = "Rohini"

current_index = stops.index(current_stop)
next_stop = stops[current_index + 1]

predicted_enter = avg_passengers.get(next_stop,0)

final_passengers = people_count + predicted_enter
available_seats = TOTAL_SEATS - final_passengers

print("\n----------- BUS PREDICTION RESULT -----------")

print("Bus Number:", bus_number)
print("Current Stop:", current_stop)
print("Next Stop:", next_stop)

print("Current passengers in bus:", people_count)

print("Predicted passengers entering at", next_stop, ":", predicted_enter)

print("Passengers when bus reaches", user_stop, ":", final_passengers)

if available_seats > 0:
    print("Available seats:", available_seats)
else:
    print("Bus will likely be FULL")
