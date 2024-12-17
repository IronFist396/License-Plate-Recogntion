from ultralytics import YOLO
import cv2
from sort.sort import *
import numpy as np
from utils import get_car, read_license_plate, write_csv

results = {}

mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')  #Detects cars
license_plate_detector = YOLO('./detect/train3/weights/best.pt')  #Detects license plates

# Load video
capture = cv2.VideoCapture('./cars-video.mp4')

# Of all the objects detected by yolo, we are interested in vehicles, their class_id can be found on their websites
# vehicles is a list of class_ids of different vehicles (cars, trucks, bikes, etc)
vehicles = [2, 3, 5, 6, 7]

# Read frames
ret = True
frame_num = -1
while ret:
    ret, frame = capture.read()
    frame_num += 1
    if ret:
        results[frame_num] = {}
        # Detect cars
        detections = coco_model(frame)[0]

        # save bounding boxes of all the vehicles
        detections_ =  []

        for detection in detections.boxes.data.tolist():
            # Detection is x1, y1, x2, y2, confidence, class
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles (using sort)
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detecting license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, card_id = get_car(license_plate, track_ids)

            if card_id != -1:

                # Crop license plate
                license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]

                # Process license plate for easy ocr to recognise
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

                # All pixels lower than 64 will be 255 and all that are higher than 64 will be 0
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                # Write results
                if license_plate_text is not None:
                    results[frame_num][card_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                   'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                     'text': license_plate_text,
                                                                     'bbox_score': score,
                                                                     'text_score': license_plate_text_score
                                                                     }}

    write_csv(results, './new.csv')



