import sys
import argparse
from camera import Camera, add_camera_args, open_cam_rtsp
import time
from threading import Thread, Lock
import cv2
import numpy as np
import math
from sort import *
import pandas as pd
import os
import torch
from ultralytics import YOLO

# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

class ObjectDetection():
    def __init__(self, capture, result):
        self.cam = capture
        self.args = result
        self.capture = capture
        self.result = result
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.frame_skip = 5  # Process every 5th frame
        self.process_fps = 10  # Process frames at 10 FPS
        self.last_process_time = time.time()
        self.lock = Lock()  # Lock for shared variables
        self.df = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2', 'id', 'cx', 'cy'])

    def load_model(self):
        model = YOLO("mlruns/249666638938616916/150fb9c000244a52a30db0218a6dac67/artifacts/weights/best.pt")
        # model.fuse()
        return model

    def predict(self, img):
        results = self.model(img ,conf=0.7,iou=0.9,save=True)
        return results

    def plot_boxes(self, results, detections, counter):
        for r in results:
            counter += 1
            boxes = r.boxes
            print(boxes)
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil(box.conf[0] * 100) / 100
                cls = box.cls[0].cpu().numpy()
                currentArray = np.array([x1, y1, x2, y2, conf, cls])
                detections = np.vstack((detections, currentArray))
                # break
        return detections, counter
    
    def append_to_dataframe(self, df, detections, video_id, frame_number):
        for detection in detections:
            x1, y1, x2, y2, id = detection
            class_name = self.CLASS_NAMES_DICT[id]
            confidence = detection[4]
            bb_left = x1
            bb_top = y1
            bb_width = x2 - x1
            bb_height = y2 - y1
            df1 = pd.DataFrame({
                'video_id': [video_id],
                'frame': [frame_number],
                'bb_left': [bb_left],
                'bb_top': [bb_top],
                'bb_width': [bb_width],
                'bb_height': [bb_height],
                'class': [class_name],
                'confidence': [confidence]
            })
            df = pd.concat([df, df1], ignore_index=True)

            
        return df

    def track_detect(self, img, detections, tracker):
        resdes = detections[:, :5]
        resultTracker = tracker.update(resdes)
        
        # create a line
        cv2.line(img, (0, 280), (350, 280), (255, 0, 0), 5)
        # plt.imshow(crop_img)

        for res in resultTracker:
            x1, y1, x2, y2, id = res
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'ID: {id}', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

           
        return img, resultTracker

    # # def append_to_dataframe(self, df, detections):
    #     for detection in detections:
    #         x1 = detection[0]
    #         y1 = detection[1]
    #         x2 = detection[2]
    #         y2 = detection[3]
    #         id = detection[4]
    #         # conf = detection[4]
    #         cx = (x1 + x2) // 2
    #         cy = (y1 + y2) // 2
    #         df1 = pd.DataFrame({'x1': [x1], 'y1': [y1], 'x2': [x2], 'y2': [y2], 'id': id, 'cx': [cx], 'cy': [cy]})
    #         df = pd.concat([df, df1], ignore_index=True)
            
    #     return df

    def process_frames(self):
        cap = self.cam.read()
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('result/output1.mp4', fourcc, 20.0, (640, 480))

        tracker = Sort(max_age=40, min_hits=10, iou_threshold=0.6)
        counter = 0
        

        start_time = time.time()
        while True:
            img = self.cam.read()
            video_id = cam.args.rtsp.split('/')[-1]
            detections = np.empty((0, 6))
            results = self.predict(img)
            detections, counter = self.plot_boxes(results, detections, counter)
            print(detections)
            detect_frame,restrack = self.track_detect(img, detections, tracker)

            # print

        
            out.write(detect_frame)

            # self.df = self.append_to_dataframe(self.df, detections, video_id, frame_number)

            # show the time elapsed
            current_time = time.time()
            elapsed_time = current_time - start_time
            fps = counter / elapsed_time
            cv2.putText(detect_frame, f'FPS: {str(int(fps))}', (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv2.imshow('Object Detection', detect_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam._stop()
        out.release()
        print("FPS:", counter / (time.time() - start_time))

    def start_processing(self):

        self.process_frames()

def parse_args():
    desc = 'Capture and display live camera video, while doing '
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="threshold when applyong non-maxima suppression")
    parser.add_argument("-C", "--CamDID", required=True, help="Camera DID for AI")
    parser.add_argument("-Objapi", "--HeadApi", required=True, help="Object / Head API for camera")
    parser.add_argument("-o", "--output", type=str, help="path to optional output video file")
    parser.add_argument("-url", "--url", required=True, help="url alert to be sent")
    parser.add_argument("-time_int", "-time_int", type=int, default=30, help="Time interval for sending request to server")
    args = parser.parse_args()
    return args

args = parse_args()
# cam = open_cam_rtsp(args.rtsp)
cam = Camera(args)

detector = ObjectDetection(cam, args)
detector.start_processing()