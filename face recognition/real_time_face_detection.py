#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 15:06:41 2021

@author: ecekurnaz
"""
import cv2
import face_recognition

all_face_locations=[]
cv2.namedWindow("preview")
webcam_video_stream = cv2.VideoCapture(0)
import time
time.sleep(3.0)

rval, frame = webcam_video_stream.read()
while True:
  if frame is not None:   
     cv2.imshow("preview", frame)
  rval, frame = webcam_video_stream.read()

  if cv2.waitKey(1) & 0xFF == ord('q'):
     break
  ret,current_frame=webcam_video_stream.read()
  current_frame_small=cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
  all_face_locations=face_recognition.face_locations(current_frame_small,model="hog")
  for index,current_face_location in enumerate(all_face_locations):
   top_pos,right_pos,bottom_pos,left_pos=current_face_location
   top_pos=top_pos*4
   right_pos=right_pos*4
   bottom_pos=bottom_pos*4
   left_pos=left_pos*4
   print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
   
   cv2.rectangle(current_frame, (left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
  cv2.imshow("Webcam Video",current_frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
webcam_video_stream.release()
cv2.destroyAllWindows()