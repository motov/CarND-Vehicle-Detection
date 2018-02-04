from lesson_functions import *
import matplotlib.pyplot as plt
import numpy as np
import cv2


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('project_video.mp4')
# create output video
out = cv2.VideoWriter('project_result.mp4',fourcc, 25.0, (640,480))

# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video stream or file")

numFrame = 0
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        numFrame += 1
        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # get box list for current frame
        out_img,box_list = find_cars()

        if numFrame <= 10:
            # add box_list to list
        else:
            # delete old list, add new list, apply threshold

            # draw_labeled_bboxes

            # save current frame
            out.write(xxx)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()