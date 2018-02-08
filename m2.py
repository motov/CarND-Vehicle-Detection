from heat_map import *
from hog_subsample import *
from lesson_functions import *
import matplotlib.pyplot as plt
import numpy as np
import cv2


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

#  project video
# cap = cv2.VideoCapture('project_video_cut.mp4')
# test video
cap = cv2.VideoCapture('test_video.mp4')

# create output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('project_cut_result.mp4',fourcc, 25.0, (1280,720))

# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video stream or file")

finalList = []
imgResult = np.zeros((720, 1280, 3), dtype=float)
numFrame = 0
heatmap = np.zeros((720, 1280), dtype=float)
bbox = []
hashmapSh = np.zeros((720, 1280), dtype=float)
threshold = 2


while cap.isOpened():
    # Capture frame-by-frame
    # cap.read() generate a BGR????!!!!!!! frame 0-255
    ret, frame = cap.read()

    if ret == True:
        numFrame += 1
        print(numFrame)
        # get box list for current frame
        # note that when training the data, the input is RGB because of mpimg.imread() function.
        # Here, frame is BGR format because of cv2 --> cap.read() function.
        # We need to transfer it to RGB for prediction because model is trained in RGB with value 0 - 1.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # plt.imshow(frame)
        # plt.show()

        bbox = find_cars(frame)

        # transfer frame to BGR for cv2 process.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # imgResult, heatmap, finalList = gen_frame_result(frame, numFrame, finalList, bbox, heatmap, threshold)

        if numFrame == 1:
            imgResult = frame

        # update heatmap so that it sums latest 10 frames of boxes.
        if numFrame <= 3:
            # add box_list to list
            finalList.append(bbox)
            if len(bbox) > 0:
                for boxlist in bbox:
                    for box in boxlist:
                        heatmap[box[1]:box[3], box[0]:box[2]] += 1
        else:
            oldList = finalList.pop(0)
            finalList.append(bbox)
            if len(oldList) > 0:
                for boxlist in oldList:
                    for box in boxlist:
                        heatmap[box[1]:box[3], box[0]:box[2]] -= 1
            if len(bbox) > 0:
                for boxlist in bbox:
                    for box in boxlist:
                        heatmap[box[1]:box[3], box[0]:box[2]] += 1
            if numFrame % 3 == 1:
                heatmapSh = apply_threshold(np.copy(heatmap), threshold)
                # draw_labeled_bboxes
                labels = label(heatmapSh)
                imgResult = draw_labeled_bboxes(frame, labels)
            # plt.imshow(imgResult)
            # plt.show()


        # save current frame
        out.write(imgResult)

        # Display the resulting frame
        cv2.imshow('Frame', imgResult)

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