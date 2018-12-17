#%%
# Global Variables
juhayna_video_path = "videos/juhayna_youghart.mp4"
we_video_path = "videos/WE Mobile Internet.mp4"
juhayna_logo_path = "images/juhayna.jpg"

#%%
####################################################3
# Reading and displaying videos
import numpy as np
import cv2

cap = cv2.VideoCapture("videos/WE Mobile Internet.mp4")

while True:
    if not cap.isOpened():
        print("Video Capture is not initialized properly")
        break
    ret, frame = cap.read()
    cv2.putText(frame, str(frame.shape[:2]), (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0,0,255))
    cv2.imshow("Video", frame)

    if cv2.waitKey(35) & 0xFF == 13:
        break

cap.release()
cv2.destroyAllWindows()
####################################################################333

#%%
############################################
# Writing videos
import numpy as np
import cv2

cap = cv2.VideoCapture("videos/WE Mobile Internet.mp4")
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('flipped.mp4', fourcc, 25, (1280,720))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 0)
        out.write(frame)
    
        cv2.imshow("Reversed", frame)
        if cv2.waitKey(35) & 0xFF == 13:
            break
    else:
        print("No more frames retrieved")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
##################################################################3

#%%
#################################################
# Juhayna Commercial. A snapshot to experiment on
import numpy as np
import cv2

cap = cv2.VideoCapture("videos/juhayna_youghart.mp4")
juhayna_logo = cv2.imread("images/juhayna.jpg")
cv2.imshow("Juhayna Logo", juhayna_logo)

while cap.isOpened():
    ret, frame = cap.read()
    
    cv2.imshow("Video", frame)
    if cv2.waitKey(40) & 0xFF == 13:
        break

    if not ret:
        break

cap.release()
cv2.destroyAllWindows()
###############################################################33

#%%
##########################
# Collect snapshots from the video
import numpy as np
import cv2

cap = cv2.VideoCapture(juhayna_video_path)
name_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow("Video", frame)
    k = cv2.waitKey(40) & 0xFF
    if k == ord('s'):
        cv2.imwrite("images/snapshots/snapshot-" + str(name_counter) + ".jpg", frame)
        name_counter += 1
    elif k == 13:
        break
cap.release()
cv2.destroyAllWindows()
####################################

#%%
########################################
# Detecting features in the logo and the picture using AKAZE
import numpy as np
import cv2
import imutils

cap = cv2.VideoCapture(juhayna_video_path)

# fourcc is the codec
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# cv2.VideoWriter(video_name, codec, frame_rate, dimensions)
out = cv2.VideoWriter('matches.mp4', fourcc, 25, (640, 480))

logo = cv2.imread("images/juhayna.jpg")
# image = cv2.imread("images/snapshots/snapshot-1.jpg")
orb = cv2.ORB_create(500, 1.2)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

while cap.isOpened():
    # cv2.imshow("Image", image)
    # cv2.imshow("logo", logo)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # MIN_MAX_MATCH = 10
    # To do here in the future, select the dominant colors in the logo image and filter them out
    # from the target image
    ret, image = cap.read()
    logo_gry = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    image_gry = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    logo_key_points, logo_descriptors = orb.detectAndCompute(logo_gry, None)
    image_key_points, image_descriptors = orb.detectAndCompute(image_gry, None)

    image2 = image.copy()
    logo2 = logo.copy()
    cv2.drawKeypoints(logo, logo_key_points, logo2)
    cv2.drawKeypoints(image, image_key_points, image2)


    # cv2.imshow("Image Key Points", image2)
    # cv2.imshow("Logo Key Points", logo2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    
    matches = sorted(bf.match(logo_descriptors, image_descriptors),
                                                    key = lambda x: x.distance)
    # print(len(matches), "were found.")

    image3 = cv2.drawMatches(logo_gry, logo_key_points, image_gry, image_key_points, matches[:50], image, flags=2)
    # cv2.imwrite("matches.jpg", image3)
    f = imutils.resize(image3, width=640)
    out.write(f)
    cv2.imshow("Matches", image3)
    if cv2.waitKey(40) & 0xFF == 13:
        break
    elif not ret:
        break

out.release()
cap.release()
cv2.destroyAllWindows()
###################################

#%%
########################################
# Detecting features in the logo and the picture using AKAZE
import numpy as np
import cv2
import imutils

cap = cv2.VideoCapture(juhayna_video_path)

# fourcc is the codec
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# cv2.VideoWriter(video_name, codec, frame_rate, dimensions)
out = cv2.VideoWriter('matches.mp4', fourcc, 25, (640, 480))

logo = cv2.imread("images/juhayna.jpg")
# image = cv2.imread("images/snapshots/snapshot-1.jpg")
orb = cv2.AKAZE_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

while cap.isOpened():
    # cv2.imshow("Image", image)
    # cv2.imshow("logo", logo)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # MIN_MAX_MATCH = 10
    # To do here in the future, select the dominant colors in the logo image and filter them out
    # from the target image
    ret, image = cap.read()
    logo_gry = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    image_gry = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    logo_key_points, logo_descriptors = orb.detectAndCompute(logo_gry, None)
    image_key_points, image_descriptors = orb.detectAndCompute(image_gry, None)

    image2 = image.copy()
    logo2 = logo.copy()
    cv2.drawKeypoints(logo, logo_key_points, logo2)
    cv2.drawKeypoints(image, image_key_points, image2)


    cv2.imshow("Image Key Points", image2)
    cv2.imshow("Logo Key Points", logo2)
    cv2.waitKey()
    # cv2.destroyAllWindows()

    
    matches = sorted(bf.match(logo_descriptors, image_descriptors),
                                                    key = lambda x: x.distance)
    # # print(len(matches), "were found.")

    image3 = cv2.drawMatches(logo_gry, logo_key_points, image_gry, image_key_points, matches[:50], image, flags=2)
    # # cv2.imwrite("matches.jpg", image3)
    # f = imutils.resize(image3, width=640)
    # out.write(f)
    cv2.imshow("Matches", image3)
    cv2.waitKey()
    break
    # if cv2.waitKey(40) & 0xFF == 13:
    #     break
    # elif not ret:
    #     break

out.release()
cap.release()
cv2.destroyAllWindows()
###################################






#%%
# Experimenting with color spaces and filtering
# Experiment: Filter out the images by the most dominant colors in the logo image
import numpy as np
import cv2
from matplotlib import pyplot as plt

# This to get the maximum local of the histogram of HSV channels
from scipy.signal import argrelextrema

MAX_NUMBER_OF_PEAKS = 2
THRESHOLD = np.array([50., 0., 255.])

we_logo_path = "images/we_logo.png"
logo = cv2.imread(we_logo_path)
# cv2.imshow("logo", logo)
logo_hsv = cv2.cvtColor(logo, cv2.COLOR_BGR2HSV)
max_vals = np.zeros(shape=(3))
max_vals[0] = 179
max_vals[1] = 255
max_vals[2] = 255

for i, upper_bound in enumerate(max_vals):
# upper_bound = 255
    hist = cv2.calcHist([logo_hsv], [i], None, [upper_bound + 1], [0,upper_bound])
    max_vals[i] = np.argmax(hist)
# To get MAX_NUMBER_OF_PEAKS local maxima
# extrema = argrelextrema(hist, np.greater, mode='wrap')[0]
# extrema_l = list(extrema)
# extrema_l.sort(key = lambda x: hist[x], reverse=True)

# plt.title("Histogram of H Channel")
# plt.plot(hist)
# plt.plot(extrema, hist[extrema], 'r*')
# plt.show()

# max_values = extrema_l[:MAX_NUMBER_OF_PEAKS]
# print(max_vals - 15)

mask = cv2.inRange(logo_hsv, max_vals - THRESHOLD, max_vals + THRESHOLD)
print(mask)
# print(logo_hsv.shape)
filtered_logo = np.zeros(shape=logo_hsv.shape)
filtered_logo[:,:,0] = cv2.bitwise_and(logo_hsv[:,:,0], mask)
filtered_logo[:,:,1] = cv2.bitwise_and(logo_hsv[:,:,1], mask)
filtered_logo[:,:,2] = cv2.bitwise_and(logo_hsv[:,:,2], mask)
cv2.imshow("Filtered Logo", filtered_logo)


cv2.waitKey()
cv2.destroyAllWindows()




#%%
# Experimenting with color spaces and filtering
# Experiment: Filter out the video by the most dominant colors in the logo image
import numpy as np
import cv2
# from matplotlib import pyplot as plt

# This to get the maximum local of the histogram of HSV channels
# from scipy.signal import argrelextrema

# MAX_NUMBER_OF_PEAKS = 2
THRESHOLD = np.array([50., 50., 120.])
juhayna_video_path = "videos/juhayna_youghart.mp4"
cap = cv2.VideoCapture(juhayna_video_path)


while cap.isOpened():
    ret, logo = cap.read()
    # cv2.imshow("logo", logo)
    logo_hsv = cv2.cvtColor(logo, cv2.COLOR_BGR2HSV)
    max_vals = np.zeros(shape=(3))
    max_vals[0] = 179
    max_vals[1] = 255
    max_vals[2] = 255

    for i, upper_bound in enumerate(max_vals):
    # upper_bound = 255
        hist = cv2.calcHist([logo_hsv], [i], None, [upper_bound + 1], [0,upper_bound])
        max_vals[i] = np.argmax(hist)
    # To get MAX_NUMBER_OF_PEAKS local maxima
    # extrema = argrelextrema(hist, np.greater, mode='wrap')[0]
    # extrema_l = list(extrema)
    # extrema_l.sort(key = lambda x: hist[x], reverse=True)

    # plt.title("Histogram of H Channel")
    # plt.plot(hist)
    # plt.plot(extrema, hist[extrema], 'r*')
    # plt.show()

    # max_values = extrema_l[:MAX_NUMBER_OF_PEAKS]
    # print(max_vals - 15)

    mask = cv2.inRange(logo_hsv, max_vals - THRESHOLD, max_vals + THRESHOLD)
    
    # It's important to make dtype = uint8
    flogo = np.zeros(shape=logo.shape, dtype=np.uint8)
    flogo[:,:,0] = cv2.bitwise_and(mask, logo[:,:,0])
    flogo[:,:,1] = cv2.bitwise_and(mask, logo[:,:,1])
    flogo[:,:,2] = cv2.bitwise_and(mask, logo[:,:,2])
    
    cv2.imshow("Filtered Logo", flogo)
    if cv2.waitKey(40) & 0xFF == 13:
        break
    if not ret:
        break

cap.release()
cv2.destroyAllWindows()



#%%
# New experiment
