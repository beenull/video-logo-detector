# Logo detection in videos
This repo contains an experimental code used to try to detect and localize predefined logo inside videos. The aim of the project was to count how many times a specific logo has appeared in a video so that this information can be used by media agencies.

The methods that have been tried:

1. Feature matching using SIFT, AKAZE and ORB.
2. Using Color Description of the logo by filtering the video from main logo's color then search the filtered are for feature matching with the reference logo.