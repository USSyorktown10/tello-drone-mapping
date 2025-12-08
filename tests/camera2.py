from djitellopy import Tello 
import cv2
import time
from threading import Thread

speed = 25
command_time_seconds= 3 

print("create tello object")
tello = Tello()

print("connect to drone")
tello.connect()

battery_level = tello.get_battery()
print(f"battery percentage: {battery_level}")


print("turn stream on")
tello.streamon()

print("read tello image")
frame_read = tello.get_frame_read()

print('press q to quit')
while True:
   tello_video_image = frame_read.frame

   if tello_video_image is not None:
      tello_video_image = cv2.cvtColor(frame_read.frame, cv2.COLOR_RGB2BGR)
      cv2.imshow("tellovideo", tello_video_image)

   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

tello.land()

time.sleep(2)

tello.streamoff()

cv2.destroyWindow("tellovideo")

cv2.destroyAllWindows()
