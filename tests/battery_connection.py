from djitellopy import Tello 
import cv2
import time
# Note: 'opencv' library import is redundant; functionality comes from 'cv2'

print("create tello object")
tello = Tello()

print("connect to drone")
tello.connect()

battery_level = tello.get_battery()
print(f"battery percentage: {battery_level}")

state = tello.get_current_state()
print(f"current state: {state}")

barometer = tello.get_barometer()
print(f"barometer: {barometer}")