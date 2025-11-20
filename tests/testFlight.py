from djitellopy import Tello

print("create tello object")
tello = Tello()

tello.connect()
battery_level = tello.get_battery()
print(f"battery percentage: {battery_level}")
tello.takeoff()
tello.move_forward(50)
tello.move_right(50)
tello.rotate_clockwise(180)
tello.move_forward(50)
tello.rotate_clockwise(90)
tello.move_forward(90)
tello.rotate_counter_clockwise(90)
tello.move_forward(50)
tello.rotate_clockwise(90)
tello.move_forward(50)
running = True
counter = 0
while running:
    tello.flip_forward()
    counter = counter + 1
    if counter == 3:
        running = False
tello.flip_right()
tello.flip_forward()
tello.flip_back()
tello.rotate_clockwise(90)
tello.move_forward(65)
tello.land()
