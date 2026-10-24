# Vision-Guided Pick-and-Place

A pick-and-place setup where a camera finds the blocks, a Dobot Magician picks them up, and a stepper motor makes sure each block is turned the right way before it goes into the pallet.

We added the stepper because the Dobot's end effector can't do the rotation we needed on its own. So the Dobot handles moving around, and the stepper handles turning the block to match the slot.

## How it works

A USB camera (1920x1080) looks at the blocks and the pallet. In OpenCV we convert the image to HSV and pick out the objects by colour, then clean up the masks and find contours. `minAreaRect()` gives the centre and angle of each object.

A rectangle only gives an axis, so it can't tell which end is the front. To fix that, each object has a red marker. We compare the direction from the centre to the marker with the rectangle's axis (dot product), and then `atan2()` gives the final angle.

To get from pixels to robot coordinates, we use an ArUco marker (ID 1) as the reference and a calibrated scale of 0.511635 mm per pixel. The camera axes then get swapped over to match the robot's (`robot_x = dy`, `robot_y = dx`).

The pallet slots aren't detected one by one. We calculate them from the pallet's centre and angle, using the known spacing (51.5 mm along the long side, 19.5 mm along the short side, 50 x 25 mm slots). So if the pallet is rotated, the slots rotate with it. To check if a slot is already full, we look at a small area around its centre and see how much of it is yellow. Over 40% means filled.

Then comes the rotation. The angle needed is just `slot angle - block angle`. The stepper turns by that amount using two half-step sequences (one per direction) at 0.9° per step, so steps = angle / 0.9, rounded. After that the Dobot is meant to place the block in the slot.

## Steps in order

1. Capture image
2. Detect blocks and pallet
3. Get position and angle of each
4. Convert to robot coordinates
5. Work out slot positions and pick an empty one
6. Calculate the angle difference
7. Rotate with the stepper
8. Pick and place with the Dobot

## What we used

- USB camera
- Dobot Magician (PyDobot)
- Stepper motor
- NI-DAQ hardware (NI-DAQmx) to drive the stepper
- Python and OpenCV

## Challenges

- **Dobot offset (the main problem):** Integration was the hardest part. The Dobot's actual position had a big offset from the coordinates we sent it. We tried correcting it with a single global offset, but the error wasn't constant and changed from one coordinate to another, so one fixed value couldn't fix it everywhere. Because of this we couldn't get accurate pick-and-place working across the workspace.
- Lighting changed the colour detection a lot, so the HSV values needed tuning
- The 180° problem with the angle (the red marker fixed it)
- Getting the calibration and axis mapping right
- Making sure the vision, Dobot, and stepper all used the same angle direction
- Turning the angle into the right number of steps

## Status

The vision side works: it detects the blocks and pallet, estimates position and angle, calculates the slots, and works out the rotation needed. The stepper control is also in place. We couldn't finish the full pick-and-place because of the Dobot's position offset, which varied across the workspace and couldn't be fixed with a global correction.

## Future work

- Map the Dobot's offset at many points and use position-dependent correction instead of one global value
- Recalibrate the camera-to-robot transform with more reference points
- Complete the full end-to-end run
