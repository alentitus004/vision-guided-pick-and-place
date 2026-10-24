import nidaqmx
from nidaqmx.constants import LineGrouping
import time

# Half step sequence — clockwise
CW_SEQUENCE  = [1, 5, 4, 6, 2, 10, 8, 9]
# Anticlockwise = reverse
ACW_SEQUENCE = [9, 8, 10, 2, 6, 4, 5, 1]

DEGREES_PER_HALF_STEP = 0.9  # half step mode

def rotate_stepper(delta_angle):
    """
    Rotate stepper by delta_angle degrees.
    Positive = clockwise
    Negative = anticlockwise
    """
    if abs(delta_angle) < 0.5:
        print("Angle too small, skipping rotation.")
        return

    steps = round(abs(delta_angle) / DEGREES_PER_HALF_STEP)
    sequence = CW_SEQUENCE if delta_angle < 0 else ACW_SEQUENCE

    print(f"Delta: {delta_angle:.1f}° → {steps} half-steps → {steps * DEGREES_PER_HALF_STEP:.1f}°")

    step_index = 0
    for step in range(steps):
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(
                'Dev1/port0/line0:7',
                line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
            task.write([sequence[step_index % 8]])
        time.sleep(0.01)  # tune this if missing steps
        step_index += 1

    print("Rotation complete!")


# --- Usage from vision system ---
value=-26 #just enter the value from the vision angle put -ve if negative
pallet_angle =0   # from vision code
block_angle  = value # just write the negative value and write #-(180+value) if negative # from vision code

delta = pallet_angle - block_angle 

if delta > 180:
    delta -= 360
elif delta < -180:
    delta += 360 # 0 - (-27) = +27° clockwise
rotate_stepper(delta)