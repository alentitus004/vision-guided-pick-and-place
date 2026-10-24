import cv2
import numpy as np
import sys
# ───────── CAMERA ─────────
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cam.set(cv2.CAP_PROP_FOCUS, 6)

cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)
cam.set(cv2.CAP_PROP_EXPOSURE, -5)

if not cam.isOpened():
    print("Camera failed to open.")
    sys.exit()

# ───────── HSV ─────────
RED_LOWER1  = np.array([0, 89, 129])
RED_UPPER1  = np.array([10, 255, 255])
RED_LOWER2  = np.array([170, 89, 129])
RED_UPPER2  = np.array([180, 255, 255])

YELLOW_LOWER = np.array([24, 72, 97])
YELLOW_UPPER = np.array([40, 255, 255])

kernel = np.ones((3,3), np.uint8)

# ───────── SLOT POSITIONS ─────────
def get_slot_positions(cx, cy, angle_deg, mm_per_pixel):
    angle = np.deg2rad(angle_deg)

    ux =  np.cos(angle)
    uy = -np.sin(angle)

    vx = -uy
    vy =  ux

    d_long_px  = 51.5 / mm_per_pixel
    d_short_px = 19.5 / mm_per_pixel

    return [
        (int(cx + ux * d_long_px),  int(cy + uy * d_long_px)),   # S1
        (int(cx - ux * d_long_px),  int(cy - uy * d_long_px)),   # S2
        (int(cx + vx * d_short_px), int(cy + vy * d_short_px)),  # S3
        (int(cx - vx * d_short_px), int(cy - vy * d_short_px)),  # S4
    ]

# ───────── ARUCO ─────────
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
detector   = cv2.aruco.ArucoDetector(aruco_dict)

MM_PER_PIXEL = 0.511635

OFFSET_MM_X = 68.5
OFFSET_MM_Y = 68.5

OFFSET_PX_X = OFFSET_MM_X / MM_PER_PIXEL
OFFSET_PX_Y = OFFSET_MM_Y / MM_PER_PIXEL

prev_origin = None
prev_coords = {}

# ───────── DOBOT SETTINGS ─────────
DOBOT_IDS = [1]

PAD_LEFT   = 420
PAD_TOP    = 400
PAD_RIGHT  = 160
PAD_BOTTOM = 100

last_excl = [0, 0, 0, 0]
excl_x1 = excl_y1 = excl_x2 = excl_y2 = 0

# ───────── FROZEN PALLET STATE (single image capture) ─────────
frozen_slots      = None
frozen_pallet_angle = None

# ───────── WINDOW ─────────
cv2.namedWindow("Final Vision", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Final Vision", 1280, 720)

print("=== Point camera at EMPTY pallet, press SPACE to capture slot positions ===")
print("=== After capture, place blocks. Press Q to quit. ===")

# ───────── HELPER: normalize angle to [-180, 180] ─────────
def normalize_angle(a):
    while a > 180:
        a -= 360
    while a < -180:
        a += 360
    return a

# ───────── LOOP ─────────
while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, -1)
    display = frame.copy()

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ───────── ARUCO + DOBOT ─────────
    corners, ids, _ = detector.detectMarkers(gray)
    origin_px = None
    origin_found = False
    dobot_found = False
    dobot_pts_x = []
    dobot_pts_y = []

    if ids is not None:
        for i, mid in enumerate(ids.flatten()):
            mid = int(mid)

            pts = corners[i][0].astype(int)

            cx_a = int(np.mean(pts[:,0]))
            cy_a = int(np.mean(pts[:,1]))

            if mid == 1:
                raw_origin = (
                    int(cx_a - OFFSET_PX_X),
                    int(cy_a - OFFSET_PX_Y)
                )

                if prev_origin is None:
                    prev_origin = raw_origin

                alpha = 0.8
                origin_px = (
                    int(alpha * prev_origin[0] + (1 - alpha) * raw_origin[0]),
                    int(alpha * prev_origin[1] + (1 - alpha) * raw_origin[1])
                )

                prev_origin  = origin_px
                origin_found = True

                cv2.circle(display, origin_px, 10, (0,255,0), -1)

            cv2.putText(display,
                        f"Aruco ID = {mid}",
                        (cx_a + 10, cy_a - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2)

            if mid in DOBOT_IDS:
                dobot_found = True
                dobot_pts_x.extend(pts[:,0])
                dobot_pts_y.extend(pts[:,1])

    # ───────── EXCLUSION ZONE ─────────
    if dobot_found:
        excl_x1 = max(0, int(min(dobot_pts_x)) - PAD_LEFT)
        excl_y1 = max(0, int(min(dobot_pts_y)) - PAD_TOP)
        excl_x2 = min(1920, int(max(dobot_pts_x)) + PAD_RIGHT)
        excl_y2 = min(1080, int(max(dobot_pts_y)) + PAD_BOTTOM)
        last_excl = [excl_x1, excl_y1, excl_x2, excl_y2]
    else:
        excl_x1, excl_y1, excl_x2, excl_y2 = last_excl

    cv2.rectangle(display, (excl_x1,excl_y1),(excl_x2,excl_y2),(0,0,255),2)

    # ───────── MASKS ─────────
    red_mask = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1) | \
               cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)

    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

    red_mask[excl_y1:excl_y2, excl_x1:excl_x2] = 0
    yellow_mask[excl_y1:excl_y2, excl_x1:excl_x2] = 0

    red_mask    = cv2.morphologyEx(red_mask,    cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN,  kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    # ───────── RED MARKERS ─────────
    red_markers = []
    r_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt_r in r_contours:
        if cv2.contourArea(cnt_r) > 200:
            rect_r = cv2.minAreaRect(cnt_r)
            rx, ry = int(rect_r[0][0]), int(rect_r[0][1])
            red_markers.append((rx, ry))
            cv2.circle(display, (rx, ry), 8, (0,0,255), -1)

    # ───────── YELLOW CONTOURS ─────────
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pallet_angle = None
    block_angles = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4000:
            continue

        rect = cv2.minAreaRect(cnt)
        box  = np.intp(cv2.boxPoints(rect))

        cx, cy = int(rect[0][0]), int(rect[0][1])
        w, h   = rect[1]

        if len(red_markers) == 0:
            continue

        nearest_red = min(red_markers, key=lambda r: np.hypot(cx - r[0], cy - r[1]))

        long_side  = max(w, h)
        short_side = min(w, h)

        if 200 < long_side <= 320 and 150 < short_side <= 200:
            label = "PALLET"
            color = (0,255,0)
        elif long_side < 210 and short_side < 150:
            label = "BLOCK"
            color = (180,105,255)
        else:
            continue

        (center, (w, h), angle) = rect
        if w < h:
            angle += 90

        rad = np.deg2rad(angle)
        vx = np.cos(rad)
        vy = np.sin(rad)

        dx_r = nearest_red[0] - cx
        dy_r = nearest_red[1] - cy

        if vx * dx_r + vy * dy_r > 0:
            vx = -vx
            vy = -vy

        # arctan2 already returns [-180, 180] — no conversion needed.
        # The old code turned -1° → 359° → -179°, flipping ux/uy and swapping S1↔S2.
        final_angle = np.degrees(np.arctan2(-vy, vx))

        if label == "PALLET":
            pallet_angle = final_angle

            if frozen_slots is not None:
                slots = frozen_slots
                disp_pallet_angle = frozen_pallet_angle
            else:
                slots = get_slot_positions(cx, cy, final_angle, MM_PER_PIXEL)
                disp_pallet_angle = final_angle
            arc_values = [None, None, None, None]
            for i, (sx, sy) in enumerate(slots):

                # ── FIX: normalize to [-180, 180] correctly for both +ve and -ve pallet angles ──
                if i < 2:
                    slot_orient = disp_pallet_angle + 90
                else:
                    slot_orient = disp_pallet_angle

                slot_orient = normalize_angle(slot_orient)  # ← replaces the old single `if > 180` check

                if origin_found:
                    cv2.line(display, origin_px, (sx, sy), (255,105,180), 2)

                cv2.circle(display, (sx, sy), 6, (255,0,0), -1)

                if origin_found:
                    dx_px = sx - origin_px[0]
                    dy_px = sy - origin_px[1]
                    robot_x = dy_px * MM_PER_PIXEL
                    robot_y = dx_px * MM_PER_PIXEL
                    cv2.putText(display,
                                f"X:{round(robot_x,1)} Y:{round(robot_y,1)}",
                                (sx-80, sy+20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,140,255), 2)

                cv2.putText(display,
                            f"S{i+1} ({round(slot_orient,1)}d)",
                            (sx+5, sy-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255,0,0), 2)

              

                # ── Arc angle: signed angle at dobot origin from block line to slot line ──
                if len(block_angles) > 0 and origin_found:
                    bx, by, _ = block_angles[0]
                    vec_slot  = np.array([sx - origin_px[0], sy - origin_px[1]], dtype=float)
                    vec_block = np.array([bx - origin_px[0], by - origin_px[1]], dtype=float)
                    # signed angle: positive = CCW, negative = CW
                    arc_angle = np.degrees(np.arctan2(
                        vec_slot[0]*vec_block[1] - vec_slot[1]*vec_block[0],
                        np.dot(vec_slot, vec_block)
                    ))
                    arc_values[i] = arc_angle
            # ───────── SIDE DISPLAY: ARC ANGLES ─────────
            start_x = 20
            start_y = 80
            
            for i in range(4):
                if arc_values[i] is not None:
                    text = f"S{i+1} arc = {round(arc_values[i],1)}°"
                else:
                    text = f"S{i+1} arc = --"
            
                cv2.putText(display,
                            text,
                            (start_x, start_y + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,255,128),
                            2)

        elif label == "BLOCK":
            block_angles.append((cx, cy, final_angle))
        
        if origin_found:
            dx_px = cx - origin_px[0]
            dy_px = cy - origin_px[1]
            robot_x = dy_px * MM_PER_PIXEL
            robot_y = dx_px * MM_PER_PIXEL
            cv2.line(display, origin_px, (cx, cy), (0,255,255), 1)
            cv2.putText(display,
                        f"X:{round(robot_x,1)} Y:{round(robot_y,1)}",
                        (cx-120, cy-55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255,25,25), 2)

        arm = int(max(w,h)/2) + 10
        cv2.arrowedLine(display, (cx-arm, cy), (cx+arm, cy), (255,255,255), 2)

        x1 = int(cx - vx * arm); y1 = int(cy - vy * arm)
        x2 = int(cx + vx * arm); y2 = int(cy + vy * arm)
        cv2.arrowedLine(display, (x1,y1), (x2,y2), color, 2)
        cv2.drawContours(display, [box], 0, color, 3)

        cv2.putText(display,
                    f"{label} | {round(final_angle,1)} deg",
                    (cx-120, cy-30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

    # ───────── ARC ANGLE CONSOLE (after loop) ─────────
    # Signed arc angle at dobot origin: from block line → each slot line
    # Positive = CCW, Negative = CW
    if pallet_angle is not None and len(block_angles) > 0 and origin_found:
        bx, by, _ = block_angles[0]
        vec_block = np.array([bx - origin_px[0], by - origin_px[1]], dtype=float)
        active_slots = frozen_slots if frozen_slots is not None else []
        for si, (sx, sy) in enumerate(active_slots):
            vec_slot = np.array([sx - origin_px[0], sy - origin_px[1]], dtype=float)
            arc_angle = np.degrees(np.arctan2(
                vec_slot[0]*vec_block[1] - vec_slot[1]*vec_block[0],
                np.dot(vec_slot, vec_block)
            ))
            s_orient = normalize_angle(pallet_angle + 90 if si < 2 else pallet_angle)
            print(f"  S{si+1} | orient: {round(s_orient,1)}d | arc from block: {round(arc_angle,1)}d")

    # ───────── FROZEN SLOT OVERLAY ─────────
    if frozen_slots is not None and pallet_angle is None:
        for i, (sx, sy) in enumerate(frozen_slots):

            # ── FIX: same normalize_angle fix applied here too ──
            if i < 2:
                slot_orient = frozen_pallet_angle + 90
            else:
                slot_orient = frozen_pallet_angle

            slot_orient = normalize_angle(slot_orient)  # ← replaces the old single `if > 180` check

            cv2.circle(display, (sx, sy), 6, (255,0,0), -1)
            cv2.putText(display,
                        f"S{i+1} ({round(slot_orient,1)}d)",
                        (sx+5, sy-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,0,0), 2)

            if origin_found:
                cv2.line(display, origin_px, (sx, sy), (255,105,180), 2)
                dx_px = sx - origin_px[0]
                dy_px = sy - origin_px[1]
                robot_x = dy_px * MM_PER_PIXEL
                robot_y = dx_px * MM_PER_PIXEL
                cv2.putText(display,
                            f"X:{round(robot_x,1)} Y:{round(robot_y,1)}",
                            (sx-80, sy+20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,140,255), 2)

            # ── Slot orientation label ──
            cv2.putText(display,
                        f"S{i+1} orient: {round(slot_orient,1)}d",
                        (sx+5, sy-20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,200,0), 2)

            # ── Arc angle: signed angle at dobot origin from block line to slot line ──
            if len(block_angles) > 0 and origin_found:
                bx, by, _ = block_angles[0]
                vec_slot  = np.array([sx - origin_px[0], sy - origin_px[1]], dtype=float)
                vec_block = np.array([bx - origin_px[0], by - origin_px[1]], dtype=float)
                arc_angle = np.degrees(np.arctan2(
                    
                    vec_slot[0]*vec_block[1] - vec_slot[1]*vec_block[0],
                    np.dot(vec_slot, vec_block)
                ))
                

    # ───────── STATUS ─────────
    status_txt = "FROZEN" if frozen_slots is not None else "LIVE — SPACE to capture"
    cv2.putText(display, status_txt, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0,255,255) if frozen_slots is not None else (0,165,255), 2)

    display_small = cv2.resize(display, (0,0), fx=0.8, fy=0.8)
    cv2.imshow("Final Vision", display_small)
    cv2.imshow("Red Mask",     red_mask)
    cv2.imshow("Yellow Mask",  yellow_mask)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord(' '):
        if pallet_angle is not None:
            for cnt in contours:
                if cv2.contourArea(cnt) < 4000:
                    continue
                rect = cv2.minAreaRect(cnt)
                cx_f, cy_f = int(rect[0][0]), int(rect[0][1])
                w_f, h_f = rect[1]
                if not (200 < max(w_f,h_f) <= 320 and 150 < min(w_f,h_f) <= 200):
                    continue
                frozen_slots        = get_slot_positions(cx_f, cy_f, pallet_angle, MM_PER_PIXEL)
                frozen_pallet_angle = pallet_angle
                print(f"=== Slots FROZEN at pallet angle {round(pallet_angle,1)} deg ===")
                for j, (sx, sy) in enumerate(frozen_slots):
                    orient = normalize_angle(pallet_angle + 90 if j < 2 else pallet_angle)
                    print(f"  S{j+1}: pixel=({sx},{sy})  orientation={round(orient,1)} deg")
                break
        else:
            print("No pallet detected — cannot freeze slots")
    elif key == ord('r'):
        frozen_slots        = None
        frozen_pallet_angle = None
        print("=== Slots UNFROZEN ===")

cam.release()
cv2.destroyAllWindows()