import cv2


# === User Constants ===

# Number of frames
max_frames_num = 14

# Pathname to the dir where photos of the chessboard pattern are stored
chessboard_pattern_dirname = 'chessboard_patterns/'

# ======================

# Initialize video client
cap = cv2.VideoCapture(0)

frames_num = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Save frame
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(chessboard_pattern_dirname + 'pic' + str(frames_num) + '.jpg', frame)
        frames_num += 1

    # If enough frames, then break the loop
    if frames_num > max_frames_num:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
