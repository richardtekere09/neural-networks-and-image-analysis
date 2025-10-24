import cv2, numpy as np

win = "test"
cv2.namedWindow(win)
cv2.createTrackbar("v", win, 50, 100, lambda x: None)
img = np.zeros((120, 320, 3), np.uint8)
while True:
    v = cv2.getTrackbarPos("v", win)
    frame = img.copy()
    cv2.putText(
        frame, f"v={v}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    cv2.imshow(win, frame)
    if (cv2.waitKey(20) & 0xFF) in (27, ord("q")):
        break
cv2.destroyAllWindows()
