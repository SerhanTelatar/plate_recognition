import cv2
import numpy as np

# Create a black image
image = np.zeros((512, 512, 3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
cv2.line(image, (0, 0), (511, 511), (255, 0, 0), 5)

cv2.imshow("Test Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()