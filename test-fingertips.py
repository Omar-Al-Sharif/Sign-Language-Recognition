# import cv2
# import numpy as np

# def preprocess_image(image):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply pixel-wise adaptive Wiener filtering
#     filtered = cv2.ximgproc.adaptiveWienerFilter(gray, (5, 5))
    
#     return filtered

# def detect_fingertips(image):
#     # Compute edges using the Sobel operator
#     edges = cv2.Sobel(image, cv2.CV_8U, 1, 1, ksize=3)
    
#     # Perform morphological processing (dilation)
#     kernel = np.ones((5, 5), np.uint8)
#     dilated = cv2.dilate(edges, kernel)
    
#     # Perform image fill operation
#     _, filled = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY)
    
#     # Compute circular Hough transform
#     circles = cv2.HoughCircles(filled, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
#                                param1=50, param2=30, minRadius=10, maxRadius=30)
    
#     # If circles are detected, extract fingertip coordinates
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype(int)
#         fingertips = circles[:, :2]
#     else:
#         fingertips = []
    
#     return fingertips

# def count_fingers(image, fingertips):
#     # Scan the image to determine the longest Hough lines near fingertip circles
#     lines = cv2.HoughLines(image, rho=1, theta=np.pi / 180, threshold=50)
    
#     # If lines are detected, compare with fingertip coordinates to identify fingers
#     if lines is not None:
#         lines = lines[:, 0, :]
#         fingers = []
        
#         for x1, y1, x2, y2 in lines:
#             # Check if the line is close to any fingertip coordinates
#             distances = np.sqrt(np.power(fingertips[:, 0] - x1, 2) +
#                                 np.power(fingertips[:, 1] - y1, 2))
#             closest_fingertip = np.argmin(distances)
            
#             if distances[closest_fingertip] < 20:
#                 fingers.append(fingertips[closest_fingertip])
        
#         num_fingers = len(fingers)
#     else:
#         num_fingers = 0
    
#     return num_fingers

# # Load the hand image
# image = cv2.imread('3_men (1).JPG')

# # Preprocess the image
# preprocessed = preprocess_image(image)

# # Detect fingertips
# fingertips = detect_fingertips(preprocessed)

# # Count the number of fingers
# num_fingers = count_fingers(preprocessed, fingertips)

# # Display the results
# cv2.putText(image, f'Fingers: {num_fingers}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
#             1, (0, 0, 255), 2)
# cv2.imshow('Hand Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






# import cv2
# import numpy as np

# def preprocess_image(image):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply pixel-wise adaptive filtering using filter2D
#     kernel = np.ones((5, 5), np.float32) / 25
#     filtered = cv2.filter2D(gray, -1, kernel)
    
#     return filtered

# def detect_fingertips(image):
#     # Compute edges using the Sobel operator
#     edges = cv2.Sobel(image, cv2.CV_8U, 1, 1, ksize=3)
    
#     # Perform morphological processing (dilation)
#     kernel = np.ones((5, 5), np.uint8)
#     dilated = cv2.dilate(edges, kernel)
    
#     # Perform image fill operation
#     _, filled = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY)
    
#     # Compute circular Hough transform
#     circles = cv2.HoughCircles(filled, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
#                                param1=50, param2=30, minRadius=10, maxRadius=30)
    
#     # If circles are detected, extract fingertip coordinates
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype(int)
#         fingertips = circles[:, :2]
#     else:
#         fingertips = []
    
#     return fingertips

# def count_fingers(image, fingertips):
#     # Scan the image to determine the longest Hough lines near fingertip circles
#     lines = cv2.HoughLines(image, rho=1, theta=np.pi / 180, threshold=50)
    
#     # If lines are detected, compare with fingertip coordinates to identify fingers
#     if lines is not None:
#         lines = lines[:, 0, :]
#         fingers = []
        
#         for x1, y1, x2, y2 in lines:
#             # Check if the line is close to any fingertip coordinates
#             distances = np.sqrt(np.power(fingertips[:, 0] - x1, 2) +
#                                 np.power(fingertips[:, 1] - y1, 2))
#             closest_fingertip = np.argmin(distances)
            
#             if distances[closest_fingertip] < 20:
#                 fingers.append(fingertips[closest_fingertip])
        
#         num_fingers = len(fingers)
#     else:
#         num_fingers = 0
    
#     return num_fingers

# # Load the hand image
# image = cv2.imread('3_men (1).JPG')

# # Preprocess the image
# preprocessed = preprocess_image(image)

# # Detect fingertips
# fingertips = detect_fingertips(preprocessed)

# # Count the number of fingers
# num_fingers = count_fingers(preprocessed, fingertips)

# # Display the results
# cv2.putText(image, f'Fingers: {num_fingers}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
#             1, (0, 0, 255), 2)
# cv2.imshow('Hand Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





import cv2
import numpy as np

def preprocess_image(image):
    # Check if the image is grayscale, convert to BGR if necessary
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply pixel-wise adaptive filtering using filter2D
    kernel = np.ones((5, 5), np.float32) / 25
    filtered = cv2.filter2D(gray, -1, kernel)
    
    return filtered

def detect_fingertips(image):
    # Compute edges using the Sobel operator
    edges = cv2.Sobel(image, cv2.CV_8U, 1, 1, ksize=3)
    
    # Perform morphological processing (dilation)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel)
    
    # Perform image fill operation
    _, filled = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY)
    
    # Compute circular Hough transform
    circles = cv2.HoughCircles(filled, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=50, param2=30, minRadius=10, maxRadius=30)
    
    # If circles are detected, extract fingertip coordinates
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        fingertips = circles[:, :2]
    else:
        fingertips = []
    
    return fingertips

def count_fingers(image, fingertips):
    # Check if the image is grayscale, convert to BGR if necessary
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform HoughLinesP to detect lines
    lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    
    # If lines are detected, compare with fingertip coordinates to identify fingers
    if lines is not None:
        lines = lines[:, 0, :]
        fingers = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            # Convert fingertips list to NumPy array
            fingertips_array = np.array(fingertips)
            # Check if the line is close to any fingertip coordinates
            distances = np.sqrt(np.power(fingertips_array[:, 0] - x1, 2) +
                                np.power(fingertips_array[:, 1] - y1, 2))
            closest_fingertip = np.argmin(distances)
            
            if distances[closest_fingertip] < 20:
                fingers.append(fingertips[closest_fingertip])
        
        num_fingers = len(fingers)
    else:
        num_fingers = 0
    
    return num_fingers

# Load the hand image
image = cv2.imread('3_men (1).JPG')

# Preprocess the image
preprocessed = preprocess_image(image)

# Detect fingertips
fingertips = detect_fingertips(preprocessed)

# Count the number of fingers
num_fingers = count_fingers(preprocessed, fingertips)

# Display the results
cv2.putText(image, f'Fingers: {num_fingers}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2)
cv2.imshow('Hand Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
