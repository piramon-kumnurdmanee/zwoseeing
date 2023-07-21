"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
frame = np.array([
    1, 2, 3, 4, 3, 2, 1,
    1, 2, 3, 4, 3, 2, 1,
    1, 2, 3, 4, 3, 2, 1,
    1, 2, 3, 4, 3, 2, 1
])

def find_image_centroid(frame):
    height, width = frame.shape
    sum_x, sum_y, total = 0, 0, 0
    
    for y, row in enumerate(frame):
        for x, pixel in enumerate(row):
            sum_x += x * pixel
            sum_y += y * pixel
            total += pixel

    centroid_x = sum_x / total
    centroid_y = sum_y / total

    return centroid_x, centroid_y

frame = frame.reshape(frame[frame.columns[0]].count(),len(frame.index))  # Reshape the frame to a 2D array
centroid_x, centroid_y = find_image_centroid(frame)

print("Centroid X:", centroid_x)
print("Centroid Y:", centroid_y)

plt.imshow(frame, cmap='gray')
plt.show()

import numpy as np
import matplotlib.pyplot as plt


frame = np.array([
    1, 2, 3, 4, 3, 2, 1,
    1, 2, 3, 4, 3, 2, 1,
    1, 2, 3, 4, 3, 2, 1,
    1, 2, 4, 4, 3, 2, 1
])

#frame_array = frame.np()  # Convert DataFrame to NumPy array

def find_image_centroid(frame):
    height, width = frame.shape
    sum_x, sum_y, total = 0, 0, 0
    
    for y in range(height):
        for x in range(width):
            pixel = frame[y, x]
            sum_x += x * pixel
            sum_y += y * pixel
            total += pixel

    centroid_x = sum_x / total
    centroid_y = sum_y / total

    return centroid_x, centroid_y

centroid_x, centroid_y = find_image_centroid(frame)

frame2 = frame.reshape(7,4)
print("Centroid X:", centroid_x)
print("Centroid Y:", centroid_y)

plt.imshow(frame2, cmap='gray')
plt.show()
"""
import numpy as np
import matplotlib.pyplot as plt

frame = np.array([
    1, 2, 3, 4, 3, 2, 1,
    1, 2, 3, 4, 3, 2, 1,
    1, 2, 3, 4, 3, 2, 1,
    1, 2, 4, 4, 3, 2, 1
])

def find_image_centroid(frame, height, width):
    sum_x, sum_y, total = 0, 0, 0

    for y in range(height):
        for x in range(width):
            pixel = frame[y * width + x]
            sum_x += x * pixel
            sum_y += y * pixel
            total += pixel

    centroid_x = sum_x / total
    centroid_y = sum_y / total

    return centroid_x, centroid_y

height = 4
width = 7
centroid_x, centroid_y = find_image_centroid(frame, height, width)

frame2 = frame.reshape(height, width)

print("Centroid X:", centroid_x)
print("Centroid Y:", centroid_y)

plt.imshow(frame2, cmap='gray')
plt.show()