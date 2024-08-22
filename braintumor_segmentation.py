#openCV
import os
import cv2
import numpy as np
import copy

_ALPHA = 300
_BETA = 2
_W_LINE = 80
_W_EDGE = 80
_NUM_NEIGHBORS = 9

# Define the 8 possible movements in a 2D grid (neighbors)
neighbors = np.array([[i, j] for i in range(-1, 2) for j in range(-1, 2)])

def internalEnergy(snake):
    iEnergy = 0
    snakeLength = len(snake)
    for index in range(snakeLength-1, -1, -1):
        nextPoint = (index + 1) % snakeLength
        currentPoint = index % snakeLength
        previousPoint = (index - 1) % snakeLength
        iEnergy += (_ALPHA * (np.linalg.norm(snake[nextPoint] - snake[currentPoint]) ** 2)) \
                   + (_BETA * (np.linalg.norm(snake[nextPoint] - 2 * snake[currentPoint] + snake[previousPoint]) ** 2))
    return iEnergy

def totalEnergy(gradient, image, snake):
    iEnergy = internalEnergy(snake)
    eEnergy = externalEnergy(gradient, image, snake)
    return iEnergy + eEnergy

def externalEnergy(gradient, image, snake):
    sum_pixel = 0
    snaxels_Len = len(snake)
    for index in range(snaxels_Len - 1):
        point = snake[index]
        sum_pixel += image[point[1], point[0]]
    pixel = 255 * sum_pixel

    eEnergy = _W_LINE * pixel - _W_EDGE * imageGradient(gradient, snake)
    return eEnergy

def imageGradient(gradient, snake):
    sum_gradient = 0
    snaxels_Len = len(snake)
    for index in range(snaxels_Len - 1):
        point = snake[index]
        sum_gradient += gradient[point[1], point[0]]
    return sum_gradient

def isPointInsideImage(image, point):
    return np.all(point < np.shape(image)) and np.all(point >= 0)

def _pointsOnCircle(center, radius, num_points=12):
    points = np.zeros((num_points, 2), dtype=np.int32)
    for i in range(num_points):
        theta = float(i) / num_points * (2 * np.pi)
        x = int(center[0] + radius * np.cos(theta))
        y = int(center[1] + radius * np.sin(theta))
        points[i] = [x, y]
    return points

def basicImageGradient(image):
    s_mask = 17
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=s_mask)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=s_mask)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)
    return gradient

def display(image, snake=None):
    display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if snake is not None:
        for s in snake:
            cv2.circle(display_img, tuple(s), 1, (0, 255, 0), -1)
    cv2.imshow("Snake", display_img)
    cv2.waitKey(1)

def activeContour(image_file, center, radius):
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    snake = _pointsOnCircle(center, radius, 20)
    gradient = basicImageGradient(image)

    for _ in range(100):
        for index, point in enumerate(snake):
            min_energy = float("inf")
            best_movement = None
            for movement in neighbors:
                next_point = point + movement
                if not isPointInsideImage(image, next_point):
                    continue

                snake_copy = copy.deepcopy(snake)
                snake_copy[index] = next_point

                totalEnergyNext = totalEnergy(gradient, image, snake_copy)
                if totalEnergyNext < min_energy:
                    min_energy = totalEnergyNext
                    best_movement = movement

            snake[index] += best_movement
        display(image, snake)

    # Save the final segmented image
    display(image, snake)
    cv2.imwrite(os.path.splitext(image_file)[0] + "-segmented.png", image)

def _test():
    activeContour(r"C:\Users\abc\Desktop\BrainTumor_segmentation\brainTumor.png", (98, 152), 30)

if __name__ == '__main__':
    _test()
