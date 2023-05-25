import os
import math
import cv2
import numpy as np

from heapq import *
from skimage.filters import threshold_otsu, sobel
from skimage.io import imsave, imread
from skimage.color import rgb2gray, gray2rgb
from skimage import img_as_ubyte


def horizontal_projections(sobel_image):
    return np.sum(sobel_image, axis=1)


def find_peak_regions(hpp, divider=3.5):
    threshold = (np.max(hpp) - np.min(hpp)) / divider
    peaks = []
    peaks_index = []
    for i, hppv in enumerate(hpp):
        if hppv < threshold:
            peaks.append([i, hppv])
    return peaks


def get_hpp_walking_regions(peaks_index):
    hpp_clusters = []
    cluster = []
    for index, value in enumerate(peaks_index):
        cluster.append(value)
        
        if index < len(peaks_index) - 1 and peaks_index[index + 1] - value > 1:
            hpp_clusters.append(cluster)
            cluster = []
        
        # get the last cluster
        if index == len(peaks_index) - 1:
            hpp_clusters.append(cluster)
            cluster = []
    
    return hpp_clusters


def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2


def astar(array, start, goal):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    
    heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heappop(oheap)[1]
        
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data
        
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
            
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
    
    return []


def get_binary(img):
    mean = np.mean(img)
    if mean == 0.0 or mean == 1.0:
        return img
    
    thresh = threshold_otsu(img)
    binary = img <= thresh
    binary = binary * 1
    return binary


def path_exists(window_image):
    # very basic check first then proceed to A* check
    if 0 in horizontal_projections(window_image):
        return True
    
    padded_window = np.zeros((window_image.shape[0], 1))
    world_map = np.hstack((padded_window, np.hstack((window_image, padded_window))))
    path = np.array(
        astar(world_map, (int(world_map.shape[0] / 2), 0), (int(world_map.shape[0] / 2), world_map.shape[1])))
    if len(path) > 0:
        return True
    
    return False


def get_road_block_regions(nmap):
    road_blocks = []
    needtobreak = False
    
    for col in range(nmap.shape[1]):
        start = col
        end = col + 20
        if end > nmap.shape[1] - 1:
            end = nmap.shape[1] - 1
            needtobreak = True
        
        if path_exists(nmap[:, start:end]) == False:
            road_blocks.append(col)
        
        if needtobreak == True:
            break
    
    return road_blocks


def group_the_road_blocks(road_blocks):
    # group the road blocks
    road_blocks_cluster_groups = []
    road_blocks_cluster = []
    size = len(road_blocks)
    for index, value in enumerate(road_blocks):
        road_blocks_cluster.append(value)
        if index < size - 1 and (road_blocks[index + 1] - road_blocks[index]) > 1:
            road_blocks_cluster_groups.append(
                [road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster) - 1]])
            road_blocks_cluster = []
        
        if index == size - 1 and len(road_blocks_cluster) > 0:
            road_blocks_cluster_groups.append(
                [road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster) - 1]])
            road_blocks_cluster = []
    
    return road_blocks_cluster_groups


def extract_line_from_image(image, lower_line, upper_line):
    lower_boundary = np.min(lower_line[:, 0])
    upper_boundary = np.min(upper_line[:, 0])
    img_copy = np.copy(image)
    r, c = img_copy.shape
    for index in range(c - 1):
        img_copy[0:lower_line[-index, 0], index] = img_copy[int((lower_line[index, 0] + upper_line[index, 0]) / 2), 1]
        img_copy[upper_line[-index, 0]:r, index] = img_copy[int((lower_line[index, 0] + upper_line[index, 0]) / 2), 1]
    
    return img_copy[lower_boundary:upper_boundary, :]


def lines_segmentation(path):
    img = rgb2gray(cv2.imread(path))
    sobel_image = sobel(img)
    hpp = horizontal_projections(sobel_image)

    peaks = find_peak_regions(hpp)
    peaks_index = np.array(peaks)[:, 0].astype(int)

    hpp_clusters = get_hpp_walking_regions(peaks_index)
    binary_image = get_binary(img)

    for cluster_of_interest in hpp_clusters:
        nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest) - 1], :]
        road_blocks = get_road_block_regions(nmap)
        road_blocks_cluster_groups = group_the_road_blocks(road_blocks)
        for index, road_blocks in enumerate(road_blocks_cluster_groups):
            window_image = nmap[:, road_blocks[0]: road_blocks[1] + 10]
            binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest) - 1], :][:,
            road_blocks[0]: road_blocks[1] + 10][int(window_image.shape[0] / 2), :] *= 0

    line_segments = []
    for i, cluster_of_interest in enumerate(hpp_clusters):
        nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest) - 1], :]
        path = np.array(astar(nmap, (int(nmap.shape[0] / 2), 0), (int(nmap.shape[0] / 2), nmap.shape[1] - 1)))
        offset_from_top = cluster_of_interest[0]
        path[:, 0] += offset_from_top
        line_segments.append(path)

    line_images = []
    line_count = len(line_segments)
    for line_index in range(line_count - 1):
        line_image = extract_line_from_image(img, line_segments[line_index], line_segments[line_index + 1])
        line_images.append(line_image)
    
    if not os.path.isdir('./lines_dataset/'):
        os.mkdir('./lines_dataset/')
    for i in range(len(line_images)):
        good_img = line_images[i]
        imsave(f'./lines_dataset/{i}_line.jpg', img_as_ubyte(gray2rgb(good_img)))

    return line_images


def percent_of_white_pixels(img, thresh_index):
    h, w, _ = img.shape
    image = img
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,thresh_index,255,cv2.THRESH_BINARY_INV)
    white_pixels = 0
    h, w = thresh.shape
    for i in range(h):
        for j in range(w):
            if thresh[i][j] == 255:
                white_pixels += 1
    all_pixels = h * w

    return round(white_pixels / all_pixels, 2)


def compare_thresh_indexes(img):
    thresh_indexes = [100 + i for i in range(18, 22)]
    dict_with_norm_white_percent = {}
    dict_with_big_white_percent = {}
    best_percent = 100
    best_index = 0
    best_val = 100
    
    for index in thresh_indexes:
        percent = percent_of_white_pixels(img, index)
        if 0.01 <= percent <= 0.15:
            dict_with_norm_white_percent[index] = percent
        else:
            dict_with_big_white_percent[index] = percent
    
    if len(dict_with_norm_white_percent) == 0:
        for index in dict_with_big_white_percent.keys():
            if dict_with_big_white_percent[index] <= best_percent:
                best_percent = dict_with_big_white_percent[index]
                best_index = index
    
    else:
        for index in dict_with_norm_white_percent.keys():
            new_value = math.fabs(dict_with_norm_white_percent[index] - 0.07)
            if new_value <= best_val:
                best_val = new_value
                best_percent = dict_with_norm_white_percent[index]
                best_index = index
    return index


class RECT:
    """Class that helps to  find where one contours are inside others"""
    def __init__(self, x, y, h, w):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.area = h*w


def get_rectangles_from_contours(contours, h_img, w_img):
    """Get all rectangles from contours, delete internal contours"""
    rectangles = []
    for i, ctr in enumerate(contours):
        x, y, w, h = cv2.boundingRect(ctr)
        if w >= w_img / 100 and h >= h_img / 100:
            r = RECT(x, y, h, w)
            rectangles += [r]
    resulted_rectangles = []
    tmp = -1
    for i in range(len(rectangles)):
        R1 = rectangles[i]
        if R1.area < 400:
            continue
        resulted_rectangles += [R1]
    return resulted_rectangles


def contours_extraction(img_path, thresh_index, path_to_save):
    img = cv2.imread(img_path)
    h_img, w_img, _ = img.shape
    image = img
    words = []
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, thresh_index, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((10, 13), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    resulted_rectangles = get_rectangles_from_contours(sorted_ctrs, h_img, w_img)
    
    for rect in resulted_rectangles:
        cv2.rectangle(thresh, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (255, 255, 255), 2)
    
    img = cv2.imread(img_path)
    image_to_cut = img
    if not os.path.isdir(path_to_save):
        os.mkdir(path_to_save)

    for i, rect in enumerate(resulted_rectangles):
        roi = image_to_cut.copy()[rect.y:rect.y + rect.h, rect.x:rect.x + rect.w]
        num_of_file = len(os.listdir(path_to_save))
        cv2.imwrite(path_to_save + f'{num_of_file}.jpg', roi)
        words.append(roi)
    
    return words


def words_segmentation(img_path, path_to_save):
    img = cv2.imread(img_path)
    h_img, w_img, _ = img.shape
    res_index = compare_thresh_indexes(img)

    contours = contours_extraction(img_path, res_index, path_to_save)


# imgs = lines_segmentation('../../data/24.jpg')
# words_segmentation('./lines_dataset/0_line.jpg', './words_dataset/')
