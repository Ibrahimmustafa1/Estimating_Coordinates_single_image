import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from geopy import distance
from transformers import AutoImageProcessor, DPTForDepthEstimation
import torch



def estimate_tilt(image_path):
    """
    Estimate the tilt of the image using Hough Line Transform.
    
    Args:
    image_path: str, path to the image file
    """
    
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Use Hough Line Transform to detect lines in the image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # If no lines detected, return 0 tilt
    if lines is None:
        return 0

    angles = []

    # Iterate over the lines and calculate their angles
    for line in lines:
        rho, theta = line[0]
        # Convert from radians to degrees
        angle = np.rad2deg(theta) - 90
        angles.append(angle)

    # Compute the median angle as the estimated tilt
    median_angle = np.median(angles)

    return median_angle



def get_camera_matrix(focal_length_mm, image_width_px, image_height_px, tilt, altitude):
    """
    Compute the camera matrix K, rotation matrix R and translation vector T.
    
    Args:
    focal_length_mm: float, focal length of the camera in mm
    image_width_px: int, image width in pixels
    image_height_px: int, image height in pixels
    
    tilt: float, tilt angle of the camera in degrees
    
    altitude: float, altitude of the camera in meters
    
    Returns:
    K: np.array, camera matrix
    R: np.array, rotation matrix
    T: np.array, translation vector
    
    
    """
    sensor_width_mm = 36.0
    sensor_height_mm = 24.0

    focal_length_x_px = focal_length_mm * (image_width_px / sensor_width_mm)
    focal_length_y_px = focal_length_mm * (image_height_px / sensor_height_mm)

    c_x = image_width_px / 2
    c_y = image_height_px / 2

    K = np.array([
        [focal_length_x_px, 0, c_x],
        [0, focal_length_y_px, c_y],
        [0, 0, 1]
    ])

    tilt_rad = np.deg2rad(tilt)

    R = np.array([
        [1, 0, 0],
        [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
        [0, np.sin(tilt_rad), np.cos(tilt_rad)]
    ])

    # Translation vector (altitude as translation along Z-axis)
    T = np.array([0, 0, -altitude])

    return K, R, T


def estimate_depth_map(image):
    """
    Estimate the depth map of the image using the DPT model.
    
    Args:
    image: PIL.Image, input image
    
    Returns:
    depth_map: np.array, depth map of the image
    """
    image_processor = AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-small-kitti")
    model = DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-small-kitti")

    # prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    depth_map = prediction.squeeze().cpu().numpy()
    return depth_map


def estimate_tile_coordinates(K, R, T, depth_map, image_width_px, image_height_px, latitude_origin, longitude_origin, tile_size):
    """
    Estimate the latitude and longitude coordinates of the center of each tile in the image.
    
    Args:
    K: np.array, camera matrix
    R: np.array, rotation matrix
    T: np.array, translation vector
    depth_map: np.array, depth map of the image
    image_width_px: int, image width in pixels
    image_height_px: int, image height in pixels
    latitude_origin: float, latitude of the camera
    longitude_origin: float, longitude of the camera
    tile_size: int, size of the tile in pixels
    
    Returns:
    coordinates: list of lists, latitude and longitude coordinates of the center of each
                    tile in the image
    """
    num_tiles_x = image_width_px // tile_size
    num_tiles_y = image_height_px // tile_size

    coordinates = []

    for tile_y in range(num_tiles_y):
        for tile_x in range(num_tiles_x):
            center_u = tile_x * tile_size + tile_size // 2
            center_v = tile_y * tile_size + tile_size // 2

            depth = depth_map[center_v, center_u] if depth_map[center_v, center_u] > 0 else 1  # Avoid zero depth

            pixel_coordinates = np.array([center_u, center_v, 1])
            camera_coordinates = np.linalg.inv(K) @ pixel_coordinates * depth

            world_coordinates = R @ camera_coordinates + T

            latitude = (world_coordinates[1] / 111320) + latitude_origin
            longitude = (world_coordinates[0] / (111320 * np.cos(np.radians(latitude_origin)))) + longitude_origin

            coordinates.append([latitude, longitude])

    return coordinates ,num_tiles_x,num_tiles_y



def evaluate_estimated_coordinates(real_coordinates_list,estimated_coordinates):
    """
    Evaluate the estimated coordinates using the ground truth coordinates.

    Args:
    real_coordinates_list: list, ground truth coordinates
    estimated_coordinates: list, estimated coordinates
    
    Returns:
    error: float, error in meters
    
    
    """
    predicted_coords = (estimated_coordinates[0], estimated_coordinates[1])
    true_coords = (real_coordinates_list[0], real_coordinates_list[1])
    error = distance.distance(predicted_coords, true_coords).meters
    return error


def estimation_error(true_coordinates, tile_coordinates, tile_size, min_index):
    """
    Compute the estimation error of the tile coordinates.
    
    Args:
    true_coordinates: list, ground truth coordinates
    tile_coordinates: list, estimated coordinates of the tiles
    tile_size: int, size of the tile in pixels
    min_index: int, index of the tile with the least error
    
    Returns:
    error: float, error in meters
    index: int, index of the tile with the least error
    errors: list, errors of all tiles

        
    """
    estimating_cord_error = [] # list of the diffrence between all tile coordinates and true coordinates
    for cord in tile_coordinates:
        estimating_cord_error.append(evaluate_estimated_coordinates(true_coordinates,cord))
    min_index = np.argmin(estimating_cord_error) # get the index of least error in the list which will be the tile number
    return estimating_cord_error[min_index] , min_index , estimating_cord_error



def create_labeled_image(image_path, geo_coords, tile_size, min_index):
    """
    Create an image with labeled tiles.
    
    Args:
    image_path: str, path to the image file
    geo_coords: list, latitude and longitude coordinates of the center of each tile
    tile_size: int, size of the tile in pixels
    min_index: int, index of the tile with the least error
    
    Returns:
    image: np.array, image with labeled tiles
    """
    
    image = cv2.imread(image_path)
    if image.shape[0] > 1920:
       image = cv2.resize(image, (1920, 1080))
    height, width, _ = image.shape
    count = 1
    camera_center = (width // 2, height)

    for tile_y in range(num_tiles_y):
        for tile_x in range(num_tiles_x):
            top_left_x = tile_x * tile_size
            top_left_y = tile_y * tile_size
            bottom_right_x = top_left_x + tile_size
            bottom_right_y = top_left_y + tile_size
            center_x = top_left_x + tile_size // 2
            center_y = top_left_y + tile_size // 2

            # Draw border for the highlighted tile
            if count == min_index :
                rect_color = (255, 0, 0)  # Highlight color red for the minimum index tile
                cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), rect_color, 2)  # Border for highlighted tile

            count += 1

    return image



image_path = "test_img.jpeg"
image = Image.open(image_path)
# if image width > 1920 resize it as the model will take longer time and my be you face ou of memory error
if image.width > 1920:
  image = image.resize((1920, 1080))
depth_map = estimate_depth_map(image)

# SAVE The depth map using matplotlib 
plt.imshow(depth_map,cmap='inferno')
plt.colorbar()
plt.show()
plt.imsave("depth_map.png", depth_map,cmap='inferno')

focal_length_mm = 5.2 # Focal length in mm
image_width_px = 1920 # Image width in pixels
image_height_px = 1080 # Image height in pixels
tilt = 6.999  # Tilt angle in degrees if you don't have you could use es 
altitude = 301.3  # Altitude in meters
tile_size = 100  # Tile size in pixels

latitude_origin = 30.018633  # Camera latitude
longitude_origin = 31.194741  # Camera longitude

# Get camera matrix
K, R, T = get_camera_matrix(focal_length_mm, image_width_px, image_height_px, tilt, altitude)

# Estimate tile coordinates
tile_coordinates,num_tiles_x,num_tiles_y = estimate_tile_coordinates(K, R, T, depth_map , image_width_px, image_height_px, latitude_origin, longitude_origin, tile_size)

true_coordinates = [30.018715, 31.194654] # True coordinates of the lught pole
least_error , min_index , estimating_cord_error  = estimation_error(true_coordinates, tile_coordinates, tile_size, num_tiles_x) # Get the tile with the least error

labeled_image = create_labeled_image(image_path, tile_coordinates, tile_size,min_index)
# save the image
cv2.imwrite("labeled_image.png", labeled_image)

print("Least error:", least_error)
print("Tile with least error:", min_index)
print("Coordinates of the least Error", tile_coordinates[min_index])

print("Tile numbers with it's coordinates whose error less than or equal 20 meter:")
for i in range(len(estimating_cord_error)):
    if estimating_cord_error[i] <= 20:
        print("Tile number:", i+1,"Error:", estimating_cord_error[i], "Coordinates:", tile_coordinates[i])
