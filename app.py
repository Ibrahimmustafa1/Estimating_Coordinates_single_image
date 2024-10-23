import depth_pro 
import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2
import numpy as np
from geopy import distance
from uuid import uuid4
import math


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, transform = depth_pro.create_model_and_transforms()
model = model.to(device)



def estimate_depth_map(image_path):
    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image)
    image = image.to(device)
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  
    focallength_px = prediction["focallength_px"] 
    depth = depth.squeeze().cpu().numpy()
    return depth



# Function to calculate field of view (FoV)
def calculate_fov(focal_length_mm, sensor_width_mm, sensor_height_mm):
    """
    Calculate the horizontal and vertical field of view (FoV) based on focal length and sensor size.
    """
    hfov = 2 * np.arctan(sensor_width_mm / (2 * focal_length_mm)) * (180 / np.pi)
    vfov = 2 * np.arctan(sensor_height_mm / (2 * focal_length_mm)) * (180 / np.pi)
    return hfov, vfov


def estimate_tile_coordinates( depth_map, image_width_px, image_height_px, latitude_origin, longitude_origin, tile_size, hfov, vfov, bearing):
    """
    Estimate the latitude and longitude coordinates of the center of each tile in the image, incorporating FoV and bearing.
    Adjusts based on the tile's position (left/right and up/down) relative to the image center.
    """
    
    # Number of tiles along the X and Y axes
    num_tiles_x = image_width_px // tile_size
    num_tiles_y = image_height_px // tile_size

    # Calculate the angle per pixel in both horizontal and vertical directions
    angle_per_pixel_horizontal = hfov / image_width_px
    angle_per_pixel_vertical = vfov / image_height_px
    
    coordinates = []
    
    for tile_y in range(num_tiles_y):
        for tile_x in range(num_tiles_x):
            center_u = tile_x * tile_size + tile_size // 2
            center_v = tile_y * tile_size + tile_size // 2

            # Distance from the depth map for this tile
            distance = depth_map[center_v, center_u]
            
            # Calculate the angle offsets from the center of the image (left/right and up/down)
            angle_offset_horizontal = ((center_u - (image_width_px / 2)) * angle_per_pixel_horizontal)
            angle_offset_vertical = ((center_v - (image_height_px / 2)) * angle_per_pixel_vertical)
            
            # Adjust the bearing angle based on the horizontal offset
            total_bearing = bearing + angle_offset_horizontal
            
            # Adjust the distance based on the vertical angle
            adjusted_distance = distance  + (math.cos(math.radians(angle_offset_vertical)))
            
            # Earth radius in meters
            R = 6371000

            # Convert latitude and longitude from degrees to radians
            lat_rad = math.radians(latitude_origin)
            lon_rad = math.radians(longitude_origin)
            bearing_rad = math.radians(total_bearing)

            # Calculate new latitude based on the adjusted distance and bearing
            new_lat = math.asin(math.sin(lat_rad) * math.cos(adjusted_distance / R) +
                                math.cos(lat_rad) * math.sin(adjusted_distance / R) * math.cos(bearing_rad))

            # Calculate new longitude
            new_lon = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(adjusted_distance / R) * math.cos(lat_rad),
                                           math.cos(adjusted_distance / R) - math.sin(lat_rad) * math.sin(new_lat))

            # Convert new latitude and longitude from radians to degrees
            new_latitude = math.degrees(new_lat)
            new_longitude = math.degrees(new_lon)
            
            coordinates.append([new_latitude, new_longitude])
    
    return coordinates, num_tiles_x, num_tiles_y

# Function to calculate distance error between estimated and true coordinates
def evaluate_estimated_coordinates(real_coordinates_list, estimated_coordinates):
    """
    Evaluate the estimated coordinates by comparing them with the true coordinates.

    Args:
    real_coordinates_list: list, ground truth coordinates (latitude, longitude)
    estimated_coordinates: list, estimated coordinates (latitude, longitude)

    Returns:
    float: error distance in meters
    """
    predicted_coords = (estimated_coordinates[0], estimated_coordinates[1])
    true_coords = (real_coordinates_list[0], real_coordinates_list[1])

    # Compute the distance between the true and estimated coordinates
    error = distance.distance(predicted_coords, true_coords).meters
    return error


# Function to compute error for each tile and find the one with the least error
def estimation_error(true_coordinates, tile_coordinates, tile_size):
    """
    Compute the error for all tiles and return the tile with the least error.

    Args:
    true_coordinates: list, ground truth coordinates
    tile_coordinates: list, estimated coordinates of each tile
    tile_size: int, size of the tile in pixels
    min_index: int, index of the tile with the least error

    Returns:
    tuple: least error, index of the tile with least error, and errors for all tiles
    """

    estimating_cord_error = []

    # Calculate error for each tile's coordinates
    for cord in tile_coordinates:
        estimating_cord_error.append(evaluate_estimated_coordinates(true_coordinates, cord))

    # Find the index of the tile with the least error
    min_index = np.argmin(estimating_cord_error)

    return estimating_cord_error[min_index], min_index, estimating_cord_error


# Function to create an image with labeled tiles and highlight the tile with the least error
def create_labeled_image(image_path, geo_coords, tile_size, low_tiles_error, num_tiles_x, num_tiles_y):
    """
    Create an image with labeled tiles and highlight the tile with the least error.

    Args:
    image_path: str, path to the image file
    geo_coords: list, latitude and longitude of the center of each tile
    tile_size: int, size of the tile in pixels
    low_tiles_error: list, indices of the tiles with the least error
    num_tiles_x: int, number of tiles in the x-axis
    num_tiles_y: int, number of tiles in the y-axis

    Returns:
    np.array: image with labeled tiles
    """

    # Load the image and resize if necessary
    image = cv2.imread(image_path)
    if image.shape[0] > 1920:
        image = cv2.resize(image, (1920, 1080))

    height, width, _ = image.shape
    count = 0  # Tile counter

    # Loop over all tiles and draw a rectangle for each
    for tile_y in range(num_tiles_y):
        for tile_x in range(num_tiles_x):
            # Top-left and bottom-right coordinates of the tile
            top_left_x = tile_x * tile_size
            top_left_y = tile_y * tile_size
            bottom_right_x = top_left_x + tile_size
            bottom_right_y = top_left_y + tile_size

            # Calculate the center of the tile for text placement
            center_x = top_left_x + tile_size // 2
            center_y = top_left_y + tile_size // 2

            # Draw a red border around the tile with the least error
            rect_color = (255, 0, 0)  # Red color for the highlighted tile
            cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), rect_color, 1)

            # Place the tile number in the center of the tile
            cv2.putText(image, f"{count}", (center_x - 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 1)
            count += 1

    return image

# Main script
def main(image_path,focal_length_mm,altitude,tile_size,latitude_origin,longitude_origin,bearing,sensor_width_mm,sensor_height_mm,true_coordinates):
    image = Image.open(image_path)
    depth_map = estimate_depth_map(image_path)
    image_width_px = image.width   
    image_height_px = image.height  


    # Calculate field of view (FoV)
    hfov, vfov = calculate_fov(focal_length_mm, sensor_width_mm, sensor_height_mm)

    # Estimate tile coordinates (latitude, longitude) using FoV and bearing
    tile_coordinates, num_tiles_x, num_tiles_y = estimate_tile_coordinates(
        depth_map, image_width_px, image_height_px, latitude_origin, longitude_origin, tile_size, hfov, vfov, bearing
    )

    # Calculate the error for each tile and find the tile with the least error
    least_error, min_index, estimating_cord_error = estimation_error(true_coordinates, tile_coordinates, tile_size)

    # Output the results
    print("Least error:", least_error)
    print("Tile with least error:", min_index)
    print("Coordinates of the least Error", tile_coordinates[min_index])

    # Print all tiles with an error less than or equal to 20 meters
    print("Tiles with error <= 20 meters:")

    low_tiles_error = []
    for i in range(len(estimating_cord_error)):
        if estimating_cord_error[i] <= 10:
            low_tiles_error.append(i)
            print(f"Tile number: {i}, Error: {estimating_cord_error[i]}, Coordinates: {tile_coordinates[i]}")

    labeled_image = create_labeled_image(image_path, tile_coordinates, tile_size, low_tiles_error,num_tiles_x , num_tiles_y)
    plt.imshow(labeled_image)
    plt.show()
    uu = str(uuid4())
    im = image_path.split("/")[-1]
    cv2.imwrite(f"/kaggle/working/labeled{im}", labeled_image)
    
image_path = "/kaggle/input/frames-data/22371809.jpg"
focal_length_mm = 4
altitude = 1270.3753662109375        # Altitude in meters
tile_size = 50           # Size of each tile in pixels
# Camera's geographical coordinates (origin point)
latitude_origin = 17.568050291667  # Latitude of the camera
longitude_origin = 44.257057704972
bearing = 111 
sensor_width_mm = 22.678644
sensor_height_mm = 19.84375
# Ground truth coordinates of the object
true_coordinates = [17.568055, 44.257227] # Example of a real object
main(image_path,focal_length_mm,altitude,tile_size,latitude_origin,longitude_origin,bearing,sensor_width_mm,sensor_height_mm,true_coordinates)
