# Important Note 
## You need To download Apple Depth pro Model From Github
### Please Clone the repo and it's instruction repo link : https://github.com/apple/ml-depth-pro

##### Project Structure
Depth Estimation: The code uses a depth estimation model to generate a depth map for an image, which tells how far each pixel is from the camera.

FoV Calculation: The field of view is calculated based on the camera's focal length and sensor size.

Tile-based Coordinate Estimation: The image is divided into tiles, and each tile's geographical coordinates are estimated using depth information and the camera's orientation (bearing).

Error Evaluation: The estimated coordinates are compared against ground-truth coordinates, and the tile with the least error is highlighted.

Visualization: The final image is displayed with tiles labeled, and the tile with the least error is highlighted in red.
##### Usage
Depth Map Estimation: The function estimate_depth_map(image_path) loads the image, applies the depth estimation model, and returns a depth map.

Field of View Calculation: The function calculate_fov(focal_length_mm, sensor_width_mm, sensor_height_mm) calculates the horizontal and vertical FoV from the camera specifications.

Geographical Coordinates Estimation: The function estimate_tile_coordinates() computes the coordinates (latitude and longitude) for each tile in the image, based on the depth and the camera's field of view and bearing.

Error Evaluation: The function estimation_error() compares the estimated coordinates to the true coordinates and finds the tile with the least error.

Labeled Image Creation: The function create_labeled_image() draws labeled tiles on the image and highlights the tile with the least error.
