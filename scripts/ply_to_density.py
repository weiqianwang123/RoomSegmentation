import numpy as np
import os
import cv2
from plyfile import PlyData, PlyElement
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import argparse
def read_scene_pc(file_path):
    with open(file_path, 'rb') as f:
        plydata = PlyData.read(f)
        dtype = plydata['vertex'].data.dtype
    print('dtype of file {}: {}'.format(file_path, dtype))

    points_data = np.array(plydata['vertex'].data.tolist())
    return points_data


def calculate_covered_area(point_cloud, min_coords, max_coords, width=256, height=256):
    # Determine the bounding_box size
    bounding_box_size = max_coords - min_coords

    # Create a grid to map the point cloud onto the XZ plane
    grid = np.zeros((height, width), dtype=int)
    
    # Map the points to the grid using X and Z coordinates
    xz_coords = point_cloud[:, [0, 2]] 
    for point in xz_coords:
        x, z = (point - min_coords[[0, 2]]) / bounding_box_size[[0, 2]] * [width, height]
        grid_x = int(np.floor(x))
        grid_z = int(np.floor(z))
        if 0 <= grid_x < width and 0 <= grid_z < height:
            grid[grid_z, grid_x] = 1
    
    # Calculate the covered area
    covered_area = np.sum(grid > 0)
    return covered_area

def get_wall_mask(point_cloud, min_coords, max_coords, width=256, height=256):
    # Step 1: Determine the bounding box size
    bounding_box_size = max_coords - min_coords
    
    # Step 2: Segment the point cloud into layers along the Y axis
    y_coords = point_cloud[:, 1] 
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)
    layer_thickness = (max_y - min_y) / 8
    
    layers = []
    for i in range(8):
        lower_bound = min_y + i * layer_thickness
        upper_bound = lower_bound + layer_thickness
        layer = point_cloud[(y_coords >= lower_bound) & (y_coords < upper_bound)]
        layers.append(layer)
    
    # Step 3: Project each layer onto a 256x256 grid map (XZ plane)
    layer_maps = []
    for layer in layers:
        layer_xz = layer[:, [0, 2]]  
        grid = np.zeros((height, width), dtype=int)
        
        for point in layer_xz:
            x, z = (point - min_coords[[0, 2]]) / bounding_box_size[[0, 2]] * [width, height]
            grid_x = int(np.floor(x))
            grid_z = int(np.floor(z))
            if 0 <= grid_x < width and 0 <= grid_z < height:
                grid[grid_z, grid_x] = 1
        
        layer_maps.append(grid)

    # # Visualization of individual layers
    # fig, axs = plt.subplots(1, len(layer_maps), figsize=(15, 5))
    # for i, layer_map in enumerate(layer_maps):
    #     axs[i].imshow(layer_map, cmap='gray')
    #     axs[i].set_title(f'Layer {i+1}')
    #     axs[i].axis('off')

    # plt.show()
    
    # Step 4: Eliminate layers based on occupancy criteria
    total_area = calculate_covered_area(point_cloud, min_coords, max_coords, width, height)
    valid_layers = []
    # print(total_area)
    for grid in layer_maps:
        occupancy = np.sum(grid > 0)
        # print("occupancy")
        # print(occupancy)
        if 1 / 15 * total_area <= occupancy <= 1 / 5 * total_area:
            valid_layers.append(grid)
    
    # Step 5: Merge the remaining layers into one final map
    final_map = np.zeros((height, width), dtype=int)
    l = len(valid_layers)
    if l > 0:
        for grid in valid_layers:
            final_map += grid
        
        final_map = (final_map > 3 / 4 * l).astype(int)
    
    # Step 6: Dilate the final map in x and z directions only
    structuring_element = np.array([[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]])
    
    dilated_map = binary_dilation(final_map, structure=structuring_element).astype(int)
    
    # # Visualization of the final map
    # plt.imshow(dilated_map, cmap='gray')
    # plt.title('Dilated Final Map')
    # plt.axis('off')
    # plt.show()

    return dilated_map
def generate_density(point_cloud, width=256, height=256):

    height_range=(0.0, 3.0)
    angle=0
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Rotation matrix around the y-axis
    rotation_matrix = np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    
    # Filter the point cloud to only include points within the specified height range
    height_filtered_point_cloud = point_cloud[(point_cloud[:, 1] >= height_range[0]) & 
                                              (point_cloud[:, 1] <= height_range[1])]
    
    if height_filtered_point_cloud.shape[0] == 0:
        return np.zeros((height, width), dtype=np.float32), {}
    
    # Rotate the point cloud
    rotated_point_cloud = np.dot(height_filtered_point_cloud, rotation_matrix.T)
    
    # Invert the y-up coordinate system
    ps = rotated_point_cloud.copy()
    ps[:, 1] *= -1  # Invert the y-axis
    ps[:, 2] *= -1  # Invert the z-axis
    
    image_res = np.array((width, height))
    
    # Calculate the min and max coordinates in the filtered point cloud
    max_coords = np.max(ps, axis=0)
    min_coords = np.min(ps, axis=0)
    max_m_min = max_coords - min_coords
    
    # Apply padding to avoid edge cases
    max_coords = max_coords + 0.1 * max_m_min
    min_coords = min_coords - 0.1 * max_m_min
    
    wall_mask = get_wall_mask(ps, min_coords, max_coords, width, height)
    
    # Store normalization data for later use
    normalization_dict = {}
    normalization_dict["min_coords"] = min_coords
    normalization_dict["max_coords"] = max_coords
    normalization_dict["image_res"] = image_res
    
    # Normalize the coordinates to fit within the image resolution
    coordinates = np.round(
        (ps[:, [0, 2]] - min_coords[None, [0, 2]]) / (max_coords[None, [0, 2]] - min_coords[None, [0, 2]]) * image_res[None]
    )
    coordinates = np.minimum(np.maximum(coordinates, np.zeros_like(image_res)),
                             image_res - 1)
    
    # Create the density map
    density = np.zeros((height, width), dtype=np.float32)
    
    # Count occurrences of each coordinate and update the density map
    unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)
    unique_coordinates = unique_coordinates.astype(np.int32)
    
    density[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts
    density += (wall_mask * np.max(density) // 10).astype(int)
    # Normalize the density map to a range of [0, 1]
    density = density / np.max(density)
    
    return density, normalization_dict

def save_density_map(density, output_path):
    density_scaled = (density * 255).astype(np.uint8)
    cv2.imwrite(output_path, density_scaled)


# print(f"Density map saved to {output_path}")
def main(ply_path, output_path):
    points = read_scene_pc(ply_path)
    xyz = points[:, :3]

    density, normalization_dict = generate_density(xyz, width=256, height=256)
    
    save_density_map(density, output_path)

    print(f"Density map saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process point cloud and save density map.')
    parser.add_argument('--ply_path', type=str,default ="Data/umich_ggbl.ply",required=True, help='Path to the input PLY file.')
    parser.add_argument('--output_path', type=str,default = "Results/demo/density.png", required=True, help='Path to save the output density map.')

    args = parser.parse_args()

    main(args.ply_path, args.output_path)