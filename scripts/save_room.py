import os
import json
import numpy as np
import open3d as o3d
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
import matplotlib.pyplot as plt
import copy
import argparse

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data["room_polys"]

def reverse_projection(polygon, min_coords, max_coords, resolution=(256, 256), angle=0):
    """
    Reverses the projection of a 2D polygon back to the original 3D coordinates,
    including the handling of padding applied during the projection.
    
    Args:
        polygon (list): List of points representing the 2D room polygon.
        min_coords (tuple): Minimum x, z coordinates in the original 3D point cloud with padding.
        max_coords (tuple): Maximum x, z coordinates in the original 3D point cloud with padding.
        resolution (tuple): Resolution of the 2D plane.
        angle (float): The angle used for rotation during the initial projection.

    Returns:
        np.ndarray: The 2D polygon corresponding to the original 3D coordinates.
    """
    # Extract the min and max coordinates in x and z (with padding)
    min_x, min_z = min_coords
    max_x, max_z = max_coords

    # Remove the padding to get the original min and max coordinates
    original_min_x = min_x - 0.1 * (max_x - min_x)
    original_max_x = max_x + 0.1 * (max_x - min_x)
    original_min_z = min_z - 0.1 * (max_z - min_z)
    original_max_z = max_z + 0.1 * (max_z - min_z)

    # Calculate scale factors based on the original (unpadded) coordinates
    scale_x = (original_max_x - original_min_x) / resolution[0]
    scale_z = (original_max_z - original_min_z) / resolution[1]

    # Reverse normalization to obtain original x, z coordinates
    reversed_polygon = np.zeros((len(polygon), 2), dtype=np.float32)
    reversed_polygon[:, 0] = polygon[:, 0] * scale_x + original_min_x
    reversed_polygon[:, 1] = polygon[:, 1] * scale_z + original_min_z

    # Reverse the axis inversion applied during projection
    reversed_polygon[:, 1] *= -1  # Invert the z-axis back

    # Reverse the rotation applied during projection
    angle_rad = -np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), np.sin(angle_rad)],
        [-np.sin(angle_rad), np.cos(angle_rad)]
    ])
    reversed_polygon = np.dot(reversed_polygon, rotation_matrix.T)
    
    return reversed_polygon

def crop_room_from_mesh(mesh, polygon, min_coords, max_coords, resolution=(256, 256)):
    """
    Crop the points and mesh that are within the given polygon from the original mesh.
    
    Args:
        mesh (open3d.geometry.TriangleMesh): The mesh data.
        polygon (list): List of points representing the room polygon.
        min_coords (tuple): Minimum x, z coordinates in the original 3D point cloud.
        max_coords (tuple): Maximum x, z coordinates in the original 3D point cloud.
        resolution (tuple): Resolution of the 2D plane.

    Returns:
        open3d.geometry.TriangleMesh: The cropped mesh.
        np.ndarray: Mask array of the cropped points in the original point cloud.
        float: Minimum height of the points within the room.
        float: Maximum height of the points within the room.
    """
    # Reverse the projection of the polygon to match the original 3D point cloud coordinates
    polygon_2d = reverse_projection(np.array(polygon), min_coords, max_coords, resolution,0)
    
    # Convert polygon to a shapely object
    room_polygon = ShapelyPolygon(polygon_2d)

    # Filter points inside the polygon
    points = np.asarray(mesh.vertices)
    mask = np.array([room_polygon.contains(ShapelyPoint(p[0], p[2])) for p in points])  # Only consider x, z

    # Apply mask to vertices and colors
    temp_mesh = copy.deepcopy(mesh)
    temp_mesh.remove_vertices_by_mask(~mask)
    
    # Get the min and max height (y-values)
    min_height = np.min(points[mask][:, 1])
    max_height = np.max(points[mask][:, 1])

    return temp_mesh, mask, min_height, max_height

def save_room(mesh, room_id, min_height, max_height, output_dir):
    # Save the cropped mesh
    room_ply_path = os.path.join(output_dir, f"{room_id}.ply")
    o3d.io.write_triangle_mesh(room_ply_path, mesh)

    # Save JSON file
    room_info = {
        "id": room_id,
        "min_height": min_height,
        "max_height": max_height
    }
    room_json_path = os.path.join(output_dir, f"{room_id}.json")
    with open(room_json_path, 'w') as f:
        json.dump(room_info, f, indent=4)

def process_ply_file(ply_file_path, json_file_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the mesh from the PLY file
    mesh = o3d.io.read_triangle_mesh(ply_file_path)
    print("read done")
    points = np.asarray(mesh.vertices)
    
    # Create a copy of the points to apply inversion only for min/max calculations
    transformed_points = points.copy()
    
    # Invert the y-axis and z-axis on the copy, similar to the original projection process
    transformed_points[:, 1] *= -1  # Invert the y-axis
    transformed_points[:, 2] *= -1  # Invert the z-axis

    # Get the min and max bounds of the transformed point cloud (consider only x, z)
    min_coords = np.min(transformed_points[:, [0, 2]], axis=0)
    max_coords = np.max(transformed_points[:, [0, 2]], axis=0)
    
    # Read room polygons from the JSON file
    room_polys = read_json(json_file_path)

    # Initialize an array for room colors
    # original_colors = np.asarray(mesh.vertex_colors)
    num_rooms = len(room_polys)
    color_map = plt.get_cmap("tab20")  # Get a colormap for visualization
    
    # Process each room polygon
    for i, polygon in enumerate(room_polys):
        cropped_mesh, mask, min_height, max_height = crop_room_from_mesh(mesh, polygon, min_coords, max_coords)
        
        # Save the cropped room mesh
        save_room(cropped_mesh, i, min_height, max_height, output_dir)
        
        # Assign a unique color to the room points in the original mesh
        # room_color = color_map(i / num_rooms)[:3]  # Get RGB values
        # original_colors[mask] = room_color
        
        print(f"Saved room {i} with heights {min_height} - {max_height}")

    # Update the original mesh with the new colors
    # mesh.vertex_colors = o3d.utility.Vector3dVector(original_colors)
    
    # # Save the colored mesh
    # colored_ply_path = os.path.join(output_dir, "colored_rooms.ply")
    # o3d.io.write_triangle_mesh(colored_ply_path, mesh)
    
    # Visualize the colored mesh
    # o3d.visualization.draw_geometries([mesh])

# Example usage
# ply_file_path = '/home/qianwei/RoomFormer-main/rotated.ply'
# json_file_path = '/home/qianwei/RoomFormer-main/my_method/data/room_polygons.json'
# output_dir = '/home/qianwei/RoomFormer-main/my_method/data/rooms'

# process_ply_file(ply_file_path, json_file_path, output_dir)


def main(ply_file_path, json_file_path, output_dir):
    process_ply_file(ply_file_path, json_file_path, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a PLY file and generate room polygons.')
    parser.add_argument('--ply_file_path',type=str, default = "Data/umich_ggbl.ply", help='Path to the input PLY file.')
    parser.add_argument('--json_file_path', type=str, default = "Results/demo/room_polygons.json",help='Path to the JSON file with room polygons.')
    parser.add_argument('--output_dir', type=str, default = "Results/demo/rooms",help='Directory to save the output rooms.')

    args = parser.parse_args()
    main(args.ply_file_path, args.json_file_path, args.output_dir)
