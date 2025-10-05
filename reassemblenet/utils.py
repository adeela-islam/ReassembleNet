import io
import cv2
import os
import random
import torch
import cairosvg
import imageio
import webcolors
import numpy as np
import torch as th
import drawsvg as drawsvg
import PIL.Image as Image
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.geometry import Point, LineString



dst_thresh = 0.1


def generate_id_color(num_ids):
    id_color = {}
    for i in range(1, num_ids + 1):
        # Generate a random color in hexadecimal format
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        id_color[i] = color
    return id_color


def calculate_f1_score(precision, recall):
    """
    Calculate the F1 score given precision and recall.

    Parameters:
        precision (float): Precision value (between 0 and 1).
        recall (float): Recall value (between 0 and 1).

    Returns:
        float: F1 score, or 0 if precision and recall are both zero.
    """
    if precision == 0 and recall == 0:
        return 0  # Avoid division by zero
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def calculate_rotation_angles_and_rmse(pred_rotation, gt_rotation):

    # Extracting cos and -sin components
    pred_cos = pred_rotation[..., 0]  
    pred_neg_sin = pred_rotation[..., 1]  

    gt_cos = gt_rotation[..., 0] 
    gt_neg_sin = gt_rotation[..., 1] 

    # Calculate rotation angles in radians
    pred_rotation_angle_rad = torch.atan2(pred_neg_sin, pred_cos)  
    gt_rotation_angle_rad = torch.atan2(gt_neg_sin, gt_cos) 

    # Convert radians to degrees
    pred_rotation_angle_deg = torch.rad2deg(pred_rotation_angle_rad) 
    gt_rotation_angle_deg = torch.rad2deg(gt_rotation_angle_rad) 

    # Calculate RMSE
    squared_diff = (pred_rotation_angle_deg - gt_rotation_angle_deg) ** 2  
    mean_squared_error = torch.mean(squared_diff, dim=2)  
    rmse = torch.sqrt(mean_squared_error)

    return rmse.mean().item()




def calculate_rmse(ground_truth, predicted):
    """
    Calculate the Root Mean Square Error (RMSE) for translation tensors.

    Args:
        ground_truth (torch.Tensor): Ground truth translation tensor of shape 
        predicted (torch.Tensor): Predicted translation tensor of shape 

    Returns:
        float: The RMSE value.
    """
    # Ensure the tensors are of the same shape
    if ground_truth.shape != predicted.shape:
        raise ValueError("Ground truth and predicted tensors must have the same shape.")

    differences = predicted - ground_truth

    squared_differences = differences ** 2
    
    mean_squared_error = torch.mean(squared_differences)
    
    rmse = torch.sqrt(mean_squared_error)

    return rmse.item()  



def polygon_area(points):
    """
    Calculate the area of a polygon using the Shoelace formula.
    
    Parameters:
    - points: A numpy array of shape (n, 2) where n is the number of vertices in the polygon.

    Returns:
    - The area of the polygon.
    """
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def calculate_areas(pieces):
    """
    Calculate the area for each piece.
    
    Parameters:
    - pieces: A list of numpy arrays, where each array represents a piece's points (polygon).
    
    Returns:
    - A torch tensor containing the area of each piece.
    """
    areas = []
    
    # Calculate the area of each piece
    for piece in pieces:
        area = polygon_area(piece)
        areas.append(area)
    
    return torch.tensor(areas)

def calculate_area_matrix(pieces):
    """
    Calculate the area matrix by summing the areas of each pair of pieces.
    
    Parameters:
    - pieces: A list of numpy arrays, where each array represents a piece's points (polygon).
    
    Returns:
    - A torch tensor representing the area matrix.
    """

    areas = calculate_areas(pieces)
    
    area_matrix = areas[:, np.newaxis] + areas[np.newaxis, :]
    
    return area_matrix



def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(point1 - point2)

def adjacency_matrix_from_puzzle(puzzle, distance_threshold):
    """
    Calculate the adjacency matrix for the given puzzle pieces based on their points.

    Parameters:
    - puzzle: A list of numpy arrays, where each array represents a piece's points.
    - distance_threshold: A float that defines the maximum distance for two pieces to be considered adjacent.

    Returns:
    - A 2D numpy array representing the adjacency matrix.
    """
    num_pieces = len(puzzle)
    adjacency_matrix = np.zeros((num_pieces, num_pieces), dtype=int)

    # Calculate adjacency
    for i in range(num_pieces):
        for j in range(num_pieces):
            if i == j:  # Skip self-comparison
                continue

            piece_i = puzzle[i]
            piece_j = puzzle[j]


            for point_i in piece_i:
                for point_j in piece_j:
                    if calculate_distance(point_i, point_j) < distance_threshold:
                        adjacency_matrix[i, j] = 1
                        break 
                if adjacency_matrix[i, j] == 1:
                    break  

    return adjacency_matrix



def precision_2d(adj_pred, adj_true, areas_matrix):
    
    both = torch.tensor(np.logical_and(adj_pred, adj_true))
    both_areas = torch.sum(both * areas_matrix)
    true_areas = torch.sum(torch.tensor(adj_true) * areas_matrix)
    
    return both_areas / true_areas if true_areas > 0 else 0


def recall_2d(adj_pred, adj_true, areas_matrix):
    
    both = torch.tensor(np.logical_and(adj_pred, adj_true))
    
    both_areas = torch.sum(both * areas_matrix)
    
    pred_areas = torch.sum(torch.tensor(adj_pred) * areas_matrix)
    
    return both_areas / pred_areas if pred_areas > 0 else 0


def f1_2d(adj_pred, adj_true, areas_matrix):
    _precision = precision_2d(adj_pred, adj_true, areas_matrix)
    _recall = recall_2d(adj_pred, adj_true, areas_matrix)
    return 2 * _precision * _recall / (_precision + _recall) if _precision + _recall > 0 else 0



def calculate_intersection_area(polygon1, polygon2):
    """
    Calculate the intersection area between two polygons using Shapely.
    
    Parameters:
    - polygon1, polygon2: numpy arrays representing the points of each polygon.

    Returns:
    - The area of the intersection of the two polygons.
    """
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)
    
    
    try:
        intersection = poly1.intersection(poly2)
    except:
        poly1 = poly1.buffer(0)
        poly2 = poly2.buffer(0)
        intersection = poly1.intersection(poly2)
    
    if not intersection.is_empty:
        return intersection.area
    else:
        return 0

def calculate_qpos(gt_puzzle, pred_puzzle):
    """
    Calculate the Qpos between two 2D puzzles (ground truth and predicted).
    
    Parameters:
    - gt_puzzle: A list of numpy arrays representing the ground truth puzzle pieces.
    - pred_puzzle: A list of numpy arrays representing the predicted puzzle pieces.

    Returns:
    - Qpos: The calculated Qpos score.
    """
    gt_areas = np.array([polygon_area(piece) for piece in gt_puzzle])
    pred_areas = np.array([polygon_area(piece) for piece in pred_puzzle])

    total_gt_area = np.sum(gt_areas)
    
    intersection_areas = []
    for gt_piece, pred_piece in zip(gt_puzzle, pred_puzzle):
        intersection_area = calculate_intersection_area(gt_piece, pred_piece)
        intersection_areas.append(intersection_area)
    
    volume_weights = gt_areas / total_gt_area
    
    Qpos = 0
    for i, w in zip(intersection_areas, volume_weights):
        pred_area = pred_areas[list(intersection_areas).index(i)] 
        if pred_area > 0:
            Qpos += w * (i / pred_area)
    
    return Qpos



def rotate_points(points, cos_theta, sin_theta):
    shape = points.shape
    theta = -th.atan2(-sin_theta, cos_theta)
    cos_theta = th.cos(theta)
    sin_theta = -th.sin(theta)
    sin_theta = th.sin(theta)
    cos_theta = th.cos(theta)

    rotation_matrix = th.stack([
        th.stack([cos_theta, -sin_theta]),
        th.stack([sin_theta, cos_theta])
    ])
    rotation_matrix = rotation_matrix.permute([2,3,4,0,1])
    points = points.reshape(-1, 2, 1)
    rotation_matrix = rotation_matrix.reshape(-1, 2, 2)
    rotated_points = th.bmm(rotation_matrix.double(), points.double())
    return rotated_points.reshape(shape)




def save_samples(sample, ext, model_kwargs, rotation, tmp_count, save_gif=True, save_edges=True, ID_COLOR=None, save_svg=True, max_num_points=100, device='cpu'): #TODO
    if not save_gif:
        sample = sample[-1:]
    for k in range(sample.shape[0]):
        if rotation:
            rot_s_total=[]
            rot_c_total=[]
            for nb in range(model_kwargs[f'room_indices'].shape[0]):
                array_a = np.array(model_kwargs[f'room_indices'][nb].cpu())
                room_types = np.where(array_a == array_a.max())[1]
                room_types = np.append(room_types, -10)
                rot_s =[]
                rot_c =[]
                rt =0
                no=0
                for ri in range(len(room_types)):
                    if rt!=room_types[ri]:
                        for nn in range(no):
                            rot_s.append(np.array(rot_s_tmp).mean())
                            rot_c.append(np.array(rot_c_tmp).mean())
                        rt=room_types[ri]
                        no=1
                        rot_s_tmp = [sample[k:k+1,:,:,3][0][nb][ri].cpu().data.numpy()]
                        rot_c_tmp = [sample[k:k+1,:,:,2][0][nb][ri].cpu().data.numpy()]
                    else:
                        no+=1
                        rot_s_tmp.append(sample[k:k+1,:,:,3][0][nb][ri].cpu().data.numpy())
                        rot_c_tmp.append(sample[k:k+1,:,:,2][0][nb][ri].cpu().data.numpy())
                while len(rot_s)<max_num_points:
                    rot_s.append(0)
                    rot_c.append(0)
                rot_s_total.append(rot_s)
                rot_c_total.append(rot_c)
          
            poly = rotate_points(model_kwargs['poly'].unsqueeze(0),th.unsqueeze(th.Tensor(rot_c_total).to(device),0), th.unsqueeze(th.Tensor(rot_s_total).to(device),0))
            # poly = rotate_points(model_kwargs['poly'].unsqueeze(0), sample[k:k+1,:,:,2], sample[k:k+1,:,:,3])
        else:
            poly = model_kwargs['poly'].unsqueeze(0)


        center_total = []
        for nb in range(model_kwargs[f'room_indices'].shape[0]):
            array_a = np.array(model_kwargs[f'room_indices'][nb].cpu())
            room_types = np.where(array_a == array_a.max())[1]
            room_types = np.append(room_types, -10)
            center =[]
            rt =0
            no=0
            for ri in range(len(room_types)):
                if rt!=room_types[ri]:
                    for nn in range(no):
                        center.append(np.array(center_tmp).mean(0))
                    rt=room_types[ri]
                    no=1
                    center_tmp = [sample[k:k+1,:,:,:2][0][nb][ri].cpu().data.numpy()]
                else:
                    no+=1
                    center_tmp.append(sample[k:k+1,:,:,:2][0][nb][ri].cpu().data.numpy())
            while len(center)<max_num_points:
                center.append([0, 0])
            center_total.append(center)

        sample[k:k+1,:,:,:2] = th.Tensor(center_total).to(device) + poly
        # sample[k:k+1,:,:,:2] = sample[k:k+1,:,:,:2] + poly
    sample = sample[:,:,:,:2]
    return sample[-1]



def weighted_points(polys):
    areas = [np.full((len(poly), 1),
                     cv2.contourArea(poly.astype(np.float32)))
             for poly in polys]
    weights = np.vstack(areas)
    points = np.vstack(polys)
    return (weights, points)

def translate_to_gt(pieces_gt, pieces_sol):
    W, p_gt = weighted_points(pieces_gt)
    p_sol = np.vstack(pieces_sol)
    center_gt, center_sol = [np.sum(W * p, axis=0) / np.sum(W)
                             for p in [p_gt, p_sol]]
    X = p_sol - center_sol
    Y = p_gt - center_gt
    S = X.T @ np.diag(W.squeeze()) @ Y
    U, _, V = np.linalg.svd(S)
    R = (V @ np.array([[1, 0],
                       [0, np.linalg.det(V @ U.T)]]) @ U.T)
    t = center_gt - center_sol @ R.T
    return (R.T, t)




def get_metric(gt, pred, model_kwargs):
    gt_puzzles = []
    pred_puzzles = []
    pres_total = 0
    recall_total = 0
    f1_total = 0
    qpos_total = 0
    
  
    
    for i in range(gt.shape[0]):
        gt_puzzle = []
        pred_puzzle = []
        gt_poly = []
        pred_poly = []
        for j in range(gt.shape[1]):
            if j>0:
                if(model_kwargs['room_indices'][i][j].argmax()!=model_kwargs['room_indices'][i][j-1].argmax()) or j==gt.shape[1]-1:
                    gt_puzzle.append(np.array(gt_poly))
                    pred_puzzle.append(np.array(pred_poly))
                    gt_poly, pred_poly = [], []
            gt_poly.append([gt[i][j][0].cpu().data, gt[i][j][1].cpu().data])
            pred_poly.append([pred[i][j][0].cpu().data, pred[i][j][1].cpu().data])
        gt_puzzles.append(gt_puzzle[:-1]) # -1 for padding
        pred_puzzles.append(pred_puzzle[:-1]) # -1 for padding

        R, t = translate_to_gt(gt_puzzles[-1], pred_puzzles[-1]) # selecting the last one
        sol_transformed = [p @ R + t for p in pred_puzzles[-1]]
        
        gt_puzzle_points = gt_puzzles[-1]
        
        
        qpos_total += calculate_qpos(gt_puzzle_points, sol_transformed)
        
       
        

    
    return qpos_total/gt.shape[0]


def calc_rels(B, N,  K, D, keypoints):


    device = keypoints.device
    idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)  # shape: (B, N)
    real_counts = (D * K).unsqueeze(1).to(device)  # shape: (B, 1)
    valid_mask = idx < real_counts     # shape: (B, N); True for real keypoints
    kp_exp1 = keypoints.unsqueeze(2)  # shape: (B, N, 1, 2)
    kp_exp2 = keypoints.unsqueeze(1)  # shape: (B, 1, N, 2)
    distances = torch.norm(kp_exp1 - kp_exp2, dim=-1)  # shape: (B, N, N)

    piece_indices = idx // K  # shape: (B, N)
    
    valid_pair_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)  # (B, N, N)
    
    same_piece_mask = (piece_indices.unsqueeze(2) == piece_indices.unsqueeze(1))  # (B, N, N)
    
    final_mask = valid_pair_mask & (~same_piece_mask)
    
    final_mask[:, torch.arange(N), torch.arange(N)] = False
    distances_masked = distances.clone()
    distances_masked[~final_mask] = float('inf')

    nearest_neighbor_idx = torch.argmin(distances_masked, dim=-1)  # shape: (B, N)
    output = torch.zeros(B, N, 2, dtype=torch.long, device=device)
    
    output[valid_mask] = torch.stack([idx[valid_mask], nearest_neighbor_idx[valid_mask]], dim=-1)

    return output

