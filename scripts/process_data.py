import os
import json
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms

SEED = 42



np.random.seed(SEED)
torch.manual_seed(SEED)


model_path = 'feature_extractor.pt'

# Build the model architecture (without final FC)
resnet18 = models.resnet18()
resnet18 = nn.Sequential(*list(resnet18.children())[:-1])

if os.path.exists(model_path):
    print(f"Loading model weights from {model_path}")
    resnet18.load_state_dict(torch.load(model_path))


resnet18.eval()

    

transform = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])




scale_factor = 0.0001
maxKP = 100

# Harris parameters
blockSize = 2
ksize = 3
k = 0.04

epsilon = 1e-6


def extract_global_feature(pil_image):
    # Convert RGBA to RGB if needed
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0) 
    with torch.no_grad():
        feat = resnet18(input_tensor)  
    feat = feat.squeeze().cpu().tolist() 
    return feat



def extract_local_patch_feature(image_np, keypoint, patch_size=32, resnet_model=None, transform=None):
    """
    Extract a square patch around the keypoint from image_np, pad with white if needed,
    and compute feature vector using pretrained ResNet-18.
    
    Args:
        image_np (np.array): H x W x 3 RGB image
        keypoint (tuple): (x, y) coordinates of keypoint
        patch_size (int): size of square patch to extract
        resnet_model (torch.nn.Module): pretrained ResNet-18 model (without final layer)
        transform (torchvision.transforms.Compose): preprocessing transforms for ResNet input
    
    Returns:
        np.array: 512-dimensional feature vector
    """
    x, y = keypoint
    half_patch = patch_size // 2
    
    x1 = max(x - half_patch, 0)
    y1 = max(y - half_patch, 0)
    x2 = min(x + half_patch + 1, image_np.shape[1])
    y2 = min(y + half_patch + 1, image_np.shape[0])
    
    patch = image_np[y1:y2, x1:x2]
    
    # Pad if patch is smaller than expected
    bottom_pad = max(0, patch_size - patch.shape[0])
    right_pad = max(0, patch_size - patch.shape[1])
    
    if bottom_pad > 0 or right_pad > 0:
        patch = cv2.copyMakeBorder(patch, 0, bottom_pad, 0, right_pad,
                                   borderType=cv2.BORDER_CONSTANT,
                                   value=(255, 255, 255))

    
    patch_pil = Image.fromarray(patch).convert('RGB')
    
    
    
    # plt.imshow(patch_pil)
    # plt.title("Patch")
    # plt.axis("off")
    # plt.show()


    
    input_tensor = transform(patch_pil).unsqueeze(0)
    
    with torch.no_grad():
        feat = resnet_model(input_tensor)
    
    feat = feat.squeeze().cpu().tolist()
    
    return feat


def load_data_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def sort_polygon_points(points):
    centroid = np.mean(points, axis=0)
    sorted_points = sorted(points, key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0]))
    return np.array(sorted_points)


def farthest_point_sampling(points, k):
    if len(points) == 0:
        return points
    k = min(k, len(points))
    sampled_indices = [np.random.choice(len(points))]
    for _ in range(1, k):
        distances = np.min(np.linalg.norm(points[:, np.newaxis] - points[sampled_indices], axis=2), axis=1)
        sampled_indices.append(np.argmax(distances))
    return points[sampled_indices]



def draw_keypoints_on_image(image_path, kp_data, output_path, radius=5, color=(0, 255, 0), thickness=2):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not load image at {image_path}")
        return

    # with open(json_path, 'r') as f:
    #     kp_data = json.load(f)

    for key, points in kp_data.items():
        if key == "0":
            continue
        for point in points[:-1]:   # cause last is global feature
            
            if isinstance(point, list) and isinstance(point[0], list) and len(point[0]) == 2:
                x, y = point[0]
                cv2.circle(image, (int(x), int(y)), radius, color, thickness)

                # Optional: draw edge direction arrow using angle
                if len(point) >= 2:
                
                    angle, curvature = point[1]
                    dx = int(10 * np.cos(np.deg2rad(angle)))
                    dy = int(10 * np.sin(np.deg2rad(angle)))
                    cv2.arrowedLine(image, (int(x), int(y)), (int(x + dx), int(y + dy)), (0, 0, 255), 1)

    cv2.imwrite(output_path, image)
    print(f"Saved image with keypoints to: {output_path}")




def construct_puzzle_image(puzzle_folder, save_folder, json_save_folder, kp_visual_folder, puzzle_counter):
    
    json_path = os.path.join(puzzle_folder, 'data.json')
    data = load_data_json(json_path)
    fragments = data['fragments']

    pieces = []
    keypoints_data = {}

    for idx, frag in enumerate(fragments, start=1):
        img_name = frag['filename'].replace('.obj', '.png')
        img_path = os.path.join(puzzle_folder, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_name} not found in {puzzle_folder}")
            continue

        img_pil = Image.open(img_path).convert('RGBA')
        x, y = frag['pixel_position'][0], frag['pixel_position'][1]
        x_scaled = x * scale_factor
        y_scaled = y * scale_factor
        pieces.append((img_pil, x_scaled, y_scaled, idx, img_name))

    if not pieces:
        print(f"No pieces loaded for {puzzle_folder}")
        return

    min_x = min(p[1] for p in pieces)
    min_y = min(p[2] for p in pieces)
    max_x = max(p[1] + p[0].width for p in pieces)
    max_y = max(p[2] + p[0].height for p in pieces)
    width = int(max_x - min_x)
    height = int(max_y - min_y)

    solution_np = np.ones((height, width, 4), dtype=np.uint8) * 255  # RGBA
    puzzle_name = os.path.basename(puzzle_folder)
    keypoints_data["puzzle_name"] = puzzle_name
    keypoints_data["0"] = [height, width]

    for img_pil, x, y, idx, img_name in pieces:
        
        # Extract global feature for this piece
        global_feature = extract_global_feature(img_pil)



        pos_x = int(x - min_x)
        pos_y = int(y - min_y)
        img_np = np.array(img_pil)

        if img_np.shape[2] == 4:
            alpha_channel = img_np[..., 3]
            mask = alpha_channel > 0
        else:
            mask = np.any(img_np != 0, axis=-1)

        non_zero_coords = np.argwhere(mask)
        

        for coord in non_zero_coords:
            canvas_y = pos_y + coord[0]
            canvas_x = pos_x + coord[1]
            if 0 <= canvas_y < height and 0 <= canvas_x < width:
                solution_np[canvas_y, canvas_x] = img_np[coord[0], coord[1]]

        gray_piece = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2GRAY)
        harris_corners = cv2.cornerHarris(gray_piece, blockSize, ksize, k)
        harris_corners = cv2.dilate(harris_corners, None)
        threshold = 0.001 * harris_corners.max()
        corners = np.argwhere(harris_corners > threshold)
        
        
        
        
        
        # Find contours — piece boundaries
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundary_mask = np.zeros_like(mask, dtype=np.uint8)

        cv2.drawContours(boundary_mask, contours, -1, 255, 1)
        
        # Filter corners to keep only those on boundaries
        filtered_keypoints = []
        for y_c, x_c in corners:
            if boundary_mask[y_c, x_c] != 0:  # corner lies on piece boundary
                # filtered_keypoints.append([pos_x + x_c, pos_y + y_c])
                filtered_keypoints.append([x_c, y_c])

        
        
        keypoints = filtered_keypoints




        keypoints_np = farthest_point_sampling(np.array(keypoints), maxKP)
        keypoints_np = sort_polygon_points(keypoints_np)
        
        # Compute gradients using Sobel on grayscale piece
        Gx = cv2.Sobel(gray_piece, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(gray_piece, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute second derivatives (Hessian)
        Ixx = cv2.Sobel(Gx, cv2.CV_64F, 1, 0, ksize=3)
        Iyy = cv2.Sobel(Gy, cv2.CV_64F, 0, 1, ksize=3)
        Ixy = cv2.Sobel(Gx, cv2.CV_64F, 0, 1, ksize=3)
        
        corner_features = []
        for (x, y) in keypoints_np:
            # Make sure x,y are integer indices inside image bounds
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < gray_piece.shape[1] and 0 <= iy < gray_piece.shape[0]:
                angle = np.arctan2(Gy[iy, ix], Gx[iy, ix]) * (180 / np.pi)
                detH = (Ixx[iy, ix] * Iyy[iy, ix]) - (Ixy[iy, ix] ** 2)
                traceH = Ixx[iy, ix] + Iyy[iy, ix]
                
                curvature = (traceH ** 2) / (detH + epsilon)

                

            else:
                angle, curvature = 0.0, 0.0
            
            
            # Extract local patch feature from the original RGB piece image
            local_feature = extract_local_patch_feature(img_np, (ix, iy), patch_size=32, resnet_model=resnet18, transform=transform)
            
            corner_features.append([[int(x + pos_x), int(y + pos_y)], [round(angle, 2), round(curvature, 2)], local_feature])
        
        corner_features.append(['piece_name', img_name])
        corner_features.append(['global', global_feature])
        keypoints_data[str(idx)] = corner_features

        
        
        
        
        
        
    # raw_image_path = os.path.join(save_folder, f"{puzzle_name}_solution.png")
    # #Save reconstructed puzzle image
    # os.makedirs(save_folder, exist_ok=True)
    # cv2.imwrite(raw_image_path, cv2.cvtColor(solution_np, cv2.COLOR_RGBA2BGRA))
    # print(f"Saved solution image at: {raw_image_path}")

    # Save JSON
    os.makedirs(json_save_folder, exist_ok=True)
    json_path_out = os.path.join(json_save_folder, f"{puzzle_name}.json")
    # with open(json_path_out, 'w') as jf:
    #     json.dump(keypoints_data, jf, indent=2)
        
    print(f"Saved keypoints JSON at: {json_path_out}")
    torch.save(keypoints_data, os.path.join(json_save_folder, f"data_{puzzle_counter}.pt"))
    

    # #Save visualized keypoints image
    # os.makedirs(kp_visual_folder, exist_ok=True)
    # visual_output_path = os.path.join(kp_visual_folder, f"{puzzle_name}_KP.png")
    # draw_keypoints_on_image(raw_image_path, keypoints_data, visual_output_path)

def process_all_puzzles(root_folder, save_folder, json_save_folder, kp_visual_folder):
    puzzle_counter = 0
    for entry in os.listdir(root_folder):
        puzzle_folder = os.path.join(root_folder, entry)
        if os.path.isdir(puzzle_folder) and entry.startswith('puzzle_'):
            construct_puzzle_image(puzzle_folder, save_folder, json_save_folder, kp_visual_folder, puzzle_counter)
            puzzle_counter = puzzle_counter + 1




if __name__ == "__main__":
    
    created_dataset_path = 'RePAIR_dataset'
    
    print('\n----------- TRAIN -----------')
    root_puzzle_folder = r"RePAIR_V2/2D_FINAL/2d_fixed/train"
    save_gt_folder = f"{created_dataset_path}/train_gt_saved"
    save_json_folder = f"{created_dataset_path}/train_kpLocal_jsons_saved"
    save_kp_vis_folder = f"{created_dataset_path}/train_kp_images_saved"

    os.makedirs(save_gt_folder, exist_ok=True)
    os.makedirs(save_json_folder, exist_ok=True)
    os.makedirs(save_kp_vis_folder, exist_ok=True)

    process_all_puzzles(root_puzzle_folder, save_gt_folder, save_json_folder, save_kp_vis_folder)
    
    
    print('\n----------- TEST -----------')
    root_puzzle_folder = r"RePAIR_V2/2D_FINAL/2d_fixed/test"
    save_gt_folder = f"{created_dataset_path}/test_gt_saved"
    save_json_folder = f"{created_dataset_path}/test_kpLocal_jsons_saved"
    save_kp_vis_folder = f"{created_dataset_path}/test_kp_images_saved"

    os.makedirs(save_gt_folder, exist_ok=True)
    os.makedirs(save_json_folder, exist_ok=True)
    os.makedirs(save_kp_vis_folder, exist_ok=True)

    process_all_puzzles(root_puzzle_folder, save_gt_folder, save_json_folder, save_kp_vis_folder)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
