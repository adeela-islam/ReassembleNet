import os
import math
import json
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader, Dataset, DistributedSampler



import warnings
warnings.filterwarnings('ignore')



def rotate_points(points, indices):
    indices = np.argmax(indices,1)
    indices[indices==0] = 1000
    unique_indices = np.unique(indices)
    num_unique_indices = len(unique_indices)
    rotated_points = np.zeros_like(points)
    rotation_angles = []
    for i in unique_indices:
        idx = (indices == i)
        selected_points = points[idx]
        rotation_degree = 0 if i==1 else (np.random.rand() * 360)
        # rotation_angle = 0 
        # rotation_angle = 0 if i==0 else (np.random.randint(4) * 90)
        rotation_angle = np.deg2rad(rotation_degree)
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)], # this is selected for return
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])
        rotated_selected_points = np.matmul(rotation_matrix, selected_points.T).T
        rotated_points[idx] = rotated_selected_points
        # rotation_matrix[0,1] = 1 if rotation_angle<np.pi else -1
        rotation_angles.extend(rotation_matrix[0:1].repeat(rotated_selected_points.shape[0], axis=0))
    return rotated_points, rotation_angles, rotation_degree



def load_repair_data(
    batch_size,
    set_name,
    rotation,
    dataset_path,
    max_num_points,
    maxcount,
    number_larger_than_pieces_and_points_in_puzzle,
    device,
    loader_num_workers,
    rank,
    world_size,
    use_geometry_only,
    use_global_texture_only,
    images_folder_path,
    use_local_texture_only,
    use_geometry_and_global_texture,
    use_geometry_and_local_texture,
    use_local_and_global_texture,
    use_geometry_global_local_texture,
    use_learnable_kp_selection,
    ):
    
    if rank == 0:
        print(f"\n================Loading {set_name} of RePAIR...")
    

    dataset = repair(set_name, rotation, dataset_path,
                      max_num_points,
                      maxcount,
                      number_larger_than_pieces_and_points_in_puzzle, 
                      device,
                      loader_num_workers,
                      rank,
                      use_geometry_only,
                      use_global_texture_only,
                      images_folder_path,
                      use_local_texture_only,
                      use_geometry_and_global_texture,
                      use_geometry_and_local_texture,
                      use_local_and_global_texture,
                      use_geometry_global_local_texture,
                      use_learnable_kp_selection,
                      )
    

    
    if set_name == 'test':
        test_data_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) 
        test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=loader_num_workers, drop_last=False, sampler=test_data_sampler)

        return test_loader, test_data_sampler

    
    elif set_name == 'val':
        val_data_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) 
        val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=loader_num_workers, drop_last=False, sampler=val_data_sampler)

        return val_loader, val_data_sampler
        
    

    elif set_name == 'train':

        train_data_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) 
        train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=loader_num_workers, drop_last=False, sampler=train_data_sampler)

        return train_loader, train_data_sampler
        

    else:
        raise ValueError("set_name must be either 'train' 'val' or 'test'.")
    
    


    


class repair(Dataset):
    def __init__(self, set_name, rotation, dataset_path,
                 max_num_points,
                 maxcount,
                 number_larger_than_pieces_and_points_in_puzzle,
                 device,
                 loader_num_workers,
                 rank,
                 use_geometry_only,
                 use_global_texture_only,
                 images_folder_path,
                 use_local_texture_only,
                 use_geometry_and_global_texture,
                 use_geometry_and_local_texture,
                 use_local_and_global_texture,
                 use_geometry_global_local_texture,
                 use_learnable_kp_selection,
                 ):
        super().__init__()
        sizes = {
            'center': 2,
            'angles': 2,
            'poly': 2,
            'corner_index': number_larger_than_pieces_and_points_in_puzzle,
            'piece_index': number_larger_than_pieces_and_points_in_puzzle,
            'padding_mask': 1,
            'connections': 2
        }
        indices = np.cumsum([0] + list(sizes.values()))  # Add 0 for starting index
        self.start_idx = {key: indices[i] for i, key in enumerate(sizes)}
        self.end_idx = {key: indices[i+1] for i, key in enumerate(sizes)}
        
        if rank == 0:
            print(f'\nself.start_idx : {self.start_idx }')
            print(f'self.end_idx : {self.end_idx }')
        
        
        if set_name == "train":
            path = f'{dataset_path}/jsons'
        elif set_name == "test":
            path = f'{dataset_path}/jsons_test'
        elif set_name == "val":
            path = f'{dataset_path}/jsons_test'
            
        self.use_geometry_only = use_geometry_only
        self.use_global_texture_only = use_global_texture_only
        self.images_folder_path = images_folder_path
        self.use_local_texture_only = use_local_texture_only
        self.use_geometry_and_global_texture = use_geometry_and_global_texture
        self.use_geometry_and_local_texture = use_geometry_and_local_texture
        self.use_local_and_global_texture = use_local_and_global_texture
        self.use_geometry_global_local_texture = use_geometry_global_local_texture
        self.use_learnable_kp_selection = use_learnable_kp_selection

        
        self.set_name = set_name
        self.rotation = rotation
        self.puzzles = []
        self.rels = []
        houses = {}
        pairss = {}
        files = glob(f'{path}/*')
        
    
        files = [x.split('/')[-1][:-2].split('_') for x in files] 


        notused = set()
        num_p_c =np.zeros(maxcount)
        num_h_min =12345678
        num_h_max = -1
        num_h_sum = []
        num_av = []
        min_num_av = 123456
        max_num_av = -1
        
        for name in tqdm(files, desc ='loading data files'):   
            used = True
            image_size =[0,0]
            name[1] = name[1][:-1]
            if name[1] not in houses:
                houses[name[1]] = []
                cnt = torch.load(f'{path}/{name[0]}_{name[1]}.pt') 
                
                if self.use_learnable_kp_selection:
                    if len(cnt.keys())-1 > maxcount-1:
                        continue
                
                pairs =[]
                numbers ={}
                
                if 1+int(list(cnt.keys())[-1])<=3:           
                    continue
                hss = 0
                all_numbers = []
                num_p_c[ 1+int(list(cnt.keys())[-1])]+=1      
                num_av_t = 0
                num_av_c = 0
                for i in range(1, 1+int(list(cnt.keys())[-1])):   
                    contours = cnt[str(i)]                     
                    piece_global_texture = contours[-1][1] 
                    piece_name = contours[-2][1]
                    contours = contours[:100]
                    
                    if( len(contours)<3):                      
                        used =False
                        notused.add(int(name[1][:-1]))
                        houses[name[1]] = []
                        continue
                    img_size =cnt["0"]
                    image_size = img_size
                    wxx = 2*img_size[0]/256.0
                    wyy =2* img_size[1]/256.0
                    numbers[i] = []  
                    if (len(contours)> max_num_av): 
                        max_num_av = len(contours)
                    if (len(contours)< min_num_av):
                        min_num_av = len(contours)
                    num_av_t+=len(contours)       
                    num_av_c+=1 
                    
                    for cnc in contours:                    
                            all_numbers.append([cnc[0], hss, i])
                            numbers[i].append([cnc[0], hss, i])
                            hss+=1

                    
                    if used == True:
                        
                        poly = []
                        if self.use_geometry_only or self.use_geometry_and_global_texture or self.use_geometry_and_local_texture or self.use_geometry_global_local_texture or self.use_learnable_kp_selection:
                            geom_feats = []

                  
                        if self.use_global_texture_only or self.use_geometry_and_global_texture or self.use_local_and_global_texture or self.use_geometry_global_local_texture:
                            piece_image_name = piece_name
                            

                        if self.use_local_texture_only or self.use_geometry_and_local_texture or self.use_local_and_global_texture or self.use_geometry_global_local_texture:
                            local_texture_feats = []

                       
                        for cntt in contours:     
                            ax = 0 #random.uniform(-2, 2)*img_size[0]/256.0   ## change it if you want to add noise
                            ay = 0 # random.uniform(-2, 2)*img_size[1]/256.0  ## change it if you want to add noise
                            a = (cntt[0][0] +ax)/ img_size[0]
                            b = (cntt[0][1]+ay) /img_size[1]
                            poly.append([a,b])

                            if self.use_geometry_only or self.use_geometry_and_global_texture or self.use_geometry_and_local_texture or self.use_geometry_global_local_texture or self.use_learnable_kp_selection:
                                edge_angle = cntt[1][0]
                                curvature = cntt[1][1]
                                geom_feats.append([edge_angle, curvature])
                            
                            if self.use_local_texture_only or self.use_geometry_and_local_texture or self.use_local_and_global_texture or self.use_geometry_global_local_texture:
                                local_texture_feats.append(cntt[-1])     
                            
                        
                        cx = np.mean(np.array(poly)[:,0])
                        cy =  np.mean(np.array(poly)[:,1])

                        if not self.use_learnable_kp_selection:

                            if self.use_geometry_only:
                                houses[name[1]].append( {'poly' : np.array(poly) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': name[1], 
                                                        'geom_feats':  np.array(geom_feats)})
                            
                            elif self.use_global_texture_only:
                                
                                
                                houses[name[1]].append( {'poly' : np.array(poly) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': name[1],
                                                        'pc_img': piece_global_texture})
                            
                            elif self.use_local_texture_only:
                                houses[name[1]].append( {'poly' : np.array(poly) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': name[1], 
                                                        'local_texture_feats':  np.array(local_texture_feats)})
                                
                            elif self.use_geometry_and_global_texture:
                                
                                houses[name[1]].append( {'poly' : np.array(poly) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': name[1],
                                                        'geom_feats':  np.array(geom_feats), 'pc_img': piece_global_texture})  
                            
                            elif self.use_geometry_and_local_texture:
                                houses[name[1]].append( {'poly' : np.array(poly) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': name[1], 
                                                        'geom_feats':  np.array(geom_feats), 'local_texture_feats':  np.array(local_texture_feats)})

                            elif self.use_local_and_global_texture:
                                
                                houses[name[1]].append( {'poly' : np.array(poly) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': name[1],
                                                        'local_texture_feats':  np.array(local_texture_feats), 'pc_img': piece_global_texture}) 

                            elif self.use_geometry_global_local_texture:
                                
                                houses[name[1]].append( {'poly' : np.array(poly) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': name[1],
                                                        'geom_feats':  np.array(geom_feats), 'pc_img': piece_global_texture, 'local_texture_feats':  np.array(local_texture_feats)})   
                            
                            else:
                                # simple only with poly coordinates
                                houses[name[1]].append( {'poly' : np.array(poly) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': name[1]})
                        
                        
                        elif self.use_learnable_kp_selection:
                            if self.use_geometry_only:
                                houses[name[1]].append( {'poly' : np.array(poly) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': name[1], 
                                                        'geom_feats':  np.array(geom_feats)})
                            
                            elif self.use_global_texture_only:
                                
                                houses[name[1]].append( {'poly' : np.array(poly) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': name[1],
                                                        'geom_feats':  np.array(geom_feats), 'pc_img': piece_global_texture}) 
                            
                            elif self.use_local_texture_only:
                                houses[name[1]].append( {'poly' : np.array(poly) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': name[1], 
                                                        'geom_feats':  np.array(geom_feats), 'local_texture_feats':  np.array(local_texture_feats)})
                                
                            elif self.use_geometry_and_global_texture:
                               
                                houses[name[1]].append( {'poly' : np.array(poly) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': name[1],
                                                        'geom_feats':  np.array(geom_feats), 'pc_img': piece_global_texture})   
                            
                            elif self.use_geometry_and_local_texture:
                                houses[name[1]].append( {'poly' : np.array(poly) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': name[1], 
                                                        'geom_feats':  np.array(geom_feats), 'local_texture_feats':  np.array(local_texture_feats)})

                            elif self.use_local_and_global_texture:
                                
                                houses[name[1]].append( {'poly' : np.array(poly) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': name[1],
                                                        'geom_feats':  np.array(geom_feats), 'local_texture_feats':  np.array(local_texture_feats), 'pc_img': piece_global_texture}) 

                            
                            elif self.use_geometry_global_local_texture:
                                
                                houses[name[1]].append( {'poly' : np.array(poly) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': name[1],
                                                        'geom_feats':  np.array(geom_feats), 'pc_img': piece_global_texture, 'local_texture_feats':  np.array(local_texture_feats)})  
                            
                            else:
                                houses[name[1]].append( {'poly' : np.array(poly) -np.array([cx,cy]), 'center': np.array([cx,cy]),'image_size': img_size, 'name': name[1], 
                                                        'geom_feats':  np.array(geom_feats)})

                
                if num_av_c != 0:
                    num_av.append(num_av_t/num_av_c)        
                pairs = []
                for tk in numbers.keys():
                        number = numbers[tk]
                        for a in number:
                            min_diss =10000
                            pair_b = -1
                            point = a[0]
                            index =a [1]
                            room_index =a[2]
                            for  nn in all_numbers:
                                point_b =nn[0]
                                #print(point , point_b, all_numbers)
                                index_b =nn [1]
                                room_index_b =nn[2]
                                if room_index == room_index_b:
                                    continue
                                if abs(math.dist(point , point_b)) <min_diss and abs(math.dist(point , point_b)) < 10*abs(math.sqrt(wxx**2 + wyy**2)):
                                    min_diss = abs(math.dist(point , point_b))
                                    pair_b =index_b
                            if pair_b!=-1 :
                                pairs.append([index, pair_b])
                        pairss[name[1]] = pairs
                        pairss[name[1]] = pairs
                        if image_size[0] <50 or image_size[1] <50:
                            pairss[name[1]] = []
                            houses[name[1]] = []
                        if len(all_numbers)>max_num_points or len(all_numbers) <10 or len(pairs)>max_num_points :
                            pairss[name[1]] = []
                            houses[name[1]] = []

        keyss = houses.keys()
        self.puzzles1 =[]
        self.rels =[]
        
        for ke in keyss:
            if len(houses[ke])>1 and len(pairss[ke]) >= 3:
                self.puzzles1.append(houses[ke])
                padding = np.zeros((max_num_points-len(pairss[ke]), 2))
                rel = np.concatenate((np.array(pairss[ke]), padding), 0)
                self.rels.append(rel)

        get_one_hot = lambda x, z: np.eye(z)[x]
        puzzles = []
        self_masks = []
        gen_masks = []
        for p in (self.puzzles1):
            puzzle = []
            corner_bounds = []
            num_points = 0
            
            for i, piece in enumerate(p):   
                poly = piece['poly']
                center = np.ones_like(poly) * piece['center']
                # Adding conditions
                num_piece_corners = len(poly)       
                piece_index = np.repeat(np.array([get_one_hot(len(puzzle)+1, number_larger_than_pieces_and_points_in_puzzle)]), num_piece_corners, 0)   
                corner_index = np.array([get_one_hot(x, number_larger_than_pieces_and_points_in_puzzle) for x in range(num_piece_corners)]) 
                if self.rotation:
                    poly, angles, degree = rotate_points(poly, piece_index)
                    

                padding_mask = np.repeat(1, num_piece_corners)   
                padding_mask = np.expand_dims(padding_mask, 1)
                connections = np.array([[i,(i+1)%num_piece_corners] for i in range(num_piece_corners)])
                connections += num_points
                corner_bounds.append([num_points, num_points+num_piece_corners])
                num_points += num_piece_corners

                if not self.use_learnable_kp_selection:

                    if self.use_geometry_only:
                        geom_feats = piece['geom_feats']
                        piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections, geom_feats), 1)
                    
                    elif self.use_global_texture_only:

                        global_texture_of_piece = np.array(piece['pc_img'])
                        global_texture_of_piece = global_texture_of_piece.reshape(1, -1).repeat(poly.shape[0], axis=0)
                        
                        piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections, global_texture_of_piece), 1)
                    
                    elif self.use_local_texture_only:
                        local_text_feats = piece['local_texture_feats']
                        piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections, local_text_feats), 1)

                    elif self.use_geometry_and_global_texture:
                        geom_feats = piece['geom_feats']
                        global_texture_of_piece = np.array(piece['pc_img'])
                        global_texture_of_piece = global_texture_of_piece.reshape(1, -1).repeat(poly.shape[0], axis=0)
                        piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections, geom_feats, global_texture_of_piece), 1)
                    
                    elif self.use_geometry_and_local_texture:
                        geom_feats = piece['geom_feats']
                        local_text_feats = piece['local_texture_feats']
                        piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections, geom_feats, local_text_feats), 1)


                    elif self.use_local_and_global_texture:
                        local_text_feats = piece['local_texture_feats']
                        global_texture_of_piece = np.array(piece['pc_img'])
                        global_texture_of_piece = global_texture_of_piece.reshape(1, -1).repeat(poly.shape[0], axis=0)
                        piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections, local_text_feats, global_texture_of_piece), 1)

                    elif self.use_geometry_global_local_texture:
                        geom_feats = piece['geom_feats']
                        local_text_feats = piece['local_texture_feats']
                        global_texture_of_piece = np.array(piece['pc_img'])
                        global_texture_of_piece = global_texture_of_piece.reshape(1, -1).repeat(poly.shape[0], axis=0)
                        piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections, geom_feats, global_texture_of_piece, local_text_feats), 1)

                    else:
                        piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections), 1)
                 
                elif self.use_learnable_kp_selection:
                    if self.use_global_texture_only or self.use_geometry_and_global_texture:
                        geom_feats = piece['geom_feats']
                        global_texture_of_piece = np.array(piece['pc_img'])
                        global_texture_of_piece = global_texture_of_piece.reshape(1, -1).repeat(poly.shape[0], axis=0)
                        piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections, geom_feats, global_texture_of_piece), 1)

                    elif self.use_local_texture_only or self.use_geometry_and_local_texture:
                        geom_feats = piece['geom_feats']
                        local_text_feats = piece['local_texture_feats']
                        piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections, geom_feats, local_text_feats), 1)


                    elif self.use_local_and_global_texture:
                        geom_feats = piece['geom_feats']
                        local_text_feats = piece['local_texture_feats']
                        global_texture_of_piece = np.array(piece['pc_img'])
                        global_texture_of_piece = global_texture_of_piece.reshape(1, -1).repeat(poly.shape[0], axis=0)
                        piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections, geom_feats, local_text_feats, global_texture_of_piece), 1)

                    elif self.use_geometry_global_local_texture:
                        geom_feats = piece['geom_feats']
                        local_text_feats = piece['local_texture_feats']
                        global_texture_of_piece = np.array(piece['pc_img'])
                        global_texture_of_piece = global_texture_of_piece.reshape(1, -1).repeat(poly.shape[0], axis=0)
                        piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections, geom_feats, global_texture_of_piece, local_text_feats), 1)

                    else:
                        geom_feats = piece['geom_feats']
                        piece = np.concatenate((center, angles, poly, corner_index, piece_index, padding_mask, connections, geom_feats), 1)

                
                horizontal_dim = piece.shape[1] 
                
                puzzle.append(piece)
            
            puzzle_layouts = np.concatenate(puzzle, 0)
            if len(puzzle_layouts)>max_num_points:
                assert False
            num_h_sum.append(len(puzzle_layouts))
            if num_h_min >len(puzzle_layouts):
                num_h_min =len(puzzle_layouts)
            if num_h_max <len(puzzle_layouts):
                num_h_max =len(puzzle_layouts)
            padding = np.zeros((max_num_points-len(puzzle_layouts), horizontal_dim))
            gen_mask = np.ones((max_num_points, max_num_points))
            gen_mask[:len(puzzle_layouts), :len(puzzle_layouts)] = 0
            puzzle_layouts = np.concatenate((puzzle_layouts, padding), 0)
            self_mask = np.ones((max_num_points, max_num_points))
            for i in range(len(corner_bounds)):
                self_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[i][0]:corner_bounds[i][1]] = 0
            puzzles.append(puzzle_layouts)
            self_masks.append(self_mask)
            gen_masks.append(gen_mask)

        
        
        self.max_num_points = max_num_points 
        self.puzzles = puzzles   
        self.self_masks = self_masks   
        self.gen_masks = gen_masks
        self.num_coords = 4
       
    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        arr = self.puzzles[idx][:, :self.num_coords]

        trans_rot_gt = self.puzzles[idx][:, 0:4]
        trans = self.puzzles[idx][:, 0:2]
        rots = self.puzzles[idx][:, 2:4]


        polys = self.puzzles[idx][:, self.num_coords:self.num_coords+2]
        cond = {
                'self_mask': self.self_masks[idx],
                'gen_mask': self.gen_masks[idx],
                'poly': polys,
                'corner_indices': self.puzzles[idx][:, self.start_idx['corner_index']:self.end_idx['corner_index']], # corner_indices of puzzle
                'room_indices': self.puzzles[idx][:, self.start_idx['piece_index']:self.end_idx['piece_index']], # piece indices of puzzle
                'src_key_padding_mask': 1-self.puzzles[idx][:, self.start_idx['padding_mask']],          # padding_mask of puzzle
                'connections': self.puzzles[idx][:, self.start_idx['connections']:self.end_idx['connections']],  # point's connections of puzzle
                'rels': self.rels[idx],

                'trans': trans,
                'rots': rots,
                'trans_rot_gt': trans_rot_gt,
                }
        
        if self.use_learnable_kp_selection:
            num_pieces = np.max(np.where(cond['room_indices'].any(0))) if cond['room_indices'].any() else None
            cond['num_pieces'] = num_pieces

            if self.use_global_texture_only or self.use_geometry_and_global_texture:
                cond['geom_feats'] = self.puzzles[idx][:, -514:-512]
                cond['global_texture_feats'] = self.puzzles[idx][:, -512:]
            
            elif self.use_local_texture_only or self.use_geometry_and_local_texture: 
                cond['geom_feats'] = self.puzzles[idx][:, -514:-512]
                cond['local_texture_feats'] = self.puzzles[idx][:, -512:]

            elif self.use_local_and_global_texture:                      
                cond['geom_feats'] = self.puzzles[idx][:, -1026:-1024]
                cond['local_texture_feats'] = self.puzzles[idx][:, -1024:-512]
                cond['global_texture_feats'] = self.puzzles[idx][:, -512:]

            
            elif self.use_geometry_global_local_texture:              
                cond['geom_feats'] = self.puzzles[idx][:, -1026:-1024]
                cond['local_texture_feats'] = self.puzzles[idx][:, -512:]
                cond['global_texture_feats'] = self.puzzles[idx][:, -1024:-512]
            
            else:
                cond['geom_feats'] = self.puzzles[idx][:, -2:]
        

        elif not self.use_learnable_kp_selection:
            if self.use_geometry_only:
                cond['geom_feats'] = self.puzzles[idx][:, -2:]


            elif self.use_global_texture_only:       
                cond['global_texture_feats'] = self.puzzles[idx][:, -512:]

            elif self.use_local_texture_only:                               
                cond['local_texture_feats'] = self.puzzles[idx][:, -512:]

            elif self.use_geometry_and_global_texture: 
                cond['geom_feats'] = self.puzzles[idx][:, -514:-512]
                cond['global_texture_feats'] = self.puzzles[idx][:, -512:]

            elif self.use_geometry_and_local_texture:    
                cond['geom_feats'] = self.puzzles[idx][:, -514:-512]
                cond['local_texture_feats'] = self.puzzles[idx][:, -512:]

            elif self.use_local_and_global_texture:    
                cond['local_texture_feats'] = self.puzzles[idx][:, -1024:-512]
                cond['global_texture_feats'] = self.puzzles[idx][:, -512:]

            elif self.use_geometry_global_local_texture:           
                cond['geom_feats'] = self.puzzles[idx][:, -1026:-1024]  
                cond['global_texture_feats'] = self.puzzles[idx][:, -1024:-512]
                cond['local_texture_feats'] = self.puzzles[idx][:, -512:]
            

        arr = np.transpose(arr, [1, 0])
        return arr.astype(float), cond

if __name__ == '__main__':
    dataset = repair('test')
