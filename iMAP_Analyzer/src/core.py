# src/core.py
import numpy as np
import cv2
from skimage.filters import threshold_otsu, gaussian
from skimage import morphology
from skimage.morphology import dilation, disk
from scipy.ndimage import convolve, distance_transform_edt
from skimage.measure import label, regionprops

class VascularAnalyzer:
    """
    Core Logic for iMAP Analyzer.
    Optimized for high-speed detailed component analysis.
    """

    @staticmethod
    def preprocess_image(image, thresh_value, white_min, black_min, gaussian_blur):
        # 1. Binarization
        binary_image = image > thresh_value

        # 2. Debris Removal (White Min)
        cleaned_image = morphology.remove_small_objects(binary_image, min_size=white_min)

        # 3. Hole Filling (Black Min)
        inverted_image = ~cleaned_image
        filtered_black_holes = morphology.remove_small_objects(inverted_image, min_size=black_min)
        final_binary_image = ~filtered_black_holes

        # 4. Smoothing
        if gaussian_blur > 0:
            blurred_image = gaussian(final_binary_image, sigma=gaussian_blur)
            try:
                thresh_otsu = threshold_otsu(blurred_image)
                binary_blurred_image = blurred_image > thresh_otsu
            except ValueError:
                binary_blurred_image = final_binary_image
        else:
            binary_blurred_image = final_binary_image

        # 5. Skeletonization
        skeleton_original = morphology.skeletonize(binary_blurred_image)
        skeleton_dilated = dilation(skeleton_original, disk(2))

        return skeleton_dilated, binary_blurred_image, skeleton_original

    @staticmethod
    def find_branchpoints(skeleton):
        # Optimized convolution for nodes
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbors = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
        return (neighbors > 2) & skeleton

    @staticmethod
    def find_endpoints(skeleton):
        # Optimized convolution for endpoints
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        neighbors = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
        endpoints = (neighbors == 11)
        
        # [수정된 부분] 개수(int)와 마스크(array) 두 개를 반환해야 GUI에서 에러가 안 납니다.
        return np.sum(endpoints), endpoints

    @staticmethod
    def calculate_thickness(binary_image, skeleton):
        distance = distance_transform_edt(binary_image)
        skeleton_vals = distance[skeleton > 0]
        if skeleton_vals.size == 0:
            return 0, 0
        thicknesses = skeleton_vals * 2
        return np.mean(thicknesses), np.std(thicknesses)

    @staticmethod
    def analyze_network_detailed(skeleton, binary, filename=""):
        """
        Performs optimized component-wise analysis.
        Avoids re-running convolution for every fragment.
        """
        # --- 1. Pre-calculate Global Maps (Speed Optimization) ---
        global_nodes_map = VascularAnalyzer.find_branchpoints(skeleton)
        
        # [수정] 여기서도 튜플 언패킹을 맞춰줍니다.
        _, global_endpoints_map = VascularAnalyzer.find_endpoints(skeleton)
        
        dist_map = distance_transform_edt(binary) # For thickness
        
        # --- 2. Label Connected Components ---
        labeled_skel, num_labels = label(skeleton, connectivity=2, return_num=True)
        regions = regionprops(labeled_skel)
        
        component_details = []
        
        # --- 3. Iterate Regions with BBox Slicing (Fast) ---
        for i, region in enumerate(regions):
            minr, minc, maxr, maxc = region.bbox
            
            sub_skel_mask = region.image
            
            # Extract features within the bounding box masked by the fragment shape
            sub_nodes = global_nodes_map[minr:maxr, minc:maxc] & sub_skel_mask
            sub_ends = global_endpoints_map[minr:maxr, minc:maxc] & sub_skel_mask
            
            n_nodes = np.sum(sub_nodes)
            n_ends = np.sum(sub_ends)
            length_px = region.area 
            
            # Thickness
            sub_dist = dist_map[minr:maxr, minc:maxc]
            fragment_thicknesses = sub_dist[sub_skel_mask] * 2
            mean_th = np.mean(fragment_thicknesses) if fragment_thicknesses.size > 0 else 0
            std_th = np.std(fragment_thicknesses) if fragment_thicknesses.size > 0 else 0
            
            # Mesh (Hole) Count
            padded_sub = np.pad(sub_skel_mask.astype(np.uint8), 1, mode='constant', constant_values=0)
            contours, hierarchy = cv2.findContours(padded_sub * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            n_meshes = 0
            if hierarchy is not None:
                n_meshes = np.count_nonzero(hierarchy[0][:, 3] != -1)

            component_details.append({
                'Image Name': filename,
                'Fragment ID': i + 1,
                'Length (px)': length_px,
                'Area (Binary px)': np.sum(binary[minr:maxr, minc:maxc]), 
                'Nodes': n_nodes,
                'Endpoints': n_ends,
                'Meshes': n_meshes,
                'Mean Thickness': mean_th,
                'Std Thickness': std_th
            })

        # --- 4. Total Summary ---
        total_nodes = np.sum(global_nodes_map)
        total_endpoints = np.sum(global_endpoints_map)
        total_length = np.sum(skeleton)
        total_area = np.count_nonzero(binary)
        
        # Calculate LMF
        if component_details:
            lengths = [d['Length (px)'] for d in component_details]
            lmf = max(lengths) / sum(lengths) if sum(lengths) > 0 else 0
        else:
            lmf = 0
            
        thickness_vals = dist_map[skeleton > 0] * 2
        total_mean_th = np.mean(thickness_vals) if thickness_vals.size > 0 else 0
        total_std_th = np.std(thickness_vals) if thickness_vals.size > 0 else 0
        wtu = (total_mean_th / total_std_th) if total_std_th > 0 else 0
        
        total_meshes = sum([d['Meshes'] for d in component_details])
        
        connectivity_index = num_labels
        safe_conn = connectivity_index if connectivity_index > 0 else 1

        summary = {
            'Image Name': filename,
            'Total Area (px)': total_area,
            'Total Length (px)': total_length,
            'Mean Thickness': total_mean_th,
            'Std Thickness': total_std_th,
            'WTU': wtu,
            'Total Nodes': total_nodes,
            'Total Endpoints': total_endpoints,
            'Total Meshes': total_meshes,
            'Connectivity Index': connectivity_index,
            'LMF': lmf,
            'Endpoints/Connectivity': total_endpoints / safe_conn,
            'Nodes/Connectivity': total_nodes / safe_conn,
            'Meshes/Connectivity': total_meshes / safe_conn
        }

        return summary, component_details