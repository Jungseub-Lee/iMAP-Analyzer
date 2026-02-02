# src/gui.py
import sys
import os
import datetime
import re
import pandas as pd
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QSlider, QVBoxLayout, QWidget,
    QPushButton, QHBoxLayout, QMessageBox, QFileDialog, 
    QComboBox, QLineEdit, QApplication, QGroupBox
)
from PyQt5.QtCore import Qt
from skimage import morphology
from skimage.morphology import dilation, disk

# Import Core Logic
from .core import VascularAnalyzer

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # State Variables
        self.current_index = 0
        self.image_files = []
        self.saved_parameters = []
        self.vessel_color = 2
        self.save_path = ""
        self.base_filename = "HUVEC"
        
        # Image Containers
        self.original_image_bgr = None 
        self.image = None              
        
        # Manual Edit States
        self.flood_fill_applied = False
        self.binary_image_final = None 
        self.is_analysis_started = False

        self.initUI()

    def initUI(self):
        self.setWindowTitle("iMAP Analyzer v3.6 (Perfect Dashed & Full Contours)")
        self.setGeometry(100, 100, 1600, 950)
        
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # 1. Top Configuration Panel
        config_group = QGroupBox("Project Configuration")
        config_layout = QVBoxLayout()
        
        row1 = QHBoxLayout()
        self.line_input_path = QLineEdit()
        self.line_input_path.setPlaceholderText("Select Image Folder...")
        self.line_input_path.setReadOnly(True)
        btn_browse_input = QPushButton("Browse...")
        btn_browse_input.clicked.connect(self.browse_input_folder)
        row1.addWidget(QLabel("Input Images:"))
        row1.addWidget(self.line_input_path)
        row1.addWidget(btn_browse_input)
        
        row2 = QHBoxLayout()
        self.line_output_path = QLineEdit()
        self.line_output_path.setPlaceholderText("Select Save Folder...")
        self.line_output_path.setReadOnly(True)
        btn_browse_output = QPushButton("Browse...")
        btn_browse_output.clicked.connect(self.browse_output_folder)
        row2.addWidget(QLabel("Save Results:"))
        row2.addWidget(self.line_output_path)
        row2.addWidget(btn_browse_output)
        
        row3 = QHBoxLayout()
        self.line_project_name = QLineEdit("HUVEC")
        self.btn_start = QPushButton("Load Images & Start Analysis")
        self.btn_start.setStyleSheet("font-weight: bold; background-color: #d1e7dd;")
        self.btn_start.clicked.connect(self.start_analysis_session)
        
        row3.addWidget(QLabel("Project Name:"))
        row3.addWidget(self.line_project_name)
        row3.addWidget(self.btn_start)
        
        config_layout.addLayout(row1)
        config_layout.addLayout(row2)
        config_layout.addLayout(row3)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # 2. Visualization Canvas
        self.figure, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        # 3. Controls
        self.controls_widget = QWidget()
        controls_layout = QVBoxLayout(self.controls_widget)
        
        hbox_chan = QHBoxLayout()
        hbox_chan.addWidget(QLabel("Vessel Color Channel:"))
        self.combo_channel = QComboBox()
        self.combo_channel.addItems(["Blue", "Green", "Red"])
        self.combo_channel.setCurrentIndex(2) 
        self.combo_channel.currentIndexChanged.connect(self.change_channel)
        hbox_chan.addWidget(self.combo_channel)
        controls_layout.addLayout(hbox_chan)
        
        self.params = {'thresh': 10, 'white_min': 100, 'black_min': 200, 'blur': 3, 'alpha': 0.4}
        
        self.add_slider("Threshold", 0, 255, self.params['thresh'], 'thresh', controls_layout)
        self.add_slider("Remove Small Objects (White)", 0, 10000, self.params['white_min'], 'white_min', controls_layout)
        self.add_slider("Fill Holes (Black)", 0, 10000, self.params['black_min'], 'black_min', controls_layout)
        self.add_slider("Gaussian Blur", 0, 20, self.params['blur'], 'blur', controls_layout)
        self.add_slider("Overlay Opacity %", 0, 100, int(self.params['alpha']*100), 'alpha', controls_layout)
        
        btn_layout = QHBoxLayout()
        btn_reset = QPushButton("Reset Noise Removal")
        btn_reset.clicked.connect(self.reset_flood_fill)
        self.btn_next = QPushButton("Next Image / Save & Finish")
        self.btn_next.clicked.connect(self.next_image)
        
        btn_layout.addWidget(btn_reset)
        btn_layout.addWidget(self.btn_next)
        controls_layout.addLayout(btn_layout)
        
        layout.addWidget(self.controls_widget)
        self.controls_widget.setEnabled(False)

    # --- Utility Functions ---
    def get_custom_directory(self, title):
        dialog = QFileDialog(self, title)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.resize(900, 600)
        if dialog.exec_():
            return dialog.selectedFiles()[0]
        return None
        
    def sanitize_sheet_name(self, name):
        clean_name = re.sub(r'[\[\]:*?/\\]', '', name)
        if len(clean_name) > 31:
            clean_name = clean_name[:31]
        return clean_name

    def browse_input_folder(self):
        folder = self.get_custom_directory("Select Image Folder")
        if folder:
            self.line_input_path.setText(folder)

    def browse_output_folder(self):
        folder = self.get_custom_directory("Select Save Folder")
        if folder:
            self.line_output_path.setText(folder)

    def start_analysis_session(self):
        input_dir = self.line_input_path.text()
        output_dir = self.line_output_path.text()
        proj_name = self.line_project_name.text().strip()
        
        if not input_dir or not os.path.exists(input_dir):
            QMessageBox.warning(self, "Error", "Please select a valid Input Folder.")
            return
        if not output_dir or not os.path.exists(output_dir):
            QMessageBox.warning(self, "Error", "Please select a valid Save Folder.")
            return
        if not proj_name:
            QMessageBox.warning(self, "Error", "Please enter a Project Name.")
            return

        valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
        self.image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])
        
        if not self.image_files:
            QMessageBox.warning(self, "Error", "No image files found.")
            return
            
        self.image_folder = input_dir
        self.base_filename = proj_name
        today = datetime.datetime.today().strftime("%y%m%d")
        self.save_path = os.path.join(output_dir, f"{today} - {self.base_filename}")
        os.makedirs(self.save_path, exist_ok=True)
        
        self.current_index = 0
        self.saved_parameters = []
        self.is_analysis_started = True
        self.controls_widget.setEnabled(True)
        
        self.line_input_path.setEnabled(False)
        self.line_output_path.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_start.setText("Analysis In Progress...")
        
        self.load_image()

    # --- Core Interaction ---
    def load_image(self):
        if self.current_index >= len(self.image_files): return
        
        path = os.path.join(self.image_folder, self.image_files[self.current_index])
        
        self.original_image_bgr = cv2.imread(path)
        if self.original_image_bgr is None: return
        
        self.image = self.original_image_bgr[:, :, self.vessel_color]
        
        self.flood_fill_applied = False
        self.binary_image_final = None
        
        self.process_current_image()
        self.update_display()

    def process_current_image(self):
        self.skel_dilated, self.binary, self.skel_raw = VascularAnalyzer.preprocess_image(
            self.image, 
            self.params['thresh'], 
            self.params['white_min'], 
            self.params['black_min'], 
            self.params['blur']
        )

    def update_display(self):
        if not self.is_analysis_started: return

        if self.flood_fill_applied and self.binary_image_final is not None:
            display_binary = self.binary_image_final
            skel_raw = morphology.skeletonize(display_binary)
            display_skel = dilation(skel_raw, disk(2))
        else:
            display_binary = self.binary
            display_skel = self.skel_dilated

        self.axes[0].clear()
        self.axes[0].imshow(self.image, cmap='gray')
        self.axes[0].imshow(display_binary, cmap='jet', alpha=self.params['alpha'])
        self.axes[0].set_title(f"Binary: {self.image_files[self.current_index]} ({self.current_index+1}/{len(self.image_files)})")
        self.axes[0].axis('off')

        self.axes[1].clear()
        self.axes[1].imshow(self.image, cmap='gray')
        self.axes[1].imshow(display_skel, cmap='hot', alpha=self.params['alpha'])
        self.axes[1].set_title("Skeleton Network")
        self.axes[1].axis('off')
        
        self.canvas.draw()

    def add_slider(self, label_text, min_val, max_val, init_val, param_key, layout):
        hbox = QHBoxLayout()
        lbl = QLabel(f"{label_text}: {init_val}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(init_val)
        
        def update_val(val):
            lbl.setText(f"{label_text}: {val}")
            
        def release_slider():
            val = slider.value()
            if param_key == 'alpha':
                self.params[param_key] = val / 100.0
            else:
                self.params[param_key] = val
                self.process_current_image()
            self.update_display()

        slider.valueChanged.connect(update_val)
        slider.sliderReleased.connect(release_slider)
        
        hbox.addWidget(lbl)
        hbox.addWidget(slider)
        layout.addLayout(hbox)

    def change_channel(self, idx):
        self.vessel_color = idx
        if self.is_analysis_started:
            self.load_image()

    def on_canvas_click(self, event):
        if not self.is_analysis_started: return
        if event.inaxes != self.axes[0] or event.xdata is None: return
        x, y = int(event.xdata), int(event.ydata)
        
        if self.binary_image_final is None:
            work_img = (self.binary.astype(np.uint8)) * 255
        else:
            work_img = (self.binary_image_final.astype(np.uint8)) * 255
            
        h, w = work_img.shape
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(work_img, mask, (x, y), 0)
        
        self.binary_image_final = (work_img > 0)
        self.flood_fill_applied = True
        self.update_display()

    def reset_flood_fill(self):
        self.flood_fill_applied = False
        self.binary_image_final = None
        self.update_display()

    def apply_colored_overlay(self, base_bgr, mask, color_bgr, alpha):
        """Applies color overlay on original BGR image."""
        overlay = base_bgr.copy()
        roi = overlay[mask]
        if roi.size == 0: return overlay
        color_layer = np.full_like(roi, color_bgr)
        blended = cv2.addWeighted(roi, 1 - alpha, color_layer, alpha, 0)
        overlay[mask] = blended
        return overlay

    def draw_uniform_dashed_lines(self, img, contours, color, thickness=2, dash_len=10, gap_len=5):
        """
        Draws uniform dashed lines along contours using vector interpolation.
        This fixes the 'irregular dash' issue caused by point density.
        """
        for cnt in contours:
            # Flatten to simpler point list
            pts = cnt.reshape(-1, 2)
            
            # Skip too short
            if len(pts) < 2: continue
            
            total_dist = 0.0
            
            for i in range(len(pts) - 1):
                p1 = pts[i]
                p2 = pts[i+1]
                
                # Vector calc
                vec = p2 - p1
                seg_dist = np.linalg.norm(vec)
                
                if seg_dist == 0: continue
                
                unit_vec = vec / seg_dist
                
                # Walk along the segment
                current_walk = 0.0
                while current_walk < seg_dist:
                    # Current phase in the dash pattern (0 to dash+gap)
                    phase = (total_dist + current_walk) % (dash_len + gap_len)
                    
                    if phase < dash_len:
                        # We are in drawing mode
                        dist_to_end_dash = dash_len - phase
                        dist_to_end_seg = seg_dist - current_walk
                        
                        draw_amount = min(dist_to_end_dash, dist_to_end_seg)
                        
                        start_pt = p1 + unit_vec * current_walk
                        end_pt = start_pt + unit_vec * draw_amount
                        
                        cv2.line(img, tuple(start_pt.astype(int)), tuple(end_pt.astype(int)), color, thickness)
                        
                        current_walk += draw_amount
                    else:
                        # We are in gap mode
                        dist_to_end_gap = (dash_len + gap_len) - phase
                        dist_to_end_seg = seg_dist - current_walk
                        
                        skip_amount = min(dist_to_end_gap, dist_to_end_seg)
                        current_walk += skip_amount
                
                total_dist += seg_dist

    def save_overlays_high_res(self, final_binary, final_skel, idx):
        base_alpha = self.params['alpha']
        base_bgr = self.original_image_bgr.copy()
        
        # =========================================================
        # 1. Binary Overlay (Red Mask + Yellow Dashed Boundary)
        # =========================================================
        # 1.1 Red Mask (High Transparency for Visibility)
        overlay_bin = self.apply_colored_overlay(base_bgr, final_binary > 0, (0, 0, 255), base_alpha * 0.7)
        
        # 1.2 Yellow Dashed Boundary (0, 255, 255)
        bin_uint8 = (final_binary.astype(np.uint8)) * 255
        
        # [CRITICAL CHANGE] Use RETR_LIST to get ALL contours (Outer + Inner Holes)
        contours, _ = cv2.findContours(bin_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw uniform dashed lines
        self.draw_uniform_dashed_lines(overlay_bin, contours, color=(0, 255, 255), thickness=2, dash_len=10, gap_len=5)
        
        cv2.imwrite(os.path.join(self.save_path, f"{self.base_filename}-{idx}-overlay_binary.png"), overlay_bin)
        
        # =========================================================
        # 2. Skeleton Overlay (Bright Red)
        # =========================================================
        thick_skel = dilation(final_skel, disk(4))
        overlay_skel = self.apply_colored_overlay(base_bgr, thick_skel > 0, (0, 0, 255), 0.8) 
        cv2.imwrite(os.path.join(self.save_path, f"{self.base_filename}-{idx}-overlay_skeleton.png"), overlay_skel)

        # =========================================================
        # 3. Topology Overlay (Red Skel + Blue Branch + Yellow End)
        # =========================================================
        topo_final = base_bgr.copy()
        
        # 3.1 Skeleton: Bright Red (0, 0, 255)
        topo_final = self.apply_colored_overlay(topo_final, thick_skel > 0, (0, 0, 255), 0.8)
        
        # Get Features
        _, endpoints = VascularAnalyzer.find_endpoints(final_skel)
        branchpoints = VascularAnalyzer.find_branchpoints(final_skel)
        
        thick_ends = dilation(endpoints, disk(8))
        thick_branches = dilation(branchpoints, disk(8))
        
        # 3.2 Branch Points: Bright Blue (255, 0, 0)
        topo_final[thick_branches > 0] = (255, 0, 0)
        
        # 3.3 Endpoints: Bright Yellow (0, 255, 255)
        topo_final[thick_ends > 0] = (0, 255, 255)
        
        cv2.imwrite(os.path.join(self.save_path, f"{self.base_filename}-{idx}-overlay_topology.png"), topo_final)

    def next_image(self):
        if not self.is_analysis_started: return

        final_binary = self.binary_image_final if (self.flood_fill_applied and self.binary_image_final is not None) else self.binary
        final_binary = final_binary > 0
        final_skel = morphology.skeletonize(final_binary)
        
        save_idx = self.current_index + 1
        
        bin_path = os.path.join(self.save_path, f"{self.base_filename}-{save_idx}-binary.png")
        skel_path = os.path.join(self.save_path, f"{self.base_filename}-{save_idx}-skeleton.png")
        cv2.imwrite(bin_path, (final_binary * 255).astype(np.uint8))
        cv2.imwrite(skel_path, (final_skel * 255).astype(np.uint8))
        
        self.save_overlays_high_res(final_binary, final_skel, save_idx)

        self.saved_parameters.append({
            'filename': self.image_files[self.current_index],
            'bin_path': bin_path,
            'skel_path': skel_path,
            **self.params
        })
        
        self.current_index += 1
        if self.current_index < len(self.image_files):
            self.load_image()
        else:
            self.run_batch_analysis()

    def run_batch_analysis(self):
        self.btn_next.setText("Calculating Full Topology... Do Not Close")
        self.btn_next.setEnabled(False)
        QApplication.processEvents()
        
        summary_results = []
        all_mesh_data = {} 
        
        for data in self.saved_parameters:
            binary = cv2.imread(data['bin_path'], cv2.IMREAD_GRAYSCALE) > 127
            skeleton = cv2.imread(data['skel_path'], cv2.IMREAD_GRAYSCALE) > 127
            
            summary, detail_list = VascularAnalyzer.analyze_network_detailed(skeleton, binary, filename=data['filename'])
            summary_results.append(summary)
            
            if detail_list:
                raw_name = data['filename']
                sheet_name = self.sanitize_sheet_name(raw_name)
                
                counter = 1
                base_sheet_name = sheet_name
                while sheet_name in all_mesh_data:
                    sheet_name = f"{base_sheet_name[:28]}_{counter}"
                    counter += 1
                
                all_mesh_data[sheet_name] = detail_list

        excel_path = os.path.join(self.save_path, f"{self.base_filename}_Results.xlsx")
        
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df_summary = pd.DataFrame(summary_results)
                df_summary.to_excel(writer, sheet_name='Total Data', index=False)
                
                for sheet_name, d_list in all_mesh_data.items():
                    df_detail = pd.DataFrame(d_list)
                    if 'Area (Binary px)' in df_detail.columns:
                        df_detail = df_detail.sort_values(by='Area (Binary px)', ascending=False)
                    df_detail.to_excel(writer, sheet_name=sheet_name, index=False)
                    
            QMessageBox.information(self, "Analysis Complete", f"Saved results to:\n{excel_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save Excel file.\nError: {str(e)}")
            
        self.close()