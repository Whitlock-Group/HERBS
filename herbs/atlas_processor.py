import os


import pickle
import nrrd
import csv
import nibabel as nib
import numpy as np
import pandas as pd
import cv2
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import pyqtgraph.opengl as gl

from .uuuuuu import read_qss_file, make_contour_img, read_excel_file, hex2rgb
from .obj_items import render_volume, render_small_volume
from .atlas_loader import process_atlas_raw_data, AtlasLoader, check_data_path_and_load


class CustomerAtlasWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(float)
    error_occur = pyqtSignal(str)

    def __init__(self):
        super(CustomerAtlasWorker, self).__init__()
        self.saving_folder = None
        self.data_local = None
        self.segmentation_local = None
        self.label_local = None
        self.mask_local = None
        self.b_val = None
        self.l_val = None
        self.vox_size = None
        self.factor = 2
        self.axis_info = None

    def set_data(
        self,
        saving_folder,
        data_local,
        segmentation_local,
        label_local,
        axis_info,
        b_val,
        vox_size,
        mask_local=None,
        factor=2,
    ):
        self.saving_folder = saving_folder
        self.data_local = data_local
        self.segmentation_local = segmentation_local
        self.label_local = label_local
        self.mask_local = mask_local
        self.b_val = b_val
        self.vox_size = vox_size
        self.axis_info = axis_info
        self.factor = factor

    def progress_control(self, total_count):
        self.progress.emit(total_count)

    def run(self):
        self.progress.emit(1)
        if self.vox_size < 1e-4:
            self.error_occur.emit("Please set voxel size.")
            return
        print(self.saving_folder)
        print(self.label_local)
        df, msg = read_excel_file(os.path.join(self.saving_folder, self.label_local))
        if msg is not None:
            self.error_occur.emit(msg)
            return

        reformat_keys = list(df.columns)
        for i in range(len(reformat_keys)):
            reformat_keys[i] = reformat_keys[i].lower()
        df.columns = reformat_keys
        self.progress.emit(2)
        try:
            da_labels = df["name"].values
            self.progress.emit(3)

            da_short_label = df["acronym"].values

        except KeyError:
            self.error_occur.emit('Label file missing columns "name" or "acronym".')
            return
        self.progress.emit(4)

        try:
            levels = []
            structure_id_path = df["structure_id_path"].values
            for i in range(len(structure_id_path)):
                da_path = structure_id_path[i]
                da_path_split = da_path.split("/")
                for j in np.arange(len(da_path_split))[::-1]:
                    if da_path_split[j] == "":
                        da_path_split.pop(j)
                levels.append(len(da_path_split))
        except KeyError:
            self.error_occur.emit('Label file missing columns "structure_id_path".')
            return
        self.progress.emit(5)

        try:
            hex_colors = df["color_hex_triplet"].values
            rgb_colors = []
            for i in range(len(hex_colors)):
                r, g, b = hex2rgb(hex_colors[i])
                rgb_colors.append([r, g, b])
            rgb_colors = np.asarray(rgb_colors)
        except KeyError:
            rgb_colors = []
            for i in range(len(da_short_label)):
                r, g, b = np.random.randint(0, 255, 3)
                rgb_colors.append([r, g, b])
            rgb_colors = np.asarray(rgb_colors)
        self.progress.emit(6)

        try:
            parent = df["parent_id"].values.astype(int)
            ids = df["id"].values.astype(int)
        except KeyError:
            self.error_occur.emit(
                'Label file missing columns "parent_structure_id" or "id".'
            )
            return

        label_info = {
            "index": ids,
            "label": da_labels,
            "parent": parent,
            "abbrev": da_short_label,
            "color": rgb_colors,
            "level_indicator": levels,
        }

        with open(os.path.join(self.saving_folder, "atlas_labels.pkl"), "wb") as handle:
            pickle.dump(label_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.progress.emit(9)

        # load atlas
        atlas_data, success = check_data_path_and_load(
            os.path.join(self.saving_folder, self.data_local)
        )
        if not success:
            self.error_occur.emit(
                "Failed to load atlas data file. Currently only support for .nii and .nrrd."
            )
            return
        atlas_data = atlas_data - np.min(atlas_data)
        atlas_size = atlas_data.shape
        if np.any(np.ravel(atlas_size) < self.factor):
            self.error_occur.emit("Factor can not be larger than atlas size.")
        self.progress.emit(14)
        # load segmentation data
        segmentation_data, success = check_data_path_and_load(
            os.path.join(self.saving_folder, self.segmentation_local)
        )
        if not success:
            self.error_occur.emit(
                "Failed to load segmentation data file. Currently only support for .nii and .nrrd."
            )
            return
        self.progress.emit(19)
        if not (segmentation_data.shape == atlas_data.shape):
            self.error_occur.emit("Atlas shape is different than segmentation shape.")
            return

        if self.mask_local is not None:
            mask_data, success = check_data_path_and_load(
                os.path.join(self.saving_folder, self.mask_local)
            )
            if not success:
                self.error_occur.emit(
                    "Failed to load mask data file. Currently only support for .nii and .nrrd."
                )
                return
            self.progress.emit(23)
            if not mask_data.shape == atlas_data.shape:
                self.error_occur.emit("Atlas shape is different than mask shape.")
                return

            # make segmentation with mask
            progress_step = np.linspace(23, 26, len(mask_data))
            for i in range(len(mask_data)):
                self.progress.emit(progress_step[i])
                segmentation_data[i][mask_data[i] == 0] = 0
            segmentation_data = segmentation_data.astype("int")

            progress_step = np.linspace(26, 28, len(mask_data))
            for i in range(len(mask_data)):
                self.progress.emit(progress_step[i])
                atlas_data[i][mask_data[i] == 0] = 0

        atlas_data = atlas_data / np.max(atlas_data)
        self.progress.emit(30)

        unique_label = np.unique(segmentation_data)
        n_unique_labels = len(unique_label)
        self.progress.emit(35)

        if self.axis_info["direction_change"][0]:
            segmentation_data = segmentation_data[::-1, :, :]
            atlas_data = atlas_data[::-1, :, :]
        self.progress.emit(36)
        if self.axis_info["direction_change"][1]:
            segmentation_data = segmentation_data[:, ::-1, :]
            atlas_data = atlas_data[:, ::-1, :]
        self.progress.emit(37)
        if self.axis_info["direction_change"][2]:
            segmentation_data = segmentation_data[:, :, ::-1]
            atlas_data = atlas_data[:, :, ::-1]
        self.progress.emit(38)

        segmentation_data = np.transpose(segmentation_data, self.axis_info["to_HERBS"])
        segmentation_data = segmentation_data.astype(int)
        # print(segmentation_data.shape)
        self.progress.emit(39)

        segment = {"data": segmentation_data, "unique_label": unique_label}

        outfile = open(os.path.join(self.saving_folder, "segment_pre_made.pkl"), "wb")
        pickle.dump(segment, outfile)
        outfile.close()
        self.progress.emit(42)

        new_atlas_shape = atlas_data.shape
        b_val = np.ravel(self.b_val)[np.array(self.axis_info["to_HERBS"])]
        if b_val[0] == 0:
            b_val[0] = int(new_atlas_shape[0] / 2)
        self.progress.emit(43)
        if b_val[1] == 0:
            b_val[1] = int(new_atlas_shape[1] / 2)
        self.progress.emit(44)

        for i in range(3):
            if self.axis_info["direction_change"][i]:
                b_val[i] = new_atlas_shape[i] - 1 - b_val[i]

        if b_val[2] == 0:
            b_val[2] = new_atlas_shape[2] - 1 - b_val[2]

        atlas_info = [
            {
                "name": "anterior",
                "values": np.arange(atlas_data.shape[0]) * self.vox_size,
                "units": "um",
            },
            {
                "name": "dorsal",
                "values": np.arange(atlas_data.shape[1]) * self.vox_size,
                "units": "um",
            },
            {
                "name": "right",
                "values": np.arange(atlas_data.shape[2]) * self.vox_size,
                "units": "um",
            },
            {"vxsize": self.vox_size, "Bregma": [b_val[0], b_val[1], b_val[2]]},
        ]
        self.progress.emit(45)
        atlas = {"data": atlas_data, "info": atlas_info}

        outfile = open(os.path.join(self.saving_folder, "atlas_pre_made.pkl"), "wb")
        pickle.dump(atlas, outfile)
        outfile.close()
        self.progress.emit(50)

        mesh_data = render_volume(
            atlas_data, self.saving_folder, factor=self.factor, level=0.1
        )
        self.progress.emit(55)

        mesh_path = os.path.join(self.saving_folder, "meshes")
        if not os.path.exists(mesh_path):
            os.mkdir(mesh_path)

        progress_step = np.linspace(55, 68, n_unique_labels)
        for i in range(n_unique_labels):
            self.progress.emit(progress_step[i])
            label_id = int(unique_label[i])

            if label_id == 0:
                continue
            render_small_volume(
                label_id,
                mesh_path,
                atlas_data,
                segmentation_data,
                factor=self.factor,
                level=0.1,
            )

        small_mesh_list = {}
        file_list = os.listdir(mesh_path)
        for da_file in file_list:
            file_name = os.path.basename(da_file)
            da_name, file_extension = os.path.splitext(file_name)
            if file_extension == ".pkl":
                infile = open(os.path.join(mesh_path, da_file), "rb")
                md = pickle.load(infile)
                infile.close()

                small_mesh_list[str(da_name)] = md

        self.progress.emit(69)
        outfile = open(
            os.path.join(self.saving_folder, "atlas_small_meshdata.pkl"), "wb"
        )
        pickle.dump(small_mesh_list, outfile)
        outfile.close()
        self.progress.emit(70)

        segment_data_shape = segmentation_data.shape

        sagital_contour_img = np.zeros(segment_data_shape, "i")
        coronal_contour_img = np.zeros(segment_data_shape, "i")
        horizontal_contour_img = np.zeros(segment_data_shape, "i")

        # pre-process boundary ----- todo: change this part as optional
        process_index = np.linspace(70, 78, segment_data_shape[0])
        for i in range(segment_data_shape[0]):
            self.progress.emit(process_index[i])
            da_slice = segmentation_data[i, :, :].copy()
            contour_img = make_contour_img(da_slice)
            sagital_contour_img[i, :, :] = contour_img

        outfile_ct = open(
            os.path.join(self.saving_folder, "sagital_contour_pre_made.pkl"), "wb"
        )
        pickle.dump(sagital_contour_img, outfile_ct)
        outfile_ct.close()
        self.progress.emit(80)

        process_index = np.linspace(80, 88, segment_data_shape[1])
        for i in range(segment_data_shape[1]):
            self.progress.emit(process_index[i])
            da_slice = segmentation_data[:, i, :].copy()
            contour_img = make_contour_img(da_slice)
            coronal_contour_img[:, i, :] = contour_img

        outfile_ct = open(
            os.path.join(self.saving_folder, "coronal_contour_pre_made.pkl"), "wb"
        )
        pickle.dump(coronal_contour_img, outfile_ct)
        outfile_ct.close()
        self.progress.emit(90)

        process_index = np.linspace(90, 98, segment_data_shape[2])
        for i in range(segment_data_shape[2]):
            self.progress.emit(process_index[i])
            da_slice = segmentation_data[:, :, i].copy()
            contour_img = make_contour_img(da_slice)
            horizontal_contour_img[:, :, i] = contour_img

        outfile_ct = open(
            os.path.join(self.saving_folder, "horizontal_contour_pre_made.pkl"), "wb"
        )
        pickle.dump(horizontal_contour_img, outfile_ct)
        outfile_ct.close()

        # saving atlas axis changing information
        self.axis_info["size"] = tuple(atlas_size)
        outfile_axis = open(
            os.path.join(self.saving_folder, "atlas_axis_info.pkl"), "wb"
        )
        pickle.dump(self.axis_info, outfile_axis)
        outfile_axis.close()

        self.progress.emit(100)

        self.finished.emit()


class AtlasProcessor(QDialog):
    def __init__(self):
        super().__init__()
        qss_file_name = "qss/dialogs.qss"
        qss_style_sheet = read_qss_file(qss_file_name)
        self.setStyleSheet(qss_style_sheet)
        self.setWindowTitle("Atlas Processor")

        self.folder_path = None
        self.data_local = None
        self.segmentation_local = None
        self.mask_local = None
        self.label_local = None
        self.bregma_coord = [0, 0, 0]
        self.lambda_coord = [0, 0, 0]
        self.voxel_size = 0
        self.factor_val = 2
        self.axis_info = None
        self.directions = [0, 0, 0]
        self.dim_group = [-1, -1, -1]
        self.info_flag = True

        self.thread = None
        self.worker = None

        self.selector_vals = [
            "L.H. --> R.H.",
            "R.H. --> L.H.",
            "Post. --> Ant.",
            "Ant. --> Post.",
            "Sup. --> Inf.",
            "Inf. --> Sup.",
        ]
        self.group_maps = np.array([1, 1, 2, 2, 3, 3])

        layout = QGridLayout(self)

        box_label_style = read_qss_file("qss/box_label.qss")
        data_label = QLabel("Volume File:")
        self.data_btn = QPushButton("Select File")
        self.data_btn.setAutoDefault(False)
        self.data_btn.setFocus(False)
        self.data_line = QLabel()
        self.data_line.setStyleSheet(box_label_style)

        seg_label = QLabel("Segmentation File: ")
        self.seg_btn = QPushButton("Select File")
        self.seg_btn.setAutoDefault(False)
        self.seg_btn.setFocus(False)
        self.seg_line = QLabel()
        self.seg_line.setStyleSheet(box_label_style)

        mask_label = QLabel("Mask File (optional): ")
        self.mask_btn = QPushButton("Select File")
        self.mask_btn.setAutoDefault(False)
        self.mask_btn.setFocus(False)
        self.mask_line = QLabel()
        self.mask_line.setStyleSheet(box_label_style)

        labinf_label = QLabel("Label Information File:")
        self.labinf_btn = QPushButton("Select File")
        self.labinf_btn.setAutoDefault(False)
        self.labinf_btn.setFocus(False)
        self.labinf_line = QLabel()
        self.labinf_line.setStyleSheet(box_label_style)

        valid_input = QIntValidator(0, 99999)

        float_input = QRegExpValidator(QRegExp(r"[0-9].+"))
        # float_input = QDoubleValidator(0.0, 100.0, 6)

        bregma_label = QLabel("Bregma Coordinates (voxel): ")
        self.bregma_input1 = QLineEdit("0")
        self.bregma_input1.setValidator(valid_input)
        self.bregma_input2 = QLineEdit("0")
        self.bregma_input2.setValidator(valid_input)
        self.bregma_input3 = QLineEdit("0")
        self.bregma_input3.setValidator(valid_input)

        lambda_label = QLabel("Lambda Coordinates (voxel): ")
        self.lambda_input1 = QLineEdit("0")
        self.lambda_input1.setValidator(valid_input)
        self.lambda_input2 = QLineEdit("0")
        self.lambda_input2.setValidator(valid_input)
        self.lambda_input3 = QLineEdit("0")
        self.lambda_input3.setValidator(valid_input)

        vox_size_label = QLabel("Voxel Size (um): ")
        self.vox_size_input1 = QLineEdit("0")
        self.vox_size_input1.setValidator(float_input)

        factor_label = QLabel("Factor (voxel): ")
        self.factor_input1 = QLineEdit("2")
        self.factor_input1.setValidator(QIntValidator(2, 99999))

        dim_selector_label = QLabel("CSys Selector: ")

        self.x_axis_combo = QComboBox()
        self.x_axis_combo.addItem("x-axis: ")
        self.x_axis_combo.addItems(self.selector_vals)
        self.x_axis_combo.currentIndexChanged.connect(
            lambda: self.dim_combo_changed("x")
        )

        self.y_axis_combo = QComboBox()
        self.y_axis_combo.addItem("y-axis: ")
        self.y_axis_combo.addItems(self.selector_vals)
        self.y_axis_combo.currentIndexChanged.connect(
            lambda: self.dim_combo_changed("y")
        )

        self.z_axis_combo = QComboBox()
        self.z_axis_combo.addItem("z-axis: ")
        self.z_axis_combo.addItems(self.selector_vals)
        self.z_axis_combo.currentIndexChanged.connect(
            lambda: self.dim_combo_changed("z")
        )

        self.process_btn = QPushButton("Start Process")
        self.process_btn.setAutoDefault(False)
        self.process_btn.setFocus(False)
        self.process_info = QLabel(
            "The whole process takes some time. \n"
            "This window will be closed automatically when processing finished."
        )

        self.progress = QProgressBar(self)
        self.progress.setMinimumWidth(100)
        self.progress.setTextVisible(False)
        self.progress_label = QLabel("0 %")
        self.progress_label.setFixedWidth(50)

        progress_wrap = QFrame()
        pw_layout = QHBoxLayout(progress_wrap)
        pw_layout.setSpacing(5)
        pw_layout.setContentsMargins(0, 0, 0, 0)
        pw_layout.setAlignment(Qt.AlignRight)
        pw_layout.addWidget(self.progress)
        pw_layout.addWidget(self.progress_label)

        layout.addWidget(data_label, 0, 0, 1, 1)
        layout.addWidget(self.data_btn, 0, 1, 1, 1)
        layout.addWidget(self.data_line, 0, 2, 1, 2)
        layout.addWidget(seg_label, 1, 0, 1, 1)
        layout.addWidget(self.seg_btn, 1, 1, 1, 1)
        layout.addWidget(self.seg_line, 1, 2, 1, 2)
        layout.addWidget(mask_label, 2, 0, 1, 1)
        layout.addWidget(self.mask_btn, 2, 1, 1, 1)
        layout.addWidget(self.mask_line, 2, 2, 1, 2)
        layout.addWidget(labinf_label, 3, 0, 1, 1)
        layout.addWidget(self.labinf_btn, 3, 1, 1, 1)
        layout.addWidget(self.labinf_line, 3, 2, 1, 2)
        layout.addWidget(bregma_label, 4, 0, 1, 1)
        layout.addWidget(self.bregma_input1, 4, 1, 1, 1)
        layout.addWidget(self.bregma_input2, 4, 2, 1, 1)
        layout.addWidget(self.bregma_input3, 4, 3, 1, 1)
        layout.addWidget(lambda_label, 5, 0, 1, 1)
        layout.addWidget(self.lambda_input1, 5, 1, 1, 1)
        layout.addWidget(self.lambda_input2, 5, 2, 1, 1)
        layout.addWidget(self.lambda_input3, 5, 3, 1, 1)
        layout.addWidget(vox_size_label, 6, 0, 1, 1)
        layout.addWidget(self.vox_size_input1, 6, 1, 1, 1)
        layout.addWidget(factor_label, 6, 2, 1, 1)
        layout.addWidget(self.factor_input1, 6, 3, 1, 1)

        layout.addWidget(dim_selector_label, 7, 0, 1, 1)
        layout.addWidget(self.x_axis_combo, 7, 1, 1, 1)
        layout.addWidget(self.y_axis_combo, 7, 2, 1, 1)
        layout.addWidget(self.z_axis_combo, 7, 3, 1, 1)

        layout.addWidget(self.process_info, 8, 0, 1, 4)
        layout.addWidget(self.process_btn, 9, 0, 1, 4)
        layout.addWidget(progress_wrap, 10, 0, 1, 4)

        # connect all buttons
        self.data_btn.clicked.connect(self.get_data_file)
        self.seg_btn.clicked.connect(self.get_seg_file)
        self.mask_btn.clicked.connect(self.get_mask_file)
        self.labinf_btn.clicked.connect(self.get_info_file)
        self.process_btn.clicked.connect(self.process_data_called)
        self.bregma_input1.textChanged.connect(self.bregma_input1_changed)
        self.bregma_input2.textChanged.connect(self.bregma_input2_changed)
        self.bregma_input3.textChanged.connect(self.bregma_input3_changed)
        self.lambda_input1.textChanged.connect(self.lambda_input1_changed)
        self.lambda_input2.textChanged.connect(self.lambda_input2_changed)
        self.lambda_input3.textChanged.connect(self.lambda_input3_changed)
        self.vox_size_input1.textChanged.connect(self.vox_size_input_changed)
        self.factor_input1.textChanged.connect(self.factor_input_changed)
        self.factor_input1.editingFinished.connect(self.factor_input_changed)

    def dim_combo_changed(self, ax):
        if not self.info_flag:
            self.process_info.setText("")
        if ax == "x":
            self.directions[0] = self.x_axis_combo.currentIndex()
        elif ax == "y":
            self.directions[1] = self.y_axis_combo.currentIndex()
        else:
            self.directions[2] = self.z_axis_combo.currentIndex()

        if np.all(np.ravel(self.directions) != 0):
            dir_groups = self.group_maps[np.array(self.directions) - 1]

            dir_goal = ["Post. --> Ant.", "Inf. --> Sup.", "L.H. --> R.H."]
            direction_change = [False, False, False]
            if self.x_axis_combo.currentText() not in dir_goal:
                direction_change[0] = True
            if self.y_axis_combo.currentText() not in dir_goal:
                direction_change[1] = True
            if self.z_axis_combo.currentText() not in dir_goal:
                direction_change[2] = True

            transpose_order = dir_groups - 1
            transpose_order = transpose_order.tolist()

            if transpose_order == [1, 2, 0]:
                inv_transpose_order = [2, 0, 1]
            elif transpose_order == [2, 0, 1]:
                inv_transpose_order = [1, 2, 0]
            else:
                inv_transpose_order = transpose_order.copy()

            self.axis_info = {
                "to_HERBS": tuple(inv_transpose_order),
                "from_HERBS": tuple(transpose_order),
                "direction_change": tuple(direction_change),
            }

            print(self.axis_info)

    def get_folder_path(self, file_path):
        self.folder_path = os.path.dirname(file_path)

    def get_data_file(self):
        file_options = QFileDialog.Options()
        file_options |= QFileDialog.DontUseNativeDialog
        dlg = QFileDialog()
        if self.folder_path is not None:
            data_path = dlg.getOpenFileName(
                self, "Select Atlas Volume File", self.folder_path, options=file_options
            )
        else:
            data_path = dlg.getOpenFileName(
                self, "Select Atlas Volume File", options=file_options
            )
            self.get_folder_path(data_path[0])
        self.data_local = os.path.basename(data_path[0])
        self.data_line.setText(self.data_local)

    def get_seg_file(self):
        file_options = QFileDialog.Options()
        file_options |= QFileDialog.DontUseNativeDialog
        dlg = QFileDialog()
        if self.folder_path is not None:
            seg_path = dlg.getOpenFileName(
                self, "Select Segmentation File", self.folder_path, options=file_options
            )
        else:
            seg_path = dlg.getOpenFileName(
                self, "Select Segmentation File", options=file_options
            )
            self.get_folder_path(seg_path[0])
        self.segmentation_local = os.path.basename(seg_path[0])
        self.seg_line.setText(self.segmentation_local)

    def get_mask_file(self):
        file_options = QFileDialog.Options()
        file_options |= QFileDialog.DontUseNativeDialog
        dlg = QFileDialog()
        if self.folder_path is not None:
            mask_path = dlg.getOpenFileName(
                self, "Select Mask File", self.folder_path, options=file_options
            )
        else:
            mask_path = dlg.getOpenFileName(
                self, "Select Mask File", options=file_options
            )
            self.get_folder_path(mask_path[0])
        self.mask_local = os.path.basename(mask_path[0])
        self.mask_line.setText(self.mask_local)

    def get_info_file(self):
        file_options = QFileDialog.Options()
        file_options |= QFileDialog.DontUseNativeDialog
        dlg = QFileDialog()
        if self.folder_path is not None:
            info_path = dlg.getOpenFileName(
                self, "Select Label File", self.folder_path, options=file_options
            )
        else:
            info_path = dlg.getOpenFileName(
                self, "Select Label File", options=file_options
            )
            self.get_folder_path(info_path[0])
        self.label_local = os.path.basename(info_path[0])
        self.labinf_line.setText(os.path.basename(self.label_local))

    def bregma_input1_changed(self, text):
        if text == "":
            return
        self.bregma_coord[0] = int(text)

    def bregma_input2_changed(self, text):
        if text == "":
            return
        self.bregma_coord[1] = int(text)

    def bregma_input3_changed(self, text):
        if text == "":
            return
        self.bregma_coord[2] = int(text)

    def lambda_input1_changed(self, text):
        if text == "":
            return
        self.lambda_coord[0] = int(text)

    def lambda_input2_changed(self, text):
        if text == "":
            return
        self.lambda_coord[1] = int(text)

    def lambda_input3_changed(self, text):
        if text == "":
            return
        self.lambda_coord[2] = int(text)

    def vox_size_input_changed(self, text):
        if text == "":
            return
        self.voxel_size = float(text)

    def factor_input_changed(self, text):
        if text == "":
            return
        self.factor_val = int(text)

    def check_empty_file(self):
        if self.data_local is None:
            msg = "Please select atlas data."
            return msg
        if self.segmentation_local is None:
            msg = "Please select segmentation file."
            return msg
        if self.label_local is None:
            msg = "Please select label information file."
            return msg
        return None

    def process_data_called(self):
        if np.any(np.array(self.directions) == 0):
            self.info_flag = False
            self.process_info.setText("Please select directions for each axis.")
            return
        dir_groups = self.group_maps[np.array(self.directions) - 1]
        if len(np.unique(dir_groups)) != 3:
            self.info_flag = False
            self.process_info.setText("Axis directions can not duplicate.")
            return

        msg = self.check_empty_file()
        if msg is not None:
            self.process_info.setText("Please select file path.")
            return

        dir_goal = ["Post. --> Ant.", "Inf. --> Sup.", "L.H. --> R.H."]
        direction_change = [False, False, False]
        if self.x_axis_combo.currentText() not in dir_goal:
            direction_change[0] = True
        if self.y_axis_combo.currentText() not in dir_goal:
            direction_change[1] = True
        if self.z_axis_combo.currentText() not in dir_goal:
            direction_change[2] = True

        transpose_order = dir_groups - 1

        self.process_btn.setVisible(False)
        self.process_info.setText(
            "The whole process takes some time. \n"
            "This window will be closed automatically when processing finished."
        )

        self.thread = QThread()
        self.worker = CustomerAtlasWorker()
        self.worker.set_data(
            self.folder_path,
            self.data_local,
            self.segmentation_local,
            self.label_local,
            self.axis_info,
            self.bregma_coord,
            self.voxel_size,
            self.mask_local,
            self.factor_val,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finish)
        self.worker.error_occur.connect(self.report_error)
        # self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.report_progress)
        self.thread.start()

    def report_error(self, msg):
        self.thread.quit()
        self.process_btn.setVisible(True)
        self.progress.setValue(int(0))
        self.progress_label.setText("%.02f %%" % 0)
        self.process_info.setText(msg)

    def report_progress(self, val):
        val = np.round(val, 2)
        self.progress.setValue(int(val))
        # self.progress.setFormat("%.02f %%" % val)
        self.progress_label.setText("%.02f %%" % val)

    def on_finish(self):
        self.thread.quit()
        self.close()
