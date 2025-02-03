import os
import sys
import numpy as np
import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.functions as fn
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from .image_stacks import SliceStack
from .slice_stacks import SliceStacks
from .label_tree import LabelTree
from .uuuuuu import read_qss_file, get_corner_line_from_rect, get_slice_atlas_coord, make_contour_img,  \
    rotate_base_points, rotation_x, rotation_y, rotation_z
from .probe_utiles import get_tilt_sign#, get_direction_rotation


class PageController(QWidget):
    class SignalProxy(QObject):
        sigPageChanged = pyqtSignal(object)

    def __init__(self):
        self._sigprox = PageController.SignalProxy()
        self.sig_page_changed = self._sigprox.sigPageChanged

        QWidget.__init__(self)
        page_control_style = read_qss_file('qss/page_control.qss')
        self.setStyleSheet(page_control_style)

        self.max_val = None

        self.page_slider = QSlider(Qt.Horizontal)
        self.page_slider.setMinimum(0)
        self.page_slider.valueChanged.connect(self.slider_value_changed)

        self.page_label = QLabel()
        self.page_label.setFixedSize(50, 20)

        self.page_left_btn = QPushButton()
        self.page_left_btn.setIcon(QIcon("icons/backward.svg"))
        self.page_left_btn.setIconSize(QSize(16, 16))
        self.page_left_btn.clicked.connect(self.left_btn_clicked)

        self.page_right_btn = QPushButton()
        self.page_right_btn.setIcon(QIcon("icons/forward.svg"))
        self.page_right_btn.setIconSize(QSize(16, 16))
        self.page_right_btn.clicked.connect(self.right_btn_clicked)

        self.page_fast_left_btn = QPushButton()
        self.page_fast_left_btn.setIcon(QIcon("icons/fast_backward.svg"))
        self.page_fast_left_btn.setIconSize(QSize(16, 16))
        self.page_fast_left_btn.clicked.connect(self.fast_left_btn_clicked)

        self.page_fast_right_btn = QPushButton()
        self.page_fast_right_btn.setIcon(QIcon("icons/fast_forward.svg"))
        self.page_fast_right_btn.setIconSize(QSize(16, 16))
        self.page_fast_right_btn.clicked.connect(self.fast_right_btn_clicked)

        page_ctrl_layout = QHBoxLayout()
        page_ctrl_layout.setSpacing(0)
        page_ctrl_layout.setContentsMargins(10, 5, 10, 5)
        page_ctrl_layout.addWidget(self.page_left_btn)
        page_ctrl_layout.addSpacing(10)
        page_ctrl_layout.addWidget(self.page_fast_left_btn)
        page_ctrl_layout.addSpacing(10)
        page_ctrl_layout.addWidget(self.page_slider)
        page_ctrl_layout.addSpacing(5)
        page_ctrl_layout.addWidget(self.page_label)
        page_ctrl_layout.addWidget(self.page_fast_right_btn)
        page_ctrl_layout.addSpacing(10)
        page_ctrl_layout.addWidget(self.page_right_btn)

        self.setLayout(page_ctrl_layout)

    def set_val(self, val):
        self.page_slider.setValue(val)

    def set_max(self, val):
        self.max_val = val
        self.page_slider.setMaximum(val)

    def slider_value_changed(self):
        val = self.page_slider.value()
        self.page_label.setText(str(val))
        self.sig_page_changed.emit(val)

    def left_btn_clicked(self):
        if self.max_val is None:
            return
        val = self.page_slider.value() - 1
        if val < 0:
            val = 0
        self.set_val(val)

    def right_btn_clicked(self):
        if self.max_val is None:
            return
        val = self.page_slider.value() + 1
        if val > self.max_val:
            val = self.max_val
        self.set_val(val)

    def fast_left_btn_clicked(self):
        if self.max_val is None:
            return
        val = self.page_slider.value() - 10
        if val < 0:
            val = 0
        self.set_val(val)

    def fast_right_btn_clicked(self):
        if self.max_val is None:
            return
        val = self.page_slider.value() + 10
        if val > self.max_val:
            val = self.max_val
        self.set_val(val)


class SliceRotation(QWidget):

    class SignalProxy(QObject):
        sigSliceRotated = pyqtSignal(object)  # id

    def __init__(self):
        self._sigprox = SliceRotation.SignalProxy()
        self.sig_slice_rotated = self._sigprox.sigSliceRotated

        QWidget.__init__(self)

        spinbox_line_edit_style = read_qss_file('qss/hidden_line_edit.qss')
        self.prec = 0.01
        self.rot_range = 30
        self.n_steps = int(self.rot_range / self.prec)

        self.h_slider = QSlider(Qt.Horizontal)
        self.h_slider.valueChanged.connect(self.h_slider_changed)
        self.h_slider.setValue(0)
        self.h_slider.setRange(-self.n_steps, self.n_steps)

        self.h_spinbox = QDoubleSpinBox()
        self.h_spinbox.lineEdit().setStyleSheet(spinbox_line_edit_style)
        self.h_spinbox.valueChanged.connect(self.h_spinbox_changed)
        self.h_spinbox.setDecimals(2)
        self.h_spinbox.setValue(0)
        self.h_spinbox.setRange(-self.rot_range, self.rot_range)
        self.h_spinbox.setSingleStep(self.prec)
        self.h_spinbox.setMinimumSize(50, 20)

        self.v_slider = QSlider(Qt.Horizontal)
        self.v_slider.valueChanged.connect(self.v_slider_changed)
        self.v_slider.setValue(0)
        self.v_slider.setRange(-self.n_steps, self.n_steps)

        self.v_spinbox = QDoubleSpinBox()
        self.v_spinbox.lineEdit().setStyleSheet(spinbox_line_edit_style)
        self.v_spinbox.valueChanged.connect(self.v_spinbox_changed)
        self.v_spinbox.setDecimals(2)
        self.v_spinbox.setValue(0)
        self.v_spinbox.setRange(-self.rot_range, self.rot_range)
        self.v_spinbox.setSingleStep(self.prec)
        self.v_spinbox.setMinimumSize(50, 20)

    def set_prec(self, prec):
        self.prec = prec

    def h_slider_changed(self):
        val = self.h_slider.value() * self.prec
        self.h_spinbox.setValue(val)

    def h_spinbox_changed(self):
        val = int(self.h_spinbox.value() / self.prec)
        if self.h_slider.value() != val:
            self.h_slider.setValue(val)
        self.sig_slice_rotated.emit(np.deg2rad([self.h_spinbox.value(), self.v_spinbox.value()]))

    def v_slider_changed(self):
        val = self.v_slider.value() * self.prec
        self.v_spinbox.setValue(val)

    def v_spinbox_changed(self):
        val = int(self.v_spinbox.value() / self.prec)
        if self.v_slider.value() != val:
            self.v_slider.setValue(val)
        self.sig_slice_rotated.emit(np.deg2rad([self.h_spinbox.value(), self.v_spinbox.value()]))


class ImageLabel(QWidget):
    def __init__(self, img, title):
        super(QWidget, self).__init__()
        layout = QHBoxLayout(self)
        self.label3 = QLabel(self)
        self.title = QLabel(title)
        self.pixmap = QPixmap(img)
        self.pixmap = self.pixmap.scaled(QSize(20, 10))
        self.label3.setPixmap(self.pixmap)
        self.label3.setAlignment(Qt.AlignCenter)
        self.title.setMinimumHeight(self.pixmap.height())
        self.title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label3)
        layout.addWidget(self.title)
        self.label3.setStyleSheet('background-color: transparent;')
        self.title.setStyleSheet("""
            background-color: transparent;
            color: black;
            padding: 0px 3px 0px 0px;
        """)
        layout.setSpacing(0)
        layout.addStretch()


class RotationBtn(QPushButton):
    def __init__(self, icon_path):
        super(QPushButton, self).__init__()
        btn_style = read_qss_file('qss/rotation_button.qss')
        self.setStyleSheet(btn_style)
        self.setFixedSize(15, 15)
        self.setIcon(QIcon(icon_path))
        self.setIconSize(QSize(15, 15))


class AtlasView(QObject):
    """
    A collection of user interface elements bound together:


    """
    def __init__(self):
        super(AtlasView, self).__init__()
        QObject.__init__(self)

        self.atlas_data = None
        self.atlas_label = None
        self.label_info = None
        self.atlas_boundary = None
        self.slice_size = None
        self.slice_tb_size_base = 80
        self.coronal_tb_size = None
        self.sagital_tb_size = None
        self.horizontal_tb_size = None
        self.slice_tb_size = None

        self.slice_image_data = None
        self.processing_slice = None
        self.slice_cut = 'Coronal'
        self.slice_width = 0
        self.slice_height = 0
        self.slice_distance = 0
        self.slice_bregma = []
        self.slice_boundary_points = []
        self.slice_info_ready = False

        self.remain_angle = False
        self.coronal_rotated = False
        self.sagittal_rotated = False
        self.horizontal_rotated = False
        self.coronal_rads = (0, 0)
        self.sagittal_rads = (0, 0)
        self.horizontal_rads = (0, 0)

        self.atlas_size = None
        self.anterior_info = None
        self.dorsal_info = None
        self.right_info = None
        self.vox_size_um = None
        self.Bregma = None
        self.rotate_origin_3d = None
        self.origin_3d = None

        self.c_size = None
        self.s_size = None
        self.h_size = None

        self.c_rotm_2d = None
        self.s_rotm_2d = None
        self.h_rotm_2d = None

        self.c_rotm_3d = None
        self.s_rotm_3d = None
        self.h_rotm_3d = None

        self.side_lines = None
        self.corner_points = None

        self.current_coronal_index = None
        self.current_sagital_index = None
        self.current_horizontal_index = None

        self.anchor_coronal_index = None
        self.anchor_sagital_index = None
        self.anchor_horizontal_index = None

        self.has_display_objects = False

        rotation_gb_style = read_qss_file('qss/atlas_view_group_box.qss')
        button_style = read_qss_file('qss/side_bar.qss')
        spinbox_line_edit_style = read_qss_file('qss/hidden_line_edit.qss')

        # self.display_atlas = None
        # self.display_label = None
        # self.scale = None
        # self.interpolate = True
        # self.oy_vector = np.array([0, 1, 0])
        # self.oz_vector = np.array([0, 0, 1])
        # self.ox_vector = np.array([1, 0, 0])

        self.axis = gl.GLAxisItem()
        self.axis.setSize(250, 700, 250)

        self.grid = gl.GLGridItem()
        self.grid.scale(2, 2, 1)

        self.mesh = gl.GLMeshItem(smooth=True, color=[0.5, 0.5, 0.5, 0.2], shader='balloon')
        self.mesh.setGLOptions('additive')

        # 3d things
        plate_color = [0.5, 0.5, 0.5, 0.2]
        self.ap_plate_verts = np.array([[-1, 0, -1], [-1, 0, 1], [1, 0, 1], [1, 0, -1]])
        self.ap_plate_faces = np.array([[0, 1, 2], [0, 2, 3]])
        ap_plate_md = gl.MeshData(vertexes=self.ap_plate_verts, faces=self.ap_plate_faces)
        self.ap_plate_mesh = gl.GLMeshItem(meshdata=ap_plate_md, smooth=False, color=plate_color)
        self.ap_plate_mesh.setGLOptions('additive')

        self.dv_plate_verts = np.array([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]])
        self.dv_plate_faces = np.array([[0, 1, 2], [0, 2, 3]])
        dv_plate_md = gl.MeshData(vertexes=self.dv_plate_verts, faces=self.dv_plate_faces)
        self.dv_plate_mesh = gl.GLMeshItem(meshdata=dv_plate_md, smooth=False, color=plate_color)
        self.dv_plate_mesh.setGLOptions('additive')

        self.ml_plate_verts = np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]])
        self.ml_plate_faces = np.array([[0, 1, 2], [0, 2, 3]])
        ml_plate_md = gl.MeshData(vertexes=self.ml_plate_verts, faces=self.ml_plate_faces)
        self.ml_plate_mesh = gl.GLMeshItem(meshdata=ml_plate_md, smooth=False, color=plate_color)
        self.ml_plate_mesh.setGLOptions('additive')

        # ap - coronal
        self.cimg = SliceStacks()
        self.cpage_ctrl = PageController()
        self.cpage_ctrl.sig_page_changed.connect(self.coronal_slice_page_changed)
        self.clut = pg.HistogramLUTWidget()
        self.clut.setImageItem(self.cimg.img)

        self.simg = SliceStacks()
        self.spage_ctrl = PageController()
        self.spage_ctrl.sig_page_changed.connect(self.sagital_slice_page_changed)
        self.slut = pg.HistogramLUTWidget()
        self.slut.setImageItem(self.simg.img)

        self.himg = SliceStacks()
        self.hpage_ctrl = PageController()
        self.hpage_ctrl.sig_page_changed.connect(self.horizontal_slice_page_changed)
        self.hlut = pg.HistogramLUTWidget()
        self.hlut.setImageItem(self.himg.img)

        self.label_tree = LabelTree()

        self.working_atlas = self.cimg
        self.working_page_control = self.cpage_ctrl

        self.slice_stack = SliceStack()


        # radio buttons
        self.radio_group = QFrame()
        self.radio_group.setFixedHeight(50)
        self.radio_group.setStyleSheet('QFrame{border: 1px solid gray; border-radius: 3px}')
        radio_group_layout = QHBoxLayout(self.radio_group)
        radio_group_layout.setContentsMargins(5, 0, 5, 0)
        radio_group_layout.setAlignment(Qt.AlignCenter)
        self.section_rabnt1 = QRadioButton('Coronal')
        self.section_rabnt1.setChecked(True)
        self.section_rabnt2 = QRadioButton('Sagittal')
        self.section_rabnt3 = QRadioButton('Horizontal')
        self.section_rabnt1.setStyleSheet('color: white')
        self.section_rabnt2.setStyleSheet('color: white')
        self.section_rabnt3.setStyleSheet('color: white')
        radio_group_layout.addWidget(self.section_rabnt1)
        radio_group_layout.addWidget(self.section_rabnt2)
        radio_group_layout.addWidget(self.section_rabnt3)

        # angles
        self.crotation_ctrl = SliceRotation()
        self.crotation_ctrl.sig_slice_rotated.connect(self.coronal_slice_rotated)
        self.srotation_ctrl = SliceRotation()
        self.srotation_ctrl.sig_slice_rotated.connect(self.sagital_slice_rotated)
        self.hrotation_ctrl = SliceRotation()
        self.hrotation_ctrl.sig_slice_rotated.connect(self.horizontal_slice_rotated)


        # opacity
        atlas_op_label = QLabel('Opacity: ')
        self.atlas_op_slider = QSlider(Qt.Horizontal)
        self.atlas_op_slider.setValue(100)
        self.atlas_op_slider.setMinimum(0)
        self.atlas_op_slider.setMaximum(100)
        self.atlas_op_slider.valueChanged.connect(self.change_opacity_spinbox_value)
        # self.atlas_op_slider.valueChanged.connect(self.sig_rescale_slider)

        self.atlas_op_spinbox = QDoubleSpinBox()
        self.atlas_op_spinbox.lineEdit().setStyleSheet(spinbox_line_edit_style)
        self.atlas_op_spinbox.setValue(1)
        self.atlas_op_spinbox.setRange(0, 1)
        self.atlas_op_spinbox.setSingleStep(0.05)
        self.atlas_op_spinbox.setMinimumSize(50, 20)
        self.atlas_op_spinbox.valueChanged.connect(self.opacity_changed)

        opacity_wrap = QFrame()
        opacity_layout = QHBoxLayout(opacity_wrap)
        opacity_layout.setSpacing(2)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_layout.addWidget(atlas_op_label)
        opacity_layout.addWidget(self.atlas_op_slider)
        opacity_layout.addWidget(self.atlas_op_spinbox)

        # show brain region boundary of a volume atlas slice
        self.show_boundary_btn = QPushButton('Show Boundary')
        self.show_boundary_btn.setStyleSheet(button_style)
        self.show_boundary_btn.setCheckable(True)

        # turn on/off navigation
        self.navigation_btn = QPushButton('Navigation')
        self.navigation_btn.setStyleSheet(button_style)
        self.navigation_btn.setCheckable(True)
        self.navigation_btn.clicked.connect(self.navigation_btn_clicked)

        # turn on/off rotation remain
        self.rotation_remain_btn = QPushButton('Keep Slice Angles')
        self.rotation_remain_btn.setStyleSheet(button_style)
        self.rotation_remain_btn.setCheckable(True)
        self.rotation_remain_btn.clicked.connect(self.rotation_remain_btn_clicked)

        # coronal section control
        coronal_rotation_wrap = QGroupBox('Coronal Section')
        coronal_rotation_wrap.setStyleSheet(rotation_gb_style)
        coronal_wrap_layout = QGridLayout(coronal_rotation_wrap)
        coronal_wrap_layout.setContentsMargins(0, 0, 0, 0)
        coronal_wrap_layout.setSpacing(10)

        c_section_label_img = QPixmap('icons/sidebar/c_section.png')
        c_section_label_img = c_section_label_img.scaled(QSize(40, 40))
        c_section_label = QLabel('Tilt Z: ')
        c_section_label.setPixmap(c_section_label_img)

        c_lr_btn = RotationBtn("icons/sidebar/rotation_horizontal.svg")
        c_ud_btn = RotationBtn("icons/sidebar/rotation_vertical.svg")

        coronal_wrap_layout.addWidget(c_section_label, 0, 0, 2, 1)
        coronal_wrap_layout.addWidget(c_lr_btn, 0, 1, 1, 1)
        coronal_wrap_layout.addWidget(self.crotation_ctrl.h_slider, 0, 2, 1, 1)
        coronal_wrap_layout.addWidget(self.crotation_ctrl.h_spinbox, 0, 3, 1, 1)
        coronal_wrap_layout.addWidget(c_ud_btn, 1, 1, 1, 1)
        coronal_wrap_layout.addWidget(self.crotation_ctrl.v_slider, 1, 2, 1, 1)
        coronal_wrap_layout.addWidget(self.crotation_ctrl.v_spinbox, 1, 3, 1, 1)

        # sagittal section control
        sagital_rotation_wrap = QGroupBox('Sagittal Section')
        sagital_rotation_wrap.setStyleSheet(rotation_gb_style)
        sagital_wrap_layout = QGridLayout(sagital_rotation_wrap)
        sagital_wrap_layout.setContentsMargins(0, 0, 0, 0)
        sagital_wrap_layout.setSpacing(10)

        s_section_label_img = QPixmap('icons/sidebar/s_section.png')
        s_section_label_img = s_section_label_img.scaled(QSize(40, 40))
        s_section_label = QLabel()
        s_section_label.setPixmap(s_section_label_img)

        s_lr_btn = RotationBtn("icons/sidebar/rotation_horizontal.svg")
        s_ud_btn = RotationBtn("icons/sidebar/rotation_vertical.svg")

        sagital_wrap_layout.addWidget(s_section_label, 0, 0, 2, 1)
        sagital_wrap_layout.addWidget(s_lr_btn, 0, 1, 1, 1)
        sagital_wrap_layout.addWidget(self.srotation_ctrl.h_slider, 0, 2, 1, 1)
        sagital_wrap_layout.addWidget(self.srotation_ctrl.h_spinbox, 0, 3, 1, 1)
        sagital_wrap_layout.addWidget(s_ud_btn, 1, 1, 1, 1)
        sagital_wrap_layout.addWidget(self.srotation_ctrl.v_slider, 1, 2, 1, 1)
        sagital_wrap_layout.addWidget(self.srotation_ctrl.v_spinbox, 1, 3, 1, 1)

        # horizontal section control
        horizontal_rotation_wrap = QGroupBox('Horizontal Section')
        horizontal_rotation_wrap.setStyleSheet(rotation_gb_style)
        horizontal_wrap_layout = QGridLayout(horizontal_rotation_wrap)
        horizontal_wrap_layout.setContentsMargins(0, 0, 0, 0)
        horizontal_wrap_layout.setSpacing(10)

        h_section_label_img = QPixmap('icons/sidebar/h_section.png')
        h_section_label_img = h_section_label_img.scaled(QSize(40, 40))
        h_section_label = QLabel()
        h_section_label.setPixmap(h_section_label_img)

        h_lr_btn = RotationBtn("icons/sidebar/rotation_horizontal.svg")
        h_ud_btn = RotationBtn("icons/sidebar/rotation_vertical.svg")

        horizontal_wrap_layout.addWidget(h_section_label, 0, 0, 2, 1)
        horizontal_wrap_layout.addWidget(h_lr_btn, 0, 1, 1, 1)
        horizontal_wrap_layout.addWidget(self.hrotation_ctrl.h_slider, 0, 2, 1, 1)
        horizontal_wrap_layout.addWidget(self.hrotation_ctrl.h_spinbox, 0, 3, 1, 1)
        horizontal_wrap_layout.addWidget(h_ud_btn, 1, 1, 1, 1)
        horizontal_wrap_layout.addWidget(self.hrotation_ctrl.v_slider, 1, 2, 1, 1)
        horizontal_wrap_layout.addWidget(self.hrotation_ctrl.v_spinbox, 1, 3, 1, 1)

        self.sidebar_wrap = QFrame()
        sidebar_wrap_layout = QVBoxLayout(self.sidebar_wrap)
        sidebar_wrap_layout.setSpacing(0)
        sidebar_wrap_layout.addWidget(self.radio_group)
        sidebar_wrap_layout.addSpacing(10)
        sidebar_wrap_layout.addWidget(opacity_wrap)
        sidebar_wrap_layout.addSpacing(10)
        sidebar_wrap_layout.addWidget(self.show_boundary_btn)
        sidebar_wrap_layout.addSpacing(10)
        sidebar_wrap_layout.addWidget(coronal_rotation_wrap)
        sidebar_wrap_layout.addSpacing(10)
        sidebar_wrap_layout.addWidget(sagital_rotation_wrap)
        sidebar_wrap_layout.addSpacing(10)
        sidebar_wrap_layout.addWidget(horizontal_rotation_wrap)
        sidebar_wrap_layout.addSpacing(5)
        sidebar_wrap_layout.addWidget(self.rotation_remain_btn)
        sidebar_wrap_layout.addStretch(1)
        sidebar_wrap_layout.addWidget(self.navigation_btn)

        # self.lut.sigLookupTableChanged.connect(self.histlut_changed)
        # self.lut.sigLevelsChanged.connect(self.histlut_changed)

    def reset_3d_plates(self):
        self.ap_plate_verts = np.array([[-1, 0, -1], [-1, 0, 1], [1, 0, 1], [1, 0, -1]])
        self.ap_plate_faces = np.array([[0, 1, 2], [0, 2, 3]])

        self.dv_plate_verts = np.array([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]])
        self.dv_plate_faces = np.array([[0, 1, 2], [0, 2, 3]])

        self.ml_plate_verts = np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]])
        self.ml_plate_faces = np.array([[0, 1, 2], [0, 2, 3]])

    def set_slice_data(self, slice_data):
        self.slice_image_data = slice_data.copy()
        self.slice_stack.set_data(slice_data)
        self.slice_size = np.ravel(self.slice_stack.img_layer.image.shape[:2])
        slice_scale = np.max(self.slice_size / self.slice_tb_size_base)
        self.slice_tb_size = self.slice_size / slice_scale
        self.slice_tb_size = self.slice_tb_size[::-1].astype(int)

        self.working_atlas = self.slice_stack

        rect = (0, 0, int(self.slice_size[1]), int(self.slice_size[0]))
        self.corner_points, self.side_lines = get_corner_line_from_rect(rect)
        self.working_atlas.image_dict['tri_pnts'].set_range(x_range=self.slice_size[1] - 1,
                                                            y_range=self.slice_size[0] - 1)

    def set_slice_info(self, slice_info):
        self.slice_cut = slice_info.cut
        self.slice_width = slice_info.width
        self.slice_height = slice_info.height
        self.slice_distance = slice_info.distance
        if self.slice_height == 0 or self.slice_width == 0 or self.slice_distance == 0 or not self.slice_bregma:
            self.slice_info_ready = False
        else:
            self.slice_info_ready = True

    def save_slice_data_and_info(self):
        data = {'data': self.slice_image_data,
                'cut': self.slice_cut,
                'width': self.slice_width,
                'height': self.slice_height,
                'distance': self.slice_distance,
                'Bregma': self.slice_bregma,
                'ready': self.slice_info_ready}
        return data

    def set_slice_data_and_info(self, slice_data):
        img_data = slice_data['data'].copy()
        self.slice_cut = slice_data['cut']
        self.slice_width = slice_data['width']
        self.slice_height = slice_data['height']
        self.slice_distance = slice_data['distance']
        self.slice_bregma = slice_data['Bregma']
        self.slice_info_ready = slice_data['ready']
        self.set_slice_data(img_data)

    def get_slice_coords(self, ruler_data):
        x, y, z = get_slice_atlas_coord(ruler_data, self.slice_cut, self.slice_size,
                                        self.slice_width, self.slice_height, self.slice_distance, self.slice_bregma)
        return x, y, z

    def check_info_ready(self):
        if self.slice_height == 0 or self.slice_width == 0 or self.slice_distance == 0:
            self.slice_info_ready = False
        else:
            self.slice_info_ready = True

    def clear_slice_info(self):
        if self.slice_image_data is not None:
            self.slice_cut = 'Coronal'
            self.slice_width = 0
            self.slice_height = 0
            self.slice_distance = 0
            self.slice_bregma = []
            self.slice_info_ready = False
        else:
            return

    def set_data(self, atlas_data, atlas_label, atlas_info, label_info, boundaries):
        if self.atlas_data is not None:
            self.reset_3d_plates()
            self.clear_all_display_obj()

        self.atlas_data = atlas_data
        self.atlas_label = atlas_label
        self.label_info = label_info
        self.atlas_boundary = boundaries
        self.label_tree.set_labels(label_info)

        self.atlas_size = self.atlas_data.shape
        self.anterior_info = atlas_info[0]
        self.dorsal_info = atlas_info[1]
        self.right_info = atlas_info[2]
        self.vox_size_um = atlas_info[3]['vxsize']

        if np.all(np.ravel(atlas_info[3]['Bregma']) == 0):
            da_bregma = (int(self.atlas_size[1] / 2), int(self.atlas_size[2] / 2), int(self.atlas_size[0] / 2))
        else:
            da_bregma = atlas_info[3]['Bregma']
        inverse_b2 = self.atlas_size[0] - 1 - da_bregma[2]
        self.Bregma = (inverse_b2, da_bregma[0], da_bregma[1])
        print(self.Bregma, 'bregma')
        self.rotate_origin_3d = np.array(da_bregma)
        self.origin_3d = np.array(da_bregma)

        self.current_coronal_index = self.origin_3d[1]
        self.current_sagital_index = self.origin_3d[0]
        self.current_horizontal_index = self.origin_3d[2]

        self.cpage_ctrl.set_max(self.atlas_size[2] - 1)
        self.spage_ctrl.set_max(self.atlas_size[1] - 1)
        self.hpage_ctrl.set_max(self.atlas_size[0] - 1)

        if self.current_coronal_index == 0:
            self.coronal_slice_page_changed(self.current_coronal_index)
            self.cpage_ctrl.page_label.setText('0')
        else:
            self.cpage_ctrl.set_val(self.current_coronal_index)

        if self.current_sagital_index == 0:
            self.sagital_slice_page_changed(self.current_sagital_index)
            self.spage_ctrl.page_label.setText('0')
        else:
            self.spage_ctrl.set_val(self.current_sagital_index)

        if self.current_horizontal_index == 0:
            self.horizontal_slice_page_changed(self.current_horizontal_index)
            self.hpage_ctrl.page_label.setText('0')
        else:
            self.hpage_ctrl.set_val(self.current_horizontal_index)

        self.cimg.label_img.setLevels(levels=[0, self.label_tree.label_level])
        self.simg.label_img.setLevels(levels=[0, self.label_tree.label_level])
        self.himg.label_img.setLevels(levels=[0, self.label_tree.label_level])

        lut = self.label_tree.lookup_table()
        self.cimg.label_img.setLookupTable(lut=lut)
        self.simg.label_img.setLookupTable(lut=lut)
        self.himg.label_img.setLookupTable(lut=lut)

        self.c_size = np.array([self.atlas_size[0], self.atlas_size[1]])
        coronal_scale = np.max(self.c_size / self.slice_tb_size_base)
        self.coronal_tb_size = self.c_size[::-1] / coronal_scale
        self.coronal_tb_size = self.coronal_tb_size.astype(int)

        self.s_size = np.array([self.atlas_size[0], self.atlas_size[2]])
        sagital_scale = np.max(self.s_size / self.slice_tb_size_base)
        self.sagital_tb_size = self.s_size[::-1] / sagital_scale
        self.sagital_tb_size = self.sagital_tb_size.astype(int)

        self.h_size = np.array([self.atlas_size[1], self.atlas_size[2]])
        horizontal_scale = np.max(self.h_size / self.slice_tb_size_base)
        self.horizontal_tb_size = self.h_size[::-1] / horizontal_scale
        self.horizontal_tb_size = self.horizontal_tb_size.astype(int)

        self.update_3d_plate_component()

    def update_3d_plate_component(self):
        self.ap_plate_verts[:, 0] = self.ap_plate_verts[:, 0] * self.atlas_size[1] * 0.5
        self.ap_plate_verts[:, 2] = self.ap_plate_verts[:, 2] * self.atlas_size[0] * 0.5

        self.dv_plate_verts[:, 0] = self.dv_plate_verts[:, 0] * self.atlas_size[1] * 0.5
        self.dv_plate_verts[:, 1] = self.dv_plate_verts[:, 1] * self.atlas_size[2] * 0.5

        self.ml_plate_verts[:, 1] = self.ml_plate_verts[:, 1] * self.atlas_size[2] * 0.5
        self.ml_plate_verts[:, 2] = self.ml_plate_verts[:, 2] * self.atlas_size[0] * 0.5

        z_slide_dist = (self.atlas_size[0] * 0.5 - self.origin_3d[2])
        y_slide_dist = (self.atlas_size[2] * 0.5 - self.origin_3d[1])
        x_slide_dist = (self.atlas_size[1] * 0.5 - self.origin_3d[0])

        self.ap_plate_verts = self.ap_plate_verts + np.array([x_slide_dist, 0, z_slide_dist])
        self.dv_plate_verts = self.dv_plate_verts + np.array([x_slide_dist, y_slide_dist, 0])
        self.ml_plate_verts = self.ml_plate_verts + np.array([0, y_slide_dist, z_slide_dist])

        ap_plate_md = gl.MeshData(vertexes=self.ap_plate_verts, faces=self.ap_plate_faces)
        self.ap_plate_mesh.setMeshData(meshdata=ap_plate_md)

        dv_plate_md = gl.MeshData(vertexes=self.dv_plate_verts, faces=self.dv_plate_faces)
        self.dv_plate_mesh.setMeshData(meshdata=dv_plate_md)

        ml_plate_md = gl.MeshData(vertexes=self.ml_plate_verts, faces=self.ml_plate_faces)
        self.ml_plate_mesh.setMeshData(meshdata=ml_plate_md)

    def working_cut_changed(self, atlas_display):
        self.pre_trajectory_changed()
        if atlas_display == "coronal":
            self.working_atlas = self.cimg
            self.working_page_control = self.cpage_ctrl
            self.slice_tb_size = self.coronal_tb_size
            self.slice_size = self.c_size
        elif atlas_display == "sagittal":
            self.working_atlas = self.simg
            self.working_page_control = self.spage_ctrl
            self.slice_tb_size = self.sagital_tb_size
            self.slice_size = self.s_size
        else:
            self.working_atlas = self.himg
            self.working_page_control = self.hpage_ctrl
            self.slice_tb_size = self.horizontal_tb_size
            self.slice_size = self.h_size
        rect = (0, 0, int(self.slice_size[1]), int(self.slice_size[0]))
        self.corner_points, self.side_lines = get_corner_line_from_rect(rect)
        self.working_atlas.image_dict['tri_pnts'].set_range(x_range=self.slice_size[1] - 1,
                                                            y_range=self.slice_size[0] - 1)

    def navigation_btn_clicked(self):
        if self.navigation_btn.isChecked():
            # self.
            self.cimg.v_line.setVisible(True)
            self.cimg.h_line.setVisible(True)
            self.himg.v_line.setVisible(True)
            self.himg.h_line.setVisible(True)
            self.simg.v_line.setVisible(True)
            self.simg.h_line.setVisible(True)
        else:
            self.cimg.v_line.setVisible(False)
            self.cimg.h_line.setVisible(False)
            self.himg.v_line.setVisible(False)
            self.himg.h_line.setVisible(False)
            self.simg.v_line.setVisible(False)
            self.simg.h_line.setVisible(False)

    def rotation_remain_btn_clicked(self):
        if self.rotation_remain_btn.isChecked():
            self.remain_angle = True
        else:
            self.remain_angle = False

    def change_opacity_spinbox_value(self):
        val = self.atlas_op_slider.value()
        self.atlas_op_spinbox.setValue(val / 100)

    def opacity_changed(self):
        val = self.atlas_op_spinbox.value()
        if self.atlas_op_slider.value() != int(val * 100):
            self.atlas_op_slider.setValue(int(val * 100))
        self.cimg.label_img.setOpts(opacity=val)
        self.simg.label_img.setOpts(opacity=val)
        self.himg.label_img.setOpts(opacity=val)

    # coronal page number changed and slice rotated
    def set_perfect_coronal_slice(self):
        da_atlas_slice = self.atlas_data[:, :, self.current_coronal_index]
        da_atlas_label = self.atlas_label[:, :, self.current_coronal_index]
        da_atlas_contour = self.atlas_boundary['c_contour'][:, :, self.current_coronal_index]
        self.cimg.set_data(da_atlas_slice, da_atlas_label, da_atlas_contour, scale=None)

        slide_dist = (self.current_coronal_index - self.origin_3d[1])
        ap_plate_verts = self.ap_plate_verts + np.array([0, slide_dist, 0])
        ap_plate_md = gl.MeshData(vertexes=ap_plate_verts, faces=self.ap_plate_faces)
        self.ap_plate_mesh.setMeshData(meshdata=ap_plate_md)
        self.get_3d_origin()

    def rotate_coronal_current_slice(self):
        # calculate for 2d rotation
        c_id = self.current_coronal_index
        s_id = self.current_sagital_index
        h_id = self.atlas_size[0] - 1 - self.current_horizontal_index

        z_angle = self.coronal_rads[0]
        x_angle = self.coronal_rads[1]
        self.c_rotm_2d = np.dot(rotation_x(z_angle), rotation_y(x_angle))

        o_val = np.array([0, 0, c_id])
        o_rot = np.array([h_id, s_id, c_id])

        oz_vector = np.dot(self.c_rotm_2d, np.array([1, 0, 0]))
        ox_vector = np.dot(self.c_rotm_2d, np.array([0, 1, 0]))

        oval_new = o_rot + np.dot(self.c_rotm_2d, o_val - o_rot)

        ox_length = self.atlas_size[1]
        oz_length = self.atlas_size[0]

        atlas = fn.affineSlice(self.atlas_data, shape=(oz_length, ox_length), vectors=[oz_vector, ox_vector],
                               origin=(oval_new[0], oval_new[1], oval_new[2]), axes=(0, 1, 2), order=1)
        label = fn.affineSlice(self.atlas_label, shape=(oz_length, ox_length), vectors=[oz_vector, ox_vector],
                               origin=(oval_new[0], oval_new[1], oval_new[2]), axes=(0, 1, 2), order=0)
        da_contour = make_contour_img(label)
        self.cimg.set_data(atlas, label, da_contour, scale=None)

        self.c_rotm_3d = np.dot(rotation_z(z_angle), rotation_x(-x_angle))

        slide_dist = self.current_coronal_index - self.origin_3d[1]
        ap_plate_verts = self.ap_plate_verts + np.array([0, slide_dist, 0])
        ap_plate_verts = np.dot(self.c_rotm_3d, (ap_plate_verts - self.rotate_origin_3d).T).T + self.rotate_origin_3d
        ap_plate_md = gl.MeshData(vertexes=ap_plate_verts, faces=self.ap_plate_faces)
        self.ap_plate_mesh.setMeshData(meshdata=ap_plate_md)

    def coronal_slice_page_changed(self, page_number):
        self.clear_all_display_obj()
        if self.atlas_data is None or self.atlas_label is None:
            return
        self.current_coronal_index = page_number
        self.set_perfect_coronal_slice()
        if self.remain_angle:
            if self.coronal_rotated:
                self.rotate_coronal_current_slice()
        else:
            if self.coronal_rotated:
                self.crotation_ctrl.h_spinbox.blockSignals(True)
                self.crotation_ctrl.v_spinbox.blockSignals(True)
                self.crotation_ctrl.h_slider.setValue(0)
                self.crotation_ctrl.v_slider.setValue(0)
                self.crotation_ctrl.h_spinbox.blockSignals(False)
                self.crotation_ctrl.v_spinbox.blockSignals(False)
                self.coronal_rotated = False

    def coronal_slice_rotated(self, rads):
        self.clear_all_display_obj()
        if self.atlas_data is None or self.atlas_label is None:
            return
        self.coronal_rads = rads
        if np.all(np.ravel(rads) == 0):
            if self.coronal_rotated:
                self.set_perfect_coronal_slice()
                self.coronal_rotated = False
        else:
            self.rotate_coronal_current_slice()
            self.coronal_rotated = True

    # sagittal page number changed and slice rotated
    def set_perfect_sagittal_slice(self):
        da_atlas_slice = self.atlas_data[:, self.current_sagital_index, :]
        da_atlas_label = self.atlas_label[:, self.current_sagital_index, :]
        da_atlas_contour = self.atlas_boundary['s_contour'][:, self.current_sagital_index, :]
        self.simg.set_data(da_atlas_slice, da_atlas_label, da_atlas_contour, scale=None)

        slide_dist = (self.current_sagital_index - self.origin_3d[0])
        ml_plate_verts = self.ml_plate_verts + np.array([slide_dist, 0, 0])
        ml_plate_md = gl.MeshData(vertexes=ml_plate_verts, faces=self.ml_plate_faces)
        self.ml_plate_mesh.setMeshData(meshdata=ml_plate_md)
        self.get_3d_origin()

    def rotate_sagittal_current_slice(self):
        # calculate for 2d rotation
        c_id = self.current_coronal_index
        s_id = self.current_sagital_index
        h_id = self.atlas_size[0] - 1 - self.current_horizontal_index

        z_angle = self.sagittal_rads[0]
        y_angle = self.sagittal_rads[1]
        self.s_rotm_2d = np.dot(rotation_x(z_angle), rotation_z(y_angle))

        o_val = np.array([0, s_id, 0])
        o_rot = np.array([h_id, s_id, c_id])

        oz_vector = np.dot(self.s_rotm_2d, np.array([1, 0, 0]))
        oy_vector = np.dot(self.s_rotm_2d, np.array([0, 0, 1]))

        oval_new = o_rot + np.dot(self.s_rotm_2d, o_val - o_rot)

        oy_length = self.atlas_size[2]
        oz_length = self.atlas_size[0]

        atlas = fn.affineSlice(self.atlas_data, shape=(oz_length, oy_length), vectors=[oz_vector, oy_vector],
                               origin=(oval_new[0], oval_new[1], oval_new[2]), axes=(0, 1, 2), order=1)
        label = fn.affineSlice(self.atlas_label, shape=(oz_length, oy_length), vectors=[oz_vector, oy_vector],
                               origin=(oval_new[0], oval_new[1], oval_new[2]), axes=(0, 1, 2), order=0)
        da_contour = make_contour_img(label)
        self.simg.set_data(atlas, label, da_contour, scale=None)

        self.s_rotm_3d = np.dot(rotation_z(z_angle), rotation_y(-y_angle))

        slide_dist = (self.current_sagital_index - self.Bregma[1])
        ml_plate_verts = self.ml_plate_verts + np.array([slide_dist, 0, 0])
        ml_plate_verts = np.dot(self.s_rotm_3d, (ml_plate_verts - self.rotate_origin_3d).T).T + self.rotate_origin_3d
        ml_plate_md = gl.MeshData(vertexes=ml_plate_verts, faces=self.ml_plate_faces)
        self.ml_plate_mesh.setMeshData(meshdata=ml_plate_md)

    def sagital_slice_page_changed(self, page_number):
        self.clear_all_display_obj()
        if self.atlas_data is None or self.atlas_label is None:
            return
        self.current_sagital_index = page_number
        self.set_perfect_sagittal_slice()
        if self.remain_angle:
            if self.sagittal_rotated:
                self.rotate_sagittal_current_slice()
        else:
            if self.sagittal_rotated:
                self.srotation_ctrl.h_spinbox.blockSignals(True)
                self.srotation_ctrl.v_spinbox.blockSignals(True)
                self.srotation_ctrl.h_slider.setValue(0)
                self.srotation_ctrl.v_slider.setValue(0)
                self.srotation_ctrl.h_spinbox.blockSignals(False)
                self.srotation_ctrl.v_spinbox.blockSignals(False)
                self.sagittal_rotated = False

    def sagital_slice_rotated(self, rads):
        self.clear_all_display_obj()
        if self.atlas_data is None or self.atlas_label is None:
            return
        self.sagittal_rads = rads
        if np.all(np.ravel(rads) == 0):
            if self.sagittal_rotated:
                self.set_perfect_sagittal_slice()
                self.sagittal_rotated = False
        else:
            self.rotate_sagittal_current_slice()
            self.sagittal_rotated = True

    # horizontal page number changed and slice rotated
    def set_perfect_horizontal_slice(self):
        slice_number = self.atlas_size[0] - 1 - self.current_horizontal_index
        da_atlas_slice = self.atlas_data[slice_number, :, :]
        da_atlas_label = self.atlas_label[slice_number, :, :]
        da_atlas_contour = self.atlas_boundary['h_contour'][slice_number, :, :]
        self.himg.set_data(da_atlas_slice, da_atlas_label, da_atlas_contour, scale=None)

        slide_dist = (self.current_horizontal_index - self.origin_3d[2])
        dv_plate_verts = self.dv_plate_verts + np.array([0, 0, slide_dist])
        dv_plate_md = gl.MeshData(vertexes=dv_plate_verts, faces=self.dv_plate_faces)
        self.dv_plate_mesh.setMeshData(meshdata=dv_plate_md)
        self.get_3d_origin()

    def rotate_horizontal_current_slice(self):
        c_id = self.current_coronal_index
        s_id = self.current_sagital_index
        h_id = self.atlas_size[0] - 1 - self.current_horizontal_index

        x_angle = self.horizontal_rads[0]
        y_angle = self.horizontal_rads[1]
        self.h_rotm_2d = np.dot(rotation_z(y_angle), rotation_y(x_angle))

        o_val = np.array([h_id, 0, 0])
        o_rot = np.array([h_id, s_id, c_id])

        ox_vector = np.dot(self.h_rotm_2d, np.array([0, 1, 0]))
        oy_vector = np.dot(self.h_rotm_2d, np.array([0, 0, 1]))

        oval_new = o_rot + np.dot(self.h_rotm_2d, o_val - o_rot)

        oy_length = self.atlas_size[2]
        ox_length = self.atlas_size[1]

        atlas = fn.affineSlice(self.atlas_data, shape=(ox_length, oy_length), vectors=[ox_vector, oy_vector],
                               origin=(oval_new[0], oval_new[1], oval_new[2]), axes=(0, 1, 2), order=1)
        label = fn.affineSlice(self.atlas_label, shape=(ox_length, oy_length), vectors=[ox_vector, oy_vector],
                               origin=(oval_new[0], oval_new[1], oval_new[2]), axes=(0, 1, 2), order=0)
        da_contour = make_contour_img(label)
        self.himg.set_data(atlas, label, da_contour, scale=None)

        self.h_rotm_3d = np.dot(rotation_y(-y_angle), rotation_x(-x_angle))

        slide_dist = (self.Bregma[0] - h_id)
        dv_plate_verts = self.dv_plate_verts + np.array([0, 0, slide_dist])
        dv_plate_verts = np.dot(self.h_rotm_3d, (dv_plate_verts - self.rotate_origin_3d).T).T + self.rotate_origin_3d
        dv_plate_md = gl.MeshData(vertexes=dv_plate_verts, faces=self.dv_plate_faces)
        self.dv_plate_mesh.setMeshData(meshdata=dv_plate_md)

    def horizontal_slice_page_changed(self, page_number):
        self.clear_all_display_obj()
        if self.atlas_data is None or self.atlas_label is None:
            return
        self.current_horizontal_index = page_number

        self.set_perfect_horizontal_slice()
        if self.remain_angle:
            if self.horizontal_rotated:
                self.rotate_horizontal_current_slice()
        else:
            if self.horizontal_rotated:
                self.hrotation_ctrl.h_spinbox.blockSignals(True)
                self.hrotation_ctrl.v_spinbox.blockSignals(True)
                self.hrotation_ctrl.h_slider.setValue(0)
                self.hrotation_ctrl.v_slider.setValue(0)
                self.hrotation_ctrl.h_spinbox.blockSignals(False)
                self.hrotation_ctrl.v_spinbox.blockSignals(False)
                self.horizontal_rotated = False

    def horizontal_slice_rotated(self, rads):
        self.clear_all_display_obj()
        if self.atlas_data is None or self.atlas_label is None:
            return
        self.horizontal_rads = rads
        if np.all(np.ravel(rads) == 0):
            if self.horizontal_rotated:
                self.set_perfect_horizontal_slice()
                self.horizontal_rotated = False
        else:
            self.rotate_horizontal_current_slice()
            self.horizontal_rotated = True

    # get 3d rotation origin
    def get_3d_origin(self):
        c_id = self.current_coronal_index
        s_id = self.current_sagital_index
        h_id = self.current_horizontal_index
        o_rot = np.array([s_id, c_id, h_id])
        self.rotate_origin_3d = o_rot - self.origin_3d

    #
    def rotate_cs_plane_after_merging_probe(self, display_data):
        # display data has to be the merged probe data
        self.cpage_ctrl.set_val(display_data['insertion_vox'][1])
        self.spage_ctrl.set_val(display_data['insertion_vox'][0])
        self.hpage_ctrl.set_val(display_data['insertion_vox'][2])

        ap_sign, ml_sign = get_tilt_sign(display_data['insertion_coords'], display_data['terminus_coords'])

        ap_angle = -1 * ap_sign * display_data['ap_angle']
        ml_angle = ml_sign * display_data['ml_angle']

        start_pnt = display_data['insertion_coords_3d'].copy()
        end_pnt = display_data['terminus_coords_3d'].copy()
        # if ap_angle <= 30 and ap_angle >= -30:
        if ap_angle <= -1e-6 or ap_angle >= 1e-6:
            self.coronal_slice_rotated((0, np.radians(ap_angle)))
            rotated_back_p1 = np.dot(self.c_rotm_3d.T, start_pnt - self.rotate_origin_3d)
            rotated_back_p2 = np.dot(self.c_rotm_3d.T, end_pnt - self.rotate_origin_3d)

            rotated_back_p1 = rotated_back_p1 + self.origin_3d + self.rotate_origin_3d
            rotated_back_p2 = rotated_back_p2 + self.origin_3d + self.rotate_origin_3d

            c_start_pnt = [rotated_back_p1[0], self.atlas_size[0] - rotated_back_p1[2]]
            c_end_pnt = [rotated_back_p2[0], self.atlas_size[0] - rotated_back_p2[2]]
        else:
            c_start_pnt_3d = start_pnt + self.origin_3d
            c_end_pnt_3d = end_pnt + self.origin_3d

            c_start_pnt = [c_start_pnt_3d[0], self.atlas_size[0] - c_start_pnt_3d[2]]
            c_end_pnt = [c_end_pnt_3d[0], self.atlas_size[0] - c_end_pnt_3d[2]]

        if ml_angle <= -1e-6 or ml_angle >= 1e-6:
            self.sagital_slice_rotated((0, np.radians(ml_angle)))
            rotated_back_p1 = np.dot(self.s_rotm_3d.T, start_pnt - self.rotate_origin_3d)
            rotated_back_p2 = np.dot(self.s_rotm_3d.T, end_pnt - self.rotate_origin_3d)

            rotated_back_p1 = rotated_back_p1 + self.origin_3d + self.rotate_origin_3d
            rotated_back_p2 = rotated_back_p2 + self.origin_3d + self.rotate_origin_3d

            s_start_pnt = [rotated_back_p1[1], self.atlas_size[0] - rotated_back_p1[2]]
            s_end_pnt = [rotated_back_p2[1], self.atlas_size[0] - rotated_back_p2[2]]
        else:
            s_start_pnt_3d = start_pnt + self.origin_3d
            s_end_pnt_3d = end_pnt + self.origin_3d

            s_start_pnt = [s_start_pnt_3d[0], self.atlas_size[0] - s_start_pnt_3d[2]]
            s_end_pnt = [s_end_pnt_3d[0], self.atlas_size[0] - s_end_pnt_3d[2]]

        c_pos = np.stack([c_start_pnt, c_end_pnt], axis=0)
        s_pos = np.stack([s_start_pnt, s_end_pnt], axis=0)

        self.cimg.add_display_obj_to_view(c_pos, display_data['vis_color'], 'probe')
        self.simg.add_display_obj_to_view(s_pos, display_data['vis_color'], 'probe')
        self.has_display_objects = True


    def clear_all_display_obj(self):
        if not self.has_display_objects:
            return
        c_n_obj = len(self.cimg.display_objects)
        c_obj_indexes = np.arange(c_n_obj)[::-1]
        for i in c_obj_indexes:
            self.cimg.remove_display_obj_from_view(i)

        s_n_obj = len(self.simg.display_objects)
        s_obj_indexes = np.arange(s_n_obj)[::-1]
        for j in s_obj_indexes:
            self.simg.remove_display_obj_from_view(j)

        self.cimg.display_objects.clear()
        self.simg.display_objects.clear()

        self.has_display_objects = False




    def get_atlas_angles(self):
        c_ang = (self.crotation_ctrl.h_spinbox.value(), self.crotation_ctrl.v_spinbox.value())
        s_ang = (self.srotation_ctrl.h_spinbox.value(), self.srotation_ctrl.v_spinbox.value())
        h_ang = (self.hrotation_ctrl.h_spinbox.value(), self.hrotation_ctrl.v_spinbox.value())
        return [c_ang, s_ang, h_ang]

    def set_volume_atlas_ctrl_data(self, atlas_ctrl_data):
        self.cpage_ctrl.page_slider.setValue(atlas_ctrl_data['coronal_index'])
        self.spage_ctrl.page_slider.setValue(atlas_ctrl_data['sagittal_index'])
        self.hpage_ctrl.page_slider.setValue(atlas_ctrl_data['horizontal_index'])
        atlas_rotation = atlas_ctrl_data['slice_rotation']
        self.crotation_ctrl.h_spinbox.setValue(atlas_rotation[0][0])
        self.crotation_ctrl.v_spinbox.setValue(atlas_rotation[0][1])
        self.srotation_ctrl.h_spinbox.setValue(atlas_rotation[1][0])
        self.srotation_ctrl.v_spinbox.setValue(atlas_rotation[1][1])
        self.hrotation_ctrl.h_spinbox.setValue(atlas_rotation[2][0])
        self.hrotation_ctrl.v_spinbox.setValue(atlas_rotation[2][1])

    def get_coronal_3d(self, points2, coronal_index=None):
        if coronal_index is None:
            da_y = np.ones(len(points2)) * self.current_coronal_index
        else:
            da_y = coronal_index
        points3 = np.vstack([points2[:, 0], da_y, self.atlas_size[0] - points2[:, 1]]).T
        points3 = points3 - self.origin_3d
        if self.coronal_rotated:
            rot_mat = self.c_rotm_3d
            rotation_origin = self.rotate_origin_3d
            points3 = np.dot(rot_mat, (points3 - rotation_origin).T).T + rotation_origin
        return points3

    def get_sagittal_3d(self, points2, sagital_index=None):
        if sagital_index is None:
            da_x = np.ones(len(points2)) * self.current_sagital_index
        else:
            da_x = sagital_index
        points3 = np.vstack([da_x, points2[:, 0], self.atlas_size[0] - points2[:, 1]]).T
        points3 = points3 - self.origin_3d
        if self.sagittal_rotated:
            rot_mat = self.s_rotm_3d
            rotation_origin = self.rotate_origin_3d
            points3 = np.dot(rot_mat, (points3 - rotation_origin).T).T + rotation_origin
        return points3

    def get_horizontal_3d(self, points2, horizontal_index=None):
        if horizontal_index is None:
            da_z = np.ones(len(points2)) * self.current_horizontal_index
        else:
            da_z = horizontal_index
        points3 = np.vstack([points2[:, 1], points2[:, 0], da_z]).T
        points3 = points3 - self.origin_3d
        if self.horizontal_rotated:
            rot_mat = self.h_rotm_3d
            rotation_origin = self.rotate_origin_3d
            points3 = np.dot(rot_mat, (points3 - rotation_origin).T).T + rotation_origin
        return points3

    def get_3d_data_from_2d_view(self, processing_data, atlas_display):
        if atlas_display == 'coronal':
            data = self.get_coronal_3d(processing_data)
        elif atlas_display == 'sagittal':
            data = self.get_sagittal_3d(processing_data)
        else:
            data = self.get_horizontal_3d(processing_data)
        return data

    def pre_trajectory_changed(self):
        self.working_atlas.remove_pre_trajectories_vis_lines()

    def get_pre_vis_data_for_volume_atlas(self, data, base_loc):
        base_loc = np.ravel(base_loc) / self.vox_size_um
        start_pnt, end_pnt = rotate_base_points(data, base_loc)
        return start_pnt, end_pnt

    def get_pre_vis_data_for_slice_atlas(self, data, base_loc):
        start_pnt, end_pnt = rotate_base_points(data, base_loc)
        return start_pnt, end_pnt

    def set_pre_vis_data(self, start_pnt, end_pnt):
        line_data = []
        for i in range(len(start_pnt)):
            da_pnts = np.asarray([start_pnt[i], end_pnt[i]])
            line_data.append(da_pnts)
        self.working_atlas.set_pre_design_vis_data(line_data)
        self.working_atlas.image_dict['atlas-probe'].setData(pos=np.asarray(start_pnt))

    def draw_pre_2d_vis_data_for_volume_atlas(self, data, base_loc):
        if len(data) == 1:
            base_loc = np.ravel(base_loc) / self.vox_size_um
            temp = np.zeros((len(base_loc), 2))
            temp[:, 0] = base_loc + data[0][0]
            temp[:, 1] = data[0][1]
            self.working_atlas.image_dict['atlas-probe'].setData(pos=np.asarray(temp))
        elif len(data) == 2:
            start_pnt, end_pnt = self.get_pre_vis_data_for_volume_atlas(data, base_loc)
            temp = np.vstack([start_pnt, end_pnt])
            self.working_atlas.image_dict['atlas-probe'].setData(pos=temp)
        else:
            self.working_atlas.image_dict['atlas-probe'].clear()
            self.working_atlas.remove_pre_trajectories_vis_lines()

    def draw_pre_2d_vis_data_for_slice_atlas(self, data, base_loc):
        if self.slice_info_ready:
            base_loc = np.ravel(base_loc) / 1000 * self.slice_size[1] / self.slice_width
            if len(data) == 1:
                temp = np.zeros((len(base_loc), 2))
                temp[:, 0] = base_loc + data[0][0]
                temp[:, 1] = data[0][1]
                self.working_atlas.image_dict['atlas-probe'].setData(pos=np.asarray(temp))
            elif len(data) == 2:
                start_pnt, end_pnt = self.get_pre_vis_data_for_slice_atlas(data, base_loc)
                temp = np.vstack([start_pnt, end_pnt])
                self.working_atlas.image_dict['atlas-probe'].setData(pos=temp)
            else:
                self.working_atlas.image_dict['atlas-probe'].clear()
                self.working_atlas.remove_pre_trajectories_vis_lines()
        else:
            if len(data) == 1:
                self.working_atlas.image_dict['atlas-probe'].setData(pos=np.asarray(data))
            elif len(data) == 2:
                self.working_atlas.image_dict['atlas-probe'].setData(pos=np.asarray(data))
                self.working_atlas.set_pre_design_vis_data([np.asarray(data)])
            else:
                self.working_atlas.image_dict['atlas-probe'].clear()
                self.working_atlas.remove_pre_trajectories_vis_lines()


    # def get_pre_multi_probe_data(self, data, atlas_display, multi_settings):
    #     x_vals = multi_settings['x_vals']
    #     y_vals = multi_settings['y_vals']
    #     n_probes = len(x_vals)
    #
    #     if atlas_display == 'coronal':
    #         data3 = self.get_coronal_3d(data)
    #     elif atlas_display == 'sagittal':
    #         data3 = self.get_sagittal_3d(data)
    #     else:
    #         data3 = self.get_horizontal_3d(data)
    #
    #     pdata = np.asarray(data3)
    #     p_order = np.argsort(pdata[:, 1])
    #     pdata = pdata[p_order, :]
    #
    #     direction = pdata[0] - pdata[1]
    #     direction = direction / np.linalg.norm(direction)
    #
    #     rot_m = get_direction_rotation(direction)
    #     base_end = np.dot(rot_m.T, pdata[1] - pdata[0])
    #
    #     base_start_3d = np.zeros((n_probes, 3))
    #     base_start_3d[:, 1] = x_vals.copy()
    #     base_start_3d[:, 2] = y_vals.copy()
    #
    #     base_end_3d = base_start_3d.copy()
    #     base_end_3d[:, 0] = base_end
    #
    #     points3_list = []
    #     for i in range(n_probes):
    #         tp3 = np.vstack([base_start_3d[i], base_end_3d[i]])
    #         points3 = np.dot(rot_m, tp3.T).T
    #         points3_list.append(points3)
    #
    #     return points3_list

    def get_multi_probe_2d_vis_data(self, data, multi_settings):
        x_vals = multi_settings['x_vals']
        y_vals = multi_settings['y_vals']
        n_probes = len(x_vals)

        vind = np.where(y_vals == 0)[0]
        if len(vind) == 0:
            vis_data = [data]
        else:
            vis_data = []
            base_loc = np.ravel(x_vals)[vind]

    def get_plane_norm_vector(self, atlas_display):
        if atlas_display == 'coronal':
            if self.coronal_rotated:
                n_vec = np.dot(self.c_rotm_3d, np.array([0, -1, 0]))
            else:
                n_vec = np.array([0, -1, 0])
        elif atlas_display == 'sagittal':
            if self.sagittal_rotated:
                n_vec = np.dot(self.s_rotm_3d, np.array([1, 0, 0]))
            else:
                n_vec = np.array([1, 0, 0])
        else:
            if self.horizontal_rotated:
                n_vec = np.dot(self.h_rotm_3d, np.array([0, 0, 1]))
            else:
                n_vec = np.array([0, 0, 1])
        return n_vec


    #
    def draw_volume_pre_trajectory(self, data, n_pre_trajectory):
        if len(data) == 1:
            if n_pre_trajectory == 1:

                self.working_atlas.image_dict['atlas-probe'].setData(pos=np.asarray(data))
            else:
                base_loc = np.array([-375, -125, 125, 375]) / self.vox_size_um
                temp = np.stack([data[0][0] + base_loc, np.repeat(data[0][1], 4)], axis=1)
                self.working_atlas.image_dict['atlas-probe'].setData(pos=np.asarray(temp))
        elif len(data) == 2:
            if n_pre_trajectory == 1:
                print('data', data)
                start_pnt, end_pnt = rotate_base_points(np.asarray(data), np.array([0]))
                print(start_pnt, end_pnt)
                self.working_atlas.image_dict['atlas-probe'].setData(pos=np.asarray(data))
            else:
                base_loc = np.array([-375, -125, 125, 375]) / self.vox_size_um
                start_pnt, end_pnt = rotate_base_points(np.asarray(data), base_loc)
                temp = np.vstack([start_pnt, end_pnt])
                self.working_atlas.image_dict['atlas-probe'].setData(pos=temp)
        else:
            self.working_atlas.image_dict['atlas-probe'].clear()

    def vis_3d_planes(self, vis):
        self.ap_plate_mesh.setVisible(vis)
        self.dv_plate_mesh.setVisible(vis)
        self.ml_plate_mesh.setVisible(vis)

    def vis_3d_axes(self, vis):
        self.axis.setVisible(vis)

    # ----------- clear atlas
    def clear_atlas_stacks(self):
        valid_keys = ['atlas-overlay', 'atlas-contour', 'atlas-mask', 'circle_follow', 'lasso_path', 'atlas-virus',
                      'atlas-probe', 'atlas-cells', 'atlas-drawing']
        for da_key in valid_keys:
            self.working_atlas.image_dict[da_key].clear()

    def clear_volume_atlas(self):
        if self.atlas_label is None or self.atlas_data is None:
            return
        valid_keys = ['atlas-overlay', 'atlas-contour', 'atlas-mask', 'circle_follow', 'lasso_path', 'atlas-virus',
                      'atlas-probe', 'atlas-cells', 'atlas-drawing']
        for da_key in valid_keys:
            self.cimg.image_dict[da_key].clear()
            self.simg.image_dict[da_key].clear()
            self.himg.image_dict[da_key].clear()
        self.cimg.img.clear()
        self.simg.img.clear()
        self.himg.img.clear()


    def clear_slice_atlas(self):
        pass

    def clear_atlas(self):
        self.clear_volume_atlas()
        self.clear_slice_atlas()












