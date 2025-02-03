import os
import numpy as np
import math
import pandas as pd
import cv2
import pickle
import colorsys
import pyqtgraph as pg
from scipy.interpolate import interp1d, splprep, splev
from .uuuuuu import rotation_z, rotation_x, rotation_y


def get_closest_point_to_line(p0, r, p):
    ap = p - p0
    res = p0 + np.dot(ap, r) * r
    return res


def line_fit_2d(points_2d, image=None):
    msg = None
    points = np.asarray(points_2d)
    sort_order = np.argsort(points[:, 1])
    points = points[sort_order, :]
    avg = np.mean(points, 0)
    subtracted = points - avg
    u, s, vh = np.linalg.svd(subtracted)
    direction = vh[0, :] / np.linalg.norm(vh[0, :])
    p1 = points[0, :]
    p2 = points[-1, :]
    sp = get_closest_point_to_line(avg, direction, p1)
    ep = get_closest_point_to_line(avg, direction, p2)

    if image is not None:
        direction = sp - ep
        direction = direction / np.linalg.norm(direction)

        valid_ind = np.where(image[:, int(sp[0])] != 0)[0]
        if len(valid_ind) > 0:
            valid_ind = valid_ind[0]
        else:
            msg = "Something went wrong for the probe location. Please contact maintainers."

        if int(sp[1]) < valid_ind:
            sign_flag = -1
        elif int(sp[1]) > valid_ind:
            sign_flag = 1
        else:
            sign_flag = 0

        stop_steps = 0
        temp = sp + 0
        while int(temp[1]) != valid_ind:
            stop_steps += 1
            temp = sp + sign_flag * stop_steps * direction
            valid_ind = np.where(image[:, int(temp[0])] != 0)[0]
            if len(valid_ind) > 0:
                valid_ind = valid_ind[0]
            else:
                msg = "Something went wrong for the probe location. Please contact maintainers."
                break

        sp = sp + sign_flag * stop_steps * direction

    p2 = np.array([sp, ep])
    return p2, msg


def line_fit(points):
    points = np.asarray(points)
    sort_order = np.argsort(points[:, 2])[::-1]
    points = points[sort_order, :]
    avg = np.mean(points, 0)
    subtracted = points - avg
    u, s, vh = np.linalg.svd(subtracted)
    direction = vh[0, :] / np.linalg.norm(vh[0, :])
    p1 = points[0, :]
    p2 = points[-1, :]
    sp = get_closest_point_to_line(avg, direction, p1)
    ep = get_closest_point_to_line(avg, direction, p2)
    return sp, ep, avg, direction


def get_angles(direction):
    direction = direction / np.linalg.norm(direction)

    vertical_vec = np.array([0, 0, 1])
    ap_proj = direction.copy()
    ap_proj[0] = 0
    ap_proj = ap_proj / np.linalg.norm(ap_proj)
    ml_proj = direction.copy()
    ml_proj[1] = 0
    ml_proj = ml_proj / np.linalg.norm(ml_proj)

    ap_val = math.acos(np.max([np.min([np.dot(ap_proj, vertical_vec), 1]), -1]))
    ml_val = math.acos(np.max([np.min([np.dot(ml_proj, vertical_vec), 1]), -1]))

    ap_angle = np.degrees(ap_val)
    ml_angle = np.degrees(ml_val)

    if ap_angle > 90:
        ap_angle = 180 - ap_angle
    if ml_angle > 90:
        ml_angle = 180 - ml_angle

    return ap_angle, ml_angle


def get_tilt_info(sp, ep):
    if ep[1] < sp[1]:
        ap_tilt = "posterior"
    elif ep[1] > sp[1]:
        ap_tilt = "anterior"
    else:
        ap_tilt = "no tilt"

    if sp[0] > 0:
        if ep[0] < sp[0]:
            ml_tilt = "medial"
        elif ep[0] > sp[0]:
            ml_tilt = "lateral"
        else:
            ml_tilt = "no tilt"
    elif sp[0] < 0:
        if ep[0] < sp[0]:
            ml_tilt = "lateral"
        elif ep[0] > sp[0]:
            ml_tilt = "medial"
        else:
            ml_tilt = "no tilt"
    else:
        if ep[0] != sp[0]:
            ml_tilt = "lateral"
        else:
            ml_tilt = "no tilt"
    return ap_tilt, ml_tilt


def get_tilt_sign(sp, ep):
    if ep[1] < sp[1]:
        ap_tilt = -1
    elif ep[1] > sp[1]:
        ap_tilt = 1
    else:
        ap_tilt = 0

    if ep[0] < sp[0]:
        ml_tilt = -1
    elif ep[0] > sp[0]:
        ml_tilt = 1
    else:
        ml_tilt = 0
    return ap_tilt, ml_tilt


def pandas_to_str(label_name, label_ano, length, channels):
    df = pd.DataFrame(
        {
            "Brain Regions": label_name,
            "Ano": label_ano,
            "Probe Length": length,
            "Probe Channels": channels,
        }
    )
    return df.to_string(col_space=30, justify="justify")


def correct_start_pnt(
    label_data, start_pnt, start_vox, direction, bregma, verbose=False
):
    error_index = 0
    # print('start_vox', start_vox)
    direction = direction / np.linalg.norm(direction)
    check_vec = label_data[int(start_vox[0]), int(start_vox[1]), :]
    top_vox = np.where(check_vec != 0)[0]
    # print(np.where(check_vec != 0))
    if len(top_vox) == 0:
        print("wrong")
    else:
        top_vox = top_vox[-1]
    check_vox = start_vox.astype(int)
    new_sp = start_pnt.copy()

    # if int(start_vox[2]) < top_vox:
    #     steps = 0
    #     while label_data[check_vox[0], check_vox[1], check_vox[2]] != 0:
    #         steps += 1
    #         new_sp = start_pnt - steps * direction
    #         check_vox = new_sp + bregma
    #         check_vox = check_vox.astype(int)
    #         if steps == 1000:
    #             raise Exception('higher limit, please contact maintainer')
    # elif int(start_vox[2]) > top_vox:
    #     steps = 0
    #     while label_data[check_vox[0], check_vox[1], check_vox[2]] == 0:
    #         steps += 1
    #         new_sp = start_pnt + steps * direction
    #         check_vox = new_sp + bregma
    #         check_vox = check_vox.astype(int)
    #         if steps == 1000:
    #             raise Exception('higher limit, please contact maintainer')
    # else:
    #     new_sp = start_pnt
    enter_label = label_data[check_vox[0], check_vox[1], check_vox[2]]
    z_diff = new_sp[2] + bregma[2] - top_vox
    if z_diff >= 1:
        sign_flag = -1
        if enter_label != 0:
            error_index = 10
            return new_sp, error_index
    elif z_diff < 0:
        sign_flag = 1
        if enter_label == 0:
            error_index = 11
            return new_sp, error_index
    else:
        sign_flag = 0

    # ToDo: the following code only works when there is no 0 label inside the brain (label data)
    # that means label 0 only indicates the area outside the brain
    # if some atlas uses label 0 to indicate the unspecified area inside the brain, this code may have problems
    # should make a pseudo label index (xxxx) inside the brain for unspecified regions ===> need to re-process atlas

    stop_steps = 0
    if sign_flag != 0:
        enter_condition = (enter_label == 0) if z_diff >= 1 else (enter_label != 0)
        while enter_condition:
            stop_steps += 1
            new_sp = start_pnt - sign_flag * stop_steps * direction
            check_vox = new_sp + bregma
            check_vox = check_vox.astype(int)
            if np.any(check_vox >= np.ravel(label_data.shape)) or np.any(check_vox) < 0:
                error_index = 12
                break
            enter_label = label_data[check_vox[0], check_vox[1], check_vox[2]]
            enter_condition = (enter_label == 0) if z_diff >= 1 else (enter_label != 0)
        if error_index != 0:
            return new_sp, error_index

        new_sp = (
            start_pnt - sign_flag * (stop_steps - 1) * direction
            if z_diff < 0
            else new_sp
        )
        if verbose:
            print("correct enter pnt with {} steps".format(stop_steps))
            # print('old sp', start_pnt)
            # print('new pc_sp', new_sp)

    return new_sp, error_index


def correct_end_point(
    sp, ep, direction, vox_size, tip_length, max_probe_length, verbose=False
):
    probe_length_with_tip = np.sqrt(np.sum((sp - ep) ** 2)) * vox_size  # in um
    probe_length_without_tip = probe_length_with_tip - tip_length  # in um
    if max_probe_length is not None:
        if probe_length_with_tip > max_probe_length:
            probe_length_with_tip = max_probe_length
            probe_length_without_tip = probe_length_with_tip - tip_length
            new_ep = sp + direction * probe_length_with_tip / vox_size
        else:
            new_ep = ep
    else:
        new_ep = ep
    if verbose:
        print("old pc_ep", ep)
        print("corrected pc_ep", new_ep)
        print("old probe length with tip", probe_length_with_tip)
        print("corrected probe length with tip", probe_length_with_tip)
    return new_ep, probe_length_with_tip, probe_length_without_tip


def check_parallel_to_z(direction):
    is_parallel = False
    if abs(abs(direction[2]) - 1) < 1e-6:
        is_parallel = True
    return is_parallel


def angle_between_2vectors(vector1, vector2):
    unit_vector_1 = vector1 / np.linalg.norm(vector1)
    unit_vector_2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle


def get_probe_info(probe_type):
    if probe_type == 0:
        tip_length = 175
        channel_size = 20
        channel_number_in_banks = (384, 384, 192)
    elif probe_type == 1:
        tip_length = 175
        channel_size = 15
        channel_number_in_banks = (384, 384, 192)
    else:
        tip_length = 0
        channel_size = None
        channel_number_in_banks = None
    return tip_length, channel_size, channel_number_in_banks


def group_labels(fine_label_mat, verbose=False):
    n_row, n_column = fine_label_mat.shape
    group_mat = np.zeros(fine_label_mat.shape)
    group_mat[:] = np.nan
    group_id = 0
    group_id_label = []

    previous_row = np.zeros(n_column)
    previous_row[:] = np.nan
    for base_level in range(0, n_row):
        check_label = fine_label_mat[base_level]
        unique_label = np.unique(check_label)
        for da_label in unique_label:
            if da_label in previous_row:
                continue
            add_row = 0
            row_index = add_row + base_level
            label_check = fine_label_mat[row_index] == da_label
            group_mat[row_index, label_check] = group_id
            while np.any(label_check):
                add_row += 1
                row_index = add_row + base_level
                if row_index == n_row:
                    break
                label_check = fine_label_mat[row_index] == da_label
                group_mat[row_index, label_check] = group_id
            group_id_label.append(da_label)
            group_id += 1
        previous_row = fine_label_mat[base_level]

    if verbose:
        print("group mat nan index", np.where(np.isnan(group_mat)))
        print("group id label", group_id_label)
        print("group_mat")
        indexes = np.arange(0, len(group_mat), 20)
        for i in range(len(indexes) - 1):
            print(group_mat[indexes[i] : indexes[i + 1]])
        print(group_mat[indexes[-1] : len(group_mat)])
        print(len(group_mat))

    return group_mat, group_id_label


def get_column_grouped_info(group_column, column_bound):
    diff_labels = np.where(np.diff(group_column) != 0)[0]
    if len(diff_labels) != 0:
        change_index = np.append(np.array([0]), diff_labels + 1)
        group_id = group_column[change_index]
        start_loc = column_bound[change_index]
        end_loc = column_bound[
            np.append(change_index[1:], np.array([len(group_column)]))
        ]
    else:
        group_id = group_column[0]
        start_loc = column_bound[0]
        end_loc = column_bound[-1]

    group_column_length = np.abs(end_loc - start_loc)

    return group_id, start_loc, end_loc, group_column_length


def get_vis_data(group_mat, column_loc, sites_loc, sites_line_count, vox_size):
    n_column = len(column_loc)

    p_bounds = np.array([0])
    p_bounds = np.append(
        p_bounds, column_loc[0][:-1, 0] + np.diff(column_loc[0][:, 0]) * 0.5
    )
    p_bounds = np.append(p_bounds, column_loc[0][-1, 0])

    vis_data = []
    column_group_length = []
    column_n_sites = []
    for i in range(n_column):
        sites_column = sites_loc[i][:, 0]
        # print(sites_column)
        if sites_line_count is None:
            line_counts = np.ones(len(sites_column))
        else:
            line_counts = np.ravel(sites_line_count)
        group_column = group_mat[:, i]
        group_id, start_loc, end_loc, gc_length = get_column_grouped_info(
            group_column, p_bounds
        )
        n_sites_group = []
        for j in range(len(group_id)):
            valid_sites_lines = np.logical_and(
                sites_column < end_loc[j], sites_column >= start_loc[j]
            )
            n_sites_group.append(np.sum(valid_sites_lines * line_counts))
        # print('n_sites_group', n_sites_group)
        vis_column = {
            "group_id": group_id,
            "start_loc": start_loc / vox_size,
            "end_loc": end_loc / vox_size,
            "sites": sites_column / vox_size,
        }
        column_group_length.append(gc_length)
        column_n_sites.append(np.ravel(n_sites_group))
        vis_data.append(vis_column)
    # print(vis_data)
    n_column = len(vis_data)
    unique_groups = np.unique(group_mat)
    group_length = []
    group_n_sites = []
    for group_ind in unique_groups:
        temp = []
        temp_sites = []
        for i in range(n_column):
            valid_ind = np.where(vis_data[i]["group_id"] == group_ind)[0]
            if len(valid_ind) != 0:
                valid_length = np.sum(column_group_length[i][valid_ind])
                valid_sites = np.sum(column_n_sites[i][valid_ind])
                temp.append(valid_length)
                temp_sites.append(valid_sites)
        group_length.append(np.sum(temp) / n_column)
        group_n_sites.append(np.sum(temp_sites))

    text_loc = []
    for group_ind in unique_groups:
        inds = np.where(group_mat == group_ind)[0]
        low_level = np.min(inds)
        high_level = np.max(inds)
        text_loc.append(
            (
                p_bounds[low_level]
                + 0.7 * (p_bounds[high_level + 1] - p_bounds[low_level])
            )
            / vox_size
        )
    return vis_data, group_length, group_n_sites, text_loc


def get_n_sites_in_region(gr_start, gr_end, plot_sites_column):
    n_groups, n_column = gr_start.shape

    region_site_num = []
    for i in range(n_groups):
        n_sites = 0
        for j in range(n_column):
            if np.isnan(gr_start[i, j]):
                continue
            valid_ind = np.logical_and(
                plot_sites_column[j] >= gr_start[i, j],
                plot_sites_column[j] < gr_end[i, j],
            )
            n_sites += np.sum(valid_ind)
        region_site_num.append(n_sites)

    return region_site_num


def get_label_name(label_info, region_label):
    label_names = []
    label_acronym = []
    label_color = []
    for i in range(len(region_label)):
        if region_label[i] == 0:
            label_names.append(" ")
            label_acronym.append(" ")
            label_color.append((128, 128, 128))
        else:
            da_ind = np.where(np.ravel(label_info["index"]) == region_label[i])[0][0]
            label_names.append(label_info["label"][da_ind])
            label_acronym.append(label_info["abbrev"][da_ind])
            label_color.append(label_info["color"][da_ind])

    label_color = np.asarray(label_color)
    return label_names, label_acronym, label_color


def get_sites_loc_related_to_base_center(probe_settings, probe_length_without_tip_um):
    probe_type_name = probe_settings["probe_type_name"]
    probe_thickness_um = probe_settings["probe_thickness"]
    per_max_sites = probe_settings["per_max_sites"]
    sites_distance_um = probe_settings["sites_distance"]
    x_bias_um = probe_settings["x_bias"]
    y_bias_um = probe_settings["y_bias"]

    if probe_type_name == "Tetrode":
        sites_loc = [np.array([[0, 0, 0]]) for _ in range(4)]
    else:
        sites_loc = []  # inverse r-vals, u-vals, n-vals
        n_val = 0.5 * probe_thickness_um
        for i in range(len(x_bias_um)):
            possible_n_sites = int(
                (probe_length_without_tip_um - y_bias_um[i]) / sites_distance_um[i]
            )
            if possible_n_sites <= per_max_sites[i]:
                r_vals = (
                    np.arange(possible_n_sites) * sites_distance_um[i] + y_bias_um[i]
                )
            else:
                r_vals = (
                    np.arange(per_max_sites[i]) * sites_distance_um[i] + y_bias_um[i]
                )
            num_sites = len(r_vals)
            u_vals = np.repeat(x_bias_um[i], num_sites)
            n_vals = np.repeat(n_val, num_sites)
            temp = np.stack([r_vals, u_vals, n_vals], axis=1)
            sites_loc.append(temp)
    return sites_loc


def merge_sites_into_line(sites_loc):
    y_vec = []
    for i in range(len(sites_loc)):
        y_vec.append(sites_loc[i][:, 0])
    y_vec = np.concatenate(y_vec)
    y_vals = np.unique(y_vec)
    sites_loc_line = []
    sites_count = []
    temp = []
    for i in range(len(y_vals)):
        n = len(np.where(y_vec == y_vals[i])[0])
        temp.append(np.array([y_vals[i], 0, 0]))
        sites_count.append(n)
    temp = np.asarray(temp)
    sites_loc_line.append(temp)

    return sites_loc_line, sites_count


def get_pnt_from_loc(sct, loc, n_vec, u_vec, r_vec, bregma, vox_size):
    pnt = []
    pnt_vox = []
    for i in range(len(loc)):
        temp = [
            sct * vox_size
            + r_vec * loc[i][ind, 0]
            + u_vec * loc[i][ind, 1]
            + n_vec * loc[i][ind, 2]
            for ind in range(len(loc[i]))
        ]
        temp = np.asarray(temp)
        temp = temp / vox_size
        pnt.append(temp)
        vox_temp = temp + bregma
        vox_temp = vox_temp.astype(int)
        pnt_vox.append(vox_temp)

    return pnt, pnt_vox


def get_column_loc(
    sites_loc_to_base, probe_length_without_tip_um, vxsize_um, verbose=False
):
    sites_r_vals = []
    for i in range(len(sites_loc_to_base)):
        sites_r_vals.append(sites_loc_to_base[i][:, 0])
    sites_r_vals = np.concatenate(sites_r_vals)
    uni_sites_r_vals = np.unique(sites_r_vals)

    if len(uni_sites_r_vals) > 1:
        if uni_sites_r_vals[0] == 0:
            vis_r_vals = np.arange(uni_sites_r_vals[0], uni_sites_r_vals[1], vxsize_um)
            for i in range(1, len(uni_sites_r_vals) - 1):
                vis_r_vals = np.append(
                    vis_r_vals,
                    np.arange(uni_sites_r_vals[i], uni_sites_r_vals[i + 1], vxsize_um),
                )
        else:
            vis_r_vals = np.arange(0, uni_sites_r_vals[0], vxsize_um)
            for i in range(len(uni_sites_r_vals) - 1):
                vis_r_vals = np.append(
                    vis_r_vals,
                    np.arange(uni_sites_r_vals[i], uni_sites_r_vals[i + 1], vxsize_um),
                )

        if uni_sites_r_vals[-1] != probe_length_without_tip_um:
            vis_r_vals = np.append(
                vis_r_vals,
                np.arange(uni_sites_r_vals[-1], probe_length_without_tip_um, vxsize_um),
            )
        vis_r_vals = np.append(vis_r_vals, probe_length_without_tip_um)
    else:
        vis_r_vals = np.arange(0, probe_length_without_tip_um, vxsize_um)

    column_loc = []
    for i in range(len(sites_loc_to_base)):
        temp = np.zeros((len(vis_r_vals), 3))
        temp[:, 0] = vis_r_vals
        temp[:, 1] = sites_loc_to_base[i][0, 1]
        temp[:, 2] = sites_loc_to_base[i][0, 2]
        column_loc.append(temp)

    if verbose:
        print("uni_sites_r_vals")
        print(uni_sites_r_vals)

        print("vis_r_vals")
        print(vis_r_vals)

        print("column_loc")
        print(column_loc)

    return column_loc


def get_loc_related_to_start(loc_to_base, probe_length_without_tip_um):
    loc_to_start = []  # inverse r-vals, u-vals, n-vals
    for i in range(len(loc_to_base)):
        temp_loc = loc_to_base[i].copy()
        temp_loc[:, 0] = probe_length_without_tip_um - temp_loc[:, 0]
        loc_to_start.append(temp_loc)
    return loc_to_start


def get_fine_label_matrix(column_vox, label_data, verbose=False):
    fine_label_mat = []
    for i in range(len(column_vox)):
        fine_label_mat.append(
            label_data[column_vox[i][:, 0], column_vox[i][:, 1], column_vox[i][:, 2]]
        )
    fine_label_mat = np.asarray(fine_label_mat).T

    if verbose:
        # print out fine label matrix
        indexes = np.arange(0, len(fine_label_mat), 20)
        for i in range(len(indexes) - 1):
            print(fine_label_mat[indexes[i] : indexes[i + 1]])
        print(fine_label_mat[indexes[-1] : len(fine_label_mat)])

    return fine_label_mat


def calculate_probe_info(
    data_list,
    pieces_names,
    label_data,
    label_info,
    vxsize_um,
    probe_settings,
    merge_sites,
    bregma,
    site_face,
    n_hat,
    pre_plan,
):
    """

    :param data: 3d coordinates for all the points
    :param label_data: original brain region segmentation
    :param label_info:
    :param vxsize_um:
    :param tip_length:
    :param channel_size:
    :param bregma:
    :return:
    """

    data_dict = None
    data = data_list[0]
    for i in range(1, len(data_list)):
        data = np.vstack([data, data_list[i]])
    # print('data', data)

    # start_pnt and end_pnt are coordinates related to the given Bregma
    probe_type_name = probe_settings["probe_type_name"]
    probe_max_length_um = probe_settings["probe_length"]
    tip_length_um = probe_settings["tip_length"]

    # get direction and probe center start and end (pc - probe center)
    pc_start_pnt, pc_end_pnt, avg, direction = line_fit(data)
    direction = pc_end_pnt - pc_start_pnt
    direction = direction / np.linalg.norm(direction)
    pc_start_vox = pc_start_pnt + bregma
    pc_end_vox = pc_end_pnt + bregma

    # print('direction', direction)

    # get angels
    ap_angle, ml_angle = get_angles(direction)
    # print('ap_angle', ap_angle)
    # print(ml_angle)

    # print('check-start_pnt', pc_start_pnt)
    # # print(pc_start_vox)
    # correct probe center start point
    pc_sp, error_index = correct_start_pnt(
        label_data, pc_start_pnt, pc_start_vox, direction, bregma, verbose=True
    )
    if error_index != 0:
        return data_dict, 15

    pc_ep, probe_length_with_tip_um, probe_length_without_tip_um = correct_end_point(
        pc_sp,
        pc_end_pnt,
        direction,
        vxsize_um,
        tip_length_um,
        probe_max_length_um,
        verbose=True,
    )

    pv_sp = pc_sp + bregma
    pv_sp = pv_sp.astype(int)
    pv_ep = pc_ep + bregma
    pv_ep = pv_ep.astype(int)

    enter_coords = pc_sp * vxsize_um
    end_coords = pc_ep * vxsize_um

    # print('')

    dv = (pc_sp[2] - pc_ep[2]) * vxsize_um
    ap_tilt, ml_tilt = get_tilt_info(pc_sp, pc_ep)

    if pre_plan:
        r_hat = direction.copy()
        u_hat = np.cross(n_hat, r_hat)

        n_vec, u_vec = get_vector_according_to_site_face(n_hat, u_hat, site_face)
    else:
        r_hat, u_vec, n_vec = calculate_vector_according_to_site_face(
            direction, site_face
        )

    # sites location, list of  [(n_sites, 3),...], in um
    sites_loc_to_base_temp = get_sites_loc_related_to_base_center(
        probe_settings, probe_length_without_tip_um
    )
    # print('sites_loc_to_base')
    # print(sites_loc_to_base)

    if merge_sites or probe_type_name == "Tetrode":
        sites_loc_to_base, sites_line_count = merge_sites_into_line(
            sites_loc_to_base_temp
        )
    else:
        sites_loc_to_base = sites_loc_to_base_temp.copy()
        sites_line_count = None

    # column location list of [(n_locs, 3), ...], in um
    column_loc_to_base = get_column_loc(
        sites_loc_to_base, probe_length_without_tip_um, vxsize_um, verbose=False
    )

    # column location relative to the start, from bottom to top, in um
    column_loc = get_loc_related_to_start(
        column_loc_to_base, probe_length_without_tip_um
    )
    # print('column_loc')
    # print(column_loc)

    # column points (related to bregma) and column vox
    column_pnt, column_vox = get_pnt_from_loc(
        pc_sp, column_loc, n_vec, u_vec, r_hat, bregma, vxsize_um
    )
    # print('column_pnt')
    # print(column_pnt)
    # print('column_vox')
    # print(column_vox)

    fine_label_mat = get_fine_label_matrix(column_vox, label_data, verbose=False)
    group_mat, group_id_label = group_labels(fine_label_mat)
    label_names, label_acronym, label_color = get_label_name(label_info, group_id_label)

    if probe_type_name != "Tetrode":
        # sites loc related to the start, in um
        sites_loc = get_loc_related_to_start(
            sites_loc_to_base, probe_length_without_tip_um
        )
        # print(sites_loc)

        sites_pnt, sites_vox = get_pnt_from_loc(
            pc_sp, sites_loc, n_vec, u_vec, r_hat, bregma, vxsize_um
        )
        # print(sites_pnt)
        # print('sites_vox')
        # print(sites_vox)
    else:
        sites_pnt = [np.array([pc_ep])]
        sites_vox_temp = sites_pnt[0] + bregma
        sites_vox = [sites_vox_temp.astype(int)]

    sites_label = []
    for i in range(len(sites_vox)):
        sites_label.append(
            label_data[sites_vox[i][:, 0], sites_vox[i][:, 1], sites_vox[i][:, 2]]
        )

    # print('sites_label')
    # print(sites_label)

    vis_data, region_length, region_site_num, region_text_loc = get_vis_data(
        group_mat, column_loc_to_base, sites_loc_to_base, sites_line_count, vxsize_um
    )

    data_dict = {
        "object_name": "probe",
        "probe_type_name": probe_type_name,
        "data": data_list,
        "pieces_names": pieces_names,
        "ap_tilt": ap_tilt,
        "ml_tilt": ml_tilt,
        "insertion_coords_3d": pc_sp,
        "terminus_coords_3d": pc_ep,
        "direction": direction,
        "probe_length": probe_length_with_tip_um,
        "dv": dv,
        "ap_angle": ap_angle,
        "ml_angle": ml_angle,
        "insertion_coords": enter_coords,
        "insertion_vox": pv_sp,
        "terminus_coords": end_coords,
        "terminus_vox": pv_ep,
        "sites_label": sites_label,
        "sites_loc_b": sites_pnt,
        "sites_vox": sites_vox,
        "region_label": group_id_label,
        "region_length": region_length,
        "region_sites": region_site_num,
        "text_loc": region_text_loc,
        "label_name": label_names,
        "label_acronym": label_acronym,
        "label_color": label_color,
        "vis_data": vis_data,
    }
    return data_dict, error_index


def get_pre_multi_shank_vis_base(x_vals, y_vals):
    x_vals = np.ravel(x_vals)
    y_vals = np.ravel(y_vals)
    valid_ind = np.where(y_vals == 0)[0]
    if len(valid_ind) != 0:
        base_loc = x_vals[valid_ind]
    else:
        base_loc = np.array([0])
    # if site_face in [0, 1]:
    #     valid_ind = np.where(y_vals == 0)[0]
    #     if len(valid_ind) != 0:
    #         base_loc = x_vals[valid_ind]
    #     else:
    #         base_loc = np.array([0])
    # else:
    #     valid_ind = np.where(x_vals == 0)[0]
    #     if len(valid_ind) != 0:
    #         base_loc = y_vals[valid_ind]
    #     else:
    #         base_loc = np.array([0])
    return base_loc


def get_vector_according_to_site_face(n_hat, u_hat, site_face):
    # for pre-plan - 2d plan
    if site_face == 0:
        # site face out, facing to you
        n_vec = n_hat.copy()
        u_vec = u_hat.copy()
    elif site_face == 1:
        # site face in, facing away from you
        n_vec = -n_hat
        u_vec = -u_hat
    elif site_face == 2:
        # site face left, facing to you left-hand side
        n_vec = -u_hat
        u_vec = n_hat.copy()
    elif site_face == 3:
        # site face right, facing to you right-hand side
        n_vec = u_hat.copy()
        u_vec = -n_hat
    else:
        raise ValueError("no such site face, stupid!!!!!!")

    # print('n_vec', n_vec)
    # print('u_vec', u_vec)

    return n_vec, u_vec


def calculate_vector_according_to_site_face(direction, site_face):
    # for after surgery
    r_hat = direction.copy()
    if check_parallel_to_z(direction):
        if site_face == 0:
            n_hat = np.array([0, 1, 0])
            u_hat = np.cross(n_hat, r_hat)
        elif site_face == 1:
            n_hat = np.array([0, -1, 0])
            u_hat = np.cross(n_hat, r_hat)
        elif site_face == 2:
            n_hat = np.array([-1, 0, 0])
            u_hat = np.cross(n_hat, r_hat)
        elif site_face == 3:
            n_hat = np.array([1, 0, 0])
            u_hat = np.cross(n_hat, r_hat)
        else:
            n_hat = None
            u_hat = None
            print("Site face can only be 0-Up, 1-Down, 2-Left, 3-Right.")
    else:
        if site_face == 0:
            t_hat = np.array([-r_hat[1], r_hat[0], 0])
            u_hat = t_hat / np.linalg.norm(t_hat)
            n_hat = np.cross(r_hat, u_hat)
        elif site_face == 1:
            t_hat = np.array([r_hat[1], r_hat[0], 0])
            u_hat = t_hat / np.linalg.norm(t_hat)
            n_hat = np.cross(r_hat, u_hat)
        elif site_face == 2:
            t_hat = np.array([-r_hat[1], r_hat[0], 0])
            n_hat = t_hat / np.linalg.norm(t_hat)
            u_hat = np.cross(n_hat, r_hat)
        elif site_face == 3:
            t_hat = np.array([r_hat[1], r_hat[0], 0])
            n_hat = t_hat / np.linalg.norm(t_hat)
            u_hat = np.cross(n_hat, r_hat)
        else:
            n_hat = None
            u_hat = None
            print("Site face can only be 0-Up, 1-Down, 2-Left, 3-Right.")
        # print('t_hat', t_hat)
        # print(n_hat)
        # print(u_hat)
    return r_hat, u_hat, n_hat


def get_center_lines(
    pnts, r_hat, n_hat, u_hat, x_vals, y_vals, length, site_face, vox_size
):
    n_vec, u_vec = get_vector_according_to_site_face(n_hat, u_hat, site_face)

    # print(x_vals)
    # print(y_vals)

    line_data = []
    for i in range(len(x_vals)):
        s_val = u_vec * x_vals[i] + n_vec * y_vals[i]
        # print('s_val', s_val)
        temp = [s_val, s_val + r_hat * length * vox_size]
        # print(temp)
        line_data.append(pnts[0] + np.asarray(temp) / vox_size)

    return line_data


def get_pre_ms_vis_base(multi_shanks, site_face, atlas_display):
    if multi_shanks is None:
        base_loc = np.array([0])
    else:
        if atlas_display == "sagittal":
            if site_face not in [0, 1]:
                base_loc = np.ravel(multi_shanks)
            else:
                base_loc = np.array([0])
        else:
            if site_face in [0, 1]:
                base_loc = np.ravel(multi_shanks)
            else:
                base_loc = np.array([0])
    return base_loc


def get_pre_mp_vis_base(base_loc, atlas_display):
    if atlas_display == "sagittal":
        valid_ind = np.where(base_loc[:, 1] == 0)[0]
        if len(valid_ind) == 0:
            vis_base = np.array([0])
        else:
            vis_base = base_loc[valid_ind, 1]
    else:
        valid_ind = np.where(base_loc[:, 0] == 0)[0]
        if len(valid_ind) == 0:
            vis_base = np.array([0])
        else:
            vis_base = base_loc[valid_ind, 0]
    return vis_base


class Probe(object):
    def __init__(self):
        self.probe_length = None
        self.tip_length = None
        self.site_width = None
        self.site_height = None
        self.total_sites = None
        self.site_number_in_banks = None
        self.sites_distance = None
        self.probe_type = None
        self.x_bias = None
        self.per_max_sites = None
        self.sites_distance = None
        self.y_bias = None
        self.probe_type_name = None
        self.probe_thickness = None
        self.multi_shanks = None
        self.exist_probes = None
        self.faces = None

        self.set_np1()

    def extend_exist_probes(self, prb):
        pass

    def get_exist_probes(self):
        self.exist_probes = {"NP1.0": None, "NP2.0": None, "Tetrode": None}

    def set_np2(self):
        self.probe_type = 1
        self.probe_type_name = "NP2.0"
        self.probe_thickness = 24
        self.probe_length = 10000
        self.tip_length = 175
        self.site_width = 16
        self.site_height = 15
        self.x_bias = [-8, 16]
        self.per_max_sites = [480, 480]
        self.sites_distance = [15, 15]
        self.y_bias = [7.5, 7.5]
        self.site_number_in_banks = (384, 384, 192)
        self.multi_shanks = [-375, -125, 125, 375]
        self.faces = "Front"

    def set_np1(self):
        self.probe_type = 0
        self.probe_type_name = "NP1.0"
        self.probe_thickness = 24
        self.probe_length = 10000
        self.tip_length = 175
        self.site_width = 16
        self.site_height = 20
        self.x_bias = [-16, -8, 8, 16]
        self.per_max_sites = [240, 240, 240, 240]
        self.sites_distance = [40, 40, 40, 40]
        self.y_bias = [30, 10, 30, 10]
        self.site_number_in_banks = (384, 384, 192)
        self.multi_shanks = None
        self.faces = "Front"

    def set_tetrode(self):
        self.probe_type = 3
        self.probe_type_name = "Tetrode"
        self.probe_thickness = 0
        self.probe_length = None
        self.tip_length = 0
        self.site_width = None
        self.site_height = None
        self.x_bias = 0
        self.per_max_sites = 4
        self.sites_distance = 0
        self.y_bias = 0
        self.site_number_in_banks = None
        self.multi_shanks = None
        self.faces = None

    def set_linear_silicon(self, pss):
        self.probe_type = 2
        self.probe_type_name = "Linear-Silicon"
        self.probe_length = pss["probe_length"]
        self.probe_thickness = pss["probe_thickness"]
        self.tip_length = pss["tip_length"]
        self.site_width = pss["site_width"]
        self.site_height = pss["site_height"]

        self.per_max_sites = pss["per_max_sites"]
        self.sites_distance = pss["sites_distance"]
        self.x_bias = pss["x_bias"]
        self.y_bias = pss["y_bias"]
        self.site_number_in_banks = None
        self.multi_shanks = None
        self.faces = "Front"

    def get_settings(self):
        data = {
            "probe_type": self.probe_type,
            "probe_type_name": self.probe_type_name,
            "probe_thickness": self.probe_thickness,
            "probe_length": self.probe_length,
            "tip_length": self.tip_length,
            "site_height": self.site_height,
            "site_width": self.site_width,
            "per_max_sites": self.per_max_sites,
            "sites_distance": self.sites_distance,
            "x_bias": self.x_bias,
            "y_bias": self.y_bias,
            "site_number_in_banks": self.site_number_in_banks,
            "multi_shanks": self.multi_shanks,
        }
        return data

    def probe_faces_changed(self, face_direction):
        self.faces = face_direction

    def get_multi_shank_3d_base(self):
        if self.multi_shanks is None:
            return
        if self.faces in ["Front", "Back"]:
            points3 = np.zeros((len(self.multi_shanks), 3))
            points3[:, 1] = self.multi_shanks
        else:
            points3 = np.zeros((len(self.multi_shanks), 3))
            points3[:, 2] = self.multi_shanks
        return points3


class MultiProbes(object):
    def __init__(self):
        self.x_vals = None
        self.y_vals = None
        self.faces = None

    def set_multi_probes(self, multi_settings):
        self.x_vals = multi_settings["x_vals"]
        self.y_vals = multi_settings["y_vals"]
        self.faces = multi_settings["faces"]

    def get_multi_settings(self):
        if self.x_vals is None:
            multi_settings = None
        else:
            if not isinstance(self.x_vals, list):
                x_vals = self.x_vals.tolist()
            else:
                x_vals = self.x_vals.copy()

            if not isinstance(self.y_vals, list):
                y_vals = self.y_vals.tolist()
            else:
                y_vals = self.y_vals.copy()

            if not isinstance(self.faces, list):
                faces = self.faces.tolist()
            else:
                faces = self.faces.copy()
            multi_settings = {"x_vals": x_vals, "y_vals": y_vals, "faces": faces}
        return multi_settings

    def check_multi_settings(self):
        x_unique = np.unique(self.x_vals)

        for x_val in x_unique:
            v_ind = np.where(np.ravel(self.x_vals) == x_val)[0]
            if len(v_ind) > 1:
                if len(np.unique(np.ravel(self.y_vals)[v_ind])) != len(v_ind):
                    msg = "There are at least 2 probes located at the same location."
                    return msg

        return

    # def get_base_loc_3d(self, multi_shank):
    #     if multi_shank is None:
    #         points3 = np.zeros((len(self.x_vals), 3))
    #         points3[:, 1] = self.x_vals
    #         points3[:, 2] = self.y_vals
    #     else:
    #         n_shank = len(multi_shank)
    #         points3 = []
    #         if self.faces[0] in ['Top', 'Bottom']:
    #             for i in range(len(self.x_vals)):
    #                 p3 = np.zeros((n_shank, 3))
    #                 p3[:, 1] = multi_shank + self.x_vals[i]
    #                 p3[:, 2] = self.y_vals[i]
    #                 points3.append(p3)
    #         else:
    #             for i in range(len(self.x_vals)):
    #                 p3 = np.zeros((n_shank, 3))
    #                 p3[:, 1] = self.x_vals[i]
    #                 p3[:, 2] = multi_shank + self.y_vals[i]
    #                 points3.append(p3)
    #         points3 = np.concatenate(points3)
    #     return points3
    #
    # def get_base_loc_2d(self, multi_shank):
    #     if multi_shank is None:
    #         points2 = np.zeros((len(self.x_vals), 2))
    #         points2[:, 0] = self.x_vals
    #         points2[:, 1] = self.y_vals
    #     else:
    #         n_shank = len(multi_shank)
    #         points2 = []
    #         if self.faces[0] in ['Top', 'Bottom']:
    #             for i in range(len(self.x_vals)):
    #                 p2 = np.zeros((n_shank, 2))
    #                 p2[:, 0] = multi_shank + self.x_vals[i]
    #                 p2[:, 1] = self.y_vals[i]
    #                 points2.append(p2)
    #         else:
    #             for i in range(len(self.x_vals)):
    #                 p2 = np.zeros((n_shank, 2))
    #                 p2[:, 0] = self.x_vals[i]
    #                 p2[:, 1] = multi_shank + self.y_vals[i]
    #                 points2.append(p2)
    #         points2 = np.concatenate(points2)
    #     return points2
