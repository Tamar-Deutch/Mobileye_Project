import numpy as np
import math

def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if(abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_prev_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container

def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(curr_container.EM)
    return norm_prev_pts, norm_curr_pts, R, foe, tZ

def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec

def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    normal_pts = []
    for pt in pts:
        pt_x = (pt[0] - pp[0]) / focal
        pt_y = (pt[1] - pp[1]) / focal
        normal_pts.append([pt_x,pt_y])
    return np.array(normal_pts)
def unnormalize(pts, focal, pp):
    unnormal_pts = []
    for pt in pts:
        pt_x = pt[0] * focal + pp[0]
        pt_y = pt[1] * focal + pp[1]
        unnormal_pts.append([pt_x,pt_y])
    return np.array(unnormal_pts)

def decompose(EM):
    R = EM[:3, :3]
    Tz = EM[:3, 3]
    foe = [Tz[0] / Tz[2], Tz[1] / Tz[2]]
    return R, foe, Tz[-1]

def rotate(pts, R):
    rotate_points = []
    for p in pts:
        prev = np.array([p[0],p[1],1])
        result=R @ prev
        rotate_points.append([result[0]/result[2],result[1]/result[2]])
    return np.array(rotate_points)

def distance(pt, m, n):
    x, y = pt[0], pt[1]
    return abs((m * x + n - y) / math.sqrt(m ** 2 + 1))


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    ex, ey = foe[0], foe[1]
    x, y = p[0], p[1]
    m = (ey - y) / (ex - x)
    n = (y * ex - ey * x) / (ex - x)
    min_distance = 999
    closest_point = None
    closest_index = None
    for i, pt in enumerate(norm_pts_rot):
        d = distance(pt, m, n)
        if min_distance > d:
            min_distance = d
            closest_index = i
            closest_point = pt
    return  closest_index,closest_point

def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z
    print((tZ*(foe[0] - p_rot[0]))/(p_curr[0] - p_rot[0]) if abs(p_curr[0] - p_rot[0]) > abs(p_curr[1] - p_rot[1]) else (tZ*(foe[1] - p_rot[1]))/(p_curr[1] - p_rot[1]))
    print((tZ * (math.sqrt((foe[0] - p_rot[0])** 2 + (foe[1] - p_rot[1])** 2 ) )/ (math.sqrt((p_curr[0] - p_rot[0])**2 +(p_curr[1] - p_rot[1])**2 ))))
    return (tZ*(foe[0] - p_rot[0]))/(p_curr[0] - p_rot[0]) if abs(p_curr[0] - p_rot[0]) > abs(p_curr[1] - p_rot[1]) else (tZ*(foe[1] - p_rot[1]))/(p_curr[1] - p_rot[1])
    # return abs((tZ * (math.sqrt((foe[0] - p_rot[0])** 2 + (foe[1] - p_rot[1])** 2 ) )/ (math.sqrt((p_curr[0] - p_rot[0])**2 +(p_curr[1] - p_rot[1])**2 ))))
    # ex, ey = foe[0], foe[1]
    # xc, yc = p_curr[0], p_curr[1]
    # xr, yr = p_rot[0], p_rot[1]
    # a=(xc - xr) + (yc - yr)
    # b=(xc - xr)/a
    # c=(yc - yr)/b
    # return ((tZ * (ex - xr)) / (xc - xr))*b + ((tZ * (ey - yr)) / (yc - yr))*c
