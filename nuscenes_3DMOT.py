from dataset.nuscenes_dataset import nuscenesTrackingDataset
from dataset.kitti_data_base import velo_to_cam
from tracker.tracker import Tracker3D
import time
import tqdm
import os
from tracker.config import cfg, cfg_from_yaml_file
from tracker.box_op import *
import numpy as np
import argparse

from evaluation_HOTA.scripts.run_kitti import eval_kitti
from nuscenes.utils.geometry_utils import view_points,transform_matrix

from nuscenes import NuScenes
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.data_classes import TrackingMetrics, TrackingMetricDataList, TrackingConfig, TrackingBox, \
    TrackingMetricData
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion


def translate(self, x: np.ndarray) -> None:
    """
    Applies a translation.
    :param x: <np.float: 3, 1>. Translation in x, y, z direction.
    """
    self.center += x


def rotate(center, quaternion: Quaternion) -> None:
    """
    Rotates box.
    :param quaternion: Rotation to apply.
    """
    center = np.dot(quaternion.rotation_matrix, center)
    # orientation = quaternion * self.orientation

def track_one_seq_prev(seq_id,config):

    """
    tracking one sequence
    Args:
        seq_id: int, the sequence id
        config: config
    Returns: dataset: KittiTrackingDataset
             tracker: Tracker3D
             all_time: float, all tracking time
             frame_num: int, num frames
    """
    dataset_path = config.dataset_path
    detections_path = config.detections_path
    tracking_type = config.tracking_type
    detections_path += "/" + str(seq_id).zfill(4)
    verbose = config.verbose

    nusc = NuScenes(version=config.nusc_version, verbose=verbose, dataroot=dataset_path)

    for i in range(10):
        my_scene = nusc.scene[i]
        first_token = my_scene['first_sample_token']
        last_token = my_scene['last_sample_token']
        nbr_samples = my_scene['nbr_samples']
        current_token = first_token
        for i in range(nbr_samples):
            current_sample = nusc.get('sample', current_token)
            # file_idx = scene_token_list_timestamps[current_token][0]
            # file_name = os.path.join('/media/storage/nuscenes_labels_new', str(scene_name), str(file_idx) + '.txt')
            LIDAR_data = nusc.get('sample_data', current_sample['data']['LIDAR_TOP'])
            if config.load_data == "True":
                # nusc.render_sample_data(cam_front_data['token'])
                pcl_path = os.path.join(config.dataset_path, LIDAR_data['filename'])
                lidar_pcl = LidarPointCloud.from_file(pcl_path)

            ego_pose = nusc.get('ego_pose', LIDAR_data['ego_pose_token'])
            pose = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']),
                                       inverse=False)
            next_token = current_sample['next']
            current_token = next_token


    # Load data.
    if verbose:
        print('Initializing nuScenes tracking evaluation')
    pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, TrackingBox,
                                            verbose=verbose)
    gt_boxes = load_gt(nusc, self.eval_set, TrackingBox, verbose=verbose)

    # tracker = Tracker3D(box_type="Kitti", tracking_features=False, config = config)
    # dataset = KittiTrackingDataset(dataset_path, seq_id=seq_id, ob_path=detections_path,type=[tracking_type])

    all_time = 0
    frame_num = 0

    for i in range(len(dataset)):
        P2, V2C, points, image, objects, det_scores, pose = dataset[i]

        mask = det_scores>config.input_score
        objects = objects[mask]
        det_scores = det_scores[mask]

        start = time.time()

        tracker.tracking(objects[:,:7],
                             features=None,
                             scores=det_scores,
                             pose=pose,
                             timestamp=i)
        end = time.time()
        all_time+=end-start
        frame_num+=1

    return dataset, tracker, all_time, frame_num

def track_one_seq(seq_id,config,nusc,my_scene):

    """
    tracking one sequence
    Args:
        seq_id: int, the sequence id
        config: config
    Returns: dataset: KittiTrackingDataset
             tracker: Tracker3D
             all_time: float, all tracking time
             frame_num: int, num frames
    """
    dataset_path = config.dataset_path
    detections_path = config.detections_path
    tracking_type = config.tracking_type
    detections_path += "/" + str(seq_id).zfill(4)
    verbose = config.verbose
    dataset = nuscenesTrackingDataset(config, nusc, type=[config.tracking_type])
    tracker = Tracker3D(box_type="OpenPCDet", tracking_features=False, config=config)
    data_length = 0

    first_token = my_scene['first_sample_token']
    last_token = my_scene['last_sample_token']
    nbr_samples = my_scene['nbr_samples']
    current_token = first_token

    all_time = 0
    frame_num = 0
    data_length += nbr_samples

    for i in range(nbr_samples):
        current_sample = nusc.get('sample', current_token)
        camera_intrinsic, Ego2Cam, points, image, objects, det_scores, pose,_ = dataset[my_scene,current_token,current_sample]
        next_token = current_sample['next']
        current_token = next_token
        mask = det_scores > config.input_score
        objects = objects[mask]
        det_scores = det_scores[mask]

        start = time.time()

        #TODO Reset this after every scene
        tracker.tracking(objects[:, :7],
                         features=None,
                         scores=det_scores,
                         pose=None,
                         timestamp=i)
        end = time.time()
        all_time += end - start
        frame_num += 1

    dataset.add_total_samples(nbr_samples)
    return dataset, tracker, all_time, frame_num

def save_one_seq(dataset,
                 seq_id,
                 tracker,
                 config,
                 nusc,
                 my_scene):
    """
    saving tracking results
    Args:
        dataset: KittiTrackingDataset, Iterable dataset object
        seq_id: int, sequence id
        tracker: Tracker3D
    """

    save_path = config.save_path
    tracking_type = config.tracking_type
    s =time.time()
    tracks = tracker.post_processing(config)
    proc_time = s-time.time()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    first_token = my_scene['first_sample_token']
    last_token = my_scene['last_sample_token']
    nbr_samples = my_scene['nbr_samples']
    current_token = first_token

    all_time = 0
    frame_num = 0

    save_name = os.path.join(save_path,str(seq_id).zfill(4)+'.txt')

    frame_first_dict = {}
    for ob_id in tracks.keys():
        track = tracks[ob_id]

        for frame_id in track.trajectory.keys():

            ob = track.trajectory[frame_id]
            if ob.updated_state is None:
                continue
            if ob.score<config.post_score:
                continue

            if frame_id in frame_first_dict.keys():
                frame_first_dict[frame_id][ob_id]=(np.array(ob.updated_state.T),ob.score)
            else:
                frame_first_dict[frame_id]={ob_id:(np.array(ob.updated_state.T),ob.score)}

    with open(save_name,'w+') as f:
        for i in range(nbr_samples):
            current_sample = nusc.get('sample', current_token)
            camera_intrinsic, Ego2Cam, points, image, _, _, pose,lidar_sensor_pose = dataset[my_scene,current_token,current_sample]
            new_pose = np.mat(pose).I
            next_token = current_sample['next']
            current_token = next_token

            if i in frame_first_dict.keys():
                objects = frame_first_dict[i]

                for ob_id in objects.keys():
                    updated_state,score = objects[ob_id]

                    box_template = np.zeros(shape=(1,7))
                    box_template[0,0:3]=updated_state[0,0:3]
                    box_template[0,3:7]=updated_state[0,9:13]

                    box = register_bbs(box_template,new_pose)

                    box[:, 6] = -box[:, 6] - np.pi / 2
                    box[:, 2] -= box[:, 5] / 2
                    # box[:,0:3] = velo_to_cam(box[:,0:3],V2C)[:,0:3]

                    #global to ego-vehicle
                    np.matmul(pose,xyz)
                    #ego to calibrate sensor
                    np.matmul(lidar_sensor_pose, xyz)


                    box2d = view_points(bb3d_corners_nuscenes(box)[:,:3].T, P2, normalize=True)

                    print('%d %d %s -1 -1 -10 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                          % (i,ob_id,tracking_type,box2d[0][0],box2d[0][1],box2d[0][2],
                             box2d[0][3],box[5],box[4],box[3],box[0],box[1],box[2],box[6],score),file = f)

    return proc_time


def tracking_val_seq(arg):

    yaml_file = arg.cfg_file

    config = cfg_from_yaml_file(yaml_file,cfg)

    print("\nconfig file:", yaml_file)
    print("data path: ", config.dataset_path)
    print('detections path: ', config.detections_path)
    nusc = NuScenes(version=config.nusc_version, verbose=config.verbose, dataroot=config.dataset_path)

    save_path = config.save_path                       # the results saving path

    os.makedirs(save_path,exist_ok=True)

    seq_list = config.tracking_seqs    # the tracking sequences

    print("tracking seqs: ", seq_list)

    all_time,frame_num = 0,0

    for scene_no in tqdm.trange(len(seq_list)):
        current_scene = nusc.scene[scene_no]

        all_time = 0
        frame_num = 0
        seq_id = seq_list[scene_no]
        dataset,tracker, this_time, this_num = track_one_seq(seq_id,config,nusc,current_scene)
        proc_time = save_one_seq(dataset,seq_id,tracker,config,nusc,current_scene)

        all_time+=this_time
        # all_time+=proc_time
        frame_num+=this_num

    print("Tracking time: ", all_time)
    print("Tracking frames: ", frame_num)
    print("Tracking FPS:", frame_num/all_time)
    print("Tracking ms:", all_time/frame_num)

    # eval_kitti()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="config/online/pvrcnn_mot_nuscenes.yaml",
                        help='specify the config for tracking')
    args = parser.parse_args()
    tracking_val_seq(args)

