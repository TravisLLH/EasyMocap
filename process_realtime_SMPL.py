# import json
# import numpy as np
# from easymocap.socket.base_client import BaseSocketClient  # Assuming EasyMocap's client module
# import glob
# import cv2
# import os

# # For keypoint extraction
# import torch
# from easymocap.estimator.YOLOv4 import YOLOv4
# from easymocap.estimator.HRNet import SimpleHRNet

# from easymocap.affinity.affinity import ComposedAffinity
# from easymocap.assignment.associate import simple_associate

# from easymocap.dataset import CONFIG
# from easymocap.config.mvmp1f import Config
# from easymocap.estimator.openpose_wrapper import FeetEstimatorByCrop
# from easymocap.dataset import MVMPMF

# from easymocap.mytools.camera_utils import read_camera
# from easymocap.assignment.group import PeopleGroup
# from easymocap.mytools import Timer


# config = {
#     'openpose':{
#         'root': '',
#         'res': 1,
#         'hand': False,
#         'face': False,
#         'vis': False,
#         'ext': '.jpg'
#     },
#     'openposecrop': {},
#     'feet':{
#         'root': '',
#         'res': 1,
#         'hand': False,
#         'face': False,
#         'vis': False,
#         'ext': '.jpg'
#     },
#     'feetcrop':{
#         'root': '',
#         'res': 1,
#         'hand': False,
#         'face': False,
#         'vis': False,
#         'ext': '.jpg'
#     },
#     'yolo':{
#         'ckpt_path': 'data/models/yolov4.weights',
#         'conf_thres': 0.3,
#         'box_nms_thres': 0.5, # means keeping the bboxes that IOU<0.5
#         # 'ext': '.jpg',
#         'isWild': False,
#     },
#     'hrnet':{
#         'nof_joints': 17,
#         'c': 48,
#         'checkpoint_path': 'data/models/pose_hrnet_w48_384x288.pth'
#     },
#     'yolo-hrnet':{},
#     'mp-pose':{
#         'model_complexity': 2,
#         'min_detection_confidence':0.5,
#         'min_tracking_confidence': 0.5
#     },
#     'mp-holistic':{
#         'model_complexity': 2,
#         # 'refine_face_landmarks': True,
#         'min_detection_confidence':0.5,
#         'min_tracking_confidence': 0.5
#     },
#     'mp-handl':{
#         'model_complexity': 1,
#         'min_detection_confidence':0.3,
#         'min_tracking_confidence': 0.1,
#         'static_image_mode': False,
#     },
#     'mp-handr':{
#         'model_complexity': 1,
#         'min_detection_confidence':0.3,
#         'min_tracking_confidence': 0.1,
#         'static_image_mode': False,
#     },
# }

# config_yolo = config['yolo']
# config_hrnet = config['hrnet']
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print(f"Using device: {device}")

# detector = YOLOv4(device=device, **config_yolo)
# estimator = SimpleHRNet(device=device, **config_hrnet)


# # Send data via socket client
# host = '127.0.0.1'
# port = 9999
# client = BaseSocketClient(host, port)


# def load_cameras(path):
#     intri_name = os.path.join(path, 'intri.yml')
#     extri_name = os.path.join(path, 'extri.yml')
#     if os.path.exists(intri_name) and os.path.exists(extri_name):
#         cameras = read_camera(intri_name, extri_name)
#         cams = cameras.pop('basenames')
#         return cameras, cams

#     else:
#         print('\n\n!!!there is no camera parameters, maybe bug: \n', intri_name, extri_name, '\n')
#         cameras = None
#         return cameras, None

# cameras, cams = load_cameras('camera_config')
# # print(f"cameras, cams: {cameras, cams}")

# Pall = np.stack([cameras[cam]['P'] for cam in cams])


# # ------------------------------------------------------------------- #
# # TODO: STEP 1: Detect Keypoint and set annot
# def create_annot(frame, camera_id=0):
#     height, width = frame.shape[0], frame.shape[1]
#     annot = {
#         'filename': f'camera_{camera_id}',
#         'height':height,
#         'width':width,
#         'annots': [],
#         'isKeyframe': False
#     }
#     return annot

# def extreact_keypoints_from_yolo_hrnet(frame):
#     annot = create_annot(frame)
#     detections = detector.predict_single(frame)
#     # forward_hrnet
#     points2d = estimator.predict(frame, detections)
#     annots = []
#     pid = 0
#     for i in range(len(detections)):
#         annot_ = {
#             'bbox': [float(d) for d in detections[i]],
#             'keypoints': points2d[i],
#             'isKeyframe': False
#         }
#         annot_['area'] = max(annot_['bbox'][2] - annot_['bbox'][0], annot_['bbox'][3] - annot_['bbox'][1])**2
#         annots.append(annot_)
#     annots.sort(key=lambda x:-x['area'])
#     # re-assign the person ID
#     for i in range(len(annots)):
#         annots[i]['personID'] = i + pid
#     annot['annots'] = annots
#     return annot

# def extreact_keypoints_from_yolo(frame):
#     annot = create_annot(frame)
#     detections = detector.predict_single(frame)
#     annots = []
#     pid = 0
#     for i in range(len(detections)):
#         annot_ = {
#             'bbox': [float(d) for d in detections[i]],
#             'isKeyframe': False
#         }
#         annot_['area'] = max(annot_['bbox'][2] - annot_['bbox'][0], annot_['bbox'][3] - annot_['bbox'][1])**2
#         annots.append(annot_)
#     annots.sort(key=lambda x:-x['area'])
#     # re-assign the person ID
#     for i in range(len(annots)):
#         annots[i]['personID'] = i + pid
#     annot['annots'] = annots

#     print(f"annot: {annot}")
#     return annot


# def extreact_keypoints_from_feetcrop(frame):
#     pass


# # def extract_keypoints_by_feetcrop(frame):
# #     # config[mode]['openpose'] = args.openpose
# #     estimator = FeetEstimatorByCrop(openpose='openpose', 
# #         tmpdir=None,
# #         fullbody=mode=='openposecrop',
# #         hand=(mode=='openposecrop')or args.hand,
# #         face=args.face)
# #     estimator.detect_foot(image_root, annot_root, args.ext)


# def mvposev1(frame, annot, cfg=Config.load('config/exp/mvmp1f.yml')):
#     affinity_model = ComposedAffinity(cameras=cameras, basenames=cams, cfg=cfg.affinity)
#     group = PeopleGroup(Pall=Pall, cfg=cfg.group)

#     group.clear()

#     ## With Timer ##
#     # with Timer('compute affinity'):
#     #     affinity, dimGroups = affinity_model(annot, images=frame)
#     # with Timer('associate'):
#     #     group = simple_associate(annot, affinity, dimGroups, Pall, group, cfg=cfg.associate)
#     # Timer.report()
#     # results = group

#     ## NO Timer ##
#     affinity, dimGroups = affinity_model(annot, images=frame)

#     print(f"affinity: {affinity}")
#     print(f"dimGroups: {dimGroups}")

#     group = simple_associate(annot, affinity, dimGroups, Pall, group, cfg=cfg.associate)
#     results = group

#     print(f"results.keys(): {results.keys()}")
#     return results

 

# def mvmp(frame, annot):
#     cfg = Config.load('config/exp/mvmp1f.yml')
#     # # affinity_model = ComposedAffinity(cameras=dataset.cameras, basenames=dataset.cams, cfg=cfg.affinity)
#     # affinity_model = ComposedAffinity(cameras=['01','02','03','04','05'], basenames=[0,1,2,3,4], cfg=cfg.affinity)
#     # # affinity, dimGroups = affinity_model(annots, images=images)

#     # print(f"affinity_model: {affinity_model}")
#     # print(f"annot: {annot}")

#     # affinity, dimGroups = affinity_model([annot], images=[frame])
#     # # group = simple_associate(annots, affinity, dimGroups, dataset.Pall, group, cfg=cfg.associate)
#     # group = simple_associate([annot], affinity, dimGroups, dataset.Pall, group, cfg=cfg.associate)
#     # results = group
#     # return results
#     dataset = MVMPMF('../', cams=['01','02','03','04','05'], annot_root='annot',
#         config=CONFIG['body25'], kpts_type='body25',
#         undis=True, no_img=True, out="../output", filter2d=cfg.dataset)
#     mvposev1(dataset, args, cfg)



# def write_keypoints3d(peopleDict):
#     results = []
#     for pid, people in peopleDict.items():
#         result = {'id': pid, 'keypoints3d': people.keypoints3d}
#         results.append(result)
#     return results


# # TODO: STEP 2: Auto-Tracking Keypoint3d
# def auto_track():
#     pass

# # TODO: STEP 3: SMPL Estimation from Keypoint3d
# def smpl_from_keypoints3d():
#     pass







# # TODO: Send SMPL Data to the Visualization Server
# def send_SMPL_data(smpl_json):
#     # Convert keypoints3d to numpy.ndarray for each person in the data
#     for person in d:
#         if 'keypoints3d' in person:
#             # Convert list to numpy array (shape: [num_keypoints, 4] for x, y, z, confidence)
#             if person['keypoints3d'] is not None:
#                 person['keypoints3d'] = np.array(person['keypoints3d'], dtype=np.float32)
        
#         if 'poses' in person:
#             if person['poses'] is not None:
#                 person['poses'] = np.array(person['poses'], dtype=np.float32)
        
#         if 'shapes' in person:
#             if person['shapes'] is not None:
#                 person['shapes'] = np.array(person['shapes'], dtype=np.float32)

#         if 'Rh' in person:
#             if person['Rh'] is not None:
#                 person['Rh'] = np.array(person['Rh'], dtype=np.float32)

#         if 'Th' in person: 
#             if person['Th'] is not None:
#                 person['Th'] = np.array(person['Th'], dtype=np.float32)

#     client.send_smpl(data=d)  # For 3D SMPL Mesh












# if __name__ == "__main__":
#     # Open the video file ONCE, outside the loop
#     cap1 = cv2.VideoCapture('test_video/01.mp4')
#     cap2 = cv2.VideoCapture('test_video/02.mp4')
#     cap3 = cv2.VideoCapture('test_video/03.mp4')
#     # cap4 = cv2.VideoCapture('test_video/04.mp4')
#     # cap5 = cv2.VideoCapture('test_video/05.mp4')

#     # cap1 = cv2.VideoCapture(0)
#     # cap2 = cv2.VideoCapture(1)
#     # cap3 = cv2.VideoCapture(2)
#     # cap4 = cv2.VideoCapture(3)
#     # cap5 = cv2.VideoCapture(4)
#     # cap = cv2.VideoCapture(0)
#     # set FPS to 30
#     cap1.set(cv2.CAP_PROP_FPS, 30)
#     cap2.set(cv2.CAP_PROP_FPS, 30)
#     cap3.set(cv2.CAP_PROP_FPS, 30)
#     # cap4.set(cv2.CAP_PROP_FPS, 30)
#     # cap5.set(cv2.CAP_PROP_FPS, 30)

#     # Check if video opened successfully
#     if not cap1.isOpened():
#         print("Error: Could not open video file")
#         exit()

#     while True:
#         ret1, frame1 = cap1.read()  # Read frame from the SAME capture object
#         ret2, frame2 = cap2.read()  # Read frame from the SAME capture object
#         ret3, frame3 = cap3.read()  # Read frame from the SAME capture object
#         # ret4, frame4 = cap4.read()  # Read frame from the SAME capture object
#         # ret5, frame5 = cap5.read()  # Read frame from the SAME capture object
#         if not ret1:
#             cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
#             cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
#             cap3.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
#             # cap4.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
#             # cap5.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
#             continue 

#         annot1 = extreact_keypoints_from_yolo_hrnet(frame1)
#         annot2 = extreact_keypoints_from_yolo_hrnet(frame2)
#         annot3 = extreact_keypoints_from_yolo_hrnet(frame3)
#         # annot4 = extreact_keypoints_from_yolo_hrnet(frame4)
#         # annot5 = extreact_keypoints_from_yolo_hrnet(frame5)
#         # frame, annot = extreact_keypoints_from_yolo(frame)


#         if len(annot1['annots']) > 0:
#             # Draw YOLOv4 bounding box
#             cv2.rectangle(frame1, (int(annot1['annots'][0]['bbox'][0]), int(annot1['annots'][0]['bbox'][1])), 
#                         (int(annot1['annots'][0]['bbox'][2]), int(annot1['annots'][0]['bbox'][3])), (0, 255, 0), 2)
        
#         # print(f"annot: {annot}")

#         # mvmp(frame, annot)
#         # print(f"annot: {annot}")
#         frames = [frame1, frame2, frame3] #, frame4, frame5]
#         annots = [annot1['annots'], annot2['annots'], annot3['annots']] #, annot4['annots'], annot5['annots']]
#         # results = mvposev1(frames, [annot['annots']])
#         print(f"annots: {annots}")
#         results = mvposev1(frames,annots)
#         res_keypoint3d = write_keypoints3d(results)

#         print(f"res_keypoint3d: {res_keypoint3d}")

#         client.send(data=res_keypoint3d)  # For 3D Skeletons


#         cv2.imshow('Video', frame1)  # Display the frame
        
#         # Break on 'q' key press to exit
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break

#     # Cleanup
#     cap.release()
#     cv2.destroyAllWindows()




import json
import numpy as np
from easymocap.socket.base_client import BaseSocketClient
import cv2
import os
import torch
import threading
import time

# --- Your Existing Imports ---
from easymocap.estimator.YOLOv4 import YOLOv4
from easymocap.estimator.HRNet import SimpleHRNet
from easymocap.affinity.affinity import ComposedAffinity
from easymocap.assignment.associate import simple_associate
from easymocap.config.mvmp1f import Config
from easymocap.pipeline.config import Config as MeshConfig
from easymocap.mytools.camera_utils import read_camera
from easymocap.assignment.group import PeopleGroup
from easymocap.dataset import CONFIG
# --- Imports for SMPL Fitting ---
from easymocap.smplmodel import load_model
from easymocap.pipeline.weight import load_weight_shape, load_weight_pose
# This function is the logical next step after optimizeShape in the easymocap pipeline
from easymocap.pyfitting.optimize_simple import optimizeShape, optimizePose3D
from easymocap.mytools import Timer

# --- CONFIGURATION (Remains the same) ---
config = {
    'yolo':{
        'ckpt_path': 'data/models/yolov4.weights',
        'conf_thres': 0.3,
        'box_nms_thres': 0.5,
        'isWild': False,
    },
    'hrnet':{
        'nof_joints': 17,
        'c': 48,
        'checkpoint_path': 'data/models/pose_hrnet_w48_384x288.pth'
    },
}

# ==================== 1. INITIALIZATION (Done ONCE) ====================

# --- Device Setup ---
print("Initializing models...")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# --- Load 2D Detection Models ---
detector = YOLOv4(device=device, **config['yolo'])
estimator = SimpleHRNet(device=device, **config['hrnet'])

# --- Initialize Socket Client ---
host = '127.0.0.1'
port = 9999
client = BaseSocketClient(host, port)
print(f"Connecting to server at {host}:{port}")

# --- Load Camera Parameters ---
def load_cameras(path):
    intri_name = os.path.join(path, 'intri.yml')
    extri_name = os.path.join(path, 'extri.yml')
    if os.path.exists(intri_name) and os.path.exists(extri_name):
        cameras = read_camera(intri_name, extri_name)
        cams = cameras.pop('basenames')
        return cameras, cams
    else:
        print(f"\n\n!!! ERROR: Camera parameters not found in '{path}' !!!\n")
        return None, None
cameras, cams = load_cameras('camera_config')
if cameras is None:
    exit()
Pall = np.stack([cameras[cam]['P'] for cam in cams])
print("Camera parameters loaded.")

# --- Load Multi-view Pose Configuration ---
print("Loading multi-view pose configuration...")
cfg_mvmp = Config.load('config/exp/mvmp1f.yml')
affinity_model = ComposedAffinity(cameras=cameras, basenames=cams, cfg=cfg_mvmp.affinity)

# ==================== MODIFIED: Load SMPL Model and Weights ONCE ====================
print("Loading SMPL model...")
GENDER = 'neutral'
MODEL_TYPE = 'smpl'
# Load the model and move it to the GPU immediately
body_model = load_model(gender=GENDER, model_type=MODEL_TYPE).to(device)
weight_shape = load_weight_shape(MODEL_TYPE, {'smooth_pose': 1e1})
weight_pose = load_weight_pose(MODEL_TYPE, {'smooth_pose': 1e1})
print("SMPL model loaded.")
# ==================================================================================

# --- 2. YOUR FUNCTIONS (Modified for Performance) ---

def create_annot(frame, camera_id=0):
    height, width = frame.shape[0], frame.shape[1]
    return {'filename': f'camera_{camera_id}', 'height':height, 'width':width, 'annots': [], 'isKeyframe': False}

def extract_keypoints_from_yolo_hrnet(frame, camera_id):
    annot = create_annot(frame, camera_id)
    detections = detector.predict_single(frame)
    points2d = estimator.predict(frame, detections)
    annots = []
    for i, (det, pts) in enumerate(zip(detections, points2d)):
        annots.append({
            'bbox': [float(d) for d in det], 'keypoints': pts, 'isKeyframe': False,
            'area': max(det[2] - det[0], det[3] - det[1])**2, 'personID': i
        })
    annots.sort(key=lambda x: -x['area'])
    for i, ann in enumerate(annots):
        ann['personID'] = i
    annot['annots'] = annots
    return annot

def mvposev1(frames, annots, group):
    group.clear()
    affinity, dimGroups = affinity_model(annots, images=frames)
    group = simple_associate(annots, affinity, dimGroups, Pall, group, cfg=cfg_mvmp.associate)
    return group

def write_keypoints3d(peopleDict):
    return [{'id': pid, 'keypoints3d': person.keypoints3d} for pid, person in peopleDict.items()]

def read_keypoint3d(json_data):
    # This function logic remains the same
    results = {}
    for d in json_data:
        pid = d['id']
        if pid not in results:
            results[pid] = {'keypoints3d': []}
        results[pid]['keypoints3d'].append(d['keypoints3d'])
    for pid, res in results.items():
        res['keypoints3d'] = np.stack(res['keypoints3d'])
    return results

# ==================== MODIFIED FUNCTION ====================
def smpl_from_skel(skel3d, body_model, weight_shape, weight_pose, smpl_params_tracker):
    # This function no longer loads models. It receives them as arguments.
    config = CONFIG['body25']
    results3d = read_keypoint3d(skel3d)
    
    current_pids = set(results3d.keys())

    for pid, result in results3d.items():
        kp3ds_np = result['keypoints3d']
        
        # Get previous frame's parameters for initialization to speed up fitting
        prev_params = smpl_params_tracker.get(pid)

        # Call the fitting function with pre-loaded models and previous params
        body_params = smpl_from_keypoints3d(
            body_model, kp3ds_np, config,
            weight_shape=weight_shape, weight_pose=weight_pose,
            prev_params=prev_params
        )
        
        # Update the tracker with the latest results for this person
        smpl_params_tracker[pid] = body_params
        result['body_params'] = body_params
    
    # Clean up tracker: remove people who are no longer in the scene
    lost_pids = set(smpl_params_tracker.keys()) - current_pids
    for pid in lost_pids:
        del smpl_params_tracker[pid]

    return results3d # Return results to be sent
# ==========================================================

# ==================== MODIFIED FUNCTION ====================
def smpl_from_keypoints3d(body_model, kp3ds, config, weight_shape, weight_pose, prev_params=None):
    # This function now accepts previous parameters for temporal consistency
    
    # --- OPTIMIZATION 1: Move computation to GPU ---
    # Convert incoming numpy array to a GPU tensor
    kp3ds_torch = torch.from_numpy(kp3ds).float().to(body_model.device)

    # --- OPTIMIZATION 2: Use temporal information ---
    if prev_params is not None:
        # If we have a history for this person, start from their last known state
        params = prev_params
        # Keep the shape consistent, don't re-optimize it every frame
        # params['shapes'] = prev_params['shapes'].clone()
        params['shapes'] = prev_params['shapes'].copy()
    else:
        # If this is a new person, initialize from scratch and optimize shape
        params = body_model.init_params(nFrames=kp3ds_torch.shape[0])
        params = optimizeShape(
            body_model, params, kp3ds_torch,
            weight_loss=weight_shape, kintree=CONFIG['body15']['kintree'][1:]
        )
    


    # Optimize 3D pose for the current frame using the initialized parameters
    # params = optimizePose(
    #     body_model, params, kp3ds_torch,
    #     weight_loss=weight_pose, kintree=config['kintree']
    # )

    cfg = MeshConfig({"smooth_poses": 1e1, 'verbose': False, 'device': device})
    cfg.OPT_R = True
    cfg.OPT_T = True
    cfg.OPT_POSE = True
    cfg.ROBUST_3D = False
    params = optimizePose3D(body_model, params, keypoints3d=kp3ds, weight=weight_pose, cfg=cfg)

    
    return params
# ==========================================================

# --- WORKER FUNCTION FOR THREADING (Unchanged) ---
def process_camera_stream(cam_id, frame, results_list):
    annot = extract_keypoints_from_yolo_hrnet(frame, cam_id)
    results_list[cam_id] = annot





# Convert keypoints3d to numpy.ndarray for each person in the data
def modify_SMPL_data(person_data, send=True):
    person_list =[]
    for pid, person in person_data.items():
        person = person['body_params']
        person['id'] = pid  # Add ID for sending
        if 'keypoints3d' in person:
            # Convert list to numpy array (shape: [num_keypoints, 4] for x, y, z, confidence)
            if person['keypoints3d'] is not None:
                person['keypoints3d'] = np.array(person['keypoints3d'], dtype=np.float32)
        
        if 'poses' in person:
            if person['poses'] is not None:
                person['poses'] = np.array(person['poses'], dtype=np.float32)
        
        if 'shapes' in person:
            if person['shapes'] is not None:
                person['shapes'] = np.array(person['shapes'], dtype=np.float32)

        if 'Rh' in person:
            if person['Rh'] is not None:
                person['Rh'] = np.array(person['Rh'], dtype=np.float32)

        if 'Th' in person:
            if person['Th'] is not None:
                person['Th'] = np.array(person['Th'], dtype=np.float32)

        person_list.append(person)

    if send:
        send_SMPL_data(person_list)  # For 3D SMPL Mesh

def send_SMPL_data(smpl_data):
    client.send_smpl(data=smpl_data)  # For 3D SMPL Mesh







# ==================== 4. MAIN EXECUTION LOOP (Modified) ====================
if __name__ == "__main__":
    # video_sources = ['test_video/01.mp4', 'test_video/02.mp4', 'test_video/03.mp4']
    # video_sources = ["usb-046d_Brio_100_2533ZB10TMM8", "usb-046d_Brio_100_2533ZB40U2U8", "usb-046d_Brio_100_2533ZBA0U2E8", "usb-046d_Brio_100_2533ZB20U488"]
    video_sources = ["usb-046d_Brio_100_2533ZB10TMM8", "usb-046d_Brio_100_2533ZB40U2U8", "usb-046d_Brio_100_2533ZBA0U2E8"]
    video_sources = [f"/dev/v4l/by-id/{unique_id}-video-index0" for unique_id in video_sources]

    caps = [cv2.VideoCapture(src) for src in video_sources]
    for cap in caps:
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not all([cap.isOpened() for cap in caps]):
        print("Error: Could not open one or more video files/cameras.")
        exit()
        
    group = PeopleGroup(Pall=Pall, cfg=cfg_mvmp.group)

    frame_counter = 0
    PROCESSING_INTERVAL = 3 # Increase if still laggy

    # NEW: Tracker dictionary to store SMPL parameters between frames
    smpl_params_tracker = {}

    while True:
        frames = []
        rets = []
        for cap in caps:
            ret, frame = cap.read()
            rets.append(ret)
            frames.append(frame)

        if not all(rets):
            print("End of video stream, resetting.")
            for cap in caps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Step 1: 2D Keypoint Extraction (runs every frame)
        threads = []
        thread_results = [None] * len(caps)
        for i, frame in enumerate(frames):
            thread = threading.Thread(target=process_camera_stream, args=(i, frame, thread_results))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        
        # Only run the expensive 3D calculation on intervals
        if frame_counter % PROCESSING_INTERVAL == 0:
            annots_for_mvpose = [res['annots'] for res in thread_results]

            # Step 2: Heavy 3D Skeleton Processing
            results_group = mvposev1(frames, annots_for_mvpose, group)
            res_keypoint3d = write_keypoints3d(results_group)
            
            # Send 3D keypoints to 3D Skeleton Server (can be visualized immediately)
            # client.send(data=res_keypoint3d)  # For 3D Skeletons

            # --- MODIFIED: SMPL Fitting Logic is now INSIDE the interval block ---
            if res_keypoint3d:
                print(f"Frame {frame_counter}: Detected {len(res_keypoint3d)} person(s). Fitting SMPL...")
                
                # --- Step 3: Heavy SMPL Model Fitting ---
                # Pass the pre-loaded models and the tracker to the function
                smpl_results = smpl_from_skel(
                    res_keypoint3d, body_model, weight_shape, 
                    weight_pose, smpl_params_tracker
                )
            

            #     # TODO: You would now send the 'smpl_results' to your visualization server
                modify_SMPL_data(smpl_results, send=True)


        # Step 4: Visualization (runs every frame)
        display_frame = frames[0]
        if thread_results[0] and thread_results[0]['annots']:
            bbox = thread_results[0]['annots'][0]['bbox']
            cv2.rectangle(display_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        cv2.imshow('Video', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_counter += 1

    # --- Cleanup ---
    print("Shutting down...")
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()