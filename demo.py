import argparse
import sys
import os

import torch
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import joblib
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
try:
    from scripts.scripts_test_video.hawor_slam import hawor_slam
except Exception as _e:          # DROID-SLAM / Metric3D are CUDA-only; --skip-slam never calls this
    _slam_import_error = _e
    def hawor_slam(*a, **k):
        raise SystemExit(f"hawor_slam is unavailable ({_slam_import_error}). Use --skip-slam.")
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video, run_vis2_on_video_cam


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default='example/video_0.mp4')
    parser.add_argument("--input_type", type=str, default='file')
    parser.add_argument("--checkpoint",  type=str, default='./weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument("--infiller_weight",  type=str, default='./weights/hawor/checkpoints/infiller.pt')
    parser.add_argument("--vis_mode",  type=str, default='world', help='cam | world')
    parser.add_argument("--device", type=str, default='auto',
                        help="auto | cpu | cuda | cuda:N | mps. 'auto' picks CUDA when available.")
    parser.add_argument("--skip-slam", dest='skip_slam', action='store_true',
                        help="Do not run DROID-SLAM. Its droid_backends extension is CUDA-only, so "
                             "this is required on CPU. Supply --camera-npz to still get world-space "
                             "output; without it an identity trajectory is written and the result is "
                             "CAMERA-space.")
    parser.add_argument("--camera-npz", dest='camera_npz', default=None,
                        help="A 6DoF npz with 'T' (N,4,4) camera-to-world (e.g. head_pose_6dof.npz) "
                             "used in place of SLAM. Only meaningful with --skip-slam.")
    parser.add_argument("--no-vis", dest='no_vis', action='store_true',
                        help="Skip the aitviewer render; write the motion only.")
    args = parser.parse_args()

    from hawor.utils.device import set_device
    device = set_device(args.device)
    print(f"[hawor] device: {device}")
    if args.skip_slam:
        # The SLAM-mask render in hawor_motion_estimation exists only to feed masked DROID-SLAM.
        # Without SLAM it is pure cost: a pytorch3d render per frame and a ~3.8 GB memmap.
        os.environ['HAWOR_SKIP_MASKS'] = '1'
    elif device.type != 'cuda':
        raise SystemExit(
            "DROID-SLAM requires CUDA (droid_backends has no CPU build). Re-run with --skip-slam, "
            "optionally adding --camera-npz <6dof npz> to keep world-space output.")

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)

    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        if args.skip_slam:
            from scripts.scripts_test_video.make_camera_npz import main as _make_cam
            import sys as _sys
            argv = ['make_camera_npz', '--out', slam_path, '--frames', str(end_idx - start_idx)]
            argv += (['--mode', 'headpose', '--headpose', args.camera_npz] if args.camera_npz
                     else ['--mode', 'identity'])
            _saved, _sys.argv = _sys.argv, argv
            try:
                _make_cam()
            finally:
                _sys.argv = _saved
            if not args.camera_npz:
                print("[hawor] NOTE: identity trajectory -- these hands are CAMERA-space, not world.")
        else:
            hawor_slam(args, start_idx, end_idx)
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(args, start_idx, end_idx, frame_chunks_all)

    # vis sequence for this video
    hand2idx = {
        "right": 1,
        "left": 0
    }
    vis_start = 0
    vis_end = pred_trans.shape[1] - 1
            
    # get faces
    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)

    # get right hand vertices
    hand = 'right'
    hand_idx = hand2idx[hand]
    pred_glob_r = run_mano(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    right_verts = pred_glob_r['vertices'][0]
    right_dict = {
            'vertices': right_verts.unsqueeze(0),
            'faces': faces_right,
        }

    # get left hand vertices
    faces_left = faces_right[:,[0,2,1]]
    hand = 'left'
    hand_idx = hand2idx[hand]
    pred_glob_l = run_mano_left(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    left_verts = pred_glob_l['vertices'][0]
    left_dict = {
            'vertices': left_verts.unsqueeze(0),
            'faces': faces_left,
        }

    R_x = torch.tensor([[1,  0,  0],
                        [0, -1,  0],
                        [0,  0, -1]]).float()
    R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
    left_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, left_dict['vertices'].cpu())
    right_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, right_dict['vertices'].cpu())
    
    # Here we use aitviewer(https://github.com/eth-ait/aitviewer) for simple visualization.
    if args.no_vis:
        print(f"[hawor] --no-vis: motion written under {seq_folder}, skipping render")
    elif args.vis_mode == 'world': 
        output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
        if not os.path.exists(output_pth):
            os.makedirs(output_pth)
        image_names = imgfiles[vis_start:vis_end]
        print(f"vis {vis_start} to {vis_end}")
        run_vis2_on_video(left_dict, right_dict, output_pth, img_focal, image_names, R_c2w=R_c2w_sla_all[vis_start:vis_end], t_c2w=t_c2w_sla_all[vis_start:vis_end])
    elif args.vis_mode == 'cam':
        output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
        if not os.path.exists(output_pth):
            os.makedirs(output_pth)
        image_names = imgfiles[vis_start:vis_end]
        print(f"vis {vis_start} to {vis_end}")
        run_vis2_on_video_cam(left_dict, right_dict, output_pth, img_focal, image_names, R_w2c=R_w2c_sla_all[vis_start:vis_end], t_w2c=t_w2c_sla_all[vis_start:vis_end])

    print("finish")



