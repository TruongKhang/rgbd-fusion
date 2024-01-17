import argparse, os
from pprint import pprint
import numpy as np

from plyfile import PlyData, PlyElement
import torch
from torch.utils.data import DataLoader, SequentialSampler

from .datasets.rgbd_dataset import MVSRGBD
from .utils import to_device
from .modules import geometry as fusion_func


def parse_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--dataset_dir', type=str, help='evaluation dataset')
    parser.add_argument(
        '--nview_thr', type=str, help='number of consistent views')
    parser.add_argument(
        '--disp_thr', type=float, default=1.0, help='disparity threshold for geometric consistency')
    parser.add_argument(
        '--conf_thr', type=float, default=0.8, help="a confidence threshold for photometric consistentcy")
    parser.add_argument(
        '--n_src_views', type=int, default=10, help='number of source images considered for each reference image.')
    
    return parser.parse_args()


def main(args, rgbd_dir, img_pair_file, out_filename, device=torch.device("cpu")):
    mvsrgbd_dataset = MVSRGBD(img_pair_file, rgbd_dir, n_src_views=args.n_src_views)
    sampler = SequentialSampler(mvsrgbd_dataset)
    dataloader = DataLoader(mvsrgbd_dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=2,
                               pin_memory=True, drop_last=False)
    views = {}
    prob_threshold = args.conf_thr
    # prob_threshold = [float(p) for p in prob_threshold.split(',')]
    for batch_idx, sample_np in enumerate(dataloader):
        sample = to_device(sample_np, device)
        for ids in range(sample["src_depths"].size(1)):
            src_prob_mask = fusion_func.prob_filter(sample['src_confs'][:, ids, ...], prob_threshold)
            sample["src_depths"][:, ids, ...] *= src_prob_mask.float()

        prob_mask = fusion_func.prob_filter(sample['ref_conf'], prob_threshold)

        reproj_xyd, in_range = fusion_func.get_reproj(
                *[sample[attr] for attr in ['ref_depth', 'src_depths', 'ref_cam', 'src_cams']])
        vis_masks, vis_mask = fusion_func.vis_filter(sample['ref_depth'], reproj_xyd, in_range, args.disp_thr, 0.01, args.nview_thr)

        ref_depth_ave = fusion_func.ave_fusion(sample['ref_depth'], reproj_xyd, vis_masks)

        mask = fusion_func.bin_op_reduce([prob_mask, vis_mask], torch.min)

        idx_img = fusion_func.get_pixel_grids(*ref_depth_ave.size()[-2:]).unsqueeze(0)
        idx_cam = fusion_func.idx_img2cam(idx_img, ref_depth_ave, sample['ref_cam'])
        points = fusion_func.idx_cam2world(idx_cam, sample['ref_cam'])[..., :3, 0].permute(0, 3, 1, 2)
        #cam_center = (- sample['ref_cam'][:,0,:3,:3].transpose(-2,-1) @ sample['ref_cam'][:,0,:3,3:])[...,0]
        #dir_vecs = cam_center.unsqueeze(-1).unsqueeze(-1) - points

        points_np = points.cpu().data.numpy()
        mask_np = mask.cpu().data.numpy().astype(np.bool)
        #dir_vecs = dir_vecs.cpu().data.numpy()
        ref_img = sample_np['ref_img'].data.numpy()
        for i in range(points_np.shape[0]):
            print(np.sum(np.isnan(points_np[i])))
            p_f_list = [points_np[i, k][mask_np[i, 0]] for k in range(3)]
            p_f = np.stack(p_f_list, -1)
            c_f_list = [ref_img[i, k][mask_np[i, 0]] for k in range(3)]
            c_f = np.stack(c_f_list, -1) * 255
            #d_f_list = [dir_vecs[i, k][mask_np[i, 0]] for k in range(3)]
            #d_f = np.stack(d_f_list, -1)
            ref_id = str(sample_np['ref_id'][i].item())
            views[ref_id] = (p_f, c_f.astype(np.uint8))
            print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(rgbd_dir, int(ref_id), prob_mask[i].float().mean().item(), vis_mask[i].float().mean().item(), mask[i].float().mean().item()))

    print('Write combined PCD')
    p_all, c_all = [np.concatenate([v[k] for key, v in views.items()], axis=0) for k in range(2)]

    vertexs = np.array([tuple(v) for v in p_all], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in c_all], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(out_filename)
    print("saving the final model to", out_filename)


if __name__ == "__main__":
    args = parse_args()
    pprint.pprint(vars(args))

    dataset_dir = args.dataset_dir
    img_pair_file = f"{dataset_dir}/pair.txt"
    out_filename = f"{dataset_dir}/fused_model.ply"

    # device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(args, dataset_dir, img_pair_file, out_filename, device)
