import os
from model.utils.opt import get_default_parser
import torch
import numpy as np
import random
from glob import glob
import csv

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def main(args):

    data_path = args.data_path
    outdir = os.path.join(args.out_path, args.exp_name)
    report_dir_name = outdir.split('/')[-1]
    
    with open(os.path.join(data_path, "test_scenes.txt"),'r') as fp:
        test_scenes = fp.readlines()
        test_scenes = [x.strip('\n') for x in test_scenes]

    # edit here for testing configs
    sample_percents = [5, 10, 20, 50]
    blur_kernel_sizes = [-1, 5]
    # test_scenes = test_scenes
    print("Get results from:", test_scenes)

    for blur_kernel_size in blur_kernel_sizes:
        metric_dict = {}

        for percent in sample_percents:
            metric_dict[percent] = {}
            metric_dict[percent]['psnr'] = []
            metric_dict[percent]['ssim'] = []
            metric_dict[percent]['mse'] = []
            metric_dict[percent]['lpips'] = []
            metric_dict[percent]['is_mean'] = []
            metric_dict[percent]['is_std'] = []
            metric_dict[percent]['fid_with_gt'] = []
            metric_dict[percent]['fid_with_input'] = []
            metric_dict[percent]['cs_with_gt'] = []
            metric_dict[percent]['cs_with_input'] = []
            metric_dict[percent]['chamfer_distance'] = []
            metric_dict[percent]['completeness'] = []

            for scene_name in test_scenes:
                
                scene_outdir = os.path.join(outdir, scene_name)
                sampled_scene_outdir = os.path.join(scene_outdir, str(percent))
                test_dir_name = ("test_down3"+f"_blur_k{blur_kernel_size}") if blur_kernel_size != -1 else "test_down3"
                metric_file_paths = glob(os.path.join(sampled_scene_outdir, '*', test_dir_name, 'metrics.txt'))
                chamfer_dist_file_paths = glob(os.path.join(sampled_scene_outdir, '*', test_dir_name, 'chamfer_distance.txt'))
                completeness_file_paths = glob(os.path.join(sampled_scene_outdir, '*', test_dir_name, 'completeness.txt'))

                if len(metric_file_paths) > 1:
                    print('Find more than one dirs. Use the last one')
                elif len(metric_file_paths) == 0:
                    print('Not find target dirs at:', sampled_scene_outdir)
                    continue

                with open(metric_file_paths[-1],'r') as fp:
                    lines = fp.readlines()
                    data1 = lines[0].strip('\n').split(' ')
                    data2 = lines[1].strip('\n').split(' ')
                    psnr, ssim, mse = float(data1[2][:-1]), float(data1[4][:-1]), float(data1[6][:-1])
                    lpips, is_mean, is_std, fid_with_gt, fid_with_input, cs_with_gt, cs_with_input = float(data2[2][:-1]), float(data2[4][:-1]), float(data2[6][:-1]), float(data2[8][:-1]), float(data2[10][:-1]), float(data2[12][:-1]), float(data2[14][:-1])
                    metric_dict[percent]['psnr'].append(psnr)
                    metric_dict[percent]['ssim'].append(ssim)
                    metric_dict[percent]['mse'].append(mse)
                    metric_dict[percent]['lpips'].append(lpips)
                    metric_dict[percent]['is_mean'].append(is_mean)
                    metric_dict[percent]['is_std'].append(is_std)
                    metric_dict[percent]['fid_with_gt'].append(fid_with_gt)
                    metric_dict[percent]['fid_with_input'].append(fid_with_input)
                    metric_dict[percent]['cs_with_gt'].append(cs_with_gt)
                    metric_dict[percent]['cs_with_input'].append(cs_with_input)
                
                if len(chamfer_dist_file_paths) == 0:
                    print("Chamfer Distance not found. Please run calculate_chanfer.py first.")
                    metric_dict[percent]['chamfer_distance'].append(0)
                else:
                    with open(chamfer_dist_file_paths[-1],'r') as fp:
                        lines = fp.readlines()
                        data1 = lines[0].strip('\n').split(' ')
                        chamfer_dist = float(data1[0])
                        metric_dict[percent]['chamfer_distance'].append(chamfer_dist)
                
                if len(completeness_file_paths) == 0:
                    print("Completeness not found. Please run calculate_chamfer_completeness.py first.")
                    metric_dict[percent]['completeness'].append(0)
                else:
                    with open(completeness_file_paths[-1],'r') as fp:
                        lines = fp.readlines()
                        data1 = lines[0].strip('\n').split(' ')
                        completeness = float(data1[0])
                        metric_dict[percent]['completeness'].append(completeness)

        metric_avg_dict = {}
        for percent in sample_percents:
            metric_avg_dict[percent] = {}

            for k, v in metric_dict[percent].items():
                metric_avg_dict[percent][k] = np.array(v).mean()
        
        print(metric_avg_dict)

        # write the results to a csv file
        if not os.path.exists('report'):
            os.makedirs('report')
        if not os.path.exists(os.path.join('report', report_dir_name)):
            os.makedirs(os.path.join('report', report_dir_name))

        csv_file_name = (report_dir_name+f"_blur_k{blur_kernel_size}.csv") if blur_kernel_size != -1 else (report_dir_name+".csv")
        with open(os.path.join('report', report_dir_name, csv_file_name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(['**OVERALL**'])
            writer.writerow(['Percent(%)', 'PSNR', 'SSIM', 'LPIPS', 'MSE', 'IS(mean)', 'IS(std)', 'FID(with gt)', 'FID(with input)', 'CS(with gt)', 'CS(with input)', 'Chamfer Distance', 'Completeness'])

            metric_names = ['psnr', 'ssim', 'lpips', 'mse', 'is_mean', 'is_std', 'fid_with_gt', 'fid_with_input', 'cs_with_gt', 'cs_with_input', 'chamfer_distance', 'completeness']

            for percent in sample_percents:
                metric_values = [metric_avg_dict[percent][metric_name] for metric_name in metric_names]
                writer.writerow([percent] + metric_values) 

            writer.writerow([' '])
            writer.writerow(['**PER SCENES**'])

            for metric_name in metric_names:
                writer.writerow([metric_name.upper()]) 
                writer.writerow(['Percent(%)'] + test_scenes) 
                for percent in sample_percents:
                    metric_values = [metric_dict[percent][metric_name][i] for i in range(len(test_scenes))]
                    writer.writerow([percent] + metric_values) 
                writer.writerow([' '])


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()
    main(args)
