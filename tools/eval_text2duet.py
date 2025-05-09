import sys
sys.path.append(sys.path[0]+r"/../")
import numpy as np
import torch

from datetime import datetime
from datasets import get_dataset_motion_loader, get_motion_loader
from models import *
from utils.metrics import *
from datasets import EvaluatorModelWrapper
from collections import OrderedDict
from utils.plot_script import *
from utils.utils import *
from configs import get_config
from os.path import join as pjoin
from tqdm import tqdm

os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
torch.multiprocessing.set_sharing_strategy('file_system')

def build_models(cfg):
    if cfg.NAME == "DuetModel":
        model = DuetModel(cfg)
    elif cfg.NAME == "ReactModel":
        model = ReactModel(cfg)
    else:
        raise NotImplementedError
    return model

def evaluate_matching_score(motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating MM Distance ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        mm_dist_sum = 0
        top_k_count = 0
        print(f"Processing loader: {motion_loader_name}")
        
        # Check if the loader is empty by attempting to get the first batch
        try:
            first_batch = next(iter(motion_loader))
            has_data = True
        except StopIteration:
            has_data = False
            print(f"Warning: The loader for {motion_loader_name} is empty!")
            print(f"Warning: The loader for {motion_loader_name} is empty!", file=file, flush=True)
            # Add placeholder values for empty loaders
            match_score_dict[motion_loader_name] = float('nan')
            R_precision_dict[motion_loader_name] = np.zeros(3)  # assuming top_k=3
            activation_dict[motion_loader_name] = np.zeros((1, motion_embeddings.shape[1])) if 'motion_embeddings' in locals() else np.zeros((1, 1))
            continue
            
        with torch.no_grad():
            # If we have data, process it
            if has_data:
                # Process the first batch we already retrieved
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(first_batch)
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                    motion_embeddings.cpu().numpy())
                mm_dist_sum += dist_mat.trace()
                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)
                all_size += text_embeddings.shape[0]
                all_motion_embeddings.append(motion_embeddings.cpu().numpy())
                
                # Process the rest of the batches
                for idx, batch in tqdm(enumerate(motion_loader)):
                    text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(batch)
                    dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                        motion_embeddings.cpu().numpy())
                    mm_dist_sum += dist_mat.trace()
                    argsmax = np.argsort(dist_mat, axis=1)
                    top_k_mat = calculate_top_k(argsmax, top_k=3)
                    top_k_count += top_k_mat.sum(axis=0)
                    all_size += text_embeddings.shape[0]
                    all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            # Check if we have any embeddings to concatenate
            if len(all_motion_embeddings) == 0:
                print(f"Warning: No motion embeddings collected for {motion_loader_name}")
                print(f"Warning: No motion embeddings collected for {motion_loader_name}", file=file, flush=True)
                # Add placeholder values
                match_score_dict[motion_loader_name] = float('nan')
                R_precision_dict[motion_loader_name] = np.zeros(3)  # assuming top_k=3
                activation_dict[motion_loader_name] = np.zeros((1, motion_embeddings.shape[1])) if 'motion_embeddings' in locals() else np.zeros((1, 1))
            else:
                all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
                mm_dist = mm_dist_sum / all_size
                R_precision = top_k_count / all_size
                match_score_dict[motion_loader_name] = mm_dist
                R_precision_dict[motion_loader_name] = R_precision
                activation_dict[motion_loader_name] = all_motion_embeddings

                print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')
                print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}', file=file, flush=True)

                line = f'---> [{motion_loader_name}] R_precision: '
                for i in range(len(R_precision)):
                    line += '(top %d): %.4f ' % (i+1, R_precision[i])
                print(line)
                print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict

def evaluate_fid(groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    
    # Check if groundtruth_loader is empty
    try:
        first_batch = next(iter(groundtruth_loader))
        has_gt_data = True
    except StopIteration:
        has_gt_data = False
        print(f"Warning: The ground truth loader is empty!")
        print(f"Warning: The ground truth loader is empty!", file=file, flush=True)
        # Return placeholder values if we have no ground truth data
        for model_name in activation_dict.keys():
            eval_dict[model_name] = float('nan')
            print(f'---> [{model_name}] FID: NaN (no ground truth data)')
            print(f'---> [{model_name}] FID: NaN (no ground truth data)', file=file, flush=True)
        return eval_dict
    
    with torch.no_grad():
        if has_gt_data:
            # Process the first batch we already retrieved
            motion_embeddings = eval_wrapper.get_motion_embeddings(first_batch)
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
            
            # Process the rest of the batches
            for idx, batch in tqdm(enumerate(groundtruth_loader)):
                motion_embeddings = eval_wrapper.get_motion_embeddings(batch)
                gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    
    # Check if we have any ground truth embeddings
    if len(gt_motion_embeddings) == 0:
        print(f"Warning: No ground truth motion embeddings collected")
        print(f"Warning: No ground truth motion embeddings collected", file=file, flush=True)
        # Return placeholder values if we have no ground truth embeddings
        for model_name in activation_dict.keys():
            eval_dict[model_name] = float('nan')
            print(f'---> [{model_name}] FID: NaN (no ground truth embeddings)')
            print(f'---> [{model_name}] FID: NaN (no ground truth embeddings)', file=file, flush=True)
        return eval_dict
    
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        if np.isnan(motion_embeddings).any() or motion_embeddings.size == 1:
            print(f'---> [{model_name}] FID: NaN (invalid model embeddings)')
            print(f'---> [{model_name}] FID: NaN (invalid model embeddings)', file=file, flush=True)
            eval_dict[model_name] = float('nan')
            continue
            
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict

def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    
    if not activation_dict:
        print("Warning: activation_dict is empty, no models to evaluate")
        print("Warning: activation_dict is empty, no models to evaluate", file=file, flush=True)
        return eval_dict
        
    for model_name, motion_embeddings in activation_dict.items():
        if np.isnan(motion_embeddings).any() or motion_embeddings.size == 1:
            print(f'---> [{model_name}] Diversity: NaN (invalid embeddings)')
            print(f'---> [{model_name}] Diversity: NaN (invalid embeddings)', file=file, flush=True)
            eval_dict[model_name] = float('nan')
            continue
            
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict

def evaluate_multimodality(mm_motion_loaders, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    
    if not mm_motion_loaders:
        print("Warning: mm_motion_loaders is empty, no models to evaluate")
        print("Warning: mm_motion_loaders is empty, no models to evaluate", file=file, flush=True)
        return eval_dict
        
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        
        # Check if the loader is empty
        try:
            first_batch = next(iter(mm_motion_loader))
            has_data = True
        except StopIteration:
            has_data = False
            print(f"Warning: The MM loader for {model_name} is empty!")
            print(f"Warning: The MM loader for {model_name} is empty!", file=file, flush=True)
            eval_dict[model_name] = 0
            print(f'---> [{model_name}] Multimodality: 0.0000 (empty loader)')
            print(f'---> [{model_name}] Multimodality: 0.0000 (empty loader)', file=file, flush=True)
            continue
            
        with torch.no_grad():
            if has_data:
                # Process first batch
                batch = first_batch
                batch[2] = batch[2][0]
                batch[3] = batch[3][0]
                batch[4] = batch[4][0]
                motion_embedings = eval_wrapper.get_motion_embeddings(batch)
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
                
                # Process remaining batches
                for idx, batch in enumerate(mm_motion_loader):
                    # (1, mm_replications, dim_pos)
                    batch[2] = batch[2][0]
                    batch[3] = batch[3][0]
                    batch[4] = batch[4][0]
                    motion_embedings = eval_wrapper.get_motion_embeddings(batch)
                    mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
                    
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
            print(f'---> [{model_name}] Multimodality: 0.0000 (no embeddings collected)')
            print(f'---> [{model_name}] Multimodality: 0.0000 (no embeddings collected)', file=file, flush=True)
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
            print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
            print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
            
        eval_dict[model_name] = multimodality
    return eval_dict

def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval

def debug_datasets_and_loaders():
    print("\n=== Debugging Datasets and Loaders ===")
    print(f"Ground truth dataset size: {len(gt_dataset)}")
    
    # Check if loaders have items
    try:
        gt_batch = next(iter(gt_loader))
        print("Ground truth loader has data")
    except StopIteration:
        print("Ground truth loader is empty!")
    
    # Check if model loaders have items
    for model_name, loader_getter in eval_motion_loaders.items():
        print(f"\nChecking {model_name}:")
        try:
            motion_loader, mm_motion_loader = loader_getter()
            
            # Check dataset underlying the motion_loader
            if hasattr(motion_loader.dataset, 'generated_motions'):
                print(f"  Dataset has {len(motion_loader.dataset.generated_motions)} generated motions")
            else:
                print("  Dataset has no 'generated_motions' attribute")
                
            # Check dataset underlying the mm_motion_loader
            if hasattr(mm_motion_loader.dataset, 'dataset'):
                print(f"  MM Dataset has {len(mm_motion_loader.dataset.dataset)} motion sets")
            else:
                print("  MM Dataset has no 'dataset' attribute")
            
            # Try to get a batch from each loader
            try:
                batch = next(iter(motion_loader))
                print("  Motion loader has data")
            except StopIteration:
                print("  Motion loader is empty!")
                
            try:
                mm_batch = next(iter(mm_motion_loader))
                print("  MM motion loader has data")
            except StopIteration:
                print("  MM motion loader is empty!")
                
        except Exception as e:
            print(f"  Error checking loaders: {e}")

def evaluation(log_file):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'MM Distance': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({})})
                                   
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mm_score_dict = evaluate_multimodality(mm_motion_loaders, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['MM Distance']:
                    all_metrics['MM Distance'][key] = [item]
                else:
                    all_metrics['MM Distance'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

            for key, item in mm_score_dict.items():
                if key not in all_metrics['MultiModality']:
                    all_metrics['MultiModality'][key] = [item]
                else:
                    all_metrics['MultiModality'][key] += [item]

        # print(all_metrics['Diversity'])
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values))
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)

if __name__ == '__main__':
    mm_num_samples = 10 #100
    mm_num_repeats = 3 #30
    mm_num_times = 10 #10

    diversity_times = 10 # 100
    replication_times = 20 # 20
    batch_size = 128 # fixed batch size for now

    data_cfg = data_cfg = get_config("configs/datasets_duet.yaml").test_set
    cfg_path_list = ["configs/model_duet_debug.yaml"]
    print(cfg_path_list)

    eval_motion_loaders = {}
    for cfg_path in cfg_path_list:
        model_cfg = get_config(cfg_path)
        device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(0)
        model = build_models(model_cfg)
        checkpoint = torch.load(model_cfg.CHECKPOINT, map_location=torch.device("cpu"))
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            for k in list(checkpoint["state_dict"].keys()):
                if "model" in k:
                    checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)

        eval_motion_loaders[model_cfg.NAME] = lambda: get_motion_loader(
                                                batch_size,
                                                model,
                                                gt_dataset,
                                                device,
                                                mm_num_samples,
                                                mm_num_repeats
                                                )

    device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')
    gt_loader, gt_dataset = get_dataset_motion_loader(data_cfg, batch_size)
    evalmodel_cfg = get_config("configs/eval_duet_debug.yaml")
    eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device)

    # Add this line before evaluation() function call
    debug_datasets_and_loaders()
    log_file = f'./evaluation_{1}.log'
    with torch.no_grad():
        evaluation(log_file)
