#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMTK到PKL格式转换器（仅输入数据）

此脚本将BMTK格式的输入数据转换为input_dat.pkl格式
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import h5py
import json
import pickle as pkl
from collections import defaultdict
from tqdm import tqdm
import time


def convert_input_data(bmtk_dir, output_dir='.'):
    """将BMTK输入数据转换为pkl格式"""
    
    print("\n=== 开始转换输入数据 ===")
    input_populations = []
    
    # 1. 处理LGN输入
    print("步骤 1/4: 处理LGN输入数据...")
    lgn_node_types = pd.read_csv(os.path.join(bmtk_dir, 'network/lgn_node_types.csv'), sep=' ')
    lgn_nodes_h5 = h5py.File(os.path.join(bmtk_dir, 'network/lgn_nodes.h5'), 'r')
    lgn_edge_types = pd.read_csv(os.path.join(bmtk_dir, 'network/lgn_v1_edge_types.csv'), sep=' ')
    lgn_edges_h5 = h5py.File(os.path.join(bmtk_dir, 'network/lgn_v1_edges.h5'), 'r')
    print(f"  - LGN节点类型: {len(lgn_node_types)}")
    print(f"  - LGN边类型: {len(lgn_edge_types)}")
    
    # 读取LGN spike数据
    lgn_spikes_file = os.path.join(bmtk_dir, 'inputs/lgn_inputs/spikes.trial_0.h5')
    if os.path.exists(lgn_spikes_file):
        lgn_spikes_h5 = h5py.File(lgn_spikes_file, 'r')
        lgn_spike_times = np.array(lgn_spikes_h5['spikes']['lgn']['timestamps'])
        lgn_spike_ids = np.array(lgn_spikes_h5['spikes']['lgn']['node_ids'])
    else:
        # 如果没有spike文件，创建空的spike数据
        lgn_spike_times = np.array([])
        lgn_spike_ids = np.array([])
    
    # 获取LGN节点信息
    lgn_node_ids = np.array(lgn_nodes_h5['nodes']['lgn']['node_id'])
    print(f"  - LGN节点数量: {len(lgn_node_ids)}")
    print(f"  - LGN spike数量: {len(lgn_spike_times)}")
    
    # 构建spike字典
    print("步骤 2/4: 构建LGN spike字典...")
    lgn_spikes_dict = defaultdict(list)
    for spike_id, spike_time in tqdm(zip(lgn_spike_ids, lgn_spike_times), total=len(lgn_spike_times), desc="处理LGN spikes"):
        lgn_spikes_dict[spike_id].append(spike_time)
    
    # 构建LGN节点信息
    lgn_nodes_info = {
        'ids': lgn_node_ids.tolist(),
        'spikes': [np.array(lgn_spikes_dict.get(node_id, [])) for node_id in lgn_node_ids]
    }
    
    # 构建LGN到V1的连接信息
    print("构建LGN到V1连接...")
    lgn_edges_info = []
    edge_groups = lgn_edges_h5['edges']['lgn_to_v1']
    
    # 读取全局的source和target节点ID
    all_source_ids = np.array(edge_groups['source_node_id'])
    all_target_ids = np.array(edge_groups['target_node_id'])
    all_edge_type_ids = np.array(edge_groups['edge_type_id'])
    
    # 预读取LGN权重数据
    lgn_weight_cache = {}
    if 'syn_weight' in edge_groups:
        try:
            lgn_weight_cache['global'] = np.array(edge_groups['syn_weight'])
        except:
            pass
    
    # 预读取LGN edge_group相关数据
    lgn_edge_group_ids = None
    lgn_edge_group_indices = None
    if 'edge_group_id' in edge_groups:
        lgn_edge_group_ids = np.array(edge_groups['edge_group_id'])
    if 'edge_group_index' in edge_groups:
        lgn_edge_group_indices = np.array(edge_groups['edge_group_index'])
    
    # 预读取LGN子组权重
    lgn_subgroup_weights_cache = {}
    for key in edge_groups.keys():
        if key.isdigit() and 'syn_weight' in edge_groups[key]:
            try:
                lgn_subgroup_weights_cache[key] = np.array(edge_groups[key]['syn_weight'])
            except:
                pass
    
    # 统计有效LGN边类型数量用于进度条（避免内存问题）
    valid_lgn_edge_types = []
    for edge_type_idx in range(len(lgn_edge_types)):
        edge_type_row = lgn_edge_types.iloc[edge_type_idx]
        edge_type_id = edge_type_row['edge_type_id']
        if edge_type_id in all_edge_type_ids:
            valid_lgn_edge_types.append(edge_type_idx)
    
    for edge_type_idx in tqdm(valid_lgn_edge_types, desc="构建LGN到V1连接", unit="边类型"):
        edge_type_row = lgn_edge_types.iloc[edge_type_idx]
        edge_type_id = edge_type_row['edge_type_id']
        
        # 使用分批处理策略避免内存问题
        matching_indices = np.where(all_edge_type_ids == edge_type_id)[0]
        if len(matching_indices) == 0:
            continue
            
        source_ids = all_source_ids[matching_indices]
        target_ids = all_target_ids[matching_indices]
        
        # 获取权重 - 使用缓存的数据
        weights = None
        
        # 首先尝试从全局权重缓存读取
        if 'global' in lgn_weight_cache:
            try:
                weights = lgn_weight_cache['global'][matching_indices]
            except:
                pass
        
        # 如果没有全局权重，尝试从子组读取
        if weights is None and lgn_edge_group_ids is not None and lgn_edge_group_indices is not None:
            try:
                edge_group_ids = lgn_edge_group_ids[matching_indices]
                if len(edge_group_ids) > 0:
                    subgroup_id = str(edge_group_ids[0])
                    if subgroup_id in lgn_subgroup_weights_cache:
                        edge_group_indices = lgn_edge_group_indices[matching_indices]
                        weights = lgn_subgroup_weights_cache[subgroup_id][edge_group_indices]
            except:
                pass
        
        # 如果还是没有权重数据，使用默认值
        if weights is None:
            weights = np.full(len(source_ids), 0.3, dtype=np.float32)  # 默认LGN权重
        
        # 获取receptor_type
        syn_params_file = edge_type_row.get('dynamics_params', 'LGN_to_GLIF.json')
        if syn_params_file == 'LGN_to_GLIF.json' or syn_params_file == 'LGN_to_LIF.json':
            receptor_type = 1  # LGN输入通常是兴奋性的
        else:
            receptor_type = 1
        
        edge_info = {
            'source': source_ids,
            'target': target_ids,
            'params': {
                'weight': weights,
                'delay': edge_type_row.get('delay', 1.0),
                'receptor_type': receptor_type
            }
        }
        lgn_edges_info.append(edge_info)
    
    input_populations.append((lgn_nodes_info, lgn_edges_info))
    
    # 2. 处理背景输入
    print("步骤 3/4: 处理背景输入数据...")
    bkg_nodes_h5 = h5py.File(os.path.join(bmtk_dir, 'network/bkg_nodes.h5'), 'r')
    bkg_edge_types = pd.read_csv(os.path.join(bmtk_dir, 'network/bkg_v1_edge_types.csv'), sep=' ')
    bkg_edges_h5 = h5py.File(os.path.join(bmtk_dir, 'network/bkg_v1_edges.h5'), 'r')
    print(f"  - 背景边类型: {len(bkg_edge_types)}")
    
    # 读取背景spike数据
    bkg_spikes_file = os.path.join(bmtk_dir, 'inputs/bkg_inputs/spikes_3s_1500fr.bkg.h5')
    if os.path.exists(bkg_spikes_file):
        bkg_spikes_h5 = h5py.File(bkg_spikes_file, 'r')
        bkg_spike_times = np.array(bkg_spikes_h5['spikes']['bkg']['timestamps'])
        bkg_spike_ids = np.array(bkg_spikes_h5['spikes']['bkg']['node_ids'])
    else:
        # 如果没有spike文件，创建泊松分布的背景活动
        bkg_node_ids = np.array(bkg_nodes_h5['nodes']['bkg']['node_id'])
        bkg_spike_times = []
        bkg_spike_ids = []
        # 为每个背景节点生成1500Hz的泊松spike
        print("  - 生成背景泊松spikes...")
        for node_id in tqdm(bkg_node_ids, desc="生成背景spikes"):
            n_spikes = np.random.poisson(1500 * 3)  # 3秒，1500Hz
            spike_times = np.sort(np.random.uniform(0, 3000, n_spikes))
            bkg_spike_times.extend(spike_times)
            bkg_spike_ids.extend([node_id] * n_spikes)
        bkg_spike_times = np.array(bkg_spike_times)
        bkg_spike_ids = np.array(bkg_spike_ids)
    
    # 获取背景节点信息
    bkg_node_ids = np.array(bkg_nodes_h5['nodes']['bkg']['node_id'])
    print(f"  - 背景节点数量: {len(bkg_node_ids)}")
    print(f"  - 背景spike数量: {len(bkg_spike_times)}")
    
    # 构建spike字典
    print("构建背景spike字典...")
    bkg_spikes_dict = defaultdict(list)
    for spike_id, spike_time in tqdm(zip(bkg_spike_ids, bkg_spike_times), total=len(bkg_spike_times), desc="处理背景spikes"):
        bkg_spikes_dict[spike_id].append(spike_time)
    
    # 构建背景节点信息
    bkg_nodes_info = {
        'ids': bkg_node_ids.tolist(),
        'spikes': [np.array(bkg_spikes_dict.get(node_id, [])) for node_id in bkg_node_ids]
    }
    
    # 构建背景到V1的连接信息
    print("构建背景到V1连接...")
    bkg_edges_info = []
    edge_groups = bkg_edges_h5['edges']['bkg_to_v1']
    
    # 读取全局的source和target节点ID
    all_source_ids = np.array(edge_groups['source_node_id'])
    all_target_ids = np.array(edge_groups['target_node_id'])
    all_edge_type_ids = np.array(edge_groups['edge_type_id'])
    
    # 预读取背景权重数据
    bkg_weight_cache = {}
    if 'syn_weight' in edge_groups:
        try:
            bkg_weight_cache['global'] = np.array(edge_groups['syn_weight'])
        except:
            pass
    
    # 预读取背景edge_group相关数据
    bkg_edge_group_ids = None
    bkg_edge_group_indices = None
    if 'edge_group_id' in edge_groups:
        bkg_edge_group_ids = np.array(edge_groups['edge_group_id'])
    if 'edge_group_index' in edge_groups:
        bkg_edge_group_indices = np.array(edge_groups['edge_group_index'])
    
    # 预读取背景子组权重
    bkg_subgroup_weights_cache = {}
    for key in edge_groups.keys():
        if key.isdigit() and 'syn_weight' in edge_groups[key]:
            try:
                bkg_subgroup_weights_cache[key] = np.array(edge_groups[key]['syn_weight'])
            except:
                pass
    
    # 统计有效背景边类型数量用于进度条（避免内存问题）
    valid_bkg_edge_types = []
    for edge_type_idx in range(len(bkg_edge_types)):
        edge_type_row = bkg_edge_types.iloc[edge_type_idx]
        edge_type_id = edge_type_row['edge_type_id']
        if edge_type_id in all_edge_type_ids:
            valid_bkg_edge_types.append(edge_type_idx)
    
    for edge_type_idx in tqdm(valid_bkg_edge_types, desc="构建背景到V1连接", unit="边类型"):
        edge_type_row = bkg_edge_types.iloc[edge_type_idx]
        edge_type_id = edge_type_row['edge_type_id']
        
        # 使用分批处理策略避免内存问题
        matching_indices = np.where(all_edge_type_ids == edge_type_id)[0]
        if len(matching_indices) == 0:
            continue
            
        source_ids = all_source_ids[matching_indices]
        target_ids = all_target_ids[matching_indices]
        
        # 获取权重 - 使用缓存的数据
        weights = None
        
        # 首先尝试从全局权重缓存读取
        if 'global' in bkg_weight_cache:
            try:
                weights = bkg_weight_cache['global'][matching_indices]
            except:
                pass
        
        # 如果没有全局权重，尝试从子组读取
        if weights is None and bkg_edge_group_ids is not None and bkg_edge_group_indices is not None:
            try:
                edge_group_ids = bkg_edge_group_ids[matching_indices]
                if len(edge_group_ids) > 0:
                    subgroup_id = str(edge_group_ids[0])
                    if subgroup_id in bkg_subgroup_weights_cache:
                        edge_group_indices = bkg_edge_group_indices[matching_indices]
                        weights = bkg_subgroup_weights_cache[subgroup_id][edge_group_indices]
            except:
                pass
        
        # 如果还是没有权重数据，使用默认值
        if weights is None:
            weights = np.full(len(source_ids), 0.02, dtype=np.float32)  # 背景输入通常较弱
        
        edge_info = {
            'source': source_ids,
            'target': target_ids,
            'params': {
                'weight': weights,
                'delay': edge_type_row.get('delay', 1.0),
                'receptor_type': 1  # 背景输入通常是兴奋性的
            }
        }
        bkg_edges_info.append(edge_info)
    
    input_populations.append((bkg_nodes_info, bkg_edges_info))
    
    # 保存数据
    print("步骤 4/4: 保存输入数据...")
    
    output_file = os.path.join(output_dir, 'input_dat.pkl')
    os.makedirs(output_dir, exist_ok=True)
    
    # 直接保存input_populations列表以兼容load_sparse.py
    with open(output_file, 'wb') as f:
        pkl.dump(input_populations, f)
    
    print(f"✓ 输入数据已保存到 {output_file}")
    print(f"✓ LGN输入: {len(lgn_node_ids)} 个节点")
    print(f"✓ 背景输入: {len(bkg_node_ids)} 个节点")
    print("=== 输入数据转换完成 ===")
    
    return input_populations


def main():
    """主函数，处理命令行参数并执行转换"""
    parser = argparse.ArgumentParser(
        description='将BMTK格式的输入数据转换为input_dat.pkl格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python bmtk_to_pkl_converter.py Allen_V1_param
  python bmtk_to_pkl_converter.py Allen_V1_param Allen_V1_param"""
    )
    
    parser.add_argument('input_dir', 
                       help='BMTK数据文件夹路径')
    parser.add_argument('output_dir', 
                       nargs='?', 
                       default='.', 
                       help='输出文件夹路径（默认为当前目录）')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录 '{args.input_dir}' 不存在")
        sys.exit(1)
    
    # 检查是否是有效的BMTK目录
    network_dir = os.path.join(args.input_dir, 'network')
    if not os.path.exists(network_dir):
        print(f"错误: '{args.input_dir}' 不是有效的BMTK数据目录（缺少network文件夹）")
        sys.exit(1)
    
    start_time = time.time()
    print("\n" + "="*60)
    print("    BMTK到PKL格式转换器（仅输入数据）")
    print("="*60)
    print(f"输入目录: {os.path.abspath(args.input_dir)}")
    print(f"输出目录: {os.path.abspath(args.output_dir)}")
    
    try:
        # 转换输入数据
        input_start = time.time()
        input_data = convert_input_data(args.input_dir, args.output_dir)
        input_time = time.time() - input_start
        print(f"输入数据转换耗时: {input_time:.2f} 秒")
        
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("    转换完成！")
        print("="*60)
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"输入数据: {os.path.join(args.output_dir, 'input_dat.pkl')}")
            
    except Exception as e:
        print(f"\n错误: 转换过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()