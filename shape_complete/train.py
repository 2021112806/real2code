import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
# 设置 wandb API key 环境变量，避免后续登录问题
os.environ['WANDB_API_KEY'] = '5dc1a2e658c0c37de621f76763c96805cc8bc8d0'

import torch
import wandb 
import json
import trimesh 
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join
from models import ShapeCompletionModel
from dataset import ShapeCompletionDataset, ShapeCompletionEvalDataset
from torch.nn import DataParallel
from einops import rearrange
from torch.utils.data import DataLoader
from kaolin.ops.conversions import voxelgrids_to_trianglemeshes

"""
python train.py --use_dp -dp 3 -rn scissors_eyeglasses --data_dir /store/real/mandi/real2code_shape_dataset_v0 --load_voxelgrid
"""

def get_loss(model, pred, target):
    if isinstance(model, DataParallel):
        model = model.module
    return model.compute_loss(pred, target)

def collate_fn(batch):
    """
    自定义collate函数，处理不同大小的点云
    对于input点云和query点云，如果大小不一致，进行padding或截断到批次中的最大大小
    直接在GPU上处理数据
    """
    # 获取各个样本的大小和设备
    input_sizes = [item['input'].shape[0] for item in batch]
    query_sizes = [item['query'].shape[0] for item in batch]
    
    # 获取第一个样本的设备（假设所有样本在同一设备上）
    device = batch[0]['input'].device
    
    # 计算批次中的最大大小
    max_input_size = max(input_sizes)
    max_query_size = max(query_sizes)
    
    collated_batch = {}
    
    # 处理input点云
    inputs = []
    for item in batch:
        pcd = item['input']
        
        if pcd.shape[0] < max_input_size:
            # Padding with zeros
            pad_size = max_input_size - pcd.shape[0]
            pad = torch.zeros(pad_size, pcd.shape[1], dtype=pcd.dtype, device=device)
            pcd = torch.cat([pcd, pad], dim=0)
        elif pcd.shape[0] > max_input_size:
            # Truncate (理论上不应该发生，但为了安全起见)
            pcd = pcd[:max_input_size]
        inputs.append(pcd)
    collated_batch['input'] = torch.stack(inputs, dim=0)
    
    # 处理query点云
    queries = []
    for item in batch:
        query = item['query']
        
        if query.shape[0] < max_query_size:
            # Padding with zeros
            pad_size = max_query_size - query.shape[0]
            pad = torch.zeros(pad_size, query.shape[1], dtype=query.dtype, device=device)
            query = torch.cat([query, pad], dim=0)
        elif query.shape[0] > max_query_size:
            # Truncate
            query = query[:max_query_size]
        queries.append(query)
    collated_batch['query'] = torch.stack(queries, dim=0)
    
    # 处理labels
    labels = []
    for item in batch:
        label = item['label']
        
        if label.shape[0] < max_query_size:
            # Padding with zeros (表示无效点)
            pad_size = max_query_size - label.shape[0]
            pad = torch.zeros(pad_size, dtype=label.dtype, device=device)
            label = torch.cat([label, pad], dim=0)
        elif label.shape[0] > max_query_size:
            # Truncate
            label = label[:max_query_size]
        labels.append(label)
    collated_batch['label'] = torch.stack(labels, dim=0)
    
    # 处理其他字段（如果存在）
    if 'raw_pcd' in batch[0]:
        # raw_pcd可能有不同大小，使用列表存储
        collated_batch['raw_pcd'] = [item['raw_pcd'] for item in batch]
    
    return collated_batch

def get_vis_table(model, loader, output_dir, num_vis_steps, skip_extents=True, use_max_extents=False, voxel_size=96):
    if isinstance(model, DataParallel):
        model = model.module
    table = wandb.Table(columns=["Input", "Target", "Prediction", "Loss"])
    rows = []
    for i, batch in enumerate(loader): 
        with torch.no_grad():
            pred = model(batch["input"].cuda(), batch["query"].cuda())
            label = batch["label"].cuda()
            logits = torch.sigmoid(pred).detach()
            loss = model.compute_loss(pred, label).item()
            R = batch['R'][0].cpu().numpy()
            center = batch['center'][0].cpu().numpy()
            extent = batch['extent'][0].cpu().numpy()

            pred_voxelgrid = rearrange(logits, 'b (x y z) -> b x y z', x=voxel_size, y=voxel_size, z=voxel_size)
            verts, faces = voxelgrids_to_trianglemeshes(pred_voxelgrid, iso_value=0.5)
            vertices = verts[0].cpu().numpy()
            faces = faces[0].cpu().numpy()
            if not skip_extents and not use_max_extents:
                vertices = vertices * extent
            if use_max_extents:
                vertices = vertices * np.max(extent)
            vertices = vertices @ R.T + center
            _mesh = trimesh.Trimesh(vertices, faces)
            _mesh.export(join(output_dir, f"pred_{i}.obj"))
            
            label_grid = rearrange(label, 'b (x y z) -> b x y z', x=voxel_size, y=voxel_size, z=voxel_size)
            verts, faces = voxelgrids_to_trianglemeshes(label_grid, iso_value=0.6)
            vertices = verts[0].cpu().numpy()
            faces = faces[0].cpu().numpy() 
            if not skip_extents and not use_max_extents:
                vertices = vertices * extent
            if use_max_extents:
                vertices = vertices * np.max(extent)
            vertices = vertices @ R.T + center
            _mesh = trimesh.Trimesh(vertices, faces)
            _mesh.export(join(output_dir, f"label_{i}.obj"))

            inp = wandb.Object3D(batch["input"][0].cpu().numpy())
            row_data = [
                inp, 
                wandb.Object3D(join(output_dir, f"label_{i}.obj")),
                wandb.Object3D(join(output_dir, f"pred_{i}.obj")),
                loss
            ]
            rows.append(row_data)
            table.add_data(*row_data)
        if i == num_vis_steps - 1:
            break

    return table, rows

def run(args):
    # setup dataset and loader
    dataset_kwargs = dict(
        data_dir=args.data_dir,  
        input_size=args.num_input_points,
        query_size=args.num_query_points,
        obj_type=args.obj_type,
        obj_folder=args.obj_folder,
        loop_dir=args.loop_dir,
        query_surface_ratio=args.query_surface_ratio,
        cache_mesh=args.cache_mesh,
        load_voxelgrid=args.load_voxelgrid,
        rot_aug=args.rot_aug,
        aug_obb=args.aug_obb,
        voxel_size=args.voxel_size,
        load_pcd_size=args.load_pcd_size,
        skip_extents=args.skip_extents,
        use_max_extents=args.use_max_extents,
        use_aabb=args.use_aabb
        )
    dataset = ShapeCompletionDataset(split="train", **dataset_kwargs)
    val_dataset_kwargs = dataset_kwargs.copy()
    val_dataset_kwargs["split"] = "test"
    val_dataset_kwargs["loop_dir"] = "0"
    val_dataset = ShapeCompletionDataset(**val_dataset_kwargs)

    vis_dataset = ShapeCompletionEvalDataset(**val_dataset_kwargs)
    vis_loader = DataLoader(
        vis_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
        )
    print(f"Loaded {len(dataset)} samples per epoch, {len(val_dataset)} samples for validation")
    val_dataloader = None 
    if len(val_dataset) > 0:
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            collate_fn=collate_fn
            ) 
    # setup model
    model = ShapeCompletionModel(
        agg_args=args.agg_args,
        unet_args=args.unet_args,
        decoder_args=args.decoder_args
        )
    model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.resume is not None:
        assert args.load_step is not None, "Must provide load step if resuming"
        model_dir = join(args.log_dir, args.resume, f"step_{args.load_step}")
        model_fname = join(model_dir, f"model_{args.load_step}.pth")
        assert os.path.exists(model_fname), f"Model file {model_fname} does not exist"
        print(f"Loading model from {model_fname}")
        model.load_state_dict(torch.load(model_fname))

        optimizer_fname = join(model_dir, "optimizer.pth")
        if os.path.exists(optimizer_fname):
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            print(f"Loading optimizer from {optimizer_fname}")
            optimizer.load_state_dict(torch.load(optimizer_fname))
    if args.use_dp:
        model = DataParallel(model, device_ids=[i for i in range(args.dp_devices)])

    run_name = f"{args.run_name}_query{args.num_query_points}_inp{args.num_input_points}_qr{args.query_surface_ratio}_bs{args.batch_size}_lr{args.learning_rate}"

    if args.wandb:
        run = wandb.init(project="real2code", entity="1785115532-harbin-institute-of-technology", group="shape", name=run_name, reinit=False)
        wandb.config.update(args)

    output_dir = join(args.log_dir, run_name)
    os.makedirs(output_dir, exist_ok=(args.overwrite or args.resume is not None))
    save_args = vars(args)
    save_args["run_name"] = run_name
    save_args["dataset_kwargs"] = dataset_kwargs
    with open(join(output_dir, "args.json"), "w") as f:
        json.dump(save_args, f)
        
    # training loop
    total_steps = 0
    if args.resume is not None:
        total_steps = int(args.load_step)
    print("Begin Training")
    for epoch in range(args.num_epochs):
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}"
        ):
            input_points, query_points, labels = data["input"], data["query"], data["label"]
            # 数据已经在GPU上（由collate_fn处理），不需要再移到GPU
            optimizer.zero_grad() 
            pred = model(input_points, query_points) 
            loss = get_loss(model, pred, labels)
            loss.backward()
            optimizer.step()
            total_steps += 1

            tolog = dict(loss=loss.item(), train_step=total_steps)
            
            if total_steps % args.save_interval == 0:
                step_outdir = join(output_dir, f"step_{total_steps}")
                os.makedirs(step_outdir, exist_ok=True)
                save_model = model.module if args.use_dp else model
                save_model.save(os.path.join(step_outdir, f"model_{total_steps}.pth"))
                optimizer_fname = join(step_outdir, "optimizer.pth")
                torch.save(optimizer.state_dict(), optimizer_fname)
            
            if total_steps % args.val_interval == 0 and val_dataloader is not None:
                with torch.no_grad():
                    val_loss = 0
                    for j, val_data in enumerate(val_dataloader):
                        input_points, query_points, labels = val_data["input"], val_data["query"], val_data["label"]
                        # 数据已经在GPU上（由collate_fn处理），不需要再移到GPU
                        pred = model(input_points, query_points)
                        val_loss += get_loss(model, pred, labels).item()
                        if j > args.num_val_steps:
                            break
                val_loss /= args.num_val_steps
                tolog['val/loss'] = val_loss
                print(f"Validation Loss {val_loss}")

            if total_steps % args.vis_interval == 0:
                visualize_table, rows = get_vis_table(model, vis_loader, output_dir, args.num_vis_steps, args.skip_extents, args.use_max_extents, args.voxel_size)
                for i, row in enumerate(rows):
                    for j, key in enumerate(["Input", "Target", "Prediction", "Loss"]):
                        tolog[f"vis/{i}_{key}"] = row[j]
            if total_steps % args.log_interval == 0:
                print(f"Epoch {epoch}, Trained Step {total_steps}, Loss {loss.item()}")
                if args.wandb:
                    wandb.log(tolog)
                else:
                    print(tolog)

    return 

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    # dataset and loader:
    parser.add_argument("--data_dir", type=str, default="/local/real/mandi/shape_dataset_v4")
    parser.add_argument("--batch_size", "-b", type=int, default=96)
    parser.add_argument("--num_input_points", "-i", type=int, default=1024)
    parser.add_argument("--num_query_points", "-q", type=int, default=6000)
    parser.add_argument("--query_surface_ratio", "-qr", type=float, default=0.2)
    parser.add_argument("--num_workers", "-w", type=int, default=0)
    parser.add_argument("--obj_type", type=str, default="*")    
    parser.add_argument("--obj_folder", type=str, default="*")  
    parser.add_argument("--loop_dir", type=str, default="*")
    parser.add_argument("--cache_mesh", action="store_true")
    parser.add_argument("--load_pcd_size", type=int, default=0)
    parser.add_argument("--skip_extents", action="store_true")
    parser.add_argument("--use_max_extents", "-me", action="store_true")
    parser.add_argument("--use_aabb", action="store_true")
    parser.add_argument("--voxel_size", "-vs", type=int, default=96)
    parser.add_argument("--rot_aug", action="store_true")
    parser.add_argument("--aug_obb", action="store_true")
    parser.add_argument("--load_voxelgrid", action="store_true")
    # model:
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", "-e", type=int, default=5000)
    parser.add_argument("--agg_args", type=dict, default=dict())
    parser.add_argument("--unet_args", type=dict, default=dict(in_channels=128, out_channels=128))
    parser.add_argument("--decoder_args", type=dict, default=dict())
    parser.add_argument("--val_interval", "-vi", type=int, default=200)
    parser.add_argument("--num_val_steps", "-v", type=int, default=100)
    parser.add_argument("--use_dp", action="store_true")
    parser.add_argument("--dp_devices", "-dp", type=int, default=2)
    parser.add_argument("--resume", "-r", type=str, default=None)
    parser.add_argument("--load_step", "-ls", type=int, default=None)
    # logging:
    parser.add_argument("--log_dir", "-ld", type=str, default="/mnt/data/zhangzhaodong/real2code/outputs/shape_models/")
    parser.add_argument("--log_interval", "-log", type=int, default=200)
    parser.add_argument("--save_interval", "-save", type=int, default=2000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", "-rn", type=str, default="test")
    parser.add_argument("--overwrite", "-o", action="store_true")
    parser.add_argument("--vis_interval", "-vis", type=int, default=5000)
    parser.add_argument("--num_vis_steps", "-nv", type=int, default=5)
    args = parser.parse_args()
    run(args)
    print("Done")