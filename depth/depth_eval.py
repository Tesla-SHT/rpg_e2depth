import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import numpy as np
from dust3r.utils.image import imread_cv2
import torch
from plyfile import PlyData, PlyElement
import cv2  # noqa
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
import matplotlib.pyplot as plt
from tools import depth_evaluation  # 引入tools中的depth_evaluation方法
from rich.console import Console
console = Console()
def save_depth_image(depth, out_path, cmap='jet_r'):
    #change the depth=0 to depth 1
    depth[depth==0]=1.0
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,4))
    plt.axis('off')
    plt.imshow(depth, cmap=cmap)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=150)
    plt.close()

def visualize_results_separate(depth1, depth2, gt1, gt2, frame_start, frame_end, output_dir, scene_name):
    # view 0 (frame_start)
    if isinstance(depth1, torch.Tensor):
        depth1 = depth1.cpu().detach().numpy()
    if isinstance(depth2, torch.Tensor):
        depth2 = depth2.cpu().detach().numpy()
    if isinstance(gt1, torch.Tensor):
        gt1 = gt1.cpu().detach().numpy()
    if isinstance(gt2, torch.Tensor):
        gt2 = gt2.cpu().detach().numpy()
    f0_pred = os.path.join(output_dir, f"{scene_name}_frame{frame_start}_view0_pred.png")
    f0_gt   = os.path.join(output_dir, f"{scene_name}_frame{frame_start}_view0_gt.png")
    save_depth_image(depth1, f0_pred)
    print("depth1 range", depth1.min(), depth1.max(), depth1.mean())
    print("gt1 range", gt1.min(), gt1.max(), gt1.mean())
    #if gt1>0.5 give it 0.5(limit)
    #gt1 = np.clip(gt1, 0, depth1.max()*2)
    save_depth_image(gt1, f0_gt)
    # view 1 (frame_end)
    f1_pred = os.path.join(output_dir, f"{scene_name}_frame{frame_end}_view1_pred.png")
    f1_gt   = os.path.join(output_dir, f"{scene_name}_frame{frame_end}_view1_gt.png")
    save_depth_image(depth2, f1_pred)
    #gt2 = np.clip(gt2, 0,depth2.max()*2)
    print("gt2 mean", gt2.min(), gt2.max(), gt2.mean())
    save_depth_image(gt2, f1_gt)
    console.print(f"[green]保存视角图像: {f0_pred}, {f0_gt}, {f1_pred}, {f1_gt}[/green]")
def process_and_evaluate_depth_pair(depth, ground_truths, mask, frame_start, output_dir, 
                               iterations=20000, lr=0.5):
    # 确保输入数据为numpy格式
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().detach().numpy()
    if isinstance(ground_truths, torch.Tensor):
        ground_truths = ground_truths.cpu().detach().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()

    results, _, refined_depth, _ = depth_evaluation(
        predicted_depth_original=depth,
        ground_truth_depth_original=ground_truths,
        custom_mask=mask,
        lr=lr,
        max_iters=iterations,
        max_depth=80,
        use_gpu=False,
        align_with_lad=True  
    )

    rmse = results["RMSE"]
    abs_rel = results["Abs Rel"]
    sq_rel = results["Sq Rel"]
    thresh = results["δ < 1.25"]
    console.print(f"[bold yellow]Evaluation Results Frame {frame_start}[/bold yellow]")
    console.print(f"RMSE: {rmse}  AbsRel: {abs_rel}  SqRel: {sq_rel}  δ<1.25: {thresh}")

    refined_depth[mask == 0] = 0
    np.savetxt(os.path.join(output_dir, f"depth_{frame_start}_CUT3R.txt"), refined_depth, fmt='%f')
    # 返回 refined_depth 和评估指标
    return refined_depth, results

def run_pair(model, scene_name, base_path, frame_start_int, frame_end_int, output_dir,
             input_type="frame_voxel", size=512, niter=300, schedule='cosine', lr=0.01):
    """
    复用单对视角深度估计流程，供批处理脚本调用
    捕获所有异常，返回 None 时表示失败
    """
    device = next(model.parameters()).device
    frame_start = str(frame_start_int).zfill(6)
    frame_end = str(frame_end_int).zfill(6)

    try:
        scene_path = f"{base_path}/{scene_name}/1/images"
        p1_path = os.path.join(scene_path, f"frame{frame_start}.jpg")
        p2_path = os.path.join(scene_path, f"frame{frame_end}.jpg")
        if not (os.path.exists(p1_path) and os.path.exists(p2_path)):
            console.print(f"[red]Skip pair {scene_name} {frame_start}-{frame_end} (image missing)[/red]")
            return None

        img1 = cv2.imread(p1_path)
        img2 = cv2.imread(p2_path)
        original_size1 = (img1.shape[1], img1.shape[0])
        original_size2 = (img2.shape[1], img2.shape[0])

        images = load_images([p1_path, p2_path], square_ok=True, size=size)

        if "voxel" in input_type:
            v1_path = os.path.join(scene_path, f"frame{frame_start}.npz")
            v2_path = os.path.join(scene_path, f"frame{frame_end}.npz")
            if os.path.exists(v1_path) and os.path.exists(v2_path):
                voxel1 = load_and_preprocess_voxel(v1_path, original_size1, target_size=size, square_ok=True)
                voxel2 = load_and_preprocess_voxel(v2_path, original_size2, target_size=size, square_ok=True)
                images[0]["voxel"] = voxel1[None, ...]
                images[1]["voxel"] = voxel2[None, ...]
            else:
                console.print(f"[yellow]Voxel files missing for {scene_name} {frame_start}-{frame_end}, continue without voxel[/yellow]")

        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=1, input_type=input_type)

        # ⭐ 关键修改：捕获位姿对齐异常
        try:
            scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)
            scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
        except torch._C._LinAlgError as e:
            console.print(f"[red]❌ Singular matrix error for {scene_name} {frame_start}-{frame_end}[/red]")
            console.print(f"[yellow]尝试使用备用初始化（无MST）...[/yellow]")
            try:
                scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)
                scene.compute_global_alignment(init="none", niter=niter, schedule=schedule, lr=lr)
                console.print(f"[green]✓ 备用方法成功[/green]")
            except Exception as e2:
                console.print(f"[red]❌ 备用方法也失败: {e2}，跳过此对[/red]")
                return None

        mask_path = f"{base_path}/{scene_name}/1/masks"
        mask1 = load_and_preprocess_mask(os.path.join(mask_path, f"frame{frame_start}.png"),
                                         original_size1, target_size=size, square_ok=True)
        mask2 = load_and_preprocess_mask(os.path.join(mask_path, f"frame{frame_end}.png"),
                                         original_size2, target_size=size, square_ok=True)

        depth_path = f"{base_path}/{scene_name}/1/depths"
        gt1_path = os.path.join(depth_path, f"frame{frame_start}.jpg.geometric.png")
        gt2_path = os.path.join(depth_path, f"frame{frame_end}.jpg.geometric.png")
        if not (os.path.exists(gt1_path) and os.path.exists(gt2_path)):
            console.print(f"[red]GT missing for {scene_name} {frame_start}-{frame_end}, skip metrics[/red]")
            return None

        gt1 = load_and_preprocess_mask(gt1_path, original_size1, target_size=size, square_ok=True).astype(np.float32)/255.0
        gt2 = load_and_preprocess_mask(gt2_path, original_size2, target_size=size, square_ok=True).astype(np.float32)/255.0

        depth1, _ = compute_depth(scene, mask1, frame_start, output_dir, 0)
        depth2, _ = compute_depth(scene, mask2, frame_end, output_dir, 1)

        refined_depth1, res1 = process_and_evaluate_depth_pair(depth1, gt1, mask1, frame_start, output_dir)
        refined_depth2, res2 = process_and_evaluate_depth_pair(depth2, gt2, mask2, frame_end, output_dir)

        visualize_results_separate(refined_depth1, refined_depth2, gt1, gt2, frame_start, frame_end, output_dir, scene_name)

        return {
            "scene": scene_name,
            "frame_start": frame_start_int,
            "frame_end": frame_end_int,
            "rmse1": res1["RMSE"],
            "abs_rel1": res1["Abs Rel"],
            "sq_rel1": res1["Sq Rel"],
            "delta1_25_1": res1["δ < 1.25"],
            "rmse2": res2["RMSE"],
            "abs_rel2": res2["Abs Rel"],
            "sq_rel2": res2["Sq Rel"],
            "delta1_25_2": res2["δ < 1.25"],
            "delta1_3_1": res1["δ < 1.03"],
            "delta1_3_2": res2["δ < 1.03"],
        }
    
    except Exception as e:
        # ⭐ 捕获所有其他异常
        console.print(f"[red]❌ Unexpected error for {scene_name} {frame_start_int}-{frame_end_int}: {type(e).__name__}: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return None
def save_tensor_to_ply(tensor, filename='output.ply'):

    # 检查tensor的维度
    assert tensor.dim() == 3 and tensor.shape[2] == 3, "Tensor must have shape [H, W, 3]"
    
    # 将tensor转为numpy数组并展平为 [H*W, 3] 形状
    points = tensor.view(-1, 3).cpu().detach().numpy()

    # 定义点云的顶点
    vertices = np.array([(p[0], p[1], p[2]) for p in points],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    # 创建PlyElement并写入文件
    ply_element = PlyElement.describe(vertices, 'vertex') 
    PlyData([ply_element]).write(filename)

def load_ground_truth(gt_path):
    ground_truths = imread_cv2(gt_path, options=cv2.IMREAD_GRAYSCALE)
    ground_truths = ground_truths.astype(np.float32) / 255.0
    return ground_truths

def compute_depth(scene, mask, frame_start, output_dir, idx):
    depthmaps = scene.get_depthmaps()
    
    depthmap = depthmaps[idx].cpu().detach().numpy()
    camera_intrinsics = scene.get_intrinsics()[idx].cpu().detach().numpy()
    camera_pose = scene.get_im_poses()[idx].cpu().detach().numpy()

    X_world, valid_mask = depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose)
    depth = X_world[..., 2]
    depth[~valid_mask] = 0
    if np.min(depth) < 0:
        depth = -depth
    depth[mask == 0] = 0
    np.savetxt(os.path.join(output_dir, f"original_depth_{frame_start}.txt"), depth, fmt='%f')
    return depth, mask

def process_and_evaluate_depth(depth, ground_truths, mask, frame_start, output_dir, 
                               iterations=20000, lr=0.5):
    # 确保输入数据为numpy格式
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().detach().numpy()
    if isinstance(ground_truths, torch.Tensor):
        ground_truths = ground_truths.cpu().detach().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()

    # 使用tools中的depth_evaluation进行深度优化和评估
    results, _, refined_depth, _ = depth_evaluation(
        predicted_depth_original=depth,
        ground_truth_depth_original=ground_truths,
        custom_mask=mask,
        lr=lr,
        max_iters=iterations,
        max_depth=80,
        use_gpu=False,
        align_with_lad=True  
    )

    # 提取并打印评估指标
    rmse = results["RMSE"]
    abs_rel = results["Abs Rel"]
    sq_rel = results["Sq Rel"]
    thresh = results["δ < 1.25"]
    console.print("[bold yellow]Evaluation Results:[/bold yellow]")
    console.print(f"RMSE: {rmse}")
    console.print(f"Abs Rel: {abs_rel}")
    console.print(f"Sq Rel: {sq_rel}")
    console.print(f"Threshold Inliers: {thresh}")

    # 保存优化后的深度结果
    refined_depth[mask == 0] = 0
    np.savetxt(os.path.join(output_dir, f"depth_{frame_start}_CUT3R.txt"), refined_depth, fmt='%f')
    return refined_depth

def visualize_results(depth1, depth2, ground_truth1, ground_truth2, frame_start, frame_end, output_dir,scene_name):
    plt.subplot(2, 2, 1)
    plt.imshow(depth1, cmap='jet')
    plt.title('Depth 1')
    plt.subplot(2, 2, 2)
    plt.imshow(ground_truth1, cmap='jet')
    plt.title('Ground Truth 1')
    plt.subplot(2, 2, 3)
    plt.imshow(depth2, cmap='jet')
    plt.title('Depth 2')
    plt.subplot(2, 2, 4)
    plt.imshow(ground_truth2, cmap='jet')
    plt.title('Ground Truth 2')
    plt.savefig(os.path.join(output_dir, f"comparison_{scene_name}_{frame_start}_{frame_end}_CUT3R.png"))
    #plt.show()
def load_and_preprocess_mask(mask_path, original_size, target_size, square_ok=False):
    """
    对mask应用与load_images相同的变换
    
    参数:
        mask_path: mask文件路径
        original_size: 原始图像尺寸 (W, H)
        target_size: 目标尺寸
        square_ok: 是否允许正方形
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    W1, H1 = original_size
    
    # 与load_images中相同的resize逻辑
    S = max(W1, H1)
    if target_size == 224:
        new_size = round(target_size * max(W1/H1, H1/W1))
    else:
        new_size = target_size
    
    scale = new_size / S
    new_w, new_h = int(round(W1 * scale)), int(round(H1 * scale))
    
    # 使用最近邻插值来保持mask的二值性
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    W, H = new_w, new_h
    cx, cy = W//2, H//2
    
    # 与load_images中相同的crop逻辑
    if target_size == 224:
        half = min(cx, cy)
        mask = mask[cy-half:cy+half, cx-half:cx+half]
    else:
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        if not square_ok and W == H:
            halfh = 3*halfw//4
        mask = mask[cy-halfh:cy+halfh, cx-halfw:cx+halfw]
    
    return mask
def load_and_preprocess_voxel(voxel_path, original_size, target_size, square_ok=False):
    """
    对voxel应用与load_images相同的变换
    
    参数:
        voxel_path: voxel文件路径
        original_size: 原始图像尺寸 (W, H)
        target_size: 目标尺寸
        square_ok: 是否允许正方形
    """
    # 加载voxel数据
    input_metadata = np.load(voxel_path)
    voxel = input_metadata["voxel"]  # 假设shape为 [C, H, W] 或 [H, W, C]
    
    # 确定voxel的维度顺序
    if voxel.ndim == 3:
        if voxel.shape[0] < voxel.shape[2]:  # [C, H, W]
            C, H_orig, W_orig = voxel.shape
            voxel = voxel.transpose(1, 2, 0)  # 转为 [H, W, C]
        else:  # [H, W, C]
            H_orig, W_orig, C = voxel.shape
    else:
        raise ValueError(f"Unexpected voxel shape: {voxel.shape}")
    
    W1, H1 = original_size
    
    # 与load_images中相同的resize逻辑
    S = max(W1, H1)
    if target_size == 224:
        new_size = round(target_size * max(W1/H1, H1/W1))
    else:
        new_size = target_size
    
    scale = new_size / S
    new_w, new_h = int(round(W1 * scale)), int(round(H1 * scale))
    
    # 对每个通道分别resize
    voxel_resized = np.zeros((new_h, new_w, C), dtype=voxel.dtype)
    for c in range(C):
        voxel_resized[:, :, c] = cv2.resize(
            voxel[:, :, c], 
            (new_w, new_h), 
            interpolation=cv2.INTER_LINEAR
        )
    
    W, H = new_w, new_h
    cx, cy = W//2, H//2
    
    # 与load_images中相同的crop逻辑
    if target_size == 224:
        half = min(cx, cy)
        voxel_cropped = voxel_resized[cy-half:cy+half, cx-half:cx+half, :]
    else:
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        if not square_ok and W == H:
            halfh = 3*halfw//4
        voxel_cropped = voxel_resized[cy-halfh:cy+halfh, cx-halfw:cx+halfw, :]
    
    # 转回 [C, H, W] 格式
    #voxel_cropped = voxel_cropped.transpose(2, 0, 1)
    
    return voxel_cropped
if __name__ == '__main__':
    import argparse
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    # Define the type of the input and model among (frame, voxel, frame_voxel), no voxel ckpt now
    type = "voxel"
    parser = argparse.ArgumentParser(description="Depth estimation script.")
    parser.add_argument("--scene_name", type=str, required=True, help="The name of the scene.")
    parser.add_argument("--output_dir", type=str, default="/home/w/Documents/project/data/dust3r_event_output/EvGGS", help="The output directory.")
    parser.add_argument("--frame_start", type=int, required=True, help="The starting frame index.")
    parser.add_argument("--frame_end", type=int, required=True, help="The ending frame index.")
    args = parser.parse_args()

    scene_name = args.scene_name
    output_dir = args.output_dir
    idx = args.frame_start
    frame_start = str(args.frame_start).zfill(6)
    frame_end = str(args.frame_end).zfill(6)
    if (type == "frame"):
        model_name = "checkpoints/dust3r_demo_224/checkpoint-frame-224-best.pth"
    elif (type == "frame_voxel"):
        model_name = "checkpoints/indoor_dpt512_old/checkpoint-best.pth"
    elif (type == "voxel"):
        model_name = "checkpoints/indoor_dpt512_old/checkpoint-best.pth"
    
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name, input_type=type).to(device)
    base_path =  f"/run/user/1000/gvfs/sftp:host=10.0.1.67,port=22332/datasets/feed_forward_event/Tartanair_tmp/hospital_no_overlap"
    #base_path = f"/run/determined/workdir/data/feed_forward_event/Tartanair_tmp/indoor"
    scene_path = f"{base_path}/{scene_name}/1/images"

    # Load the frame data
    p1_path = os.path.join(scene_path, f"frame{frame_start}.jpg")
    p2_path = os.path.join(scene_path, f"frame{frame_end}.jpg")
    img1 = cv2.imread(p1_path)
    original_size1 = (img1.shape[1], img1.shape[0])  # (W, H)
    
    img2 = cv2.imread(p2_path)
    original_size2 = (img2.shape[1], img2.shape[0])  # (W, H)
    
    if ("frame" in type):
        # load_images can take a list of images or a directory, remember to change the image size
        images = load_images([p1_path, p2_path], square_ok=True, size=512)

    # Load the voxel data
    if ("voxel" in type):
        images = load_images([p1_path, p2_path], square_ok=True, size=512)
        p1_path = os.path.join(scene_path, f"frame{frame_start}.npz")
        p2_path = os.path.join(scene_path, f"frame{frame_end}.npz")
        # 对voxel应用相同的预处理
        voxel1 = load_and_preprocess_voxel(p1_path, original_size1, target_size=512, square_ok=True)
        voxel2 = load_and_preprocess_voxel(p2_path, original_size2, target_size=512, square_ok=True)
        images[0]["voxel"] = voxel1[None, ...]  # 添加batch维度
        images[1]["voxel"] = voxel2[None, ...]

    # print(images)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

    output = inference(pairs, model, device, batch_size=batch_size, input_type=type)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    mask_path = f"{base_path}/{scene_name}/1/masks"

    # 加载并预处理mask
    mask1_path = os.path.join(mask_path, f"frame{frame_start}.png")
    mask1 = load_and_preprocess_mask(mask1_path, original_size1, target_size=512, square_ok=True)
    
    mask2_path = os.path.join(mask_path, f"frame{frame_end}.png")
    mask2 = load_and_preprocess_mask(mask2_path, original_size2, target_size=512, square_ok=True)
    
    #get depthmap from get_depthmaps
    # depthmap = scene.get_depthmaps()
    # depth = depthmap[0].cpu().detach().numpy()
    # depth[mask1 == 0] = 0

    console.rule("Starting the depth processing...")
    depth_path = base_path+ f"/{scene_name}/1/depths"
    #use png to load GT
    gt_path = os.path.join(depth_path, f"frame{frame_start}.jpg.geometric.png")
    #create output folder
    os.makedirs(output_dir, exist_ok=True)
    #load gt image, do not use imageio
    console.rule("[bold red] Left View")
    ground_truth1 = load_ground_truth(gt_path)    
    ground_truth1 = load_and_preprocess_mask(gt_path, original_size1, target_size=512, square_ok=True)
    ground_truth1 = ground_truth1.astype(np.float32) / 255.0
    depth1, mask1 = compute_depth(scene, mask1, frame_start, output_dir, 0)
    refined_depth1 = process_and_evaluate_depth(depth1, ground_truth1, mask1, frame_start, output_dir)

    # 处理第二张图片
    gt_path2 = os.path.join(depth_path, f"frame{frame_end}.jpg.geometric.png")
    console.rule("[bold red] Right View")
    ground_truth2 = load_ground_truth(gt_path2)
    ground_truth2 = load_and_preprocess_mask(gt_path2, original_size2, target_size=512, square_ok=True)
    ground_truth2 = ground_truth2.astype(np.float32) / 255.0
    depth2, mask2 = compute_depth(scene, mask2, frame_end, output_dir, 1)
    refined_depth2 = process_and_evaluate_depth(depth2, ground_truth2, mask2, frame_end, output_dir)

    # 可视化结果
    visualize_results(refined_depth1, refined_depth2, ground_truth1, ground_truth2, frame_start, frame_end, output_dir,scene_name)


'''
python eval/depth/depth_eval.py --scene_name japanesealley_easy_P003 --frame_start 485 --frame_end 495 --output output/japanesealley_easy_P003
python eval/depth/depth_eval.py --scene_name hospital_easy_P002 --frame_start 10 --frame_end 15 --output output/hospital_easy_P002
'''