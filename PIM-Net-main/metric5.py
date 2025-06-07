import os
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score # 新增 accuracy_score
from tqdm import tqdm

def preprocess_gt(gt_path, target_size=(384, 384)):
    # 读取GT图像并将其像素值归一化为0或1
    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        print(f"错误: 无法读取GT图像 {gt_path}")
        return None
    gt_img = cv2.resize(gt_img, target_size)  # 调整到目标尺寸
    # 对于GT，通常我们期望它是二值的 (0 或 255)，然后归一化到 (0 或 1)
    _, gt_img_binary = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY) # 先二值化
    gt_img_normalized = gt_img_binary / 255.0
    return gt_img_normalized

def preprocess_pred(pred_path, target_size=(384, 384)):
    # 读取预测图像并将其像素值归一化到0-1范围
    pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    if pred_img is None:
        print(f"错误: 无法读取预测图像 {pred_path}")
        return None
    pred_img = cv2.resize(pred_img, target_size)  # 调整到目标尺寸
    pred_img_normalized = pred_img / 255.0  # 预测图通常是概率图，直接归一化
    return pred_img_normalized


def calculate_iou(gt_binary_flat, pred_binary_flat):
    # gt_binary_flat 和 pred_binary_flat 应该是布尔型或0/1的整数型
    intersection = np.logical_and(gt_binary_flat, pred_binary_flat).sum()
    union = np.logical_or(gt_binary_flat, pred_binary_flat).sum()
    iou = intersection / (union + 1e-8)  # 添加一个小的epsilon以避免除以零
    return iou

def calculate_fpr(gt_binary_flat, pred_binary_flat):
    # gt_binary_flat 和 pred_binary_flat 应该是布尔型或0/1的整数型
    true_negative = np.logical_and(np.logical_not(gt_binary_flat), np.logical_not(pred_binary_flat)).sum()
    false_positive = np.logical_and(np.logical_not(gt_binary_flat), pred_binary_flat).sum()
    fpr = false_positive / (false_positive + true_negative + 1e-8)  # 添加一个小的epsilon以避免除以零
    return fpr

def calculate_metrics_on_files(gt_root, pred_folder): # 函数名稍作修改以反映其作用
    pred_files = sorted(os.listdir(pred_folder))
    num_images_total = len([f for f in pred_files if f.endswith('.png')]) # 统计png文件总数

    auc_scores, f1_scores, mcc_scores, iou_scores, fpr_scores, accuracy_scores = [], [], [], [], [], [] # 新增 accuracy_scores

    processed_images_count = 0

    for pred_file_name in tqdm(pred_files, desc="评估进度"):
        if not pred_file_name.endswith('.png'): # 或者其他你使用的图像格式
            continue

        pred_path = os.path.join(pred_folder, pred_file_name)

        # 假设GT文件名与预测文件名完全相同
        # 并且GT文件直接位于gt_root目录下
        gt_file_name = pred_file_name # 例如: "mask_image_00000_96.png"
        gt_path = os.path.join(gt_root, gt_file_name)

        if not os.path.exists(gt_path):
            print(f"警告: 找不到对应的GT图像: {gt_path}")
            print(f"  (预测文件为: {pred_path})")
            continue

        # 加载和归一化
        gt_img_normalized = preprocess_gt(gt_path)
        pred_img_normalized = preprocess_pred(pred_path)

        if gt_img_normalized is None or pred_img_normalized is None:
            print(f"跳过文件 {pred_file_name} 因为图像加载失败。")
            continue

        # 将GT图像转换为扁平化的0/1数组 (确保是整数类型用于sklearn度量)
        gt_flat_binary = gt_img_normalized.flatten().astype(np.uint8) # 使用uint8或int
        # 将预测图像转换为扁平化的概率值数组 (0-1范围)
        pred_flat_scores = pred_img_normalized.flatten()

        # 计算AUC (使用概率值)
        try:
            # 确保gt_flat_binary中至少有两个类别
            if len(np.unique(gt_flat_binary)) < 2:
                # 如果GT全黑或全白，AUC未定义或无意义，可以设为0.5或忽略
                auc = 0.5
                # print(f"警告: GT图像 {gt_file_name} 只有一个类别（全黑或全白），AUC设为0.5。")
                if np.all(gt_flat_binary == 0) and np.all(pred_flat_scores <= 0.5): # 全黑GT，预测也全黑
                     pass # 这种情况AUC可能是1，但roc_auc_score会报错
                elif np.all(gt_flat_binary == 1) and np.all(pred_flat_scores >= 0.5): # 全白GT，预测也全白
                     pass # 这种情况AUC可能是1
            else:
                auc = roc_auc_score(gt_flat_binary, pred_flat_scores)
            auc = max(auc, 1 - auc) # 处理标签反转的情况
        except ValueError as e:
            print(f"计算AUC时出错 {pred_file_name}: {e}. GT unique values: {np.unique(gt_flat_binary)}. AUC设为0.5")
            auc = 0.5 # 发生错误时给一个默认值
        auc_scores.append(auc)

        # 为了计算其他指标，需要将预测概率图二值化
        # 你可以调整这个阈值0.5
        pred_flat_binary = (pred_flat_scores > 0.5).astype(np.uint8)

        f1_scores.append(f1_score(gt_flat_binary, pred_flat_binary, zero_division=0))
        mcc_scores.append(matthews_corrcoef(gt_flat_binary, pred_flat_binary))
        iou_scores.append(calculate_iou(gt_flat_binary, pred_flat_binary))
        fpr_scores.append(calculate_fpr(gt_flat_binary, pred_flat_binary))
        accuracy_scores.append(accuracy_score(gt_flat_binary, pred_flat_binary)) # 计算并存储Accuracy

        processed_images_count +=1
        # print(f"{pred_file_name} → AUC: {auc:.4f}, F1: {f1_scores[-1]:.4f}, MCC: {mcc_scores[-1]:.4f}, IoU: {iou_scores[-1]:.4f}, FPR: {fpr_scores[-1]:.4f}, Acc: {accuracy_scores[-1]:.4f}")

    if processed_images_count == 0: # 如果没有成功处理任何图片
        print("错误：没有图片被成功处理和评估。请检查路径和文件格式。")
        return (0,0,0,0,0,0) # 返回对应数量的0

    print("\n--- 平均指标 ---")
    print(f"成功处理图片数量: {processed_images_count} / {num_images_total}")
    print(f"平均 Accuracy (准确率): {np.mean(accuracy_scores):.4f}")
    print(f"平均 AUC (Area Under Curve): {np.mean(auc_scores):.4f}")
    print(f"平均 F1-Score: {np.mean(f1_scores):.4f}")
    print(f"平均 MCC (Matthews Correlation Coefficient): {np.mean(mcc_scores):.4f}")
    print(f"平均 IoU / mIoU (Intersection over Union, for foreground class): {np.mean(iou_scores):.4f}")
    print(f"平均 FPR (False Positive Rate): {np.mean(fpr_scores):.4f}")

    return (
        np.mean(accuracy_scores),
        np.mean(auc_scores),
        np.mean(f1_scores),
        np.mean(mcc_scores),
        np.mean(iou_scores),
        np.mean(fpr_scores)
    )


if __name__ == "__main__":
    # 确保这里的路径是正确的
    # gt_root 应该指向包含 mask_image_XXXXX_YY.png 这种GT文件的文件夹
    gt_root = "/mnt/d/BaiduDownload/data/UltraEdit/test/mask_image"
    # pred_folder_path 指向包含模型输出的 mask_image_XXXXX_YY.png 文件的文件夹
    pred_folder_path = "results/polyp/UltraEdit"

    if not os.path.isdir(gt_root):
        print(f"错误: GT根目录不存在或不是一个目录: {gt_root}")
    elif not os.path.isdir(pred_folder_path):
        print(f"错误: 预测结果目录不存在或不是一个目录: {pred_folder_path}")
    else:
        (
            average_accuracy, # 新增
            average_auc,
            average_f1,
            average_mcc,
            average_iou, # 这个在二分类中可以视为前景的mIoU
            average_fpr
        ) = calculate_metrics_on_files(gt_root, pred_folder_path) # 函数名更新