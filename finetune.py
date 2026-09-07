"""
FHR-FineTune: 基于预训练编码器的胎心监护信号微调系统

使用预训练的Net1D编码器对FHR信号进行特征提取，结合临床特征进行下游任务微调。

核心功能：
1. 预训练编码器加载与冻结
2. 临床特征融合 (可选)
3. 滑窗数据增强 (训练集)
4. 类别不平衡处理
5. 完整评估指标 (AUC, AUPRC, F1, Sensitivity, Specificity)
6. Bootstrap置信区间估计

"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, 
    f1_score, precision_score, recall_score, roc_curve
)
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import argparse

warnings.filterwarnings('ignore')


# ===================================================================
# 1. 配置模块
# ===================================================================

@dataclass
class Config:
    """训练配置"""
    # 设备
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 随机种子
    seed: int = 33
    
    # 数据参数
    window_size: int = 1200
    stride: int = 300
    
    # 训练参数
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # 调度器参数
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    
    # 早停参数
    early_stop_patience: int = 8
    
    # 特征开关
    use_clinical: bool = True
    use_hand_features: bool = False
    
    # 数据增强
    noise_std: float = 0.02
    mask_prob: float = 0.15
    
    # 评估参数
    n_bootstraps: int = 1000
    
    # 路径配置 (用户修改这里)
    filled_data_path: str = "./fhr_final_filled.npz"
    pretrained_model_path: str = "./pretrain_net1d_multidata_20260727.pth"
    output_dir: str = "./pth/"
    
    # 数据集CSV路径
    train_csv: str = "./data/临床数据_TRAIN_train_with_gap.csv"
    val_csv: str = "./data/临床数据_TRAIN_val_with_gap.csv"
    test_csv: str = "./data/临床数据_TRAIN_test_with_gap.csv"
    
    # 任务配置
    selected_task: str = "preterm"
    binary_delivery: bool = True  # True: 二分类(1+2 vs 3), False: 三分类


# ===================================================================
# 2. 任务配置
# ===================================================================

# 临床特征列表
CLINICAL_FEATURES = ['年龄', '孕龄', 'BMI-孕前', 'CTGage-gap']

# CSV列名映射
COLUMN_MAP = {
    '年龄': '年龄',
    '孕龄': '孕龄',
    'BMI-孕前': 'BMI-孕前',
    'CTGage-gap': 'CTGage-gap',
}


# ===================================================================
# 3. 数据处理模块
# ===================================================================

class FHRDataset(Dataset):
    """
    FHR数据集
    
    支持滑窗增强 (训练集) 和固定窗口 (验证/测试集)
    支持临床特征融合
    """
    
    def __init__(self, config: Config, filled_npz: str, csv_path: str, 
                 task_cfg: Dict, is_train: bool = True):
        """
        Args:
            config: 配置对象
            filled_npz: 填补后的FHR数据路径
            csv_path: CSV标签文件路径
            task_cfg: 任务配置
            is_train: 是否为训练集
        """
        self.config = config
        self.task_cfg = task_cfg
        self.is_train = is_train
        self.window = config.window_size
        self.stride = config.stride
        
        # 加载数据
        data = np.load(filled_npz, allow_pickle=True)
        self.fhr = data['fhr_filled']
        self.filenames = data['filenames']
        
        # 加载标签
        df = pd.read_csv(csv_path)
        df['胎心数据文件名'] = df['胎心数据文件名'].astype(str).str.strip()
        
        # 构建文件信息映射
        self.file_info = self._build_file_info(df)
        
        # 构建窗口数据
        self.windows, self.labels, self.clinical_features = self._build_windows()
        
        # 标准化临床特征
        self._normalize_clinical_features()
        
        print(f"✅ [{os.path.basename(csv_path)}] 样本数: {len(self.windows)}")
    
    def _build_file_info(self, df: pd.DataFrame) -> Dict:
        """构建文件名到标签和临床特征的映射"""
        file_info = {}
        
        for _, row in df.iterrows():
            fname = str(row['胎心数据文件名']).strip()
            
            # 提取标签
            label = row[self.task_cfg['label_col']]
            
            # 标签转换
            label = self._transform_label(label)
            
            # 提取临床特征
            clinical_vals = []
            for std_col in CLINICAL_FEATURES:
                mapped = COLUMN_MAP.get(std_col)
                if mapped and mapped in row:
                    val = row[mapped]
                    if val == -1 or pd.isna(val):
                        val = 0.0  # 默认值
                else:
                    val = 0.0
                clinical_vals.append(float(val))
            
            file_info[fname] = (label, clinical_vals)
        
        return file_info
    
    def _transform_label(self, label: int) -> int:
        """转换标签格式"""
        # 处理无效标签
        if label == -1 or label in self.task_cfg['invalid']:
            return -1
        
        # 分娩方式特殊处理
        if self.task_cfg['label_col'] == '分娩方式':
            if self.config.binary_delivery:
                # 二分类: 1+2 -> 0, 3 -> 1
                return 0 if label in [1, 2] else 1
            else:
                return label - 1
        
        return label
    
    def _build_windows(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构建滑动窗口"""
        windows, labels, clinicals = [], [], []
        
        kept = 0
        for i, fname in enumerate(self.filenames):
            fname = str(fname).strip()
            if fname not in self.file_info:
                continue
            
            label, clinical_vals = self.file_info[fname]
            if label == -1:
                continue
            
            # 处理信号
            sig = self.fhr[i]
            sig = np.nan_to_num(sig, nan=0.0)
            sig_mean = np.nanmean(sig)
            sig_std = np.nanstd(sig)
            sig = (sig - sig_mean) / (sig_std if sig_std > 1e-8 else 1.0)
            
            # 提取窗口
            if self.is_train:
                wins = self._split_windows(sig)
            else:
                wins = self._get_fixed_window(sig)
            
            if not wins:
                continue
            
            windows.extend(wins)
            labels.extend([label] * len(wins))
            clinicals.extend([clinical_vals] * len(wins))
            kept += 1
        
        return np.array(windows), np.array(labels), np.array(clinicals)
    
    def _split_windows(self, signal: np.ndarray) -> List[np.ndarray]:
        """滑动窗口分割 (训练集)"""
        windows = []
        n = len(signal)
        for i in range(0, n - self.window + 1, self.stride):
            windows.append(signal[i:i + self.window])
        return windows
    
    def _get_fixed_window(self, signal: np.ndarray) -> List[np.ndarray]:
        """固定窗口提取 (验证/测试集)"""
        if len(signal) >= self.window:
            return [signal[-self.window:]]
        return []
    
    def _normalize_clinical_features(self):
        """标准化临床特征"""
        if not self.config.use_clinical or self.clinical_features.shape[1] == 0:
            self.clinical_features = np.zeros((len(self.windows), 0))
            return
        
        mean = np.mean(self.clinical_features, axis=0)
        std = np.std(self.clinical_features, axis=0)
        std[std < 1e-6] = 1.0
        self.clinical_features = (self.clinical_features - mean) / std
        self.clinical_features = np.nan_to_num(self.clinical_features, nan=0.0)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        x = self.windows[idx].astype(np.float32)[None, :]
        
        # 训练时数据增强
        if self.is_train:
            # 添加噪声
            noise = np.random.normal(0, self.config.noise_std, x.shape).astype(np.float32)
            x = x + noise
            # 随机掩码
            mask = (np.random.rand(*x.shape) > self.config.mask_prob).astype(np.float32)
            x = x * mask
        
        x = torch.tensor(x, dtype=torch.float32)
        clin = torch.tensor(self.clinical_features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32 if self.task_cfg['type'] == 'binary' else torch.long)
        
        return x, clin, y


# ===================================================================
# 4. 模型模块
# ===================================================================

def get_encoder(pretrained_path: str, device: torch.device) -> nn.Module:
    """
    加载预训练编码器
    
    注意: 需要从 net1d 导入 Net1D 类
    """
    try:
        from net1d import Net1D
    except ImportError:
        print("⚠️ 无法导入 Net1D，请确保 net1d.py 在路径中")
        raise
    
    backbone = Net1D(
        in_channels=1,
        base_filters=64,
        ratio=1,
        filter_list=[32, 64, 128],
        m_blocks_list=[1, 1, 1],
        kernel_size=16,
        stride=2,
        groups_width=16,
        n_classes=128,
        use_bn=True,
        use_do=True,
        verbose=False
    )
    
    # 加载预训练权重
    state_dict = torch.load(pretrained_path, map_location=device)
    backbone.load_state_dict(state_dict)
    
    # 冻结所有参数
    for param in backbone.parameters():
        param.requires_grad = False
    
    return backbone


class FHRClassifier(nn.Module):
    """FHR分类器 (编码器 + 分类头)"""
    
    def __init__(self, backbone: nn.Module, n_classes: int, extra_features: int = 0):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = backbone.dense.in_features
        self.backbone.dense = nn.Identity()
        self.fc = nn.Linear(self.feat_dim + extra_features, n_classes)
    
    def forward(self, x: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if clinical.shape[1] > 0:
            features = torch.cat([features, clinical], dim=1)
        return self.fc(features)


# ===================================================================
# 5. 评估模块
# ===================================================================

def evaluate_metrics(y_true: np.ndarray, y_score: np.ndarray, task_type: str = 'binary') -> Dict:
    """
    计算所有评估指标
    
    Args:
        y_true: 真实标签
        y_score: 预测分数
        task_type: 'binary' 或 'multiclass'
    
    Returns:
        指标字典
    """
    if task_type == 'binary':
        # 寻找最佳阈值 (Sensitivity = Specificity)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        diff = np.abs(tpr - (1 - fpr))
        best_idx = np.argmin(diff)
        best_thresh = thresholds[best_idx]
        y_pred = (y_score >= best_thresh).astype(int)
        
        return {
            'auc': roc_auc_score(y_true, y_score),
            'auprc': average_precision_score(y_true, y_score),
            'acc': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'sensitivity': recall_score(y_true, y_pred, zero_division=0),
            'specificity': (1 - fpr)[best_idx],
            'threshold': float(best_thresh)
        }
    else:
        # 多分类
        y_pred = np.argmax(y_score, axis=1)
        return {
            'auc': roc_auc_score(y_true, y_score, multi_class='ovr', average='macro'),
            'auprc': average_precision_score(y_true, y_score),
            'acc': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'sensitivity': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'specificity': np.nan,
            'threshold': np.nan
        }


def bootstrap_metrics(y_true: np.ndarray, y_score: np.ndarray, 
                      task_type: str = 'binary', n_bootstraps: int = 1000) -> Dict:
    """
    Bootstrap估计指标均值和标准差
    """
    np.random.seed(42)
    rng = np.random.RandomState(42)
    
    results = {k: [] for k in ['auc', 'auprc', 'acc', 'f1', 'precision', 'sensitivity', 'specificity']}
    
    for _ in range(n_bootstraps):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        metrics = evaluate_metrics(y_true[indices], y_score[indices], task_type)
        for k in results:
            results[k].append(metrics[k])
    
    return {
        f'{k}_mean': float(np.mean(v)),
        f'{k}_std': float(np.std(v))
        for k, v in results.items()
    }


# ===================================================================
# 6. 训练模块
# ===================================================================

class Trainer:
    """训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        # 获取任务配置
        self.task_cfg = TASKS[config.selected_task]
        self.n_classes = 1 if self.task_cfg['type'] == 'binary' else 3
        
        # 创建输出目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(
            config.output_dir, 
            f"finetune_{config.selected_task}_{self.timestamp}"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 构建数据
        self._build_data()
        
        # 构建模型
        self._build_model()
        
        # 构建优化器
        self._build_optimizer()
        
        print(f"\n🚀 任务: {config.selected_task}")
        print(f"📊 类型: {self.task_cfg['type']}")
        print(f"📁 保存目录: {self.save_dir}")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def _build_data(self):
        """构建数据集"""
        self.train_ds = FHRDataset(
            self.config, self.config.filled_data_path, 
            self.config.train_csv, self.task_cfg, is_train=True
        )
        self.val_ds = FHRDataset(
            self.config, self.config.filled_data_path, 
            self.config.val_csv, self.task_cfg, is_train=False
        )
        self.test_ds = FHRDataset(
            self.config, self.config.filled_data_path, 
            self.config.test_csv, self.task_cfg, is_train=False
        )
        
        self.train_loader = DataLoader(
            self.train_ds, batch_size=self.config.batch_size, 
            shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=self.config.batch_size, 
            shuffle=False, num_workers=4
        )
        self.test_loader = DataLoader(
            self.test_ds, batch_size=self.config.batch_size, 
            shuffle=False, num_workers=4
        )
        
        # 处理类别不平衡
        self._setup_criterion()
    
    def _setup_criterion(self):
        """设置损失函数 (处理类别不平衡)"""
        labels = self.train_ds.labels
        counts = Counter(labels)
        print(f"训练集标签分布: {dict(counts)}")
        
        if self.task_cfg['type'] == 'binary':
            pos_count = counts.get(1, 0)
            neg_count = counts.get(0, 0)
            if pos_count > 0 and neg_count / pos_count > 2:
                pos_weight = torch.tensor(neg_count / pos_count, dtype=torch.float32).to(self.device)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                print(f"⚠️ 使用加权损失 (正类权重: {pos_weight.item():.2f})")
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            total = len(labels)
            weights = [total / counts[i] for i in range(3)]
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    
    def _build_model(self):
        """构建模型"""
        # 加载编码器
        backbone = get_encoder(self.config.pretrained_model_path, self.device)
        
        # 计算额外特征维度
        extra_features = len(CLINICAL_FEATURES) if self.config.use_clinical else 0
        
        # 创建分类器
        self.model = FHRClassifier(backbone, self.n_classes, extra_features).to(self.device)
    
    def _build_optimizer(self):
        """构建优化器"""
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 'min', 
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            verbose=True
        )
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for x, clin, y in tqdm(self.train_loader, desc="训练"):
            x, clin, y = x.to(self.device), clin.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            out = self.model(x, clin).squeeze(-1)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * x.size(0)
        
        return total_loss / len(self.train_ds)
    
    def evaluate(self, loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        """评估"""
        self.model.eval()
        total_loss = 0
        y_true, y_score = [], []
        
        with torch.no_grad():
            for x, clin, y in loader:
                x, clin, y = x.to(self.device), clin.to(self.device), y.to(self.device)
                out = self.model(x, clin).squeeze(-1)
                total_loss += self.criterion(out, y).item() * x.size(0)
                
                y_true.append(y.cpu().numpy())
                if self.task_cfg['type'] == 'binary':
                    y_score.append(torch.sigmoid(out).cpu().numpy())
                else:
                    y_score.append(torch.softmax(out, dim=1).cpu().numpy())
        
        return total_loss / len(loader.dataset), np.concatenate(y_true), np.concatenate(y_score)
    
    def train(self):
        """主训练循环"""
        print(f"\n🔄 开始训练 (Epochs: {self.config.epochs})")
        
        best_val_auc = 0
        best_val_loss = float('inf')
        patience = 0
        
        for epoch in range(self.config.epochs):
            print(f"\n=== Epoch {epoch+1}/{self.config.epochs} ===")
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss, y_true, y_score = self.evaluate(self.val_loader)
            val_auc = roc_auc_score(y_true, y_score) if self.task_cfg['type'] == 'binary' else 0
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | 验证AUC: {val_auc:.4f}")
            
            # 保存最佳模型
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best.pth"))
                print("⭐ 最佳模型已更新!")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= self.config.early_stop_patience:
                    print(f"早停触发 (耐心: {self.config.early_stop_patience})")
                    break
            
            # 保存检查点
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_auc': best_val_auc,
            }, os.path.join(self.save_dir, "checkpoint.pth"))
        
        # 最终评估
        self._final_evaluation()
    
    def _final_evaluation(self):
        """最终评估"""
        # 加载最佳模型
        best_path = os.path.join(self.save_dir, "best.pth")
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path))
        
        print("\n" + "=" * 60)
        
        # 验证集评估
        _, y_true_val, y_score_val = self.evaluate(self.val_loader)
        val_metrics = bootstrap_metrics(
            y_true_val, y_score_val, 
            self.task_cfg['type'], 
            self.config.n_bootstraps
        )
        print("🔍 验证集结果")
        print("-" * 40)
        self._print_metrics(val_metrics)
        
        # 测试集评估
        _, y_true_test, y_score_test = self.evaluate(self.test_loader)
        test_metrics = bootstrap_metrics(
            y_true_test, y_score_test,
            self.task_cfg['type'],
            self.config.n_bootstraps
        )
        print("🔍 测试集结果")
        print("-" * 40)
        self._print_metrics(test_metrics)
        
        print("=" * 60)
        
        # 保存结果
        self._save_results(val_metrics, test_metrics)
    
    def _print_metrics(self, metrics: Dict):
        """打印指标"""
        for key in ['auc', 'auprc', 'acc', 'f1', 'precision', 'sensitivity', 'specificity']:
            mean_key = f'{key}_mean'
            std_key = f'{key}_std'
            if mean_key in metrics and std_key in metrics:
                print(f"{key.upper():12s}: {metrics[mean_key]:.3f} ± {metrics[std_key]:.3f}")
    
    def _save_results(self, val_metrics: Dict, test_metrics: Dict):
        """保存结果"""
        results = {
            'task': self.config.selected_task,
            'task_type': self.task_cfg['type'],
            'use_clinical': self.config.use_clinical,
            'clinical_features': CLINICAL_FEATURES if self.config.use_clinical else [],
            'val': val_metrics,
            'test': test_metrics,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'epochs': self.config.epochs,
                'window_size': self.config.window_size,
            }
        }
        
        with open(os.path.join(self.save_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"✅ 结果已保存: {os.path.join(self.save_dir, 'results.json')}")


# ===================================================================
# 7. 主入口
# ===================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FHR微调训练')
    
    parser.add_argument('--task', '-t', type=str, default='oligohydramnios',
                        choices=list(TASKS.keys()),
                        help='选择任务')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--no_clinical', action='store_true',
                        help='不使用临床特征')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出目录')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建配置
    config = Config(
        selected_task=args.task,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        use_clinical=not args.no_clinical
    )
    
    if args.output:
        config.output_dir = args.output
    
    # 创建训练器并训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()