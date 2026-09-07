"""
FHR-SimCLR: 基于对比学习的胎心监护信号预训练系统

使用SimCLR框架对FHR数据进行自监督预训练，提取通用特征表示。

核心功能：
1. 数据源加载 (NPZ格式)
2. 安全归一化与插值
3. 时序数据增强 (缩放 + 噪声)
4. SimCLR对比学习预训练

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os
import warnings
from dataclasses import dataclass
import argparse

warnings.filterwarnings('ignore')


# ===================================================================
# 1. 配置模块
# ===================================================================

@dataclass
class Config:
    """训练配置"""
    # 设备
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    # 训练参数
    batch_size: int = 128
    epochs: int = 200
    learning_rate: float = 1e-4
    window_size: int = 1200
    
    # SimCLR参数
    latent_dim: int = 128
    projection_dim: int = 64
    temperature: float = 0.5
    
    # 数据增强参数
    scale_range: float = 0.05
    noise_std: float = 0.01
    clamp_range: float = 5.0
    
    # 保存参数
    save_interval: int = 10
    output_path: str = "./pretrain_net1d.pth"
    
    # 数据路径 (用户修改这里)
    data_path: str = "./pretrain_contrastive_windows.npz"


# ===================================================================
# 2. 数据处理模块
# ===================================================================

class FHRDataLoader:
    """FHR数据加载与预处理工具"""
    
    @staticmethod
    def safe_normalize(signal: np.ndarray) -> np.ndarray:
        """
        安全归一化，处理NaN和全零序列
        
        Args:
            signal: 输入信号
        
        Returns:
            归一化后的信号 [0, 1]
        """
        signal = np.asarray(signal, dtype=np.float32)
        valid = signal[~np.isnan(signal)]
        
        if len(valid) < 1:
            return np.zeros_like(signal)
        
        min_val, max_val = np.min(valid), np.max(valid)
        diff = max_val - min_val
        
        if diff < 1e-6:
            return np.full_like(signal, 0.5)
        
        norm = (signal - min_val) / diff
        norm[np.isnan(norm)] = 0.0
        return norm
    
    @staticmethod
    def to_fixed_length(signal: np.ndarray, target_len: int) -> torch.Tensor:
        """将信号插值到固定长度"""
        signal = torch.tensor(signal, dtype=torch.float32)
        signal = signal.view(1, 1, -1)
        return F.interpolate(signal, size=target_len).squeeze()
    
    @staticmethod
    def augment(signal: torch.Tensor, config: Config) -> torch.Tensor:
        """时序数据增强"""
        # 缩放增强
        scale = 1 + config.scale_range * torch.randn_like(signal)
        signal = signal * scale
        
        # 噪声增强
        noise = config.noise_std * torch.randn_like(signal)
        signal = signal + noise
        
        # 数值裁剪
        return torch.clamp(signal, -config.clamp_range, config.clamp_range)


class FHRPretrainDataset(Dataset):
    """
    FHR预训练数据集
    
    从NPZ文件中加载FHR信号，支持两种格式:
    1. 键名为 'fhr_windows' 或 'fhr_filled' 或 'fhr'
    2. 自动检测第一个可用数组
    """
    
    def __init__(self, npz_path: str, config: Config):
        """
        Args:
            npz_path: NPZ文件路径
            config: 配置对象
        """
        self.config = config
        self.signals = self._load_data(npz_path)
        print(f"✅ 加载数据集: {os.path.basename(npz_path)} | {len(self.signals)} 样本")
    
    def _load_data(self, path: str) -> np.ndarray:
        """加载NPZ数据"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在: {path}")
        
        data = np.load(path, allow_pickle=True)
        
        # 自动检测数据键
        for key in ['fhr_windows', 'fhr_filled', 'fhr', 'signals']:
            if key in data:
                return data[key]
        
        # 如果都没有，取第一个数组
        return data[data.files[0]]
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        
        # 安全归一化
        signal = FHRDataLoader.safe_normalize(signal)
        
        # 固定长度
        signal = FHRDataLoader.to_fixed_length(signal, self.config.window_size)
        
        # 双增强 (SimCLR需要两个视图)
        v1 = FHRDataLoader.augment(signal, self.config)
        v2 = FHRDataLoader.augment(signal, self.config)
        
        return v1, v2


# ===================================================================
# 3. 模型模块
# ===================================================================

class SimCLR(nn.Module):
    """
    SimCLR对比学习模型
    
    包含编码器(Net1D)和投影头
    """
    
    def __init__(self, encoder: nn.Module, latent_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.encoder = encoder
        
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入信号 (batch, signal_length)
        
        Returns:
            归一化的投影特征 (batch, proj_dim)
        """
        x = x.unsqueeze(1)  # (batch, 1, signal_length)
        features = self.encoder(x)
        projected = self.projection(features)
        return F.normalize(projected, dim=-1)


def get_encoder(config: Config) -> nn.Module:
    """
    获取编码器 (Net1D)
    
    注意: 需要从 net1d 导入 Net1D 类
    """
    try:
        from net1d import Net1D
        
        return Net1D(
            in_channels=1,
            base_filters=64,
            ratio=1,
            filter_list=[32, 64, 128],
            m_blocks_list=[1, 1, 1],
            kernel_size=16,
            stride=2,
            groups_width=16,
            n_classes=config.latent_dim,
            use_bn=True,
            use_do=True,
            verbose=False
        )
    except ImportError:
        print("⚠️ 无法导入 Net1D，请确保 net1d.py 在路径中")
        raise


# ===================================================================
# 4. 损失函数模块
# ===================================================================

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    NT-Xent损失 (Normalized Temperature-scaled Cross Entropy)
    
    Args:
        z1: 第一视图投影 (batch, dim)
        z2: 第二视图投影 (batch, dim)
        temperature: 温度参数
    
    Returns:
        损失值
    """
    batch_size = z1.size(0)
    device = z1.device
    
    # 拼接两个视图
    z = torch.cat([z1, z2], dim=0)
    
    # 计算相似度矩阵
    sim = torch.matmul(z, z.T) / temperature
    sim = torch.clamp(sim, min=-50, max=50)  # 防止数值溢出
    sim.fill_diagonal_(-1e4)
    
    # 正样本对
    pos = torch.cat([
        torch.diag(sim, batch_size),
        torch.diag(sim, -batch_size)
    ], dim=0)
    
    # 负样本
    neg_mask = ~torch.eye(2 * batch_size, device=device, dtype=bool)
    neg = sim[neg_mask].view(2 * batch_size, -1)
    
    # 计算损失
    logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=device)
    
    return F.cross_entropy(logits, labels)


# ===================================================================
# 5. 训练模块
# ===================================================================

class Trainer:
    """SimCLR训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # 创建模型
        encoder = get_encoder(config).to(self.device)
        self.model = SimCLR(
            encoder,
            latent_dim=config.latent_dim,
            proj_dim=config.projection_dim
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate, 
            betas=(0.5, 0.999)
        )
        self.scaler = torch.cuda.amp.GradScaler()
        
        # 创建输出目录
        os.makedirs(os.path.dirname(config.output_path) or '.', exist_ok=True)
    
    def train(self):
        """执行训练"""
        # 构建数据集
        dataset = FHRPretrainDataset(self.config.data_path, self.config)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"\n🚀 开始SimCLR预训练")
        print(f"📊 设备: {self.device}")
        print(f"📊 总样本数: {len(dataset):,}")
        print(f"📈 批次大小: {self.config.batch_size}")
        print(f"🔄 Epochs: {self.config.epochs}\n")
        
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0.0
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            
            for v1, v2 in pbar:
                v1, v2 = v1.to(self.device), v2.to(self.device)
                
                self.optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    z1 = self.model(v1)
                    z2 = self.model(v2)
                    loss = nt_xent_loss(z1, z2, self.config.temperature)
                
                # 检测NaN
                if torch.isnan(loss):
                    print(f"\n⚠️ 检测到NaN损失 (Epoch {epoch+1})")
                    print(f"  v1 NaN: {torch.isnan(v1).any().item()}")
                    print(f"  v2 NaN: {torch.isnan(v2).any().item()}")
                    print(f"  v1范围: [{v1.min():.3f}, {v1.max():.3f}]")
                    raise RuntimeError("训练中断: 损失为NaN")
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_loss = total_loss / len(loader)
            print(f"✅ Epoch {epoch+1:2d} | 平均损失: {avg_loss:.6f}")
            
            # 定期保存
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(epoch)
        
        # 保存最终模型
        self._save_checkpoint("final")
        print(f"\n🎉 预训练完成！模型已保存至: {self.config.output_path}")
    
    def _save_checkpoint(self, epoch):
        """保存检查点"""
        path = self.config.output_path
        if epoch != "final":
            base, ext = os.path.splitext(path)
            path = f"{base}_epoch{epoch}{ext}"
        
        # 保存编码器权重
        torch.save(self.model.encoder.state_dict(), path)
        print(f"💾 检查点已保存: {path}")


# ===================================================================
# 6. 主入口
# ===================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FHR-SimCLR 预训练')
    
    parser.add_argument('--data', '-d', type=str, default=None,
                        help='数据NPZ文件路径')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--window', type=int, default=1200,
                        help='信号窗口长度')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='模型输出路径')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='计算设备')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 创建配置
    config = Config(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        window_size=args.window,
        device=args.device,
        output_path=args.output if args.output else Config.output_path
    )
    
    # 如果指定了数据路径，覆盖配置
    if args.data:
        config.data_path = args.data
    
    print("=" * 60)
    print("FHR-SimCLR 预训练系统")
    print("=" * 60)
    print(f"数据路径: {config.data_path}")
    print(f"输出路径: {config.output_path}")
    print("=" * 60)
    
    # 创建训练器并训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()