"""
FHR-GAN Filler: 基于条件生成对抗网络的胎心信号缺失填补工具

这是一个利用预训练GAN模型对4Hz胎心监护(FHR)信号进行质量增强和缺失填补的工具。
主要功能:
1. 基于质量掩码的严格信号清洗
2. 小缺口线性插值填补 (<60s)
3. 大缺口GAN生成式填补 (>=60s)
4. 滑动窗口加权融合策略
5. 填补前后可视化对比

"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
import warnings
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import argparse
import json

warnings.filterwarnings('ignore')


# ===================================================================
# 1. 配置模块
# ===================================================================

class Config:
    """全局配置类"""
    # 信号参数
    FS = 4  # 采样频率 (Hz)
    SIGNAL_LENGTH = 1800  # 窗口长度 (对应 7.5分钟 @ 4Hz)
    OVERLAP_RATIO = 0.25  # 滑动窗口重叠比例
    
    # 缺口分类阈值 (秒)
    SMALL_GAP_THRESHOLD = 15  # <15s为小缺口，线性插值填补
    LARGE_GAP_THRESHOLD = 60  # >=60s为大缺口，GAN填补
    
    # GAN参数
    NOISE_DIM = 50
    FEATURE_DIM = 3
    MAX_RETRY = 5
    
    # 质量范围
    FHR_MIN = 50
    FHR_MAX = 210
    FHR_NORMAL_MIN = 80
    FHR_NORMAL_MAX = 160
    
    # 训练参数
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # 随机种子
    SEED = 42
    
    # 输出配置
    OUTPUT_DIR = "./outputs"
    PLOT_DPI = 150
    PLOT_SAMPLES = 10  # 每种类型绘制的样本数


class PathConfig:
    """路径配置类 - 用户需要根据实际情况修改"""
    # 输入: 4Hz FSQI质量筛查结果
    INPUT_NPZ = "./fhr_fsqi_4hz_quality_result.npz"
    
    # 模型: 预训练的GAN权重
    GAN_CHECKPOINT = "./checkpoint_epoch_99.pth"
    
    # 输出: 填补完成的数据
    OUTPUT_PATH = "./4hz_filled.npz"
    
    # 可视化输出目录
    PLOT_DIR = "./fill_compare_plots"
    
    @classmethod
    def create_output_dirs(cls):
        """创建所有输出目录"""
        os.makedirs(cls.OUTPUT_PATH.replace(os.path.basename(cls.OUTPUT_PATH), ''), exist_ok=True)
        os.makedirs(cls.PLOT_DIR, exist_ok=True)


# ===================================================================
# 2. 数据增强与模型模块
# ===================================================================

class FHRNormalizer:
    """FHR信号归一化工具"""
    
    @staticmethod
    def normalize(signal: np.ndarray, fhr_min: float = 50, fhr_max: float = 210) -> np.ndarray:
        """
        将FHR信号归一化到 [-1, 1] 范围
        
        Args:
            signal: 原始FHR信号 (bpm)
            fhr_min: 最小FHR值
            fhr_max: 最大FHR值
        
        Returns:
            归一化后的信号
        """
        return np.clip((signal - fhr_min) / (fhr_max - fhr_min) * 2 - 1, -1, 1)
    
    @staticmethod
    def denormalize(normalized: np.ndarray, fhr_min: float = 50, fhr_max: float = 210) -> np.ndarray:
        """
        将归一化信号还原为FHR值
        
        Args:
            normalized: 归一化信号 [-1, 1]
            fhr_min: 最小FHR值
            fhr_max: 最大FHR值
        
        Returns:
            还原后的FHR信号 (bpm)
        """
        return (normalized + 1) / 2 * (fhr_max - fhr_min) + fhr_min
    
    @staticmethod
    def extract_features(signal: np.ndarray) -> np.ndarray:
        """
        提取信号的上下文特征
        
        Args:
            signal: FHR信号
        
        Returns:
            特征向量 [均值归一化, 标准差归一化, 变异性归一化]
        """
        valid = signal[(signal > 40) & (signal < 220)]
        if len(valid) < 5:
            return np.array([0.0, 0.1, 0.1], dtype=np.float32)
        
        # 均值 (归一化到0-1)
        mean_val = np.mean(valid) / 200.0
        mean_val = np.clip(mean_val, 0, 1)
        
        # 标准差 (归一化)
        std_val = np.clip(np.std(valid), 1, 20) / 20.0
        
        # 变异性 (相邻差分绝对值的均值)
        var_val = np.mean(np.abs(np.diff(valid))) if len(valid) > 1 else 0.1
        var_val = np.clip(var_val, 0, 10) / 10.0
        
        return np.array([mean_val, std_val, var_val], dtype=np.float32)


class StabilizedConditionalFHRGenerator(nn.Module):
    """
    条件FHR生成器 (与预训练权重兼容)
    
    输入: 噪声(50) + 标签(1) + 特征(3)
    输出: 归一化FHR信号 (1800)
    """
    
    def __init__(self, noise_dim: int = 50, label_dim: int = 1, 
                 signal_length: int = 1800, feature_dim: int = 3):
        super(StabilizedConditionalFHRGenerator, self).__init__()
        
        self.signal_length = signal_length
        self.noise_dim = noise_dim
        self.label_dim = label_dim
        self.feature_dim = feature_dim
        
        input_size = noise_dim + label_dim + feature_dim
        
        self.main = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, True),
            
            nn.Linear(2048, signal_length),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, noise: torch.Tensor, labels: Optional[torch.Tensor] = None,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            noise: 噪声向量 (batch, noise_dim)
            labels: 标签 (batch, label_dim)
            features: 特征 (batch, feature_dim)
        
        Returns:
            生成的信号 (batch, signal_length)
        """
        x = torch.cat([noise, labels, features], dim=1)
        return self.main(x)


class GANFiller:
    """GAN填充器 - 使用预训练模型填充大缺口"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化GAN填充器
        
        Args:
            checkpoint_path: 预训练模型路径
            device: 计算设备
        """
        self.device = device
        self.normalizer = FHRNormalizer()
        
        # 加载模型
        self.generator = StabilizedConditionalFHRGenerator(
            noise_dim=Config.NOISE_DIM,
            label_dim=1,
            signal_length=Config.SIGNAL_LENGTH,
            feature_dim=Config.FEATURE_DIM
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()
        
        print(f"✅ GAN模型加载成功: {checkpoint_path}")
    
    def generate_window(self, signal_segment: np.ndarray) -> np.ndarray:
        """
        为单个窗口生成填充信号
        
        Args:
            signal_segment: 信号段 (长度=窗口长度)
        
        Returns:
            生成的重建信号 (归一化后)
        """
        # 提取特征
        features = self.normalizer.extract_features(signal_segment)
        
        for attempt in range(Config.MAX_RETRY):
            with torch.no_grad():
                noise = torch.randn(1, Config.NOISE_DIM).to(self.device)
                label = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
                feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                gen_norm = self.generator(noise, label, feature_tensor).cpu().numpy()[0]
                gen = self.normalizer.denormalize(gen_norm)
            
            # 质量检查
            if (Config.FHR_NORMAL_MIN < np.mean(gen) < Config.FHR_NORMAL_MAX and 
                np.std(gen) > 2):
                return gen
        
        return gen  # 返回最后一次生成的结果


# ===================================================================
# 3. 信号处理模块
# ===================================================================

class FHRQualityCleaner:
    """FHR信号质量清洗工具"""
    
    @staticmethod
    def clean_signal(signal: np.ndarray, mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        严格清洗信号，去除前后端大片无效区域
        
        Args:
            signal: 原始FHR信号
            mask: 质量掩码 (True=有效, False=无效)
        
        Returns:
            (清洗后的信号, 清洗后的掩码) 或 (None, None)
        """
        signal = np.array(signal, dtype=np.float32).copy()
        mask = np.array(mask, dtype=bool).copy()
        L = len(signal)
        
        # 检查是否有NaN
        if not np.isnan(signal).any():
            return signal, mask
        
        # 检查缺失位置
        nan_pos = np.where(np.isnan(signal))[0]
        is_front = (nan_pos < 100).all()
        is_back = (nan_pos > L - 100).all()
        is_middle = not is_front and not is_back
        
        if is_front or is_back:
            # 截断前后端的大片缺失
            valid_idx = np.where(~np.isnan(signal))[0]
            if len(valid_idx) < 200:
                return None, None
            signal = signal[valid_idx[0]:valid_idx[-1] + 1]
            mask = mask[valid_idx[0]:valid_idx[-1] + 1]
        else:
            # 中间缺失用0填充
            signal = np.nan_to_num(signal, nan=0.0)
        
        # 再次检查NaN
        if np.isnan(signal).any():
            return None, None
        
        # 长度检查
        if len(signal) < 500:
            return None, None
        
        return signal, mask
    
    @staticmethod
    def get_gap_lengths(mask: np.ndarray) -> List[int]:
        """
        获取所有无效段落的长度
        
        Args:
            mask: 质量掩码 (True=有效, False=无效)
        
        Returns:
            无效段落的长度列表
        """
        bad = ~mask
        if not bad.any():
            return []
        
        jumps = np.diff(bad.astype(int))
        starts = np.where(jumps == 1)[0] + 1
        ends = np.where(jumps == -1)[0] + 1
        
        if bad[0]:
            starts = np.r_[0, starts]
        if bad[-1]:
            ends = np.r_[ends, len(bad)]
        
        return [e - s for s, e in zip(starts, ends)]


class FHRFiller:
    """FHR信号填补器"""
    
    def __init__(self, gan_filler: GANFiller):
        self.gan_filler = gan_filler
        self.normalizer = FHRNormalizer()
        self.fs = Config.FS
        self.window_len = Config.SIGNAL_LENGTH
        self.overlap_ratio = Config.OVERLAP_RATIO
        self.small_threshold = Config.SMALL_GAP_THRESHOLD * Config.FS  # 转为采样点数
        self.large_threshold = Config.LARGE_GAP_THRESHOLD * Config.FS  # 转为采样点数
    
    def fill_small_gap(self, signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        使用线性插值填补小缺口
        
        Args:
            signal: FHR信号
            mask: 质量掩码 (True=有效)
        
        Returns:
            填补后的信号
        """
        signal = np.asarray(signal, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        valid = mask & (signal > 10) & (signal < 210)
        indices = np.arange(len(signal))
        
        # 插值
        good_x = indices[valid]
        good_y = signal[valid]
        linear = np.interp(indices, good_x, good_y)
        
        # 前向填充
        forward = signal.copy()
        last_val = np.nan
        for i in range(len(signal)):
            if valid[i]:
                last_val = signal[i]
            forward[i] = last_val
        
        # 后向填充
        backward = signal.copy()
        next_val = np.nan
        for i in reversed(range(len(signal))):
            if valid[i]:
                next_val = signal[i]
            backward[i] = next_val
        
        # 三种方法平均
        filled = (forward + backward + linear) / 3
        
        # 只在无效区域应用填补
        result = signal.copy()
        result[~mask] = filled[~mask]
        
        return result
    
    def _sliding_window(self, signal: np.ndarray) -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]:
        """
        生成滑动窗口
        
        Args:
            signal: FHR信号
        
        Returns:
            (起始位置列表, 窗口列表, 权重列表)
        """
        win = self.window_len
        step = int(win * (1 - self.overlap_ratio))
        
        # 生成起始位置
        starts = list(range(0, len(signal) - win + 1, step))
        if len(signal) > win and starts[-1] + win < len(signal):
            starts.append(len(signal) - win)
        
        # 提取窗口
        windows = [signal[s:s + win] for s in starts]
        
        # 生成权重 (中间权重高，边缘权重低)
        weights = []
        for s in starts:
            w = np.ones(win)
            edge = win // 4
            w[:edge] = np.linspace(0.3, 1.0, edge)
            w[-edge:] = np.linspace(1.0, 0.3, edge)
            
            # 首尾窗口特殊处理
            if s == 0:
                w[:edge] = 1.0
            if s + win >= len(signal):
                w[-edge:] = 1.0
            weights.append(w)
        
        return starts, windows, weights
    
    def _merge_windows(self, starts: List[int], windows: List[np.ndarray], 
                       weights: List[np.ndarray], orig_len: int) -> np.ndarray:
        """
        加权融合窗口
        
        Args:
            starts: 起始位置列表
            windows: 窗口列表
            weights: 权重列表
            orig_len: 原始信号长度
        
        Returns:
            融合后的信号
        """
        out = np.zeros(orig_len, dtype=np.float32)
        wsum = np.zeros(orig_len, dtype=np.float32)
        
        for s, w, wt in zip(starts, windows, weights):
            e = s + self.window_len
            if e > orig_len:
                e = orig_len
            out[s:e] += w[:e - s] * wt[:e - s]
            wsum[s:e] += wt[:e - s]
        
        wsum[wsum < 0.01] = 1.0
        return out / wsum
    
    def fill_large_gap(self, signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        使用GAN填补大缺口
        
        Args:
            signal: FHR信号
            mask: 质量掩码 (True=有效)
        
        Returns:
            填补后的信号
        """
        result = np.asarray(signal, dtype=np.float32).copy()
        bad = ~mask
        
        # 生成滑动窗口
        starts, windows, weights = self._sliding_window(signal)
        
        # 对每个窗口使用GAN生成
        gen_windows = []
        for win in tqdm(windows, desc="GAN生成窗口", leave=False):
            gen_windows.append(self.gan_filler.generate_window(win))
        
        # 融合生成结果
        gan_full = self._merge_windows(starts, gen_windows, weights, len(signal))
        
        # 只填补无效区域
        result[bad] = gan_full[bad]
        
        return result
    
    def fill(self, signal: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        完整填补流程
        
        Args:
            signal: FHR信号
            mask: 质量掩码 (True=有效)
        
        Returns:
            (填补后的信号, 统计信息)
        """
        stats = {
            'original_length': len(signal),
            'small_gaps': 0,
            'large_gaps': 0,
            'small_gap_samples': 0,
            'large_gap_samples': 0
        }
        
        # 获取缺口信息
        gap_lengths = FHRQualityCleaner.get_gap_lengths(mask)
        if gap_lengths:
            stats['small_gaps'] = sum(1 for g in gap_lengths if g < self.small_threshold)
            stats['large_gaps'] = sum(1 for g in gap_lengths if g >= self.large_threshold)
            stats['small_gap_samples'] = sum(g for g in gap_lengths if g < self.small_threshold)
            stats['large_gap_samples'] = sum(g for g in gap_lengths if g >= self.large_threshold)
        
        # 第一步: 小缺口线性插值
        filled = self.fill_small_gap(signal, mask)
        
        # 更新掩码 (所有NaN都已填补)
        updated_mask = mask.copy()
        updated_mask[np.isnan(filled)] = False
        
        # 第二步: 大缺口GAN填补
        if stats['large_gaps'] > 0:
            filled = self.fill_large_gap(filled, updated_mask)
        
        return filled, stats


# ===================================================================
# 4. 可视化模块
# ===================================================================

class FHRVisualizer:
    """FHR信号可视化工具"""
    
    def __init__(self, output_dir: str, fs: float = 4):
        self.output_dir = output_dir
        self.fs = fs
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_comparison(self, original: np.ndarray, filled: np.ndarray, mask: np.ndarray,
                        filename: str, suffix: str = "", title_extra: str = "") -> str:
        """
        绘制原始与填补信号的对比图
        
        Args:
            original: 原始信号
            filled: 填补后的信号
            mask: 质量掩码
            filename: 文件名 (用于标题)
            suffix: 文件后缀
            title_extra: 额外标题信息
        
        Returns:
            保存的文件路径
        """
        t = np.arange(len(original)) / self.fs / 60  # 转换为分钟
        
        plt.figure(figsize=(18, 5))
        
        # 绘制原始信号
        plt.plot(t, original, c='#555', lw=1.2, alpha=0.7, label='原始信号')
        
        # 绘制填补信号
        plt.plot(t, filled, c='#e63946', lw=1.4, alpha=0.95, label='GAN填补信号')
        
        # 标记无效区域
        bad = ~mask
        if bad.any():
            for i in range(len(original)):
                if bad[i]:
                    plt.axvspan(t[i], t[i] + 1/self.fs/60, color='#ffaaaa', alpha=0.25)
        
        plt.ylim(50, 210)
        plt.grid(alpha=0.3)
        plt.title(f"文件: {filename} | {title_extra}", fontsize=14)
        plt.xlabel("时间 (分钟)", fontsize=12)
        plt.ylabel("胎心率 (bpm)", fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        save_name = f"{suffix}_{filename}" if suffix else f"{filename}"
        save_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=Config.PLOT_DPI)
        plt.close()
        
        return save_path


# ===================================================================
# 5. 主处理流程
# ===================================================================

class FHRDatasetProcessor:
    """FHR数据集处理器"""
    
    def __init__(self, config: PathConfig, gan_filler: GANFiller):
        self.config = config
        self.gan_filler = gan_filler
        self.cleaner = FHRQualityCleaner()
        self.filler = FHRFiller(gan_filler)
        self.visualizer = FHRVisualizer(config.PLOT_DIR, Config.FS)
        
        self.results = {
            'filenames': [],
            'fhr_original': [],
            'fhr_filled': [],
            'quality_mask': [],
            'stats': []
        }
    
    def process_single(self, signal: np.ndarray, mask: np.ndarray, 
                       filename: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        处理单个样本
        
        Args:
            signal: FHR信号
            mask: 质量掩码
            filename: 文件名
        
        Returns:
            (填补后的信号, 统计信息)
        """
        # 1. 清洗
        cleaned_signal, cleaned_mask = self.cleaner.clean_signal(signal, mask)
        if cleaned_signal is None:
            print(f"⚠️ 样本 {filename} 清洗后无效，跳过")
            return None, None
        
        # 2. 填补
        filled_signal, stats = self.filler.fill(cleaned_signal, cleaned_mask)
        
        return filled_signal, stats
    
    def process_all(self):
        """处理所有样本"""
        print("\n" + "=" * 60)
        print("开始FHR信号填补流程")
        print("=" * 60)
        
        # 加载数据
        print(f"\n[1/4] 加载数据: {self.config.INPUT_NPZ}")
        data = np.load(self.config.INPUT_NPZ, allow_pickle=True)
        
        fhr_origin = data["fhr"]
        quality_mask = data["quality_mask"]
        filenames = data["filenames"]
        
        print(f"样本总数: {len(fhr_origin)}")
        print(f"采样频率: {Config.FS} Hz")
        
        # 处理每个样本
        print("\n[2/4] 处理样本...")
        successful = 0
        skipped = 0
        
        for i in tqdm(range(len(fhr_origin)), desc="处理进度"):
            signal = fhr_origin[i]
            mask = quality_mask[i]
            filename = str(filenames[i])
            
            filled, stats = self.process_single(signal, mask, filename)
            
            if filled is not None:
                self.results['filenames'].append(filename)
                self.results['fhr_original'].append(signal)
                self.results['fhr_filled'].append(filled)
                self.results['quality_mask'].append(mask)
                self.results['stats'].append(stats)
                successful += 1
            else:
                skipped += 1
        
        print(f"✅ 成功处理: {successful}, 跳过: {skipped}")
        
        # 保存数据
        print("\n[3/4] 保存数据...")
        self._save_results()
        
        # 生成可视化
        print("\n[4/4] 生成可视化...")
        self._generate_visualizations()
        
        # 打印统计
        self._print_statistics()
        
        print("\n" + "=" * 60)
        print("✅ 处理完成!")
        print(f"输出文件: {self.config.OUTPUT_PATH}")
        print(f"可视化目录: {self.config.PLOT_DIR}")
        print("=" * 60)
    
    def _save_results(self):
        """保存结果到npz文件"""
        np.savez_compressed(
            self.config.OUTPUT_PATH,
            filenames=np.array(self.results['filenames'], dtype=object),
            fhr_original=np.array(self.results['fhr_original'], dtype=object),
            fhr_filled=np.array(self.results['fhr_filled'], dtype=object),
            quality_mask=np.array(self.results['quality_mask'], dtype=object),
            fs=np.array(Config.FS),
            metadata=np.array({
                'total_samples': len(self.results['filenames']),
                'fs': Config.FS,
                'timestamp': datetime.now().isoformat()
            }, dtype=object)
        )
        print(f"✅ 数据已保存: {self.config.OUTPUT_PATH}")
    
    def _generate_visualizations(self):
        """生成可视化对比图"""
        if len(self.results['filenames']) == 0:
            print("⚠️ 无有效样本，跳过可视化")
            return
        
        # 分类样本
        small_gap_ids = []
        large_gap_ids = []
        
        for i, stats in enumerate(self.results['stats']):
            if stats['large_gaps'] > 0:
                large_gap_ids.append(i)
            elif stats['small_gaps'] > 0:
                small_gap_ids.append(i)
        
        # 随机选择
        random.seed(Config.SEED)
        sample_count = min(Config.PLOT_SAMPLES, len(small_gap_ids))
        selected_small = random.sample(small_gap_ids, sample_count) if small_gap_ids else []
        
        sample_count = min(Config.PLOT_SAMPLES, len(large_gap_ids))
        selected_large = random.sample(large_gap_ids, sample_count) if large_gap_ids else []
        
        # 绘制小缺口样本
        if selected_small:
            print(f"  绘制小缺口样本: {len(selected_small)} 个")
            for idx in tqdm(selected_small, desc="小缺口可视化"):
                self._plot_sample(idx, "small_gap")
        
        # 绘制大缺口样本
        if selected_large:
            print(f"  绘制大缺口样本: {len(selected_large)} 个")
            for idx in tqdm(selected_large, desc="大缺口可视化"):
                self._plot_sample(idx, "large_gap_gan")
        
        # 绘制统计汇总
        self._plot_summary()
    
    def _plot_sample(self, idx: int, suffix: str):
        """绘制单个样本的对比图"""
        filename = self.results['filenames'][idx]
        original = self.results['fhr_original'][idx]
        filled = self.results['fhr_filled'][idx]
        mask = self.results['quality_mask'][idx]
        stats = self.results['stats'][idx]
        
        title_extra = (
            f"小缺口: {stats['small_gaps']} 处, "
            f"大缺口: {stats['large_gaps']} 处"
        )
        
        self.visualizer.plot_comparison(
            original, filled, mask, filename, suffix, title_extra
        )
    
    def _plot_summary(self):
        """绘制整体统计摘要图"""
        if not self.results['filenames']:
            return
        
        # 收集统计信息
        small_gap_counts = [s['small_gaps'] for s in self.results['stats']]
        large_gap_counts = [s['large_gaps'] for s in self.results['stats']]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 分布图
        axes[0].hist(small_gap_counts, bins=20, alpha=0.7, color='blue', label='小缺口')
        axes[0].hist(large_gap_counts, bins=20, alpha=0.7, color='red', label='大缺口')
        axes[0].set_xlabel('缺口数量')
        axes[0].set_ylabel('样本数')
        axes[0].set_title('缺口分布统计')
        axes[0].legend()
        
        # 统计信息
        stats_text = [
            f"总样本数: {len(self.results['filenames'])}",
            f"含小缺口样本: {sum(1 for s in self.results['stats'] if s['small_gaps'] > 0)}",
            f"含大缺口样本: {sum(1 for s in self.results['stats'] if s['large_gaps'] > 0)}",
            "",
            f"小缺口总数: {sum(s['small_gaps'] for s in self.results['stats'])}",
            f"大缺口总数: {sum(s['large_gaps'] for s in self.results['stats'])}",
        ]
        axes[1].axis('off')
        axes[1].text(0.05, 0.95, '\n'.join(stats_text), transform=axes[1].transAxes,
                    fontsize=12, verticalalignment='top')
        
        plt.tight_layout()
        save_path = os.path.join(self.config.PLOT_DIR, "summary_statistics.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  ✅ 统计摘要图: {save_path}")
    
    def _print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("处理统计摘要")
        print("=" * 60)
        
        total = len(self.results['filenames'])
        print(f"有效样本数: {total}")
        
        small_gap_samples = sum(1 for s in self.results['stats'] if s['small_gaps'] > 0)
        large_gap_samples = sum(1 for s in self.results['stats'] if s['large_gaps'] > 0)
        
        print(f"含小缺口样本: {small_gap_samples} ({small_gap_samples/total*100:.1f}%)")
        print(f"含大缺口样本: {large_gap_samples} ({large_gap_samples/total*100:.1f}%)")
        
        total_small_gaps = sum(s['small_gaps'] for s in self.results['stats'])
        total_large_gaps = sum(s['large_gaps'] for s in self.results['stats'])
        total_small_samples = sum(s['small_gap_samples'] for s in self.results['stats'])
        total_large_samples = sum(s['large_gap_samples'] for s in self.results['stats'])
        
        print(f"\n总缺口数: {total_small_gaps + total_large_gaps}")
        print(f"  - 小缺口: {total_small_gaps} (填补样本数: {total_small_samples})")
        print(f"  - 大缺口: {total_large_gaps} (填补样本数: {total_large_samples})")


# ===================================================================
# 6. 主入口
# ===================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FHR-GAN信号填补工具')
    
    parser.add_argument('--input', type=str, default=None,
                        help='输入npz文件路径 (4Hz FSQI质量筛查结果)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='GAN预训练权重路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出npz文件路径')
    parser.add_argument('--plot_dir', type=str, default=None,
                        help='可视化输出目录')
    parser.add_argument('--fs', type=int, default=4,
                        help='采样频率 (Hz)')
    parser.add_argument('--samples', type=int, default=10,
                        help='可视化样本数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()


def setup_from_args(args):
    """从命令行参数更新配置"""
    if args.input:
        PathConfig.INPUT_NPZ = args.input
    if args.checkpoint:
        PathConfig.GAN_CHECKPOINT = args.checkpoint
    if args.output:
        PathConfig.OUTPUT_PATH = args.output
    if args.plot_dir:
        PathConfig.PLOT_DIR = args.plot_dir
    if args.fs:
        Config.FS = args.fs
    if args.samples:
        Config.PLOT_SAMPLES = args.samples
    if args.seed:
        Config.SEED = args.seed
    
    # 设置随机种子
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    setup_from_args(args)
    
    # 创建输出目录
    PathConfig.create_output_dirs()
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化GAN填充器
    print(f"\n加载GAN模型: {PathConfig.GAN_CHECKPOINT}")
    gan_filler = GANFiller(PathConfig.GAN_CHECKPOINT, device)
    
    # 初始化处理器
    processor = FHRDatasetProcessor(PathConfig, gan_filler)
    
    # 执行处理
    processor.process_all()


if __name__ == "__main__":
    main()