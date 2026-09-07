"""
FHR-GAN: 基于条件生成对抗网络的胎心监护数据质量增强系统

这是一个用于处理胎心监护(FHR)数据不平衡问题的深度学习工具。
支持不良妊娠结局的条件生成和信号填补。

主要功能:
1. 多结局条件GAN训练 (可选择开启/关闭条件)
2. 差异化数据增强策略 (阳性样本增强)
3. 信号缺失填补 (基于生成器的inpainting)
4. 特定结局样本生成

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime
import random
from scipy import signal as scipy_signal
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings('ignore')


# ======================== 配置模块 ========================

class Config:
    """全局配置类"""
    # 数据配置
    SIGNAL_LENGTH = 2400
    BATCH_SIZE = 16
    WINDOW_SIZE = 1800
    
    # 结局配置
    ALL_OUTCOMES = []
    
    # ========== 关键开关：是否使用结局条件 ==========
    USE_OUTCOME_CONDITION = False  # True: 使用结局标签条件, False: 不使用
    
    # 训练配置
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    NOISE_DIM = 50
    
    # 增强配置
    POSITIVE_AUGMENT_FACTOR = 5
    NEGATIVE_AUGMENT_FACTOR = 0
    POSITIVE_SLIDING_STEP = 200
    NEGATIVE_SLIDING_STEP = 2400
    
    # 模型配置
    USE_AUTOENCODER_PRETRAIN = True
    USE_SLIDING_WINDOW = True
    
    # 增强方法
    AUGMENTATION_METHODS = [
        'time_warp', 'scaling', 'noise', 'time_shift', 
        'frequency_shift', 'smoothing', 'cutout'
    ]
    
    # 路径配置（用户需要修改）
    DATA_PATH = './data/train_data.npy'
    QUALITY_CSV_PATH = './data/good_quality_files.csv'
    MATERNAL_PATH = './data/name_age_bmi_ga_maternal.npy'
    OUTPUT_DIR = './outputs'

# ======================== 数据增强模块 ========================

class FHRDataAugmentation:
    """
    胎心数据增强方法集
    
    提供7种数据增强方法，可组合使用：
    - time_warp: 时间扭曲
    - scaling: 幅度缩放
    - noise: 添加噪声
    - time_shift: 时间偏移
    - frequency_shift: 频率偏移
    - smoothing: 平滑处理
    - cutout: 随机遮挡
    """
    
    @staticmethod
    def time_warp(signal, warp_factor=0.1):
        """时间扭曲增强"""
        signal_len = len(signal)
        original_time = np.linspace(0, 1, signal_len)
        warp_points = np.linspace(0, 1, 5)
        warp_values = np.random.uniform(-warp_factor, warp_factor, 5)
        warp_values[0] = warp_values[-1] = 0
        
        cs = CubicSpline(warp_points, warp_points + warp_values)
        warped_time = cs(original_time)
        warped_time = np.clip(warped_time, 0, 1)
        warped_signal = np.interp(original_time, warped_time, signal)
        return warped_signal

    @staticmethod
    def scaling(signal, scale_range=(0.8, 1.2)):
        """信号缩放增强"""
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return signal * scale_factor

    @staticmethod
    def add_noise(signal, noise_level=0.05):
        """添加高斯噪声"""
        noise = np.random.normal(0, noise_level * np.std(signal), len(signal))
        return signal + noise

    @staticmethod
    def time_shift(signal, max_shift_ratio=0.1):
        """时间偏移增强"""
        signal_len = len(signal)
        max_shift = int(signal_len * max_shift_ratio)
        shift = np.random.randint(-max_shift, max_shift)
        if shift > 0:
            shifted = np.concatenate([signal[shift:], signal[:shift]])
        elif shift < 0:
            shifted = np.concatenate([signal[shift:], signal[:shift]])
        else:
            shifted = signal.copy()
        return shifted

    @staticmethod
    def frequency_shift(signal, shift_range=(-0.1, 0.1)):
        """频率偏移增强"""
        phase_shift = np.random.uniform(shift_range[0], shift_range[1]) * 2 * np.pi
        freqs = np.fft.fft(signal)
        shifted_freqs = freqs * np.exp(1j * phase_shift * np.arange(len(signal)))
        return np.real(np.fft.ifft(shifted_freqs))

    @staticmethod
    def smoothing(signal, window_size=5):
        """Savitzky-Golay平滑"""
        if window_size % 2 == 0:
            window_size += 1
        return scipy_signal.savgol_filter(signal, window_size, 2)

    @staticmethod
    def cutout(signal, num_cutouts=3, max_cutout_length=50):
        """随机遮挡增强"""
        signal_len = len(signal)
        augmented = signal.copy()
        for _ in range(num_cutouts):
            cutout_length = np.random.randint(10, max_cutout_length)
            start_pos = np.random.randint(0, signal_len - cutout_length)
            if start_pos > 0 and start_pos + cutout_length < signal_len:
                left_val = augmented[start_pos - 1]
                right_val = augmented[start_pos + cutout_length]
                augmented[start_pos:start_pos + cutout_length] = np.linspace(
                    left_val, right_val, cutout_length
                )
        return augmented

    @staticmethod
    def random_augment(signal, num_augmentations=2):
        """随机组合多种增强方法"""
        augmented = signal.copy()
        methods = random.sample(Config.AUGMENTATION_METHODS, num_augmentations)
        for method in methods:
            if method == 'time_warp':
                augmented = FHRDataAugmentation.time_warp(augmented)
            elif method == 'scaling':
                augmented = FHRDataAugmentation.scaling(augmented)
            elif method == 'noise':
                augmented = FHRDataAugmentation.add_noise(augmented)
            elif method == 'time_shift':
                augmented = FHRDataAugmentation.time_shift(augmented)
            elif method == 'frequency_shift':
                augmented = FHRDataAugmentation.frequency_shift(augmented)
            elif method == 'smoothing':
                augmented = FHRDataAugmentation.smoothing(augmented)
            elif method == 'cutout':
                augmented = FHRDataAugmentation.cutout(augmented)
        return augmented


# ======================== 数据集模块 ========================

class EnhancedBalancedMultiOutcomeFHRQualityDataset(Dataset):
    """
    增强的平衡多结局胎心数据集
    
    特点：
    1. 差异化增强：阳性样本增强5倍，阴性样本不增强
    2. 滑动窗口：阳性样本步长200，阴性样本步长2400
    3. 人工缺失：随机创建1-3个缺失区域用于训练填补
    """
    
    def __init__(self, fhr_data, labels_dict=None, features=None, signal_length=2400, 
                 use_sliding_window=True, positive_aug_factor=5, negative_aug_factor=0,
                 use_advanced_augmentation=True, use_outcome_condition=True):
        """
        Args:
            fhr_data: 胎心数据
            labels_dict: 标签字典 (如果 use_outcome_condition=False，可为None)
            features: 额外特征
            signal_length: 信号长度
            use_sliding_window: 是否使用滑动窗口
            positive_aug_factor: 阳性样本增强倍数
            negative_aug_factor: 阴性样本增强倍数
            use_advanced_augmentation: 是否使用高级数据增强
            use_outcome_condition: 是否使用结局条件
        """
        self.fhr_data = torch.tensor(fhr_data, dtype=torch.float32)
        self.signal_length = signal_length
        self.use_sliding_window = use_sliding_window
        self.positive_aug_factor = positive_aug_factor
        self.negative_aug_factor = negative_aug_factor
        self.use_advanced_augmentation = use_advanced_augmentation
        self.use_outcome_condition = use_outcome_condition
        
        # 如果不用结局条件，标签字典设为None
        self.labels_dict = labels_dict if use_outcome_condition else None
        self.outcome_names = list(labels_dict.keys()) if labels_dict is not None and use_outcome_condition else []
        
        if features is not None:
            self.features = torch.tensor(features, dtype=torch.float32)
        else:
            self.features = None
            
        self.data_mean = torch.mean(self.fhr_data)
        self.data_std = torch.std(self.fhr_data)
        
        if self.use_sliding_window:
            if self.use_outcome_condition:
                self.processed_data, self.processed_labels_dict = self._apply_enhanced_balanced_sliding_window()
            else:
                self.processed_data, self.processed_labels_dict = self._apply_simple_sliding_window()
            self.signal_length = Config.WINDOW_SIZE
        else:
            self.processed_data = self.fhr_data
            self.processed_labels_dict = self.labels_dict
    
    def _apply_simple_sliding_window(self):
        """简单滑动窗口（不使用结局条件）"""
        print("应用简单滑动窗口（无结局条件）...")
        all_windows = []
        
        for i in range(len(self.fhr_data)):
            signal = self.fhr_data[i].numpy()
            windows = self._extract_windows(signal, Config.WINDOW_SIZE, 200, is_negative_sample=False)
            for window in windows:
                all_windows.append(torch.tensor(window, dtype=torch.float32))
        
        print(f"总样本数: {len(all_windows)} (原始: {len(self.fhr_data)})")
        return torch.stack(all_windows), None
        
    def _apply_enhanced_balanced_sliding_window(self):
        """应用差异化滑动窗口增强（使用结局条件）"""
        print("应用差异化滑动窗口增强...")
        all_windows = []
        all_labels_dict = {outcome: [] for outcome in self.outcome_names}
        
        total_samples = len(self.fhr_data)
        
        # 分类样本
        enhancement_types = []
        for i in range(total_samples):
            is_all_negative = True
            for outcome in self.outcome_names:
                if self.labels_dict[outcome][i] == 1:
                    is_all_negative = False
                    break
            enhancement_types.append('all_negative' if is_all_negative else 'any_positive')
        
        print(f"样本统计: 全阴性 {enhancement_types.count('all_negative')}, "
              f"含阳性 {enhancement_types.count('any_positive')}")
        
        for i in range(total_samples):
            signal = self.fhr_data[i].numpy()
            enhancement_type = enhancement_types[i]
            
            if enhancement_type == 'all_negative':
                aug_factor = self.negative_aug_factor
                step_size = Config.NEGATIVE_SLIDING_STEP
                use_augmentation = False
            else:
                aug_factor = self.positive_aug_factor
                step_size = Config.POSITIVE_SLIDING_STEP
                use_augmentation = self.use_advanced_augmentation
            
            windows = self._extract_windows(signal, Config.WINDOW_SIZE, step_size, 
                                           enhancement_type == 'all_negative')
            
            enhanced_windows = []
            for window in windows:
                if enhancement_type == 'all_negative':
                    enhanced_windows.append(torch.tensor(window, dtype=torch.float32))
                else:
                    if use_augmentation:
                        for aug_idx in range(aug_factor):
                            if aug_idx == 0:
                                enhanced_windows.append(torch.tensor(window, dtype=torch.float32))
                            else:
                                augmented_window = FHRDataAugmentation.random_augment(window)
                                enhanced_windows.append(torch.tensor(augmented_window, dtype=torch.float32))
                    else:
                        enhanced_windows.extend([torch.tensor(window, dtype=torch.float32)] * aug_factor)
            
            all_windows.extend(enhanced_windows)
            for outcome in self.outcome_names:
                label = self.labels_dict[outcome][i]
                all_labels_dict[outcome].extend([label] * len(enhanced_windows))
        
        print(f"增强后总样本数: {len(all_windows)} (原始: {total_samples})")
        return torch.stack(all_windows), {outcome: torch.tensor(labels) for outcome, labels in all_labels_dict.items()}
    
    def _extract_windows(self, signal, window_size, step_size, is_negative_sample=False):
        """提取滑动窗口"""
        windows = []
        signal_length = len(signal)
        
        if signal_length == window_size:
            return [signal]
        
        if is_negative_sample:
            # 阴性样本：只取中心窗口
            center_start = (signal_length - window_size) // 2
            windows.append(signal[center_start:center_start + window_size])
        else:
            # 阳性样本：滑动窗口
            start_idx = 0
            while start_idx + window_size <= signal_length:
                windows.append(signal[start_idx:start_idx + window_size])
                start_idx += step_size
            if not windows:
                windows.append(signal[:window_size])
        
        return windows
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        signal = self.processed_data[idx]
        signal_with_gaps, gap_mask = self._create_artificial_gaps(signal)
        
        if self.processed_labels_dict is not None and self.features is not None:
            labels_tensor = torch.stack([self.processed_labels_dict[outcome][idx] for outcome in self.outcome_names])
            return signal_with_gaps, labels_tensor, self.features[idx], gap_mask, signal
        elif self.processed_labels_dict is not None:
            labels_tensor = torch.stack([self.processed_labels_dict[outcome][idx] for outcome in self.outcome_names])
            return signal_with_gaps, labels_tensor, gap_mask, signal
        elif self.features is not None:
            return signal_with_gaps, self.features[idx], gap_mask, signal
        else:
            return signal_with_gaps, gap_mask, signal
    
    def _create_artificial_gaps(self, signal):
        """创建人工缺失区域"""
        signal_with_gaps = signal.clone()
        gap_mask = torch.zeros_like(signal)
        
        num_gaps = torch.randint(1, 4, (1,)).item()
        for _ in range(num_gaps):
            gap_length = torch.randint(50, 201, (1,)).item()
            start_pos = torch.randint(0, self.signal_length - gap_length, (1,)).item()
            signal_with_gaps[start_pos:start_pos+gap_length] = 0
            gap_mask[start_pos:start_pos+gap_length] = 1
                
        return signal_with_gaps, gap_mask


# ======================== GAN模型模块 ========================
class MultiOutcomeStabilizedConditionalFHRGenerator(nn.Module):
    """
    多结局稳定的条件胎心数据生成器
    
    输入: 噪声(50) + 结局标签(7) + 上下文特征(3)
    输出: 胎心信号(2400)
    """
    
    def __init__(self, noise_dim=50, num_outcomes=7, signal_length=2400, feature_dim=3):
        super(MultiOutcomeStabilizedConditionalFHRGenerator, self).__init__()
        
        self.signal_length = signal_length
        self.noise_dim = noise_dim
        self.num_outcomes = num_outcomes
        self.feature_dim = feature_dim
        
        # 如果 num_outcomes=0，则输入维度不包含结局标签
        input_size = noise_dim + num_outcomes + feature_dim
        
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
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, noise, labels=None, features=None):
        if labels is not None and features is not None:
            x = torch.cat([noise, labels, features], dim=1)
        elif labels is not None:
            x = torch.cat([noise, labels], dim=1)
        elif features is not None:
            x = torch.cat([noise, features], dim=1)
        else:
            x = noise
        return self.main(x)


class MultiOutcomeBalancedConditionalFHRDiscriminator(nn.Module):
    """
    多结局平衡的条件胎心数据判别器
    
    使用卷积网络提取特征，并融合结局标签和上下文特征进行判别
    """
    
    def __init__(self, signal_length=2400, num_outcomes=7, feature_dim=3):
        super(MultiOutcomeBalancedConditionalFHRDiscriminator, self).__init__()
        
        self.signal_length = signal_length
        self.num_outcomes = num_outcomes
        self.feature_dim = feature_dim
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=31, stride=4, padding=15),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.2),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        with torch.no_grad():
            x = torch.randn(1, 1, signal_length)
            x = self.conv_layers(x)
            conv_output_size = x.size(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(conv_output_size + num_outcomes + feature_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, True),
            
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, signal, labels=None, features=None):
        x = signal.unsqueeze(1)
        conv_features = self.conv_layers(x)
        
        if labels is not None and features is not None:
            x_combined = torch.cat([conv_features, labels, features], dim=1)
        elif labels is not None:
            x_combined = torch.cat([conv_features, labels], dim=1)
        elif features is not None:
            x_combined = torch.cat([conv_features, features], dim=1)
        else:
            x_combined = conv_features
            
        output = self.classifier(x_combined)
        return output * 0.1


class SimpleAutoencoder(nn.Module):
    """简单的自编码器，用于预训练"""
    
    def __init__(self, signal_length=2400, latent_dim=100):
        super(SimpleAutoencoder, self).__init__()
        self.signal_length = signal_length
        
        hidden1 = max(512, signal_length // 4)
        hidden2 = max(256, signal_length // 8)
        
        self.encoder = nn.Sequential(
            nn.Linear(signal_length, hidden1),
            nn.ReLU(True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(True),
            nn.Linear(hidden2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden2),
            nn.ReLU(True),
            nn.Linear(hidden2, hidden1),
            nn.ReLU(True),
            nn.Linear(hidden1, signal_length),
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ======================== 训练管理模块 ========================

class FHRQualityEnhancer:
    """
    胎心数据质量增强器
    
    整合了自编码器预训练和条件GAN训练，提供：
    1. 数据增强训练
    2. 信号缺失填补
    3. 特定结局样本生成
    """
    
    def __init__(self, signal_length=2400, noise_dim=50, feature_dim=3, num_outcomes=7,
                 use_autoencoder_pretrain=True, use_outcome_condition=True,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            signal_length: 信号长度
            noise_dim: 噪声维度
            feature_dim: 上下文特征维度
            num_outcomes: 结局数量 (设为0表示不使用结局条件)
            use_autoencoder_pretrain: 是否使用自编码器预训练
            use_outcome_condition: 是否使用结局条件
            device: 计算设备
        """
        self.signal_length = signal_length
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim
        self.num_outcomes = num_outcomes if use_outcome_condition else 0  # 不用条件则设为0
        self.use_autoencoder_pretrain = use_autoencoder_pretrain
        self.use_outcome_condition = use_outcome_condition
        self.device = device
        
        # 初始化生成器和判别器
        # num_outcomes=0 时，模型自动变为无条件GAN
        self.generator = MultiOutcomeStabilizedConditionalFHRGenerator(
            noise_dim, self.num_outcomes, signal_length, feature_dim).to(device)
        self.discriminator = MultiOutcomeBalancedConditionalFHRDiscriminator(
            signal_length, self.num_outcomes, feature_dim).to(device)
        
        # 自编码器
        self.autoencoder = SimpleAutoencoder(signal_length).to(device)
        
        # 优化器
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), 
                                           lr=Config.LEARNING_RATE, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), 
                                           lr=Config.LEARNING_RATE, betas=(0.5, 0.999))
        
        # 训练历史
        self.history = {
            'g_loss': [], 'd_loss': [], 
            'real_scores': [], 'fake_scores': [], 'ae_loss': []
        }
        
        # 输出目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        condition_str = "conditional" if use_outcome_condition else "unconditional"
        self.output_dir = os.path.join(Config.OUTPUT_DIR, f"fhr_enhancer_{condition_str}_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.outcome_names = Config.ALL_OUTCOMES if use_outcome_condition else []
        
        print(f"初始化完成:")
        print(f"  - 使用结局条件: {use_outcome_condition}")
        print(f"  - 结局数量: {self.num_outcomes}")
        print(f"  - 输出目录: {self.output_dir}")
    
    def train_autoencoder(self, dataloader, epochs=50):
        """训练自编码器"""
        print("开始训练自编码器...")
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"AE Epoch {epoch+1}/{epochs}"):
                if len(batch) >= 3:
                    clean = batch[-1].to(self.device)
                else:
                    clean = batch[0].to(self.device)
                
                optimizer.zero_grad()
                reconstructed = self.autoencoder(clean)
                loss = criterion(reconstructed, clean)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.history['ae_loss'].append(avg_loss)
            print(f"AE Epoch {epoch+1}: Loss: {avg_loss:.6f}")
        
        print("自编码器训练完成!")
    
    def train_gan(self, dataloader, epochs=100):
        """训练GAN"""
        mode_str = "条件" if self.use_outcome_condition else "无条件"
        print(f"开始{mode_str}GAN训练...")
        if self.use_outcome_condition:
            print(f"结局: {self.outcome_names}")
        else:
            print("无结局条件，生成器将学习整体数据分布")
        
        for epoch in range(epochs):
            g_losses, d_losses = [], []
            real_scores, fake_scores = [], []
            
            for batch in tqdm(dataloader, desc=f"GAN Epoch {epoch+1}/{epochs}"):
                # 解析batch数据
                if len(batch) >= 3 and self.use_outcome_condition:
                    # 有标签模式
                    if len(batch) == 4:  # signal, labels, gap_mask, clean
                        corrupted, all_labels, gap_mask, clean = batch[0], batch[1], batch[2], batch[3]
                    else:  # signal, labels, features, gap_mask, clean
                        corrupted, all_labels, features, gap_mask, clean = batch[0], batch[1], batch[2], batch[3], batch[4]
                    
                    all_labels = all_labels.float().to(self.device)
                    context_features = self._extract_context_features(gap_mask, clean)
                    if context_features is None or context_features.shape[1] != 3:
                        continue
                    context_features = context_features.to(self.device)
                    
                    batch_size = clean.size(0)
                    if context_features.size(0) != batch_size:
                        min_batch = min(batch_size, context_features.size(0))
                        clean = clean[:min_batch]
                        all_labels = all_labels[:min_batch] if all_labels is not None else None
                        context_features = context_features[:min_batch]
                        batch_size = min_batch
                else:
                    # 无标签模式
                    if len(batch) == 3:  # signal, gap_mask, clean
                        corrupted, gap_mask, clean = batch[0], batch[1], batch[2]
                    else:
                        corrupted, gap_mask, clean = batch[0], batch[-2], batch[-1]
                    all_labels = None
                    context_features = None
                    
                clean = clean.to(self.device)
                batch_size = clean.size(0)
                
                if batch_size == 0:
                    continue
                
                # ===== 训练判别器 =====
                self.d_optimizer.zero_grad()
                
                if self.use_outcome_condition and all_labels is not None:
                    real_output = self.discriminator(clean, all_labels, context_features)
                else:
                    real_output = self.discriminator(clean)
                
                real_output = torch.clamp(real_output, -1.0, 1.0)
                real_loss = -torch.mean(real_output)
                
                # 生成假数据
                with torch.no_grad():
                    noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                    if self.use_outcome_condition and all_labels is not None:
                        fake_data = self.generator(noise, all_labels, context_features)
                    else:
                        fake_data = self.generator(noise)
                
                if self.use_outcome_condition and all_labels is not None:
                    fake_output = self.discriminator(fake_data.detach(), all_labels, context_features)
                else:
                    fake_output = self.discriminator(fake_data.detach())
                
                fake_output = torch.clamp(fake_output, -1.0, 1.0)
                fake_loss = torch.mean(fake_output)
                
                gradient_penalty = self._compute_gradient_penalty(
                    clean, fake_data.detach(), context_features, all_labels
                )
                
                d_loss = real_loss + fake_loss + 0.5 * gradient_penalty
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=0.05)
                self.d_optimizer.step()
                
                d_losses.append(d_loss.item())
                real_scores.append(torch.mean(real_output).item())
                fake_scores.append(torch.mean(fake_output).item())
                
                # ===== 训练生成器 =====
                self.g_optimizer.zero_grad()
                
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                if self.use_outcome_condition and all_labels is not None:
                    fake_data = self.generator(noise, all_labels, context_features)
                    fake_output = self.discriminator(fake_data, all_labels, context_features)
                else:
                    fake_data = self.generator(noise)
                    fake_output = self.discriminator(fake_data)
                
                fake_output = torch.clamp(fake_output, -1.0, 1.0)
                g_loss = -torch.mean(fake_output)
                
                # 特征匹配损失
                if epoch > 0:
                    with torch.no_grad():
                        if self.use_outcome_condition and all_labels is not None:
                            real_features = self._get_discriminator_features(clean, all_labels)
                            fake_features = self._get_discriminator_features(fake_data, all_labels)
                        else:
                            real_features = self._get_discriminator_features(clean)
                            fake_features = self._get_discriminator_features(fake_data)
                    
                    feature_matching_loss = F.mse_loss(fake_features, real_features.detach())
                    g_loss = g_loss + 0.3 * feature_matching_loss
                
                output_range_loss = torch.mean(torch.relu(torch.abs(fake_data) - 0.9))
                g_loss = g_loss + 0.05 * output_range_loss
                
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=0.03)
                self.g_optimizer.step()
                
                g_losses.append(g_loss.item())
            
            # 记录训练指标
            avg_g_loss = np.mean(g_losses) if g_losses else 0
            avg_d_loss = np.mean(d_losses)
            avg_real_score = np.mean(real_scores)
            avg_fake_score = np.mean(fake_scores)
            
            self.history['g_loss'].append(avg_g_loss)
            self.history['d_loss'].append(avg_d_loss)
            self.history['real_scores'].append(avg_real_score)
            self.history['fake_scores'].append(avg_fake_score)
            
            print(f"Epoch {epoch+1}: G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}, "
                  f"D(real): {avg_real_score:.4f}, D(fake): {avg_fake_score:.4f}")
            
            # 定期保存和生成样本
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)
                self._generate_sample_images(epoch)
        
        print("GAN训练完成!")
    
    def generate_samples(self, outcome_name=None, num_samples=5, outcome_value=1):
        """
        生成样本
        
        Args:
            outcome_name: 指定结局名称，None则随机生成
            num_samples: 生成样本数量
            outcome_value: 结局值 (1=阳性, 0=阴性)
        
        Returns:
            generated_signals: 生成的信号数组
            labels: 对应的标签数组 (如果不使用条件，返回None)
        """
        self.generator.eval()
        with torch.no_grad():
            if self.use_outcome_condition and outcome_name is not None and outcome_name in self.outcome_names:
                outcome_idx = self.outcome_names.index(outcome_name)
                labels = torch.zeros(num_samples, self.num_outcomes).to(self.device)
                labels[:, outcome_idx] = outcome_value
            elif self.use_outcome_condition:
                # 随机生成标签
                labels = torch.randint(0, 2, (num_samples, self.num_outcomes)).float().to(self.device)
            else:
                labels = None
            
            features = self._create_default_context_features(num_samples).to(self.device)
            noise = torch.randn(num_samples, self.noise_dim).to(self.device)
            
            if self.use_outcome_condition:
                generated_signals = self.generator(noise, labels, features)
            else:
                generated_signals = self.generator(noise)
            
            return generated_signals.cpu().numpy(), labels.cpu().numpy() if labels is not None else None
    
    def inpaint(self, corrupted_signal, gap_mask, labels=None):
        """
        信号缺失填补
        
        Args:
            corrupted_signal: 有缺失的信号
            gap_mask: 缺失区域掩码 (1表示缺失)
            labels: 结局标签 (可选，仅在条件模式下使用)
        
        Returns:
            reconstructed: 填补后的信号
        """
        self.generator.eval()
        with torch.no_grad():
            batch_size = corrupted_signal.size(0)
            corrupted_signal = corrupted_signal.to(self.device)
            gap_mask = gap_mask.to(self.device)
            
            context_features = self._extract_context_features(gap_mask, corrupted_signal)
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            
            if self.use_outcome_condition:
                if labels is not None:
                    labels = labels.float().to(self.device)
                else:
                    labels = torch.zeros(batch_size, self.num_outcomes).to(self.device)
                generated_content = self.generator(noise, labels, context_features)
            else:
                generated_content = self.generator(noise)
            
            reconstructed = corrupted_signal * (1 - gap_mask) + generated_content * gap_mask
            return reconstructed
    
    def _extract_context_features(self, gap_mask, signal):
        """提取上下文特征 (均值、标准差、变异性)"""
        batch_size, signal_length = signal.shape
        important_features = []
        
        for i in range(batch_size):
            sample_signal = signal[i]
            sample_signal = torch.nan_to_num(sample_signal, nan=0.0)
            sample_signal = torch.clamp(sample_signal, -1.0, 1.0)
            
            global_mean = torch.mean(sample_signal).unsqueeze(0)
            global_std = torch.std(sample_signal).unsqueeze(0)
            if len(sample_signal) > 1:
                diff = torch.diff(sample_signal)
                variability = torch.mean(torch.abs(diff)).unsqueeze(0)
            else:
                variability = torch.tensor(0.0).unsqueeze(0)
            
            feature_tensor = torch.cat([global_mean, global_std, variability])
            important_features.append(feature_tensor.unsqueeze(0))
        
        if important_features:
            result = torch.cat(important_features, dim=0)
        else:
            result = torch.zeros(batch_size, 3, device=signal.device)
        
        if torch.isnan(result).any() or torch.isinf(result).any():
            result = torch.zeros(batch_size, 3, device=result.device)
        
        if result.shape[1] != 3:
            if result.shape[1] < 3:
                padding = torch.zeros(result.shape[0], 3 - result.shape[1], device=result.device)
                result = torch.cat([result, padding], dim=1)
            else:
                result = result[:, :3]
        
        return result
    
    def _create_default_context_features(self, batch_size):
        """创建默认上下文特征"""
        default_features = torch.tensor([[0.0, 0.5, 0.1]], device=self.device)
        return default_features.repeat(batch_size, 1)
    
    def _get_discriminator_features(self, signal, labels=None):
        """获取判别器的中间特征"""
        x = signal.unsqueeze(1)
        if hasattr(self.discriminator, 'conv_layers'):
            features = self.discriminator.conv_layers(x)
            return torch.flatten(features, 1)
        else:
            return self.discriminator(signal, labels)
    
    def _compute_gradient_penalty(self, real_data, fake_data, context_features, labels=None):
        """计算梯度惩罚"""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
        
        if self.use_outcome_condition and labels is not None:
            disc_interpolates = self.discriminator(interpolates, labels, context_features)
        else:
            disc_interpolates = self.discriminator(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        if gradients is None:
            return torch.tensor(0.0, device=self.device)
        
        gradients = gradients.view(batch_size, -1)
        gradients = torch.clamp(gradients, -10.0, 10.0)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return torch.clamp(gradient_penalty, 0.0, 10.0)
    
    def _generate_sample_images(self, epoch):
        """生成样本对比图"""
        self.generator.eval()
        
        with torch.no_grad():
            # 获取随机噪声
            num_samples = 5
            noise = torch.randn(num_samples, self.noise_dim).to(self.device)
            
            if self.use_outcome_condition:
                # 为每个结局生成样本
                fig, axes = plt.subplots(num_samples, self.num_outcomes + 1, 
                                        figsize=(4*(self.num_outcomes + 1), 3*num_samples))
                
                for i in range(num_samples):
                    # 第一列：随机生成的样本
                    labels = torch.randint(0, 2, (1, self.num_outcomes)).float().to(self.device)
                    features = self._create_default_context_features(1).to(self.device)
                    sample = self.generator(noise[i:i+1], labels, features)
                    axes[i, 0].plot(denormalize_fhr(sample.cpu().numpy()[0]), linewidth=1.5, color='blue')
                    axes[i, 0].set_title(f"Random", fontsize=10)
                    axes[i, 0].set_ylim(50, 200)
                    axes[i, 0].grid(True, alpha=0.3)
                    
                    # 为每个结局生成样本
                    for j, outcome in enumerate(self.outcome_names):
                        labels = torch.zeros(1, self.num_outcomes).to(self.device)
                        labels[:, j] = 1  # 设置为阳性
                        features = self._create_default_context_features(1).to(self.device)
                        sample = self.generator(noise[i:i+1], labels, features)
                        axes[i, j+1].plot(denormalize_fhr(sample.cpu().numpy()[0]), linewidth=1.5, color='orange')
                        axes[i, j+1].set_title(f"{outcome}+", fontsize=10)
                        axes[i, j+1].set_ylim(50, 200)
                        axes[i, j+1].grid(True, alpha=0.3)
            else:
                # 无条件模式：只生成随机样本
                fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
                samples = self.generator(noise)
                for i in range(num_samples):
                    axes[i].plot(denormalize_fhr(samples[i].cpu().numpy()), linewidth=1.5, color='blue')
                    axes[i].set_title(f"Sample {i+1}", fontsize=10)
                    axes[i].set_ylim(50, 200)
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/samples_epoch_{epoch}.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"样本图已保存: {self.output_dir}/samples_epoch_{epoch}.png")
    
    def plot_training_progress(self):
        """绘制训练进度图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. GAN训练损失
        if len(self.history['g_loss']) > 0:
            epochs_range = range(1, len(self.history['g_loss']) + 1)
            axes[0, 0].plot(epochs_range, self.history['g_loss'], 'b-', label='Generator', linewidth=1)
            axes[0, 0].plot(epochs_range, self.history['d_loss'], 'r-', label='Discriminator', linewidth=1)
            axes[0, 0].set_title('GAN Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 判别器得分
        if len(self.history['real_scores']) > 0:
            epochs_range = range(1, len(self.history['real_scores']) + 1)
            axes[0, 1].plot(epochs_range, self.history['real_scores'], 'g-', label='Real', linewidth=1)
            axes[0, 1].plot(epochs_range, self.history['fake_scores'], 'm-', label='Fake', linewidth=1)
            axes[0, 1].set_title('Discriminator Scores')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 自编码器损失
        if len(self.history['ae_loss']) > 0:
            epochs_range = range(1, len(self.history['ae_loss']) + 1)
            axes[1, 0].plot(epochs_range, self.history['ae_loss'], 'purple', label='Autoencoder', linewidth=1)
            axes[1, 0].set_title('Autoencoder Training Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 训练信息
        axes[1, 1].axis('off')
        info_text = [
            "Training Summary:",
            f"Mode: {'Conditional' if self.use_outcome_condition else 'Unconditional'}",
            f"Outcomes: {len(self.outcome_names) if self.use_outcome_condition else 'N/A'}",
            f"Total Epochs: {len(self.history['g_loss'])}",
            f"Final G Loss: {self.history['g_loss'][-1]:.4f}" if self.history['g_loss'] else "N/A",
            f"Final D Loss: {self.history['d_loss'][-1]:.4f}" if self.history['d_loss'] else "N/A"
        ]
        axes[1, 1].text(0.05, 0.95, '\n'.join(info_text), transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_progress.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"训练进度图已保存: {self.output_dir}/training_progress.png")
    
    def _save_checkpoint(self, epoch):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'history': self.history,
            'use_autoencoder_pretrain': self.use_autoencoder_pretrain,
            'use_outcome_condition': self.use_outcome_condition,
            'signal_length': self.signal_length,
            'num_outcomes': self.num_outcomes,
            'outcome_names': self.outcome_names
        }
        
        if self.autoencoder is not None:
            checkpoint['autoencoder_state_dict'] = self.autoencoder.state_dict()
        
        torch.save(checkpoint, f"{self.output_dir}/checkpoint_epoch_{epoch}.pth")
        print(f"检查点已保存: {self.output_dir}/checkpoint_epoch_{epoch}.pth")


# ======================== 工具函数 ========================

def robust_normalize(signals):
    """鲁棒的标准化方法"""
    signals = np.clip(signals, 50, 200)
    p1, p99 = np.percentile(signals, [1, 99])
    signals = (signals - p1) / (p99 - p1)
    signals = 2 * signals - 1
    return np.clip(signals, -1, 1)


def denormalize_fhr(normalized_signals):
    """将标准化信号反标准化回原始胎心数值"""
    p1, p99 = 60, 180
    original_signals = (normalized_signals + 1) / 2 * (p99 - p1) + p1
    return np.clip(original_signals, 50, 200)


def extract_data(path, fixed_columns):
    """提取数据"""
    data = np.load(path, allow_pickle=True)
    fixed_data = data[:, :len(fixed_columns)]
    dynamic_data = data[:, len(fixed_columns):]
    return fixed_data, dynamic_data


def load_sample_data():
    """
    加载示例数据（用于Demo）
    
    如果没有真实数据，生成模拟数据
    """
    try:
        # 尝试加载真实数据
        fhr_data = np.load(Config.DATA_PATH, allow_pickle=True)
        print(f"成功加载数据: {fhr_data.shape}")
        return fhr_data
    except FileNotFoundError:
        print("未找到数据文件，生成模拟数据用于演示...")
        # 生成模拟胎心数据
        num_samples = 100
        signal_length = Config.SIGNAL_LENGTH
        fhr_data = []
        
        for _ in range(num_samples):
            # 模拟胎心信号：基线+变异+偶尔减速
            t = np.linspace(0, 10, signal_length)
            baseline = 140 + np.random.normal(0, 2)  # 基线130-150
            variability = 5 + np.random.exponential(3)  # 变异性
            signal = baseline + variability * np.sin(2 * np.pi * 0.02 * t + np.random.random() * np.pi)
            
            # 加入随机减速
            for _ in range(np.random.randint(0, 3)):
                idx = np.random.randint(100, signal_length - 100)
                duration = np.random.randint(30, 100)
                depth = np.random.randint(10, 30)
                signal[idx:idx+duration] -= depth * np.exp(-np.arange(duration) / 30)
            
            # 加入噪声
            signal += np.random.normal(0, 2, signal_length)
            signal = np.clip(signal, 50, 200)
            fhr_data.append(signal)
        
        fhr_data = np.array(fhr_data)
        print(f"生成模拟数据: {fhr_data.shape}")
        return fhr_data


# ======================== 主程序 ========================

def main():
    """主函数"""
    print("=" * 60)
    print("FHR-GAN: 胎心监护数据质量增强系统")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    fhr_data = load_sample_data()
    
    # 2. 创建模拟标签（如果没有真实标签）
    print("\n[2] 准备标签...")
    if Config.USE_OUTCOME_CONDITION:
        # 生成模拟标签
        labels_dict = {}
        for outcome in Config.ALL_OUTCOMES:
            labels_dict[outcome] = np.random.choice([0, 1], len(fhr_data), p=[0.85, 0.15])
            print(f"  {outcome}: 阳性{sum(labels_dict[outcome])}, 阴性{len(fhr_data) - sum(labels_dict[outcome])}")
    else:
        labels_dict = None
        print("  不使用结局条件")
    
    # 3. 标准化数据
    print("\n[3] 数据标准化...")
    fhr_data_normalized = robust_normalize(fhr_data)
    print(f"  数据范围: [{fhr_data_normalized.min():.3f}, {fhr_data_normalized.max():.3f}]")
    
    # 4. 创建数据集
    print("\n[4] 创建数据集...")
    dataset = EnhancedBalancedMultiOutcomeFHRQualityDataset(
        fhr_data_normalized,
        labels_dict=labels_dict,
        use_sliding_window=Config.USE_SLIDING_WINDOW,
        positive_aug_factor=Config.POSITIVE_AUGMENT_FACTOR,
        negative_aug_factor=Config.NEGATIVE_AUGMENT_FACTOR,
        use_advanced_augmentation=True,
        use_outcome_condition=Config.USE_OUTCOME_CONDITION
    )
    
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # 确定信号长度
    signal_length = Config.WINDOW_SIZE if Config.USE_SLIDING_WINDOW else Config.SIGNAL_LENGTH
    num_outcomes = len(Config.ALL_OUTCOMES) if Config.USE_OUTCOME_CONDITION else 0
    
    # 5. 初始化增强器
    print("\n[5] 初始化FHR质量增强器...")
    enhancer = FHRQualityEnhancer(
        signal_length=signal_length,
        noise_dim=Config.NOISE_DIM,
        feature_dim=3,
        num_outcomes=num_outcomes,
        use_autoencoder_pretrain=Config.USE_AUTOENCODER_PRETRAIN,
        use_outcome_condition=Config.USE_OUTCOME_CONDITION,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 6. 训练
    print("\n[6] 开始训练...")
    print(f"  训练配置:")
    print(f"    - 结局条件: {'开启' if Config.USE_OUTCOME_CONDITION else '关闭'}")
    print(f"    - 自编码器预训练: {'开启' if Config.USE_AUTOENCODER_PRETRAIN else '关闭'}")
    print(f"    - 滑动窗口: {'开启' if Config.USE_SLIDING_WINDOW else '关闭'}")
    print(f"    - Epochs: {Config.NUM_EPOCHS}")
    
    # 训练自编码器
    if Config.USE_AUTOENCODER_PRETRAIN:
        enhancer.train_autoencoder(dataloader, epochs=min(50, Config.NUM_EPOCHS // 2))
    
    # 训练GAN
    enhancer.train_gan(dataloader, epochs=Config.NUM_EPOCHS)
    
    # 7. 生成样本示例
    print("\n[7] 生成样本示例...")
    if Config.USE_OUTCOME_CONDITION:
        for outcome in Config.ALL_OUTCOMES[:3]:  # 只演示前3个
            samples, labels = enhancer.generate_samples(outcome, num_samples=2, outcome_value=1)
            print(f"  生成 {outcome}+ 样本: {samples.shape}")
    else:
        samples, _ = enhancer.generate_samples(num_samples=5)
        print(f"  生成随机样本: {samples.shape}")
    
    # 8. 绘制训练进度
    print("\n[8] 生成训练报告...")
    enhancer.plot_training_progress()
    
    # 9. 保存最终模型
    print("\n[9] 保存最终模型...")
    enhancer._save_checkpoint("final")
    
    print("\n" + "=" * 60)
    print(f"训练完成！所有结果保存在: {enhancer.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()