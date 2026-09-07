"""
FHR-FSQI: 4Hz胎心监护信号质量筛查工具

基于FSQI (Fetal Signal Quality Index) 算法对4Hz胎心监护信号进行质量评估。
主要功能：
1. 信号质量检测（基于窗口统计特征）
2. 质量掩码生成（标记低质量片段）
3. 可视化质量报告（红绿对比图）
4. 批量处理与统计分析

FSQI算法原理：
- 滑动窗口计算信号质量
- 基于均值、最小值、最大值等统计特征
- 累积坏窗口计数生成质量掩码

"""

import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import argparse
from datetime import datetime

warnings.filterwarnings('ignore')


# ===================================================================
# 1. 配置模块
# ===================================================================

@dataclass
class FSQIConfig:
    """FSQI质量筛查配置"""
    # 信号参数
    fs: int = 4  # 采样频率 (Hz)
    window_size_seconds: int = 60  # 窗口大小 (秒)
    
    # 质量阈值
    bad_threshold: int = 50  # 坏窗口累计阈值
    good_ratio_threshold: float = 0.5  # 高质量样本判定阈值
    
    # 可视化参数
    plot_dpi: int = 150
    plot_figsize: tuple = (20, 5)
    fhr_ylim: tuple = (60, 210)
    fhr_normal_range: tuple = (110, 160)  # 正常胎心范围
    
    # 随机种子
    seed: int = 42
    
    @property
    def window_size(self) -> int:
        """窗口大小 (采样点数)"""
        return self.window_size_seconds * self.fs


@dataclass
class PathConfig:
    """路径配置"""
    # 输入: 原始FHR数据
    input_npz: str = "./fhr_all_samples.npz"
    
    # 输出: 筛查结果
    output_npz: str = "./fhr_fsqi_quality_result.npz"
    
    # 输出: 可视化图片目录
    plot_dir: str = "./fhr_quality_plots/"
    
    def create_dirs(self):
        """创建所有输出目录"""
        Path(self.plot_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_npz).parent.mkdir(parents=True, exist_ok=True)


# ===================================================================
# 2. 质量检测模块
# ===================================================================

class FHRQualityDetector:
    """
    FHR信号质量检测器
    
    基于FSQI算法检测信号质量，使用滑动窗口统计特征。
    """
    
    def __init__(self, config: FSQIConfig):
        """
        Args:
            config: FSQI配置对象
        """
        self.config = config
    
    def detect_window_quality(self, window: np.ndarray) -> int:
        """
        检测单个窗口的信号质量
        
        FSQI判定规则：
        1. 均值在 [80, 200] 范围内
        2. 最小值 >= 60
        3. 最大值/最小值 <= 1.5
        
        Args:
            window: 窗口信号数据
        
        Returns:
            1: 高质量, 0: 低质量
        """
        # 去除NaN值
        valid = window[~np.isnan(window)]
        
        # 全为NaN视为低质量
        if len(valid) == 0:
            return 0
        
        # 计算统计特征
        mean_val = np.mean(valid)
        min_val = np.min(valid)
        max_val = np.max(valid)
        
        # FSQI判定条件
        conditions = [
            mean_val <= 200 and mean_val >= 80,  # 均值范围
            min_val >= 60,                       # 最小值阈值
            max_val / min_val <= 1.5             # 变异范围
        ]
        
        return 1 if all(conditions) else 0
    
    def process_signal(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理单个信号，生成质量掩码
        
        Args:
            signal: 原始FHR信号 (一维数组)
        
        Returns:
            (处理后的信号, 质量掩码)
            - 处理后的信号: NaN标记为10的信号
            - 质量掩码: True=高质量, False=低质量
        """
        # 复制并展平
        hr = signal.ravel().copy()
        seq_len = len(hr)
        
        # 将10标记为NaN (与原数据格式兼容)
        hr[hr == 10] = np.nan
        
        # 初始化坏窗口计数器
        bad_counter = np.zeros(seq_len, dtype=int)
        window_size = self.config.window_size
        step = self.config.fs  # 滑动步长 = 采样频率
        
        # 滑动窗口检测
        for start_idx in range(0, seq_len - window_size + 1, step):
            window = hr[start_idx:start_idx + window_size]
            quality = self.detect_window_quality(window)
            
            if quality == 0:
                # 坏窗口，计数器加1
                bad_counter[start_idx:start_idx + window_size] += 1
        
        # 生成质量掩码: 坏窗口累计次数 <= 阈值
        quality_mask = bad_counter <= self.config.bad_threshold
        
        return hr, quality_mask
    
    def classify_sample(self, quality_mask: np.ndarray) -> Tuple[str, float]:
        """
        根据质量掩码分类样本
        
        Args:
            quality_mask: 质量掩码
        
        Returns:
            (类别, 高质量比例)
        """
        good_ratio = np.mean(quality_mask)
        category = "good" if good_ratio >= self.config.good_ratio_threshold else "bad"
        return category, good_ratio


# ===================================================================
# 3. 可视化模块
# ===================================================================

class FHRVisualizer:
    """FHR信号可视化工具"""
    
    def __init__(self, config: FSQIConfig, output_dir: str):
        """
        Args:
            config: FSQI配置
            output_dir: 输出目录
        """
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_quality(self, signal: np.ndarray, mask: np.ndarray, 
                     filename: str, dpi: int = 150) -> str:
        """
        绘制信号质量可视化图
        
        Args:
            signal: 原始信号
            mask: 质量掩码
            filename: 文件名
            dpi: 图片分辨率
        
        Returns:
            保存的文件路径
        """
        seq_len = len(signal)
        fs = self.config.fs
        
        # 时间轴 (分钟)
        time_axis = np.arange(seq_len) / fs / 60
        
        # 创建图形
        plt.figure(figsize=self.config.plot_figsize)
        
        # 绘制信号
        plt.plot(time_axis, signal, c='k', linewidth=0.8, label='FHR信号')
        
        # 绘制正常范围区域
        plt.axhspan(
            self.config.fhr_normal_range[0], 
            self.config.fhr_normal_range[1],
            facecolor='green', alpha=0.3, label='正常范围'
        )
        
        # 标记低质量区域
        bad_mask = ~mask
        if bad_mask.any():
            # 将连续的低质量区域聚合并标记
            for i in range(seq_len):
                if bad_mask[i]:
                    # 绘制竖条
                    plt.axvspan(
                        time_axis[i], 
                        time_axis[i] + 1/fs/60,
                        facecolor='red', alpha=0.2
                    )
        
        # 设置样式
        plt.grid(True, color='#FF8C00', linewidth=0.5, alpha=0.5)
        plt.xlabel('时间 (分钟)', fontsize=12)
        plt.ylabel('胎心率 (bpm)', fontsize=12)
        plt.ylim(self.config.fhr_ylim[0], self.config.fhr_ylim[1])
        
        # 计算质量指标
        good_ratio = np.mean(mask)
        quality_text = f"高质量: {good_ratio*100:.1f}%"
        plt.title(f"FSQI质量筛查: {filename} | {quality_text}", fontsize=14)
        
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_summary(self, results: Dict[str, List]) -> str:
        """
        绘制统计汇总图
        
        Args:
            results: 处理结果字典
        
        Returns:
            保存的文件路径
        """
        # 提取统计信息
        good_ratios = results.get('good_ratios', [])
        categories = results.get('categories', [])
        
        if not good_ratios:
            return ""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 高质量比例分布直方图
        axes[0].hist(good_ratios, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='阈值 (50%)')
        axes[0].set_xlabel('高质量比例', fontsize=12)
        axes[0].set_ylabel('样本数', fontsize=12)
        axes[0].set_title('信号质量分布')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 分类统计
        good_count = sum(1 for c in categories if c == 'good')
        bad_count = sum(1 for c in categories if c == 'bad')
        labels = ['高质量', '低质量']
        counts = [good_count, bad_count]
        colors = ['#2ecc71', '#e74c3c']
        
        axes[1].pie(counts, labels=labels, autopct='%1.1f%%', 
                    colors=colors, startangle=90, explode=(0.05, 0.05))
        axes[1].set_title(f'样本分类 (总数: {len(categories)})')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "summary_statistics.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path


# ===================================================================
# 4. 主处理流程
# ===================================================================

class FSQIDataProcessor:
    """FSQI数据处理器"""
    
    def __init__(self, config: FSQIConfig, path_config: PathConfig):
        """
        Args:
            config: FSQI配置
            path_config: 路径配置
        """
        self.config = config
        self.path_config = path_config
        self.detector = FHRQualityDetector(config)
        self.visualizer = FHRVisualizer(config, path_config.plot_dir)
        
        # 结果存储
        self.results = {
            'filenames': [],
            'signals': [],
            'quality_masks': [],
            'categories': [],
            'good_ratios': []
        }
    
    def load_data(self) -> Tuple[List[str], List[np.ndarray]]:
        """
        加载数据
        
        Returns:
            (文件名列表, 信号列表)
        """
        print(f"正在加载数据: {self.path_config.input_npz}")
        
        with np.load(self.path_config.input_npz, allow_pickle=True) as data:
            filenames = data.files
            signals = [data[name].astype(np.float64) for name in filenames]
        
        print(f"已加载 {len(signals)} 个样本")
        return filenames, signals
    
    def process_all(self):
        """处理所有样本"""
        print("\n" + "=" * 60)
        print("4Hz FSQI 信号质量筛查系统")
        print("=" * 60)
        
        # 1. 加载数据
        filenames, signals = self.load_data()
        
        # 2. 处理每个样本
        print("\n开始质量筛查...")
        bad_samples = []
        
        for idx, fname in enumerate(tqdm(filenames, desc="处理进度")):
            signal = signals[idx]
            
            # 质量检测
            processed_signal, quality_mask = self.detector.process_signal(signal)
            category, good_ratio = self.detector.classify_sample(quality_mask)
            
            # 记录结果
            self.results['filenames'].append(fname)
            self.results['signals'].append(processed_signal)
            self.results['quality_masks'].append(quality_mask)
            self.results['categories'].append(category)
            self.results['good_ratios'].append(good_ratio)
            
            if category == 'bad':
                bad_samples.append(fname)
            
            # 生成可视化
            self.visualizer.plot_quality(
                processed_signal, 
                quality_mask, 
                fname,
                dpi=self.config.plot_dpi
            )
        
        # 3. 统计结果
        self._print_statistics(bad_samples)
        
        # 4. 保存结果
        self._save_results()
        
        # 5. 生成汇总图
        self.visualizer.plot_summary(self.results)
        
        print("\n" + "=" * 60)
        print("✅ FSQI质量筛查完成!")
        print(f"📊 结果保存: {self.path_config.output_npz}")
        print(f"🖼️  图片目录: {self.path_config.plot_dir}")
        print("=" * 60)
    
    def _save_results(self):
        """保存结果到npz文件"""
        np.savez_compressed(
            self.path_config.output_npz,
            filenames=np.array(self.results['filenames'], dtype=object),
            fhr=np.array(self.results['signals'], dtype=object),
            quality_mask=np.array(self.results['quality_masks'], dtype=object),
            fs=np.array(self.config.fs),
            window_size=np.array(self.config.window_size),
            threshold=np.array(self.config.bad_threshold),
            metadata=np.array({
                'total_samples': len(self.results['filenames']),
                'good_count': sum(1 for c in self.results['categories'] if c == 'good'),
                'bad_count': sum(1 for c in self.results['categories'] if c == 'bad'),
                'fs': self.config.fs,
                'window_size_seconds': self.config.window_size_seconds,
                'timestamp': datetime.now().isoformat()
            }, dtype=object)
        )
        print(f"✅ 结果已保存: {self.path_config.output_npz}")
    
    def _print_statistics(self, bad_samples: List[str]):
        """打印统计信息"""
        total = len(self.results['filenames'])
        good_count = sum(1 for c in self.results['categories'] if c == 'good')
        bad_count = len(bad_samples)
        
        print("\n" + "=" * 60)
        print("           4Hz FHR 信号质量筛查结果")
        print("=" * 60)
        print(f"总样本数        : {total}")
        print(f"高质量信号      : {good_count} ({good_count/total*100:.1f}%)")
        print(f"低质量信号      : {bad_count} ({bad_count/total*100:.1f}%)")
        print("=" * 60)
        
        if bad_samples:
            print("\n低质量信号文件列表 (前10个):")
            for f in bad_samples[:10]:
                print(f"  ❌ {f}")
            if len(bad_samples) > 10:
                print(f"  ... 还有 {len(bad_samples) - 10} 个")
        
        # 计算平均质量
        avg_ratio = np.mean(self.results['good_ratios'])
        print(f"\n平均高质量比例: {avg_ratio*100:.1f}%")


# ===================================================================
# 5. 命令行接口
# ===================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='FHR-FSQI: 4Hz胎心监护信号质量筛查工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用
  python fhr_fsqi.py --input data.npz --output result.npz
  
  # 自定义参数
  python fhr_fsqi.py --input data.npz --fs 4 --window 60 --threshold 50
        """
    )
    
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='输入npz文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出npz文件路径')
    parser.add_argument('--plot_dir', '-p', type=str, default=None,
                        help='可视化图片输出目录')
    parser.add_argument('--fs', type=int, default=4,
                        help='采样频率 (Hz), 默认: 4')
    parser.add_argument('--window', type=int, default=60,
                        help='窗口大小 (秒), 默认: 60')
    parser.add_argument('--threshold', '-t', type=int, default=50,
                        help='坏窗口累计阈值, 默认: 50')
    parser.add_argument('--good_ratio', type=float, default=0.5,
                        help='高质量样本判定阈值, 默认: 0.5')
    parser.add_argument('--dpi', type=int, default=150,
                        help='图片分辨率, 默认: 150')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子, 默认: 42')
    
    return parser.parse_args()


def setup_from_args(args):
    """从命令行参数更新配置"""
    config = FSQIConfig(
        fs=args.fs,
        window_size_seconds=args.window,
        bad_threshold=args.threshold,
        good_ratio_threshold=args.good_ratio,
        plot_dpi=args.dpi,
        seed=args.seed
    )
    
    path_config = PathConfig()
    if args.input:
        path_config.input_npz = args.input
    if args.output:
        path_config.output_npz = args.output
    if args.plot_dir:
        path_config.plot_dir = args.plot_dir
    
    return config, path_config


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    config, path_config = setup_from_args(args)
    
    # 设置随机种子
    np.random.seed(config.seed)
    
    # 创建输出目录
    path_config.create_dirs()
    
    # 创建处理器并执行
    processor = FSQIDataProcessor(config, path_config)
    processor.process_all()


if __name__ == "__main__":
    main()