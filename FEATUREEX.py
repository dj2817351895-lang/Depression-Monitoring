"""
多模态特征提取系统（无torchaudio版本）
架构：
[音频分支] MFCC+F0提取 → 病理特征滤波 → 热图生成 → EfficientNet-B0 → 音频特征
[文本分支] ERNIE 3.0编码 → 抑郁词典增强 → 分层BiLSTM-Attention → 文本特征
"""

import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
import pickle
import json
import re
import time
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==============================================================================
# 配置文件
# ==============================================================================
class Config:
    # 路径配置
    TRAIN_DATA_PATH = r"D:\Work(paper3)\EATD-AUG-WINDOWS\train"
    VAL_DATA_PATH = r"D:\Work(paper3)\EATD-AUG-WINDOWS\validation"
    FEATURE_SAVE_DIR = r"D:\Work(paper3)\EATD-Features-NoTorchaudio"

    # 音频参数
    SAMPLE_RATE = 16000
    N_MFCC = 13  # 原始MFCC维度
    N_MELS = 40  # Mel滤波器数量
    HOP_LENGTH = 256
    N_FFT = 2048
    F0_MIN = 80
    F0_MAX = 400

    # 热图参数
    HEATMAP_SIZE = 224  # EfficientNet-B0输入尺寸

    # 模型参数
    AUDIO_FEATURE_DIM = 1280  # EfficientNet-B0输出维度
    TEXT_FEATURE_DIM = 512    # 文本特征输出维度
    ERNIE_HIDDEN_SIZE = 768   # ERNIE 3.0隐藏层大小
    BILSTM_HIDDEN_SIZE = 256

    # 处理参数
    MAX_AUDIO_LENGTH = 10  # 最大音频长度（秒）
    MAX_TEXT_LENGTH = 512  # 最大文本长度

    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        os.makedirs(cls.FEATURE_SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.FEATURE_SAVE_DIR, "train"), exist_ok=True)
        os.makedirs(os.path.join(cls.FEATURE_SAVE_DIR, "validation"), exist_ok=True)
        os.makedirs(os.path.join(cls.FEATURE_SAVE_DIR, "heatmaps"), exist_ok=True)
        os.makedirs(os.path.join(cls.FEATURE_SAVE_DIR, "logs"), exist_ok=True)

# ==============================================================================
# 音频特征提取器（无torchaudio版本）
# ==============================================================================
class NoTorchaudioAudioFeatureExtractor:
    """
    音频特征提取器 - 完全不使用torchaudio
    """

    def __init__(self, config):
        self.config = config
        self.device = device

        # 加载EfficientNet-B0（不使用torchvision，使用纯PyTorch实现）
        self.efficientnet = self._load_efficientnet_b0_pure()

        # 抑郁症病理特征滤波器参数
        self.depression_params = {
            'f0_smoothing_sigma': 3.0,    # 基频平滑系数
            'mfcc_low_freq_boost': 1.2,   # 低频增强
            'mfcc_high_freq_suppress': 0.9, # 高频抑制
        }

    def _load_efficientnet_b0_pure(self):
        """纯PyTorch实现的EfficientNet-B0（不依赖torchvision）"""

        class MBConvBlock(nn.Module):
            """Mobile Inverted Residual Bottleneck Block"""
            def __init__(self, in_channels, out_channels, expansion_factor, stride):
                super().__init__()
                self.use_residual = (stride == 1 and in_channels == out_channels)
                hidden_dim = in_channels * expansion_factor

                layers = []
                if expansion_factor != 1:
                    layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
                    layers.append(nn.BatchNorm2d(hidden_dim))
                    layers.append(nn.ReLU6(inplace=True))

                layers.extend([
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels)
                ])

                self.conv = nn.Sequential(*layers)

            def forward(self, x):
                if self.use_residual:
                    return x + self.conv(x)
                return self.conv(x)

        class EfficientNetB0(nn.Module):
            def __init__(self):
                super().__init__()

                # 初始卷积层
                self.features = nn.Sequential(
                    # Conv3x3
                    nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU6(inplace=True),

                    # MBConv1, 3x3
                    MBConvBlock(32, 16, 1, 1),

                    # MBConv6, 3x3
                    MBConvBlock(16, 24, 6, 2),
                    MBConvBlock(24, 24, 6, 1),

                    # MBConv6, 5x5
                    MBConvBlock(24, 40, 6, 2),
                    MBConvBlock(40, 40, 6, 1),

                    # MBConv6, 3x3
                    MBConvBlock(40, 80, 6, 2),
                    MBConvBlock(80, 80, 6, 1),
                    MBConvBlock(80, 80, 6, 1),

                    # MBConv6, 5x5
                    MBConvBlock(80, 112, 6, 1),
                    MBConvBlock(112, 112, 6, 1),
                    MBConvBlock(112, 112, 6, 1),

                    # MBConv6, 5x5
                    MBConvBlock(112, 192, 6, 2),
                    MBConvBlock(192, 192, 6, 1),
                    MBConvBlock(192, 192, 6, 1),
                    MBConvBlock(192, 192, 6, 1),

                    # MBConv6, 3x3
                    MBConvBlock(192, 320, 6, 1),

                    # Final layers
                    nn.Conv2d(320, 1280, 1, bias=False),
                    nn.BatchNorm2d(1280),
                    nn.ReLU6(inplace=True),
                )

                # 全局平均池化
                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.output_dim = 1280

            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                return x.flatten(1)

        model = EfficientNetB0().to(self.device)
        model.eval()
        return model

    def extract_mfcc_and_f0(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用librosa提取MFCC和F0特征
        返回: (mfcc_features, f0)
        """
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE)

            # 静音去除
            if len(audio) > sr * 0.1:
                audio, _ = librosa.effects.trim(audio, top_db=20)

            # 限制音频长度
            max_samples = int(self.config.MAX_AUDIO_LENGTH * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # 1. 提取Mel频谱
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=self.config.N_FFT,
                hop_length=self.config.HOP_LENGTH,
                n_mels=self.config.N_MELS,
                fmin=0,
                fmax=sr/2
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # 2. 从Mel频谱计算MFCC
            mfcc = librosa.feature.mfcc(
                S=mel_spec_db,
                n_mfcc=self.config.N_MFCC,
                dct_type=2,
                norm='ortho'
            )

            # 3. 添加delta和delta-delta
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            # 组合MFCC特征 [39, T]
            mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

            # 4. 提取基频F0（使用pyin算法）
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y=audio,
                fmin=self.config.F0_MIN,
                fmax=self.config.F0_MAX,
                sr=sr,
                hop_length=self.config.HOP_LENGTH,
                fill_na=0.0
            )

            # 确保F0与MFCC时间维度对齐
            if len(f0) != mfcc_features.shape[1]:
                # 线性插值对齐
                from scipy import interpolate
                x_orig = np.linspace(0, 1, len(f0))
                x_new = np.linspace(0, 1, mfcc_features.shape[1])
                f = interpolate.interp1d(x_orig, f0, kind='linear', fill_value='extrapolate')
                f0 = f(x_new)

            return mfcc_features, f0

        except Exception as e:
            print(f"MFCC/F0提取失败 {audio_path}: {e}")
            # 返回默认值
            T = 100
            mfcc_features = np.zeros((self.config.N_MFCC * 3, T))
            f0 = np.zeros(T)
            return mfcc_features, f0

    def extract_energy_and_spectral_features(self, audio: np.ndarray, sr: int) -> Dict:
        """提取能量和频谱特征"""
        features = {}

        # 能量特征
        features['energy'] = librosa.feature.rms(y=audio, hop_length=self.config.HOP_LENGTH)[0]

        # 频谱质心
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=self.config.HOP_LENGTH)[0]

        # 频谱带宽
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, hop_length=self.config.HOP_LENGTH)[0]

        # 频谱对比度
        features['spectral_contrast'] = librosa.feature.spectral_contrast(
            y=audio, sr=sr, hop_length=self.config.HOP_LENGTH)

        return features

    def apply_depression_filter(self, mfcc: np.ndarray, f0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用抑郁症病理特征滤波
        基于临床研究：抑郁患者语音特征
        """
        mfcc_filtered = mfcc.copy()
        f0_filtered = f0.copy()

        # 1. 基频平滑（抑郁患者语调单调）
        if len(f0_filtered) > 10:
            from scipy.ndimage import gaussian_filter1d
            f0_filtered = gaussian_filter1d(
                f0_filtered,
                sigma=self.depression_params['f0_smoothing_sigma']
            )

        # 2. MFCC频率调整
        # 低频（前13个MFCC系数）增强
        low_freq_end = min(13, mfcc_filtered.shape[0])
        mfcc_filtered[:low_freq_end] *= self.depression_params['mfcc_low_freq_boost']

        # 高频抑制（后26个系数）
        if mfcc_filtered.shape[0] > 13:
            mfcc_filtered[13:] *= self.depression_params['mfcc_high_freq_suppress']

        # 3. 抑郁相关特征增强
        # 增强停顿特征（抑郁患者停顿增多）
        energy = np.sqrt(np.sum(mfcc_filtered ** 2, axis=0))
        energy_threshold = np.percentile(energy, 30)  # 低能量阈值
        pause_mask = (energy < energy_threshold).astype(float)
        pause_mask = gaussian_filter1d(pause_mask, sigma=1.0)

        # 应用停顿增强
        mfcc_filtered = mfcc_filtered * (1.0 + 0.3 * pause_mask[np.newaxis, :])

        return mfcc_filtered, f0_filtered

    def create_heatmap(self, mfcc: np.ndarray, f0: np.ndarray) -> np.ndarray:
        """
        创建2D热图用于EfficientNet-B0输入
        返回: [224, 224] 的灰度热图
        """
        # 确保MFCC维度正确
        if mfcc.shape[0] != self.config.N_MFCC * 3:
            if mfcc.shape[0] > self.config.N_MFCC * 3:
                mfcc = mfcc[:self.config.N_MFCC * 3, :]
            else:
                padding = self.config.N_MFCC * 3 - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, padding), (0, 0)), mode='constant')

        # 创建特征矩阵：[40, T]，其中39个MFCC特征 + 1个F0特征
        feature_matrix = np.vstack([mfcc, f0.reshape(1, -1)])

        # 对数变换增强对比度
        feature_matrix = np.log1p(np.abs(feature_matrix))

        # 归一化到[0, 1]
        min_val = feature_matrix.min(axis=1, keepdims=True)
        max_val = feature_matrix.max(axis=1, keepdims=True)
        range_val = max_val - min_val + 1e-8

        normalized = (feature_matrix - min_val) / range_val

        # 调整大小到[224, 224]
        from scipy.ndimage import zoom
        target_size = self.config.HEATMAP_SIZE

        # 计算缩放因子
        scale_h = target_size / normalized.shape[0]
        scale_w = target_size / normalized.shape[1]

        # 使用双线性插值
        heatmap = zoom(normalized, (scale_h, scale_w), order=1)

        # 确保尺寸正确
        if heatmap.shape != (target_size, target_size):
            # 裁剪或填充
            heatmap = heatmap[:target_size, :target_size]
            if heatmap.shape != (target_size, target_size):
                padding_h = target_size - heatmap.shape[0]
                padding_w = target_size - heatmap.shape[1]
                heatmap = np.pad(heatmap, ((0, padding_h), (0, padding_w)), mode='constant')

        return heatmap

    def extract_audio_features(self, audio_path: str, save_heatmap: bool = False) -> np.ndarray:
        """
        完整的音频特征提取流程
        返回: [1280,] EfficientNet-B0特征向量
        """
        try:
            # 1. 提取MFCC和F0
            mfcc, f0 = self.extract_mfcc_and_f0(audio_path)

            # 2. 应用抑郁症病理特征滤波
            mfcc_filtered, f0_filtered = self.apply_depression_filter(mfcc, f0)

            # 3. 创建2D热图
            heatmap = self.create_heatmap(mfcc_filtered, f0_filtered)

            # 4. 保存热图（可选）
            if save_heatmap:
                heatmap_dir = os.path.join(self.config.FEATURE_SAVE_DIR, "heatmaps")
                os.makedirs(heatmap_dir, exist_ok=True)
                filename = os.path.basename(audio_path).replace('.wav', '_heatmap.png')
                heatmap_path = os.path.join(heatmap_dir, filename)

                # 保存为图像
                plt.figure(figsize=(8, 6))
                plt.imshow(heatmap, cmap='viridis', aspect='auto')
                plt.colorbar(label='Normalized Feature Value')
                plt.title('Audio Heatmap (MFCC+F0)')
                plt.xlabel('Time Frame')
                plt.ylabel('Feature Dimension')
                plt.tight_layout()
                plt.savefig(heatmap_path, dpi=100)
                plt.close()

            # 5. 通过EfficientNet-B0提取特征
            # 转换为tensor [1, 1, 224, 224]
            heatmap_tensor = torch.FloatTensor(heatmap).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.efficientnet(heatmap_tensor)

            # 转换为numpy
            audio_features = features.cpu().numpy().flatten()

            return audio_features

        except Exception as e:
            print(f"音频特征提取失败 {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            # 返回零向量
            return np.zeros(self.config.AUDIO_FEATURE_DIM)

# ==============================================================================
# 文本特征提取器（无torchaudio版本）
# ==============================================================================
class NoTorchaudioTextFeatureExtractor:
    """
    文本特征提取器 - 完全不使用torchaudio
    """

    def __init__(self, config):
        self.config = config
        self.device = device

        # 加载ERNIE 3.0模型
        self.tokenizer, self.ernie_model = self._load_ernie_3_0()

        # 抑郁词典增强器
        self.lexicon_enhancer = DepressionLexiconEnhancer()

        # 分层BiLSTM-Attention网络
        self.hierarchical_bilstm = HierarchicalBiLSTMAttention(
            input_size=self.config.ERNIE_HIDDEN_SIZE,
            hidden_size=self.config.BILSTM_HIDDEN_SIZE,
            output_size=self.config.TEXT_FEATURE_DIM
        ).to(self.device)

    def _load_ernie_3_0(self):
        """加载ERNIE 3.0模型"""
        try:
            from transformers import AutoTokenizer, AutoModel

            print("加载ERNIE 3.0模型...")

            # 使用ERNIE 3.0中文模型
            model_name = "nghuyong/ernie-3.0-base-zh"

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

            model = model.to(self.device)
            model.eval()

            print("ERNIE 3.0模型加载完成")
            return tokenizer, model

        except ImportError:
            print("错误: transformers库未安装")
            print("请运行: pip install transformers")
            raise
        except Exception as e:
            print(f"ERNIE 3.0加载失败: {e}")
            print("尝试使用BERT作为替代...")
            return self._load_bert_as_fallback()

    def _load_bert_as_fallback(self):
        """加载BERT作为备用"""
        try:
            from transformers import BertTokenizer, BertModel

            print("加载BERT模型作为替代...")

            model_name = "bert-base-chinese"
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)

            model = model.to(self.device)
            model.eval()

            print("BERT模型加载完成")
            return tokenizer, model
        except Exception as e:
            print(f"BERT加载失败: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if not text:
            return ""

        # 清理文本
        text = re.sub(r'\s+', ' ', text)  # 去除多余空格
        text = text.strip()

        # 移除特殊字符（保留中文标点）
        text = re.sub(r'[^\u4e00-\u9fa5，。！？；：""''、\s\w]', '', text)

        # 截断过长的文本
        max_length = self.config.MAX_TEXT_LENGTH
        if len(text) > max_length:
            text = text[:max_length]

        return text

    def segment_text(self, text: str) -> List[str]:
        """文本分段（处理长文本）"""
        if len(text) <= 100:
            return [text]

        # 使用标点符号进行分割
        sentences = re.split(r'[。！？；]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 合并短句子
        segments = []
        current_segment = ""

        for sentence in sentences:
            if len(current_segment) + len(sentence) + 1 <= 100:
                if current_segment:
                    current_segment += "。" + sentence
                else:
                    current_segment = sentence
            else:
                if current_segment:
                    segments.append(current_segment)
                current_segment = sentence

        if current_segment:
            segments.append(current_segment)

        return segments if segments else [text[:100]]

    def encode_with_ernie(self, text: str) -> torch.Tensor:
        """使用ERNIE 3.0编码文本"""
        try:
            # 分词和编码
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding='max_length',
                max_length=self.config.MAX_TEXT_LENGTH
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.ernie_model(**inputs)
                # 使用最后一层隐藏状态
                hidden_states = outputs.last_hidden_state  # [1, seq_len, hidden_size]

            return hidden_states

        except Exception as e:
            print(f"ERNIE编码失败: {e}")
            # 返回零向量
            return torch.zeros((1, self.config.MAX_TEXT_LENGTH, self.config.ERNIE_HIDDEN_SIZE)).to(self.device)

    def extract_text_features(self, text_path: str) -> np.ndarray:
        """
        完整的文本特征提取流程
        返回: [512,] 文本特征向量
        """
        try:
            # 1. 读取文本
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            if not text:
                return np.zeros(self.config.TEXT_FEATURE_DIM)

            # 2. 文本预处理
            text = self.preprocess_text(text)

            # 3. 文本分段（处理长文本）
            segments = self.segment_text(text)

            # 4. 处理每个文本段
            segment_features = []

            for segment in segments:
                # ERNIE 3.0编码
                ernie_features = self.encode_with_ernie(segment)  # [1, seq_len, hidden_size]

                # 获取tokens用于词典增强
                tokens = self.tokenizer.tokenize(segment)
                if len(tokens) > ernie_features.shape[1]:
                    tokens = tokens[:ernie_features.shape[1]]

                # 应用抑郁词典增强
                enhanced_features = self.lexicon_enhancer.enhance_features(ernie_features, tokens)

                segment_features.append(enhanced_features)

            # 5. 如果有多段，合并特征
            if len(segment_features) > 1:
                # 将所有段特征连接
                combined_features = torch.cat(segment_features, dim=1)
            else:
                combined_features = segment_features[0]

            # 6. 分层BiLSTM-Attention
            with torch.no_grad():
                text_features = self.hierarchical_bilstm(combined_features)

            # 转换为numpy
            text_features = text_features.cpu().numpy().flatten()

            return text_features

        except Exception as e:
            print(f"文本特征提取失败 {text_path}: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(self.config.TEXT_FEATURE_DIM)

class DepressionLexiconEnhancer:
    """抑郁词典增强器"""

    def __init__(self):
        # 抑郁症核心词汇库（带权重）
        self.lexicon = {
            # 情绪状态（高权重）
            "绝望": 3.0, "无助": 3.0, "悲伤": 2.5, "痛苦": 3.0, "抑郁": 3.0,
            "忧郁": 2.5, "沮丧": 2.5, "低落": 2.0, "失落": 2.0, "孤独": 2.5,
            "寂寞": 2.5, "空虚": 2.0, "麻木": 2.0, "崩溃": 3.0, "难过": 2.0,
            "伤心": 2.0, "悲痛": 3.0, "哀伤": 2.5, "郁闷": 2.0, "消沉": 2.0,

            # 自我认知
            "自责": 2.5, "内疚": 2.5, "愧疚": 2.5, "自卑": 2.0, "无用": 2.5,
            "无能": 2.5, "无价值": 2.5, "失败": 2.0, "失败者": 2.5, "废物": 3.0,
            "负担": 2.5, "累赘": 2.5, "拖累": 2.5, "没用": 2.0, "不配": 2.5,

            # 生理症状
            "失眠": 2.0, "睡不着": 2.0, "入睡困难": 2.0, "早醒": 2.0, "噩梦": 2.0,
            "疲劳": 1.5, "乏力": 1.5, "倦怠": 1.5, "疲惫": 1.5, "劳累": 1.5,
            "食欲不振": 2.0, "没胃口": 2.0, "头痛": 1.5, "头晕": 1.5, "背痛": 1.5,

            # 认知症状
            "注意力不集中": 2.0, "注意力分散": 2.0, "记忆力下降": 2.0, "记性差": 2.0,
            "犹豫不决": 2.0, "纠结": 2.0, "思考困难": 2.0, "思维迟缓": 2.0,
            "反应迟钝": 2.0, "无法思考": 2.5, "脑子空白": 2.5,

            # 自杀相关（最高权重）
            "自杀": 4.0, "自残": 4.0, "轻生": 4.0, "不想活": 4.0, "活着没意思": 4.0,
            "解脱": 3.5, "死亡": 3.5, "死了": 3.5, "离开": 3.0, "结束": 3.0,
        }

    def enhance_features(self, features: torch.Tensor, tokens: List[str]) -> torch.Tensor:
        """
        应用抑郁词典增强
        为抑郁相关词汇分配更高的注意力权重
        """
        if features.shape[1] <= 2:  # 只有[CLS]和[SEP]
            return features

        # 创建注意力权重
        seq_len = min(len(tokens), features.shape[1] - 2)  # 减去[CLS]和[SEP]
        attention_weights = torch.ones(features.shape[1]).to(features.device)

        # 为抑郁词汇分配更高权重（从第2个位置开始，跳过[CLS]）
        for i, token in enumerate(tokens[:seq_len]):
            # 检查完全匹配
            if token in self.lexicon:
                weight = self.lexicon[token]
                attention_weights[i+1] = weight  # +1跳过[CLS]
            else:
                # 检查部分匹配
                for dep_word, weight in self.lexicon.items():
                    if dep_word in token:
                        attention_weights[i+1] = weight * 0.8  # 部分匹配权重较低
                        break

        # 归一化权重（保持平均权重为1）
        attention_weights = attention_weights / attention_weights.mean()

        # 应用权重
        enhanced_features = features * attention_weights.view(1, -1, 1)

        return enhanced_features

class HierarchicalBiLSTMAttention(nn.Module):
    """
    分层BiLSTM-Attention网络
    """

    def __init__(self, input_size=768, hidden_size=256, output_size=512, num_layers=2):
        super().__init__()

        # 第一层：BiLSTM
        self.bilstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        # 第一层注意力
        self.attention1 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # 第二层：BiLSTM（处理注意力加权后的特征）
        self.bilstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        # 第二层注意力
        self.attention2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        x: [batch_size, seq_len, input_size]
        返回: [batch_size, output_size]
        """
        batch_size = x.shape[0]

        # 第一层BiLSTM
        lstm1_out, _ = self.bilstm1(x)  # [batch, seq_len, hidden*2]

        # 第一层注意力
        attn_weights1 = self.attention1(lstm1_out)  # [batch, seq_len, 1]
        weighted1 = torch.sum(lstm1_out * attn_weights1, dim=1)  # [batch, hidden*2]

        # 第二层BiLSTM（将加权向量视为序列）
        lstm2_input = weighted1.unsqueeze(1)  # [batch, 1, hidden*2]
        lstm2_out, _ = self.bilstm2(lstm2_input)  # [batch, 1, hidden*2]

        # 第二层注意力
        attn_weights2 = self.attention2(lstm2_out)  # [batch, 1, 1]
        weighted2 = torch.sum(lstm2_out * attn_weights2, dim=1)  # [batch, hidden*2]

        # 输出层
        output = self.output_layer(weighted2)  # [batch, output_size]

        return output

# ==============================================================================
# 多模态数据集
# ==============================================================================
class NoTorchaudioMultimodalDataset(Dataset):
    """不使用torchaudio的多模态数据集"""

    def __init__(self, data_dir, config, split='train', extract_features=True):
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        self.extract_features = extract_features

        # 特征提取器
        self.audio_extractor = NoTorchaudioAudioFeatureExtractor(config)
        self.text_extractor = NoTorchaudioTextFeatureExtractor(config)

        # 特征保存路径
        self.feature_dir = Path(config.FEATURE_SAVE_DIR) / split

        # 加载样本
        self.samples = self._load_samples()

        # 提取或加载特征
        if extract_features:
            self._extract_and_save_features()

    def _load_samples(self):
        """加载数据样本"""
        samples = []

        if not self.data_dir.exists():
            print(f"错误: 数据目录不存在 {self.data_dir}")
            return samples

        # 遍历所有文件夹
        folders = [f for f in self.data_dir.iterdir() if f.is_dir()]
        print(f"在 {self.data_dir} 中找到 {len(folders)} 个文件夹")

        for folder in tqdm(folders, desc="扫描样本"):
            try:
                # 读取标签
                label = self._read_label(folder)
                if label is None:
                    continue

                # 检查三个情感片段
                emotions = ['positive', 'negative', 'neutral']
                has_all_files = True
                file_paths = {}

                for emotion in emotions:
                    # 音频文件
                    audio_file = folder / f"{emotion}.wav"
                    if not audio_file.exists():
                        audio_file = folder / f"{emotion}_out.wav"  # 兼容旧命名

                    # 文本文件
                    text_file = folder / f"{emotion}.txt"

                    if not audio_file.exists() or not text_file.exists():
                        has_all_files = False
                        break

                    file_paths[emotion] = {
                        'audio': str(audio_file),
                        'text': str(text_file)
                    }

                if has_all_files:
                    samples.append({
                        'id': folder.name,
                        'folder': str(folder),
                        'label': label,
                        'emotions': emotions,
                        'file_paths': file_paths
                    })
            except Exception as e:
                print(f"加载样本 {folder} 失败: {e}")
                continue

        print(f"成功加载 {len(samples)} 个有效样本")
        return samples

    def _read_label(self, folder):
        """读取标签（0:非抑郁, 1:抑郁）"""
        try:
            # 优先读取 new_label.txt
            label_file = folder / "new_label.txt"
            if not label_file.exists():
                label_file = folder / "label.txt"

            if label_file.exists():
                with open(label_file, 'r', encoding='utf-8') as f:
                    val = float(f.read().strip())
                    # 原始 label.txt 需要 * 1.25
                    if "new_label" not in str(label_file):
                        val *= 1.25
                    return 1 if val >= 53 else 0
        except:
            pass
        return None

    def _extract_and_save_features(self):
        """提取并保存特征"""
        print(f"\n开始提取 {self.split} 集特征...")

        # 确保特征目录存在
        self.feature_dir.mkdir(parents=True, exist_ok=True)

        # 创建进度条
        pbar = tqdm(self.samples, desc=f"提取{self.split}特征")

        for sample in pbar:
            feature_file = self.feature_dir / f"{sample['id']}_features.pkl"

            # 如果特征已存在，跳过
            if feature_file.exists():
                pbar.set_postfix({"状态": "已存在", "样本": sample['id']})
                continue

            try:
                # 提取每个情感片段的特征
                audio_features_list = []
                text_features_list = []

                for emotion in sample['emotions']:
                    file_paths = sample['file_paths'][emotion]

                    # 提取音频特征（只对第一个样本保存热图）
                    save_heatmap = (self.split == 'train' and sample == self.samples[0] and emotion == 'positive')
                    audio_features = self.audio_extractor.extract_audio_features(
                        file_paths['audio'],
                        save_heatmap=save_heatmap
                    )

                    # 提取文本特征
                    text_features = self.text_extractor.extract_text_features(
                        file_paths['text']
                    )

                    audio_features_list.append(audio_features)
                    text_features_list.append(text_features)

                # 合并三个情感片段的特征（平均）
                audio_features_combined = np.mean(audio_features_list, axis=0)
                text_features_combined = np.mean(text_features_list, axis=0)

                # 保存特征
                features = {
                    'audio_features': audio_features_combined,
                    'text_features': text_features_combined,
                    'label': sample['label'],
                    'id': sample['id']
                }

                with open(feature_file, 'wb') as f:
                    pickle.dump(features, f)

                pbar.set_postfix({"状态": "成功", "样本": sample['id']})

            except Exception as e:
                print(f"\n提取样本 {sample['id']} 特征失败: {e}")
                pbar.set_postfix({"状态": "失败", "样本": sample['id']})
                continue

        print(f"{self.split} 集特征提取完成")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """获取样本"""
        sample = self.samples[idx]
        feature_file = self.feature_dir / f"{sample['id']}_features.pkl"

        if feature_file.exists():
            # 从保存的特征文件加载
            with open(feature_file, 'rb') as f:
                features = pickle.load(f)

            audio_features = features['audio_features']
            text_features = features['text_features']
            label = features['label']
        else:
            # 如果特征文件不存在，返回零向量
            audio_features = np.zeros(self.config.AUDIO_FEATURE_DIM)
            text_features = np.zeros(self.config.TEXT_FEATURE_DIM)
            label = sample['label']

        # 转换为tensor
        audio_tensor = torch.FloatTensor(audio_features)
        text_tensor = torch.FloatTensor(text_features)
        label_tensor = torch.LongTensor([label])

        return {
            'audio': audio_tensor,
            'text': text_tensor,
            'label': label_tensor.squeeze(),
            'id': sample['id']
        }

# ==============================================================================
# 主程序
# ==============================================================================
def main():
    """主程序"""
    print("=" * 80)
    print("多模态特征提取系统 - 无torchaudio版本")
    print("=" * 80)
    print("音频分支: MFCC+F0 → 病理特征滤波 → 热图 → EfficientNet-B0")
    print("文本分支: ERNIE 3.0 → 抑郁词典增强 → 分层BiLSTM-Attention")
    print("=" * 80)

    # 加载配置
    config = Config()
    config.create_dirs()

    print(f"训练数据路径: {config.TRAIN_DATA_PATH}")
    print(f"验证数据路径: {config.VAL_DATA_PATH}")
    print(f"特征保存路径: {config.FEATURE_SAVE_DIR}")

    # 检查路径是否存在
    if not os.path.exists(config.TRAIN_DATA_PATH):
        print(f"错误: 训练数据路径不存在: {config.TRAIN_DATA_PATH}")
        return

    if not os.path.exists(config.VAL_DATA_PATH):
        print(f"错误: 验证数据路径不存在: {config.VAL_DATA_PATH}")
        return

    # 创建数据集并提取特征
    print("\n" + "=" * 80)
    print("开始特征提取...")
    print("=" * 80)

    # 训练集
    print("\n处理训练集...")
    train_dataset = NoTorchaudioMultimodalDataset(
        data_dir=config.TRAIN_DATA_PATH,
        config=config,
        split='train',
        extract_features=True
    )

    # 验证集
    print("\n处理验证集...")
    val_dataset = NoTorchaudioMultimodalDataset(
        data_dir=config.VAL_DATA_PATH,
        config=config,
        split='validation',
        extract_features=True
    )

    # 保存数据集信息
    dataset_info = {
        'architecture': {
            'audio_branch': 'MFCC+F0 → 病理特征滤波 → 热图 → EfficientNet-B0',
            'text_branch': 'ERNIE 3.0 → 抑郁词典增强 → 分层BiLSTM-Attention'
        },
        'feature_dimensions': {
            'audio_feature_dim': config.AUDIO_FEATURE_DIM,
            'text_feature_dim': config.TEXT_FEATURE_DIM
        },
        'dataset_sizes': {
            'train': len(train_dataset),
            'validation': len(val_dataset)
        },
        'audio_config': {
            'sample_rate': config.SAMPLE_RATE,
            'n_mfcc': config.N_MFCC,
            'n_mels': config.N_MELS,
            'hop_length': config.HOP_LENGTH,
            'n_fft': config.N_FFT,
            'heatmap_size': config.HEATMAP_SIZE
        },
        'extraction_time': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    info_file = os.path.join(config.FEATURE_SAVE_DIR, "dataset_info.json")
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print(f"\n数据集信息已保存到: {info_file}")

    print("\n" + "=" * 80)
    print("特征提取完成!")
    print("=" * 80)
    print(f"架构总结:")
    print(f"  音频分支: MFCC({config.N_MFCC})+F0 → 病理特征滤波 → 热图({config.HEATMAP_SIZE}x{config.HEATMAP_SIZE}) → EfficientNet-B0 → {config.AUDIO_FEATURE_DIM}维")
    print(f"  文本分支: ERNIE 3.0({config.ERNIE_HIDDEN_SIZE}) → 抑郁词典增强 → 分层BiLSTM-Attention → {config.TEXT_FEATURE_DIM}维")
    print(f"\n数据统计:")
    print(f"  训练集: {len(train_dataset)} 个样本")
    print(f"  验证集: {len(val_dataset)} 个样本")
    print(f"  特征保存位置: {config.FEATURE_SAVE_DIR}")
    print("=" * 80)

    # 创建数据加载器示例
    if len(train_dataset) > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0  # Windows下建议设为0
        )

        print(f"\n训练数据加载器创建完成")
        print(f"总批次: {len(train_loader)}")

# ==============================================================================
# 快速测试
# ==============================================================================
def quick_test():
    """快速测试"""
    print("快速测试特征提取...")

    # 创建临时测试目录
    test_dir = "test_features_no_torchaudio"
    os.makedirs(test_dir, exist_ok=True)

    # 创建测试音频文件
    import soundfile as sf
    test_audio_path = os.path.join(test_dir, "test.wav")
    duration = 2.0
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = 0.5 * np.sin(2 * np.pi * 220 * t)  # 220Hz正弦波
    sf.write(test_audio_path, test_audio, sr)

    # 创建测试文本文件
    test_text_path = os.path.join(test_dir, "test.txt")
    with open(test_text_path, 'w', encoding='utf-8') as f:
        f.write("我感到非常难过和绝望，最近总是睡不着觉，觉得自己很没用。")

    # 测试音频特征提取
    print("\n测试音频特征提取...")
    config = Config()
    audio_extractor = NoTorchaudioAudioFeatureExtractor(config)

    audio_features = audio_extractor.extract_audio_features(test_audio_path, save_heatmap=True)
    print(f"音频特征维度: {audio_features.shape}")
    print(f"音频特征范围: [{audio_features.min():.4f}, {audio_features.max():.4f}]")

    # 测试文本特征提取
    print("\n测试文本特征提取...")
    text_extractor = NoTorchaudioTextFeatureExtractor(config)

    text_features = text_extractor.extract_text_features(test_text_path)
    print(f"文本特征维度: {text_features.shape}")
    print(f"文本特征范围: [{text_features.min():.4f}, {text_features.max():.4f}]")

    # 清理测试文件
    import shutil
    shutil.rmtree(test_dir)

    print("\n快速测试完成!")

# ==============================================================================
# 运行
# ==============================================================================
if __name__ == "__main__":
    # 先运行快速测试
    try:
        quick_test()
    except Exception as e:
        print(f"快速测试失败: {e}")
        print("继续主程序...")

    # 运行主程序
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n程序运行失败: {e}")
        import traceback
        traceback.print_exc()