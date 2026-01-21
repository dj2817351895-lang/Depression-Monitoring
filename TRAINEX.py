"""抑郁症诊断训练系统 - 终极优化版本
结合多种策略提升性能
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import json
import random
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import defaultdict
import warnings
import time
import gc

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==============================================================================
# 配置
# ==============================================================================
class UltimateConfig:
    # 特征路径
    FEATURE_DIR = r"D:\Work(paper3)\EATD-Features-NoTorchaudio"

    # 模型保存路径
    MODEL_SAVE_DIR = r"D:\Work(paper3)\EATD-Models-Ultimate"

    # 训练参数
    BATCH_SIZE = 8
    EPOCHS = 200
    INITIAL_LR = 1e-4
    WEIGHT_DECAY = 1e-5

    # 特征维度
    AUDIO_DIM = 1280
    TEXT_DIM = 512
    HIDDEN_DIM = 256

    # 早停
    PATIENCE = 30

    # 数据划分
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1

    @classmethod
    def create_dirs(cls):
        """创建目录"""
        os.makedirs(cls.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.MODEL_SAVE_DIR, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cls.MODEL_SAVE_DIR, "visualizations"), exist_ok=True)

# ==============================================================================
# 高级特征处理器
# ==============================================================================
class AdvancedFeatureProcessor:
    """高级特征处理"""

    def __init__(self):
        self.audio_scaler = StandardScaler()
        self.text_scaler = StandardScaler()

    def load_and_process_features(self):
        """加载并处理特征"""
        print("加载特征数据...")

        # 加载所有特征
        train_dir = Path(UltimateConfig.FEATURE_DIR) / "train"
        val_dir = Path(UltimateConfig.FEATURE_DIR) / "validation"

        all_features = []
        all_labels = []

        # 加载训练集
        for feature_file in train_dir.glob("*_features.pkl"):
            try:
                with open(feature_file, 'rb') as f:
                    data = pickle.load(f)
                if self.validate_feature(data):
                    all_features.append(data)
                    all_labels.append(data['label'])
            except:
                continue

        # 加载验证集
        for feature_file in val_dir.glob("*_features.pkl"):
            try:
                with open(feature_file, 'rb') as f:
                    data = pickle.load(f)
                if self.validate_feature(data):
                    all_features.append(data)
                    all_labels.append(data['label'])
            except:
                continue

        print(f"加载了 {len(all_features)} 个样本")
        print(f"类别分布: 抑郁={sum(all_labels)}, 非抑郁={len(all_labels)-sum(all_labels)}")

        # 提取特征
        audio_features = []
        text_features = []
        labels = []

        for data in all_features:
            audio_features.append(data['audio_features'])
            text_features.append(data['text_features'])
            labels.append(data['label'])

        # 转换为numpy
        audio_features = np.array(audio_features)
        text_features = np.array(text_features)
        labels = np.array(labels)

        # 特征增强：创建更有信息量的特征
        enhanced_features = self.enhance_features(audio_features, text_features)

        return enhanced_features, labels

    def validate_feature(self, data):
        """验证特征"""
        if 'audio_features' not in data or 'text_features' not in data or 'label' not in data:
            return False

        if (len(data['audio_features']) != UltimateConfig.AUDIO_DIM or
            len(data['text_features']) != UltimateConfig.TEXT_DIM):
            return False

        return True

    def enhance_features(self, audio_features, text_features):
        """增强特征"""
        print("增强特征...")

        # 1. 原始特征
        enhanced = []

        # 2. 添加统计特征
        for i in range(len(audio_features)):
            # 音频统计特征
            audio_mean = np.mean(audio_features[i])
            audio_std = np.std(audio_features[i])
            audio_min = np.min(audio_features[i])
            audio_max = np.max(audio_features[i])

            # 文本统计特征
            text_mean = np.mean(text_features[i])
            text_std = np.std(text_features[i])

            # 组合所有特征
            combined = np.concatenate([
                audio_features[i],
                text_features[i],
                [audio_mean, audio_std, audio_min, audio_max, text_mean, text_std]
            ])

            enhanced.append(combined)

        enhanced = np.array(enhanced)
        print(f"增强后特征维度: {enhanced.shape}")

        return enhanced

# ==============================================================================
# 强大的神经网络模型
# ==============================================================================
class PowerfulFusionModel(nn.Module):
    """强大的融合模型"""

    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # 创建深度网络
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        # 特征提取器
        self.feature_extractor = nn.Sequential(*layers)

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)

        # 注意力权重
        attn_weights = self.attention(features)

        # 应用注意力
        attended_features = features * attn_weights

        # 分类
        logits = self.classifier(attended_features)

        return logits, attn_weights, features

# ==============================================================================
# 自定义损失函数
# ==============================================================================
class CustomLoss(nn.Module):
    """自定义损失函数"""

    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight], device=device)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits, targets):
        # 基础BCE损失
        bce_loss = self.bce(logits.squeeze(), targets.float())

        # 添加正则化项：鼓励模型输出接近0或1的预测
        probs = torch.sigmoid(logits)
        entropy_reg = -torch.mean(probs * torch.log(probs + 1e-8) +
                                 (1 - probs) * torch.log(1 - probs + 1e-8))

        # 组合损失
        total_loss = bce_loss + 0.1 * entropy_reg

        return total_loss

# ==============================================================================
# 数据平衡器
# ==============================================================================
class DataBalancer:
    """数据平衡器"""

    def __init__(self):
        self.smote = SMOTE(random_state=42, k_neighbors=3)

    def balance_data(self, X, y):
        """平衡数据"""
        print(f"平衡前: 类别0={sum(y==0)}, 类别1={sum(y==1)}")

        # 使用SMOTE过采样
        try:
            X_balanced, y_balanced = self.smote.fit_resample(X, y)
            print(f"平衡后: 类别0={sum(y_balanced==0)}, 类别1={sum(y_balanced==1)}")
            return X_balanced, y_balanced
        except:
            print("SMOTE失败，使用随机过采样")
            return self.random_oversample(X, y)

    def random_oversample(self, X, y):
        """随机过采样"""
        # 计算类别分布
        class_counts = np.bincount(y)
        majority_class = np.argmax(class_counts)
        minority_class = 1 - majority_class

        # 复制少数类样本
        minority_indices = np.where(y == minority_class)[0]
        oversampled_indices = np.random.choice(
            minority_indices,
            size=class_counts[majority_class] - class_counts[minority_class],
            replace=True
        )

        # 合并数据
        X_balanced = np.vstack([X, X[oversampled_indices]])
        y_balanced = np.concatenate([y, y[oversampled_indices]])

        print(f"随机过采样后: 类别0={sum(y_balanced==0)}, 类别1={sum(y_balanced==1)}")

        return X_balanced, y_balanced

# ==============================================================================
# 终极训练器
# ==============================================================================
class UltimateTrainer:
    """终极训练器"""

    def __init__(self, config):
        self.config = config
        self.device = device

        # 创建目录
        config.create_dirs()

        print("="*80)
        print("初始化终极训练器...")
        print("="*80)

        # 处理特征
        self.feature_processor = AdvancedFeatureProcessor()
        self.data_balancer = DataBalancer()

        # 加载和准备数据
        self.prepare_data()

        # 初始化模型
        self.model = PowerfulFusionModel(
            input_dim=self.features.shape[1]
        ).to(self.device)

        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.INITIAL_LR,
            weight_decay=config.WEIGHT_DECAY
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.INITIAL_LR * 3,
            epochs=config.EPOCHS,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )

        # 损失函数
        pos_weight = len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])
        pos_weight = max(1.0, min(pos_weight, 10.0))  # 限制在1-10之间
        self.criterion = CustomLoss(pos_weight=pos_weight)

        print(f"正样本权重: {pos_weight:.2f}")

        # 训练历史
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'val_auc': [], 'val_sensitivity': [], 'val_specificity': []
        }

        # 最佳指标
        self.best_val_f1 = 0
        self.best_val_auc = 0
        self.best_val_acc = 0

    def prepare_data(self):
        """准备数据"""
        print("\n准备数据...")

        # 加载和增强特征
        features, labels = self.feature_processor.load_and_process_features()

        # 划分数据集
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels,
            test_size=self.config.TEST_SIZE,
            stratify=labels,
            random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config.VAL_SIZE/(1-self.config.TEST_SIZE),
            stratify=y_temp,
            random_state=42
        )

        print(f"训练集: {len(X_train)} 样本")
        print(f"验证集: {len(X_val)} 样本")
        print(f"测试集: {len(X_test)} 样本")

        # 平衡训练数据
        X_train_balanced, y_train_balanced = self.data_balancer.balance_data(X_train, y_train)

        # 创建数据集
        self.train_dataset = self.create_dataset(X_train_balanced, y_train_balanced)
        self.val_dataset = self.create_dataset(X_val, y_val)
        self.test_dataset = self.create_dataset(X_test, y_test)

        # 创建数据加载器
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        # 保存特征用于分析
        self.features = features
        self.labels = labels
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def create_dataset(self, features, labels):
        """创建数据集"""
        class FeatureDataset(torch.utils.data.Dataset):
            def __init__(self, features, labels):
                self.features = torch.FloatTensor(features)
                self.labels = torch.FloatTensor(labels)

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return {
                    'features': self.features[idx],
                    'label': self.labels[idx]
                }

        return FeatureDataset(features, labels)

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"训练 Epoch {epoch+1}")

        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device)
            targets = batch['label'].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            logits, _, _ = self.model(features)

            # 计算损失
            loss = self.criterion(logits, targets)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()
            self.scheduler.step()

            # 统计
            total_loss += loss.item()

            # 预测
            preds = torch.sigmoid(logits) > 0.5
            preds = preds.float().squeeze()

            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

            # 收集结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

            # 更新进度条
            acc = 100. * correct / total if total > 0 else 0
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': f'{acc:.1f}%',
                'lr': self.optimizer.param_groups[0]['lr']
            })

        # 计算指标
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total if total > 0 else 0

        # 计算F1分数
        if len(set(all_labels)) > 1:
            f1 = f1_score(all_labels, all_preds, average='weighted')
        else:
            f1 = 0.0

        return avg_loss, avg_acc, f1

    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        all_labels = []
        all_preds = []
        all_probs = []

        for batch in self.val_loader:
            features = batch['features'].to(self.device)
            targets = batch['label'].to(self.device)

            # 前向传播
            logits, _, _ = self.model(features)

            # 计算损失
            loss = self.criterion(logits, targets)
            val_loss += loss.item()

            # 预测
            probs = torch.sigmoid(logits)
            preds = probs > 0.5
            preds = preds.float().squeeze()

            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

            # 收集结果
            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.squeeze().cpu().numpy())

        # 计算指标
        avg_loss = val_loss / len(self.val_loader)
        avg_acc = 100. * correct / total if total > 0 else 0

        # 计算AUC
        if len(set(all_labels)) > 1:
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except:
                auc = 0.5
        else:
            auc = 0.5

        # 计算F1分数
        if len(set(all_labels)) > 1:
            f1 = f1_score(all_labels, all_preds, average='weighted')
        else:
            f1 = 0.0

        # 计算敏感性和特异性
        if len(set(all_labels)) > 1:
            cm = confusion_matrix(all_labels, all_preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                sensitivity = specificity = 0
        else:
            sensitivity = specificity = 0

        print(f"验证结果 - Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%, F1: {f1:.4f}")
        print(f"          AUC: {auc:.4f}, 敏感性: {sensitivity:.4f}, 特异性: {specificity:.4f}")

        return avg_loss, avg_acc, auc, f1, sensitivity, specificity

    def train(self):
        """主训练循环"""
        print("\n" + "="*80)
        print("开始终极训练")
        print("="*80)

        start_time = time.time()
        patience_counter = 0
        best_epoch = 0

        for epoch in range(self.config.EPOCHS):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.EPOCHS}")
            print(f"{'='*60}")

            # 训练
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)

            # 验证
            val_loss, val_acc, val_auc, val_f1, sensitivity, specificity = self.validate(epoch)

            # 保存历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['val_auc'].append(val_auc)
            self.history['val_sensitivity'].append(sensitivity)
            self.history['val_specificity'].append(specificity)

            # 检查是否为最佳模型
            is_best = False
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_val_auc = val_auc
                self.best_val_acc = val_acc
                self.history['best_epoch'] = epoch
                best_epoch = epoch
                is_best = True
                patience_counter = 0
                print(f"🎉 新的最佳模型! F1: {val_f1:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.2f}%")

                # 保存最佳模型
                self.save_checkpoint(epoch, is_best=True)
            else:
                patience_counter += 1

            # 早停检查
            if patience_counter >= self.config.PATIENCE:
                print(f"\n⚠️ 早停触发! 连续{self.config.PATIENCE}个epoch验证F1分数未提升")
                break

        # 训练完成
        training_time = time.time() - start_time

        print(f"\n{'='*80}")
        print("训练完成!")
        print(f"总时间: {training_time/60:.2f}分钟")
        print(f"总轮次: {len(self.history['train_loss'])}")
        print(f"最佳Epoch: {best_epoch + 1}")
        print(f"最佳F1分数: {self.best_val_f1:.4f}")
        print(f"最佳AUC: {self.best_val_auc:.4f}")
        print(f"最佳准确率: {self.best_val_acc:.2f}%")
        print(f"{'='*80}")

        # 测试最终模型
        self.test_model()

        # 绘制训练历史
        self.plot_training_history()

        # 生成报告
        self.generate_report()

        return self.model

    def test_model(self):
        """测试模型"""
        print(f"\n{'='*80}")
        print("测试模型性能")
        print(f"{'='*80}")

        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in self.test_loader:
                features = batch['features'].to(self.device)
                targets = batch['label'].to(self.device)

                logits, _, _ = self.model(features)
                probs = torch.sigmoid(logits)
                preds = probs > 0.5
                preds = preds.float().squeeze()

                all_labels.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.squeeze().cpu().numpy())

        # 计算测试指标
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5

        cm = confusion_matrix(all_labels, all_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            sensitivity = specificity = 0

        print(f"测试集性能:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  敏感性: {sensitivity:.4f}")
        print(f"  特异性: {specificity:.4f}")

        # 打印混淆矩阵
        print(f"\n混淆矩阵:")
        print(f"       预测非抑郁  预测抑郁")
        print(f"真实非抑郁    {tn:5d}       {fp:5d}")
        print(f"真实抑郁      {fn:5d}       {tp:5d}")

    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': vars(self.config)
        }

        if is_best:
            path = os.path.join(self.config.MODEL_SAVE_DIR, "best_model.pt")
            torch.save(checkpoint, path, _use_new_zipfile_serialization=False)
            print(f"✅ 保存最佳模型到: {path}")

    def plot_training_history(self):
        """绘制训练历史"""
        if len(self.history['train_loss']) == 0:
            return

        epochs = range(1, len(self.history['train_loss']) + 1)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 损失曲线
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='训练损失', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='验证损失', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 准确率曲线
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='训练准确率', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='验证准确率', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('准确率 (%)')
        axes[0, 1].set_title('训练和验证准确率')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # F1分数曲线
        axes[0, 2].plot(epochs, self.history['train_f1'], 'b-', label='训练F1', linewidth=2)
        axes[0, 2].plot(epochs, self.history['val_f1'], 'r-', label='验证F1', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1分数')
        axes[0, 2].set_title('训练和验证F1分数')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # AUC曲线
        axes[1, 0].plot(epochs, self.history['val_auc'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_title('验证集AUC')
        axes[1, 0].grid(True, alpha=0.3)

        # 敏感性和特异性
        axes[1, 1].plot(epochs, self.history['val_sensitivity'], 'c-', label='敏感性', linewidth=2)
        axes[1, 1].plot(epochs, self.history['val_specificity'], 'm-', label='特异性', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('分数')
        axes[1, 1].set_title('敏感性和特异性')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 留空
        axes[1, 2].axis('off')

        plt.tight_layout()

        # 保存图像
        plot_path = os.path.join(self.config.MODEL_SAVE_DIR, "visualizations/training_history.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"训练历史图已保存到: {plot_path}")

    def generate_report(self):
        """生成报告"""
        report = {
            'training_summary': {
                'total_epochs': len(self.history['train_loss']),
                'best_epoch': self.history.get('best_epoch', 0) + 1,
                'best_f1': float(self.best_val_f1),
                'best_auc': float(self.best_val_auc),
                'best_accuracy': float(self.best_val_acc),
                'dataset_sizes': {
                    'total': len(self.features),
                    'train': len(self.X_train),
                    'val': len(self.X_val),
                    'test': len(self.X_test)
                },
                'class_distribution': {
                    'total': {
                        'depressed': int(sum(self.labels)),
                        'non_depressed': int(len(self.labels) - sum(self.labels))
                    },
                    'train': {
                        'depressed': int(sum(self.y_train)),
                        'non_depressed': int(len(self.y_train) - sum(self.y_train))
                    }
                }
            },
            'final_metrics': {
                'train_loss': float(self.history['train_loss'][-1]),
                'val_loss': float(self.history['val_loss'][-1]),
                'train_accuracy': float(self.history['train_acc'][-1]),
                'val_accuracy': float(self.history['val_acc'][-1]),
                'train_f1': float(self.history['train_f1'][-1]),
                'val_f1': float(self.history['val_f1'][-1]),
                'val_auc': float(self.history['val_auc'][-1])
            }
        }

        # 保存报告
        report_path = os.path.join(self.config.MODEL_SAVE_DIR, "training_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"训练报告已保存到: {report_path}")

# ==============================================================================
# 特征重要性分析
# ==============================================================================
class FeatureAnalyzer:
    """特征重要性分析"""

    def __init__(self, model, feature_processor):
        self.model = model
        self.feature_processor = feature_processor

    def analyze(self, X, y):
        """分析特征重要性"""
        print("\n分析特征重要性...")

        # 使用模型获取注意力权重
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(device)

        with torch.no_grad():
            _, attn_weights, features = self.model(X_tensor)

        # 计算平均注意力权重
        avg_attn = attn_weights.mean(dim=0).cpu().numpy()

        # 找出最重要的特征
        important_indices = np.argsort(avg_attn.flatten())[-10:]  # 前10个重要特征

        print("最重要的特征索引:", important_indices)
        print("对应的注意力权重:", avg_attn.flatten()[important_indices])

        return important_indices, avg_attn

# ==============================================================================
# 主程序
# ==============================================================================
def main():
    """主程序"""
    print("="*80)
    print("终极抑郁症诊断训练系统")
    print("="*80)
    print("结合多种策略提升性能")
    print("="*80)

    # 加载配置
    config = UltimateConfig()

    try:
        # 创建训练器
        trainer = UltimateTrainer(config)

        # 开始训练
        model = trainer.train()

        # 分析特征重要性
        analyzer = FeatureAnalyzer(model, trainer.feature_processor)
        analyzer.analyze(trainer.X_test, trainer.y_test)

    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# 快速测试
# ==============================================================================
def quick_test():
    """快速测试"""
    print("快速测试终极系统...")

    config = UltimateConfig()

    # 测试特征处理器
    processor = AdvancedFeatureProcessor()

    try:
        features, labels = processor.load_and_process_features()
        print(f"✅ 特征加载成功: {features.shape}")

        # 测试数据平衡器
        balancer = DataBalancer()
        X_balanced, y_balanced = balancer.balance_data(features[:100], labels[:100])
        print(f"✅ 数据平衡成功: {X_balanced.shape}")

        # 测试模型
        model = PowerfulFusionModel(input_dim=features.shape[1])
        print(f"✅ 模型创建成功: {sum(p.numel() for p in model.parameters()):,}参数")

        print(f"\n✅ 所有组件测试通过!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

# ==============================================================================
# 运行
# ==============================================================================
if __name__ == "__main__":
    # 运行快速测试
    quick_test()

    print("\n" + "="*80)
    print("开始终极训练")
    print("="*80)

    # 运行主程序
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行失败: {e}")
        import traceback
        traceback.print_exc()