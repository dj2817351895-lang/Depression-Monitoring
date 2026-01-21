"""
Windows兼容版数据增强系统 - 专门解决Windows下音频处理问题
主要改进：
1. 移除Unix/Linux特有的signal超时机制
2. 使用更稳定的音频处理库
3. 添加详细的调试信息
"""

import os
import numpy as np
import librosa
import soundfile as sf
import random
import shutil
import jieba
import jieba.posseg as pseg
from pathlib import Path
from tqdm import tqdm
import warnings
import math
import re
from collections import defaultdict
import traceback
import time

# 忽略特定警告
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning)

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ==============================================================================
# Windows兼容的音频增强器
# ==============================================================================
class WindowsAudioAugmenter:
    """
    Windows兼容的音频增强器：简化处理，避免卡顿
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def apply_safe_augmentation(self, audio_path, output_path):
        """安全的音频增强（简化版）"""
        try:
            # 检查文件是否存在
            if not os.path.exists(audio_path):
                print(f"音频文件不存在: {audio_path}")
                return False

            # 检查文件大小（防止处理超大或损坏的文件）
            file_size = os.path.getsize(audio_path)
            if file_size < 1024:  # 小于1KB的文件可能损坏
                print(f"音频文件太小: {audio_path}, 大小: {file_size}字节")
                return False
            if file_size > 50 * 1024 * 1024:  # 大于50MB的文件跳过
                print(f"音频文件太大: {audio_path}, 大小: {file_size/1024/1024:.2f}MB")
                return False

            # 1. 加载音频（使用soundfile，更稳定）
            try:
                audio, sr = sf.read(audio_path)
                # 如果是立体声，转换为单声道
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
            except Exception as e:
                print(f"soundfile读取失败 {audio_path}: {e}")
                # 尝试使用librosa
                try:
                    audio, sr = librosa.load(audio_path, sr=None, mono=True)
                except Exception as e2:
                    print(f"librosa读取失败 {audio_path}: {e2}")
                    return False

            # 重采样到16kHz
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

            # 检查音频长度
            duration = len(audio) / self.sample_rate
            if duration < 0.3:  # 小于0.3秒的跳过
                print(f"音频太短: {audio_path}, 长度: {duration:.2f}秒")
                return False
            if duration > 60:  # 大于60秒的截断
                print(f"音频太长，截断前30秒: {audio_path}, 长度: {duration:.2f}秒")
                audio = audio[:int(30 * self.sample_rate)]

            # 2. 简单的静音去除（可选）
            try:
                if len(audio) > 1000:  # 确保音频足够长
                    audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
                    if len(audio_trimmed) > 0.5 * len(audio):  # 保留至少50%
                        audio = audio_trimmed
            except:
                pass  # 静音去除失败也没关系

            # 3. 随机选择一种简单的增强方式
            aug_audio = audio.copy()

            # 随机选择增强方法
            method = random.choice([0, 1, 2])  # 0: 无增强, 1: 加噪, 2: 轻微变速

            if method == 1 and len(audio) > 1000:
                # 轻微加噪
                noise_level = random.uniform(0.001, 0.005)
                noise = np.random.randn(len(audio)) * noise_level
                aug_audio = audio + noise

            elif method == 2 and len(audio) > 1000:
                # 轻微变速（0.9-1.1倍）
                rate = random.uniform(0.9, 1.1)
                try:
                    aug_audio = librosa.effects.time_stretch(audio, rate=rate)
                except:
                    pass  # 变速失败就用原音频

            # 4. 保存音频
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # 归一化防止削波
                if np.max(np.abs(aug_audio)) > 0:
                    aug_audio = aug_audio / (np.max(np.abs(aug_audio)) + 1e-8) * 0.95

                # 保存为16kHz WAV
                sf.write(output_path, aug_audio, self.sample_rate, subtype='PCM_16')
                return True

            except Exception as e:
                print(f"保存音频失败 {output_path}: {e}")
                # 尝试直接复制原文件
                try:
                    shutil.copy(audio_path, output_path)
                    return True
                except:
                    return False

        except Exception as e:
            print(f"音频处理失败 {audio_path}: {e}")
            # 直接复制原文件
            try:
                shutil.copy(audio_path, output_path)
                return True
            except:
                return False

    def process_audio(self, audio_path, output_path):
        """处理单个音频文件（带异常处理）"""
        try:
            success = self.apply_safe_augmentation(audio_path, output_path)
            return success
        except Exception as e:
            print(f"音频处理异常 {audio_path}: {e}")
            # 直接复制原文件
            try:
                shutil.copy(audio_path, output_path)
                return True
            except:
                return False

# ==============================================================================
# 智能文本增强器（保持原样）
# ==============================================================================
class SmartTextAugmenter:
    """智能文本增强器：解决多样性和覆盖率问题"""

    def __init__(self):
        self.lexicon = self._create_lexicon()
        self.used_replacements = defaultdict(set)

    def _create_lexicon(self):
        """创建词典"""
        return {
            'core_lexicon': {
                "绝望", "无助", "悲伤", "痛苦", "抑郁", "忧郁", "沮丧", "低落",
                "失落", "孤独", "寂寞", "空虚", "麻木", "崩溃", "难过", "伤心",
                "悲痛", "哀伤", "郁闷", "消沉", "压抑", "烦闷", "焦虑", "紧张",
            },
            'synonym_dict': {
                "难过": [("伤心", 3), ("悲伤", 4), ("痛苦", 5), ("沮丧", 4)],
                "伤心": [("难过", 3), ("悲伤", 4), ("痛苦", 5), ("心碎", 5)],
                "悲伤": [("悲痛", 5), ("哀伤", 4), ("难过", 3), ("忧郁", 4)],
                "疲劳": [("疲倦", 3), ("疲惫", 4), ("劳累", 3), ("乏力", 4)],
                "失眠": [("睡不着", 3), ("入睡困难", 4), ("早醒", 4)],
                "觉得": [("感到", 2), ("认为", 2), ("感觉", 2), ("以为", 2)],
                "认为": [("觉得", 2), ("以为", 2), ("感觉", 2), ("认定", 3)],
                "经常": [("常常", 2), ("总是", 3), ("频繁", 2), ("时常", 2)],
                "总是": [("经常", 2), ("一直", 3), ("从来", 3), ("每每", 2)],
                "非常": [("十分", 2), ("特别", 2), ("极其", 3), ("格外", 2)],
                "压力": [("负担", 2), ("重负", 3), ("紧张", 2), ("压迫感", 3)],
            }
        }

    def _reset_for_new_text(self):
        """重置替换记录"""
        self.used_replacements = defaultdict(set)

    def _get_unique_synonym(self, word, original_intensity):
        """获取唯一同义词"""
        if word not in self.lexicon['synonym_dict']:
            return None

        synonyms = self.lexicon['synonym_dict'][word]
        used = self.used_replacements[word]

        candidates = []
        for syn, intensity in synonyms:
            if syn not in used and abs(intensity - original_intensity) <= 1:
                candidates.append(syn)

        if not candidates:
            return None

        chosen = random.choice(candidates)
        self.used_replacements[word].add(chosen)
        return chosen

    def _advanced_vocab_augment(self, text):
        """高级词汇增强"""
        if len(text) < 3:
            return text

        # 分割为字符
        chars = list(text)
        new_chars = []
        i = 0

        while i < len(chars):
            # 尝试匹配2-4字词语
            matched = False
            for length in range(min(4, len(chars) - i), 1, -1):
                word = ''.join(chars[i:i+length])

                # 检查是否是核心词汇（保护）
                if word in self.lexicon['core_lexicon']:
                    new_chars.append(word)
                    i += length
                    matched = True
                    break

                # 检查是否有同义词
                if word in self.lexicon['synonym_dict'] and random.random() < 0.4:
                    synonym = self._get_unique_synonym(word, 3)
                    if synonym:
                        new_chars.append(synonym)
                        i += length
                        matched = True
                        break

            if not matched and i < len(chars):
                new_chars.append(chars[i])
                i += 1

        return ''.join(new_chars)

    def augment(self, text):
        """文本增强主函数"""
        if not text or len(text) < 2:
            return text

        self._reset_for_new_text()

        # 随机选择增强策略
        strategies = ["vocab", "struct", "semantic"]
        strategy = random.choice(strategies)

        if strategy == "vocab":
            result = self._advanced_vocab_augment(text)
        elif strategy == "struct":
            # 简单结构增强
            if random.random() < 0.3:
                prefixes = ["我", "我们", "本人"]
                result = random.choice(prefixes) + text
            else:
                suffixes = ["。", "！", "……"]
                result = text + random.choice(suffixes)
        else:  # semantic
            # 简单语义改写
            result = text
            paraphrases = [
                ("感到非常难过", "感觉特别伤心"),
                ("压力很大", "负担很重"),
                ("睡不着觉", "难以入睡"),
                ("情绪低落", "心情不好"),
                ("食欲不振", "没有胃口"),
                ("注意力不集中", "无法集中注意力"),
                ("觉得生活没有希望", "感到生活无望"),
            ]

            for pattern, replacement in paraphrases:
                if pattern in result and random.random() < 0.5:
                    result = result.replace(pattern, replacement)
                    break

        # 质量检查
        if (result == text or
            len(result) < len(text) * 0.5 or
            len(result) > len(text) * 2.0 or
            re.search(r'(.)\1{3,}', result)):

            # 轻微修改
            if random.random() < 0.5:
                suffixes = ["。", "！", "……"]
                result = text + random.choice(suffixes)
            else:
                prefixes = ["有时候", "经常", "总是"]
                result = random.choice(prefixes) + text

        return result

# ==============================================================================
# 主处理器（Windows兼容版）
# ==============================================================================
class WindowsEATDProcessor:
    """
    Windows兼容版EATD处理器
    """

    def __init__(self, input_root, output_root):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.audio_aug = WindowsAudioAugmenter()
        self.text_aug = SmartTextAugmenter()

        # 统计信息
        self.stats = {
            'total_processed': 0,
            'audio_success': 0,
            'audio_failed': 0,
            'text_success': 0,
            'text_failed': 0
        }

    def is_depressed(self, folder):
        """读取标签判断是否抑郁"""
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
                    return val >= 53
        except:
            pass
        return False

    def process(self):
        """主处理函数"""
        print(f"源目录: {self.input_root}")
        print(f"目标目录: {self.output_root}")

        # 只处理训练集（验证集不需要增强）
        split = 'train'
        in_split = self.input_root / split
        out_split = self.output_root / split

        if not in_split.exists():
            print(f"错误: 找不到目录 {in_split}")
            return

        # 扫描统计
        volunteers = [d for d in in_split.iterdir() if d.is_dir()]
        dep_vols = [v for v in volunteers if self.is_depressed(v)]
        non_vols = [v for v in volunteers if not self.is_depressed(v)]

        print(f"\n处理 {split} 集...")
        print(f"原始: 抑郁={len(dep_vols)}, 非抑郁={len(non_vols)}")

        # 动态平衡（抑郁样本上采样）
        target_count = len(non_vols)
        if len(dep_vols) > 0:
            aug_ratio = min(math.ceil(target_count / len(dep_vols)), 5)  # 最大5倍
        else:
            aug_ratio = 1

        print(f"动态平衡策略: 抑郁样本增强 {aug_ratio} 倍")

        # 确保输出目录存在
        out_split.mkdir(parents=True, exist_ok=True)

        # 创建进度条
        total_samples = len(dep_vols) + len(non_vols)
        pbar = tqdm(total=total_samples, desc="处理样本")

        # 处理抑郁样本（增强）
        for vol_idx, vol in enumerate(dep_vols):
            try:
                vol_name = vol.name
                # 复制原始样本
                self._copy_sample(vol, out_split / vol_name)
                self.stats['total_processed'] += 1

                # 生成增强副本
                for i in range(aug_ratio - 1):
                    dst_name = f"{vol_name}_aug{i+1}"
                    dst_path = out_split / dst_name

                    # 创建增强样本
                    success = self._create_augmented_sample(vol, dst_path)
                    if success:
                        self.stats['total_processed'] += 1

                pbar.update(1)
                pbar.set_postfix({
                    '音频成功': self.stats['audio_success'],
                    '音频失败': self.stats['audio_failed'],
                    '文本成功': self.stats['text_success']
                })

                # 每处理5个样本打印一次状态
                if (vol_idx + 1) % 5 == 0:
                    print(f"\n已处理 {vol_idx + 1}/{len(dep_vols)} 个抑郁样本")

            except Exception as e:
                print(f"\n处理抑郁样本 {vol.name} 失败: {e}")
                traceback.print_exc()
                pbar.update(1)
                continue

        # 处理非抑郁样本（只复制）
        for vol_idx, vol in enumerate(non_vols):
            try:
                self._copy_sample(vol, out_split / vol.name)
                self.stats['total_processed'] += 1
                pbar.update(1)

                # 每处理10个样本打印一次状态
                if (vol_idx + 1) % 10 == 0:
                    print(f"\n已复制 {vol_idx + 1}/{len(non_vols)} 个非抑郁样本")

            except Exception as e:
                print(f"\n复制非抑郁样本 {vol.name} 失败: {e}")
                pbar.update(1)
                continue

        pbar.close()

        # 打印统计信息
        print(f"\n{'='*60}")
        print(f"处理完成!")
        print(f"{'='*60}")
        print(f"总处理样本数: {self.stats['total_processed']}")
        print(f"音频处理成功: {self.stats['audio_success']}")
        print(f"音频处理失败: {self.stats['audio_failed']}")
        print(f"文本处理成功: {self.stats['text_success']}")
        print(f"文本处理失败: {self.stats['text_failed']}")

        if self.stats['audio_failed'] > 0:
            print(f"\n警告: 有 {self.stats['audio_failed']} 个音频文件处理失败")
            print("失败的文件已使用原文件替代")

        # 最终统计
        try:
            final_folders = [f for f in out_split.iterdir() if f.is_dir()]
            final_dep = 0
            final_non = 0

            for folder in final_folders:
                if self.is_depressed(folder):
                    final_dep += 1
                else:
                    final_non += 1

            print(f"\n{'='*60}")
            print(f"最终统计:")
            print(f"{'='*60}")
            print(f"抑郁样本: {final_dep}")
            print(f"非抑郁样本: {final_non}")
            print(f"总计: {len(final_folders)}")
            print(f"理论抑郁样本数: {len(dep_vols) * aug_ratio}")
            print(f"理论非抑郁样本数: {len(non_vols)}")
        except Exception as e:
            print(f"\n最终统计失败: {e}")

    def _copy_sample(self, src, dst):
        """复制原始样本"""
        if dst.exists():
            try:
                shutil.rmtree(dst)
            except:
                pass  # 如果删除失败，继续尝试复制

        try:
            shutil.copytree(src, dst)
            return True
        except Exception as e:
            print(f"复制样本失败 {src} -> {dst}: {e}")
            return False

    def _create_augmented_sample(self, src, dst):
        """创建增强样本"""
        try:
            # 创建目标目录
            if dst.exists():
                try:
                    shutil.rmtree(dst)
                except:
                    pass

            dst.mkdir(parents=True, exist_ok=True)

            # 复制标签文件
            for f in ['label.txt', 'new_label.txt']:
                src_file = src / f
                if src_file.exists():
                    try:
                        shutil.copy(src_file, dst / f)
                    except Exception as e:
                        print(f"复制标签文件失败 {src_file}: {e}")

            # 处理三个情感文件
            emotions = ['positive', 'negative', 'neutral']

            for emo in emotions:
                # 音频处理
                aud_src = src / f"{emo}.wav"
                if not aud_src.exists():
                    aud_src = src / f"{emo}_out.wav"  # 兼容旧命名

                if aud_src.exists():
                    output_audio = dst / f"{emo}.wav"

                    # 处理音频
                    start_time = time.time()
                    success = self.audio_aug.process_audio(str(aud_src), str(output_audio))
                    processing_time = time.time() - start_time

                    if processing_time > 5:  # 处理时间超过5秒
                        print(f"警告: 音频处理时间较长 ({processing_time:.1f}秒): {aud_src}")

                    if success:
                        self.stats['audio_success'] += 1
                    else:
                        self.stats['audio_failed'] += 1
                        print(f"音频处理失败，使用原文件: {aud_src}")

                # 文本处理
                txt_src = src / f"{emo}.txt"
                if txt_src.exists():
                    try:
                        with open(txt_src, 'r', encoding='utf-8') as f:
                            text = f.read().strip()

                        # 应用文本增强
                        aug_text = self.text_aug.augment(text)

                        with open(dst / f"{emo}.txt", 'w', encoding='utf-8') as f:
                            f.write(aug_text)

                        self.stats['text_success'] += 1

                    except Exception as e:
                        print(f"文本增强失败 {txt_src}: {e}")
                        self.stats['text_failed'] += 1
                        # 失败时直接复制
                        try:
                            shutil.copy(txt_src, dst / f"{emo}.txt")
                        except:
                            pass  # 如果复制也失败，就跳过

            return True

        except Exception as e:
            print(f"创建增强样本失败 {src} -> {dst}: {e}")
            traceback.print_exc()
            return False

# ==============================================================================
# 快速测试
# ==============================================================================
def quick_test():
    """快速测试"""
    print("=" * 60)
    print("快速测试文本增强...")
    print("=" * 60)

    test_cases = [
        "我感到非常难过，最近总是睡不着觉，觉得自己很没用。",
        "压力很大，对什么都不感兴趣，经常一个人流泪。",
    ]

    augmenter = SmartTextAugmenter()

    for text in test_cases:
        print(f"\n原始文本: {text}")
        variants = set()
        for i in range(5):
            variant = augmenter.augment(text)
            if variant != text:
                variants.add(variant)

        if variants:
            for j, variant in enumerate(list(variants)[:3], 1):
                print(f"变体 {j}: {variant}")
        else:
            print("未能生成有效的变体")

    print("\n" + "=" * 60)
    print("快速测试完成!")
    print("=" * 60)

# ==============================================================================
# 主函数
# ==============================================================================
if __name__ == "__main__":
    # 快速测试
    quick_test()

    print("\n" + "=" * 60)
    print("开始数据增强处理")
    print("=" * 60)

    # 配置路径
    RAW_DATA_ROOT = r"D:\Work(paper3)\EATD-Corpus"
    AUG_DATA_ROOT = r"D:\Work(paper3)\EATD-AUG-WINDOWS"

    if not os.path.exists(RAW_DATA_ROOT):
        print(f"错误: 源路径不存在: {RAW_DATA_ROOT}")
        print("请检查路径配置")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"源目录内容: {os.listdir(os.path.dirname(RAW_DATA_ROOT)) if os.path.exists(os.path.dirname(RAW_DATA_ROOT)) else '父目录不存在'}")
    else:
        print(f"找到源目录: {RAW_DATA_ROOT}")

        # 检查是否有train目录
        train_dir = os.path.join(RAW_DATA_ROOT, 'train')
        if not os.path.exists(train_dir):
            print(f"错误: 找不到train目录: {train_dir}")
            print("请确保EATD-Corpus目录结构正确")
        else:
            print(f"找到train目录: {train_dir}")
            print(f"train目录内容示例: {os.listdir(train_dir)[:5]}")

        # 备份原有数据（如果存在）
        if os.path.exists(AUG_DATA_ROOT):
            backup_path = AUG_DATA_ROOT + "_BACKUP"
            print(f"\n警告: 输出目录已存在，正在备份到: {backup_path}")
            try:
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path)
                shutil.copytree(AUG_DATA_ROOT, backup_path)
                print("备份完成")
            except Exception as e:
                print(f"备份失败: {e}")
                print("继续处理...")

        # 创建处理器并执行
        try:
            processor = WindowsEATDProcessor(RAW_DATA_ROOT, AUG_DATA_ROOT)
            processor.process()

            print("\n" + "=" * 60)
            print("数据增强完成!")
            print(f"增强数据保存在: {AUG_DATA_ROOT}")
            print("=" * 60)

        except Exception as e:
            print(f"\n处理过程中发生错误: {e}")
            traceback.print_exc()