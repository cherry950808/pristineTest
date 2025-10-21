import os
from pathlib import Path

# 抑制FFmpeg和OpenCV的警告信息
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'fflags;+ignidx'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

import random
import math

import numpy
import numpy as np
import pickle as pk
import cv2
from tqdm import tqdm
from PIL import Image

import torchvision.transforms as transforms
import torch
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset, RandomSampler
from dataset.randaugment import RandAugmentMC

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False

def validate_video_file(video_path):
    """
    验证视频文件是否可以被正确解码
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Cannot open video file"
        
        # 尝试读取第一帧
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return False, "Cannot read first frame"
        
        # 检查帧尺寸
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            cap.release()
            return False, "Invalid frame dimensions"
        
        # 获取总帧数
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return False, "Invalid frame count"
        
        cap.release()
        return True, f"Valid video: {width}x{height}, {frame_count} frames"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def convert_video_format(input_path, output_path=None):
    """
    使用FFmpeg转换视频格式，解决H.264解码问题
    """
    import subprocess
    
    if output_path is None:
        # 生成临时文件名
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_converted.mp4"
    
    try:
        # 使用FFmpeg重新编码视频，确保H.264兼容性
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',  # 使用H.264编码器
            '-preset', 'fast',   # 快速编码
            '-crf', '23',        # 质量参数
            '-c:a', 'aac',       # 音频编码
            '-y',                # 覆盖输出文件
            output_path
        ]
        
        # 静默运行FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True, output_path
        else:
            return False, f"FFmpeg error: {result.stderr}"
            
    except FileNotFoundError:
        return False, "FFmpeg not found. Please install FFmpeg."
    except Exception as e:
        return False, f"Conversion error: {str(e)}"
        torch.backends.cudnn.deterministic = True
    return 1


# class VideoDataset(Dataset):
#
#     def __init__(self, directory_list, local_rank=0, enable_GPUs_num=0, distributed_load=False, resize_shape=[160, 160],
#                  mode='train', clip_len=32, crop_size=160):                                   # resize_shape=[224, 224]
#
#         self.clip_len, self.crop_size, self.resize_shape = clip_len, crop_size, resize_shape
#         self.mode = mode
#         self.fnames, labels = [], []
#         self.idx = []
#         # get the directory of the specified split
#         for directory in directory_list:
#             folder = Path(directory)
#             print("Load dataset from folder : ", folder)
#             for label in sorted(os.listdir(folder)):
#                 a = os.listdir(os.path.join(folder, label))
#                 for fname in os.listdir(os.path.join(folder, label)) if mode == "train" or "weak" or "strong" or "test" else os.listdir(
#                         os.path.join(folder, label))[:10]:
#                     a = fname
#                     self.fnames.append(os.path.join(folder, label, fname))
#                     labels.append(label)
#
#         random_list = list(zip(self.fnames, labels))
#         random.shuffle(random_list)
#         self.fnames[:], labels[:] = zip(*random_list)
#
#         # self.fnames = self.fnames[:240]
#         '''
#         if mode == 'train' and distributed_load:
#             single_num_ = len(self.fnames)//enable_GPUs_num
#             self.fnames = self.fnames[local_rank*single_num_:((local_rank+1)*single_num_)]
#             labels = labels[local_rank*single_num_:((local_rank+1)*single_num_)]
#         '''
#         # prepare a mapping between the label names (strings) and indices (ints)
#         self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
#         # convert the list of label names into an array of label indices
#         self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)
#
#     def __getitem__(self, index):
#         # seed = set_seed(2048)
#         # a = seed
#         # print("Random_Seed_State:", a)
#         # loading and preprocessing. TODO move them to transform classess
#         buffer = self.loadvideo(self.fnames[index])
#
#         return buffer, self.label_array[index], index
#
#
#
#     def __len__(self):
#         return len(self.fnames)
#
#     def loadvideo(self, fname):
#         # initialize a VideoCapture object to read video data into a numpy array
#         try:
#             video_stream = cv2.VideoCapture(fname)
#             frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
#         except RuntimeError:
#             index = np.random.randint(self.__len__())
#             video_stream = cv2.VideoCapture(self.fnames[index])
#             frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
#
#         while frame_count < self.clip_len + 2:
#             index = np.random.randint(self.__len__())
#             video_stream = cv2.VideoCapture(self.fnames[index])
#             frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
#
#         speed_rate = np.random.randint(1, 3) if frame_count > self.clip_len * 2 + 2 else 1
#         time_index = np.random.randint(frame_count - self.clip_len * speed_rate)
#
#         start_idx, end_idx, final_idx = time_index, time_index + (self.clip_len * speed_rate), frame_count - 1
#         count, sample_count, retaining = 0, 0, True
#
#         buffer = np.empty((self.clip_len, 3, self.resize_shape[0], self.resize_shape[1]), np.dtype('float16'))
#
#         self.transform = transforms.Compose([
#             transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])
#         self.transform_weak = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=1),
#             transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
#             # transforms.RandomCrop(size=224,
#             #                       padding=int(224 * 0.125),
#             #                       padding_mode='reflect'),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])
#         self.transform_strong = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=1),
#             transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
#             RandAugmentMC(n=2, m=10),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])
#         self.transform_strong_rate = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=1),
#             transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])
#
#         self.transform_val = transforms.Compose([
#             # transforms.Resize([self.crop_size, self.crop_size]),
#             transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])
#
#         if self.mode == 'train':
#             while (count <= end_idx and retaining):
#                 retaining, frame = video_stream.read()
#                 if count < start_idx:
#                     count += 1
#                     continue
#                 if count % speed_rate == speed_rate - 1 and count >= start_idx and sample_count < self.clip_len:
#                     try:
#                         buffer[sample_count] = self.transform(
#                             Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
#                     except cv2.error as err:
#                         continue
#                     sample_count += 1
#                 count += 1
#             video_stream.release()
#
#
#         elif self.mode == 'val':
#             while (count <= end_idx and retaining):
#                 retaining, frame = video_stream.read()
#                 if count < start_idx:
#                     count += 1
#                     continue
#                 if count % speed_rate == speed_rate - 1 and count >= start_idx and sample_count < self.clip_len:
#                     try:
#                         buffer[sample_count] = self.transform_val(
#                             Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
#                     except cv2.error as err:
#                         continue
#                     sample_count += 1
#                 count += 1
#             video_stream.release()
#
#         elif self.mode == 'weak':
#             while (count <= end_idx and retaining):
#                 retaining, frame = video_stream.read()
#                 if count < start_idx:
#                     count += 1
#                     continue
#                 if count % speed_rate == speed_rate - 1 and count >= start_idx and sample_count < self.clip_len:
#                     try:
#                         buffer[sample_count] = self.transform_weak(
#                             Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
#                     except cv2.error as err:
#                         continue
#                     sample_count += 1
#                 count += 1
#             video_stream.release()
#
#         elif self.mode == 'strong':
#             while (count <= end_idx and retaining):
#                 retaining, frame = video_stream.read()
#                 if count < start_idx:
#                     count += 1
#                     continue
#                 if count % speed_rate == speed_rate - 1 and count >= start_idx and sample_count < self.clip_len:
#                     try:
#                         buffer[sample_count] = self.transform_strong(
#                             Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
#                     except cv2.error as err:
#                         continue
#                     sample_count += 1
#                 count += 1
#             video_stream.release()
#
#         elif self.mode == 'test':
#             while (count <= end_idx and retaining):
#                 retaining, frame = video_stream.read()
#                 if count < start_idx:
#                     count += 1
#                     continue
#                 if count % speed_rate == speed_rate - 1 and count >= start_idx and sample_count < self.clip_len:
#                     try:
#                         buffer[sample_count] = self.transform_val(
#                             Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
#                     except cv2.error as err:
#                         continue
#                     sample_count += 1
#                 count += 1
#             video_stream.release()
#
#
#         return buffer.transpose((1, 0, 2, 3))

#
# #===================================== PKL ====================================
class VideoDataset(Dataset):

    def __init__(self, directory_list, local_rank=0, enable_GPUs_num=0, distributed_load=False, resize_shape=[160, 160],
                 mode='train', clip_len=32, crop_size=160):                                   # resize_shape=[224, 224]

        self.clip_len, self.crop_size, self.resize_shape = clip_len, crop_size, resize_shape
        self.mode = mode
        self.fnames, labels = [], []
        self.idx = []
        # get the directory of the specified split
        for directory in directory_list:
            folder = Path(directory)
            print("Load dataset from folder : ", folder)
            for label in sorted(os.listdir(folder)):
                a = os.listdir(os.path.join(folder, label))
                for fname in os.listdir(os.path.join(folder, label)) if mode == "train" or "weak" or "strong" or "test" else os.listdir(
                        os.path.join(folder, label))[:10]:
                    a = fname
                    self.fnames.append(os.path.join(folder, label, fname))
                    labels.append(label)

        random_list = list(zip(self.fnames, labels))
        random.shuffle(random_list)
        self.fnames[:], labels[:] = zip(*random_list)

        # self.fnames = self.fnames[:240]
        '''
        if mode == 'train' and distributed_load:
            single_num_ = len(self.fnames)//enable_GPUs_num
            self.fnames = self.fnames[local_rank*single_num_:((local_rank+1)*single_num_)]
            labels = labels[local_rank*single_num_:((local_rank+1)*single_num_)]
        '''
        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

    def __getitem__(self, index):
        # seed = set_seed(2048)
        # a = seed
        # print("Random_Seed_State:", a)
        # loading and preprocessing. TODO move them to transform classess
        buffer = self.loadvideo(self.fnames[index])

        return buffer, self.label_array[index], index


    def __len__(self):
        return len(self.fnames)
    
    def apply_light_augmentation(self, image, frame_idx):
        """
        对静态图片应用轻微的数据增强，生成不同的帧
        """
        # 添加轻微的随机噪声
        noise = np.random.normal(0, 2, image.shape).astype(np.uint8)
        frame = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 轻微的亮度变化
        brightness_factor = 1.0 + np.random.uniform(-0.1, 0.1)
        frame = np.clip(frame * brightness_factor, 0, 255).astype(np.uint8)
        
        # 轻微的对比度变化
        contrast_factor = 1.0 + np.random.uniform(-0.05, 0.05)
        frame = np.clip((frame - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
        
        return frame

    def loadvideo(self, fname):
        # seed = set_seed(2048)
        # print("Random_Seed_State:",seed)
        # initialize a VideoCapture object to read video data into a numpy array
        
        # 根据文件扩展名选择加载方式
        if fname.endswith('.pkl'):
            # 原有的PKL加载逻辑
            with open(fname, 'rb') as Video_reader:
                try:
                    video = pk.load(Video_reader)
                except EOFError:
                    return None

            while video.shape[0] < self.clip_len + 2:
                index = np.random.randint(self.__len__())
                with open(self.fnames[index], 'rb') as Video_reader:
                    video = pk.load(Video_reader)

            height, width = video.shape[1], video.shape[2]

            speed_rate = np.random.randint(1, 3) if video.shape[0] > self.clip_len * 2 + 2 and self.mode == "train" else 1
            time_index = np.random.randint(video.shape[0] - self.clip_len * speed_rate)

            video = video[time_index:time_index + (self.clip_len * speed_rate):speed_rate, :, :, :]
            
        elif fname.endswith(('.jpg', '.jpeg', '.png')):
            # JPG/PNG静态图片加载逻辑
            image = cv2.imread(fname)
            if image is None:
                return None
            
            # 转换颜色空间
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 重复图片生成32帧的"伪视频"
            video_frames = []
            for i in range(self.clip_len):
                # 对图片进行轻微的数据增强，避免完全重复
                if i == 0:
                    frame = image.copy()
                else:
                    # 添加轻微的随机变化
                    frame = self.apply_light_augmentation(image, i)
                video_frames.append(frame)
            
            video = np.array(video_frames)
            
        elif fname.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # MP4等视频文件加载逻辑 - 优化H.264解码
            try:
                # 首先验证视频文件
                is_valid, validation_msg = validate_video_file(fname)
                if not is_valid:
                    print(f"Video validation failed for {fname}: {validation_msg}")
                    return None
                
                # 设置OpenCV解码参数，减少H.264警告
                video_stream = cv2.VideoCapture(fname)
                
                # 优化解码器参数
                video_stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区大小
                video_stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))  # 明确指定H.264解码器
                
                if not video_stream.isOpened():
                    print(f"Warning: Cannot open video file {fname}")
                    return None
                
                # 获取视频信息
                frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = video_stream.get(cv2.CAP_PROP_FPS)
                
                if frame_count < self.clip_len + 2:
                    video_stream.release()
                    return None
                
                # 随机选择起始帧
                speed_rate = np.random.randint(1, 3) if frame_count > self.clip_len * 2 + 2 and self.mode == "train" else 1
                time_index = np.random.randint(frame_count - self.clip_len * speed_rate)
                
                # 读取视频帧 - 添加错误恢复机制
                video_frames = []
                video_stream.set(cv2.CAP_PROP_POS_FRAMES, time_index)
                
                consecutive_failures = 0
                max_consecutive_failures = 5
                
                for i in range(self.clip_len * speed_rate):
                    ret, frame = video_stream.read()
                    
                    if not ret:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            print(f"Too many consecutive frame read failures in {fname}")
                            break
                        continue
                    else:
                        consecutive_failures = 0
                    
                    if i % speed_rate == 0:
                        try:
                            # 转换颜色空间
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            video_frames.append(frame)
                        except cv2.error as e:
                            print(f"Color conversion error in {fname}: {e}")
                            continue
                
                video_stream.release()
                
                if len(video_frames) < self.clip_len:
                    print(f"Insufficient frames extracted from {fname}: {len(video_frames)}/{self.clip_len}")
                    return None
                
                video = np.array(video_frames[:self.clip_len])
                
            except Exception as e:
                print(f"Error processing video {fname}: {e}")
                return None
            
        else:
            # 不支持的文件格式
            print(f"Unsupported file format: {fname}")
            return None

        self.transform = transforms.Compose([
            transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.transform_weak = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
            # transforms.RandomCrop(size=224,
            #                       padding=int(224 * 0.125),
            #                       padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.transform_strong = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
            # transforms.RandomCrop(size=224,
            #                       padding=int(224 * 0.125),
            #                       padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.transform_strong_rate = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.transform_val = transforms.Compose([
            # transforms.Resize([self.crop_size, self.crop_size]),
            transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        if self.mode == 'train':
            # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
            buffer = np.empty((self.clip_len, 3, self.resize_shape[0], self.resize_shape[1]), np.dtype('float16'))
            for idx, frame in enumerate(video):
                buffer[idx] = self.transform(Image.fromarray(frame))

        elif self.mode == 'val':
            # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
            buffer = np.empty((self.clip_len, 3, self.resize_shape[0], self.resize_shape[1]), np.dtype('float16'))
            for idx, frame in enumerate(video):
                buffer[idx] = self.transform_val(Image.fromarray(frame))

        elif self.mode == 'weak':
            buffer = np.empty((self.clip_len, 3, self.resize_shape[0], self.resize_shape[1]), np.dtype('float16'))
            for idx, frame in enumerate(video):
                buffer[idx] = self.transform_weak(Image.fromarray(frame))

        elif self.mode == 'strong':
            buffer = np.empty((self.clip_len, 3, self.resize_shape[0], self.resize_shape[1]), np.dtype('float16'))
            for idx, frame in enumerate(video):
                buffer[idx] = self.transform_strong(Image.fromarray(frame))


                    # if idx % 3 == 0:
                    #     buffer[idx] = self.transform_strong(Image.fromarray(frame))
                    # else:
                    #     buffer[idx] = self.transform_strong_rate(Image.fromarray(frame))


        elif self.mode == 'test':
            buffer = np.empty((self.clip_len, 3, self.resize_shape[0], self.resize_shape[1]), np.dtype('float16'))
            for idx, frame in enumerate(video):
                buffer[idx] = self.transform_val(Image.fromarray(frame))



        return buffer.transpose((1, 0, 2, 3))

class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


def Get_Dataloader(datapath, mode, bs, resize_shape=[160, 160]):
    dataset = VideoDataset(datapath,
                           mode=mode,
                           resize_shape=resize_shape)
    Label_dict = dataset.label2index

    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=8)


    return dataloader, list(Label_dict.keys())

def Get_lx_sux_wux_Dataloader_forFire(args, datapath, weak_datapath, strong_datapath, mode, bs):
    resize_shape = [args.input_size, args.input_size]
    dataset = VideoDataset(datapath,
                           mode=mode,
                           resize_shape=resize_shape)

    weak_dataset = VideoDataset(weak_datapath,
                                mode='weak',
                                resize_shape=resize_shape)
    strong_dataset = VideoDataset(strong_datapath,
                                  mode='strong',
                                  resize_shape=resize_shape)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split_Fire(
        args, dataset.label_array)

    # train_unlabeled_idxs, _ = x_u_split(
    #     args, dataset.label_array)

    labeled_train_dataset = get_ucf101_ssl(dataset, train_labeled_idxs)
    print("-------------------------------------------")
    dataset = VideoDataset(datapath,
                           mode=mode,
                           resize_shape=resize_shape)
    unlabeled_train_dataset = get_ucf101_ssl(dataset, train_unlabeled_idxs)
    print("-------------------------------------------")
    # dataset = VideoDataset(datapath,
    #                        resize_shape=[224, 224],
    #                        mode=mode)
    # unlabeled_train_dataset = get_ucf101_ssl(dataset, train_unlabeled_idxs)
    unlabeled_weak_dataset = get_ucf101_ssl(weak_dataset, train_unlabeled_idxs)
    print("-------------------------------------------")
    unlabeled_strong_dataset = get_ucf101_ssl(strong_dataset, train_unlabeled_idxs)
    print("------------------------------------------")

    u_bs = int(bs * args.mu)
    random_sampler = RandomSampler(unlabeled_train_dataset)
    labeled_train_dataloader = DataLoader(labeled_train_dataset,
                                          batch_size=bs,
                                          shuffle=True,
                                          num_workers=8)

    unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset,
                                            batch_size=u_bs,
                                            sampler=random_sampler,
                                            shuffle=False,
                                            num_workers=8)

    unlabeled_weak_dataloader = DataLoader(unlabeled_weak_dataset,
                                           batch_size=u_bs,
                                           sampler=random_sampler,
                                           shuffle=False,
                                           num_workers=8)

    unlabeled_strong_dataloader = DataLoader(unlabeled_strong_dataset,
                                             batch_size=u_bs,
                                             sampler=random_sampler,
                                             shuffle=False,
                                             num_workers=8)

    return labeled_train_dataloader, unlabeled_train_dataloader, unlabeled_weak_dataloader, unlabeled_strong_dataloader


def Get_lx_sux_wux_Datasets_forFire(args, datapath, weak_datapath, strong_datapath, mode, bs):
    dataset = VideoDataset(datapath,
                           mode=mode)

    weak_dataset = VideoDataset(weak_datapath,
                                mode='weak')
    strong_dataset = VideoDataset(strong_datapath,
                                  mode='strong')

    train_labeled_idxs, train_unlabeled_idxs = x_u_split_Fire(
        args, dataset.label_array)

    # train_unlabeled_idxs, _ = x_u_split(
    #     args, dataset.label_array)

    labeled_train_dataset = get_ucf101_ssl(dataset, train_labeled_idxs)
    print("-------------------------------------------")
    dataset = VideoDataset(datapath,
                           mode=mode)
    unlabeled_train_dataset = get_ucf101_ssl(dataset, train_unlabeled_idxs)
    print("-------------------------------------------")
    # dataset = VideoDataset(datapath,
    #                        resize_shape=[224, 224],
    #                        mode=mode)
    # unlabeled_train_dataset = get_ucf101_ssl(dataset, train_unlabeled_idxs)
    unlabeled_weak_dataset = get_ucf101_ssl(weak_dataset, train_unlabeled_idxs)
    print("-------------------------------------------")
    unlabeled_strong_dataset = get_ucf101_ssl(strong_dataset, train_unlabeled_idxs)
    print("------------------------------------------")



    return labeled_train_dataset, unlabeled_train_dataset, unlabeled_weak_dataset, unlabeled_strong_dataset

def adjust_label_per_class(label_per_class, num_labeled):
   
    a = sum(label_per_class)

    
    b = a - num_labeled

    if b > 0:
        
        max_value = max(label_per_class)
        max_index = label_per_class.index(max_value)
        label_per_class[max_index] -= b
    elif b < 0:
        
        min_value = min(label_per_class)
        min_index = label_per_class.index(min_value)
        label_per_class[min_index] += abs(b)

    return label_per_class

def x_u_split_Fire(args, labels):

    label_per_class = []
    for i in range(args.num_classes):
        try:
            count = np.count_nonzero(labels == i)
            len_of_dataset = len(labels)
            class_num = math.ceil(  count  * (args.num_labeled/len_of_dataset) )
            label_per_class.append(class_num)
        except:
            print("split labels error")

    # label_per_class = args.num_labeled // args.num_classes   # 12 / 2 = 6
    label_per_class = adjust_label_per_class(label_per_class, args.num_labeled)
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        try:
            idx = np.where(labels == i)[0]
            a = len(idx)
            idx = np.random.choice(idx, label_per_class[i], False)
            labeled_idx.extend(idx)
        except:
            print("error idx length = ", a, '\n')
            print("error class = ", i)

        # idx = np.where(labels == i)[0]
        # a = len(idx)
        # idx = np.random.choice(idx, label_per_class, False)
        # labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    same = []
    # same = np.array(same)
    for i in range(0,len(labeled_idx)):
        for j in range(0,len(unlabeled_idx)):
            if labeled_idx[i] == unlabeled_idx[j]:
                same.append(unlabeled_idx[j])

    unlabeled_idx = numpy.delete(unlabeled_idx, same)

    return labeled_idx, unlabeled_idx

def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes   # 12 / 2 = 6
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        try:
            idx = np.where(labels == i)[0]
            a = len(idx)
            idx = np.random.choice(idx, label_per_class, False)
            labeled_idx.extend(idx)
        except:
            print("error idx length = ", a, '\n')
            print("error class = ", i)

        # idx = np.where(labels == i)[0]
        # a = len(idx)
        # idx = np.random.choice(idx, label_per_class, False)
        # labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    a = args.num_labeled
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    same = []
    # same = np.array(same)
    for i in range(0,len(labeled_idx)):
        for j in range(0,len(unlabeled_idx)):
            if labeled_idx[i] == unlabeled_idx[j]:
                same.append(unlabeled_idx[j])

    unlabeled_idx = numpy.delete(unlabeled_idx, same)

    return labeled_idx, unlabeled_idx


def back_index(args, labels):
    # label_per_class = args.num_labeled // args.num_classes   # 12 / 2 = 6
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        try:
            idx = np.where(labels == i)[0]
            a = len(idx)
            # idx = np.random.choice(idx, label_per_class, False)
            labeled_idx.extend(idx)
        except:
            print("error idx length = ", a, '\n')
            print("error class = ", i)

        # idx = np.where(labels == i)[0]
        # a = len(idx)
        # idx = np.random.choice(idx, label_per_class, False)
        # labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    a = args.num_labeled
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    same = []
    # same = np.array(same)
    for i in range(0,len(labeled_idx)):
        for j in range(0,len(unlabeled_idx)):
            if labeled_idx[i] == unlabeled_idx[j]:
                same.append(unlabeled_idx[j])

    unlabeled_idx = numpy.delete(unlabeled_idx, same)

    return labeled_idx, unlabeled_idx

def get_ucf101_ssl(dataset,indexs):
    ssl_dataset = dataset
    ilen = len(indexs)
    data = []
    target = []
    

    for i in range(0,(len(indexs))):
        if i == 0:
            aa = 0

        # aa = ssl_dataset.fnames[indexs[i]]
        # bb = ssl_dataset.label_array[indexs[i]] + 1


        #ssl_dataset.fnames 0-13319
        #ssl_dataset.label_array 0-13319
        # try:
        a = indexs[i]
        b = ssl_dataset.fnames.__len__()
        data.append(ssl_dataset.fnames[indexs[i]])
        # target.append(ssl_dataset.label_array[indexs[i]]+1)
        target.append(ssl_dataset.label_array[indexs[i]])
        # except:
        #     print()



    ssl_dataset.fnames = data
    ssl_dataset.label_array = np.array(target)
    return ssl_dataset





