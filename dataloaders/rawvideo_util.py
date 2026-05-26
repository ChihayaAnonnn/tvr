import cv2
import numpy as np
import torch as th
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

# pytorch=1.7.1
# pip install opencv-python

class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=224, framerate=-1, ):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)


    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        # Samples a frame sample_fp X frames.
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
        
        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0:
            interval = 1

        # Use pure-Python ints to avoid OpenCV type issues with numpy scalar types.
        inds = [int(ind) for ind in range(0, fps, interval)]  # ------
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]


        ret = True
        images = []
        # ------------------------- get frame
        for sec in range(start_sec, end_sec + 1):  # seconds
            if not ret:
                break
            sec_base = int(sec * fps)
            for ind in inds:  # fps
                # cv2 expects a double-compatible scalar; numpy scalars may fail on some builds.
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(sec_base + int(ind)))
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))
        # -------------------------
        cap.release()

        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return {'video': video_data}

    def get_video_data(self, video_path, start_time=None, end_time=None):
        image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
        return image_input

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def get_raw_video_data(self, video_path, start_time=None, end_time=None):
        """解码视频帧（不做 CLIP 预处理），返回原始帧列表。

        用于 TQFS 等需要在原始帧上计算质量指标的场景。
        返回列表而非 stack 后的张量，避免 np.stack 改变数组内部结构
        导致 cv2 函数无法识别。

        Returns:
            dict: {'video': List[np.ndarray]}，每个元素为 [H, W, 3] BGR uint8，
                  若解码失败则 {'video': []}。
        """
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30

        total_duration = (frame_count + fps - 1) // fps
        start_sec, end_sec = 0, total_duration
        if start_time is not None:
            start_sec = start_time
            end_sec = min(end_time, total_duration)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        interval = max(1, fps // max(self.framerate, 1)) if self.framerate > 0 else 1
        inds = list(range(0, fps, interval))[:max(self.framerate, 1) if self.framerate > 0 else fps]

        images = []
        ret = True
        for sec in range(start_sec, end_sec + 1):
            if not ret:
                break
            sec_base = int(sec * fps)
            for ind in inds:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(sec_base + int(ind)))
                ret, frame = cap.read()
                if not ret:
                    break
                images.append(frame)  # BGR uint8 [H, W, 3]，cv2 原生数组
        cap.release()

        return {'video': images}

    def preprocess_raw_frames(self, raw_frames_list):
        """对原始帧列表应用 CLIP 预处理（Resize + CenterCrop + RGB + Normalize）。

        Args:
            raw_frames_list: List[np.ndarray]，每个元素为 [H, W, 3] BGR uint8。

        Returns:
            th.Tensor: shape [N, 1, 3, size, size] 的 float32 张量（已归一化）。
        """
        frames = []
        for frame_bgr in raw_frames_list:
            frame_rgb = cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(self.transform(pil_img))
        return th.stack(frames).unsqueeze(1)  # [N, 1, 3, size, size]

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

# An ordinary video frame extractor based CV2
RawVideoExtractor = RawVideoExtractorCV2