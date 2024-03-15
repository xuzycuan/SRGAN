# -*- coding: utf-8 -*-

import time

import cv2
import imutils
import numpy as np
from tqdm import tqdm

from models import Generator
from utils import *

# 测试图像
# imgPath = './results/girl16.jpg'

# 模型参数
large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3  # 中间层卷积的核大小
n_channels = 64  # 中间层通道数
n_blocks = 16  # 残差模块数量
scaling_factor = 4  # 放大比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 预训练模型
    srgan_checkpoint = "./models/checkpoint_srgan.pth"
    # srresnet_checkpoint = "./results/checkpoint_srresnet.pth"

    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(srgan_checkpoint, map_location=device)
    # checkpoint = torch.load(srresnet_checkpoint, map_location=device)
    generator = Generator(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor)

    # generator = SRResNet(large_kernel_size=large_kernel_size,
    #                      small_kernel_size=small_kernel_size,
    #                      n_channels=n_channels,
    #                      n_blocks=n_blocks,
    #                      scaling_factor=scaling_factor)

    generator = generator.to(device)
    generator.load_state_dict(checkpoint['generator'])

    generator.eval()
    model = generator

    # 加载视频
    video_path = "./images/beautifulGirl.mp4"
    vs = cv2.VideoCapture(video_path)
    (W, H) = (None, None)
    frameIndex = 0  # 视频帧数

    # 试运行，获取总的画面帧数
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] 视频总帧数：{}".format(total))

    # 若读取失败，报错退出
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ret, frame = vs.read()
    vw = frame.shape[1] * scaling_factor
    vh = frame.shape[0] * scaling_factor
    print("[INFO] 视频尺寸：{} * {}".format(vw, vh))
    output_video = cv2.VideoWriter(video_path.replace(".mp4", "-det.avi"), fourcc, 20.0, (vw, vh))  # 处理后的视频对象
    output_video_orig = cv2.VideoWriter(video_path.replace(".mp4", "-det-orig.avi"),
                                        fourcc, 20.0, (vw, vh))  # 处理后的视频对象

    # 遍历视频帧进行检测
    for fr in tqdm(range(total)):
        # 从视频文件中逐帧读取画面
        (grabbed, frame) = vs.read()

        # 若grabbed为空，表示视频到达最后一帧，退出
        if not grabbed:
            break

        # 获取画面长宽
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # 加载图像
        # img = Image.open(frame, mode='r')
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.convert('RGB')

        # 双线性上采样
        # Bicubic_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Bicubic_img = Image.fromarray(frame)
        Bicubic_img = Bicubic_img.convert('RGB')
        Bicubic_img = Bicubic_img.resize((int(Bicubic_img.width * scaling_factor), int(Bicubic_img.height * scaling_factor)), Image.BICUBIC)
        # Bicubic_img.save('./results/test_bicubic.jpg')

        # 图像预处理
        lr_img = convert_image(img, source='pil', target='imagenet-norm')
        lr_img.unsqueeze_(0)

        # 记录时间
        start = time.time()

        # 转移数据至设备
        lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed

        # 模型推理
        with torch.no_grad():
            sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
            sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
            # sr_img.save('./results/test_srresnet.jpg')

        # 实时显示检测画面
        sr_img = np.array(sr_img)
        Bicubic_img = np.array(Bicubic_img)

        cv2.imshow('Stream', sr_img)
        output_video.write(sr_img)  # 保存标记后的视频
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frameIndex += 1

        output_video_orig.write(Bicubic_img)
        if frameIndex >= total:  # 可设置检测的最大帧数提前退出
            print("[INFO] 运行结束...")
            output_video.release()
            output_video_orig.release()
            vs.release()
            exit()
