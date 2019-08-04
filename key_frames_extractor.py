import cv2
import numpy as np
import torch
import torch.nn.functional as f


class KeyFrameExtractor:
    def __init__(self, cap):
        self.cap = cap
        self.frameRate = int(cap.get(cv2.CAP_PROP_FPS))
        # sample rate of frames per second. so if equals to 1, then sample 1 frame per second from the video
        self.frameSampleRate = 1
        # the number of frames to extract from the video
        self.framesNum = 64
        # the window of frames that will be used to find the LUV mutual difference
        self.frameWindow = 3
        # total number of frames in the video
        self.frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # video height and width
        self.frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # video channels
        self.frameChannel = 3
        # the weight for blur measure
        self.blurWeight = 1000
        self.informationWeight = 100
        # setting the right device for pytorch tensors and datatypes
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def extract_keyframes(self):
        # defining the laplacian filter
        laplacian_filter = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                                        dtype=torch.float, device=self.device)
        laplacian_filter = laplacian_filter.expand(1, 3, 3, 3)
        diff = torch.tensor([0, 0], dtype=torch.float, device=self.device)
        diff = diff.unsqueeze(0)
        # int(self.frameRate/self.frameSampleRate) steps through the range from 0 to total framecount
        frames_indexes = range(0, self.frameCount, int(self.frameRate / self.frameSampleRate))
        frames = torch.zeros([len(frames_indexes), self.frameHeight, self.frameWidth, self.frameChannel],
                             dtype=torch.float, device=self.device)
        frames_blur = torch.zeros(len(frames_indexes), dtype=torch.float, device=self.device)
        img_entropy = torch.zeros(len(frames_indexes), dtype=torch.float, device=self.device)

        for i, frames_id in enumerate(frames_indexes):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frames_id)
            # read frame
            ret, frame = self.cap.read()

            if not ret:
                break

            try:
                # convert to LUV colorspace, this is an additive colorspace which is useful for finding
                # color differences
                frame_LUV = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
            except Exception as e:
                print(e)
                continue
            # converting frame to a numpy tensor on the device i.e. probably the GPU
            frame = torch.from_numpy(frame).to(self.device).unsqueeze(0).float()
            # convert from HxWxC to CxHxW
            frame = frame.permute(0, 3, 1, 2)
            img_entropy[i] = calculate_entropy_frame(frame)
            frames[i] = torch.from_numpy(frame_LUV).to(self.device)
            # apply laplacian filter
            frame_lap = f.conv2d(frame, laplacian_filter, stride=1, padding=1)
            # variance of laplacian as a blur measure
            frames_blur[i] = torch.var(frame_lap)

        for i in range(int(self.frameWindow / 2), len(frames_indexes) - int(self.frameWindow / 2)):
            frame_diff = 0
            frame_mutual_information = 0
            for j in range(-int(self.frameWindow / 2), int(self.frameWindow / 2) + 1):
                probs = calculate_joint_histogram(frames[i], frames[i + j])
                joint_entropy_i_j = calculate_joint_entropy(torch.from_numpy(probs).to(self.device))
                mutual_information = calculate_mutual_information(img_entropy[i], img_entropy[i + j], joint_entropy_i_j)
                frame_diff += torch.sum(torch.abs((frames[i] - frames[i + j])))
                frame_mutual_information += mutual_information

            # add the frame differences, normalized by len of frameWindow and the blur measure
            score = torch.tensor([frame_diff / self.frameWindow + frames_blur[i] -
                                  frame_mutual_information * self.informationWeight, frames_indexes[i]],
                                 dtype=torch.float, device=self.device)
            score = score.unsqueeze(0)
            diff = torch.cat((diff, score), dim=0)

        # sort the diff array wrt the difference column
        mask = torch.argsort(diff[:, 0], descending=True)
        diff = diff[mask]

        # return the frame number list
        return diff[:, 1][:self.framesNum]


def calculate_entropy_frame(frame):
    epsilon = 1e-5
    probs = torch.histc(frame, bins=255, min=0, max=255)
    probs = f.normalize(probs.double(), dim=0)
    probs += epsilon
    log_probs = torch.log(probs)
    return -torch.sum(probs * log_probs)


def calculate_joint_histogram(frame_i, frame_j):
    frame_i = frame_i.cpu().numpy()
    frame_j = frame_j.cpu().numpy()
    probs = np.histogram2d(frame_i.flatten(), frame_j.flatten(),
                           bins=np.arange(0, 257), density=True)[0]
    return probs


def calculate_joint_entropy(probs):
    epsilon = 1e-5
    probs += epsilon
    log_probs = torch.log(probs)
    return -torch.sum(probs * log_probs)


def calculate_mutual_information(entropy_i, entropy_j, joint_entropy_i_j):
    return entropy_i + entropy_j - joint_entropy_i_j
