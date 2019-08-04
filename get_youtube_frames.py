import cv2
import youtube_dl
import json
import os
import pickle
from re import sub
from decimal import Decimal
from key_frames_extractor import KeyFrameExtractor
import shutil

# the dataset root location
DATASET_ROOT = './feature_extractor/youtube_data/'


def download_frames(youtubeurl_id, video_dir):
    """Download the frames and store them in the corresponding location"""
    # resize the keyframes to this height and width
    frameHeight = 360
    frameWidth = 480

    video_url = 'https://www.youtube.com/watch?v=' + youtubeurl_id
    frmt = 0

    ydl_opts = {}

    # create youtube-dl object
    ydl = youtube_dl.YoutubeDL(ydl_opts)

    try:
        # set video url, extract video information
        info_dict = ydl.extract_info(video_url, download=False)
    except Exception as e:
        print(e)
        return False

    # make directories for saving the video and its corresponding frames
    os.mkdir(video_dir)
    os.mkdir(video_dir+'/frames/')

    # get video formats available
    formats = info_dict.get('formats', None)

    for f in formats:
        # try to select the higher resolution and proceed on to lower ones
        if f.get('format_note', None) == '480p':
            frmt = f
        elif f.get('format_note', None) == '360p':
            frmt = f
        elif f.get('format_note', None) == '240p':
            frmt = f
        elif f.get('format_note', None) == '144p':
            frmt = f

    if type(frmt) is dict:
        ydl = youtube_dl.YoutubeDL({'format': frmt['format_id'], 'outtmpl': video_dir+youtubeurl_id})
    else:
        return False

    # download the video to the corresponding location
    try:
        ydl.download([video_url])
    except Exception as e:
        print(e)

    # open url with opencv
    cap = cv2.VideoCapture(video_dir+youtubeurl_id)

    # check if url was opened
    if not cap.isOpened():
        print('video not opened')
        return False

    # find keyframes
    kf = KeyFrameExtractor(cap)
    # get a list of keyframes
    frames = kf.extract_keyframes()

    cap.set(cv2.CAP_PROP_POS_MSEC, 0.0)

    # check framerate
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frameCount):
        # read frame
        ret, frame = cap.read()

        if not ret:
            break

        # only save the keyframes
        if i in frames:
            frame = cv2.resize(frame, (frameWidth, frameHeight))
            cv2.imwrite(video_dir+'/frames/'+str(i)+'.jpg', frame)

    # release VideoCapture
    cap.release()
    # remove the YouTube video after processing is complete
    os.remove(video_dir+youtubeurl_id)

    return True


def create_trailer_dataset(json_file, start, end):
    f = open(json_file, "r")
    titles = json.load(f)
    count = 0
    titles_keys = list(titles.keys())[start:end]

    for idx, tconst in enumerate(titles_keys):
        print(idx, tconst)
        title = titles[tconst]
        if title['YoutubeURL'] is not '':
            video_dir = DATASET_ROOT+'/'+title['YoutubeURL']+'/'
            # if the download_frames returns false then there was an error, only add the YoutubeURL to video_ids
            # when the download/_frames was successfully executed
            if not download_frames(title['YoutubeURL'], video_dir):
                continue
            y = Decimal(sub(r'[^\d.]', '', title['BoxOffice']))
            # create a file titled y and save the movie budget in the video directory
            with open(video_dir+'y', "wb") as f:
                pickle.dump(y, f)
            count += 1
    return True


def data_pruning():
    for vid in os.listdir(DATASET_ROOT):
        frames_num = len(os.listdir(DATASET_ROOT + '/' + vid + '/frames/'))
        if frames_num < 32:
            # shutil.rmtree(DATASET_ROOT + '/' + vid)
            print(vid, frames_num)


# create_trailer_dataset('crawled_revenue.json', 7000, 7260)
data_pruning()
