import mat4py
import numpy as np


def load_one_data(data_path, save_path, watch_need_time, thresold_forvideo):

    datas = mat4py.loadmat(data_path)
    clicks = datas['video']
    video_num = 0
    user_num = 0
    for click in clicks:
        video_num = max(video_num, click[1])
        user_num = max(user_num, click[0])
        click[2] = (click[4] - click[3]) > (click[2] * watch_need_time)
    video_click = np.zeros(video_num, dtype=int)
    for click in clicks:
        video_click[click[2]] += 1


if __name__ == '__main__':
    data_path = '../data/one_day_dataset/video_GH_01.mat'
    watch_need_time=0.2
    datas = mat4py.loadmat(data_path)
    clicks = datas['video']
    video_num = 0
    user_num = 0
    for click in clicks:
        video_num = max(video_num, click[1])
        user_num = max(user_num, click[0])
        click[2] = (click[4] - click[3]) > (click[2] * watch_need_time)
    video_click = np.zeros(video_num+1, dtype=int)
    for click in clicks:
        video_click[click[1]] += 1
    pass
