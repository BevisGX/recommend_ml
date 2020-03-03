import mat4py
import numpy as np


def load_one_data(data_path, save_path):
    '''
    the npy file is a np array with  [user_id, video_id, score]
    the score is in range [1,5]
    '''
    datas = mat4py.loadmat(data_path)
    clicks = datas['video']
    video_num = 0
    user_num = 0
    for click in clicks:
        video_num = max(video_num, click[1])
        user_num = max(user_num, click[0])
        # click[2] = (click[4] - click[3]) > (click[2] * watch_need_time)
    # video_click = np.zeros(video_num, dtype=int)
    uesr_video_rate = dict()
    for click in clicks:
        if uesr_video_rate.get(click[0]) is None:
            uesr_video_rate[click[0]] = dict()
        if uesr_video_rate[click[0]].get(click[1]) is None:
            # 使用观看时间/节目时长 再乘上系数作为评分
            uesr_video_rate[click[0]][click[1]] = 2.5 * (click[4] - click[3]) / click[2]
        else:
            uesr_video_rate[click[0]][click[1]] += 2.5 * (click[4] - click[3]) / click[2]
    new_clicks = []
    for useri in uesr_video_rate:
        for videoi in uesr_video_rate[useri]:
            uesr_video_rate[useri][videoi] = round(max(1.0, uesr_video_rate[useri][videoi]), 3)
            uesr_video_rate[useri][videoi] = round(min(5.0, uesr_video_rate[useri][videoi]), 3)
            new_clicks.append([useri, videoi, uesr_video_rate[useri][videoi]])
    new_clicks = sorted(new_clicks, key=lambda x: (x[0]))
    np_clicks = np.asarray(new_clicks, dtype=np.float32)
    np.save(save_path, np_clicks)


if __name__ == '__main__':
    data_path = 'D:/projects/recommend_ml/data/one_day_dataset/'
    file_name = 'video_GH_01'
    data_path = data_path + file_name + '.mat'
    save_path = 'D:/projects/recommend_ml/data/dataset1/'
    save_path = save_path + file_name
    watch_need_time = 0.2
    load_one_data(data_path, save_path)
    pass
