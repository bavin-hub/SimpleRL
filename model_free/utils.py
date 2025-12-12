from moviepy.editor import VideoFileClip
import imageio.v2 as imageio
import os


# def save_video(env, env_name):
#     video_folder = './videos'

#     def video_trigger(episode_id: int) -> bool:
#         # return episode_id == 0 or episode_id % 50 == 0
#         return True
    
#     # env = RecordVideo(env, "./videos/output_videos", episode_trigger=lambda episode_id: True)

#     env = RecordVideo(env, 
#                       video_folder=video_folder,
#                       episode_trigger=video_trigger,
#                       name_prefix=f'{env_name}')
    
    # return env


def mp4_to_gif(env_name):
    curr_dir = os.getcwd()
    videos_path = os.path.join(curr_dir, 'videos')
    if os.path.isdir(videos_path):
        pass
    else:
        os.makedirs(videos_path)

    path = f'{videos_path}/test-episode-0.mp4'
    gif_path = './videos/test.gif'
    
    clip = VideoFileClip(path)

    clip = clip.resize(0.6)

    clip.write_gif(gif_path, program='imageio', fps=15)
    


def save_gif(array_of_frames, env_name, ep_num, fps=24):
    curr_dir = os.getcwd()
    videos_path = os.path.join(curr_dir, 'videos')
    if os.path.isdir(videos_path):
        pass
    else:
        os.makedirs(videos_path)


    file_name = f'{env_name}_eval_ep-{ep_num}.gif'
    video_path = f'{videos_path}/{file_name}'
    
    imageio.mimsave(video_path, array_of_frames, fps=fps)



