import torch
import gym
from simple_aloha import AlohaTransformer3
import torchvision.transforms as transforms
from PIL import Image
import imageio
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),           # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
])

def eval(model, env, with_render=False):
    model.eval()
    state = env.reset()
    # observation, goal = state['observation'], state['desired_goal']
    img = state['img']
    robot_state = state['robot_state']
    goal = state['desired_goal']




    episode_return, episode_length = 0, 0
    
    if with_render:
        render_obss = []
    
    is_jaw_close = False

    for t in range(300):
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        img = transform(img)
        img = img.unsqueeze(0).to('cuda')
        robot_state = torch.from_numpy(np.concatenate([robot_state, goal], axis=0)).unsqueeze(0).to('cuda').to(torch.float32)
        # print(f'img dtype: {img.dtype}')
        # print(f'robot_state dtype: {robot_state.dtype}')
        actions = model(img, robot_state)

        action = actions[-1][0][0]
        action = action.detach().cpu().numpy()

        if action[-1] < 0:
            is_jaw_close = True
        # if is_jaw_close:
        #     action[-1] = -1.0

        if with_render:
            render_obs, _, _ = env.ecm.render_image()
            render_obss.append(render_obs)

        state, reward, done, info = env.step(action)

        episode_return += reward
        episode_length += 1
        img = state['img']
        robot_state = state['robot_state']
        goal = state['desired_goal']

        is_success = info['is_success']
        print(f'is_success: {is_success}')

        if done or is_success:
            break

    return render_obss

if __name__ == '__main__':
    aloha_model = AlohaTransformer3(action_length=5, hidden_dim=256, robot_state_dim=10, action_size=5)
    print(f'load model success')
    aloha_model.load_state_dict(torch.load('/research/d1/gds/jwfu/SurRoL_science_robotics_experiment/tmp_workspace/surrol_aloha_needlepick_new_2_sr100.pth', map_location='cpu'))
    aloha_model.to('cuda')
    print(f'to cuda success')
    env = gym.make('NeedlePickRLVision-v0', render_mode='rgb_array')  # 'human'
    # env = gym.make('NeedleReach-v0', render_mode='rgb_array')  # 'human'


    render_obss = eval(aloha_model, env, True)
    # print(f'render_obss: {render_obss}')

    writer = imageio.get_writer('video_aloha_2.mp4', fps=20)
    for img in render_obss:
        writer.append_data(img)
    writer.close()
