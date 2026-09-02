from aloha_dataset import SurrolAlohaDataset
from simple_aloha import AlohaTransformer3
import tqdm
import torch
from torch.utils.data import DataLoader

if __name__ == '__main__':
    surrol_aloha_dataset = SurrolAlohaDataset('/research/d1/gds/jwfu/SurRoL_science_robotics_experiment/experiment_data/needlepick_aloha')
    aloha_model = AlohaTransformer3(action_length=5, hidden_dim=256, robot_state_dim=10, action_size=5)
    aloha_model.load_state_dict(torch.load('/research/d1/gds/jwfu/SurRoL_science_robotics_experiment/tmp_workspace/surrol_aloha_needlepick_new_2.pth', map_location='cpu'))

    dataloader = DataLoader(surrol_aloha_dataset, batch_size=128, shuffle=True, num_workers=4)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aloha_model.to(device)

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(aloha_model.parameters(), lr=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # Number of epochs
    num_epochs = 5000

    # Training loop
    for epoch in range(num_epochs):
        loss_epoch = 0
        for images, robot_state, actions in tqdm.tqdm(dataloader):
            images = images.to(device).to(torch.float32)
            # print(f'images size: {images.size()}')
            robot_state = robot_state.to(device).to(torch.float32)
            # print(f'robot_state size: {robot_state.size()}')
            actions = actions.to(device).to(device).to(torch.float32) 
            # print(f'actions size: {actions.size()}')
            # Forward pass
            outputs = aloha_model(images, robot_state)
            # print(f'outputs: {outputs.size()}')
            # print(f'actions size: {actions.size()}')

            loss = 0
            for lvl in range(outputs.size(0)):
                loss += criterion(outputs[0], actions)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.mean()
        # scheduler.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_epoch.item():.4f}')
        if epoch % 100 == 0:
            torch.save(aloha_model.state_dict(), 'surrol_aloha_needlepick_new_2.pth')