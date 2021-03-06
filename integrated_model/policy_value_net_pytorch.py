import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import io

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # ResNet
        self.res_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.res_conv1_bn = nn.BatchNorm2d(256)
        self.res_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.res_conv2_bn = nn.BatchNorm2d(256)

    def forward(self, input):
        x = F.relu(self.res_conv1_bn(self.res_conv1(input)))
        x = self.res_conv2_bn(self.res_conv2(x))
        x = x + input
        x = F.relu(x)
        return x


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height,n_resnet,in_channel):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.n_resent = n_resnet
        self.in_channel = in_channel

        # Inoput layer
        self.conv0=nn.Conv2d(self.in_channel, 256 , kernel_size=3 , padding=1)
        self.conv0_bn = nn.BatchNorm2d(256)

        # # ResNet
        self.resnets = nn.ModuleList([ResNet() for i in range(n_resnet)])

        # Policy head
        self.act_conv1 = nn.Conv2d(256, 2, kernel_size=1)
        self.act_conv1_bn = nn.BatchNorm2d(2)
        self.act_fc1 = nn.Linear(2*board_width*board_height, board_width*board_height)

        # Value head
        self.val_conv1 = nn.Conv2d(256, 1, kernel_size=1)
        self.val_conv1_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(board_width*board_height, 256)
        self.val_fc2 = nn.Linear(256, 1)

    def forward(self, state_input):

        # Common layers
        x = F.relu(self.conv0_bn(self.conv0(state_input)))

        for l in self.resnets:
            x = l(x)

        # Policy head
        x_act = F.relu(self.act_conv1_bn(self.act_conv1(x)))
        x_act = x_act.view(-1, 2*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act))

        # Value head
        x_val = F.relu(self.val_conv1_bn(self.val_conv1(x)))
        x_val = x_val.view(-1, self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))

        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height, n_resnet,in_channel,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.n_resnet = n_resnet
        self.in_channel = in_channel
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height,n_resnet,in_channel).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height,n_resnet,in_channel)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            self.policy_value_net.load_state_dict(torch.load(model_file))


    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state(state_representation_channel = self.in_channel).reshape(
                -1, self.in_channel, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.item(), entropy.item()
        #for pytorch version >= 0.5 please use the following line instead.
        #return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
