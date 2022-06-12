from collections import namedtuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cpu")

RecurrentBatch = namedtuple('RecurrentBatch', 'o a r d m')


def as_probas(positive_values: np.array) -> np.array:
    return positive_values / np.sum(positive_values)


def as_tensor_on_device(np_array: np.array):
    return torch.tensor(np_array).float().to(device)


class RecurrentReplayBuffer:

    """Use this version when num_bptt == max_episode_len"""

    def __init__(
        self,
        o_dim,
        a_dim,
        max_episode_len,  # this will also serve as num_bptt
        capacity,
        batch_size,
        segment_len=None,  # for non-overlapping truncated bptt, maybe need a large batch size
    ):

        # placeholders

        self.o = np.full((capacity, max_episode_len + 1, o_dim), np.nan)
        self.a = np.full((capacity, max_episode_len, a_dim), np.nan)
        self.r = np.full((capacity, max_episode_len, 1), np.nan)
        self.d = np.full((capacity, max_episode_len, 1), np.nan)
        self.m = np.full((capacity, max_episode_len, 1), np.nan)
        self.ep_len = np.zeros((capacity,))
        self.ready_for_sampling = np.zeros((capacity,))

        # pointers

        self.episode_ptr = 0
        self.time_ptr = 0

        # trackers

        self.starting_new_episode = True
        self.num_episodes = 0

        # hyper-parameters

        self.capacity = capacity
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.batch_size = batch_size

        self.max_episode_len = max_episode_len

        if segment_len is not None:
            assert max_episode_len % segment_len == 0  # e.g., if max_episode_len = 1000, then segment_len = 100 is ok

        self.segment_len = segment_len
        
    def __str__(self):
        to_print = ""        
        to_print += "Observations: {}\n{}".format(str(self.o.shape), self.o.__str__())
        to_print += "\n\nActions: {}\n{}".format(str(self.a.shape), self.a.__str__())
        to_print += "\n\nRewards: {}\n{}".format(str(self.r.shape), self.r.__str__())
        to_print += "\n\nDones: {}\n{}".format(str(self.d.shape), self.d.__str__())
        to_print += "\n\nM?: {}\n{}".format(str(self.m.shape), self.m.__str__())
        to_print += "\n\nEpisodes: {}. New episode: {}.".format(self.num_episodes, self.starting_new_episode)
        to_print += "\nEpisode {}, Step {}.".format(self.episode_ptr +1, self.time_ptr+1)
        return(to_print)

    def push(self, o, a, r, no, d, cutoff):

        # zero-out current slot at the beginning of an episode

        if self.starting_new_episode:

            self.o[self.episode_ptr] = np.nan
            self.a[self.episode_ptr] = np.nan
            self.r[self.episode_ptr] = np.nan
            self.d[self.episode_ptr] = np.nan
            self.m[self.episode_ptr] = np.nan
            self.ep_len[self.episode_ptr] = 0
            self.ready_for_sampling[self.episode_ptr] = 0

            self.starting_new_episode = False

        # fill placeholders

        self.o[self.episode_ptr, self.time_ptr] = o
        self.a[self.episode_ptr, self.time_ptr] = a
        self.r[self.episode_ptr, self.time_ptr] = r
        self.d[self.episode_ptr, self.time_ptr] = d
        self.m[self.episode_ptr, self.time_ptr] = 1
        self.ep_len[self.episode_ptr] += 1

        if d or cutoff:

            # fill placeholders

            self.o[self.episode_ptr, self.time_ptr+1] = no
            self.ready_for_sampling[self.episode_ptr] = 1

            # reset pointers

            self.episode_ptr = (self.episode_ptr + 1) % self.capacity
            self.time_ptr = 0

            # update trackers

            self.starting_new_episode = True
            if self.num_episodes < self.capacity:
                self.num_episodes += 1

        else:

            # update pointers

            self.time_ptr += 1

    def sample(self):

        assert self.num_episodes >= self.batch_size

        # sample episode indices

        options = np.where(self.ready_for_sampling == 1)[0]
        ep_lens_of_options = self.ep_len[options]
        probas_of_options = as_probas(ep_lens_of_options)
        choices = np.random.choice(options, p=probas_of_options, size=self.batch_size)

        ep_lens_of_choices = self.ep_len[choices]

        if self.segment_len is None:

            # grab the corresponding numpy array
            # and save computational effort for lstm

            max_ep_len_in_batch = int(np.max(ep_lens_of_choices))

            o = self.o[choices][:, :max_ep_len_in_batch+1, :]
            a = self.a[choices][:, :max_ep_len_in_batch, :]
            r = self.r[choices][:, :max_ep_len_in_batch, :]
            d = self.d[choices][:, :max_ep_len_in_batch, :]
            m = self.m[choices][:, :max_ep_len_in_batch, :]

            # convert to tensors on the right device

            o = as_tensor_on_device(o).view(self.batch_size, max_ep_len_in_batch+1, self.o_dim)
            a = as_tensor_on_device(a).view(self.batch_size, max_ep_len_in_batch, self.a_dim)
            r = as_tensor_on_device(r).view(self.batch_size, max_ep_len_in_batch, 1)
            d = as_tensor_on_device(d).view(self.batch_size, max_ep_len_in_batch, 1)
            m = as_tensor_on_device(m).view(self.batch_size, max_ep_len_in_batch, 1)

            return RecurrentBatch(o, a, r, d, m)

        else:

            num_segments_for_each_item = np.ceil(ep_lens_of_choices / self.segment_len).astype(int)

            o = self.o[choices]
            a = self.a[choices]
            r = self.r[choices]
            d = self.d[choices]
            m = self.m[choices]

            o_seg = np.zeros((self.batch_size, self.segment_len + 1, self.o_dim))
            a_seg = np.zeros((self.batch_size, self.segment_len, self.a_dim))
            r_seg = np.zeros((self.batch_size, self.segment_len, 1))
            d_seg = np.zeros((self.batch_size, self.segment_len, 1))
            m_seg = np.zeros((self.batch_size, self.segment_len, 1))

            for i in range(self.batch_size):
                start_idx = np.random.randint(num_segments_for_each_item[i]) * self.segment_len
                o_seg[i] = o[i][start_idx:start_idx + self.segment_len + 1]
                a_seg[i] = a[i][start_idx:start_idx + self.segment_len]
                r_seg[i] = r[i][start_idx:start_idx + self.segment_len]
                d_seg[i] = d[i][start_idx:start_idx + self.segment_len]
                m_seg[i] = m[i][start_idx:start_idx + self.segment_len]

            o_seg = as_tensor_on_device(o_seg)
            a_seg = as_tensor_on_device(a_seg)
            r_seg = as_tensor_on_device(r_seg)
            d_seg = as_tensor_on_device(d_seg)
            m_seg = as_tensor_on_device(m_seg)

            return RecurrentBatch(o_seg, a_seg, r_seg, d_seg, m_seg)
        
buffer = RecurrentReplayBuffer(o_dim = 3, a_dim = 2, capacity = 3, batch_size = 2, max_episode_len = 5)

buffer.push([1,2,3], [1,2], [10], [4, 5, 6], False, False)
buffer.push([1,2,3], [1,2], [10], [4, 5, 6], False, False)
buffer.push([1,2,3], [1,2], [10], [4, 5, 6], False, False)
buffer.push([1,2,3], [1,2], [10], [4, 5, 6], False, False)
buffer.push([1,2,3], [1,2], [10], [4, 5, 6], True,  True)
buffer.push([1,2,3], [1,2], [10], [4, 5, 6], False, False)
buffer.push([1,2,3], [1,2], [10], [4, 5, 6], True,  True)
buffer.push([1,2,3], [1,2], [10], [4, 5, 6], False, False)
buffer.push([1,2,3], [1,2], [10], [4, 5, 6], True,  True)
print(buffer)

batch = buffer.sample()
print("\n\nA batch:\n\n")
print(batch)




class Trans(nn.Module):
    
    def __init__(self):
        super(Trans, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size = 3,
            hidden_size = 10,
            batch_first = True)
        
    def forward(self, obs, hidden = None):
        self.lstm.flatten_parameters()
        if(hidden == None): x, hidden = self.lstm(obs)
        else:               x, hidden = self.lstm(obs, (hidden[0], hidden[1]))
        return(x)

trans = Trans()
encoding = trans(batch.o)

print("\n\nA transitioner output:\n\n")
print(encoding.shape)
print(encoding)





class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        
        self.lin = nn.Sequential(
            nn.Linear(10+2, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 1))
        

    def forward(self, encode, action):
        x = torch.cat((encode[:,:-1,:], action), dim=-1)
        x = self.lin(x)
        return x
    
crit = Critic()
v = crit(encoding, batch.a)

print("\n\nA critic output:\n\n")
print(v.shape)
print(v)

from torch.distributions import Normal
def careful_log_prob(mu, std, e):
    ongoing = ~torch.isnan(mu)
    mu[ongoing == False] = 0
    std[ongoing == False] = 1
    log_prob = Normal(mu, std).log_prob(mu + e * std)
    log_prob[ongoing == False] = np.nan
    return(log_prob)

def careful_loss(predict, target, loss_function):
    loss = 0
    for s in range(predict.shape[1]):
        p = predict[:,s]
        t = target[:,s]
        ongoing = ~torch.isnan(p)
        # Try making ongoing just one layer! 
        loss += loss_function(p[ongoing], t[ongoing])
    return(loss)
            

loss = careful_loss(v, v.detach(), F.mse_loss)

print("\n\nLoss:\n\n")
print(loss)