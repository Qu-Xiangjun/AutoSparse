import os
import torch
import torch.nn as nn
import numpy as np
import copy
import json
import heapq
import random

from .space import *
from .utils import Flatten

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print("Number of available GPUs:", num_gpus)

    device_id = 0
    device = torch.device("cuda:" + str(device_id))
    print("Using GPU device:", device)
else:
    print("No GPU available, using CPU instead.")
    device = torch.device("cpu")

class SimpleQNN(nn.Module):
    """
    Simple nn for DQN.
    """
    def __init__(self, state_dim: int, action_dim: int, 
                 width: int = 64, depth: int = 4) -> None:
        super().__init__()
        assert state_dim > 0 and width > 0 and depth >= 2 and action_dim > 1

        self.net = nn.Sequential()
        self.net.add_module("input", nn.Linear(state_dim, width))
        self.net.add_module("input_activate", nn.ReLU())
        for count in range(depth - 2):
            self.net.add_module(f"hidden{count}", nn.Linear(width, width))
            self.net.add_module(
                f"hidden_{count}_activate", nn.Linear(width, width))
        self.net.add_module("output", nn.Linear(width, action_dim))
        self.to(device) 
    
    def forward(self, state: torch.FloatTensor):
        return self.net(state)


class DQNAgent(object):
    def __init__(self, agent_name: str, subspace: SubSpace, input_len: int, 
                 decay: float = 0.9, lr: float = 0.02, epochs: int = 20,
                 train_batch_size: int = 1000) -> None:
        self.name = agent_name
        self.subspace = subspace
        self.major_model = SimpleQNN(
            input_len, self.subspace.num_directions, 64, 3)
        self.target_model = SimpleQNN(
            input_len, self.subspace.num_directions, 64, 3)
        self.memory = [] # (current_state, action, next_state, reward) 
        self.memory_size = 0
        self.model_path = "DQNAgent_SimpleQNN_Model_" + self.name + ".pkl"
        self.data_path = "DQNAgent_SimpleQNN_Data_" + self.name + ".txt"
        self.decay = decay
        self.lr = lr
        self.optimizer = torch.optim.Adadelta(
            self.major_model.parameters(), lr=lr)
        self.train_batch_size = train_batch_size
        self.epochs = epochs
        self.loss_fn = nn.MSELoss()
        
    
    def RandomBatch(self, batch_size):
        """
        Returns
        -------
        ret_entries: List
            All the entries list.
        batch_indices: List
            All the entires indices.
        """
        assert batch_size > 0
        # Sampling without replacement was performed.
        batch_indices = np.random.choice(
            self.subspace.size, size=batch_size)
        ret_entries = self.subspace.GetBatchEntry(batch_indices)
        return ret_entries, batch_indices
    
    def BestBatch(self, batch_size):
        """
        Returns
        -------
        ret_entries: List
            All the entries list.
        batch_indices: List
            All the entires indices.
        """
        assert batch_size > 0 and batch_size <= self.subspace.size
        all_entities = [Flatten(x) for x in self.subspace.all_entries]
        inputs = torch.FloatTensor(all_entities).to(device)
        q_values, _ = torch.max(self.major_model(inputs), -1)
        _, batch_indices = torch.topk(q_values.reshape(-1), batch_size)
        ret_entities = self.subspace.GetBatchEntry(batch_indices)
        return ret_entities, batch_indices
    
    # def RecordBest(self, best_idnex, best_q_value):
    #     self.memory[best_idnex] = best_q_value
    
    def SelectAction(self, inputs_lst, indices_lst, trial, epsilon, gamma):
        """
        Parameters
        ----------
        inputs_lst: List
            Batch states.
        indices_lst: List
            All the states position in subspace.
        trial: List
            End the current run round.
        epsilon: float
        gamma: float

        Returns
        -------
        ret_indices: List
            Next states indecies in subspace.
        ret_directions: List
            Directions set from the current state set to the next state set.
        """
        inputs = [Flatten(x) for x in inputs_lst]
        inputs = torch.FloatTensor(inputs).to(device)
        q_values_lst = self.major_model(inputs)
        ret_indices = []
        ret_directions  = []
        for i, q_value in enumerate(q_values_lst):
            p = np.random.random()
            # As the number of runs increases, the probability of 
            # randomly selecting the direction decreases.
            t = max(epsilon * np.exp(-trial * gamma), 0.1)
            if p <= t:
                direction_pos = np.random.randint(
                    0, self.subspace.num_directions)
            else:
                _, direction_pos = torch.max(q_value, -1)
            direction = self.subspace.GetDirection(direction_pos)
            new_index = self.subspace.NextEntry(
                pos = indices_lst[i], direction = direction)
            ret_indices.append(new_index)
            ret_directions.append(direction)
        return ret_indices, ret_directions

    def SelectionFull(self, index):
        """Get all next states with all directions in subspace.
        
        Returns
        -------
        ret_indices: List
            Next states indecies in subspace.
        ret_directions: List
            Directions set from the current state to all next states.
        """
        new_indices = []
        for d in self.subspace.directions:
            new_indices.append(self.subspace.NextEntry(index, d))
        return new_indices, copy.deepcopy(self.subspace.directions)
    
    def SelectOneAction(self, index, direction):
        """Get next state by direction from currect state index."""
        return self.subspace.NextEntry(index, direction)

    def Train(self, save_model = True):
        batch_size = min(self.memory_size, self.train_batch_size)
        # TODO: Can there only store least recently recorded data?
        batch_data = random.sample(self.memory, k = batch_size)
        pre_states, actions, post_states, rewards = zip(*batch_data)
        pre_states = torch.FloatTensor(pre_states).to(device)
        actions = torch.LongTensor(actions).to(device)
        post_states = torch.FloatTensor(post_states).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        
        for epoch in range(self.epochs):
            q_values = self.major_model(pre_states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            target_q_values = self.target_model(post_states).max(1)[0]
            target_q_values = rewards + self.decay * target_q_values

            loss = self.loss_fn(q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if save_model and ((epoch + 1) % 5 == 0):
                self.SaveModel()
            print(f"[cur/total]=[{epoch+1}/{self.epochs}], loss = {float(loss):.4f}")
    
    def AddData(self, pre_state, action, post_state, reward):
        direction_pos = self.subspace.GetDirectionPos(action)
        assert direction_pos >= 0
        self.memory.append((pre_state, direction_pos, post_state, reward))
        self.memory_size += 1
    
    def GetModelPath(self):
        return self.model_path
    
    def GetDataPath(self):
        return self.data_path

    def SaveModel(self):
        other_params = {
            'decay': self.decay,
            'lr': self.lr,
            'epochs': self.epochs,
        }
        model_state = self.major_model.state_dict()

        torch.save({'model_state': model_state, 'other_params': other_params}, 
                    self.model_path)
    
    def LoadModel(self):
        if not os.path.exists(self.model_path):
            print(f"[Warning] {self.model_path} File does not exist.")
            return
        checkpoint = torch.load(self.model_path)
        self.major_model.load_state_dict(checkpoint['model_state'])
        self.target_model.load_state_dict(self.major_model.state_dict())
        self.decay = checkpoint['other_params']['decay']
        self.lr = checkpoint['other_params']['lr']
        self.epochs = checkpoint['other_params']['epochs']
        print(f"[INFO] Successfully read model of {self.name} from the {self.model_path}.")
    
    def SaveData(self):
        with open(self.data_path, "a") as fout:
            for data in self.memory:
                string = json.dumps(data)
                fout.write(string + "\n")
    
    def LoadData(self):
        if not os.path.exists(self.data_path):
            print(f"[Warning] {self.data_path} File does not exist.")
            return
        with open(self.data_path, "r") as fin:
            for line in fin:
                data = tuple(json.loads(line))
                self.memory.append(data)
                self.memory_size += 1
        print(f"[INFO] Successfully read model of {self.name} from the {self.data_path}.")
    
    def LoadOrCreateModel(self):
        if os.path.exists(self.model_path):
            self.LoadModel()
        if os.path.exists(self.data_path):
            self.LoadData()
    
    def UpdateTargetModel(self):
        self.target_model.load_state_dict(self.major_model.state_dict())


class MemEntry(object):
    def __init__(self, indices, value):
        self.indices = indices
        self.value = value
    
    def __lt__(self, b):
        return self.value < b.value

class DQNAgentGroup(object):
    def __init__(self, agent_group_name: str, space: Space, 
                 decay: float = 0.9, lr: float = 0.02, epochs: int = 20, 
                 train_batch_size: int = 1000):
        """
        DQN agent group for all the subspace agent.'
        Parameters
        ----------
        agent_group_name: str
        space: Space
            The schedule total space object.
        decay: float optional(0.9)
            The decay factor in reward function.
        lr: float optional(0.02)
            Network learning ratio.
        epochs: int optional(20)
            Train epochs count in every train Q-leaning network.
        train_bacth_size: int optional(1000)
            The size of the experience return visit area.
        """
        self.name = agent_group_name
        self.space = space
        self.agent_group = dict()
        for name, subspace in space.items():
            self.agent_group[name] = DQNAgent(
                agent_group_name + "_" + name, subspace, space.dim,
                decay, lr, epochs, train_batch_size
            )
        # Keep track of the data as it appears and used by heap to get topK.
        # TODO: Record all the data as it appears will take too many memory?
        # TODO: But note the list will be used to create test data set?
        self.memory = []
        self.memory_size = 0
        self.visits_set = set()
        self.schedule_data = [] # [schedule_command, value]
    
    @property
    def action_num(self):
        cnt = 0
        for _, agent in self.agent_group.items():
            cnt += agent.subspace.num_directions
        return cnt
    
    def SelectAction(self, indices_lst, values_lst, 
                      trial, epsilon = 0.8, gamma = 0.01):
        """
        Get action from now states by every DQN agents.

        Parameters
        ----------
        indices_lst: List[Dict]
            Batch states indices in every subspace.
        values_lst: List
            All the states performance.
        trial: List
            End the current run round.
        epsilon: float
        gamma: float

        Returns
        -------
        ret_data_lst: List[]
            Element in it will be a tuple:
            (pre_state_indices: Dict, value, subspace.name, 
             action, next_state_indices: Dict)
            `pre_state_indices` indicate last state in every subspace.
            `value` indicate last state's performan for program design.
            `subspace.name` indicate which subspace will change.
            `action` indicate changing direction in the subspace.
            `next_state_indices` indicate next state in every subspace when do a action.
        """
        states_lst = [self.ConvertIndices2FeatureVec(indices) 
                        for indices in indices_lst]
        ret_data_lst = []
        # Sub space name and binded agent.
        for name, agent in self.agent_group.items(): 
            subspace_indices_lst = [indices[name] for indices in indices_lst]
            # Get Next Action
            next_indices_lst, directions_lst = agent.SelectAction(
                states_lst, subspace_indices_lst, trial, epsilon, gamma
            )
            # Pack return dict. Note that you only change the type of 
            # one subspace across all the data, which is equivalent 
            # to doing a Cartesian product.
            indices_lst_cpoy = copy.deepcopy(indices_lst)
            for i, next_indices in enumerate(indices_lst_cpoy):
                next_indices[name] = next_indices_lst[i] # Only do action for a subspace.
                if not self.EverMeet(next_indices):
                    ret_data_lst.append(
                        (indices_lst[i], values_lst[i], name, 
                         directions_lst[i], next_indices)
                    )
        return ret_data_lst
    
    def SelectionFull(self, indices: Dict, value: float, no_repeat=True):
        """Get all next states with all directions in every subspace.
        
        Parameters
        ----------
        indices: Dict
            A config indices in every subspace.
        no_repeat: bool optinal(True)
            Only add never-before-see ones.
        
        Returns
        -------
        ret_data_lst: List[]
            Element in it will be a tuple:
            (pre_state_indices: Dict, value, subspace.name, 
             action, next_state_indices: Dict)
            `pre_state_indices` indicate last state in every subspace.
            `value` indicate last state's performan for program design.
            `subspace.name` indicate which subspace will change.
            `action` indicate changing direction in the subspace.
            `next_state_indices` indicate next state in every subspace when do a action.
        """
        ret_data_lst = []
        for name, index in indices.items(): # Traverse all the sub space.
            next_index_lst, direction_lst = self.agent_group[name].SelectionFull(index)
            for next_index, direction in zip(next_index_lst, direction_lst):
                next_indices = copy.deepcopy(indices)
                next_indices[name] = next_index # Only do action for a subspace.
                if no_repeat and not self.EverMeet(next_indices):
                    ret_data_lst.append(
                        (indices, value, name, direction, next_indices)
                    )
                elif not no_repeat:
                    ret_data_lst.append(
                        (indices, value, name, direction, next_indices)
                    )
        return ret_data_lst
    
    def SelectOneAction(self, indices, name, direction, no_repeat=True):
        """Get next state by direction from currect state index.
        
        Parameters
        ----------
        indices: Dict
            A config indices in every subspace.
        name: str
            Sub space name.
        direction: int
            The action direction in the sub space.
        no_repeat: Bool 
            If ever meet the new indices in the space, return None.
        """
        new_indices = copy.deepcopy(indices)
        agent = self.agent_group[name]
        new_indices[name] = agent.SelectOneAction(indices[name], direction)
        if no_repeat and not self.EverMeet(new_indices):
            return new_indices
        elif not no_repeat:
            return new_indices
        return None

        
    def RandomBatch(self, batch_size):
        """Get a batch size data from all the sub space config.
        Return
        ------
        ret dict{subspace name: (ret_entities, ret_p_values)}
            `key` indicate subspace agent's name,
            `ret_entities` indicate config data list,
            `batch_indices` indicate every config position in subspace.
        """
        ret = dict()
        for name, agent in self.agent_group.items():
            ret_entries, batch_indices = agent.RandomBatch(batch_size)
            ret[name] = (ret_entries, batch_indices)
        return ret

    def EverMeet(self, indices):
        """Have ever meet the indices in space.

        Parameters
        ----------
        indices: List
            A config position set in all subspace. 
        """
        return str(indices) in self.visits_set
    
    def Record(self, indices: Dict, value: float, use_sa = True, gamma = 0.05):
        """
        Record the config data with SA algorithm.

        Parameters
        ----------
        indices: Dict
            A config position set in all subspace. 
        value: float
            The config's performance.
        use_sa: bool optional(True)
            A simulated annealing algorithm is used to decide recording.
        gamma: float optional(0.05)
            The SA algorithm argument.
        """
        self.visits_set.add(str(indices))

        if use_sa:
            p = np.random.random()
            # The smaller gamma is, the more likely it is that the worse points 
            # will be selected.
            t = np.exp(-gamma * (value - self.Top1Value()) / self.Top1Value()) 
            if p <= t or math.isnan(t):
                # Using heap to rank all the test records.
                heapq.heappush(self.memory, MemEntry(indices, value))
                self.memory_size += 1
                return True
            else:
                return False
        else:
            heapq.heappush(self.memory, MemEntry(indices, value))
            self.memory_size += 1
            return True
    
    def AddSchedule(self, cmd: str, value: float):
        self.schedule_data.append([cmd, value])

    def ConvertIndices2FeatureVec(self, indices: Dict):
        ret = []
        for name, index in indices.items():
            ret.extend(self.agent_group[name].subspace.GetEntry(index))
        return ret
    
    def GetConfigFfromIndices(self, indices: Dict):
        config = {}
        for name, index in indices.items():
            config[name] = self.agent_group[name].subspace.GetEntry(index)
        return config
    
    def AddData(self, pre_state_indices: Dict, subspace_name:str, action: int, 
                next_states_indices: Dict, reward: float):
        """
        `pre_state_indices` indicate last state in every subspace.
        `value` indicate last state's performan for program design.
        `subspace.name` indicate which subspace will change.
        `action` indicate changing direction in the subspace.
        `next_state_indices` indicate next state in every subspace when do a action.
        """
        self.agent_group[subspace_name].AddData(
            self.ConvertIndices2FeatureVec(pre_state_indices), action,
            self.ConvertIndices2FeatureVec(next_states_indices), reward
        )
    
    def Train(self, save_model = True):
        for _, agent in self.agent_group.items():
            agent.Train(save_model)
    
    def TopRandom(self, gamma: int = 0.5):
        """Random return a better record. """
        entry = np.random.choice(self.memory)
        p = np.random.random()
        t = np.exp(-gamma * (entry.value - self.Top1Value()) / self.Top1Value())
        if p < t:
            return entry.indices, entry.value
        else:
            return self.Top1()
    
    def Top1(self):
        if self.memory_size > 0:
            return self.memory[0].indices, self.memory[0].value
        else:
            return {}, float('inf')

    def Top1Value(self):
        if self.memory_size > 0:
            return self.memory[0].value
        else:
            return float('inf')
    
    def TopK(self, k: int, modify = False):
        if k > self.memory_size:
            k = self.memory_size
        
        ret_indices = []
        ret_values = []
        tmp_lst = []
        for i in range(k):
            tmp = heapq.heappop(self.memory)
            tmp_lst.append(tmp)
            ret_indices.append(tmp.indices)
            ret_values.append(tmp.value)
        self.memory_size -= k
        if not modify:
            for tmp in tmp_lst:
                heapq.heappush(self.memory, tmp)
            self.memory_size += k
        return ret_indices, ret_values
    
    def PopTop(self):
        if self.memory_size > 0:
            self.memory_size -= 1
            return heapq.heappop(self.memory)
        else:
            return MemEntry({}, float("inf"))
    
    def LoadorCreateAgentModel(self):
        for _, agent in self.agent_group.items():
            agent.LoadOrCreateModel()

    def GetAgentModelsPath(self):
        model_path_lst = []
        for _, agent in self.agent_group.items():
            model_path_lst.append(agent.GetModelPath())
        return model_path_lst
    
    def GetAgentDatasPath(self):
        data_path_lst = []
        for _, agent in self.agent_group.items():
            data_path_lst.append(agent.GetDataPath())
        return data_path_lst
    
    def LoadAgentModel(self, save_dirpath: str = ""):
        for name, agent in self.agent_group.items():
            agent.model_path = os.path.join(
                save_dirpath, name+"_model.pth"
            )
            agent.LoadModel()
    
    def LoadAgentData(self, save_dirpath: str = ""):
        for name, agent in self.agent_group.items():
            agent.model_path = os.path.join(
                save_dirpath, name+"_data.txt"
            )
            agent.LoadData()
    
    def SaveAgentModel(self, save_dirpath: str = ""):
        for name, agent in self.agent_group.items():
            agent.model_path = os.path.join(
                save_dirpath, name+"_model.pth"
            )
            agent.SaveModel()

    def SaveAgentData(self, save_dirpath: str = ""):
        for name, agent in self.agent_group.items():
            agent.model_path = os.path.join(
                save_dirpath, name+"_data.txt"
            )
            agent.SaveData()
    
    def UpdateAgentTargetModel(self):
        for _, agent in self.agent_group.items():
            agent.UpdateTargetModel()
    
    def SaveMemoryData(self, filepath: str):
        """Save data for all the object of (space indices, performance)."""
        assert "pth" in filepath.split(".")
        torch.save(self.memory, filepath)
    
    def LoadMemoryData(self, filepath: str):
        """From filepath load pth file path.
        
        Returns
        -------
        data List[(indices, performance)]
        """
        assert "pth" in filepath.split(".")
        self.memory = torch.load(filepath)
        self.memory_size = len(self.memory)
        print(f"[INFO] Successfully read performance data from the {filepath}.")
    
    def SaveScheduleData(self, filepath: str):
        """ Save AutoSparse schedule config and value. """
        assert "pth" in filepath.split(".")
        torch.save(self.schedule_data, filepath)
    
    @staticmethod
    def LoadScheduleData(filepath: str):
        """ Load schedule config and value. """
        assert "pth" in filepath.split(".")
        data = torch.load(filepath)
        data_val = [element[1] for element in data]
        idx = torch.argmin(torch.tensor(data_val))
        return data, data[idx][0], data[idx][1]



