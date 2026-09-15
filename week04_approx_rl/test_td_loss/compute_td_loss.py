import torch
import torch.nn as nn
from typing import Protocol
import numpy as np


class ComputeTdLossProtocol(Protocol):
    """
    An Protocol which the compute_td_loss function should match.
    """

    def __call__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        is_done: torch.Tensor,
        agent: nn.Module,
        target_network: nn.Module,
        gamma: float,
    ):
        pass


class MockAgent(nn.Module):
    """
    An nn.Module, which outputs a value which does not depend on its input.
    Designed to be used for testing the compute_td_loss function.
    """

    def __init__(self, output_q_values: torch.Tensor):
        super().__init__()
        assert output_q_values.dtype == torch.float, output_q_values.dtype
        assert output_q_values.ndim == 2, output_q_values.shape
        self.output_q_values = nn.Parameter(output_q_values)

    def forward(self, state):
        return torch.clone(self.output_q_values)


@torch.no_grad()
def test_is_done_is_used(compute_td_loss: ComputeTdLossProtocol):
    """
    Tries to catch the error when compute_td_loss ignores its is_done argument.
    """

    states = torch.empty(1)
    actions = torch.tensor([0])
    rewards = torch.tensor([1], dtype=torch.float)
    is_done_first = torch.tensor([True])
    is_done_second = torch.tensor([False])
    next_states = torch.empty(1)
    gamma = 0.99

    q_values_agent = torch.tensor([[1, 1, 1]], dtype=torch.float)
    q_values_target_network = torch.tensor([[1, 1, 1]], dtype=torch.float)
    agent = MockAgent(q_values_agent)
    target_network = MockAgent(q_values_target_network)

    loss_kwargs = dict(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        agent=agent,
        target_network=target_network,
        gamma=gamma,
    )

    loss_first = compute_td_loss(is_done=is_done_first, **loss_kwargs).item()
    loss_second = compute_td_loss(is_done=is_done_second, **loss_kwargs).item()

    abs_diff = abs(loss_first - loss_second)
    if abs_diff > 0.5:
        msg = "compute_td_loss returned close values for different is_done inputs"

    assert abs(loss_first - loss_second) > 0.5, msg


@torch.no_grad()
def test_compute_td_loss_vanilla(compute_td_loss: ComputeTdLossProtocol):
    """
    Checks compute_td_loss on manually precomputed examples.
    Note: this is a test for vanilla compute_td_loss
    and it should NOT be used for double_dqn
    """

    samples = [
        {
            "q_agent": [0, 1, 2],
            "action": 1,
            "is_done": False,
            "q_target": [0, 1, 2],
            "gamma": 0.5,
            "reward": 5,
            "answer": 25,
        },
        {
            "q_agent": [0, 1, 2],
            "action": 1,
            "is_done": False,
            "q_target": [2, 0, 1],
            "gamma": 0.5,
            "reward": 5,
            "answer": 25,
        },
        {
            "q_agent": [3, 1, 2],
            "action": 1,
            "is_done": True,
            "q_target": [0, 1, 2],
            "gamma": 0.5,
            "reward": 5,
            "answer": 16,
        },
        {
            "q_agent": [0, 1, 2],
            "action": 0,
            "is_done": False,
            "q_target": [0, 1, 2],
            "gamma": 0.5,
            "reward": 5,
            "answer": 36,
        },
    ]

    for sample in samples:
        agent = MockAgent(torch.tensor(sample["q_agent"], dtype=torch.float)[None])
        tn = MockAgent(torch.tensor(sample["q_target"], dtype=torch.float)[None])
        ans = compute_td_loss(
            states=torch.empty(1),
            actions=torch.tensor(sample["action"])[None],
            rewards=torch.tensor(sample["reward"])[None],
            next_states=torch.empty(1),
            is_done=torch.tensor(sample["is_done"])[None],
            agent=agent,
            target_network=tn,
            gamma=sample["gamma"],
        ).item()
        abs_diff = abs(ans - sample["answer"])
        assert abs_diff < 1e-8, abs_diff


@torch.no_grad()
def test_compute_td_loss_double(compute_td_loss: ComputeTdLossProtocol):
    """
    Checks compute_td_loss on manually precomputed examples.
    Note: this is a test for vanilla compute_td_loss
    and it should NOT be used for double_dqn
    """

    samples = [
        {
            "q_agent": [0, 1, 2],
            "action": 1,
            "is_done": False,
            "q_target": [0, 1, 2],
            "gamma": 0.5,
            "reward": 5,
            "answer": 25,
        },
        {
            "q_agent": [0, 1, 2],
            "action": 1,
            "is_done": False,
            "q_target": [2, 0, 1],
            "gamma": 0.5,
            "reward": 5,
            "answer": 20.25,
        },
        {
            "q_agent": [3, 1, 2],
            "action": 1,
            "is_done": False,
            "q_target": [-1, 1, 2],
            "gamma": 0.5,
            "reward": 5,
            "answer": 12.25,
        },
        {
            "q_agent": [3, 1, 2],
            "action": 1,
            "is_done": True,
            "q_target": [-1, 1, 2],
            "gamma": 0.5,
            "reward": 5,
            "answer": 16,
        },
        {
            "q_agent": [0, 1, 2],
            "action": 0,
            "is_done": False,
            "q_target": [0, 1, 2],
            "gamma": 0.5,
            "reward": 5,
            "answer": 36,
        },
    ]

    for sample in samples:
        agent = MockAgent(torch.tensor(sample["q_agent"], dtype=torch.float)[None])
        tn = MockAgent(torch.tensor(sample["q_target"], dtype=torch.float)[None])
        ans = compute_td_loss(
            states=torch.empty(1),
            actions=torch.tensor(sample["action"])[None],
            rewards=torch.tensor(sample["reward"])[None],
            next_states=torch.empty(1),
            is_done=torch.tensor(sample["is_done"])[None],
            agent=agent,
            target_network=tn,
            gamma=sample["gamma"],
        ).item()
        abs_diff = abs(ans - sample["answer"])
        assert abs_diff < 1e-8, abs_diff

def compute_td_loss(states, actions, rewards, next_states, is_done,
                    agent, target_network,
                    gamma=0.99,
                    check_shapes=False,
                    device="cpu"):
    """ Compute td loss using torch operations only. Use the formulae above. """
    states = torch.tensor(states, device=device, dtype=torch.float32)    # shape: [batch_size, *state_shape]
    actions = torch.tensor(actions, device=device, dtype=torch.int64)    # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done,
        device=device,
        dtype=torch.float32,
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)  # shape: [batch_size, n_actions]

    # compute q-values for all actions in next states
    # with torch.no_grad():
    predicted_next_qvalues = target_network(next_states)  # shape: [batch_size, n_actions]
    # print(predicted_next_qvalues)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]  # shape: [batch_size]

    # compute V*(next_states) using predicted next q-values
    
    next_state_values = predicted_next_qvalues.max(dim=-1).values

    assert next_state_values.dim() == 1 and next_state_values.shape[0] == states.shape[0], \
        "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    target_qvalues_for_actions = rewards + gamma * is_not_done * next_state_values 

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim() == 2, \
            "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim() == 1, \
            "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim() == 1, \
            "there's something wrong with target q-values, they must be a vector"

    return loss

if __name__ == "__main__":
    print(test_compute_td_loss_vanilla(compute_td_loss))
