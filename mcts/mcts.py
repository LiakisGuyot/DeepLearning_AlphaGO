from connect4.game import *
from Node import Node

NUM_SIMULATIONS = 100

test_board = np.array([
    [0, -1, -1, -1, 1, 0, -1],
    [0, 1, -1, 1, 1, 0, 1],
    [-1, 1, -1, 1, 1, 0, -1],
    [1, -1, 1, -1, -1, 0, -1],
    [-1, -1, 1, -1, 1, 1, -1],
    [-1, 1, 1, -1, 1, -1, 1]
])


def dummy_model_predict(board: np.ndarray):
    value_head = 0.5
    policy_head = [0.5, 0, 0, 0, 0, 0.5, 0]
    return value_head, policy_head


# Root init

root = Node(0, 1, test_board)

# MCTS simulate

for _ in range(NUM_SIMULATIONS):
    node = root
    path = [node]

    while len(node.children) > 0:
        action, node = node.select_child()
        path.append(node)

    value = None
    if is_draw(node.state):
        value = 0
    if is_win(node.state, 1):
        value = 1
    if is_win(node.state, -1):
        value = -1

    if value is None:
        value, action_p = dummy_model_predict(node.state)
        node.expand(action_p)

    for n in path:
        n.value += value
        n.visits += 1

render(root.children[0].state)
print(root.children[0].value)

render(root.children[5].state)
print(root.children[5].value)
