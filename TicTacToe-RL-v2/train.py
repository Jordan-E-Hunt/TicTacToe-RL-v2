import random
import pickle
import os
from src import TicTacToe, QLearningAgent


def train(num_episodes, alpha=0.2, epsilon=1.0, discount=1.0):
    agent = QLearningAgent(alpha, epsilon, discount)
    opponent_pool = []
    snapshot_interval = 15000
    max_pool_size = 10

    for i in range(num_episodes):
        game = TicTacToe()
        agent.discount = discount

        # 50% chance of random training, 50% who goes first
        training_against_random = [random.random() < 0.5, random.random() < 0.5]

        # Pick opponent type: pool (40%), random (50%), self-play (10%)
        pool_opponent = None
        if opponent_pool and random.random() < 0.4:
            pool_opponent = random.choice(opponent_pool)
            training_against_random[0] = True

        # Memory dictionaries
        prev_state = {"X": None, "O": None}
        prev_action = {"X": None, "O": None}
        prev_reward = {"X": None, "O": None}

        while not game.game_over:
            cur_player = game.current
            opp_player = "O" if cur_player == "X" else "X"

            state = game.get_state()
            avail_moves = game.avail_moves()

            if training_against_random[0] and training_against_random[1]:
                if pool_opponent is not None:
                    save_eps = pool_opponent.epsilon
                    pool_opponent.epsilon = -1
                    action = pool_opponent.choose_action(state, avail_moves, game)
                    pool_opponent.epsilon = save_eps
                else:
                    action = random.choice(avail_moves)
                training_against_random[1] = False
            elif training_against_random[0]:
                action = agent.choose_action(state, avail_moves, game)
                training_against_random[1] = True
            else:
                action = agent.choose_action(state, avail_moves, game)

            rotation = game.last_rotation
            next_state, reward, done, _ = game.make_move(action)
            canon_action = game.rotate_action(action, rotation)
            next_avail = [game.rotate_action(m, game.last_rotation) for m in game.avail_moves()]

            if game.winner is not None:
                agent.update_Q_value(state, canon_action, reward, next_state, [])
                other_reward = -2.0 if game.winner != "Draw" else reward
                agent.update_Q_value(prev_state[opp_player], prev_action[opp_player],
                                     other_reward, tuple(-s for s in next_state), [])
            elif prev_state[opp_player] is not None:
                agent.update_Q_value(prev_state[opp_player], prev_action[opp_player],
                                     prev_reward[opp_player], next_state, next_avail)

            prev_state[cur_player] = state
            prev_action[cur_player] = canon_action
            prev_reward[cur_player] = reward

        agent.decay_epsilon()
        agent.decay_alpha(i, num_episodes, alpha)

        # Snapshot agent into pool periodically
        if (i + 1) % snapshot_interval == 0:
            snapshot = QLearningAgent()
            snapshot.Q = pickle.loads(pickle.dumps(agent.Q))
            opponent_pool.append(snapshot)
            if len(opponent_pool) > max_pool_size:
                opponent_pool.pop(0)
            print(f"Episode {i+1}: Snapshot saved. Pool size: {len(opponent_pool)}")

    return agent


def test(agent, num_games=1000):
    num_wins = 0
    num_draws = 0
    save_epsilon = agent.epsilon
    agent.epsilon = -1

    for _ in range(num_games):
        game = TicTacToe()
        agent_p = "X" if random.random() < 0.5 else "O"

        while not game.game_over:
            if game.current == agent_p:
                state = game.get_state()
                action = agent.choose_action(state, game.avail_moves(), game)
                game.make_move(action)
            else:
                game.make_move(random.choice(game.avail_moves()))

        if game.winner == agent_p:
            num_wins += 1
        elif game.winner == "Draw":
            num_draws += 1

    win_rate = (num_wins / num_games) * 100
    draw_rate = (num_draws / num_games) * 100
    loss_rate = 100 - win_rate - draw_rate
    print(f"Agent vs Random (random starting player):")
    print(f"Wins: {win_rate:.1f}% | Draws: {draw_rate:.1f}% | Losses: {loss_rate:.1f}%")

    agent.epsilon = save_epsilon
    return win_rate


if __name__ == "__main__":
    print("Training agent (300,000 episodes)...")
    agent = train(num_episodes=300000, alpha=0.2, epsilon=1.0, discount=1.0)

    print("\nTesting agent...")
    test(agent, num_games=1000)

    os.makedirs("models", exist_ok=True)
    with open("models/q_table.pkl", "wb") as f:
        pickle.dump(agent.Q, f)
    print(f"\nQ-table saved. Entries: {len(agent.Q)}")
