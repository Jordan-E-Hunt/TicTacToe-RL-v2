import random


class QLearningAgent:
    def __init__(self, alpha=0.2, epsilon=1.0,
                 discount=0.95, epsilon_min=0.01, epsilon_decay=0.99996):
        self.Q = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_Q_value(self, state, action):
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state, avail_moves, game):
        canonical_moves = [game.rotate_action(m, game.last_rotation) for m in avail_moves]
        if random.uniform(0, 1) < self.epsilon:
            idx = random.randint(0, len(avail_moves) - 1)
        else:
            Q_values = [self.get_Q_value(state, cm) for cm in canonical_moves]
            max_Q = max(Q_values)
            best = [i for i, q in enumerate(Q_values) if q == max_Q]
            idx = random.choice(best)
        return avail_moves[idx]

    def update_Q_value(self, state, action, reward, next_state, next_avail_moves):
        if not next_avail_moves:
            max_next_Q = 0.0
        else:
            next_Q_values = [self.get_Q_value(next_state, a) for a in next_avail_moves]
            max_next_Q = max(next_Q_values)
        current_Q = self.get_Q_value(state, action)
        td_error = reward + self.discount * max_next_Q - current_Q
        self.Q[(state, action)] = current_Q + self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def decay_alpha(self, episode, total_episodes, start_alpha):
        min_alpha = 0.01
        self.alpha = max(min_alpha, start_alpha - (start_alpha - min_alpha) * (episode / total_episodes))
