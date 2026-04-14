import pickle
import sys
from src import TicTacToe, QLearningAgent


def load_agent(path="models/q_table.pkl"):
    agent = QLearningAgent()
    agent.epsilon = -1  # Pure greedy, no exploration
    try:
        with open(path, "rb") as f:
            agent.Q = pickle.load(f)
        print(f"Agent loaded. ({len(agent.Q)} states)")
    except FileNotFoundError:
        print(f"Error: {path} not found. Run train.py first.")
        sys.exit(1)
    return agent


def play():
    agent = load_agent()
    game = TicTacToe()

    print("\n=== Tic-Tac-Toe: You vs Q-Learning Agent ===\n")
    choice = input("Play as X (go first) or O (go second)? [X/O]: ").strip().upper()
    if choice not in ("X", "O"):
        choice = "X"

    human = choice
    ai = "O" if human == "X" else "X"
    print(f"\nYou are {human}. Agent is {ai}.\n")

    while not game.game_over:
        game.print_board()
        print()

        if game.current == human:
            while True:
                try:
                    move = input("Your move (row col, 0-indexed): ").strip().split()
                    r, c = int(move[0]), int(move[1])
                    if (r, c) in game.avail_moves():
                        break
                    print("That cell is taken. Try again.")
                except (ValueError, IndexError):
                    print("Enter two numbers like: 1 2")
            game.make_move((r, c))
        else:
            state = game.get_state()
            action = agent.choose_action(state, game.avail_moves(), game)
            game.make_move(action)
            print(f"Agent plays: {action[0]} {action[1]}")

    game.print_board()
    print()
    if game.winner == "Draw":
        print("Draw!")
    elif game.winner == human:
        print("You win!")
    else:
        print("Agent wins.")

    print()
    again = input("Play again? [y/n]: ").strip().lower()
    if again == "y":
        play()


if __name__ == "__main__":
    play()
