import pickle  # For saving and loading the Q-Table
import random
import tkinter as tk  # For the GUI
from tkinter import messagebox  # For pop-up messages

import numpy as np
from numpy import signedinteger

# Global Variables
q_table = {}
alpha: float = 0.1  # the learning rate
gamma: float = 0.9
epsilon: float = 1.0  # exploration factor 1 means it is mostly random. The lower, the more cognitive
epsilon_decay: float = 0.995  # Decay rate per episode
min_epsilon: float = 0.01  # Used to be 0.1
training_mode: bool = False  # Switch to False for human vs agent
q_table_file: str = "RLModels/q_table10milLionyTooHumble.pkl"  # File to save/load the Q-Table
EPISODES_COUNT: int = 1_000_000  # times for agent reincarnation :)

WIN_STATES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Horizontal Lines
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Vertical Lines
    [0, 4, 8], [2, 4, 6]  # Diagonal Lines
]  # Not to be tempered with


# The functions for the Reinforcement Learning Logic
def check_winner(board: list[int]) -> int:
    """This function returns the winner of the game (1 for 'X', 2 for 'O', -1 for no one) or draw (0)."""
    for line in WIN_STATES:
        if board[line[0]] == board[line[1]] == board[line[2]] and board[line[0]] != 0:
            return board[line[0]]  # the winner
    if 0 not in board:
        return 0  # draw
    return -1  # no one has won yet


def check_can_win_next_move(board: list[int]) -> list[int] | None:
    """This function returns a list of possible winners for the next move."""
    possible_winners2: dict[int, int] = {
        1: 0,
        2: 0
    }
    possible_winners: [int] = []
    for line in WIN_STATES:
        if (board[line[0]] == board[line[2]] and board[line[0]] != 0 and board[line[1]] == 0) \
                or (board[line[1]] == board[line[2]] and board[line[1]] != 0 and board[line[0]] == 0) \
                or (board[line[0]] == board[line[1]] and board[line[1]] != 0 and board[line[2]] == 0):
            temp = max(board[line[0]], board[line[1]])
            possible_winners.append(temp)  # this is between zero, and the players number (1 or 2)
            if temp == 1:
                possible_winners2[1] += max(board[line[0]], board[line[1]])
            else:
                possible_winners2[2] += max(board[line[0]], board[line[1]])
    print(possible_winners, possible_winners2)
    return possible_winners if len(possible_winners) != 0 else None


def choose_action(state: list[int]) -> int | signedinteger:
    """The function balances exploration and exploitation: \n
        During early training (high epsilon), it explores to learn about the game.\n
        During later stages (low epsilon), it exploits its learned knowledge to make optimal moves.\n
        This balance helps the agent learn a good strategy over time while avoiding getting stuck in suboptimal behaviors early on."""
    if random.uniform(0, 1) < epsilon:  # Explore
        return random.choice([i for i in range(9) if state[i] == 0])
    else:  # Exploit
        return np.argmax(q_table.get(tuple(state), [0] * 9))


def update_q_table(state: list[int], action: int | signedinteger, reward: float, next_state: list[int]):
    """This function ensures the agent learns which actions lead to better outcomes over time
        by updating the Q-Table to reflect the rewards and consequences of actions."""
    q_state = q_table.get(tuple(state), [0] * 9)
    q_next = q_table.get(tuple(next_state), [0] * 9)
    q_state[action] = q_state[action] + alpha * (reward + gamma * max(q_next) - q_state[action])
    q_table[tuple(state)] = q_state


def train_agent(episodes: int) -> None:
    global epsilon
    for episode in range(episodes):
        board: list[int] = [0] * 9
        done_training: bool = False
        current_player: int = 1
        if episode % 100000 == 0:
            print(f'{episode=:,}')
            print(f'{(episode / episodes):.2%} done')
        while not done_training:
            state: list[int] = board.copy()
            action: int | signedinteger = choose_action(state)  # The chosen index
            reward: float = 0
            if board[action] == 0:
                post_board: list[int] = board.copy()
                board[action] = current_player
                winner = check_winner(board)
                if winner != -1:
                    reward += 10 if winner == current_player else -10 if winner != 0 else 0
                    # Draw reward is zero for now... meaning balanced (tho experimenting is applicable)
                    update_q_table(state, action, reward, board)
                    done_training = True
                else:
                    # TODO fix the rewards for when the are two possible winners at the same time
                    possible_winner_before_action = check_can_win_next_move(post_board)[-1]
                    possible_winner_after_action = check_can_win_next_move(board.copy())[-1]
                    other_player = (3 - current_player)
                    if possible_winner_before_action == other_player and possible_winner_after_action == other_player:
                        # the agent hasn't blocked the other player form winning
                        reward -= 1
                    elif possible_winner_before_action == other_player and possible_winner_after_action != other_player:
                        # the agent has blocked the other player form winning
                        reward += 1
                    elif possible_winner_before_action != current_player and possible_winner_after_action == current_player:
                        # Immediate reward (a possible win for the agent)
                        reward += 0.1
                    next_state = board.copy()
                    update_q_table(state, action, reward, next_state)
                    current_player = 3 - current_player  # Switch player (1 or 2)
            else:
                update_q_table(state, action, -10, state)  # The penalty for an invalid move
                done_training = True

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)


def save_q_table(filename: str) -> None:
    """This function saves the Q-Table to a file for later use in the RLModels Folder"""
    with open(filename, "wb") as file:
        pickle.dump(q_table, file)
    print(f"Q-Table saved to {filename}")


def load_q_table(filename: str) -> None:
    """This function loads the Q-Table from the already made RL Models in the RLModels Folder"""
    global q_table
    try:
        with open(filename, "rb") as file:
            q_table = pickle.load(file)
        print(f"Q-Table loaded from {filename}")
    except FileNotFoundError:
        print(f"No Q-Table found at {filename}. Starting fresh.")


# GUI Class and game logic
# Modified Tic Tac Toe Program with Game Mode Selection
class TicTacToeGUI:
    """This class creates the GUI for the Tic Tac Toe game."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Tic Tac Toe with RL")
        self.root.geometry("650x550")
        self.root.config(bg="#333")

        self.mode: str = None  # None initially, set later to "1 vs Agent" or "2 Player"
        self.current_player: int = 1  # Default starting player (X)
        self.board: [int] = [0] * 9
        self.buttons: [tk.Button] = []

        # Show mode selection screen
        self.show_mode_selection()

        if training_mode:
            self.train_agents()

    def show_mode_selection(self):
        """Displays the mode selection screen."""
        for widget in self.root.winfo_children():
            widget.destroy()  # Clear the screen

        title_label = tk.Label(
            self.root,
            text="Choose Game Mode" if training_mode == False else "Wait for the model to train plz...",
            font=("Arial", 28, "bold"),
            bg="#333",
            fg="white",
        )
        title_label.pack(pady=20)

        one_vs_agent_button = tk.Button(
            self.root,
            text="1 vs Agent",
            font=("Arial", 20),
            bg="#444",
            fg="black",
            activebackground="#555",
            activeforeground="lightgray",
            command=self.start_one_vs_agent,
        )
        one_vs_agent_button.pack(pady=10)

        two_player_button = tk.Button(
            self.root,
            text="2 Player",
            font=("Arial", 20),
            bg="#444",
            fg="black",
            activebackground="#555",
            activeforeground="lightgray",
            command=self.start_two_player,
        )
        two_player_button.pack(pady=10)
        root.update()

    def train_agents(self):
        """This function trains the agents and saves the Q-Table to a file. (Only used when training_mode is True)"""
        train_agent(episodes=EPISODES_COUNT)
        save_q_table(q_table_file)
        global training_mode
        training_mode = False

    def start_one_vs_agent(self):
        """Starts the 1 vs Agent mode."""
        self.mode = "1_vs_agent"
        self.initialize_game()

    def start_two_player(self):
        """Starts the 2 Player mode."""
        self.mode = "2_player"
        self.initialize_game()

    def initialize_game(self):
        """Sets up the game board and starts the selected mode."""
        for widget in self.root.winfo_children():
            widget.destroy()  # Clears the screen

        self.board = [0] * 9  # Reinitialized to be empty if the game restarted
        self.buttons = []
        self.current_player = 1

        title_label = tk.Label(
            self.root,
            text="Tic Tac Toe",
            font=("Arial", 28, "bold"),
            bg="#333",
            fg="white",
        )
        title_label.pack(pady=20)

        self.frame = tk.Frame(self.root, bg="#333")
        self.frame.pack()

        self.create_board()

        if self.mode == "1_vs_agent" and self.current_player == 1:
            self.agent_turn()

    def create_board(self):
        """Creates the game board with buttons."""
        button_width: int = 8
        button_height: int = 4
        font_style: tuple[str, int, str] = ("Arial", 24, "bold")
        for i in range(9):
            button = tk.Button(
                self.frame,
                text="",
                font=font_style,
                width=button_width,
                height=button_height,
                bg="#444",
                fg="black",
                activebackground="#555",
                activeforeground="lightgray",
                disabledforeground="black",
                command=lambda index=i: self.on_click(index),
            )
            button.grid(row=i // 3, column=i % 3)  # Use grid within the frame for buttons (placeholders)
            self.buttons.append(button)

    def on_click(self, index: int):
        """This function handles the click event on a button (changes the state of the button and updates the game board and checks for winner)."""
        print(f'{self.board=}')
        print(f'{check_can_win_next_move(board=self.board)=}')
        if self.board[index] == 0:  # self.current_player == 2:  Human's turn (condition not actually necessary) x
            if self.mode == "2_player":
                self.board[index] = self.current_player
                self.buttons[index].config(
                    text="X" if self.current_player == 1 else "O", state="disabled"
                )
                root.update()
                winner = check_winner(self.board)
                if winner != -1:
                    self.end_game(winner)
                else:
                    self.current_player = 3 - self.current_player
            elif self.mode == "1_vs_agent":
                self.board[index] = 2
                self.buttons[index].config(text="O", state="disabled")
                root.update()  # To display the last move before the popup appears
                winner = check_winner(self.board)
                if winner != -1:
                    self.end_game(winner)
                else:
                    self.agent_turn()

    def agent_turn(self):
        """Executes the agent's turn in 1 vs Agent mode."""
        state = self.board.copy()
        action = choose_action(state)
        self.board[action] = 1
        self.buttons[action].config(text="X", state="disabled")  # The button
        root.update()
        winner = check_winner(self.board)
        if winner != -1:
            self.end_game(winner)

    def end_game(self, winner):
        """This function ends the game and displays a pop-up message indicating the winner or draw. And disables all buttons to prevent further moves."""
        if winner == 1:
            messagebox.showinfo("Game Over",
                                "Player X wins!" if self.mode == "2_player" else "Reinforcement Learning wins!")
        elif winner == 2:
            messagebox.showinfo("Game Over", "Player O wins!")
        else:
            messagebox.showinfo("Game Over", "It's a draw!")
        for button in self.buttons:
            button.config(state="disabled")  # To stop the buttons from being clicked after the game ends
        self.show_mode_selection()


# Main Program
if __name__ == "__main__":
    load_q_table(q_table_file)
    root = tk.Tk()
    game = TicTacToeGUI(root)
    root.mainloop()
