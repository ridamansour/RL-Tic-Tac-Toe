# 🧠 Reinforcement Learning Tic Tac Toe

A smart Tic Tac Toe game built using **Q-Learning** (Reinforcement Learning), with a GUI built in **Tkinter**. Play against a trained AI agent or train your own model from scratch.

---

## 🎯 Features

* 🧠 **AI opponent** trained using reinforcement learning
* 👥 **2-player mode** for human vs human
* 💾 **Persistent Q-Table** using `pickle`
* 🛠️ **Trainable** with millions of episodes
* 🖥️ **GUI** interface via Tkinter

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rl-tic-tac-toe.git
cd rl-tic-tac-toe
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
numpy
```

> **Note**: Tkinter is included with most Python installations. If you’re on Linux and it’s missing, install it via your package manager:
>
> ```bash
> sudo apt install python3-tk
> ```

---

## 🚀 Usage

### Play the Game

```bash
python tic_tac_toe.py
```

You’ll be able to:

* Play **1 vs Agent**
* Play **2 Player** (local)

### Train the AI

To train the agent from scratch:

1. Open `tic_tac_toe.py`
2. Set:

```python
training_mode = True
EPISODES_COUNT = 1_000_000  # or more
```

3. Run:

```bash
python tic_tac_toe.py
```

> The training will take a while depending on the number of episodes.
> The resulting Q-table is saved to: `RLModels/q_table10milLionyTooHumble.pkl`

---

## 🧠 How the AI Works

The agent uses **Q-Learning**, a model-free reinforcement learning algorithm. The Q-table maps game states to expected rewards for each possible move.

### Rewards:

| Situation      | Reward |
| -------------- | ------ |
| Win            | +10    |
| Loss           | -10    |
| Block opponent | +1     |
| Missed block   | -1     |
| Invalid move   | -10    |
| Draw           | 0      |

Exploration/exploitation is handled using the **epsilon-greedy** strategy with decay over time.

---

## ⚙️ Configurable Parameters

You can tune these in `tic_tac_toe.py`:

```python
alpha = 0.1         # Learning rate
gamma = 0.9         # Discount factor
epsilon = 1.0       # Initial exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01
EPISODES_COUNT = 1_000_000
```

---

## 📂 Project Structure

```
.
├── RLModels/
│   └── q_table10milLionyTooHumble.pkl   # Trained Q-Table (after training)
├── tic_tac_toe.py                       # Main game logic and GUI
├── requirements.txt                     # Dependencies
└── README.md
```

---

## 🧪 Known Limitations

* Agent always plays as **Player X**
* No dynamic difficulty yet
* Training is single-threaded (can be slow at high episode counts)

---

## 🙋‍♀️ FAQ

**Q: The AI makes random moves sometimes?**
A: If `epsilon` is still high or the model is undertrained, this is expected. Either train longer or decay epsilon faster.

**Q: Can I train against a human?**
A: Not currently. Training is self-play only.

---

## 📜 License

Feel free to fork, enhance, and share!
