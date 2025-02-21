# 人狼ゲームAI / Werewolf Game AI

このリポジトリは、深層強化学習（DQN）を用いて人狼ゲームをプレイするAIエージェントの学習とプレイを行うPythonコードを含んでいます。
This repository contains Python code for training and playing AI agents for the Werewolf game using Deep Q-Network (DQN).

## 概要 / Overview

このプロジェクトでは、人狼ゲームのAIエージェントを強化学習によって訓練し、人間のような振る舞いを模倣させることを目指しています。エージェントは、ゲームの状態を観測し、最適な行動を選択するように学習します。
This project aims to train AI agents for the Werewolf game using reinforcement learning, mimicking human-like behavior. Agents learn to observe the game state and select optimal actions.

## 必要な環境 / Prerequisites

* Python 3.6+
* PyTorch
* NumPy

## インストール方法 / Installation

リポジトリをクローンしてください。
Clone the repository:

```bash
git clone https://github.com/TakuShoji/ai_werewolf_game
```

必要なライブラリをインストールしてください。
Install the required libraries:

```Bash
pip install torch numpy
```

実行方法 / Usage
学習を実行するには、train_agents()関数を実行します。
To train the agents, run the train_agents() function:

学習済みモデルでゲームをプレイするには、play_trained_game()関数を実行します。
To play a game with the trained model, run the play_trained_game() function:

ファイル構成 / File Structure
.
├── your_script_name.py # メインのPythonスクリプト / Main Python script
└── README.md

コードの説明 / Code Description
DQN: 深層Qネットワークのモデルを定義します。 / Defines the Deep Q-Network model.
Agent: 人狼ゲームのAIエージェントを定義します。 / Defines the AI agent for the Werewolf game.
WerewolfGame: 人狼ゲームの環境を定義します。 / Defines the environment for the Werewolf game.
train_agents: エージェントを学習させる関数です。 / Function to train the agents.
play_trained_game: 学習済みモデルでゲームをプレイする関数です。 / Function to play a game with the trained model.
今後の展望 / Future Work
エージェントの性能向上 / Improve agent performance.
より複雑な人狼ゲームのルールの実装 / Implement more complex Werewolf game rules.
他の強化学習アルゴリズムの適用 / Apply other reinforcement learning algorithms.
貢献 / Contributing
バグ報告や機能提案など、どのような貢献も歓迎します。
Contributions such as bug reports and feature suggestions are welcome.

ライセンス / License
このプロジェクトはMITライセンスの下で提供されています。
This project is licensed under the MIT License.

作者 / Author
[庄司　拓 / Taku Shoji]
