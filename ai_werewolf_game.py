import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time

# モデル本体
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# プレイヤー（エージェント）本体
class Agent:
    def __init__(self, agent_id, true_role, input_size, output_size):
        self.agent_id = agent_id
        self.true_role = true_role
        self.recognized_role = "市民"
        self.memory = deque(maxlen=3000)  # 経験再生バッファのサイズを大きくする
        self.model = DQN(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # epilonが大きいほど挑戦的な性格になる。5はかなり大きい。通常は1程度。学習時は逓減するようになっている。実プレイ時は強制的に0にされる
        self.epsilon = 5
        # gammaの振れ幅を決定。gammaは小さいほど短絡的な性格になる。均一なプレイヤーにしたければself.gammaを固定値に変更すればよい
        min_g, max_g = 0.85, 0.99
        self.gamma = (max_g - min_g) * random.random() + min_g
        self.talking_skill = 0
        self.deception_skill = 0
        self.revealed_roles = set()
        self.has_revealed = False
        self.alive = True  # 各エージェントの生存状態
        self.previous_reveals = set() #以前にカミングアウトした役職を保持する
        self.protect_target  = None
        self.divine_targets = []
        self.werewolf_revealed = [0] * 9
        if self.true_role == "人狼":
            self.werewolf_revealed[self.agent_id] = 1
    
    def choose_action(self, state, candidates):
        if not self.alive:
            return None  # **死亡している場合は行動しない**
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state_tensor)
        best_index = torch.argmax(q_values).item()
        return candidates[best_index % len(candidates)]

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            q_values = self.model(torch.FloatTensor(state))
            next_q_values = self.model(torch.FloatTensor(next_state))
            target = reward + self.gamma * torch.max(next_q_values).item() * (1 - done)
            target_q_values = q_values.clone()
            target_q_values[action] = target
            loss = nn.MSELoss()(q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.995)

    def _decide_reveal(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.model(state_tensor)
        return torch.argmax(action).item() == 1
    
    def decide_reveal(self, state):
        """
        カミングアウトするかどうかの判断を行う。
        三役がカミングアウトしないと盛り上がりに欠けるのでこれらは必ずカミングアウトを実行するようにしている。
        役職にかかわらず随時判断させたい場合はこちらのメソッドの名前に"_"を付けて、先の_decide_revealの"_"を消せばよい。
        """
        if self.true_role == "霊能者" or self.true_role == "占い師" or self.true_role == "狂人":
            return True
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = self.model(state_tensor)
            return torch.argmax(action).item() == 1
    
    def reveal_role(self, new_role=None):
        if new_role is None:
            new_role = self.true_role
        if new_role in self.revealed_roles:
            return False
        if self.has_revealed and new_role != self.recognized_role:
            self.revealed_roles.add(new_role)
            return True
        self.recognized_role = new_role
        self.has_revealed = True
        self.revealed_roles.add(new_role)
        return True

    def divine(self, target, game):
        if self.true_role == "占い師":
            result = "黒" if target.true_role == "人狼" else "白"
            game.liar[target.agent_id][self.agent_id] = -1
            if result == "黒":
                self.werewolf_revealed[target.agent_id] = 1
            else:
                self.werewolf_revealed[target.agent_id] = -1
            return result
        else:
            seer_num = len([a for a in game.agents if a.recognized_role == "占い師"])
            with torch.no_grad():
                state_tensor = torch.FloatTensor(self.get_state(game))
                q_values = self.model(state_tensor)
                result = "黒" if torch.argmax(q_values).item() % 2 == 0 else "白"
                # ニセ占い師が市民・霊能者・狩人に黒判定を出した場合
                if self.has_revealed and result == "黒" and target.true_role in ["市民", "霊能者", "狩人"]:
                    game.update_trust(target.agent_id, self.agent_id, -1.)
                    game.liar[target.agent_id][self.agent_id] = 1
                # ニセ占い師が人狼に白判定を出した場合
                elif self.has_revealed and result == "白" and target.true_role == "人狼":
                    game.update_trust(target.agent_id, self.agent_id, 0.2)
                    game.liar[target.agent_id][self.agent_id] = -1
                elif self.has_revealed and result == "白" and target.true_role in ["市民", "霊能者", "狩人"]:
                    game.liar[target.agent_id][self.agent_id] = -1 / seer_num
                return result

    def defend(self, accuser_id):
        # 疑いをかけられたときの抗弁
        skill = self.talking_skill if self.true_role == self.recognized_role else self.deception_skill
        success_chance = 0.5 + 0.05 * skill
        success = random.random() < success_chance
        if success:
            if random.random() < 0.05:
                if self.true_role == self.recognized_role:
                    self.talking_skill = min(10, self.talking_skill + 1)
                else:
                    self.deception_skill = min(10, self.deception_skill + 1)
        return success

    def take_action(self, state, action_type, candidates):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state_tensor)
        best_index = torch.argmax(q_values).item()
        return candidates[best_index % len(candidates)]

    def judge(self, executed_player, game):
        if executed_player and self.has_revealed:
            if self.true_role == "霊能者":
                result = "黒" if executed_player.true_role == "人狼" else "白"
                if result == "黒":
                    self.werewolf_revealed[executed_player.agent_id] = 1
                else:
                    self.werewolf_revealed[executed_player.agent_id] = -1
                return result
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(self.get_state(game))
                    q_values = self.model(state_tensor)
                    return "黒" if torch.argmax(q_values).item() % 2 == 0 else "白"
        else:
            pass
    
    def get_state(self, game):
        # **生存状態**
        alive_status = [int(a.alive) for a in game.agents]
        # 信頼
        trust_levels = game.trust[self.agent_id]
        # **カミングアウト済みか**
        has_revealed = [int(a.has_revealed) for a in game.agents]
        # **投票データ**
        votes = game.votes
        # **占い結果**
        divination_results = game.divination_results
        # **認識されている役職**
        recognized_roles = [game.role_to_index[a.recognized_role] for a in game.agents]
        # 人狼同士が認識しあうための処理
        werewolf_revealed = self.werewolf_revealed
        game.werewolfship()
        if self.true_role == "人狼":
            wolf_id = [a.agent_id for a in game.agents if a.true_role == "人狼"]
            for w in wolf_id:
                divination_results[w] = -1
        
        # 明らかにうそをついたプレイヤー
        liars = game.liar[self.agent_id]

        # **状態ベクトルを結合**
        state = alive_status + trust_levels + has_revealed + votes + divination_results + recognized_roles + liars + werewolf_revealed
        
        # **72 次元に調整**
        while len(state) < 72:
            state.append(0.0)
        return state

# ゲームフィールド本体
class WerewolfGame:
    def __init__(self, agents):
        self.agents = agents
        self.day = 0
        self.trust = [[0.5] * 9 for _ in range(9)]
        self.divination_results = [0.0] * 9
        self.last_executed = None
        self.votes = [-1] * 9
        self.liar = [[0] * 9 for _ in range(9)]
        self.role_to_index = {"市民": 0, "占い師": 1, "霊能者": 2, "狩人": 3, "人狼": 4, "狂人": 5}
        self.index_to_role = {0: "市民", 1: "占い師", 2: "霊能者", 3: "狩人", 4: "人狼", 5: "狂人"}

    def reset_agents(self):
        for agent in self.agents:
            agent.recognized_role = "市民"
            agent.has_revealed = False
            agent.reveal_role = set()
            agent.alive = True
            agent.divine_targets = []
            agent.previous_reveals.clear()
            agent.werewolf_revealed = [0] * 9
            if agent.true_role == "人狼":
                agent.werewolf_revealed[agent.agent_id] = 1
            
    
    def update_trust(self, from_id=None, to_id=None, change=0):
        if all(map(lambda v: isinstance(v, type(None)), [from_id, to_id])):
            pass
        elif from_id is None:
            for i in range(9):
                self.trust[i][to_id] += change
        elif to_id is None:
            self.trust[from_id] = list(map(lambda v: v + change, self.trust[from_id]))
        else:
            self.trust[from_id][to_id] = max(0.0, min(1.0, self.trust[from_id][to_id] + change))

    def werewolfship(self):
        wolf1, wolf2 = [a for a in self.agents if a.true_role == "人狼"]
        self.update_trust(wolf1.agent_id, wolf2.agent_id, 1)
        self.update_trust(wolf2.agent_id, wolf1.agent_id, 1)
        wolf1.werewolf_revealed[wolf2.agent_id] = 1
        wolf2.werewolf_revealed[wolf1.agent_id] = 1
    
    def perform_divinations(self):
        for agent in [a for a in self.agents if a.alive]:
            if agent.recognized_role == "占い師" and agent.has_revealed:
                not_divined_agents = [a for a in self.agents if a.alive and a not in agent.divine_targets]
                candidate = [a for a in not_divined_agents if a != agent]
                if len(candidate) == 0:
                    print(f"プレイヤー{agent.agent_id}({agent.true_role})にはもう占う対象がいませんでした")
                    pass
                else:
                    target = agent.take_action(agent.get_state(self), 'divine', candidate)
                    result = agent.divine(target, self)
                    agent.divine_targets.append(target)
                    seer_num = len([a for a in self.agents if a.recognized_role == "占い師"])
                    change = 1/seer_num if result == "白" else -1/seer_num
                    self.divination_results[target.agent_id] += change
                    self.update_trust(to_id=target.agent_id, change=change)
                    if result == "黒" and target.true_role == "人狼" and agent.true_role == "占い師":
                        self.rewards[agent.agent_id] += 0.5
                        agent.werewolf_revealed[target.agent_id] = 1
                    print(f"プレイヤー{agent.agent_id}({agent.true_role})がプレイヤー{target.agent_id}を占い、{result}と判定")
            elif agent.true_role == "占い師":
                # 真の占い師はカミングアウトしていなくても占いは密かにに実行する
                not_divined_agents = [a for a in self.agents if a.alive and a not in agent.divine_targets]
                candidate = [a for a in not_divined_agents if a != agent]
                if len(candidate) == 0:
                    pass
                else:
                    target = agent.take_action(agent.get_state(self), 'divine', candidate)
                    result = agent.divine(target, self)
                    agent.divine_targets.append(target)
                    print(f"真の占い師プレイヤー{agent.agent_id}が密かにプレイヤー{target.agent_id}を占い、{result}と判定")
                    if result == "黒":
                        agent.werewolf_revealed[target.agent_id] = 1
                        self.update_trust(agent.agent_id, target.agent_id, 0)
                    else:
                        agent.werewolf_revealed[target.agent_id] = -1
                        self.update_trust(agent.agent_id, target.agent_id, 1)
                
    
    def play_game(self):
        self.rewards = {agent.agent_id: 0 for agent in self.agents}
        self.werewolfship()
        self.day = 0
        black_count = {}
        for agent in self.agents:
            agent.alive = True
        
        while True:
            self.day += 1
            print("=" * 30)
            print(f"{self.day}日目 - 昼")
            print("=" * 30)

            if self.last_executed:
                medium_num = len([a for a in self.agents if a.recognized_role == "霊能者"])
                for agent in [a for a in self.agents if a.alive]:
                    if agent.recognized_role == "霊能者":
                        if agent.agent_id not in black_count.keys():
                            black_count[agent.agent_id] = 0
                        judge_result = agent.judge(self.last_executed, self)
                        change = -0.5 / medium_num
                        if judge_result == "黒":
                            black_count[agent.agent_id] += 1
                            change *= -1
                            if agent.true_role == "霊能者":
                                self.rewards[agent.agent_id] += 1.
                                agent.werewolf_revealed[self.last_executed.agent_id] = 1
                            
                        if black_count[agent.agent_id] > 1:
                            self.update_trust(to_id=agent.agent_id, change=-1)
                            for i in range(9):
                                self.liar[i][agent.agent_id] = 1
                        
                        for i in range(9):
                            if i == self.last_executed.agent_id:
                                self.divination_results[i] -= change
                            else:
                                self.divination_results[i] += change
                        print(f"プレイヤー{agent.agent_id}({agent.true_role})が処刑されたプレイヤー{self.last_executed.agent_id}を判定: {judge_result}")
                    elif agent.true_role == "霊能者":
                        judge_result = agent.judge(self.last_executed, self)
                        print(f"真の霊能者プレイヤー{agent.agent_id}が密かに処刑されたプレイヤー{self.last_executed.agent_id}を判定: {judge_result}")
            
            self.handle_reveals()
            self.perform_divinations()
            self.handle_phase('handle_accusations_and_defenses')
            self.votes = [-1] * 9
            self.handle_phase('handle_voting')
            if self.check_victory():
                break
            print("="*30)
            print(f"{self.day}日目 - 夜")
            print("="*30)
            self.handle_phase('handle_protection')
            self.handle_phase('handle_attack')

            for agent in self.agents:
                if agent.true_role == "狩人":
                    agent.protect_target = None
            if self.check_victory():
                break
        
        print("【ゲーム終了】")
    
    
    def handle_reveals(self):
        real_seer = next((agent for agent in self.agents if agent.true_role == "占い師"), None)
        real_medium = next((agent for agent in self.agents if agent.true_role == "霊能者"), None)
        real_hunter = next((agent for agent in self.agents if agent.true_role == "狩人"), None)

        for agent in [a for a in self.agents if a.alive]:
            if agent.decide_reveal(agent.get_state(self)):
                if agent.reveal_role():
                    if agent.recognized_role in agent.previous_reveals:
                        self.update_trust(to_id=agent.agent_id, change=-0.2)  # 信頼度低下
                        continue

                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(agent.get_state(self))
                        q_values = agent.model(state_tensor)
                        best_index = torch.argmax(q_values).item()
                        possible_roles = ["市民", "占い師", "霊能者", "狩人", "狂人"]
                        agent.recognized_role = possible_roles[best_index % len(possible_roles)]
                    agent.has_revealed = True
                    if agent.true_role == "霊能者" and agent.recognized_role == "霊能者":
                            self.rewards[agent.agent_id] += 10
                    elif agent.true_role == "占い師" and agent.recognized_role == "占い師":
                            self.rewards[agent.agent_id] += 10
                    elif agent.true_role == "狂人" and agent.recognized_role == "占い師":
                            self.rewards[agent.agent_id] += 10

                    print(f"プレイヤー{agent.agent_id}が{agent.recognized_role}とカミングアウト（真の役職は{agent.true_role}）")

                    if agent.recognized_role == "占い師" and agent.true_role != "占い師":
                        if real_seer:
                        if real_seer.alive:
                            print(f"プレイヤー{real_seer.agent_id}({real_seer.true_role})は、プレイヤー{agent.agent_id}を深く疑いました")
                            self.update_trust(real_seer.agent_id, agent.agent_id, -1)
                            self.liar[real_seer.agent_id][agent.agent_id] = 1
                    if agent.recognized_role == "霊能者" and agent.true_role != "霊能者":
                        if real_medium:
                        if real_medium.alive:
                            print(f"プレイヤー{real_medium.agent_id}({real_medium.true_role})は、プレイヤー{agent.agent_id}を深く疑いました")
                            self.update_trust(real_medium.agent_id, agent.agent_id, -1)
                            self.liar[real_medium.agent_id][agent.agent_id] = 1
                    if agent.recognized_role == "狩人" and agent.true_role != "狩人":
                        if real_hunter:
                        if real_hunter.alive:
                            print(f"プレイヤー{real_hunter.agent_id}({real_hunter.true_role})は、プレイヤー{agent.agent_id}を深く疑いました")
                            self.update_trust(real_hunter.agent_id, agent.agent_id, -1)
                            self.liar[real_hunter.agent_id][agent.agent_id] = 1
                else:
                    self.update_trust(to_id=agent.agent_id, change=-0.25)  # 信頼度低下

    def handle_divinations(self):
        live_agents = [a for a in self.agents if a.alive]
        for agent in live_agents:
            if agent.recognized_role == "占い師":
                target = agent.choose_action(agent.get_state(self), [a for a in live_agents if a != agent])
                result = agent.divine(target)
    
    def handle_voting(self):
        votes = {}
        for agent in [a for a in self.agents if a.alive]:
            valid_targets = [a for a in self.agents if a != agent and a.alive]
            if not valid_targets:
                continue  # **投票対象がいない場合はスキップ**
            
            target = agent.choose_action(agent.get_state(self), valid_targets)
            self.votes[agent.agent_id] = target.agent_id
            print(f"プレイヤー{agent.agent_id}({agent.true_role})がプレイヤー{target.agent_id}に投票")

            # **報酬の処理**
            if agent.true_role in ["狂人", "人狼"]:
                if target.true_role != "人狼":
                    self.rewards[agent.agent_id] += 2.
            else:
                if target.true_role == "人狼":
                    self.rewards[agent.agent_id] += 2.

            votes[target.agent_id] = votes.get(target.agent_id, 0) + 1

        if not votes:
            return  # **投票がない場合、処刑はスキップ**

        # **最多票のプレイヤーを処刑**
        eliminated = max(votes, key=votes.get)
        eliminated_agent = next(a for a in self.agents if a.agent_id == eliminated)

        print(f"プレイヤー{eliminated}({eliminated_agent.true_role})が処刑されました")

        if eliminated_agent.true_role == "人狼":
            self.rewards[eliminated_agent.agent_id] -= 0.5

        # **削除ではなく、生存フラグを変更**
        eliminated_agent.alive = False
        self.last_executed = eliminated_agent

    
    def handle_attack(self):
        wolves = [agent for agent in [a for a in self.agents if a.alive] if agent.true_role == "人狼"]
        live_agents = [a for a in self.agents if a.alive]
        if wolves:
            wolf = random.choice(wolves)
            target = wolf.take_action(wolf.get_state(self), 'attack',
                                     [a for a in live_agents if a.true_role != "人狼" and a.agent_id != wolf.agent_id])
            print(f"人狼がプレイヤー{target.agent_id}({target.true_role})を襲撃しました")
            hunter = [a for a in self.agents if a.true_role == "狩人"]
            condition = True
            if len(hunter) != 0:
                hunter = hunter[0]
                condition = hunter.protect_target != target
            
            if condition:
                print(f"プレイヤー{target.agent_id}({target.true_role})は噛み殺されました！")
                target.alive = False
            else:
                print("襲撃は失敗しました。狩人に守られました！")
    
    def handle_protection(self):
        for agent in [a for a in self.agents if a.alive]:
            live_targets = [a for a in self.agents if a.alive]
            if agent.true_role == "狩人":
                if live_targets:
                    state = agent.get_state(self)
                    target = agent.choose_action(state, [a for a in live_targets if a != agent])
                    agent.protect_target = target
                    print(f"プレイヤー{agent.agent_id}がプレイヤー{target.agent_id}({target.true_role})を守ろうとしました")
    
    def display_outcome(self):
        if self.check_victory():
            wolf_count = sum(1 for a in self.agents if (a.true_role == "人狼") and a.alive)
            if wolf_count == 0:
                print("人間側の勝利です！")
            else:
                print("人狼側の勝利です！")
    
    def assign_rewards(self):
        for agent in self.agents:
            if (agent.true_role in ["市民", "占い師", "霊能者", "狩人"] and sum(
                    1 for a in self.agents if a.true_role == "人狼") == 0) or \
                    (agent.true_role in ["人狼", "狂人"] and sum(
                        1 for a in self.agents if a.true_role == "人狼") >= sum(
                            1 for a in self.agents if a.true_role not in ["人狼", "狂人"])):
                agent.reward = 1.0
            else:
                agent.reward = -1.0
    
    def handle_phase(self, phase):
        getattr(self, phase)()
    
    def handle_accusations_and_defenses(self):
        live_agents = [a for a in self.agents if a.alive]
        for agent in live_agents:
            target = agent.choose_action(agent.get_state(self), [a for a in live_agents if a.agent_id != agent.agent_id])
            print(f"プレイヤー{agent.agent_id}({agent.true_role})がプレイヤー{target.agent_id}({target.true_role})へ疑いを表明しました")
            success = target.defend(agent.agent_id)
            if success:
                self.update_trust(to_id=agent.agent_id, change=-0.1)
            else:
                self.update_trust(to_id=target.agent_id, change=-0.1)
            print(f"プレイヤー{target.agent_id}({target.true_role})の抗弁は{'成功' if success else '失敗'}しました")
    
    def check_victory(self):
        """ ゲームの勝敗判定を行う """
        alive_agents = [agent for agent in self.agents if agent.alive]
        alive_wolves = [agent for agent in alive_agents if agent.true_role == "人狼"]
        alive_humans = [agent for agent in alive_agents if agent.true_role != "人狼"]

        print(f"生存プレイヤー数: {len(alive_agents)}, 人狼: {len(alive_wolves)}, 人間: {len(alive_humans)}")

        # **人狼が全滅 → 人間の勝利**
        if len(alive_wolves) == 0:
            return True

        # **人狼の数が他の生存者と同じか上回る → 人狼の勝利**
        if len(alive_wolves) >= len(alive_humans):
            return True

        # **ゲーム続行**
        return False

# 学習用関数
def train_agents(episodes=1000, batch_size=32):
    roles = ["市民", "市民", "市民", "人狼", "人狼", "占い師", "霊能者", "狩人", "狂人"]
    for episode in range(1, episodes + 1):
        random.shuffle(roles)
        agents = [Agent(agent_id=i, true_role=roles[i], input_size=72, output_size=5) for i in range(9)]
        win_counts = {i: 0 for i in range(9)}  # 各プレイヤーの勝利回数
        game = WerewolfGame(agents)
        game.play_game()
        
        # 勝利陣営を判定し、各プレイヤーの勝率をカウント
        human_win = sum(1 for a in game.agents if a.true_role == "人狼") == 0  # 人狼が全滅 → 人間陣営の勝利
        for agent in agents:
            if (human_win and agent.true_role in ["市民", "占い師", "霊能者", "狩人"]) or \
            (not human_win and agent.true_role in ["人狼", "狂人"]):
                win_counts[agent.agent_id] += 1
        
        # 経験を学習
        for agent in agents:
            agent.replay(batch_size)
            agent.update_epsilon()
        
        # 100エピソードごとに勝率を表示
        # 100エピソードごとに勝率を表示する
        if episode % 100 == 0:
            print(f"\n=== Episode {episode}/{episodes} ===")
            win_rates = {i: win_counts[i] / episode for i in range(9)}
            for agent_id, win_rate in win_rates.items():
                print(f"プレイヤー{agent_id}: 勝率 {win_rate:.2%}")
        game.reset_agents()
    print("全エピソード完了。モデルが学習されました。")  
    # 学習後の勝利回数と最終エピソード番号を返す
    return win_counts, episodes

def continue_training(existing_win_counts, start_episode, additional_episodes=500, batch_size=32):
    roles = ["市民", "市民", "市民", "人狼", "人狼", "占い師", "霊能者", "狩人", "狂人"]
    
    # 既存の勝率データを継続
    win_counts = existing_win_counts.copy()
    end_episodes = start_episode

    for episode in range(start_episode + 1, start_episode + additional_episodes + 1):
        random.shuffle(roles)
        agents = [Agent(agent_id=i, true_role=roles[i], input_size=72, output_size=5) for i in range(9)]
        game = WerewolfGame(agents)
        game.play_game()

        # 勝利陣営のプレイヤーに勝利カウントを加算
        human_win = sum(1 for a in game.agents if a.true_role == "人狼") == 0  # 人狼全滅 → 人間側の勝利
        for agent in agents:
            if (human_win and agent.true_role in ["市民", "占い師", "霊能者", "狩人"]) or \
               (not human_win and agent.true_role in ["人狼", "狂人"]):
                win_counts[agent.agent_id] += 1

        # 学習
        for agent in agents:
            agent.replay(batch_size)
            agent.update_epsilon()
            epsilon_now = agent.epsilon
            epsilon_now -= 1 / additional_episodes

        # 100エピソードごとに勝率を表示
        if episode % 100 == 0:
            print(f"\n=== Episode {episode}/{start_episode + additional_episodes} ===")
            win_rates = {i: win_counts[i] / episode for i in range(9)}
            for agent_id, win_rate in win_rates.items():
                print(f"プレイヤー{agent_id}: 勝率 {win_rate:.2%}")
        end_episodes = episode
        game.reset_agents()
    print("追加学習完了。モデルがさらに学習されました。")
    return win_counts, end_episodes


# 学習済みエージェントによる実プレイ関数
def play_trained_game():
    roles = ["市民", "市民", "市民", "人狼", "人狼", "占い師", "霊能者", "狩人", "狂人"]
    random.shuffle(roles)
    agents = [Agent(agent_id=i, true_role=roles[i], input_size=72, output_size=5) for i in range(9)]
    for agent in agents:
        agent.epsilon = 0.0  # 学習済みモデルで行動

    game = WerewolfGame(agents)
    game.play_game()
    game.display_outcome()
    print("人狼ゲームが終了しました。")

if __name__ == '__main__': 
    """
    学習の実行　かっこの中の左の数字は学習繰り返し回数。本来は5000回以上がおすすめだが、結構時間がかかるので
    様子を見ながら変更するべし
    """
    
    win_counts, episodes = train_agents(500, 32)
    
    # 追加で学習させたいときはTrueにする。繰り返し実行できる。
    if False:
        for _ in range(10):
            win_counts, episodes = continue_training(win_counts, episodes, additional_episodes=500, batch_size=32)
            time.sleep(2)
    
    # 上で学習を済ませたプレイヤーたちに実際に一回りプレイさせる
    play_trained_game()