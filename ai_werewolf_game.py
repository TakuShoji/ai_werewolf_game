# ai_werewolf_game.py — DQN Werewolf (balanced preset: ~50/50を狙う推奨設定)
# - 村の証拠バイアス: 投票 α=1.00 / 疑い α=0.55
# - 占い影響: mag = 1.50 / seer_num
# - 狩人: 自己護衛禁止 + 同一対象の連続護衛禁止 + 情報役優先護衛 ~87%
# - 狼バンドワゴン: D1 β=0.22, D2+ β=0.27, トップ村人2票以上のときのみ、各狼65%の確率で適用
# - 詳細ログ: play_trained_game() 実行時のみ verbose=True で表示

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
from typing import List, Tuple

# =========================================================
# ハイパーパラメータ
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

STATE_SIZE = 72
ACTION_SIZE = 9                 # 常に0〜8（プレイヤーID）
HIDDEN = 128
LR = 1e-3
GAMMA_MIN, GAMMA_MAX = 0.85, 0.99
MEM_CAP = 5000
BATCH_SIZE = 64
TARGET_SYNC_FREQ = 200          # ステップごとのターゲット同期周期
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.9993              # 長めの探索
TRAIN_STEPS_PER_EP = 200        # 1エピソードで最大この回数だけreplay

# =========================================================
# DQN
# =========================================================
class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# =========================================================
# エージェント
# =========================================================
class Agent:
    def __init__(self, agent_id: int, true_role: str):
        self.agent_id = agent_id
        self.true_role = true_role
        self.recognized_role = "市民"
        self.alive = True

        # 学習関連
        self.gamma = (GAMMA_MAX - GAMMA_MIN) * random.random() + GAMMA_MIN
        self.epsilon = EPS_START
        self.memory = deque(maxlen=MEM_CAP)
        self.model = DQN(STATE_SIZE, ACTION_SIZE)
        self.target_model = DQN(STATE_SIZE, ACTION_SIZE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.learn_steps = 0

        # 会話スキル（簡易）
        self.talking_skill = 0
        self.deception_skill = 0

        # CO/占い/人狼相互認識
        self.revealed_roles = set()
        self.has_revealed = False
        self.protect_target = None
        self.divine_targets = []
        self.werewolf_revealed = [0] * 9
        if self.true_role == "人狼":
            self.werewolf_revealed[self.agent_id] = 1

        # 直前護衛対象（連続護衛禁止用）
        self.last_protect_id = None

    # ---------- 行動選択（ε-greedy＋PyTorchマスク） ----------
    def choose_action(self, state: List[float], valid_ids: List[int], use_epsilon: bool = True) -> int:
        if not valid_ids:
            return self.agent_id
        if use_epsilon and random.random() < self.epsilon:
            return random.choice(valid_ids)

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = self.model(x)[0]  # PyTorchのまま扱う

        mask = torch.full((ACTION_SIZE,), -1e9, dtype=torch.float32)
        mask[valid_ids] = 0.0
        a = int(torch.argmax(q + mask).item())
        if a not in valid_ids:
            a = random.choice(valid_ids)
        return a

    # ---------- 経験保存 ----------
    def store(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, float(done)))

    # ---------- 学習 ----------
    def replay(self, batch_size=BATCH_SIZE):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s = torch.tensor(states, dtype=torch.float32)
        ns = torch.tensor(next_states, dtype=torch.float32)
        a = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        r = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        d = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q = self.model(s).gather(1, a)
        with torch.no_grad():
            next_q = self.target_model(ns).max(1, keepdim=True)[0]
            y = r + self.gamma * next_q * (1 - d)

        loss = nn.MSELoss()(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % TARGET_SYNC_FREQ == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    # ---------- ゲーム固有の軽量ロジック ----------
    def defend(self, accuser_id):
        skill = self.talking_skill if self.true_role == self.recognized_role else self.deception_skill
        success_chance = 0.5 + 0.05 * skill
        success = random.random() < success_chance
        if success and random.random() < 0.05:
            if self.true_role == self.recognized_role:
                self.talking_skill = min(10, self.talking_skill + 1)
            else:
                self.deception_skill = min(10, self.deception_skill + 1)
        return success

    def get_state(self, game) -> List[float]:
        alive_status = [int(a.alive) for a in game.agents]                             # 9
        trust_levels = list(game.trust[self.agent_id])                                  # 9
        has_revealed = [int(a.has_revealed) for a in game.agents]                       # 9
        votes = list(game.votes)                                                        # 9
        divination_results = list(game.divination_results)                              # 9
        recognized_roles = [game.role_to_index[a.recognized_role] for a in game.agents] # 9
        liars = list(game.liar[self.agent_id])                                          # 9
        werewolf_revealed = list(self.werewolf_revealed)                                # 9

        state = (alive_status + trust_levels + has_revealed + votes +
                 divination_results + recognized_roles + liars + werewolf_revealed)

        while len(state) < STATE_SIZE:
            state.append(0.0)
        return state[:STATE_SIZE]

# =========================================================
# ゲーム
# =========================================================
class WerewolfGame:
    def __init__(self, agents: List[Agent], verbose: bool = False):
        self.agents = agents
        self.verbose = verbose
        self.day = 0
        self.trust = [[0.5] * 9 for _ in range(9)]
        self.divination_results = [0.0] * 9
        self.last_executed = None
        self.votes = [-1] * 9
        self.liar = [[0] * 9 for _ in range(9)]
        self.role_to_index = {"市民": 0, "占い師": 1, "霊能者": 2, "狩人": 3, "人狼": 4, "狂人": 5}
        self.index_to_role = {v: k for k, v in self.role_to_index.items()}
        self.rewards = {a.agent_id: 0.0 for a in self.agents}

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def _role(self, a: Agent) -> str:
        return self.index_to_role[self.role_to_index[a.true_role]]

    def werewolfship(self):
        # 初期の人狼相互信頼は控えめ（+0.1）
        wolves = [a for a in self.agents if a.true_role == "人狼"]
        if len(wolves) >= 2:
            for i in range(len(wolves)):
                for j in range(i + 1, len(wolves)):
                    w1, w2 = wolves[i], wolves[j]
                    self.update_trust(w1.agent_id, w2.agent_id, 0.1)
                    self.update_trust(w2.agent_id, w1.agent_id, 0.1)
                    w1.werewolf_revealed[w2.agent_id] = 1
                    w2.werewolf_revealed[w1.agent_id] = 1

    def reset_agents(self):
        for agent in self.agents:
            agent.recognized_role = "市民"
            agent.has_revealed = False
            agent.revealed_roles.clear()
            agent.alive = True
            agent.divine_targets = []
            agent.protect_target = None
            agent.werewolf_revealed = [0] * 9
            agent.last_protect_id = None   # ★ 連続護衛履歴のクリア
            if agent.true_role == "人狼":
                agent.werewolf_revealed[agent.agent_id] = 1

    def update_trust(self, from_id=None, to_id=None, change=0.0):
        if from_id is None and to_id is None:
            return
        if from_id is None:
            for i in range(9):
                self.trust[i][to_id] = self._clip01(self.trust[i][to_id] + change)
        elif to_id is None:
            for j in range(9):
                self.trust[from_id][j] = self._clip01(self.trust[from_id][j] + change)
        else:
            self.trust[from_id][to_id] = self._clip01(self.trust[from_id][to_id] + change)

    @staticmethod
    def _clip01(x): return max(0.0, min(1.0, x))

    # ----------------- フェーズ -----------------
    def handle_reveals(self):
        # 真占い師はD1:80%でCO、D2以降は必ずCO。他は従来通り25%でCO。
        for agent in [a for a in self.agents if a.alive]:
            if agent.true_role == "占い師":
                do_reveal = (self.day == 1 and random.random() < 0.8) or (self.day >= 2)
            else:
                do_reveal = (random.random() < 0.25)
            if not do_reveal:
                continue

            possible_roles = list({"市民", "占い師", "霊能者", "狩人", "狂人", "人狼"} - agent.revealed_roles)
            if not possible_roles:
                continue

            before_role = agent.recognized_role
            agent.recognized_role = random.choice(possible_roles)
            agent.revealed_roles.add(agent.recognized_role)
            agent.has_revealed = True

            self.log(f"プレイヤー{agent.agent_id}が{agent.recognized_role}とカミングアウト（真の役職は{agent.true_role}）")

            if before_role != agent.recognized_role and len(agent.revealed_roles) > 1:
                self.update_trust(to_id=agent.agent_id, change=-0.25)

            def punish_if_fake(role_name):
                real = next((x for x in self.agents if x.true_role == role_name and x.alive), None)
                if real and agent.true_role != role_name:
                    self.update_trust(real.agent_id, agent.agent_id, -1.0)
                    self.liar[real.agent_id][agent.agent_id] = 1

            if agent.recognized_role in ("占い師", "霊能者", "狩人"):
                punish_if_fake(agent.recognized_role)

    def perform_divinations(self):
        # 占いの影響：バランス用に中庸（mag=1.50/CO数）
        for agent in [a for a in self.agents if a.alive]:
            is_seer_claim = (agent.recognized_role == "占い師" and agent.has_revealed)
            is_true_seer = (agent.true_role == "占い師" and not agent.has_revealed)
            if not (is_seer_claim or is_true_seer):
                continue

            not_divined = [x for x in self.agents if x.alive and x not in agent.divine_targets and x.agent_id != agent.agent_id]
            if not not_divined:
                continue

            state = agent.get_state(self)
            valid_ids = [x.agent_id for x in not_divined]
            target_id = agent.choose_action(state, valid_ids, use_epsilon=True)
            target = self.agents[target_id]

            s = state
            result = "黒" if (target.true_role == "人狼") else "白"

            if agent.true_role == "占い師":
                r = 2.0 if result == "黒" else 0.5
            else:
                r = 0.35 if result == "黒" else 0.05

            seer_num = max(1, len([a for a in self.agents if a.recognized_role == "占い師"]))
            mag = 1.50 / seer_num
            delta = mag if (result == "白") else -mag
            self.divination_results[target.agent_id] += delta
            self.update_trust(to_id=target.agent_id, change=delta)

            if is_seer_claim and agent.true_role != "占い師":
                if result == "黒" and target.true_role in ["市民", "霊能者", "狩人"]:
                    r -= 1.2
                    self.update_trust(target.agent_id, agent.agent_id, -1.1)
                    self.liar[target.agent_id][agent.agent_id] = 1

            agent.divine_targets.append(target)
            agent.werewolf_revealed[target.agent_id] = 1 if result == "黒" else -1

            
            if is_true_seer and self.verbose: self.log(f"真の占い師プレイヤー{agent.agent_id}({agent.recognized_role})がプレイヤー{target.agent_id}を密かに占い、{result}と判定")
            
            self.log(f"プレイヤー{agent.agent_id}({agent.recognized_role})がプレイヤー{target.agent_id}を占い、{result}と判定")

            ns = agent.get_state(self)
            agent.store(s, target_id, r, ns, False)
            
    def perform_medium_check(self):
        # 前日に処刑されたプレイヤーを真霊能だけが密かに判定（公開はしない）
        if self.last_executed is None:
            return
        medium = next((a for a in self.agents if a.alive and a.true_role == "霊能者"), None)
        if medium is None:
            return
        res = "黒" if self.last_executed.true_role == "人狼" else "白"
        self.log(f"真の霊能者プレイヤー{medium.agent_id}が密かに処刑されたプレイヤー{self.last_executed.agent_id}を判定: {res}")
        # 学習への微小信号（なくても可）
        s = medium.get_state(self)
        medium.store(s, self.last_executed.agent_id, +0.02, s, False)
        # 霊能者の“内ポケット”に最新結果を保持（COしてたら次の関数で公開する）
        medium.last_medium_result = (self.last_executed.agent_id, res)
        
    def announce_medium_result_if_co(self):
        # CO済みの霊能者だけが、直近の検死結果を公開する（あれば）
        medium = next((a for a in self.agents
                       if a.alive and a.has_revealed and a.recognized_role == "霊能者"), None)
        if medium is None:
            return
        lmr = getattr(medium, "last_medium_result", None)
        if not lmr:
            return
        pid, res = lmr
        self.log(f"霊能CO中のプレイヤー{medium.agent_id}が前日の処刑者{pid}は『{res}』だったと公開")
        # 公開で霊能の信用が少し上がる（村全体から）
        self.update_trust(to_id=medium.agent_id, change=+0.10)
        # 一度公開した判定は消す（同じ内容の連呼を防止）
        medium.last_medium_result = None
    
    def handle_accusations_and_defenses(self):
        live = [a for a in self.agents if a.alive]
        for agent in live:
            valid_ids = [x.agent_id for x in live if x.agent_id != agent.agent_id]
            if not valid_ids:
                continue
            s = agent.get_state(self)

            # --- 村は証拠バイアス（弱め α=0.55）、狼は従来 ---
            if agent.true_role in ["人狼", "狂人"]:
                target_id = agent.choose_action(s, valid_ids, use_epsilon=True)
            else:
                x = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q = agent.model(x)[0].clone()
                alpha = 0.55
                bias = torch.full((ACTION_SIZE,), -1e9, dtype=torch.float32)
                for pid in valid_ids:
                    bias[pid] = -alpha * self.divination_results[pid]  # 黒いほど+方向
                mask = torch.full((ACTION_SIZE,), -1e9, dtype=torch.float32)
                mask[valid_ids] = 0.0
                target_id = int(torch.argmax(q + bias + mask).item())
                if target_id not in valid_ids:
                    target_id = random.choice(valid_ids)

            target = self.agents[target_id]
            self.log(f"プレイヤー{agent.agent_id}({agent.true_role})がプレイヤー{target.agent_id}({target.true_role})へ疑いを表明しました")
            success = target.defend(agent.agent_id)
            if success:
                self.update_trust(to_id=agent.agent_id, change=-0.1)
                r = -0.05
                self.log(f"プレイヤー{target.agent_id}({target.true_role})の抗弁は成功しました")
            else:
                self.update_trust(to_id=target.agent_id, change=-0.1)
                r = +0.05
                self.log(f"プレイヤー{target.agent_id}({target.true_role})の抗弁は失敗しました")
            ns = agent.get_state(self)
            agent.store(s, target_id, r, ns, False)

    def handle_voting(self):
        votes = {}
        self.votes = [-1] * 9
        live = [a for a in self.agents if a.alive]
        per_vote_store = []  # 遅延報酬用に保持

        # 現時点の最多得票の村人を返す（ID, 得票数）
        def _current_top_human(votes_dict):
            if not votes_dict:
                return None, 0
            max_v = max(votes_dict.values())
            tops = [pid for pid, v in votes_dict.items() if v == max_v]
            tops = [pid for pid in tops if self.agents[pid].true_role != "人狼"]
            if not tops:
                return None, 0
            return random.choice(tops), max_v

        for agent in live:
            valid_ids = [x.agent_id for x in live if x.agent_id != agent.agent_id]
            if not valid_ids:
                continue
            s = agent.get_state(self)

            if agent.true_role in ["人狼", "狂人"]:
                # --- 狼バンドワゴン（抑制付き） ---
                x = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q = agent.model(x)[0].clone()
                mask = torch.full((ACTION_SIZE,), -1e9, dtype=torch.float32)
                mask[valid_ids] = 0.0

                top_h, top_votes = _current_top_human(votes)
                band = torch.zeros(ACTION_SIZE, dtype=torch.float32)

                beta = 0.22 if self.day <= 1 else 0.27
                ok_density = (top_h is not None and top_h in valid_ids and top_votes >= 2)
                apply_prob = 0.65
                if ok_density and random.random() < apply_prob:
                    band[top_h] = beta

                target_id = int(torch.argmax(q + band + mask).item())
                if target_id not in valid_ids:
                    target_id = random.choice(valid_ids)
            else:
                # --- 村は証拠バイアス（投票は α=1.00） ---
                x = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q = agent.model(x)[0].clone()
                alpha = 0.86
                bias = torch.full((ACTION_SIZE,), -1e9, dtype=torch.float32)
                for pid in valid_ids:
                    bias[pid] = -alpha * self.divination_results[pid]
                mask = torch.full((ACTION_SIZE,), -1e9, dtype=torch.float32)
                mask[valid_ids] = 0.0
                target_id = int(torch.argmax(q + bias + mask).item())
                if target_id not in valid_ids:
                    target_id = random.choice(valid_ids)

            self.votes[agent.agent_id] = target_id
            target = self.agents[target_id]
            self.log(f"プレイヤー{agent.agent_id}({agent.true_role})がプレイヤー{target.agent_id}に投票")

            # 即時報酬（村の正解票は控えめ +0.55）
            if agent.true_role in ["狂人", "人狼"]:
                r = +0.1 if target.true_role not in ["人狼"] else -0.618
            else:
                r = +0.55 if target.true_role == "人狼" else -0.1

            # 証拠フォロー・ボーナス（上限0.30、緩やか）
            if agent.true_role not in ["人狼", "狂人"]:
                ev = self.divination_results[target_id]
                if ev < 0:
                    bonus = min(0.30, 0.12 + 0.22 * min(2.0, -ev))
                    r += bonus
                elif ev >= 1.0:
                    r -= 0.2

            ns = agent.get_state(self)
            agent.store(s, target_id, r, ns, False)

            votes[target_id] = votes.get(target_id, 0) + 1
            per_vote_store.append((agent.agent_id, target_id))

        if not votes:
            return

        max_votes = max(votes.values())
        tops = [i for i, v in votes.items() if v == max_votes]
        eliminated = random.choice(tops) if len(tops) > 1 else tops[0]
        eliminated_agent = self.agents[eliminated]
        eliminated_agent.alive = False
        self.last_executed = eliminated_agent

        # 決定ログ
        if len(tops) > 1:
            self.log("最多得票者が複数いたため信頼度の低い者を対象とします")
        self.log(f"プレイヤー{eliminated_agent.agent_id}({eliminated_agent.true_role})が処刑されました")

        # 投票の遅延報酬（弱め）
        for aid, tid in per_vote_store:
            if tid != eliminated_agent.agent_id:
                continue
            a = self.agents[aid]
            s = a.get_state(self); ns = s
            if eliminated_agent.true_role == "人狼":
                gain = 0.8 if a.true_role not in ["人狼", "狂人"] else 0.45
                a.store(s, eliminated_agent.agent_id, +gain, ns, False)
            else:
                loss = -0.55 if a.true_role not in ["人狼", "狂人"] else +0.12
                a.store(s, eliminated_agent.agent_id, loss, ns, False)

        # 処刑の即時影響（控えめ）
        if eliminated_agent.true_role == "人狼":
            for a in self.agents:
                if a.true_role in ["人狼", "狂人"]:
                    self.rewards[a.agent_id] -= 0.2
                else:
                    self.rewards[a.agent_id] += 0.2
        else:
            for a in self.agents:
                if a.true_role in ["人狼", "狂人"]:
                    self.rewards[a.agent_id] += 0.1
                else:
                    self.rewards[a.agent_id] -= 0.1

        self._log_counts()

    def handle_protection(self):
        # 自己護衛禁止 + 同一対象連続護衛禁止 + 情報役CO優先（~87%）
        live = [a for a in self.agents if a.alive]
        for agent in live:
            if agent.true_role != "狩人":
                continue

            last_pid = getattr(agent, "last_protect_id", None)

            # 自己護衛・連続護衛の禁止
            valid_ids = [
                x.agent_id for x in live
                if x.agent_id != agent.agent_id
                and x.agent_id != last_pid
            ]
            if not valid_ids:
                valid_ids = [x.agent_id for x in live if x.agent_id != agent.agent_id]
            if not valid_ids:
                continue

            s = agent.get_state(self)

            # 情報役（占い/霊能）のCO者を優先（確率的に）
            priority = [
                x.agent_id for x in live
                if x.has_revealed
                and x.recognized_role in ["占い師", "霊能者"]
                and x.agent_id != agent.agent_id
                and x.agent_id != last_pid
            ]
            use_priority = (random.random() < 0.87)

            if priority and use_priority:
                target_id = random.choice(priority)
            else:
                target_id = agent.choose_action(s, valid_ids, use_epsilon=True)

            agent.protect_target = self.agents[target_id]
            agent.last_protect_id = target_id  # 連続護衛禁止に使用

            self.log(f"プレイヤー{agent.agent_id}がプレイヤー{target_id}({self.agents[target_id].true_role})を守ろうとしました")

            ns = agent.get_state(self)
            agent.store(s, target_id, +0.01, ns, False)

    def handle_attack(self):
        # 護衛成功のご褒美・狼の罰は強め／襲撃成功の旨味はゼロ
        wolves = [a for a in self.agents if a.alive and a.true_role == "人狼"]
        live = [a for a in self.agents if a.alive]
        if not wolves:
            return
        wolf = random.choice(wolves)
        valid_ids = [x.agent_id for x in live if (x.agent_id != wolf.agent_id and x.true_role != "人狼")]
        if not valid_ids:
            return
        s = wolf.get_state(self)
        target_id = wolf.choose_action(s, valid_ids, use_epsilon=True)
        target = self.agents[target_id]

        hunters = [a for a in self.agents if a.alive and a.true_role == "狩人"]
        protected = False
        if hunters:
            h = hunters[0]
            protected = (h.protect_target is not None and h.protect_target.agent_id == target_id)

        if protected:
            r = -1.0  # 狼の罰
            for h in [a for a in self.agents if a.alive and a.true_role == "狩人"]:
                s_h = h.get_state(self)
                ns_h = s_h
                h.store(s_h, target_id, +1.0, ns_h, False)
            self.log(f"人狼がプレイヤー{target_id}({target.true_role})を襲撃→護衛により無効化！")
        else:
            target.alive = False
            r = +0.0  # 襲撃成功の旨味はほぼゼロ
            self.log(f"人狼がプレイヤー{target_id}({target.true_role})を襲撃し、噛み殺しました！")

        ns = wolf.get_state(self)
        wolf.store(s, target_id, r, ns, False)
        for a in self.agents:
            if a.true_role == "狩人":
                a.protect_target = None

        self._log_counts()

    # ----------------- 進行＆勝敗 -----------------
    def _log_counts(self):
        if not self.verbose:
            return
        alive_agents = [a for a in self.agents if a.alive]
        alive_wolves = [a for a in alive_agents if a.true_role == "人狼"]
        alive_humans = [a for a in alive_agents if a.true_role != "人狼"]
        self.log(f"生存プレイヤー数: {len(alive_agents)}, 人狼: {len(alive_wolves)}, 人間: {len(alive_humans)}")

    def check_victory(self):
        alive_agents = [a for a in self.agents if a.alive]
        alive_wolves = [a for a in alive_agents if a.true_role == "人狼"]
        alive_humans = [a for a in alive_agents if a.true_role != "人狼"]
        if len(alive_wolves) == 0:
            return 1   # 人間勝利
        if len(alive_wolves) >= len(alive_humans):
            return -1  # 人狼勝利
        return 0       # 継続

    def play_game(self, max_days=15) -> int:
        self.rewards = {a.agent_id: 0.0 for a in self.agents}
        self.reset_agents()
        self.werewolfship()

        self.day = 0
        steps_this_ep = 0
        while self.day < max_days:
            self.day += 1

            # 昼の見出し（実プレイのみ出る）
            self.log("=" * 30); self.log(f"{self.day}日目 - 昼"); self.log("=" * 30)

            # ★ 霊能：密かな検死 → CO中なら公開
            self.perform_medium_check()
            self.announce_medium_result_if_co()

            # 以降は従来どおり
            self.handle_reveals()
            self.perform_divinations()
            self.handle_accusations_and_defenses()
            self.handle_voting()
            result = self.check_victory()
            if result != 0:
                break

            # 夜
            self.log("=" * 30); self.log(f"{self.day}日目 - 夜"); self.log("=" * 30)
            self.handle_protection()
            self.handle_attack()
            result = self.check_victory()
            if result != 0:
                break

            for a in self.agents:
                a.replay()

            steps_this_ep += 1
            if steps_this_ep > TRAIN_STEPS_PER_EP:
                break
        
        # 終局報酬（村寄りだが過剰でない）
        human_win = (result == 1)
        if self.verbose:
            self.log("【ゲーム終了】")
            alive_agents = [a for a in self.agents if a.alive]
            alive_wolves = [a for a in alive_agents if a.true_role == "人狼"]
            alive_humans = [a for a in alive_agents if a.true_role != "人狼"]
            self.log(f"生存プレイヤー数: {len(alive_agents)}, 人狼: {len(alive_wolves)}, 人間: {len(alive_humans)}")
            self.log("人間側の勝利です！" if human_win else "人狼側の勝利です！")
            self.log("人狼ゲームが終了しました。")

        for a in self.agents:
            if human_win:
                r = +4.0 if a.true_role in ["市民", "占い師", "霊能者", "狩人"] else -1.0
            else:
                r = +2.0 if a.true_role in ["人狼", "狂人"] else -1.0
            s = a.get_state(self)
            a.store(s, a.agent_id, r, s, True)

        for a in self.agents:
            for _ in range(3):
                a.replay()

        return result

# =========================================================
# 学習ルーチン
# =========================================================
def make_agents() -> List[Agent]:
    roles = ["市民", "市民", "市民", "人狼", "人狼", "占い師", "霊能者", "狩人", "狂人"]
    random.shuffle(roles)
    return [Agent(agent_id=i, true_role=roles[i]) for i in range(9)]

def train_agents(episodes=400) -> Tuple[dict, int, int, int]:
    total_human_win = 0
    total_wolf_win = 0
    win_counts = {i: 0 for i in range(9)}

    for ep in range(1, episodes + 1):
        agents = make_agents()
        game = WerewolfGame(agents, verbose=False)  # 学習時は黙る
        result = game.play_game()  # 1:人間勝利 / -1:人狼勝利

        if result == 1:
            total_human_win += 1
        else:
            total_wolf_win += 1

        human_win = (result == 1)
        for a in agents:
            if (human_win and a.true_role in ["市民", "占い師", "霊能者", "狩人"]) or \
               ((not human_win) and a.true_role in ["人狼", "狂人"]):
                win_counts[a.agent_id] += 1

        for a in agents:
            a.update_epsilon()

        if ep % 50 == 0:
            print(f"[Episode {ep}/{episodes}] HumanWin {total_human_win} / WolfWin {total_wolf_win} "
                  f"({total_human_win/ep:.1%} vs {total_wolf_win/ep:.1%})")

    print("=== 学習完了 ===")
    print(f"人間陣営勝率：{total_human_win/episodes:.1%}　人狼陣営勝率：{total_wolf_win/episodes:.1%}")
    return win_counts, episodes, total_human_win, total_wolf_win

def continue_training(existing_win_counts: dict,
                      start_episode: int,
                      total_human_win: int,
                      total_wolf_win: int,
                      additional_episodes=200) -> Tuple[dict, int, int, int]:

    for ep in range(start_episode + 1, start_episode + additional_episodes + 1):
        agents = make_agents()
        game = WerewolfGame(agents, verbose=False)  # 追加学習も黙る
        result = game.play_game()

        if result == 1:
            total_human_win += 1
        else:
            total_wolf_win += 1

        human_win = (result == 1)
        for a in agents:
            if (human_win and a.true_role in ["市民", "占い師", "霊能者", "狩人"]) or \
               ((not human_win) and a.true_role in ["人狼", "狂人"]):
                existing_win_counts[a.agent_id] += 1

        for a in agents:
            a.update_epsilon()

        if ep % 50 == 0:
            total = ep
            print(f"[Cont Episode {ep}/{start_episode + additional_episodes}] "
                  f"Human {total_human_win/total:.1%} / Wolf {total_wolf_win/total:.1%}")

    end_episodes = start_episode + additional_episodes
    print("=== 追加学習完了 ===")
    print(f"累計 人間陣営勝率：{total_human_win/end_episodes:.1%}　人狼陣営勝率：{total_wolf_win/end_episodes:.1%}")
    return existing_win_counts, end_episodes, total_human_win, total_wolf_win

def play_trained_game():
    agents = make_agents()
    for a in agents:
        a.epsilon = 0.0
    game = WerewolfGame(agents, verbose=True)   # ★ ログON
    result = game.play_game()
    print("【実プレイ結果】", "人間勝利" if result == 1 else "人狼勝利")

# =========================================================
# main
# =========================================================
if __name__ == '__main__':
    # 環境に合わせてエピソード数を増減（例：400〜2000）
    win_counts, episodes, total_human_win, total_wolf_win = train_agents(episodes=10000)

    # 追加学習（必要に応じて）
    if False:
        win_counts, episodes, total_human_win, total_wolf_win = continue_training(
            win_counts, episodes, total_human_win, total_wolf_win,
            additional_episodes=200
        )
        time.sleep(1)

    # 学習済みモデルで実プレイ（詳細ログON）
    play_trained_game()
