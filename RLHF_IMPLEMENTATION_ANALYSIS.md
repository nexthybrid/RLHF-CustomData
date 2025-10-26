# 🎯 RLHF Pipeline Implementation Analysis

Based on OpenAI's paper "Learning to Summarize from Human Feedback", this repository implements all three key components of the RLHF pipeline.

## 📋 Table of Contents
1. [Human Feedback for Training](#1-human-feedback-for-training)
2. [Reward Model](#2-reward-model)
3. [Reinforcement Learning with PPO](#3-reinforcement-learning-with-ppo)

---

## 1. Human Feedback for Training {#1-human-feedback-for-training}

### 🔍 **Implementation Location**

**Files:**
- `Judger/judger.py` - Automated feedback system
- `Judger/data/querydata.csv` - Human preference data
- `Judger/result/results.csv` - Scored outputs

### 📊 **How It Works**

The **Judger** class implements an automated system to collect human feedback by:

1. **Comparing Multiple Outputs**: Takes human-written responses vs. machine-generated responses
2. **Multi-Criteria Scoring**:
   - **Correctness** (weight: 0.4)
   - **Relevance** (weight: 0.3) 
   - **Clarity** (weight: 0.1)
   - **Coherence** (weight: 0.1)
   - **Usefulness** (weight: 0.05)
   - **Conciseness** (weight: 0.03)
   - **Engagement** (weight: 0.02)

3. **Preference Labeling**: Creates preference pairs (human preferred vs machine preferred)

### 📁 **Data Format**

From `Judger/data/querydata.csv`:
```csv
prompt,human_answer,machine_answer
"What is the capital of France?", "The capital of France is Paris.", "Paris is the capital of France."
```

From `Judger/result/results.csv`:
```csv
prompt,human_answer,machine_answer,human_score,machine_score,preference
"What is the capital of France?", "The capital...", "Paris is...", 0.9367, 0.9298, human
```

### ✅ **Key Features**
- **Automated scoring** using sentence embeddings (cosine similarity)
- **Preference pairs** for reward model training
- **Multi-dimensional evaluation** capturing different aspects of quality

---

## 2. Reward Model {#2-reward-model}

### 🔍 **Implementation Location**

**Files:**
- `rewardModeling/models/reward_model.py` - Reward model trainer
- `rewardModeling/train.py` - Training script
- `rewardModeling/data/dataset.py` - Preference dataset
- `rewardModeling/utils/evaluation.py` - Evaluation metrics

### 📊 **How It Works**

The reward model learns to predict human preferences from feedback data:

1. **Input**: Preference pairs (preferred vs non-preferred responses)
2. **Learning**: Uses ranking loss to learn preferences
3. **Output**: Scalar reward scores

### 🔑 **Key Implementation**

```python
# rewardModeling/models/reward_model.py
class RewardModelTrainer:
    def train(self, dataloader, epochs=3):
        for batch in dataloader:
            # Get rewards for preferred and non-preferred
            preferred_rewards = self.model(preferred_input_ids, attention_mask).logits
            non_preferred_rewards = self.model(non_preferred_input_ids, attention_mask).logits
            
            # Compute ranking loss
            logits = preferred_rewards - non_preferred_rewards
            loss = self.loss_fn(logits, labels)
            
            # Backpropagation
            loss.backward()
```

### ✅ **Key Features**
- **Pairwise ranking** loss (preferred vs non-preferred)
- **Scalar reward** prediction
- **Preference learning** from human feedback
- **Evaluation metrics**: Accuracy and Spearman correlation

---

## 3. Reinforcement Learning with PPO {#3-reinforcement-learning-with-ppo}

### 🔍 **Implementation Location**

**Files:**
- `policyOptimiyation/train.py` - PPO training loop
- `policyOptimiyation/utils/ppo_utils.py` - PPO loss and advantages
- `policyOptimiyation/generate_trajectories.py` - Trajectory generation
- `policyOptimiyation/models/policy_model.py` - Policy model
- `policyOptimiyation/models/reward_model.py` - Reward model loading

### 📊 **How It Works**

The PPO training loop follows the standard RLHF process:

1. **Generate Trajectories**: Policy model generates responses
2. **Compute Rewards**: Reward model scores the responses
3. **Compute Advantages**: Using discounted cumulative rewards
4. **PPO Update**: Optimize policy to maximize rewards

### 🔑 **Key Implementation**

```python
# policyOptimiyation/train.py
for epoch in range(EPOCHS):
    # Generate trajectories
    trajectories = generate_trajectories(policy_model, reward_model, tokenizer, prompts)
    
    # Compute advantages
    advantages = compute_advantages(rewards)
    
    # PPO loss calculation
    loss = ppo_loss(old_action_prob, new_action_prob, advantage)
    loss.backward()
    optimizer.step()
```

```python
# policyOptimiyation/utils/ppo_utils.py
def ppo_loss(old_probs, new_probs, advantages, clip_epsilon=0.2):
    ratio = new_probs / old_probs
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    return -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

def compute_advantages(rewards, gamma=0.99):
    # Discounted cumulative rewards
    advantages = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        advantages.insert(0, R)
    return advantages
```

### ✅ **Key Features**
- **PPO loss** with clipping (ε=0.2)
- **Discounted advantages** (γ=0.99)
- **Trajectory generation** with policy model
- **Reward computation** using reward model
- **Policy optimization** to maximize rewards

---

## 🔄 Complete RLHF Pipeline

```
1. Human Feedback Collection (Judger/)
   ↓
   Generate preference pairs from human feedback
   ↓
2. Reward Model Training (rewardModeling/)
   ↓
   Learn to predict human preferences
   ↓
3. Policy Optimization (policyOptimiyation/)
   ↓
   Use PPO to maximize rewards from reward model
   ↓
4. Optimized Policy Model
```

---

## 📊 **Key Differences from OpenAI Paper**

| Component | OpenAI Paper | This Repository |
|-----------|--------------|-----------------|
| **Human Feedback** | Human labelers | Automated Judger system |
| **Reward Model** | Trained on human rankings | Trained on preference pairs |
| **Policy Optimization** | PPO | PPO implementation |
| **Base Model** | GPT-3 | Flan-T5 |

---

## 🎯 **Summary**

✅ **Human Feedback**: Implemented in `Judger/` with automated scoring system
✅ **Reward Model**: Implemented in `rewardModeling/` with pairwise ranking
✅ **PPO**: Implemented in `policyOptimiyation/` with standard PPO algorithm

**All three components are fully implemented and ready to use!** 🚀
