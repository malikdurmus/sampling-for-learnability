# The Ultimate Guide to JaxNav Sampling For Learnability (SFL)

This document provides a comprehensive, mathematically rigorous, and fully exhaustive breakdown of the JaxNav SFL training script (`sfl/train/jaxnav_sfl.py`) and its associated configuration (`jaxnav-sfl.yaml`). 

---

## 1. The Configuration (`jaxnav-sfl.yaml`)

The configuration file is managed by Hydra, which allows for nested and composable YAML configurations. It dictates everything from network architecture to curriculum strategies.

### Base Configuration Parameters
*   `SEED`: Controls all randomness in JAX. Since JAX uses explicit Pseudo-Random Number Generators (PRNGs), this single seed ensures complete reproducibility.
*   `BATCH_SIZE` (e.g., 1000) & `NUM_BATCHES` (e.g., 5): Determines how many total environments are generated per cycle. (e.g., $1000 \times 5 = 5000$ total maps).
*   `ROLLOUT_STEPS`: The number of steps the agent interacts with the environment during the empirical SFL evaluation phase.
*   `NUM_TO_SAVE`: Out of the 5000 generated maps, how many the curriculum should actually select and pass to the agent for training (e.g., 100).

### The `LEARN_METHOD` Key
This is the master switch that controls the entire pipeline's logic.
1.  `"standard"`: Classical SFL. The agent rolls out in all 5000 maps, calculates empirical learnability, and picks the best 100.
2.  `"random"`: Baseline. Randomly selects 100 maps from the 5000.
3.  `"cnn"`: Uses the pre-trained Difficulty ResNet to predict map difficulty from raw images without running the agent, then applies a curriculum strategy to pick 100 maps.
4.  `"hybrid_*"`: A fusion of `standard` and `cnn`. The agent rolls out to get empirical scores, the CNN predicts structural difficulty, and the two are mathematically combined using a specific compound scoring method (e.g., `hybrid_linear`, `hybrid_multiplicative`).

### Curriculum Strategy Parameters
These apply only when using `cnn` or `hybrid` methods.
*   `CURRICULUM_STRATEGY`: Chooses between `"time_based"`, `"performance_adaptive"`, or `"gaussian_frontier"`.
*   `CURRICULUM_START_MU`: The initial target difficulty score (e.g., `0.05` for very easy maps).
*   `CURRICULUM_STEP_SIZE`: How aggressively the target difficulty shifts (used in `performance_adaptive`).
*   `CURRICULUM_GAUSSIAN_VAR`: The variance $\sigma^2$ (spread) of the probability distribution when using `gaussian_frontier`.

---

## 2. The Core PPO Training Loop (`train_step`)

The agent is trained using Independent Proximal Policy Optimization (IPPO). This entire loop is decorated with `@jax.jit` and run via `jax.lax.scan`, meaning it executes entirely on the GPU without returning to Python.

### The Rollout Phase
The agent interacts with the `NUM_TO_SAVE` selected environments.
*   **Action Selection**: The Actor network outputs a Gaussian distribution. JAX samples an action (linear/angular velocity) from this distribution.
*   **Environment Step**: The environment processes the action, updating physics (positions, collisions) and returning the next observation, reward, and a `done` flag.
*   **Trajectory Buffer**: All observations, actions, rewards, values, and log-probabilities are stacked into large arrays for the PPO update.

### Advantage Estimation (GAE)
Once the rollout is complete, the script calculates **Generalized Advantage Estimation**.
*   **Value Target**: What the actual discounted return was for a state.
*   **Advantage**: How much *better* or *worse* the agent's chosen action was compared to what the Critic initially predicted. $A_t = \delta_t + (\gamma \lambda) \delta_{t+1} + \dots$ where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

### The PPO Update
The network weights are updated to maximize expected return.
*   **Actor Loss**: Modifies the policy to increase the probability of actions with positive advantages. It uses a **Clipped Surrogate Objective** ($L^{CLIP}$), which prevents the policy from changing too aggressively in a single step by clamping the probability ratio between $[1 - \epsilon, 1 + \epsilon]$.
*   **Critic Loss**: A simple Mean Squared Error (MSE) between the Critic's predictions and the actual Value Targets.
*   **Entropy Bonus**: A small mathematical penalty that discourages the Actor's distribution from becoming too narrow, forcing the agent to continue exploring.

---

## 3. Learnability Sampling Approaches

This is where the Unsupervised Environment Design (UED) happens. The script generates 5000 random environments and must select the best 100 (`NUM_TO_SAVE`) to train the agent on.

### A. Standard Empirical SFL (`get_learnability_set_standard`)
This method relies purely on the agent's current capabilities.
1.  The current agent policy is cloned and rolled out across all 5000 environments.
2.  The script tracks whether the agent succeeds or fails, establishing an empirical **Solvability** score ($S$) ranging from $0.0$ to $1.0$ for each map.
3.  **Learnability** ($L$) is calculated as the variance of the Bernoulli success distribution:
    $$L = S \times (1 - S)$$
    *   If $S = 0.0$ (Impossible), $L = 0.0$.
    *   If $S = 1.0$ (Trivial), $L = 0.0$.
    *   If $S = 0.5$ (Perfectly challenging), $L = 0.25$ (Maximum learnability).
4.  The environments are sorted by $L$, and the top 100 are selected.

### B. CNN-Based SFL (`get_learnability_set_cnn`)
This method avoids the costly agent rollouts.
1.  All 5000 environments are rendered natively in JAX into 2D top-down images.
2.  These images are passed through a pre-trained ResNet CNN, which outputs a raw logit predicting difficulty.
3.  **Min-Max Normalization** is applied to ensure scores are strictly between $0.0$ and $1.0$:
    $$CNN_{norm} = \frac{CNN_{raw} - \min(CNN_{raw})}{\max(CNN_{raw}) - \min(CNN_{raw}) + 1e-8}$$
4.  These normalized scores are passed to the **Curriculum Strategy** to select the 100 maps.

### C. The Hybrid Approach (`get_learnability_set_hybrid`)
This is the most advanced method. It runs *both* the empirical rollouts and the CNN scoring, and combines them mathematically to achieve the "best of both worlds". It defines four sub-modes (`HYBRID_MODE`):

First, it calculates CNN Proximity:
$$Proximity = 1.0 - |CNN_{norm} - \mu|$$
Where $\mu$ is the current curriculum target difficulty.

1.  **Linear (`hybrid_linear`)**
    Simply adds the empirical score and the CNN proximity score together. SFL is multiplied by 4 to normalize its range ($0 \to 0.25$) to match the CNN range ($0 \to 1.0$).
    $$Score = (4.0 \times SFL) + Proximity$$

2.  **Soft Handoff (`hybrid_soft_handoff`)**
    Dynamically weights the two scores based on the *average solvability of the entire batch* ($\bar{P}$). If the batch is completely impossible ($\bar{P} \approx 0$), empirical SFL is useless (cold-start problem), so the CNN dominates. As the agent gets better and the batch becomes solvable, empirical SFL takes over.
    $$\alpha = \text{clip}\left(\frac{\bar{P}}{0.1}, 0.0, 1.0\right)$$
    $$Score = \alpha \times (4.0 \times SFL) + (1 - \alpha) \times Proximity$$

3.  **Learnability Weighted (`hybrid_learnability_weighted`)**
    The weight of the CNN is inversely proportional to the empirical SFL score on a *per-map basis*. If a map has high empirical learnability, we trust it. If it doesn't, we fall back to the CNN's structural judgment. It also includes a **Mastery Safeguard**: if a map has >90% empirical solvability, the CNN weight is hard-clamped to $0.0$ to prevent the CNN from forcing the agent to re-train on trivially mastered maps.
    $$W_{cnn} = (0.25 - SFL) \times 4.0$$
    $$Score = (4.0 \times SFL) + (W_{cnn} \times Proximity)$$

4.  **Multiplicative (`hybrid_multiplicative`)**
    Applies the CNN's curriculum target as a Gaussian probability filter over the empirical SFL scores. If a map is far from the target $\mu$, its empirical score is crushed toward zero.
    $$Filter = \exp\left(-\frac{(CNN_{norm} - \mu)^2}{2\sigma^2}\right)$$
    $$Score = (SFL + \epsilon) \times Filter$$

---

## 4. The Curriculum Strategies (`select_environments`)

When using the CNN or Hybrid methods, the curriculum strategy determines how the target difficulty ($\mu$) shifts and how maps are selected relative to it.

### Shifting the Target ($\mu$)
*   **Time Based**: $\mu$ simply increments linearly from `0.05` to `1.0` as the training epochs progress. It ignores the agent's actual skill.
*   **Performance Adaptive**: $\mu$ reacts to the agent's recent empirical success rate (calculated in the outer loop over the last 50 updates).
    *   If `recent_success > 0.8`, $\mu \mathrel{{+}{=}} Step Size$ (Make it harder).
    *   If `recent_success < 0.2`, $\mu \mathrel{{-}{=}} Step Size$ (Make it easier).

### Selecting the Maps
Once $\mu$ is established, the strategy picks the 100 maps:
*   **Standard Target Selection**: Used by both `time_based` and `performance_adaptive`. It strictly sorts the maps by absolute distance to $\mu$ and takes the closest 100.
    $$Distance = |CNN_{norm} - \mu|$$
*   **Gaussian Frontier**: Instead of strictly taking the closest maps, it builds a Gaussian probability distribution centered at $\mu$ with variance $\sigma^2$ (`CURRICULUM_GAUSSIAN_VAR`). It samples maps probabilistically. This allows some maps slightly further away from $\mu$ to occasionally be selected, maintaining environmental diversity and preventing catastrophic forgetting.

---

## 5. Weights & Biases Observability

The script uses a complex image rendering buffer (`log_buffer`) to push visual grids of the generated maps to W&B every `EVAL_FREQ` steps.

*   `best_maps`: The final 100 maps successfully selected by the active algorithm. **This is what the agent actually trains on.**
*   `worst_maps`: The 20 absolute lowest scoring maps across the *entire 5000 batch* (Global worst).
*   `unsolvable_maps`: The 20 maps with the absolute lowest empirical success rate across the *entire 5000 batch*. (Only available in Standard and Hybrid modes).
*   `highest_in_curriculum`: Out of the 100 maps selected in `best_maps`, these are the 20 with the highest raw learnability score. Shows the curriculum's upper bound.
*   `lowest_in_curriculum`: Out of the 100 maps selected in `best_maps`, these are the 20 with the lowest raw learnability score. Shows the curriculum's lower bound.

By comparing these grids, you gain perfect observability into the mathematical distribution and effectiveness of your active UED algorithm.
