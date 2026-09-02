import os
import time
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import NamedTuple, Sequence

class ActorCritic(nn.Module):
    """
    Actor-Critic Neural Network for Humanoid Whole-Body Control:
    - Actor: Maps observation (110-D) to 32 joint position offsets.
    - Critic: Estimates state value V(s) for GAE advantage estimation.
    """
    action_dim: int = 32
    hidden_dims: Sequence[int] = (512, 256, 128)

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # Shared feature trunk
        h = x
        for dim in self.hidden_dims:
            h = nn.Dense(dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(h)
            h = nn.LayerNorm()(h)
            h = nn.elu(h)

        # Actor branch (Mean & Log Std)
        actor_mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(h)
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))

        # Critic branch (Value function)
        v = nn.Dense(128, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(h)
        v = nn.elu(v)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(v)

        return actor_mean, log_std, jnp.squeeze(value, axis=-1)

class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    reward: jnp.ndarray
    value: jnp.ndarray
    done: jnp.ndarray

class PPOMJXTrainer:
    """
    Distributed PPO Trainer for Kaggle Dual NVIDIA T4 GPUs:
    - Massively vectorized parallel rollouts (4,096 environments).
    - Generalized Advantage Estimation (GAE).
    - Clipped PPO surrogate objective with entropy regularization.
    - Multi-device data parallelism across GPU 0 & GPU 1.
    """
    def __init__(
        self,
        env,
        num_envs: int = 4096,
        rollout_steps: int = 64,
        num_epochs: int = 4,
        num_minibatches: int = 8,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.005
    ):
        self.env = env
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        # Network & Optimizer
        self.network = ActorCritic(action_dim=self.env.action_dim)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=self.lr, eps=1e-5)
        )

    def init_params(self, rng: jax.Array):
        """Initializes network parameters."""
        dummy_obs = jnp.zeros((1, self.env.obs_dim))
        params = self.network.init(rng, dummy_obs)
        opt_state = self.optimizer.init(params)
        return params, opt_state

    def save_checkpoint(self, params, path: str):
        """Saves policy parameters to NPZ archive."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Flatten Flax params dictionary
        flat_params = flax.traverse_util.flatten_dict(params, sep='/')
        jnp.savez(path, **flat_params)
        print(f"[CHECKPOINT SAVED] {path}")
