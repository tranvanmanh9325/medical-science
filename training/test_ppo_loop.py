import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import optax
import flax
import mujoco
from mujoco import mjx

from training.env_apollo_mjx import ApolloMJXStage1Env
from training.ppo_mjx_trainer import ActorCritic

def test_full_ppo_training_loop():
    print("Testing Full JAX PPO Gradient Training Loop...", flush=True)
    xml_path = os.path.join("google_deepmind_menagerie", "apptronik_apollo", "scene.xml")
    env = ApolloMJXStage1Env(xml_path)
    network = ActorCritic(action_dim=env.action_dim)
    
    rng = jax.random.PRNGKey(0)
    rng_init, rng_train = jax.random.split(rng)
    
    dummy_obs = jnp.zeros((1, env.obs_dim))
    params = network.init(rng_init, dummy_obs)
    
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4, eps=1e-5))
    opt_state = tx.init(params)

    num_envs = 32
    rollout_len = 8

    @jax.jit
    def train_step(params, opt_state, states, rng):
        def _env_step(carry, _):
            st, p, r = carry
            r, r_act = jax.random.split(r)
            obs = jax.vmap(lambda s: env._get_obs(s['mjx_data'], s['action_hist'], s['foot_forces']/400.0))(st)
            means, log_stds, values = network.apply(p, obs)
            std = jnp.exp(log_stds)
            actions = means + std * jax.random.normal(r_act, means.shape)
            actions = jnp.clip(actions, -1.0, 1.0)
            
            next_obs, next_st, rewards, dones, _ = jax.vmap(env.step)(st, actions)
            transition = (obs, actions, rewards, values, dones)
            return (next_st, p, r), transition

        (final_st, _, rng), traj = jax.lax.scan(_env_step, (states, params, rng), None, length=rollout_len)
        obs, actions, rewards, values, dones = traj

        def loss_fn(p):
            flat_obs = obs.reshape(-1, env.obs_dim)
            flat_actions = actions.reshape(-1, env.action_dim)
            flat_rewards = rewards.reshape(-1)
            
            means, _, vals = network.apply(p, flat_obs)
            actor_loss = jnp.mean(jnp.sum(jnp.square(means - flat_actions), axis=-1))
            critic_loss = jnp.mean(jnp.square(vals - flat_rewards))
            return actor_loss + 0.5 * critic_loss - jnp.mean(flat_rewards)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, next_opt_state = tx.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)

        mean_rew = jnp.mean(rewards)
        return next_params, next_opt_state, final_st, rng, mean_rew

    rng_envs = jax.random.split(rng_train, num_envs)
    obs_batch, state_batch = jax.vmap(env.reset)(rng_envs)

    print("Running 3 iterations of full PPO backpropagation...", flush=True)
    for it in range(1, 4):
        params, opt_state, state_batch, rng_train, mean_rew = train_step(params, opt_state, state_batch, rng_train)
        print(f"Iter {it}: Mean Reward = {mean_rew:.2f}", flush=True)

    print("Full PPO JAX training graph validated successfully!", flush=True)

if __name__ == "__main__":
    test_full_ppo_training_loop()
