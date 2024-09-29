import argparse
import os
import shutil
import jax
from functools import partial

from evojax.task.slimevolley import SlimeVolley
from evojax import util
import neat
import jax.numpy as jnp
from jax import random
from typing import Tuple
from typing import Union, Sequence
import numpy
import visualize


@jax.jit
def update_score_and_mask(score, reward, mask, done):
    new_score = score + reward * mask
    new_mask = mask * (1 - done.ravel())
    return new_score, new_mask

@jax.jit
def all_done(masks):
    return masks.sum() == 0

#@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def get_task_reset_keys(key: jnp.ndarray,
                        test: bool,
                        n_tests: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    key, subkey = random.split(key=key)
    if test:
        reset_keys = random.split(subkey, n_tests)
    else:
        reset_keys = random.split(subkey)
    return key, reset_keys


def eval_genomes(genomes, task, config, test=False):
    for genome_id, genome in genomes:
        print(genome_id)
        key = random.PRNGKey(seed=0)
        key, reset_keys = get_task_reset_keys(
            key, False, 100)
        genome.fitness = -float('Inf')
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        task_step_func = task.step
        task_reset_func = task.reset
        task_max_steps = task.max_steps

        task_state = task_reset_func(reset_keys)
        scores = jnp.zeros(1)
        valid_mask = jnp.ones(1)

        for i in range(task_max_steps):
            actions = net.activate(task_state.obs[0,:8])
            actions = jnp.array([actions,actions])
            task_state, reward, done = task_step_func(task_state, actions)
            scores, valid_mask = update_score_and_mask(
                scores, reward, valid_mask, done)
            if all_done(valid_mask):
                break
        score = jnp.mean(scores.ravel().reshape((-1)), axis=-1)

        score = score.tolist()
        genome.fitness = score

def run(config_file, show_res=False, checkpoint_path=None):
    max_steps = 3000
    train_task = SlimeVolley(test=False, max_steps=max_steps)
    test_task = SlimeVolley(test=True, max_steps=max_steps)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)

    # Create the population, which is the top-level object for a NEAT run.
    if checkpoint_path:
        p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    else:
        p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(100))
    winner = p.run(eval_genomes, train_task, 100)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    if show_res == True:
        # Show output of the most fit genome against training data.
        print('\nOutput:')
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        screens = []
        key = random.PRNGKey(seed=0)
        key, reset_keys = get_task_reset_keys(
            key, False, 100)
        task_state = test_task.reset(reset_keys)
        for _ in range(max_steps):
            actions = winner_net.activate(task_state.obs[0,:8])
            actions = jnp.array([actions, actions])
            task_state, reward, done = test_task.step(task_state, actions)
            screens.append(SlimeVolley.render(task_state))
            if done:
                break
        gif_file = os.path.join("./", 'slimevolley.gif')
        screens[0].save(gif_file, save_all=True, append_images=screens[1:],
                        duration=40, loop=0)
        visualize.draw_net(p.config, winner, True)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    config_path = os.path.join("./", 'config-feedforward')
    run(config_path, show_res=True)
