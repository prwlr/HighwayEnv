"""
Usage:
  experiments evaluate <environment> <agent> (--train|--test) [options]
  experiments benchmark <benchmark> (--train|--test) [options]
  experiments -h | --help

Options:
  -h --help              Show this screen.
  --episodes <count>     Number of episodes [default: 5].
  --no-display           Disable environment, agent, and rewards rendering.
  --name-from-config     Name the output folder from the corresponding config files
  --processes <count>    Number of running processes [default: 4].
  --recover              Load model from the latest checkpoint.
  --recover-from <file>  Load model from a given checkpoint.
  --seed <str>           Seed the environments and agents.
  --train                Train the agent.
  --test                 Test the agent.
  --verbose              Set log level to debug instead of info.
  --repeat <times>       Repeat several times [default: 1].
"""
import sys
sys.path.insert(1, '/home/marha066/Projects/TSFS12-Project/HighwayEnv')

import datetime
import os
from pathlib import Path
import gymnasium as gym
from docopt import docopt

import trainer.logger as logger
from trainer.evaluation import Evaluation
from agents.common.factory import load_agent, load_environment


VERBOSE_CONFIG = 'configs/verbose.json'
LOGGING_CONFIG = 'configs/logging.json'


def main():
    opts = docopt(__doc__)
    if opts['evaluate']:
        for _ in range(int(opts['--repeat'])):
            evaluate(opts['<environment>'], opts['<agent>'], opts)


def evaluate(environment_config, agent_config, options):
    """
        Evaluate an agent interacting with an environment.

    :param environment_config: the path of the environment configuration file
    :param agent_config: the path of the agent configuration file
    :param options: the evaluation options
    """
    logger.configure(LOGGING_CONFIG)
    if options['--verbose']:
        logger.configure(VERBOSE_CONFIG)
    env = load_environment(environment_config)
    agent = load_agent(agent_config, env)
    run_directory = None
    if options['--name-from-config']:
        run_directory = "{}_{}_{}".format(Path(agent_config).with_suffix('').name,
                                  datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
                                  os.getpid())
    options['--seed'] = int(options['--seed']) if options['--seed'] is not None else None
    evaluation = Evaluation(env,
                            agent,
                            run_directory=run_directory,
                            num_episodes=int(options['--episodes']),
                            sim_seed=options['--seed'],
                            recover=options['--recover'] or options['--recover-from'],
                            display_env=not options['--no-display'],
                            display_agent=not options['--no-display'],
                            display_rewards=not options['--no-display'])
    if options['--train']:
        evaluation.train()
    elif options['--test']:
        evaluation.test()
    else:
        evaluation.close()
    return os.path.relpath(evaluation.run_directory)
    


if __name__ == '__main__':
    main()
