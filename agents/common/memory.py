from collections.__init__ import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminal', 'info'))