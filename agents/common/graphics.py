from __future__ import division, print_function

from agents.tree_search.abstract import AbstractTreeSearchAgent
from agents.tree_search.graphics import TreeGraphics, MCTSGraphics
from agents.tree_search.mcts import MCTSAgent

class AgentGraphics(object):
    """
        Graphical visualization of any Agent implementing AbstractAgent.
    """
    @classmethod
    def display(cls, agent, agent_surface, sim_surface=None):
        """
            Display an agent visualization on a pygame surface.

        :param agent: the agent to be displayed
        :param agent_surface: the pygame surface on which the agent is displayed
        :param sim_surface: the pygame surface on which the environment is displayed
        """

        if isinstance(agent, MCTSAgent):
            MCTSGraphics.display(agent, agent_surface)
        elif isinstance(agent, AbstractTreeSearchAgent):
            TreeGraphics.display(agent, agent_surface)