import random

from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, taxi_id: int):
    robo = env.get_robot(taxi_id)

    agent_to_package = min(manhattan_distance(robo.position, env.packages[0].position),
                           manhattan_distance(robo.position, env.packages[1].position))

    # missing some score for delivering the package that will make him do it
    heuristic = robo.credit * 30

    if robo.package:
        agent_to_destination = manhattan_distance(robo.position, robo.package.destination)
        heuristic -= (agent_to_destination - 10)  # go to delivery destination / pick up package
    else:
        heuristic -= agent_to_package  # go to package position
    return heuristic


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1

    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def minimax(self, env: WarehouseEnv, agent_id: int, agent_turn: int, depth: float):
        if depth == 0 or env.done():
            return self.heuristic(env, agent_id)

        operators = env.get_legal_operators(agent_turn)
        children = [env.clone() for _ in operators]

        if agent_turn == agent_id:
            curr_max = float('-inf')
            for child, op in zip(children, operators):
                child.apply_operator(agent_turn, op)
                v = self.minimax(child, agent_id, not agent_turn, depth - 1.0)
                curr_max = max(curr_max, v)
            return curr_max
        else:
            curr_min = float('inf')
            for child, op in zip(children, operators):
                child.apply_operator(agent_turn, op)
                v = self.minimax(child, agent_id, not agent_turn, depth - 1.0)
                curr_min = min(curr_min, v)
            return curr_min

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        children_heuristics = [self.minimax(child, agent_id, not agent_id, time_limit - 1.0) for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def minimax_ab(self, env: WarehouseEnv, agent_id: int, agent_turn: int, depth: float, a: float, b: float):
        if depth == 0 or env.done():
            return self.heuristic(env, agent_id)

        operators = env.get_legal_operators(agent_turn)
        children = [env.clone() for _ in operators]

        if agent_turn == agent_id:
            curr_max = float('-inf')
            for child, op in zip(children, operators):
                child.apply_operator(agent_turn, op)
                v = self.minimax_ab(child, agent_id, not agent_turn, depth - 1.0, a, b)
                curr_max = max(curr_max, v)
                a = max(v, a)
                if b <= a:
                    break
            return curr_max
        else:
            curr_min = float('inf')
            for child, op in zip(children, operators):
                child.apply_operator(agent_turn, op)
                v = self.minimax_ab(child, agent_id, not agent_turn, depth - 1.0, a, b)
                curr_min = min(curr_min, v)
                b = min(b, v)
                if b <= a:
                    break
            return curr_min

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        children_heuristics = [
            self.minimax_ab(child, agent_id, not agent_id, time_limit - 1.0, float('-inf'), float('inf')) for child in
            children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def expectimax(self, env: WarehouseEnv, agent_id: int, agent_turn: int, depth: float):
        if depth == 0 or env.done():
            return self.heuristic(env, agent_id)

        operators = env.get_legal_operators(agent_turn)
        children = [env.clone() for _ in operators]

        if agent_turn == agent_id:
            curr_max = float('-inf')
            for child, op in zip(children, operators):
                child.apply_operator(agent_turn, op)
                v = self.expectimax(child, agent_id, not agent_turn, depth - 1.0)
                curr_max = max(curr_max, v)
            return curr_max
        else:
            e_v = 0.0
            child_count = 0
            curr_min = float('inf')

            for child, op in zip(children, operators):
                child.apply_operator(agent_turn, op)
                v = self.expectimax(child, agent_id, not agent_turn, depth - 1.0)
                if env.get_charge_station_in((env.get_robot(agent_turn)).position):
                    v = 2 * v
                    child_count += 1
                child_count += 1
                e_v += v
            return e_v / child_count

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        children_heuristics = [self.expectimax(child, agent_id, not agent_id, time_limit - 1.0) for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
