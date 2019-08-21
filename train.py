from datetime import timedelta
from timeit import default_timer as timer

import argparse 
import copy
import json
import math
import numpy as np
import os
import random
import shutil
import torch

from agents.agent import AgentDQN
from dialog_system.dialog_manager import DialogManager
from usersim.usersim_test import TestRuleSimulator      
from usersim.usersim_rule import RuleSimulator
from utils.utils import *
from tensorboardX import SummaryWriter

import dialog_config

writer = SummaryWriter()
parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', dest='data_folder', type=str, default='ad_data', help='folder to all data')
parser.add_argument('--max_turn', dest='max_turn', default=22, type=int, help='maximum ength of each dialog (default=20, 0=no maximum length)')
parser.add_argument('--meta_train_epochs', dest='meta_train_epochs', default=1, type=int, help='Total number of meta train epochs to run (default=1)')
parser.add_argument('--meta_test_epochs', dest='meta_test_epochs', default=1, type=int, help='Total number of meta test epochs to run (default=1)')
parser.add_argument('--slot_err_prob', dest='slot_err_prob', default=0.05, type=float, help='the slot err probability')
parser.add_argument('--slot_err_mode', dest='slot_err_mode', default=0, type=int, help='slot_err_mode: 0 for slot_val only; 1 for three errs')
parser.add_argument('--intent_err_prob', dest='intent_err_prob', default=0.05, type=float, help='the intent err probability')
parser.add_argument('--agt', dest='agt', default=0, type=int, help='Select an agent: 0 for a command line input, 1-6 for rule based agents')
parser.add_argument('--usr', dest='usr', default=0, type=int, help='Select a user simulator. 0 is a Frozen user simulator.')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.1, help='Epsilon to determine stochasticity of epsilon-greedy agent policies')
parser.add_argument('--priority_replay', dest='priority_replay', default=0, type=int, help='')
parser.add_argument('--fix_buffer', dest='fix_buffer', default=0, type=int, help='')
parser.add_argument('--origin_model', dest='origin_model', default=0, type=int, help='0 for not mask')
# load NLG & NLU model
parser.add_argument('--nlg_model_path', dest='nlg_model_path', type=str, default='./deep_dialog/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p', help='path to model file')
parser.add_argument('--nlu_model_path', dest='nlu_model_path', type=str, default='./deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.p', help='path to the NLU model file')
parser.add_argument('--act_level', dest='act_level', type=int, default=0, help='0 for dia_act level; 1 for NL level')
parser.add_argument('--run_mode', dest='run_mode', type=int, default=0, help='run_mode: 0 for default NL; 1 for dia_act; 2 for both')
parser.add_argument('--auto_suggest', dest='auto_suggest', type=int, default=0, help='0 for no auto_suggest; 1 for auto_suggest')
parser.add_argument('--cmd_input_mode', dest='cmd_input_mode', type=int, default=0, help='run_mode: 0 for NL; 1 for dia_act')

# RL agent parameters
parser.add_argument('--experience_replay_size', dest='experience_replay_size', type=int, default=1000, help='the size for experience replay')
parser.add_argument('--dqn_hidden_size', dest='dqn_hidden_size', type=int, default=60, help='the hidden size for DQN')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='batch size')
parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='lr for DQN')
parser.add_argument('--gamma', dest='gamma', type=float, default=0.9, help='gamma for DQN')
parser.add_argument('--predict_mode', dest='predict_mode', type=bool, default=False, help='predict model for DQN')
parser.add_argument('--num_simulation_episodes', dest='num_simulation_episodes', type=int, default=50, help='the size of validation set')
parser.add_argument('--target_net_update_freq', dest='target_net_update_freq', type=int, default=1, help='update frequency')
parser.add_argument('--warm_start', dest='warm_start', type=int, default=1, help='0: no warm start; 1: warm start for training')
parser.add_argument('--num_warm_start_episodes', dest='num_warm_start_episodes', type=int, default=100, help='the number of episodes for warm start')
parser.add_argument('--supervise', dest='supervise', type=int, default=1, help='0: no supervise; 1: supervise for training')
parser.add_argument('--supervise_episodes', dest='supervise_episodes', type=int, default=100, help='the number of episodes for supervise')

parser.add_argument('--trained_model_path', dest='trained_model_path', type=str, default=None, help='the path for trained model')
parser.add_argument('-o', '--write_model_dir', dest='write_model_dir', type=str, default='./deep_dialog/checkpoints/', help='write model to disk')
parser.add_argument('--save_check_point', dest='save_check_point', type=int, default=10, help='number of epochs for saving model')
parser.add_argument('--success_rate_threshold', dest='success_rate_threshold', type=float, default=0.3, help='the threshold for success rate')
parser.add_argument('--learning_phase', dest='learning_phase', default='train', type=str, help='train/test; default is all')
parser.add_argument('--train_set', dest='train_set', default='all', type=str, help='train/test/all; default is all')
parser.add_argument('--test_set', dest='test_set', default='all', type=str, help='train/test/all; default is all')

parser.add_argument('--meta_batch_size', dest='meta_batch_size', default=20, type=int, help='number of tasks per meta training batch')
parser.add_argument('--support_epoch', dest='support_epoch', default=20, type=int, help='number of epochs to update models with support set before collecting gradients')
parser.add_argument('--learned_domain_prob', dest='learned_domain_prob', default=0.5, type=int, help='probability of using learned domains to generate training episodes to avoid catastrophic forgetting')

args = parser.parse_args()
params = vars(args)

print('Dialog Parameters: ')
print(json.dumps(params, indent=2))

data_folder = params['data_folder']

goal_set = load_pickle('{}/goal_dict_original.p'.format(data_folder))
goal_set_tag = 'all'
act_set = text_to_dict('{}/dia_acts.txt'.format(data_folder))  # all acts
slot_set = text_to_dict('{}/slot_set.txt'.format(data_folder))  # all slots with symptoms + all disease

sym_dict = text_to_dict('{}/symptoms.txt'.format(data_folder))  # all symptoms
dise_dict = text_to_dict('{}/diseases.txt'.format(data_folder))  # all diseases
req_dise_sym_dict = load_pickle('{}/req_dise_sym_dict.p'.format(data_folder))
dise_sym_num_dict = load_pickle('{}/dise_sym_num_dict.p'.format(data_folder))
dise_sym_pro = np.loadtxt('{}/dise_sym_pro.txt'.format(data_folder))
sym_dise_pro = np.loadtxt('{}/sym_dise_pro.txt'.format(data_folder))
sp = np.loadtxt('{}/sym_prio.txt'.format(data_folder))
tran_mat = np.loadtxt('{}/action_mat.txt'.format(data_folder))
learning_phase = params['learning_phase']
train_set = params['train_set']
test_set = params['test_set']
fix_buffer = False
if params['fix_buffer'] == 1:
    fix_buffer = True
priority_replay = False
if params['priority_replay'] == 1:
    priority_replay = True
max_turn = params['max_turn']
meta_train_epochs = params['meta_train_epochs']
meta_test_epochs = params['meta_test_epochs']

agt = params['agt']
usr = params['usr']

meta_batch_size = params['meta_batch_size']
support_epoch = params['support_epoch']

dialog_config.run_mode = params['run_mode']
dialog_config.auto_suggest = params['auto_suggest']

################################################################################
#   Parameters for Agents
################################################################################
agent_params = {}
agent_params['max_turn'] = max_turn
agent_params['epsilon'] = params['epsilon']
agent_params['agent_run_mode'] = params['run_mode']
agent_params['agent_act_level'] = params['act_level']
agent_params['experience_replay_size'] = params['experience_replay_size']
agent_params['dqn_hidden_size'] = params['dqn_hidden_size']
agent_params['batch_size'] = params['batch_size']
agent_params['gamma'] = params['gamma']
agent_params['lr'] = params['lr']
agent_params['predict_mode'] = params['predict_mode']
agent_params['trained_model_path'] = params['trained_model_path']
agent_params['warm_start'] = params['warm_start']
agent_params['supervise'] = params['supervise']
agent_params['cmd_input_mode'] = params['cmd_input_mode']
agent_params['fix_buffer'] = fix_buffer
agent_params['priority_replay'] = priority_replay
agent_params['target_net_update_freq'] = params['target_net_update_freq']
agent_params['origin_model'] = params['origin_model']

meta_agent = AgentDQN(sym_dict, dise_dict, req_dise_sym_dict, dise_sym_num_dict, tran_mat, dise_sym_pro, sym_dise_pro, sp, act_set, slot_set, agent_params, static_policy=True)
agents = [AgentDQN(sym_dict, dise_dict, req_dise_sym_dict, dise_sym_num_dict, tran_mat, dise_sym_pro, sym_dise_pro, sp, act_set, slot_set, agent_params, static_policy=True) for _ in range(meta_batch_size)]
test_agents = [AgentDQN(sym_dict, dise_dict, req_dise_sym_dict, dise_sym_num_dict, tran_mat, dise_sym_pro, sym_dise_pro, sp, act_set, slot_set, agent_params, static_policy=True) for _ in range(meta_batch_size)]

################################################################################
#   Parameters for User Simulators
################################################################################
usersim_params = {}
usersim_params['max_turn'] = max_turn
usersim_params['slot_err_probability'] = params['slot_err_prob']
usersim_params['slot_err_mode'] = params['slot_err_mode']
usersim_params['intent_err_probability'] = params['intent_err_prob']
usersim_params['simulator_run_mode'] = params['run_mode']
usersim_params['simulator_act_level'] = params['act_level']
usersim_params['data_split'] = params['learning_phase']

user_sim = RuleSimulator(sym_dict, act_set, slot_set, goal_set, usersim_params)
test_user_sim = TestRuleSimulator(sym_dict, act_set, slot_set, goal_set, usersim_params)
################################################################################
# Dialog Manager
################################################################################
dm_params = {}
dialog_managers = [DialogManager(agent, user_sim, act_set, slot_set, dm_params) for agent in agents]
# TODO: test_dialog_managers = [DialogManager(test_agents, test_user_sim, act_set, slot_set, dm_params) for test_agent in test_agents]
# TODO: need to change set_dm_with_goals when simulation
test_dialog_managers = [DialogManager(test_agents, user_sim, act_set, slot_set, dm_params) for test_agent in test_agents]
meta_dialog_manager = DialogManager(meta_agent, user_sim, act_set, slot_set, dm_params)
meta_test_dialog_manager = DialogManager(meta_agent, test_user_sim, act_set, slot_set, dm_params)
status = {'successes': 0, 'count': 0, 'cumulative_reward': 0}

num_simulation_episodes = params['num_simulation_episodes']
batch_size = params['batch_size']  # default = 16
warm_start = params['warm_start']
num_warm_start_episodes = params['num_warm_start_episodes']
supervise = params['supervise']
supervise_episodes = params['supervise_episodes']
success_rate_threshold = params['success_rate_threshold']
save_check_point = params['save_check_point']

best_model, best_res, best_te_model, best_te_res = [], [], [], []
def initialize_metric_collector(num_tasks):
    """ Best Model and Performance Records """
    global best_model, best_res, best_te_model, best_te_res
    best_model = [{'model': agent.model.state_dict()} for _ in range(num_tasks+1)]
    best_res = [{'success_rate': 0, 'ave_reward': float('-inf'), 'ave_turns': float('inf'), 'epoch': 0} for _ in range(num_tasks+1)]

    best_te_model = [{'model': agent.model.state_dict()} for _ in range(num_tasks+1)]
    best_te_res = [{'success_rate': 0, 'ave_reward': float('-inf'), 'ave_turns': float('inf'), 'epoch': 0} for _ in range(num_tasks+1)]

run_mode = params['run_mode']
output = False
if run_mode < 3: output = True

episode_reward = 0

""" Save model """


def save_model(path, agt, agent, cur_epoch, best_epoch=0, best_success_rate=0.0, best_ave_turns=0.0, tr_success_rate=0.0, te_success_rate=0.0, phase="", is_checkpoint=False):
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint = {}
    checkpoint['cur_epoch'] = cur_epoch
    checkpoint['state_dict'] = agent.model.state_dict()
    if is_checkpoint:
        file_name = 'checkpoint.pth.tar'
        checkpoint['eval_success'] = tr_success_rate
        checkpoint['test_success'] = te_success_rate
    else:
        file_name = 'agt_%s_%s_%s_%s_%.3f_%.3f.pth.tar' % (agt, phase, best_epoch, cur_epoch, best_success_rate, best_ave_turns)
        checkpoint['best_success_rate'] = best_success_rate
        checkpoint['best_epoch'] = best_epoch
    file_path = os.path.join(path, file_name)
    torch.save(checkpoint, file_path)


def simulation_episodes(num_episodes, goals, dialog_manager, epoch, output=False):
    """
    simulate dialog for num_episodes episodes, and return evaluation (avg success rate, avg reward) 
    """
    if epoch < 100:
        return warm_start_simulation(num_episodes, goals, dialog_manager)

    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    res = {}

    set_dm_with_goals(goals, True)

    for episode in range(num_episodes):
        dialog_manager.initialize_episode()
        episode_over = False
        while not episode_over:
            episode_over, r, dialog_status, hit_rate = dialog_manager.next_turn()
            cumulative_reward += r
            if episode_over:
                if dialog_status == dialog_config.SUCCESS_DIALOG:
                    successes += 1
                    if output: print("simulation episode %s: Success" % episode)
                else:
                    if output: print("simulation episode %s: Fail" % episode)
                cumulative_turns += dialog_manager.state_tracker.turn_count
    res['success_rate'] = float(successes) / num_episodes
    res['ave_reward'] = float(cumulative_reward) / num_episodes
    res['ave_turns'] = float(cumulative_turns) / num_episodes
    print("simulation success rate %s, ave reward %s, ave turns %s" % (res['success_rate'], res['ave_reward'], res['ave_turns']))
    return res

'''
def eval(simu_size, data_split, out=False):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    user_sim.data_split = data_split
    res = {}
    avg_hit_rate = 0.0
    for episode in range(simu_size):
        dialog_manager.initialize_episode()
        episode_over = False
        episode_hit_rate = 0
        while not episode_over:
            episode_over, r, dialog_status, hit_rate = dialog_manager.next_turn()
            cumulative_reward += r
            if episode_over:
                # if reward > 0:
                episode_hit_rate += hit_rate
                if dialog_status == dialog_config.SUCCESS_DIALOG:
                    successes += 1
                    if out: print("%s simulation episode %s: Success" % (data_split, episode))
                else:
                    if out: print("%s simulation episode %s: Fail" % (data_split, episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count
                episode_hit_rate /= dialog_manager.state_tracker.turn_count
                avg_hit_rate += episode_hit_rate
    res['success_rate'] = float(successes) / simu_size
    res['ave_reward'] = float(cumulative_reward) / simu_size
    res['ave_turns'] = float(cumulative_turns) / simu_size
    avg_hit_rate = avg_hit_rate / simu_size
    print("%s hit rate %.4f, success rate %s, ave reward %s, ave turns %s" % (data_split, avg_hit_rate, res['success_rate'], res['ave_reward'], res['ave_turns']))
    return res
'''

def test(goals, data_split_tag, test_dialog_manager, out=False):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    res = {}
    avg_hit_rate = 0.0
    test_dialog_manager.agent.epsilon = 0
    num_episodes = len(goals)

    # no replacement when generating episodes from goals
    set_dm_with_goals(goals, False)

    for episode in range(num_episodes):
        test_dialog_manager.initialize_episode()
        episode_over = False
        episode_hit_rate = 0
        #print(len(test_dialog_manager.user.left_goal))
        while not episode_over:
            # TODO: add record_training_data=Fasle to next turn in pretrain experiment branch too
            episode_over, r, dialog_status, hit_rate = test_dialog_manager.next_turn(record_training_data=False)
            cumulative_reward += r
            if episode_over:
                # if reward > 0:
                episode_hit_rate += hit_rate
                if dialog_status == dialog_config.SUCCESS_DIALOG:
                    successes += 1
                    if out: print("%s simulation episode %s: Success" % (data_split_tag, episode))
                else:
                    if out: print("%s simulation episode %s: Fail" % (data_split_tag, episode))
                cumulative_turns += test_dialog_manager.state_tracker.turn_count
                episode_hit_rate /= test_dialog_manager.state_tracker.turn_count
                avg_hit_rate += episode_hit_rate
    res['success_rate'] = float(successes) / float(num_episodes)
    res['ave_reward'] = float(cumulative_reward) / float(num_episodes)
    res['ave_turns'] = float(cumulative_turns) / float(num_episodes)
    avg_hit_rate = avg_hit_rate / num_episodes
    print("%s hit rate %.4f, success rate %.4f, ave reward %.4f, ave turns %.4f" % (data_split_tag, avg_hit_rate, res['success_rate'], res['ave_reward'], res['ave_turns']))
    test_dialog_manager.agent.epsilon = params['epsilon']
    
    return res

def warm_start_simulation(num_episodes, goals, dialog_manager):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    res = {}
    warm_start_run_episodes = 0
    
    set_dm_with_goals(goals, True)

    dialog_manager.agent.warm_start = 1
    for episode in range(num_episodes):
        dialog_manager.initialize_episode()
        episode_over = False
        while not episode_over:
            episode_over, r, dialog_status, hit_rate = dialog_manager.next_turn()
            cumulative_reward += r
            if episode_over:
                # if reward > 0:
                if dialog_status == dialog_config.SUCCESS_DIALOG:
                    successes += 1
                # print ("warm_start simulation episode %s: Success" % episode)
                # else: print ("warm_start simulation episode %s: Fail" % episode)
                cumulative_turns += dialog_manager.state_tracker.turn_count
        warm_start_run_episodes += 1
        if len(dialog_manager.agent.memory) >= dialog_manager.agent.experience_replay_size:
            # TODO: clear part of the memory???
            break
    dialog_manager.agent.warm_start = 2
    res['success_rate'] = float(successes) / warm_start_run_episodes
    res['ave_reward'] = float(cumulative_reward) / warm_start_run_episodes
    res['ave_turns'] = float(cumulative_turns) / warm_start_run_episodes
    print("Warm_Start %s episodes, success rate %s, ave reward %s, ave turns %s" % (episode + 1, res['success_rate'], res['ave_reward'], res['ave_turns']))
    print("Current experience replay buffer size %s" % (len(dialog_manager.agent.memory)))


def set_dm_with_goals(goals, is_resample):
    # all the (test_)dialog_managers share the same (test_)user_sim
    if is_resample:
        user_sim.start_set = {goal_set_tag: copy.deepcopy(goals)}
        user_sim.data_split = goal_set_tag
    else:
        test_user_sim.left_goal = copy.deepcopy(goals)
        test_user_sim.data_split = goal_set_tag


def get_goals_by_diseases(diseases, tag=goal_set_tag):
    return [goal for goal in goal_set[tag] if goal['disease_tag'] in diseases]


def sample_from_list(elements, sample_size, sampled_elements, remaining_elements):
    sampled_index = set(random.sample(range(len(elements)), sample_size))
    for index, element in enumerate(elements):
        if index in sampled_index:
            sampled_elements.append(element)
        else:
            remaining_elements.append(element)


def sample_goals(goals, split_sizes, split_goal_sets):
    remaining_goals = goals
    for split_set_index, split_size in enumerate(split_sizes):
        new_remaining_goals = []
        if split_set_index == len(split_sizes)-1:
            not_sampled_container = split_goal_sets[split_set_index+1]
        else:
            not_sampled_container = new_remaining_goals

        sample_from_list(remaining_goals, split_size, split_goal_sets[split_set_index], not_sampled_container)

        remaining_goals = new_remaining_goals


def split_goals(diseases, split_sizes, tag=goal_set_tag):
    split_goal_sets = [[] for _ in range(len(split_sizes)+1)]

    for disease in diseases:
        goals = get_goals_by_diseases(set([disease]), tag)
        sample_goals(goals, split_sizes, split_goal_sets)

    return split_goal_sets


def training(meta_train_epochs, meta_test_epochs, meta_train_diseases, meta_test_diseases, fine_tune_size, meta_batch_size, support_epoch, learned_domain_prob):
    meta_train_goals = get_goals_by_diseases(meta_train_diseases, tag='train')
    # pretraining ???

    start_epoch = 0
    initialize_metric_collector(meta_batch_size)

    print(params['trained_model_path'])
    if params['trained_model_path'] is not None:
        trained_file = torch.load(params['trained_model_path'])
        if 'cur_epoch' in trained_file.keys():
            start_epoch = trained_file['cur_epoch']

    # support set: training set in meta-training phase
    # query set: eval set in meta-training phase
    # support and query set comes from the same disease, remaining goals/diseases correspond to dialogs in
    # not selected diseases
    support_diseases, remaining_diseases = [], []
    support_goals, query_goals, remaining_goals = [], [], []
    meta_train_disease_list = list(meta_train_diseases)
    for _ in range(meta_batch_size):
        support_diseases.append([])
        remaining_diseases.append([])
        sample_from_list(meta_train_disease_list, len(meta_test_diseases), support_diseases[-1], remaining_diseases[-1])

        support_goal, query_goal = split_goals(support_diseases[-1], [fine_tune_size], tag='train')
        remaining_goal = get_goals_by_diseases(set(remaining_diseases[-1]), tag='train')

        support_goals.append(support_goal)
        query_goals.append(query_goal)
        remaining_goals.append(remaining_goal)
        print('qqqqq', len(support_diseases[-1]), len(remaining_diseases[-1]), len(support_goals[-1]), len(query_goals[-1]), len(remaining_goals[-1]))

    # use rule policy, and record warm start experience
    if params['trained_model_path'] is None and warm_start == 1:
        print('warm_start starting ...')

        for task_index in range(meta_batch_size):
            warm_start_simulation(num_simulation_episodes, support_goals[task_index], dialog_managers[task_index])
            warm_start_simulation(num_simulation_episodes, query_goals[task_index], test_dialog_managers[task_index])

        print('warm_start finished, start RL training ...')

    # meta training
    for epoch in range(start_epoch, meta_train_epochs):
        print("Epoch: %s" % epoch)

        # initialize meta_gradients
        meta_gradients = ???

        for task_index in range(meta_batch_size):
            # train on support set

            # TODO: copy weights from meta_dialog_manager.agent to dialog_managers[task_index].agent
            for _ in range(support_epoch):
                dialog_managers[task_index].agent.predict_mode = True
                # simulate dialogs and save experience
                # TODO: consider to use remaining goals from time to time
                simulation_episodes(num_simulation_episodes, support_goals[task_index], dialog_managers[task_index])

                # train by current experience pool
                dialog_managers[task_index].agent.train()

                dialog_managers[task_index].agent.predict_mode = False

            # TODO: make 10 as a hyperparameter, 10 -> 10 // support_epoch?
            if epoch and epoch % 10 == 0:
                dialog_managers[task_index].agent.memory.clear()

            # update weights with gradients on query set
            # TODO: copy weights from dialog_managers[task_index].agent to test_dialog_managers[task_index].agent
            # TODO: try sample with replacement when simulation for test_dialog_managers
            test_dialog_managers[task_index].agent.predict_mode = True

            # simulate dialogs and save experience
            if random.random() < learned_domain_prob:
                simulation_episodes(num_simulation_episodes, remaining_goals[task_index], test_dialog_managers[task_index])
            else:
                simulation_episodes(num_simulation_episodes, query_goals[task_index], test_dialog_managers[task_index])

            # TODO: collect gradients for meta_dialog_manager
            meta_gradients += ???

            test_dialog_managers[task_index].agent.predict_mode = False

            # TODO: make 10 as a hyperparameter, 10 -> 10 // support_epoch?, different forgetting strategy???
            if epoch and epoch % 10 == 0:
                test_dialog_managers[task_index].agent.memory.clear()

        # TODO: update meta_dialog_manager.agent with meta_gradients
        eval_res = test(meta_train_goals, 'meta_train_eval', meta_test_dialog_manager)
        # both meta_dialog_manager and meta_test_dialog_manager call meta_agent. thus we don't need to copy weight as 
        # we did for dialog_managers and test_dialog_managers
        
        writer.add_scalar('meta_train_eval/accracy', torch.tensor(eval_res['success_rate'], device=dialog_config.device), epoch)
        writer.add_scalar('meta_train_eval/ave_reward', torch.tensor(eval_res['ave_reward'], device=dialog_config.device), epoch)
        writer.add_scalar('meta_train_eval/ave_turns', torch.tensor(eval_res['ave_turns'], device=dialog_config.device), epoch)

        # use last element in best_res for performance of meta_agent, the rest elements are not used
        if eval_res['success_rate'] > best_res[-1]['success_rate']:
            best_model['model'] = meta_dialog_manager.agent.model.state_dict()
            best_res[-1]['success_rate'] = eval_res['success_rate']
            best_res[-1]['ave_reward'] = eval_res['ave_reward']
            best_res[-1]['ave_turns'] = eval_res['ave_turns']
            best_res[-1]['epoch'] = epoch
            save_model(
                params['write_model_dir'], agt, meta_dialog_manager.agent, epoch, best_epoch=best_res[-1]['epoch'],
                best_success_rate=best_res[-1]['success_rate'], best_ave_turns=best_res[-1]['ave_turns'], phase="eval")
        
        save_model(params['write_model_dir'], agt, meta_dialog_manager.agent, epoch, is_checkpoint=True)  # save checkpoint each epoch

    # evaluation
    # TODO: sample multiple tasks and average the performance
    test_old_goals = get_goals_by_diseases(meta_train_diseases, tag='test')
    test_new_goal_size = len(test_old_goals) // len(meta_train_diseases)
    fine_tune_new_goals, test_new_goals, _ = split_goals(meta_test_diseases, [fine_tune_size, test_new_goal_size])

    print('qqqqq', len(test_old_goals), test_new_goal_size, len(fine_tune_new_goals), len(test_new_goals))

    initialize_metric_collector(meta_batch_size)
    num_diseases = len(meta_test_diseases) + len(meta_train_diseases)

    meta_dialog_manager.agent.memory.clear()
    warm_start_simulation(num_simulation_episodes, fine_tune_new_goals, meta_dialog_manager)

    for epoch in range(meta_test_epochs):
        print("Meta test epoch: %s" % epoch)

        meta_dialog_manager.agent.predict_mode = True

        # simulate dialogs and save experience
        if random.random() < learned_domain_prob:
            simulation_episodes(num_simulation_episodes, meta_train_goals, meta_dialog_manager)
        else:
            simulation_episodes(num_simulation_episodes, fine_tune_new_goals, meta_dialog_manager)

        # train by current experience pool
        meta_dialog_manager.agent.train()
        meta_dialog_manager.agent.predict_mode = False

        if epoch and epoch % 10 == 0:
            meta_dialog_manager.agent.memory.clear()

        test_res = test(test_new_goals, 'test_new_diseases', meta_test_dialog_manager)
        test_res = test(test_old_goals, 'test_old_diseases', meta_test_dialog_manager)
        test_res = test(test_new_goals + test_old_goals, 'test', meta_test_dialog_manager)

        writer.add_scalar('test/accracy', torch.tensor(test_res['success_rate'], device=dialog_config.device), epoch)
        writer.add_scalar('test/ave_reward', torch.tensor(test_res['ave_reward'], device=dialog_config.device), epoch)
        writer.add_scalar('test/ave_turns', torch.tensor(test_res['ave_turns'], device=dialog_config.device), epoch)

        if test_res['success_rate'] > best_te_res[-1]['success_rate']:
            best_te_model['model'] = meta_dialog_manager.agent.model.state_dict()
            best_te_res[-1]['success_rate'] = test_res['success_rate']
            best_te_res[-1]['ave_reward'] = test_res['ave_reward']
            best_te_res[-1]['ave_turns'] = test_res['ave_turns']
            best_te_res[-1]['epoch'] = epoch
            save_model(
                params['write_model_dir'], agt, meta_dialog_manager.agent, epoch, best_epoch=best_te_res[-1]['epoch'],
                best_success_rate=best_te_res[-1]['success_rate'], best_ave_turns=best_te_res[-1]['ave_turns'], phase="test")

        save_model(params['write_model_dir'], agt, meta_dialog_manager.agent, epoch, is_checkpoint=True)  # save checkpoint each epoch


meta_train_diseases = set(['小儿消化不良', '上呼吸道感染', '小儿腹泻'])
meta_test_diseases = set(['小儿支气管炎'])
fine_tune_size = 5
if agt == 9:
    training(meta_train_epochs, meta_test_epochs, meta_train_diseases, meta_test_diseases, fine_tune_size, meta_batch_size, support_epoch, learned_domain_prob)
