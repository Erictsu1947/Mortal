import prelude

import logging
import socket
import torch
import numpy as np
import time
import gc
from os import path
from model import Brain, DQN
from player import TrainPlayer
from common import send_msg, recv_msg
from config import config


def save_stats(index):
    from torch.utils.tensorboard import SummaryWriter
    from libriichi.stat import Stat
    steps = 100 * index

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stat = Stat.from_dir(path.abspath( './train_model/drain'), 'trainee')
    avg_pt = stat.avg_pt([90, 45, 0, -135])  # for display only, never used in training

    writer.add_scalar('train_play/avg_ranking', stat.avg_rank, steps)
    writer.add_scalar('train_play/avg_pt', avg_pt, steps)
    writer.add_scalars('train_play/ranking', {
        '1st': stat.rank_1_rate,
        '2nd': stat.rank_2_rate,
        '3rd': stat.rank_3_rate,
        '4th': stat.rank_4_rate,
    }, steps)
    writer.add_scalars('train_play/behavior', {
        'agari': stat.agari_rate,
        'houjuu': stat.houjuu_rate,
        'fuuro': stat.fuuro_rate,
        'riichi': stat.riichi_rate,
    }, steps)
    writer.add_scalars('train_play/agari_point', {
        'overall': stat.avg_point_per_agari,
        'riichi': stat.avg_point_per_riichi_agari,
        'fuuro': stat.avg_point_per_fuuro_agari,
        'dama': stat.avg_point_per_dama_agari,
    }, steps)
    writer.add_scalar('train_play/houjuu_point', stat.avg_point_per_houjuu, steps)
    writer.add_scalar('train_play/point_per_round', stat.avg_point_per_round, steps)
    writer.add_scalars('train_play/key_step', {
        'agari_jun': stat.avg_agari_jun,
        'houjuu_jun': stat.avg_houjuu_jun,
        'riichi_jun': stat.avg_riichi_jun,
    }, steps)
    writer.add_scalars('train_play/riichi', {
        'agari_after_riichi': stat.agari_rate_after_riichi,
        'houjuu_after_riichi': stat.houjuu_rate_after_riichi,
        'chasing_riichi': stat.chasing_riichi_rate,
        'riichi_chased': stat.riichi_chased_rate,
    }, steps)
    writer.add_scalar('train_play/riichi_point', stat.avg_riichi_point, steps)
    writer.add_scalars('train_play/fuuro', {
        'agari_after_fuuro': stat.agari_rate_after_fuuro,
        'houjuu_after_fuuro': stat.houjuu_rate_after_fuuro,
    }, steps)
    writer.add_scalar('train_play/fuuro_num', stat.avg_fuuro_num, steps)
    writer.add_scalar('train_play/fuuro_point', stat.avg_fuuro_point, steps)
    writer.flush()



def main():
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    device = torch.device(config['control']['device'])
    version = config['control']['version']
    num_blocks = config['resnet']['num_blocks']
    conv_channels = config['resnet']['conv_channels']
    oracle = None
    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels).to(device).eval()
    dqn = DQN(version=version).to(device)
    train_player = TrainPlayer()
    param_version = -1

    pts = np.array([90, 45, 0, -135])
    history_window = config['online']['history_window']
    history = []

    index = 0
    while True:
        while True:
            with socket.socket() as conn:
                conn.connect(remote)
                msg = {
                    'type': 'get_param',
                    'param_version': param_version,
                }
                send_msg(conn, msg)
                rsp = recv_msg(conn, map_location=device)
                if rsp['status'] == 'ok':
                    param_version = rsp['param_version']
                    break
                time.sleep(3)
        mortal.load_state_dict(rsp['mortal'])
        dqn.load_state_dict(rsp['dqn'])
        logging.info('param has been updated')

        rankings, file_list = train_player.train_play(oracle, mortal, dqn, device)
        avg_rank = (rankings * np.arange(1, 5)).sum() / rankings.sum()
        avg_pt = (rankings * pts).sum() / rankings.sum()

        history.append(np.array(rankings))
        if len(history) > history_window:
            del history[0]
        sum_rankings = np.sum(history, axis=0)
        ma_avg_rank = (sum_rankings * np.arange(1, 5)).sum() / sum_rankings.sum()
        ma_avg_pt = (sum_rankings * pts).sum() / sum_rankings.sum()

        index += 1
        save_stats(index)
        logging.info(f'trainee rankings: {rankings} ({avg_rank:.6}, {avg_pt:.6}pt)')
        logging.info(f'last {len(history)} sessions: {sum_rankings} ({ma_avg_rank:.6}, {ma_avg_pt:.6}pt)')

        logs = {}
        for filename in file_list:
            with open(filename, 'rb') as f:
                logs[path.basename(filename)] = f.read()

        with socket.socket() as conn:
            conn.connect(remote)
            send_msg(conn, {
                'type': 'submit_replay',
                'logs': logs,
                'param_version': param_version,
            })
            logging.info('logs have been submitted')
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
