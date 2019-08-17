
CUDA_VISIBLE_DEVICES=0 python train.py --agt 9 --usr 1 --max_turn 22 \
          --dqn_hidden_size 128 \
          --lr 0.01 \
          --epsilon 0.1 \
          --origin_model 1 \
          --fix_buffer 0 \
          --priority_replay 0\
          --experience_replay_size 50000 \
          --meta_train_epochs 2000 \
          --meta_test_epochs 2000\
          --num_simulation_episodes 100 \
          --target_net_update_freq 1 \
          --write_model_dir ./checkpoints/exp_models/KR-DQN/ \
          --data_folder dataset \
          --run_mode 3 \
          --act_level 0 \
          --slot_err_prob 0.05 \
          --intent_err_prob 0.05 \
          --batch_size 32 \
          --warm_start 1 \
          --num_warm_start_episodes 5000 \
          --learning_phase train \
          --train_set train \
          --test_set test

