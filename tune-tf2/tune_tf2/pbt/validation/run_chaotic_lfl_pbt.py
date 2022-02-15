import sys, os
import shutil
# add the PBT package path
sys.path.insert(0, '/snel/home/asedler/core/pbt-hp-opt/pbt_opt')
from server import Server
from lfads_wrapper.run_posterior_mean_sampling import run_posterior_sample_and_average
#from lfads_wrapper.run_posterior_mean_sampling import run_train


''' ------------------------ Set input and output dirs ------------------------ '''
name = 'pbt_run_chaotic_10_reps'
run_save_path = '/snel/home/asedler/ray_results/pbt_validation_reconcileDropoutAndClipping/chaotic/lfl/chaotic_10_replications/sample_0'     # where PBT will store the runs
data_dir = '/snel/share/data/lf2_pbt_validation/chaotic/chaotic_10_replications/sample_0'      # lfads data folder
lfads_output_dir = os.path.join(run_save_path, 'best_worker')  # folder to copy the best worker after PBT

# Used for compatibility with run manager. re-write values with CL arguments if passed
if len(sys.argv) > 4 :
    name = sys.argv[1]
    run_save_path = sys.argv[2]  # where PBT will store the runs
    data_dir = sys.argv[3]  # lfads data folder
    lfads_output_dir = sys.argv[4]  # folder to copy the best worker after PBT

''' ------------------------ Set machine information ------------------------ '''
computers = [
    {'id': 'sulcus',
     'ip': 'sulcus.bme.emory.edu',
     'max_processes': 16,
     'process_start_cmd': '/snel/home/asedler/core/pbt-hp-opt/pbt_opt/run_lfads_client.sh',
     'wait_for_process_start':False,
     },
    # {'id': 'cortex',
    #  'ip': 'cortex.bme.emory.edu',
    #  'max_processes': 6,
    #  'process_start_cmd': '/snel/home/mreza/projects/PBT_HP_opt/PBT_HP_opt/pbt_opt/run_lfads_client.sh',
    #  'wait_for_process_start':False,
    #  },
    # {'id': 'striate',
    # 'ip': 'striate.bme.emory.edu',
    # 'max_processes': 4,
    # 'process_start_cmd': '/snel/home/mreza/projects/PBT_HP_opt/PBT_HP_opt/pbt_opt/run_lfads_client.sh',
    # 'wait_for_process_start':False,
    # },
    # {'id': 'spike',
    # 'ip': 'spike.bme.emory.edu',
    # 'max_processes': 12,
    # 'process_start_cmd': '/snel/home/mreza/projects/PBT_HP_opt/PBT_HP_opt/pbt_opt/run_lfads_client.sh',
    # 'wait_for_process_start':False,
    # },
    # {'id': 'axon',
    # 'ip': 'axon.bme.emory.edu',
    # 'max_processes': 0,
    # 'process_start_cmd': '/snel/home/mreza/projects/PBT_HP_opt/PBT_HP_opt/pbt_opt/run_lfads_client.sh',
    # 'wait_for_process_start':False,
    # },
    # {'id': 'synapse',
    # 'ip': 'synapse.bme.emory.edu',
    # 'max_processes': 12,
    # 'process_start_cmd': '/snel/home/mreza/projects/PBT_HP_opt/PBT_HP_opt/pbt_opt/run_lfads_client.sh',
    # 'wait_for_process_start':False,
    # },
    #{'id': 'poisson',
    # 'ip': 'poisson.bme.emory.edu',
    # 'max_processes': 3,
    # 'process_start_cmd': '/snel/home/mreza/projects/PBT_HP_opt/PBT_HP_opt/pbt_opt/run_lfads_client.sh',
    # 'wait_for_process_start': False,
    #},

]

# example if you want to use resample and specify the probability of resampling per hyperparam
#resample_param = {'sample_mode':'rand',
#                  'sample_prob':0.5}

''' ------------------------ Create server object ------------------------ '''
# put the pbt run in the pbt_run folder
run_save_path = os.path.join(run_save_path, 'pbt_run')
svr = Server(name, run_save_path, num_workers=16, epochs_per_generation=50,
             max_generations=50, exploit_method='binarytournament', exploit_param=[],
             explore_method='perturb', explore_param=0.8, mode='minimize', force_overwrite=True,
             mongo_server_ip='localhost', port=27017, server_log_path=run_save_path, num_no_best_to_stop=4)

# add machines to be used by PBT
svr.add_computers(computers)

''' ------------------------ Specify model parameters ------------------------ '''
''' Searchable parameters (explorable=True) '''
# Learning rate
svr.add_hp('learning_rate_init', (0.00001, 0.005), init_sample_mode=[0.004],
           explore_method='perturb', explore_param=0.3, limit_explore=True, explorable=True)

''' Regularization '''
# Standard Dropout
svr.add_hp('keep_prob', (0.4, 1.0), init_sample_mode='rand',
           explore_method='perturb', explore_param=0.3, limit_explore=True, explorable=True)

# Coordinated Dropout
svr.add_hp('keep_ratio', (0.3, 0.99), init_sample_mode=[0.5],
           explore_method='perturb', explore_param=0.3, limit_explore=True, explorable=True)

# L2
svr.add_hp('l2_gen_scale', (1e-4, 1.0), init_sample_mode='logrand', explorable=True)
svr.add_hp('l2_con_scale', (1e-4, 1.0), init_sample_mode='logrand', explorable=True)

# KL
svr.add_hp('kl_co_weight', (1e-6, 1e-4), init_sample_mode='logrand', explorable=True)
svr.add_hp('kl_ic_weight', (1e-5, 1e-3), init_sample_mode='logrand', explorable=True)


''' Other fixed params (default: explorable=False)'''
# Batch size
svr.add_hp('batch_size', [500])
svr.add_hp('valid_batch_size', [500])

# Validation metric used for PBT
# svr.add_hp('val_cost_for_pbt', ['heldout_samp']) # uncomment for sample validation
svr.add_hp('val_cost_for_pbt', ['heldout_trial'])  # trial (standard validation)

svr.add_hp('cv_keep_ratio', [1.0]) # change (<1) if you use sample validation
svr.add_hp('cd_grad_passthru_prob', [0.0])

# Factors
svr.add_hp('factors_dim', [40])
svr.add_hp('in_factors_dim', [0])

# External inputs
svr.add_hp('ext_input_dim', [0])

# Initial Condition Encoder
svr.add_hp('ic_dim', [100])
svr.add_hp('ic_enc_dim', [100])
svr.add_hp('ic_enc_seg_len', [0])  # for causal encoder, default=0 non-causal

# Generator
svr.add_hp('gen_dim', [100])

# Controller
svr.add_hp('co_dim', [10])
svr.add_hp('ci_enc_dim', [64])
svr.add_hp('con_dim', [64])
svr.add_hp('do_causal_controller', [False])
svr.add_hp('controller_input_lag', [1])

# Unused regularizes (not searched)
svr.add_hp('l2_ci_enc_scale', [0])
svr.add_hp('l2_ic_enc_scale', [0])

# Output distribution:
svr.add_hp('output_dist', ['poisson'])

# Ramping up KL/L2 weights
svr.add_hp('kl_start_epoch', [0])
svr.add_hp('l2_start_epoch', [0])
svr.add_hp('kl_increase_epochs', [50])
svr.add_hp('l2_increase_epochs', [50])

# Data path params
svr.add_hp('data_filename_stem', ['chaotic']) # data file must start with this

# ---- frequently not changed ----
# disabling lr annealing, change if you want to use PBT framework for random search
svr.add_hp('learning_rate_decay_factor', [1])
svr.add_hp('learning_rate_stop', [1e-10])
svr.add_hp('learning_rate_n_to_compare', [0])
svr.add_hp('n_epochs_early_stop', [10000])  # not used anyway when learning_rate_n_to_compare=0
svr.add_hp('checkpoint_pb_load_name', ["checkpoint"]) # change to checkpoint_lve for random search

# optimizer params (only search if needed)
svr.add_hp('loss_scale', [1e4])
svr.add_hp('adam_epsilon', [1e-8])
svr.add_hp('beta1', [0.9])
svr.add_hp('beta2', [0.999])

# Frequently not changed
svr.add_hp('data_dir', [data_dir])
svr.add_hp('do_train_readin', [True])
svr.add_hp('do_train_encoder_only', [False])
svr.add_hp('cv_rand_seed', [1000])
svr.add_hp('output_filename_stem', [""])
svr.add_hp('max_ckpt_to_keep', [1])
svr.add_hp('max_ckpt_to_keep_lve', [1])
svr.add_hp('ckpt_save_interval', [50])
svr.add_hp('checkpoint_name', ["lfads_vae"])
svr.add_hp('csv_log', ["fitlog"])
svr.add_hp('device', ["gpu"])
svr.add_hp('ps_nexamples_to_process', [100000000])
svr.add_hp('ic_prior_var', [0.1])
svr.add_hp('ic_post_var_min', [0.0001])
svr.add_hp('prior_ar_atau', [10.0])
svr.add_hp('prior_ar_nvar', [0.1])
svr.add_hp('do_train_prior_ar_atau', [True])
svr.add_hp('do_train_prior_ar_nvar', [True])
svr.add_hp('do_feed_factors_to_controller', [True])
svr.add_hp('feedback_factors_or_rates', ['factors'])
svr.add_hp('do_reset_learning_rate', [True])
svr.add_hp('do_calc_r2', [False])
svr.add_hp('inject_ext_input_to_gen', [True])
svr.add_hp('allow_gpu_growth', [True])
svr.add_hp('max_grad_norm', [200.0])
svr.add_hp('cell_clip_value', [5.0])

''' --------------------------------------------------------------------- '''


''' ----------------------------- Start PBT ----------------------------- '''
best_worker = svr.start_pbt()


''' -------------------------- Post PBT  stuff -------------------------- '''
''' get the path to the best worker '''
best_worker_path = best_worker.run_save_path

''' do posterior mean sample and average on the best model '''
PM_param = {'batch_size':200, 'valid_batch_size':4000}
run_posterior_sample_and_average(best_worker_path, PM_param)


''' copy the best worker to lfadsOutput for run-manager to load '''
src = best_worker_path
dest = lfads_output_dir

# create lfadsOutput directory if doesn't exist
if os.path.exists(dest):
    shutil.rmtree(dest)

os.makedirs(dest)

src_files = os.listdir(src)
for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, dest)