for dataset_name in metatree_mfeat_fourier metatree_mfeat_zernike metatree_mfeat_morphological metatree_mfeat_karhunen metatree_page_blocks metatree_optdigits metatree_pendigits metatree_waveform_5000 metatree_Hyperplane_10_1E_3 metatree_Hyperplane_10_1E_4 metatree_pokerhand metatree_RandomRBF_0_0 metatree_RandomRBF_10_1E_3 metatree_RandomRBF_50_1E_3 metatree_RandomRBF_10_1E_4 metatree_RandomRBF_50_1E_4 metatree_SEA_50_ metatree_SEA_50000_ metatree_satimage metatree_BNG_labor_ metatree_BNG_breast_w_ metatree_BNG_mfeat_karhunen_ metatree_BNG_bridges_version1_ metatree_BNG_mfeat_zernike_ metatree_BNG_cmc_ metatree_BNG_colic_ORIG_ metatree_BNG_colic_ metatree_BNG_credit_a_ metatree_BNG_page_blocks_ metatree_BNG_credit_g_ metatree_BNG_pendigits_ metatree_BNG_cylinder_bands_ metatree_BNG_dermatology_ metatree_BNG_sonar_ metatree_BNG_glass_ metatree_BNG_heart_c_ metatree_BNG_heart_statlog_ metatree_BNG_vehicle_ metatree_BNG_hepatitis_ metatree_BNG_waveform_5000_ metatree_BNG_zoo_ metatree_vehicle_sensIT metatree_UNIX_user_data metatree_fri_c3_1000_25 metatree_rmftsa_sleepdata metatree_JapaneseVowels metatree_fri_c4_1000_100 metatree_abalone metatree_fri_c4_1000_25 metatree_bank8FM metatree_analcatdata_supreme metatree_ailerons metatree_cpu_small metatree_space_ga metatree_fri_c1_1000_5 metatree_puma32H metatree_fri_c3_1000_10 metatree_cpu_act metatree_fri_c4_1000_10 metatree_quake metatree_fri_c4_1000_50 metatree_fri_c0_1000_5 metatree_delta_ailerons metatree_fri_c3_1000_50 metatree_kin8nm metatree_fri_c3_1000_5 metatree_puma8NH metatree_delta_elevators metatree_houses metatree_bank32nh metatree_fri_c1_1000_50 metatree_house_8L metatree_fri_c0_1000_10 metatree_elevators metatree_wind metatree_fri_c0_1000_25 metatree_fri_c2_1000_50 metatree_pollen metatree_mv metatree_fried metatree_fri_c2_1000_25 metatree_fri_c0_1000_50 metatree_fri_c1_1000_10 metatree_fri_c2_1000_5 metatree_fri_c2_1000_10 metatree_fri_c1_1000_25 metatree_visualizing_soil metatree_socmob metatree_mozilla4 metatree_pc3 metatree_pc1
do
for tree_size in 1 5 10 20 30 40 50 60 70 80 90 100
do  
for run in 1 2 3 4 5 6 7 8 9 10
do
PYTHONPATH="./metatree":"$PYTHONPATH" python ./metatree/eval_generalization.py \
    --model_name_or_path "yzhuang/MetaTree" \
    --normalize \
    --dataset_name yzhuang/$dataset_name \
    --max_train_steps $tree_size \
    --backward_window 1 \
    --n_feature 10 \
    --n_class 10 \
    --block_size 256 \
    --depth 2 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 512 \
    --preprocessing_num_workers 8 \
    --inference_only \
    #--with_tracking \ 
    #--report_to "wandb" # In case you want to use wandb to log the results 
done
done
done