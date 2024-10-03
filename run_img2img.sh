export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0 
export NCCL_P2P_DISABLE=1 
export NCCL_TIMEOUT=14400
export MODEL_DIR=runwayml/stable-diffusion-v1-5
export SERVER_NAME=r6

accelerate launch --mixed_precision="no" --multi_gpu --num_processes=4 train_sd_img2img_orida.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --train_data_dir=/workspace/dataset/postprocess_mask/objects \
 --dataloader_num_workers=8 \
 --resolution=512 \
 --max_train_steps=500000 \
 --learning_rate=1e-5 \
 --train_batch_size=8 \
 --mixed_precision="no" \
 --validation_src_dir "validation_src" \
 --validation_tgt_dir "validation_tgt" \
 --validation_prompt "" \
 --num_validation_images 2 \
 --validation_num_inference_steps 50 \
 --validation_init_timestep 20 \
 --validation_steps 200 \
 --report_to=wandb \
 --tracker_project_name="sd-img2img-orida-v0" \
 --logging_dir="orida_sd_img2img_size512_batch8_gpu4_iter500000_$SERVER_NAME" \
 --output_dir=output_orida_sd_img2img_v0.0 \
 --checkpointing_steps 5000


MODEL_DIR=runwayml/stable-diffusion-inpainting
accelerate launch --mixed_precision="no" --multi_gpu --num_processes=4 train_sd_inpaint_orida.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --train_data_dir=/workspace/dataset/postprocess_mask/objects \
 --dataloader_num_workers=8 \
 --resolution=512 \
 --max_train_steps=500000 \
 --learning_rate=1e-5 \
 --train_batch_size=8 \
 --mixed_precision="no" \
 --validation_src_dir "validation_src" \
 --validation_tgt_dir "validation_tgt" \
 --validation_prompt "" \
 --num_validation_images 3 \
 --validation_num_inference_steps 50 \
 --validation_steps 500 \
 --report_to=wandb \
 --tracker_project_name="sd-inpaint-removal-v0" \
 --logging_dir="orida_sd_removal_size512_batch8_gpu4_iter500000_$SERVER_NAME" \
 --output_dir=output_orida_sd_inpaint_removal \
 --checkpointing_steps 5000 \
 --resume_from_checkpoint latest
