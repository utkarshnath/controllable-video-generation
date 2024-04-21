### Training
```
cd /home/unath/controllable_video_generation/diffusers/examples/controlnet
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/scratch/unath/controllable_video_generation_test" #path to save model
accelerate launch --multi_gpu train_controlnet.py --pretrained_model_name_or_path=$MODEL_DIR --output_dir=$OUTPUT_DIR --dataset_name=sayakpaul/nyu_depth_v2 --resolution=512 --learning_rate=1e-5 --train_batch_size=4 --tracker_project_name="controlnet-demo" --report_to wandb --controlnet_model_name_or_path "/scratch/unath/controllable_video_generation/control_net_depth/"
```
