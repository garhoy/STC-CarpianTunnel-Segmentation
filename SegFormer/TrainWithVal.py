import torch
from metrics import compute_metrics_wrapper
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
)
from transformers import EarlyStoppingCallback
from utils import MedicalSegmentationDataset,CustomTrainer,data_collator


"""
Train a SegFormer model for semantic segmentation on a medical image dataset.

Key Features:
- Uses NVIDIA's `mit-b5` backbone
- Custom Trainer supports Focal, Dice, or Cross Entropy loss
- Optional validation with early stopping
- Saves final model and processor to output folder

Set `val = True` to enable validation during training.
"""

if __name__ == '__main__':
    print(torch.cuda.is_available())  
    print(torch.cuda.device_count())  
    print(torch.cuda.get_device_name(0))  
    torch.cuda.empty_cache()
    val = False

    ####################################################
    #          Paths and Model Configuration           #
    ####################################################


    TRAIN_CSV_PATH = "Data_Changed/Train_data.csv"
    VAL_CSV_PATH   = "Data_Changed/Val_Data.csv"

    TRAIN_IMG_DIR  = "Data_Changed/train"
    TRAIN_MASK_DIR = "Data_Changed/mask_train"

    VAL_IMG_DIR  = "Data_Changed/val"
    VAL_MASK_DIR = "Data_Changed/mask_val"

    processor = SegformerImageProcessor.from_pretrained(
    "nvidia/mit-b5",
    num_labels=5,
    reshape_last_stage=True
    )

    train_dataset = MedicalSegmentationDataset(
        csv_file=TRAIN_CSV_PATH,
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        processor=processor,
    )
    val_dataset = MedicalSegmentationDataset(
        csv_file=VAL_CSV_PATH,
        image_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        processor=processor,
    )

    # =============================================================================
    # Load Segformer and configure Training Arguments
    # =============================================================================

    num_labels = 5
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5", num_labels=num_labels)

    if val == True:
        training_args = TrainingArguments(
            output_dir="./segformer-medical-output",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            fp16=True,
            num_train_epochs = 100,
            evaluation_strategy="epoch",  
            save_strategy="epoch",
            save_total_limit=1,           # Only last checkpoint
            load_best_model_at_end=True,  
            learning_rate=2e-5,
        )
    else:
        training_args = TrainingArguments(
            output_dir="./segformer-medical-output",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            fp16=True,
            num_train_epochs= 100,
            save_strategy="epoch",        
            save_total_limit=1,           
            learning_rate=2e-5,
        )




    if val == True:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,  
            data_collator=data_collator,
            compute_metrics=compute_metrics_wrapper,
            loss_name="focal",  #  Cross / Focal / Dice
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]  
        )
    else :
        trainer = CustomTrainer(
            model=model,    
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics_wrapper,
            loss_name="focal"  # Focal / Cross / Dice 
        )

    trainer.train()

    # Saving Models 
    
    if val == True:
        model.save_pretrained("./segformer-medical-output/final_model_val")
        processor.save_pretrained("./segformer-medical-output/final_model_val")

    else :
        model.save_pretrained("./segformer-medical-output/final_model_noVal_focal")
        processor.save_pretrained("./segformer-medical-output/final_model_noVal_focal")


