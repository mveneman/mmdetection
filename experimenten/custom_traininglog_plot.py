import json
import matplotlib.pyplot as plt


def make_train_validation_plot(log_file_path, experiment_name=""):
    # Initialize lists to store training loss and validation coco/segm_mAP
    training_loss = []
    validation_coco_segm_mAP = []
    validation_epochs = []
    

    # Read the JSON log file
    with open(log_file_path, 'r') as file:
        epoch_training_loss = []
        current_epoch = 0

        for line in file:
            data = json.loads(line)
            
            # Check if it's a training or validation entry
            if 'loss' in data:
                if data['epoch'] != current_epoch:
                    if epoch_training_loss:
                        training_loss.append(sum(epoch_training_loss) / len(epoch_training_loss))
                        epoch_training_loss = []
                    current_epoch = data['epoch']
                epoch_training_loss.append(data['loss'])
            elif 'coco/segm_mAP' in data:
                validation_coco_segm_mAP.append(data['coco/segm_mAP'])
                validation_epochs.append(current_epoch)

    # Append the training loss of the last epoch
    if epoch_training_loss:
        training_loss.append(sum(epoch_training_loss) / len(epoch_training_loss))

    print("Validation mAP: ", validation_coco_segm_mAP)
    
    # Making the plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_loss) + 1), training_loss, color = '#015293', label='Training Loss', marker='o')
    plt.plot(validation_epochs, validation_coco_segm_mAP, color = '#66b0c9', label='Validation coco/segm_mAP', marker='o')

    plt.title('Training Loss and Validation coco/segm_mAP' + '\n' + experiment_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss / coco/segm_mAP')
    plt.legend()
    plt.grid(True)
    plt.show()