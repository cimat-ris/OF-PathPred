import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
from tqdm import tqdm
from batches_data import get_batch
from testing_utils import evaluation_minadefde

# Parameters
# The only datasets that can use add_social are those of ETH/UCY
# The only datasets that can use add_kp are PETS2009-S2L1, TOWN-CENTRE
class Experiment_Parameters:
    def __init__(self,add_social=False,add_kp=False,obstacles=False):
        # Maximum number of persons in a frame
        self.person_max =70
        # Observation length (trajlet size)
        self.obs_len    = 8
        # Prediction length
        self.pred_len   = 12
        # Flag to consider social interactions
        self.add_social = add_social
        # Number of key points
        self.kp_num     = 18
        # Key point flag
        self.add_kp     = add_kp
        # Obstacles flag
        self.obstacles    = obstacles
        self.intersection = False
        self.delim        = ','
        self.output_representation = 'dxdy' #



def relative_to_abs(rel_traj, start_pos):
    """Relative x,y to absolute x,y coordinates.
    Args:
    rel_traj: numpy array [T,2]
    start_pos: [2]
    Returns:
    abs_traj: [T,2]
    """
    # batch, seq_len, 2
    # the relative xy cumulated across time first
    displacement = np.cumsum(rel_traj, axis=0)
    abs_traj = displacement + np.array([start_pos])  # [1,2]
    return abs_traj

# Training loop
def training_loop(model,train_data,val_data,config,checkpoint,checkpoint_prefix):
    train_loss_results   = []
    val_loss_results     = []
    val_metrics_results  = {'mADE': [], 'mFDE': [], 'obs_classif_accuracy': []}
    train_metrics_results= {'obs_classif_accuracy': []}
    best                 = {'mADE':999999, 'mFDE':0, 'batchId':-1}
    train_metrics        = {'obs_classif_sca':keras.metrics.SparseCategoricalAccuracy()}
    val_metrics          = {'obs_classif_sca':keras.metrics.SparseCategoricalAccuracy()}
    # TODO: Shuffle

    # Training the main system
    for epoch in range(config.num_epochs):
        print('Epoch {}.'.format(epoch + 1))
        # Cycle over batches
        total_loss = 0
        #num_batches_per_epoch = train_data.get_num_batches()
        #for idx,batch in tqdm(train_data.get_batches(config.batch_size, num_steps = num_batches_per_epoch, shuffle=True), total = num_batches_per_epoch, ascii = True):
        num_batches_per_epoch= train_data.cardinality().numpy()
        for batch in tqdm(train_data,ascii = True):
            # Format the data
            batch_inputs, batch_targets = get_batch(batch, config)
            # Run the forward pass of the layer.
            # Compute the loss value for this minibatch.
            batch_loss = model.batch_step(batch_inputs, batch_targets, train_metrics, training=True)
            total_loss+= batch_loss
        # End epoch
        total_loss = total_loss / num_batches_per_epoch
        train_loss_results.append(total_loss)

        # Saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        # Display information about the current state of the training loop
        print('[TRN] Epoch {}. Training loss {:.4f}'.format(epoch + 1, total_loss ))
        # print('[TRN] Training accuracy of classifier p(z|x)   {:.4f}'.format(float(train_metrics['obs_classif_sca'].result()),))
        train_metrics['obs_classif_sca'].reset_states()

        if config.use_validation:
            # Compute validation loss
            total_loss = 0
            # num_batches_per_epoch = val_data.get_num_batches()
            # for idx, batch in tqdm(val_data.get_batches(config.batch_size, num_steps = num_batches_per_epoch), total = num_batches_per_epoch, ascii = True):
            num_batches_per_epoch= val_data.cardinality().numpy()
            for idx,batch in tqdm(enumerate(val_data),ascii = True):
                # Format the data
                batch_inputs, batch_targets = get_batch(batch, config)
                batch_loss                  = model.batch_step(batch_inputs,batch_targets, val_metrics, training=False)
                total_loss+= batch_loss
            # End epoch
            total_loss = total_loss / num_batches_per_epoch
            print('[TRN] Epoch {}. Validation loss {:.4f}'.format(epoch + 1, total_loss ))
            val_loss_results.append(total_loss)
            # Evaluate ADE, FDE metrics on validation data
            val_quantitative_metrics = evaluation_minadefde(model,val_data,config)
            val_metrics_results['mADE'].append(val_quantitative_metrics['mADE'])
            val_metrics_results['mFDE'].append(val_quantitative_metrics['mFDE'])
            if val_quantitative_metrics["mADE"]< best['mADE']:
                best['mADE'] = val_quantitative_metrics["mADE"]
                best['mFDE'] = val_quantitative_metrics["mFDE"]
                best["patchId"]= idx
                # Save the best model so far
                checkpoint.write(checkpoint_prefix+'-best')
            print('[TRN] Epoch {}. Validation mADE {:.4f}'.format(epoch + 1, val_quantitative_metrics['mADE']))

    # Training the classifier
    for epoch in range(0):
        print('Epoch {}.'.format(epoch + 1))
        # Cycle over batches
        # num_batches_per_epoch = train_data.get_num_batches()
        # for idx, batch in tqdm(train_data.get_batches(config.batch_size, num_steps = num_batches_per_epoch, shuffle=True), total = num_batches_per_epoch, ascii = True):
        num_batches_per_epoch= train_data.cardinality().numpy()
        for batch in tqdm(train_data,ascii = True):
            # Format the data
            batch_inputs, batch_targets = get_batch(batch, config)
            # Run the forward pass of the layer.
            # Compute the loss value for this minibatch.
            batch_loss = model.batch_step(batch_inputs, batch_targets, train_metrics, training=True)
            total_loss+= batch_loss
        # End epoch
        total_loss = total_loss / num_batches_per_epoch
        train_loss_results.append(total_loss)

        # Display information about the current state of the training loop
        print('[TRN] Epoch {}. Training loss {:.4f}'.format(epoch + 1, total_loss ))
        print('[TRN] Training accuracy of classifier p(z|x)   {:.4f}'.format(float(train_metrics['obs_classif_sca'].result()),))
        train_metrics['obs_classif_sca'].reset_states()

    return train_loss_results,val_loss_results,val_metrics_results,best["patchId"]
