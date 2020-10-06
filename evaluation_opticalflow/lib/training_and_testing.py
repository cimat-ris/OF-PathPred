import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm

# Parameters
# The only datasets that can use add_social are those of ETH/UCY
# The only datasets that can use add_kp are PETS2009-S2L1, TOWN-CENTRE 
class Experiment_Parameters:
    def __init__(self,add_social=False,add_kp=False,obstacles=False):
        # Maximum number of persons in a frame
        self.person_max = 42 # 8   # Univ: 42  Hotel: 28
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
        self.neighborhood = False
        self.intersection = False

# the
class Experiment_Parameters_Various:
    def __init__(self,intersection= True,add_social=False, obstacles = False, neighborhood= False, ind_test=0):
         
        self.intersection = intersection
        self.data_dirs = ['../data1/eth-univ', '../data1/eth-hotel',
             '../data1/ucy-zara01', '../data1/ucy-zara02',
             '../data1/ucy-univ']
        if self.intersection:
            self.list_max_person = [5,8,14,14,40]
        else:
            self.list_max_person = [42,28,22,26,68]
    
        #el indice del conjunto de test
        self.ind_test = ind_test
        #the max number of people of the  test set 
        self.person_max = self.list_max_person[ind_test]
        # The direction of the test set
        self.dir_test     = self.data_dirs[ind_test]

        self.obs_len = 8
        self.pred_len = 12
        self.kp_num = 18

        self.add_social   = add_social
        self.neighborhood = neighborhood
        self.lim          = [[-7.69,14.42,-3.17,13.21,4], [-3.2881,4.3802,-10.2537,4.316,2], 
        [-0.139538367682,15.4805506734,-0.37469588555,12.3864436051, 3],
        [-0.357790686363,15.558422764,-0.273742790271,13.9427441591, 3],
        [-0.174686040989,15.4369843957,-0.222192273533,13.8542013734, 3]]
       
        self.obstacles    = obstacles
        self.add_kp       = False



class Trainer(object):
    """Trainer class for model."""
    def __init__(self, model, config):
        self.config       = config
        self.model        = model  # this is a Model instance
        self.global_step  = model.global_step
        self.learning_rate= config.init_lr

        if config.learning_rate_decay is not None:
            decay_steps = int(config.train_num_examples /
                        config.batch_size * config.num_epoch_per_decay)
            self.learning_rate = tf.compat.v1.train.exponential_decay(
                config.init_lr,
                self.global_step,
                decay_steps,  # decay every k samples used in trainingf
                config.learning_rate_decay,
                staircase=True)

        if config.optimizer == 'momentum':
            opt_emb = tf.train.MomentumOptimizer(
                self.learning_rate*config.emb_lr, momentum=0.9)
        elif config.optimizer == 'adadelta':
            opt_emb = tf.train.AdadeltaOptimizer(self.learning_rate*config.emb_lr)
        elif config.optimizer == 'adam':
            opt_emb = tf.compat.v1.train.AdamOptimizer(self.learning_rate*config.emb_lr)
        else:
            raise Exception('Optimizer not implemented')

        # losses
        self.loss = model.loss  # get the loss funcion
        self.train_op = opt_emb.minimize(self.loss) #### Diferente

    def get_lr(self):
        return self.learning_rate

    def step(self, sess, batch):
        """One training step."""
        config    = self.config
        feed_dict = self.model.get_feed_dict(batch,True)
        inputs    = [self.loss, self.train_op]
        loss, train_op = sess.run(inputs, feed_dict = feed_dict)
        return loss, train_op


class Tester(object):
    """Tester for model."""
    def __init__(self, model, config):
        self.config = config
        self.model  = model
        self.traj_pred_out = self.model.traj_pred_out
        if config.multi_decoder:
            self.traj_class_logits = self.model.traj_class_logits
            self.traj_outs = self.model.traj_pred_outs

    def step(self,batch,sess):
        """One inference step."""
        # Give one batch of Dataset, use model to get the result,
        feed_dict = self.model.get_feed_dict(batch,False)
        pred_out = sess.run(self.traj_pred_out, feed_dict = feed_dict)
        return pred_out

    def evaluate(self,dataset,sess):
        """Evaluate a dataset using the tester model.
        Args:
            dataset: the Dataset instance
            config: arguments
            sess: tensorflow session
        Returns:
            Evaluation results.
        """
        config = self.config
        l2dis = []
        num_batches_per_epoch = dataset.get_num_batches()
        for idx, batch in tqdm(dataset.get_batches(config.batch_size, num_steps = num_batches_per_epoch, shuffle=False), total = num_batches_per_epoch, ascii = True):
            #
            pred_out               = self.step(batch,sess)
            this_actual_batch_size = batch["original_batch_size"]
            d = []
            # For all the trajectories in the batch
            for i, (obs_traj_gt, pred_traj_gt) in enumerate(zip(batch["obs_traj"], batch["pred_traj"])):
                if i >= this_actual_batch_size:
                    break
                # Conserve the x,y coordinates
                this_pred_out     = pred_out[i][:, :2] #[pred,2]
                # Convert it to absolute (starting from the last observed position)
                this_pred_out_abs = relative_to_abs(this_pred_out, obs_traj_gt[-1])
                # Check shape is OK
                assert this_pred_out_abs.shape == this_pred_out.shape, (this_pred_out_abs.shape, this_pred_out.shape)
                # Error for ade/fde
                diff = pred_traj_gt - this_pred_out_abs
                diff = diff**2
                diff = np.sqrt(np.sum(diff, axis=1))
                d.append(diff)
            l2dis += d
        ade = [t for o in l2dis for t in o] # average displacement
        fde = [o[-1] for o in l2dis] # final displacement
        return { "ade": np.mean(ade), "fde": np.mean(fde)}

    def evaluate_new(self,dataset, sess):
        """Evaluate the dataset using the tester model.
        Args:
        dataset: the Dataset instance
        config: arguments
        sess: tensorflow session
        tester: the Tester instance
        Returns:
        Evaluation results.
        """

        predicho   = []
        verdadero  = []
        observado  = []

        config = self.config
        l2dis  = []
        num_batches_per_epoch = dataset.get_num_batches()

        for idx, batch in tqdm(dataset.get_batches(config.batch_size, num_steps = num_batches_per_epoch, shuffle=False), total = num_batches_per_epoch, ascii = True):

            pred_out = self.step(batch,sess)
            this_actual_batch_size = batch["original_batch_size"]
            d = []
            for i, (obs_traj_gt, pred_traj_gt) in enumerate(zip(batch["obs_traj"], batch["pred_traj"])):
                if i >= this_actual_batch_size:
                    break

                this_pred_out = pred_out[i][:, :2]
                this_pred_out_abs = relative_to_abs(this_pred_out, obs_traj_gt[-1])

                assert this_pred_out_abs.shape == this_pred_out.shape, (this_pred_out_abs.shape, this_pred_out.shape)

                diff = pred_traj_gt - this_pred_out_abs
                diff = diff**2
                diff = np.sqrt(np.sum(diff, axis=1))
                d.append(diff)

                observado.append(obs_traj_gt)
                predicho.append(this_pred_out_abs)
                verdadero.append(pred_traj_gt)
            l2dis += d


        ade = [t for o in l2dis for t in o] # average displacement
        fde = [o[-1] for o in l2dis] # final displacement
        p = { "ade": np.mean(ade), "fde": np.mean(fde)}
        return p,observado, verdadero, predicho

    def apply_on_batch(self,dataset,batchId,sess):
        """Evaluate a dataset batch using the tester model.
        Args:
            dataset: the dataset instance
            batchId: the batch to apply the predictor on
            sess:    tensorflow session
        Returns:
            traj_obs,traj_gt,traj_pred.
        """
        config   = self.config
        traj_obs = []
        traj_gt  = []
        traj_pred= []
        # Scan all the batches and simply stop when we reach the one with Id batchId
        num_batches_per_epoch = dataset.get_num_batches()
        
        for count, (idx, batch) in enumerate(dataset.get_batches(config.batch_size,num_steps = num_batches_per_epoch,shuffle = False)):
                if count==batchId:
                    # Apply the network to this batch
                    pred_out  = self.step(batch,sess)
                    this_actual_batch_size = batch["original_batch_size"]
                    # For all the trajectories in the batch
                    for i, (obs_traj_gt, pred_traj_gt) in enumerate(zip(batch["obs_traj"], batch["pred_traj"])):
                        if i >= this_actual_batch_size:
                            break
                        # Conserve the x,y coordinates
                        this_pred_out     = pred_out[i][:, :2]
                        # Convert it to absolute (starting from the last observed position)
                        this_pred_out_abs = relative_to_abs(this_pred_out, obs_traj_gt[-1])
                        # Keep all the trajectories
                        traj_obs.append(obs_traj_gt)
                        traj_gt.append(pred_traj_gt)
                        traj_pred.append(this_pred_out_abs)
                    break
        return traj_obs,traj_gt,traj_pred

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