import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm

class Trainer(object):
    """Trainer class for model."""
    def __init__(self, model, config):
        self.config = config
        self.model = model  # this is a Model instance
        self.global_step = model.global_step
        self.learning_rate = config.init_lr

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
            #opt_rest = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
        elif config.optimizer == 'adadelta':
            opt_emb = tf.train.AdadeltaOptimizer(self.learning_rate*config.emb_lr)
            #opt_rest = tf.train.AdadeltaOptimizer(self.learning_rate)
        elif config.optimizer == 'adam':
            opt_emb = tf.compat.v1.train.AdamOptimizer(self.learning_rate*config.emb_lr)
            #opt_rest = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise Exception('Optimizer not implemented')

        # losses
        #self.xyloss = model.xyloss
        self.loss = model.loss  # get the loss funcion
        #self.grid_data = model.grid_data  # get the vector_ID value
        self.train_op = opt_emb.minimize(self.loss) #### Diferente

    def get_lr(self):
        return self.learning_rate

    def step(self, sess, batch):
        """One training step."""
        config = self.config
        feed_dict = self.model.get_feed_dict(batch,True)
        # TODO: Calculate GridData
        #inputs = [self.loss, self.train_op,self.grid_data]#### Diferente
        inputs = [self.loss, self.train_op]#### Diferente
        #loss, train_op,grid= sess.run(inputs, feed_dict = feed_dict)
        loss, train_op = sess.run(inputs, feed_dict = feed_dict)
        return loss, train_op


class Tester(object):
    """Tester for model."""
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.traj_pred_out = self.model.traj_pred_out
        #self.sess = sess
        if config.multi_decoder:
            self.traj_class_logits = self.model.traj_class_logits
            self.traj_outs = self.model.traj_pred_outs

    def step(self,batch,sess):
        """One inferencing step."""
        config = self.config
        #sess = self.sess
        # give one batch of Dataset, use model to get the result,
        feed_dict = self.model.get_feed_dict(batch, is_train = False)
        pred_out = sess.run(self.traj_pred_out, feed_dict = feed_dict)
        return pred_out


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


def evaluate(dataset, tester,sess,arguments):
    """Evaluate the dataset using the tester model.
    Args:
    dataset: the Dataset instance
    config: arguments
    sess: tensorflow session
    tester: the Tester instance
    Returns:
    Evaluation results.
    """

    config = tester.config
    #sess = tester.sess

    l2dis = []
    num_batches_per_epoch = int(math.ceil(dataset.num_examples / float(config.batch_size)))
    #traj_obs = []
    #traj_ver = []
    #traj_pred = []
    for idx, batch in tqdm(dataset.get_batches(config.batch_size, num_steps = num_batches_per_epoch, shuffle=False), total = num_batches_per_epoch, ascii = True):

        pred_out = tester.step(batch,sess)
        this_actual_batch_size = batch["original_batch_size"]
        d = []
        for i, (obs_traj_gt, pred_traj_gt) in enumerate(zip(batch["obs_traj"], batch["pred_traj"])):
            if i >= this_actual_batch_size:
                break

            this_pred_out = pred_out[i][:, :2] #[pred,2]

            this_pred_out_abs = relative_to_abs(this_pred_out, obs_traj_gt[-1])

            assert this_pred_out_abs.shape == this_pred_out.shape, (this_pred_out_abs.shape, this_pred_out.shape)
            diff = pred_traj_gt - this_pred_out_abs
            #traj_obs.append(obs_traj_gt)
            #traj_ver.append(pred_traj_gt)
            #traj_pred.append(this_pred_out_abs)
            diff = diff**2
            diff = np.sqrt(np.sum(diff, axis=1))
            d.append(diff)

        l2dis += d
    ade = [t for o in l2dis for t in o] # average displacement
    fde = [o[-1] for o in l2dis] # final displacement
    p = { "ade": np.mean(ade), "fde": np.mean(fde)}
    #, traj_ver, traj_pred
    return p


def evaluate_new(dataset, tester,sess):
  """Evaluate the dataset using the tester model.
  Args:
    dataset: the Dataset instance
    config: arguments
    sess: tensorflow session
    tester: the Tester instance
  Returns:
    Evaluation results.
  """
  cont       = 0
  prom       = 0.0
  predicho   = []
  verdadero  = []
  error_prom = []

  config = tester.config
  #sess = tester.sess

  l2dis = []
  num_batches_per_epoch = int(math.ceil(dataset.num_examples / float(config.batch_size)))
  
  for idx, batch in tqdm(dataset.get_batches(config.batch_size, num_steps = num_batches_per_epoch, shuffle=False), total = num_batches_per_epoch, ascii = True):
    
    pred_out = tester.step(batch,sess)
    cont +=1

    this_actual_batch_size = batch["original_batch_size"]
    d = []
    for i, (obs_traj_gt, pred_traj_gt) in enumerate(zip(batch["obs_traj"], batch["pred_traj"])):
      if i >= this_actual_batch_size:
        break
      #print("iterador")
      #print(i)
      this_pred_out = pred_out[i][:, :2]
      this_pred_out_abs = relative_to_abs(this_pred_out, obs_traj_gt[-1])

      assert this_pred_out_abs.shape == this_pred_out.shape, (this_pred_out_abs.shape, this_pred_out.shape)

      diff = pred_traj_gt - this_pred_out_abs
      diff = diff**2
      diff = np.sqrt(np.sum(diff, axis=1))
      d.append(diff)
      #print("Trayectoria verdadera")
      #print( pred_traj_gt)
      #print("Trayectoria predicha")
      #print( this_pred_out_abs )
      #print(np.mean(diff))
      predicho.append(this_pred_out_abs)
      verdadero.append(pred_traj_gt)
      prom += np.mean(diff)
      error_prom.append(np.mean(diff))

    l2dis += d


  ade = [t for o in l2dis for t in o] # average displacement
  fde = [o[-1] for o in l2dis] # final displacement


  p = { "ade": np.mean(ade), "fde": np.mean(fde),"error_prom":prom/len(error_prom)}

  

  return p,predicho, verdadero