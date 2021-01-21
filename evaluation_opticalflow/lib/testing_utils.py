from tqdm import tqdm
from traj_utils import relative_to_abs, vw_to_abs
from batches_data import get_batch
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

mADEFDE = {
  "ETH" : {
    "S-GAN": (0.87,1.62),
    "SoPhie": (0.70,1.43),
    "Soc-BIGAT": (0.69,1.29),
    "NextP": (0.73,1.65),
    "TFq": (0.61,1.12),
    "Trajectron": (0.59,1.14),
    "Trajectron++": (0.39,0.83),
    "VVAE": (0.68,1.30),
    "20SVAE": (0.53,0.95),
    "SDVAE": (0.40,0.60),
  },
  "HOTEL" : {
    "S-GAN": (0.67,1.37),
    "SoPhie": (0.76,1.67),
    "Soc-BIGAT": (0.49,1.01),
    "NextP": (0.30,0.59),
    "TFq": (0.18,0.30),
    "Trajectron": (0.35,0.66),
    "Trajectron++": (0.12,0.21),
    "VVAE": (0.32,0.67),
    "20SVAE": (0.19,0.37),
    "SDVAE": (0.18,0.35),
  },
  "UNIV" : {
    "S-GAN": (0.76,1.52),
    "SoPhie": (0.54,1.24),
    "Soc-BIGAT": (0.55,1.32),
    "NextP": (0.60,1.27),
    "TFq": (0.35,0.65),
    "Trajectron": (0.54,1.13),
    "Trajectron++": (0.20,0.44),
    "VVAE": (0.43,0.92),
    "20SVAE": (0.27,0.47),
    "SDVAE": (0.26,0.43),
  },
  "ZARA1": {
    "S-GAN": (0.35,0.68),
    "SoPhie": (0.30,0.63),
    "Soc-BIGAT": (0.30,0.62),
    "NextP": (0.38,0.81),
    "TFq": (0.22,0.38),
    "Trajectron": (0.43,0.83),
    "Trajectron++": (0.15,0.33),
    "VVAE": (0.33,0.68),
    "20SVAE": (0.22,0.38),
    "SDVAE": (0.23,0.41)
  },
  "ZARA2": {
    "S-GAN": (0.42,0.84),
    "SoPhie": (0.38,0.78),
    "Soc-BIGAT": (0.36,0.75),
    "NextP": (0.31,0.68),
    "TFq": (0.17,0.32),
    "Trajectron": (0.43,0.85),
    "Trajectron++": (0.11,0.25),
    "VVAE": (0.28,0.55),
    "20SVAE": (0.21,0.38),
    "SDVAE": (0.17,0.29),
  }
}

def evaluation_minadefde(model,test_data,config):
    l2dis = []
    # num_batches_per_epoch = test_data.get_num_batches()
    # for idx, batch in tqdm(test_data.get_batches(config.batch_size, num_steps = num_batches_per_epoch, shuffle=True), total = num_batches_per_epoch, ascii = True):
    num_batches_per_epoch= test_data.cardinality().numpy()
    for batch in tqdm(test_data,ascii = True):
        # Format the data
        batch_inputs, batch_targets = get_batch(batch, config)
        pred_out,__                 = model.batch_predict(batch_inputs,batch_targets.shape[1],1)
        pred_out                    = pred_out[0][0]
        # this_actual_batch_size      = batch["original_batch_size"]
        d = []
        # For all the trajectories in the batch
        for i, (obs_traj_gt, pred_traj_gt) in enumerate(zip(batch["obs_traj"], batch["pred_traj"])):
            normin = 1000.0
            diffmin= None
            for k in range(model.output_samples):
                # Conserve the x,y coordinates of the kth trajectory
                this_pred_out     = pred_out[k][i][:, :2] #[pred,2]
                # Convert it to absolute (starting from the last observed position)
                if config.output_representation=='dxdy':
                    this_pred_out_abs = relative_to_abs(this_pred_out, obs_traj_gt[-1])
                else:
                    this_pred_out_abs = vw_to_abs(this_pred_out, obs_traj_gt[-1])
                # Check shape is OK
                assert this_pred_out_abs.shape == this_pred_out.shape, (this_pred_out_abs.shape, this_pred_out.shape)
                # Error for ade/fde
                diff = pred_traj_gt - this_pred_out_abs
                diff = diff**2
                diff = np.sqrt(np.sum(diff, axis=1))
                # To keep the min
                if tf.norm(diff)<normin:
                    normin  = tf.norm(diff)
                    diffmin = diff
            d.append(diffmin)
        l2dis += d
    ade = [t for o in l2dis for t in o] # average displacement
    fde = [o[-1] for o in l2dis] # final displacement
    return { "mADE": np.mean(ade), "mFDE": np.mean(fde)}


def plot_comparisons_minadefde(madefde_results):
    labels = list(mADEFDE["ZARA1"].keys())
    labels.append("This run")
    values = list(mADEFDE["ZARA1"].values())
    values.append((madefde_results["mADE"],madefde_results["mFDE"]))
    values = np.array(values)
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, values[:,0], width, label='mADE')
    rects2 = ax.bar(x + width/2, values[:,1], width, label='mFDE')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('mADE/mFDE (m)')
    ax.set_title('mADE/mFDE')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()
