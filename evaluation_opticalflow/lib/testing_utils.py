from tqdm import tqdm
from plot_utils import plot_gt_preds
from traj_utils import relative_to_abs, vw_to_abs
from batches_data import get_batch
import numpy as np
import tensorflow as tf

""" \multicolumn{1}{c|}{} & S-GAN \cite{socialGAN} & SoPhie \cite{sophie} & Soc-BIGAT \cite{bigat} & \multicolumn{1}{c}{\acrshort{vvae}}          \\ \hline
ETH                   & 0.87/1.62                               & \cellcolor[HTML]{FFCE93}0.70/1.43     & \cellcolor[HTML]{FFFC9E}0.69/1.29       & \cellcolor[HTML]{FFFC9E}0.68/1.30 \\
HOTEL                 & \cellcolor[HTML]{FFCE93}0.67/1.37       & 0.76/1.67                             & \cellcolor[HTML]{EFEFEF}0.49/1.01       & \cellcolor[HTML]{FFFC9E}0.32/0.67 \\
UNIV                  & 0.76/1.52                               & \cellcolor[HTML]{EFEFEF}0.54/1.24     & \cellcolor[HTML]{FFCE93}0.55/1.32       & \cellcolor[HTML]{FFFC9E}0.43/0.92 \\
ZARA1                 & 0.35/0.68                               & \cellcolor[HTML]{EFEFEF}0.30/0.63     & \cellcolor[HTML]{FFFC9E}0.30/0.62       & \cellcolor[HTML]{FFCE93}0.33/0.68 \\
ZARA2                 & 0.42/0.84                               & \cellcolor[HTML]{FFCE93}0.38/0.78     & \cellcolor[HTML]{EFEFEF}0.36/0.75       & \cellcolor[HTML]{FFFC9E}0.28/0.55 \\ \hline
Avg                   & 0.61/1.21                               & \cellcolor[HTML]{FFCE93}0.54/1.15     & \cellcolor[HTML]{EFEFEF}0.48/1.00       & \cellcolor[HTML]{FFFC9E}0.41/0.83 \\ \hline"""

"""& \multicolumn{1}{c|}{NextP \cite{peeking}} & \multicolumn{1}{c}{$TF_q$ \cite{transformer}}          & \multicolumn{1}{c}{$20S$VAE}      & SDVAE                             \\ \hline
ETH   & 0.73/1.65                  & \cellcolor[HTML]{FFCE93}0.61/1.12   & \cellcolor[HTML]{EFEFEF}0.53/0.95 & \cellcolor[HTML]{FFFC9E}0.40/0.60 \\
HOTEL & 0.30/0.59                  & \cellcolor[HTML]{FFFC9E}0.18/0.30   & \cellcolor[HTML]{FFCE93}0.19/0.37 & \cellcolor[HTML]{EFEFEF}0.18/0.35 \\
UNIV  & 0.60/1.27                  & \cellcolor[HTML]{FFCE93}0.35/0.65   & \cellcolor[HTML]{EFEFEF}0.27/0.47 & \cellcolor[HTML]{FFFC9E}0.26/0.43 \\
ZARA1 & 0.38/0.81                  & \cellcolor[HTML]{FFFC9E}0.22/0.38   & \cellcolor[HTML]{FFFC9E}0.22/0.38 & \cellcolor[HTML]{EFEFEF}0.23/0.41 \\
ZARA2 & 0.31/0.68                  & \cellcolor[HTML]{EFEFEF}0.17/0.32   & \cellcolor[HTML]{FFCE93}0.21/0.38 & \cellcolor[HTML]{FFFC9E}0.17/0.29 \\ \hline
Avg   & 0.46/1.00                  & \cellcolor[HTML]{FFCE93}0.31 / 0.55 & \cellcolor[HTML]{EFEFEF}0.28/0.51 & \cellcolor[HTML]{FFFC9E}0.25/0.41 \\ \hline
\end{tabular}
\end{center}
\caption[Comparison of $20$ \acrshort{vae} and \acrshort{sdvae} with Transformers~\cite{transformer} and NextP~\cite{peeking}]{Comparison of our best \acrshort{vae} and \acrshort{sdvae} with Transformers~\cite{transformer} and NextP~\cite{peeking} (best of $20$: \acrshort{made}/\acrshort{mfde} in meters). The three methods on the right are ranked with gold, plate and bronze.}
\label{soaeth1}
\end{table}"""

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
    return { "min ade": np.mean(ade), "min fde": np.mean(fde)}
