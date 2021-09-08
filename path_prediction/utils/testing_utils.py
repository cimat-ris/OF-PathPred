from tqdm import tqdm
import numpy as np
import tensorflow as tf
import cv2, heapq, math
from matplotlib import pyplot as plt
from .traj_utils import relative_to_abs
from .plot_utils import plot_gt_preds,plot_background,plot_neighbors,plot_attention
from .batches_data import get_batch

# Since it is used as a submodule, the trajnetplusplustools directory should be there
import sys
sys.path.append("../trajnetplusplustools")
from trajnetplusplustools import TrackRow

sys.path.append("../trajnetplusplusbaselines")
from trajnetplusplusbaselines.evaluator.trajnet_evaluator import TrajnetEvaluator, collision_test, eval
import evaluator.write as write
from joblib import Parallel, delayed
import os, logging, pickle

mADEFDE = {
  "eth-univ" : {
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
  "eth-hotel" : {
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
  "ucy-univ" : {
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
  "ucy-zara01": {
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
  "ucy-zara02": {
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


"""
Extract one batch from a dataset to perform testing
:param testing_data: The model to evaluate.
:param testing_data_path: Batch of data to evaluate.
:return: the trajectories data and their corresponding video frame.
"""
def get_testing_batch(testing_data,testing_data_path):
    # A trajectory id
    testing_data_arr = list(testing_data.as_numpy_iterator())
    randomtrajId     = np.random.randint(len(testing_data_arr),size=1)[0]
    frame_id         = testing_data_arr[randomtrajId]["frames_ids"][0]
    # Get the video corresponding to the testing
    cap   = cv2.VideoCapture(testing_data_path+'/video.avi')
    frame = 0
    while(cap.isOpened()):
        ret, test_bckgd = cap.read()
        if frame == frame_id:
            break
        frame = frame + 1
    # Form the batch
    filtered_data  = testing_data.filter(lambda x: x["frames_ids"][0]==frame_id)
    filtered_data  = filtered_data.batch(20)
    for element in filtered_data.as_numpy_iterator():
        return element, test_bckgd


"""
Apply prediction on a batch
:param model: The model to evaluate.
:param batch: Batch of data to apply the prediction on.
:param config: Model parameters.
:return: Nothing.
"""
def predict_from_batch(model,batch,config):
    traj_obs      = []
    traj_gt       = []
    traj_pred     = []
    neighbors     = []
    attention     = []

    batch_inputs, batch_targets = get_batch(batch, config)
    # Perform prediction
    pred_traj  = model.predict(batch_inputs,batch_targets.shape[1])
    # Cycle over the trajectories of the batch
    for i, (obs_traj_gt, obs_traj_theta, pred_traj_gt, neighbors_gt) in enumerate(zip(batch["obs_traj"],batch["obs_traj_theta"],batch["pred_traj"],batch["obs_neighbors"])):
        this_pred_out_abs_set = []
        nsamples = pred_traj.shape[1]
        # TODO: use the reconstruct function
        c, s     = np.cos(obs_traj_theta[-1,0]), np.sin(obs_traj_theta[-1,0])
        for k in range(nsamples):
            # Conserve the x,y coordinates
            if (pred_traj[i,k].shape[0]==config.pred_len):
                this_pred_out     = pred_traj[i,k,:, :2]
                # Coordinates: two cases
                if config.coords_mode=="rel_rot":
                    # Convert it to absolute (starting from the last observed position)
                    this_pred_traj_rot  = np.zeros_like(this_pred_out)
                    this_pred_traj_rot[:,0] = c*this_pred_out[:,0]-s*this_pred_out[:,1]
                    this_pred_traj_rot[:,1] = s*this_pred_out[:,0]+c*this_pred_out[:,1]
                    this_pred_out_abs = relative_to_abs(this_pred_traj_rot, obs_traj_gt[-1])
                else:
                    if config.coords_mode=="rel":
                        this_pred_out_abs = relative_to_abs(this_pred_out, obs_traj_gt[-1])
                this_pred_out_abs_set.append(this_pred_out_abs)
        this_pred_out_abs_set = tf.stack(this_pred_out_abs_set,axis=0)
        # TODO: tensors instead of lists?
        # Keep all the trajectories
        traj_obs.append(obs_traj_gt)
        traj_gt.append(pred_traj_gt)
        traj_pred.append(this_pred_out_abs_set)
        neighbors.append(neighbors_gt)
    return traj_obs,traj_gt,traj_pred,neighbors

"""
Perform a qualitative evaluation over a batch of n_trajectories
:param model: The model to evaluate.
:param batch: Batch of data to evaluate.
:param background: Background image (in case the data are reprojected).
:param homography: Homography (in case the data are reprojected).
:param flip: Flip flag to flip the image coordinates  (in case the data are reprojected).
:param n_peds_max: Maximum number of neighbors to display.
:param display_mode:
:return: Nothing.
"""
def evaluation_qualitative(model,batch,config,background=None,homography=None,flip=False,n_peds_max=1000):
    traj_obs,traj_gt,traj_pred,neighbors = predict_from_batch(model,batch,config)
    # Plot ground truth and predictions
    plt.subplots(1,1,figsize=(10,10))
    ax = plt.subplot(1,1,1)
    if background is not None:
        plot_background(ax,background)
    plot_neighbors(ax,neighbors,homography,flip=flip)
    plot_gt_preds(ax,traj_gt,traj_obs,traj_pred,homography,flip=flip,n_peds_max=n_peds_max)
    plt.show()

"""
For a multiple-output prediction, evaluate the minADE and minFDE
:param start: Starting position (last observation).
:param theta: Orientation at the last observation.
:param pred_traj: Tensor of velocities predictions.
:return: Trajectory reconstructed in absolute coordinates
"""
def reconstruct(start,theta,this_pred_traj,mode="rel_rot"):
    if mode=="rel_rot":
        # Inverse roation to
        c, s = np.cos(theta), np.sin(theta)
        # Convert it to absolute (starting from the last observed position)
        this_pred_traj_rot  = np.zeros_like(this_pred_traj)
        this_pred_traj_rot[:,:,0] = c*this_pred_traj[:,:,0]-s*this_pred_traj[:,:,1]
        this_pred_traj_rot[:,:,1] = s*this_pred_traj[:,:,0]+c*this_pred_traj[:,:,1]
    else:
        if mode=="rel":
            this_pred_traj_rot = this_pred_traj
    return  relative_to_abs(this_pred_traj_rot, start, axis=1)

"""
For a multiple-output prediction, evaluate the minADE and minFDE
:param start: Starting position (last observation).
:param theta: Orientation at the last observation.
:param pred_traj_gt: Ground truth future trajectory.
:param pred_traj: Tensor of velocities predictions.
:return: minADE, minDFE
"""
def minadefde(start, theta, pred_traj, pred_traj_gt, mode="rel_rot"):
    nsamples = pred_traj.shape[0]
    ademin   = np.inf
    fdemin   = np.inf
    # Reconstruct all samples
    this_pred_traj_abs = reconstruct(start,theta,pred_traj[:,:, :2], mode)
    for k in range(nsamples):
        # Error for ade
        diff = pred_traj_gt - this_pred_traj_abs[k]
        diff = diff**2
        diff = np.sqrt(np.sum(diff, axis=1))
        # To keep the smallest ade/fde
        if np.mean(diff)<ademin:
            ademin  = np.mean(diff)
        if diff[-1]<fdemin:
            fdemin  = diff[-1]
    return ademin,fdemin

"""
Perform quantitative evaluation for a whole batched dataset
:param model: The model to evaluate.
:param test_data: Batched dataset to evaluate.
:param config: Model parameters.
:return: Dictionary of metrics: "mADE", "mFDE"
"""
def evaluation_minadefde(model,test_data,config):
    ades = []
    fdes = []
    for batch in tqdm(test_data,ascii = True):
        if hasattr(config, 'add_social') and config.add_social:
            pred_out  = model.predict([batch['obs_traj_'+config.coords_mode],batch['obs_optical_flow']],batch['pred_traj_'+config.coords_mode].shape[1])
        else:
            pred_out  = model.predict([batch['obs_traj_'+config.coords_mode]],batch['pred_traj_'+config.coords_mode].shape[1])
        # For all the trajectories in the batch
        for i, (obs_traj_gt, obs_theta_gt, pred_traj_gt) in enumerate(zip(batch["obs_traj"], batch["obs_traj_theta"], batch["pred_traj"])):
            made,mfde = minadefde(obs_traj_gt[-1], obs_theta_gt[-1,0], pred_out[i], pred_traj_gt, mode=config.coords_mode)
            ades.append(made)
            fdes.append(mfde)
    return {"mADE": np.mean(ades), "mFDE": np.mean(fdes)}

"""
Perform quantitative evaluation for a whole batched dataset in trajnetplusplus
:param model: The model to evaluate.
:param test_data: Batched dataset to evaluate.
:param config: Model parameters.
:return: Dictionary of metrics: "mADE", "mFDE"
"""
def evaluation_trajnetplusplus_minadefde(model,test_data,test_primary_path,config,table=None):
    l2dis = []
    num_batches_per_epoch= test_data.cardinality().numpy()
    # Here we reconstruct the trajectories
    scenes_gt_batch   = []
    scenes_sub_batch  = []
    scenes_id_gt_batch= []
    for batch in tqdm(test_data,ascii = True):
        print(batch["index"].shape)
        # Primary_path
        test_primary_path_local = [ test_primary_path[row.numpy()] for row in batch["index"]]
        scenes_id_gt, scenes_gt_all, scenes_by_id = zip(*test_primary_path_local)

        ## indexes is dictionary deciding which scenes are in which type
        indexes = {}
        for i in range(1, 5):
            indexes[i] = []
        ## sub-indexes
        sub_indexes = {}
        for i in range(1, 5):
            sub_indexes[i] = []
        for scene in scenes_by_id:
            tags = scene.tag
            main_tag = tags[0:1]
            sub_tags = tags[1]
            for ii in range(1, 5):
                if ii in main_tag:
                    indexes[ii].append(scene.scene)
                if ii in sub_tags:
                    sub_indexes[ii].append(scene.scene)
        # Here we perform the prediction
        if hasattr(config, 'add_social') and config.add_social:
            pred_out  = model.predict([batch['obs_traj_'+config.coords_mode],batch['obs_optical_flow']],batch['pred_traj_'+config.coords_mode].shape[1])
        else:
            pred_out  = model.predict([batch['obs_traj_'+config.coords_mode]],batch['pred_traj_'+config.coords_mode].shape[1])
        # For all the trajectories in the batch
        for i, (obs_traj_gt, pred_traj_gt) in enumerate(zip(batch["obs_traj"], batch["pred_traj"])):
            normin = 1000.0
            diffmin= None
            if hasattr(model,'output_samples'):
                nsamples = model.output_samples
            else:
                nsamples = 1
            #for k in range(nsamples):
            # Conserve the x,y coordinates of the kth trajectory
            this_pred_out     = pred_out[i,0,:, :2] #[pred,2]
            # Convert it to absolute (starting from the last observed position)
            this_pred_out_abs = reconstruct(obs_traj_gt[-1],0.0,this_pred_out,config.coords_mode)
            # Check shape is OK
            assert this_pred_out_abs.shape == this_pred_out.shape, (this_pred_out_abs.shape, this_pred_out.shape)
            # Ground truth
            scenes_gt = scenes_gt_all[i][config.obs_len+1:]
            print((np.sum(np.square(pred_traj_gt-this_pred_out_abs),axis=1)))
            #print(scenes_gt)
            #print(pred_traj_gt)
            scenes_sub = [ TrackRow(path.frame, path.pedestrian, x , y, 0, scenes_id_gt[i] ) for path, (x,y) in zip(scenes_gt,this_pred_out_abs)]
            # Guardamos  en la lista del batch
            scenes_gt_batch.append([scenes_gt])
            scenes_sub_batch.append([scenes_sub])
            scenes_id_gt_batch.append(scenes_id_gt[i])

    # Evaluate
    logging.info("Calling TrajnetEvaluator with {} trajectories".format(len(scenes_sub_batch)))
    evaluator = TrajnetEvaluator([], scenes_gt_batch, scenes_id_gt_batch, scenes_sub_batch, indexes, sub_indexes, config)
    evaluator.aggregate('kf', True)
    results = {"Evaluation":evaluator.result(),}

    if table != None:
        logging.info("Creating results table")
        # Creamos la tabla de resultados
        table.add_collision_entry("Our_Model", "NA")
        final_result, sub_final_result = table.add_entry('Our_Model', results)
    return table

"""
Determines and plots the worst cases (in ADE) on a dataset
:param model: The model to evaluate.
:param test_data: Dataset to evaluate.
:param config: Model parameters.
:param nworst: Number of worst cases to consider.
:param background: Background image (in case the data are reprojected).
:param homography: Homography (in case the data are reprojected).
:param flip: Flip flag to flip the image coordinates  (in case the data are reprojected).
:param n_peds_max: Maximum number of neighbors to display.
:param display_mode:
:return: Nothing.
"""
def evaluation_worstcases(model,test_data,config,nworst=10,background=None,homography=None,flip=False,n_peds_max=1000):
    # Search for worst cases
    worst = []
    for batch in tqdm(test_data,ascii = True):
        # Format the data
        if hasattr(config, 'add_social') and config.add_social:
            pred_out  = model.predict([batch['obs_traj_'+config.coords_mode],batch['obs_optical_flow']],batch['pred_traj_'+config.coords_mode].shape[1])
        else:
            pred_out  = model.predict([batch['obs_traj_'+config.coords_mode]],batch['pred_traj_'+config.coords_mode].shape[1])
        # For all the trajectories in the batch
        for i, (obs_traj_gt, obs_theta_gt, pred_traj_gt) in enumerate(zip(batch["obs_traj"], batch["obs_traj_theta"], batch["pred_traj"])):
            made,__ = minadefde(obs_traj_gt[-1], obs_theta_gt[-1], pred_out[i], pred_traj_gt,mode=config.coords_mode)
            # To have unique values in the heap
            made += 0.0001*np.random.random_sample()
            heapq.heappush(worst,(made,[obs_traj_gt,obs_theta_gt[-1],pred_traj_gt,pred_out[i]]))
            if len(worst)>nworst:
                heapq.heappop(worst)

    # Plot ground truth and predictions
    plt.subplots(1,1,figsize=(10,10))
    for made,[obs_traj_gt, theta, pred_traj_gt,pred] in worst:
        logging.info("mADE : {:.4f}".format(made))
        ax = plt.subplot(1,1,1)
        if background is not None:
            plot_background(ax,background)
        pred_traj_abs = reconstruct(obs_traj_gt[-1],theta,pred,config.coords_mode)
        plot_gt_preds(ax,[pred_traj_gt],[obs_traj_gt],[pred_traj_abs],homography,flip=flip,n_peds_max=n_peds_max,title='Worst cases')
        plt.show()

def plot_comparisons_minadefde(madefde_results,dataset_name):
    labels = list(mADEFDE[dataset_name].keys())
    labels.append("This run")
    values = list(mADEFDE[dataset_name].values())
    values.append((madefde_results["mADE"],madefde_results["mFDE"]))
    values = np.array(values)
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, values[:,0], width, label='mADE')
    rects2 = ax.bar(x + width/2, values[:,1], width, label='mFDE')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('mADE/mFDE (m)')
    ax.set_title('mADE/mFDE on '+dataset_name)
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

def evaluation_trajnetplusplus_other_models(args,table):
        ## Test_pred : Folders for saving model predictions
        args.path = args.path + '/test_pred/'
        args.output = args.output if args.output is not None else []
        ## assert length of output models is not None
        if (not args.sf) and (not args.orca) and (not args.kf) and (not args.cv):
            assert len(args.output), 'No output file is provided'
        print(args)
        # Generate predictions for the other models
        write.main(args)

        ## Evaluates test_pred with test_private
        names = []
        for model in args.output:
            model_name = model.split('/')[-1].replace('.pkl', '')
            model_name = model_name + '_modes' + str(args.modes)
            names.append(model_name)
        # For
        for num, name in enumerate(names):
            # Result file
            result_file = args.path.replace('pred', 'results') + name
            ## If result was pre-calculated and saved, Load
            if os.path.exists(result_file + '/results.pkl'):
                with open(result_file + '/results.pkl', 'rb') as handle:
                    [final_result, sub_final_result, col_result] = pickle.load(handle)
                table.add_result(names[num], final_result, sub_final_result)
                table.add_collision_entry(names[num], col_result)
            ## Else, Calculate results and save
            else:
                # List of datasets to process
                list_sub = []
                for f in os.listdir(args.path + name):
                     if not f.startswith('.'):
                             list_sub.append(f)
                ## Simple Collision Test
                col_result = collision_test(list_sub, name, args)
                table.add_collision_entry(names[num], col_result)
                submit_datasets= [args.path + name + '/' + f for f in list_sub if 'collision_test.ndjson' not in f]
                true_datasets  = [args.path.replace('pred', 'private') + f for f in list_sub if 'collision_test.ndjson' not in f]
                ## Evaluate submitted datasets with True Datasets [The main eval function]
                results = {submit_datasets[i].replace(args.path, '').replace('.ndjson', ''):
                             eval(true_datasets[i], submit_datasets[i], args)
                            for i in range(len(true_datasets))}

                #results_list = Parallel(n_jobs=4)(delayed(eval)(true_datasets[i], submit_datasets[i], args) for i in range(len(true_datasets)))
                #results = {submit_datasets[i].replace(args.path, '').replace('.ndjson', ''): results_list[i]
                #           for i in range(len(true_datasets))}

                # print(results)
                ## Generate results
                final_result, sub_final_result = table.add_entry(names[num], results)

                ## Save results as pkl (to avoid a new computation)
                os.makedirs(result_file)
                with open(result_file + '/results.pkl', 'wb') as handle:
                    pickle.dump([final_result, sub_final_result, col_result], handle, protocol=pickle.HIGHEST_PROTOCOL)

        return(table)
