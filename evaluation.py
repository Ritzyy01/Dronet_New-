import gflags
import numpy as np
import os
import sys
import glob
from random import randint
from sklearn import metrics
import itertools # Added for matrix plotting iteration

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    # Set backend to 'Agg' to avoid display issues on servers
    # This must be done *before* importing pyplot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib imported successfully. Plots will be generated.")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not found. Plots will not be generated. Install with 'pip install matplotlib'")


from keras import backend as K

import utils
from constants import TEST_PHASE
from common_flags import FLAGS


# Functions to evaluate steering prediction

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y].
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary


def compute_explained_variance(predictions, real_values):
    """
    Computes the explained variance of prediction for each
    steering and the average of them
    """
    assert np.all(predictions.shape == real_values.shape)
    ex_variance = explained_variance_1d(predictions, real_values)
    print("EVA = {}".format(ex_variance))
    return ex_variance


def compute_sq_residuals(predictions, real_values):
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    sr = np.mean(sq_res, axis = -1)
    print("MSE = {}".format(sr))
    return sq_res


def compute_rmse(predictions, real_values):
    assert np.all(predictions.shape == real_values.shape)
    mse = np.mean(np.square(predictions - real_values))
    rmse = np.sqrt(mse)
    print("RMSE = {}".format(rmse))
    return rmse


def compute_highest_regression_errors(predictions, real_values, n_errors=20):
    """
    Compute the indexes with highest error
    """
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    highest_errors = sq_res.argsort()[-n_errors:][::-1]
    return highest_errors


def random_regression_baseline(real_values):
    mean = np.mean(real_values)
    std = np.std(real_values)
    return np.random.normal(loc=mean, scale=abs(std), size=real_values.shape)


def constant_baseline(real_values):
    mean = np.mean(real_values)
    return mean * np.ones_like(real_values)


def evaluate_regression(predictions, real_values, fname):
    evas = compute_explained_variance(predictions, real_values)
    rmse = compute_rmse(predictions, real_values)
    highest_errors = compute_highest_regression_errors(predictions, real_values,
            n_errors=20)
    dictionary = {"evas": evas.tolist(), "rmse": rmse.tolist(),
                  "highest_errors": highest_errors.tolist()}
    utils.write_to_file(dictionary, fname)


# Functions to evaluate collision

def read_training_labels(file_name):
    labels = []
    try:
        labels = np.loadtxt(file_name, usecols=0)
        labels = np.array(labels)
    except:
        print("File {} failed loading labels".format(file_name)) 
    return labels


def count_samples_per_class(train_dir):
    # Changed to os.path.join for Windows compatibility
    experiments = glob.glob(os.path.join(train_dir, "*"))
    num_class0 = 0
    num_class1 = 0
    for exp in experiments:
        file_name = os.path.join(exp, "labels.txt")
        try:
            labels = np.loadtxt(file_name, usecols=0)
            # Handle case where labels file might be empty or malformed
            if labels.size > 0:
                # If labels is a scalar (single file entry), make it iterable
                if labels.ndim == 0:
                    labels = np.array([labels])
                num_class1 += np.sum(labels == 1)
                num_class0 += np.sum(labels == 0)
        except:
            print("File {} failed loading labels".format(file_name)) 
            continue
    return np.array([num_class0, num_class1])


def random_classification_baseline(real_values):
    """
    Randomly assigns half of the labels to class 0, and the other half to class 1
    """
    return [randint(0,1) for p in range(real_values.shape[0])]


def weighted_baseline(real_values, samples_per_class):
    """
    Let x be the fraction of instances labeled as 0, and (1-x) the fraction of
    instances labeled as 1, a weighted classifier randomly assigns x% of the
    labels to class 0, and the remaining (1-x)% to class 1.
    """
    # Fix for division by zero if no training samples are found
    total = np.sum(samples_per_class)
    if total == 0:
        print("Warning: No training samples found for baseline calculation. Defaulting to uniform weights.")
        weights = [0.5, 0.5]
    else:
        weights = samples_per_class/total
        
    return np.random.choice(2, real_values.shape[0], p=weights)


def majority_class_baseline(real_values, samples_per_class):
    """
    Classify all test data as the most common label
    """
    # Fix for empty training data
    if np.sum(samples_per_class) == 0:
        print("Warning: No training samples found. Defaulting majority class to 0.")
        major_class = 0
    else:
        major_class = np.argmax(samples_per_class)
        
    return [major_class for i in real_values]

            
def compute_highest_classification_errors(predictions, real_values, n_errors=20):
    """
    Compute the indexes with highest error
    """
    assert np.all(predictions.shape == real_values.shape)
    dist = abs(predictions - real_values)
    highest_errors = dist.argsort()[-n_errors:][::-1]
    return highest_errors


def plot_pr_curve(real_labels, pred_prob, fname):
    """
    Calculates and plots the Precision-Recall curve.
    Saves the plot to a file.
    """
    if not MATPLOTLIB_AVAILABLE:
        return  # Skip if matplotlib wasn't imported
        
    try:
        precision, recall, _ = metrics.precision_recall_curve(real_labels, pred_prob)
        pr_auc = metrics.auc(recall, precision)

        plt.figure()
        plt.plot(recall, precision, color='darkorange',
                 lw=2, label='PR curve (area = %0.2f)' % pr_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Fraction of real crashes caught)')
        plt.ylabel('Precision (Fraction of crash predictions that were real)')
        plt.title('Precision-Recall Curve for Collision')
        plt.legend(loc="lower left")
        
        # Derive plot filename from json filename
        plot_fname = os.path.splitext(fname)[0] + '_pr_curve.png'
        plt.savefig(plot_fname)
        print(f"PR curve plot saved to {plot_fname}")
        plt.close()  # Close the figure to free up memory

    except Exception as e:
        print(f"Error: Could not plot or save PR curve: {e}")


def plot_confusion_matrix(real_labels, pred_labels, fname):
    """
    Generates and saves a visual Confusion Matrix using Matplotlib.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    try:
        cm = metrics.confusion_matrix(real_labels, pred_labels)
        
        # Create title based on filename (e.g., remove .json and directory)
        base_name = os.path.basename(fname).replace('.json', '')
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        title = f'Confusion Matrix for {base_name}'
        plt.title(title, fontsize=14, pad=20)
        
        plt.colorbar()
        
        # Set tick positions
        tick_marks = np.arange(2)
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)

        # Add descriptive text annotations to each quadrant
        # Top-left: True Negatives (Actual Safe, Predicted Safe)
        plt.text(0, 0, f'True Neg (Safe)\n{cm[0, 0]}',
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[0, 0] > cm.max() / 2. else "black",
                 fontsize=11)
        
        # Top-right: False Positives (Actual Safe, Predicted Crash)
        plt.text(1, 0, f'False Pos (Safe->Crash)\n{cm[0, 1]}',
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[0, 1] > cm.max() / 2. else "black",
                 fontsize=11)
        
        # Bottom-left: False Negatives (Actual Crash, Predicted Safe)
        plt.text(0, 1, f'False Neg (Crash->Safe)\n{cm[1, 0]}',
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[1, 0] > cm.max() / 2. else "black",
                 fontsize=11)
        
        # Bottom-right: True Positives (Actual Crash, Predicted Crash)
        plt.text(1, 1, f'True Pos (Crash)\n{cm[1, 1]}',
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[1, 1] > cm.max() / 2. else "black",
                 fontsize=11)

        plt.ylabel('Actual Safe                                        Actual Crash', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add more space around the plot
        plt.tight_layout()

        # Save plot
        plot_fname = os.path.splitext(fname)[0] + '_cm.png'
        plt.savefig(plot_fname, dpi=150, bbox_inches='tight')
        print(f"Confusion Matrix plot saved to {plot_fname}")
        plt.close()
    
    except Exception as e:
        print(f"Error plotting confusion matrix for {fname}: {e}")


def evaluate_classification(pred_prob, pred_labels, real_labels, fname):
    # Standard Metrics
    ave_accuracy = metrics.accuracy_score(real_labels, pred_labels)
    print('Average accuracy = ', ave_accuracy)
    precision = metrics.precision_score(real_labels, pred_labels)
    print('Precision = ', precision)
    recall = metrics.recall_score(real_labels, pred_labels)
    print('Recall = ', recall)
    f_score = metrics.f1_score(real_labels, pred_labels)
    print('F1-score = ', f_score)
    
    # --- Confusion Matrix Logic ---
    # 0 = No Collision (Safe), 1 = Collision
    try:
        cm = metrics.confusion_matrix(real_labels, pred_labels)
        
        # Handle cases where the matrix might not be 2x2
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print('\n--- CONFUSION MATRIX for {} ---'.format(fname))
            print('True Negatives (Actual Safe, Pred Safe): {}'.format(tn))
            print('False Positives (Actual Safe, Pred CRASH): {}'.format(fp))
            print('False Negatives (Actual CRASH, Pred Safe): {}  <-- CRITICAL RISK'.format(fn))
            print('True Positives (Actual CRASH, Pred CRASH): {}'.format(tp))
            print('----------------------------------------\n')
        else:
            print('\nConfusion Matrix (Shape {}):\n{}\n'.format(cm.shape, cm))
            
    except Exception as e:
        print("Could not print detailed confusion matrix: {}".format(e))

    # --- NEW: Generate Visual Confusion Matrix ---
    # This will run for all calls (Model, Random, Weighted, Majority)
    plot_confusion_matrix(real_labels, pred_labels, fname)
    # ---------------------------------------------
    
    # --- Plot Precision-Recall Curve ---
    # We only want to plot the PR curve for the *actual model test*
    if 'test_classification' in fname:
        plot_pr_curve(real_labels, pred_prob, fname)

    highest_errors = compute_highest_classification_errors(pred_prob, real_labels,
            n_errors=20)
    
    dictionary = {"ave_accuracy": ave_accuracy.tolist(), "precision": precision.tolist(),
                  "recall": recall.tolist(), "f_score": f_score.tolist(),
                  "highest_errors": highest_errors.tolist()}
    utils.write_to_file(dictionary, fname)



def _main():

    # Set testing mode (dropout/batchnormalization)
    K.set_learning_phase(TEST_PHASE)

    # Generate testing data
    test_datagen = utils.DroneDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(FLAGS.test_dir,
                          shuffle=False,
                          color_mode=FLAGS.img_mode,
                          target_size=(FLAGS.img_width, FLAGS.img_height),
                          crop_size=(FLAGS.crop_img_height, FLAGS.crop_img_width),
                          batch_size = FLAGS.batch_size)

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    model = utils.jsonToModel(json_model_path)

    # Load weights
    weights_load_path = FLAGS.experiment_rootdir + "/" +FLAGS.weights_fname
    print(weights_load_path)
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except Exception as e:
        print(e)
        print("Impossible to find weight path. Returning untrained model")


    # Compile model
    model.compile(loss='mse', optimizer='adam')

    # Get predictions and ground truth
    n_samples = test_generator.samples
    nb_batches = int(np.ceil(n_samples / FLAGS.batch_size))

    predictions, ground_truth, t = utils.compute_predictions_and_gt(
            model, test_generator, nb_batches, verbose = 1)

    # Param t. t=1 steering, t=0 collision
    t_mask = t==1

    # ************************* Steering evaluation ***************************
    
    # If there are steering samples (t==1) evaluate them; otherwise skip.
    if np.any(t_mask):
        # Predicted and real steerings
        pred_steerings = predictions[t_mask,0]
        real_steerings = ground_truth[t_mask,0]

        # Compute random and constant baselines for steerings
        random_steerings = random_regression_baseline(real_steerings)
        constant_steerings = constant_baseline(real_steerings)

        # Create dictionary with filenames
        dict_fname = {'test_regression.json': pred_steerings,
                      'random_regression.json': random_steerings,
                      'constant_regression.json': constant_steerings}

        # Evaluate predictions: EVA, residuals, and highest errors
        for fname, pred in dict_fname.items():
            abs_fname = os.path.join(FLAGS.experiment_rootdir, fname)
            evaluate_regression(pred, real_steerings, abs_fname)

        # Write predicted and real steerings
        dict_test = {'pred_steerings': pred_steerings.tolist(),
                     'real_steerings': real_steerings.tolist()}
        utils.write_to_file(dict_test,os.path.join(FLAGS.experiment_rootdir,
                                                   'predicted_and_real_steerings.json'))
    else:
        print("Evaluating steering samples... Done.")


    # *********************** Collision evaluation ****************************
    
    # Predicted probabilities and real labels
    # Handle cases where predictions/ground_truth may have a single column
    if predictions.ndim == 1:
        pred_prob = predictions[~t_mask]
    elif predictions.shape[1] == 1:
        pred_prob = predictions[~t_mask, 0]
    else:
        pred_prob = predictions[~t_mask, 1]

    pred_labels = np.zeros_like(pred_prob)
    pred_labels[pred_prob >= 0.5] = 1

    if ground_truth.ndim == 1:
        real_labels = ground_truth[~t_mask]
    elif ground_truth.shape[1] == 1:
        real_labels = ground_truth[~t_mask, 0]
    else:
        real_labels = ground_truth[~t_mask, 1]

    # --- Compute all baselines ---
    random_labels = random_classification_baseline(real_labels)
    
    print("Counting samples per class in training set for baselines...")
    samples_per_class = count_samples_per_class(FLAGS.train_dir)
    
    weighted_labels = weighted_baseline(real_labels, samples_per_class)
    majority_labels = majority_class_baseline(real_labels, samples_per_class)
    
    # --- Create dictionary with ALL filenames ---
    # The loop below will pass these to evaluate_classification, 
    # which will now generate confusion matrix plots for each.
    dict_fname = {'test_classification.json': pred_labels,
                  'random_classification.json': random_labels,
                  'weighted_classification.json': weighted_labels,
                  'majority_classification.json': majority_labels}
    
    # Evaluate predictions: accuracy, precision, recall, F1-score, and highest errors
    for fname, pred in dict_fname.items():
        abs_fname = os.path.join(FLAGS.experiment_rootdir, fname)
        evaluate_classification(pred_prob, pred, real_labels, abs_fname)

    # Write predicted probabilities and real labels
    dict_test = {'pred_probabilities': pred_prob.tolist(),
                 'real_labels': real_labels.tolist()}
    utils.write_to_file(dict_test,os.path.join(FLAGS.experiment_rootdir,
                                               'predicted_and_real_labels.json'))


def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)