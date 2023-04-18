# SwimFunction

This repository preprocesses videos, calculates recovery metrics, and predicts recovery outcomes as described in

Jensen et al., Functional Trajectories during Innate Spinal Cord Repair. bioRxiv doi: [https://doi.org/10.1101/2023.01.31.526502](https://doi.org/10.1101/2023.01.31.526502)

At the end of this document, there are two suggested ways to run the code: calculating and plotting everything, and calculating only rostral compensation with pose and behavior quality control plots only.

## Installation

This software has been used on Linux and Mac (macOS Catalina). We have not tested it on Windows.

Make sure python is installed and the command `conda` is accessible. Either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) should be fine. We used Miniconda.

1) Clone this repository

2) Create the virtual environment:

        conda env create --name swimfunction --file=swimfunction-env.yaml

3) Activate the virtual environment:

        conda activate swimfunction

4) Install the cloned repository via pip:

        pip install /path/to/SwimFunction

5) One plotting function (not required for metric calculation) requires both R and a forked version of the [sigclust2](https://github.com/pkimes/sigclust2) repository located [here](https://github.com/discolemur/sigclust2). You can install it in R by running this command in the R interpreter:

        devtools::install_github("discolemur/sigclust2")

Each time you use this package, be sure to activate the virtual environment: `conda activate swimfunction`

## Preparing the data

Take videos according to the protocol described in

Burris, B., Jensen, N., Mokalled, M. H. Assessment of Swim Endurance and Swim Behavior in Adult Zebrafish. J. Vis. Exp. (177), e63240, doi:10.3791/63240 (2021).

Each behavior assay should be 15 minutes long and have three periods of flow: 0 cm/s (0:00-5:00), 10 cm/s (5:00-10:00), and 20 cm/s (10:00-15:00).

Some video attributes can be set in the `config.ini` file, but be sure to update it with the correct frames per second (default is 70). All videos must be taken from the dorsal view with water flow originating from the bottom of the screen (so that the water flows from bottom to top).

For all frames, calculate eleven keypoints on the fish's dorsal centerline: ten points evenly spaced from nose to the base of the caudal fin, and one on the tip of the caudal fin. Posture annotation files are expected to be in the same format as the output of [DeepLabCut](https://deeplabcut.github.io/DeepLabCut/README.html) version 2.1.6.2 (csv format described below). SwimFunction accepts both .h5 or .csv files (each recorded assay only needs one or the other). Pose csv files should follow the following format (this example includes dummy values just to show an example):

        scorer,X,X,X,X,...,X,X,X,X
        bodyparts,p0,p0,p0,p1,...,p9,caudal_fin,caudal_fin,caudal_fin
        coords,x,y,likelihood,x,...,likelihood,x,y,likelihood
        0,25,132,1.0,32,...,1.0,38,145,1.0
        1,34,137,1.0,24,...,1.0,34,140,1.0

Body parts names do not actually matter as long as the columns are sorted from rostral tip (left) to caudal fin tip (right). The code will also accept files without the caudal fin tip (only ten keypoints), but in that case tail beat frequency and Strouhal number would be inferred from the base of the caudal fin instead, which could be unreliable.

Posture files can only have one fish per file. Please name all posture files (and videos, if you choose to import them) according to the rules described in the next section.

## Pose and video file name rules

Filenames should not have whitespace. All video and pose annotation files must follow the format below (with one exception):

        {anything here}_{assay label}_{fish labels separated by underscores}.{extension}

The {anything here} portion is where you can put miscellaneous information, like the fish stock number, experiment name, or researcher name. As long as the information does not match {assay label} as described below, any letters, numbers, or underscores are acceptable.

Assay label must match

        {number}wpi
        preinj
        pre-inj
        pretreatment
        posttreatment

Fish labels must be in the format

        {fish group letters}{fish number}{L or R (optional)}

or, in regex, a fish label matches the expression below

        ([A-Za-z]+)([0-9]+)([LRlr]{0,1})

where the first captured element is the group (think of it as a surname, like "Heterozygote"), the second is the fish number (think of it as a name, like "42"), and the optional third part is the side of the video where the fish is visible (left or right).

Internally, all characters are uppercase and all assay labels are handled as integers. Preinjury and pretreatment are converted to -1. Posttreatment is defined as 1. The "L or R" optional part is to designate whether the fish was on the left (L) or right (R) side of the original video.

Common troubleshooting: make sure nothing comes after the last fish name before the extension. Make sure group labels are letters only (numbers will be interpreted as fish names). Make sure all fish are listed in the file name.

### Exception

To accept DeepLabCut's output file names, we do allow DLC_RESNET to immediately follow the final fish name:

- some_description_3wpi_M23DLC_resnet50_bunch_of_stuff_here.h5
  - Parses as assay label 3, fish M23 (group M, number 23).

### Valid filename examples

- 3wpi_M23.avi
- 7-24-1998_26344_EKAB_1wpi_F3L_M3R.avi
- preinj_F15.avi
- Pre-inj_M23R_F43L.avi
- 19920215_23563_ntr_mtz_pretreatment_RPFW7.avi
- 2022_0819_23348_ekab_preinj_F3LDLC_resnet50_chuck_wagon_some_numbers2525324.h5
  - Parses as assay label -1, fish F3 (side L)

### Invalid filename examples

- bogus_M23_3wpi.avi
  - Assay label must come before fish name.
- bogus_4wpi_8wpi_M23.mp4
  - Only one assay label allowed.
- 3wpi_F23_bogus.csv
  - Nothing should come after the assay string and fish label strings.

If you absolutely *must* modify this source code instead of changing your file names, see `parse_details_from_filename(...)` in the file `swimfunction/data_access/data_utils.py`

## Setting up the experiment

The script `calculate_and_plot_everything.py` will guide the user through importing any necessary files (e.g., pose annotation files), and optional ones (e.g., precalculated metrics). After files are imported, it will proceed to process posture, predict behaviors, and calculate swim capacity and gait quality measurements. Then, it will produce plots similar to those presented in our manuscript.

## Directory structure

The experiment directory structure contains folders for **raw data** (e.g., pose files, videos, precalculated scores), **plots** (e.g., quality control, metric correlations, eigenfish), and **results** in csv format (e.g., calculated and imported recovery metrics and predicted outcomes, if applicable). It also contains folders with intermediate calculations including predicted behaviors and cruise waveform features.

Each experiment directory will look like this:

    experiment_root_directory
    .
    |-- behaviors_model
    |-- cruise_waveform
    |-- csv_results
    |-- plots
    |  |-- correlation_clustermaps
    |  |-- cruise_embedding
    |  |-- cruise_waveform
    |  `-- swim_capacity
    |-- precalculated
    |-- processed_pose_data
    |  |-- fish
    |  `-- pca_models
    |-- qc
    |  |-- behavior_qc
    |  `-- pose_quality_by_fish
    |-- raw_pose_annotations
    `-- videos
       `-- normalized

Most directories will still be created, even if they are unused or empty.

## Expected inputs

Minimally, the user must have posture files either in `.csv` format as described above, or `.h5` pandas DataFrames. These may be imported using the `calculate_and_plot_everything.py` script, or the user can manually place them in the experiment directory in a subfolder called `{path_to_experiment}/raw_pose_annotations`

Optionally, the user may also provide:

- Videos that produced the pose outputs. If provided, the code assumes the videos are grayscale and normalized. Place them in the experiment directory in a subdirectory `{path_to_experiment}/videos/normalized` There is also code to normalize RGB videos, see `swimfunction/video_processing/SplitCropTracker.py`. If you use the CropTracker to normalize your videos, be sure to include both the normalized video files and their associated CropTracker logfiles (extension: `.ctlog` or `.ctlog.gz`).
- A csv file defining the names of fish at the end of the experiment and whether they were predicted to recover well `{path_to_experiment}/outcome_prediction_final_name_key.csv`

### Integrating precalculated metrics

If any scores are precalculated, place a csv for each type of score in the "precalculated" subfolder of the experiment directory. Each file must be located and named as follows: `{path_to_experiment}/precalculated/{score_name}.csv`. The examples in this section have two fish, M7 and F23, and include assays labeled -1 (preinjury), 1, 2, 6, 8 (weeks post-injury). Each file must follow the format of the examples below, with a header that includes "fish" and a list of assay labels.

File defining scores that will be called "perceived_quality" in the outputs: **perceived_quality.csv**

        fish,-1,1,2,6,8
        M7,5,1,2,4,5
        F23,5,1,1,1,3
        ...

Missing values are acceptable. There is no restriction on assay labels. For example, since measures like glial bridging are gathered after fish are collected, they only exist at the final assay timepoint, so if the experiment ends at 8 weeks post-injury then the user would include the following file:

**glial_bridging.csv**

        fish,8
        M7,0.543
        F23,0.124
        M4,
        F11,0.23
        ...

CRITICAL: All glial bridging and axon regrowth scores are assumed to be percents between 0 and 1 (where 1 is 100%). If glial_bridging.csv, proximal_axon_regrowth.csv, or distal_axon_regrowth.csv include scores between 1 and 100 (where 100 is 100%), some plotting functions may misbehave.

The code expects, but does not require, the following precalculated score files. Blank files will be automatically created to remind the user that they are recommended. Expected, optional score files:

- endurance.csv
- perceived_quality.csv
- glial_bridging.csv
- proximal_axon_regrowth.csv
- distal_axon_regrowth.csv

## Expected outputs

All calculated functional metrics and precalculated metrics/scores are stored in `{path_to_experiment}/csv_results/calculated_metrics.csv`. A key mapping fish names to predicted outcomes, if the experiment is set up to do so, will be created as `{path_to_experiment}/csv_results/outcome_predictions.csv`. See the folders `{path_to_experiment}/plots` and `{path_to_experiment}/qc` for plots.

Metrics in `calculated_metrics.csv` include

- swim_distance
- activity
- time_against_flow_10_20cms
- centroid_burst_freq_0cms
- pose_burst_freq_0cms
- mean_y_10_20cms
- posture_novelty
- scoliosis
- rostral_compensation
- body_frequency
- tail_beat_freq
- tail_beat_freq_0cms
- strouhal_0cms

and any other scores provided in the `precalculated` subdirectory.

## Predicting recovery outcomes

Minimally, outcome prediction requires measurements at 1 and 2 wpi, where fish names are consistent between timepoints. It's up to the user to decide how identities will be tracked for multiple weeks. After the 2 wpi assay, ensure that `config.ini` has the experiment's name. Import all posture data into the experiment using `calculate_and_plot_everything.py` and it should detect that predictions can be calculated. It will output a file containing predictions: `{path_to_experiment}/csv_results/outcome_predictions.csv` which has two columns:

        fish,will_recover_best
        M12,False
        F5,True
        ...

At the end of the outcome prediction experiment, consider providing "outcome_prediction_final_name_key.csv" in the experiment root directory. This matches the identity of the fish at the final timepoint (its name on the pose annotation file) with the predicted outcome. It is not expected that fish will keep the same name after 2 wpi, for practical reasons. The key file tells the program what the original predictions were, but for the new fish names, and is therefore necessary for validation and plotting.

        fish,predicted_well
        M832,FALSE
        F865,TRUE
        ...

## Basic usage

### Calculate and plot everything

1) Create a directory located at /path/to/experiment_name

2) Copy `config-template.ini` to `config.ini` and update the following:

    - [EXPERIMENT DETAILS] names
        - List of all experiments in the cache root
    - [EXPERIMENT DETAILS] individuals_were_tracked
        - Parallel list of booleans, whether fish identities are consistent between assays (if a fish is named F23 in assay 3, is it the exact same fish as the one named F23 in assay 5? If so, put "true" in the list, otherwise, use "false")
    - [EXPERIMENT DETAILS] assay_labels
    - (optional) [EXPERIMENT DETAILS] experiment_name
        - Active experiment name
    - (optional) [FOLDERS] cache_root

3) Run the `calculate_and_plot_everything.py` script

        >> python calculate_and_plot_everything.py -r /path/to -e experiment_name -c /path/to/config.ini

Some functions allow parallel processing, and you can designate the number of processes you allow to run using the `-t {integer}` flag.

### Calculate rostral compensation only, plot pose and behavior QC only

In our paper, we introduced a gait quality score called "rostral compensation" which is highly correlated with neurological wellbeing. Because we want this score to be as accessible as possible, we provide a script called `calculate_rostral_compensation.py` which imports pose files, predicts behaviors, performs standard quality control on pose and behavior, and calculates rostral compensation scores. It does not calculate any other functional measurements and does not incorporate any precalculated scores. To use this script, follow the same steps in the section above, but instead of running `calculate_and_plot_everything.py` run `calculate_rostral_compensation.py` which has exactly the same commandline usage.
