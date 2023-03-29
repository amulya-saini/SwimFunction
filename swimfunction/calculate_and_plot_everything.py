from swimfunction.main_scripts import metrics, plotting, qc, setup_structure
from swimfunction import FileLocations

def prompt_yes(question):
    res = '?'
    while res not in ['Y', 'N']:
        res = input(f'{question} (Y/N) ').strip().upper()
    return res == 'Y'

def main():
    ''' This main function will import and analyze posture data.

    1) Imports pose data
    2) Plots some quality control information about the poses
    3) Predicts behaviors
    4) Plots some quality control information about the predicted behaviors
    5) Calculates functional metrics, including importing
        any precalculated scores if provided (see README)
    6) Plots all functional metrics and any precalculated scores
    7) Predicts recovery outcomes if set up correctly (see README)
    8) Performs tensor decomposition if set up correctly (see README)
    '''
    print(main.__doc__)

    print(
        'All files will be saved in:\n\t',
        FileLocations.get_experiment_cache_root().as_posix())
    print('If the output directory looks incorrect, '
          'make sure you used the correct command '
          'line arguments and updated your config.ini file.\n')
    if prompt_yes('This may take several hours. Do you wish to continue?'):
        setup_structure.setup_structure()
        qc.qc_main()
        metrics.metrics_main()
        plotting.plotting_main()

if __name__ == '__main__':
    FileLocations.parse_default_args()
    main()
