1. use LDC1-4_MLE.py to obtain the maximum likelihood estimates (MLE) across the full frequency band
1.2. use merge_signal_files.py to merge the output MLE files of multiple runs if the search for MLE is one in parallel
2. (optional) use LDC1-4_posterior_GPU.py to compute the posterior of each found MLE using the GPU 
3. use LDC1-4_evaluate.py to match the found MLE with the injected signals
4. use LDC1-4_evaluate_matches.py to evalute the quality of the matches and create plots