## Indian Buffet Process VAEs

This repository contains the code for the paper "Structured Variational Autoencoders for the Beta-Bernoulli Process", 
by Jeffrey Ling, Rachit Singh, and Finale Doshi-Velez. The paper is itself forthcoming on the ArXiv, but a workshop
version is available at the [AABI NIPS workshop page](http://approximateinference.org/2017/accepted/SinghEtAl2017.pdf).

This codebase is somewhat incomplete, missing some code that we haven't had time to clean up. We're working 
on putting all of the code up as soon as possible, and will do so over the next few weeks. Please let us know 
if you run into problems by making an issue.

We can also be contacted at [rachitsingh@college.harvard.edu](rachitsingh@college.harvard.edu), and [jeffreyling@alumni.harvard.edu](jeffreyling@alumni.harvard.edu), and feel free to send us an email.

# Quickstart

To run the code with GPU support, navigate to `src/lgamma` and do `./make.sh`.

Navigate to base directory and run

`python scripts/run_s_ibp_concrete.py --savefile testsave`

`--savefile` is a required argument. Models will be saved under `models/` (last, and best over epochs) and train, valid, test curves and timings are saved under `runs/`.

Use `--help` to see arguments.

# Note about compilation

We're working on incorporating all of the required CUDA functionality in PyTorch master, but until then the compilation process is likely a little tricky. For example, you'll almost definitely need to change `sm_35` in `make.sh` to your card's compute capability and the `CUDA_PATH` must be customized. Finally, one likely change you'll need to make is to fix `build.py` to have the correct `cuda_path` and `include_dirs`. Please feel free to send me an email if you have trouble compiling - I'm happy to help.
