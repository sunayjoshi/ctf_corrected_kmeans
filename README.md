First make a directory called data at the root of the repo. Then copy [ribosome\_images\_centered.npy](https://drive.google.com/file/d/1VBrdkhklVljOo7bGlonC6vC060ebBm0i/view?usp=sharing) and [ribosome\_angles\_centered.npy](https://drive.google.com/file/d/1ZBdXgjmj8VmDA3pxuc0YFjIA4tdf5R-u/view?usp=sharing) into data/ 

To run the Benchmark Algorithm (no CTF-correction), run
```
./make_plots_ctf_old
```

To run the Phase-flipped Benchmark Algorithm (no CTF-correction except initial phase flipping), run
```
./make_plots_ctf_old_pf
```

To run the Clean Centers Algorithm (CTF-correction with oracle initialization), run
```
./make_plots_ctf_clean
```

To run the Wiener Filtered Centers Algorithm (CTF-correction with Wiener Filtered initialization), run
```
./make_plots_ctf_wf
```

This code is based off of [wasserstein-k-means](https://github.com/4tywon/wasserstein-k-means/blob/master/README.md) and [ASPIRE-Python](https://github.com/ComputationalCryoEM/ASPIRE-Python).