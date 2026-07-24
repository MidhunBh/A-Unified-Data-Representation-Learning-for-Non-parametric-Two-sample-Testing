for name, samp in [
    ("mmd_d_HDGM_type1.py", "sample_hdgm_semi_t1"),
    ("c2st_semi_HDGM_type1.py", "sample_hdgm_semi_t1"),
    ("rl_mmd_d_HDGM_type1.py", "sample_hdgm_semi_t1"),
]:
    src = open(name).read()
    print(f"{name}:")
    print(f"  uses t1 sampler: {samp in src}")
    print(f"  no t2 sampler: {'semi_t2' not in src}")
    print(f"  type1 in output: {'type1' in src}")
