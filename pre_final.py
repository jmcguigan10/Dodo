import h5py
import numpy as np

with h5py.File("processed_emu.h5", "w") as f_out:
    N = total_num_samples
    ds_F_init = f_out.create_dataset("F_init", (N, 6, 4), dtype="f4")
    ds_F_true = f_out.create_dataset("F_true", (N, 6, 4), dtype="f4")
    ds_F_box = f_out.create_dataset("F_box", (N, 6, 4), dtype="f4")
    ds_resid = f_out.create_dataset("residual", (N, 24), dtype="f4")
    ds_inv = f_out.create_dataset("invariants", (N, 27), dtype="f4")
    # plus sim_id as a variable-length string dataset if needed

    for k in range(N):
        F_init, F_true, u4, sim_id = load_raw_emu_sample(k)
        F_box = box3d_closure(F_init, matter_fields_for_k, params)

        DeltaF = F_true - F_box
        ds_F_init[k] = F_init
        ds_F_true[k] = F_true
        ds_F_box[k] = F_box
        ds_resid[k] = DeltaF.reshape(-1)
        ds_inv[k] = compute_invariants(F_init, u4)
        # store sim_id separately
