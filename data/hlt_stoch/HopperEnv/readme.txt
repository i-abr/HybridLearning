HopperEnv:
    max_frames: 50000
    max_steps: 1000
    frame_skip: 1
    lam: 0.2
#     horizon: 10
    model_lr: !!float 3e-3
    policy_lr: !!float 3e-3
    value_lr: !!float 3e-4
    soft_q_lr: !!float 3e-4
    