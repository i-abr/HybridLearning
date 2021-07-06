default:
    method: "hlt_stoch"
    max_steps: 200
    max_frames: 10000
    frame_skip: 2
    model_lr: !!float 3e-3
    policy_lr: !!float 3e-3
    value_lr: !!float 3e-4
    soft_q_lr: !!float 3e-4
    reward_scale: 1.0
    horizon: 5
    model_iter: 5
    trajectory_samples: 20
    lam: 0.1
    activation_fun: 'ReLU'

PendulumEnv:
    activation_fun: 'sin'
    max_frames: 10000
    frame_skip: 4
    max_steps: 200
    lam: 0.1
    trajectory_samples: 60