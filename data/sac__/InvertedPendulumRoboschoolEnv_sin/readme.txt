default:
    method: "sac"
    max_steps: 200
    max_frames: 10000
    frame_skip: 2
    policy_lr: !!float 3e-3
    value_lr: !!float 3e-4
    soft_q_lr: !!float 3e-4
    reward_scale: 1.0
    activation_fun: 'ReLU'
    
InvertedPendulumRoboschoolEnv:
    activation_fun: 'sin'
    max_frames: 50000
    frame_skip: 4
    max_steps: 1000