def cnt_lr_scheduler(cfg):
    lr_scale = 0.9
    START_LR = cfg.learning_rate
    lr_schedule = [(i, START_LR * (lr_scale ** i)) for i in range(100)]

    lr_schedule_big = [(0, min(START_LR, START_LR)),
                       (30, START_LR * (lr_scale ** 1)),
                       (60, START_LR * (lr_scale ** 2)),
                       (90, START_LR * (lr_scale ** 3)),
                       (100, START_LR * (lr_scale ** 4))
                       ]

    end_learning_rate = 1e-7
    power = 1
    start_decay = max(0, cfg.epoch - 100)
    plr_schedule = [(i, (START_LR - end_learning_rate) * (
                (1 - (i - start_decay) / (cfg.epoch - start_decay)) ** power) + end_learning_rate)
                    for i in range(start_decay, cfg.epoch)]

    return plr_schedule

def cls_lr_scheduler(cfg):
    lr_scale = 0.9
    end_learning_rate = 1e-9
    power = 2
    START_LR = cfg.learning_rate
    # (learning_rate - end_learning_rate) *(1 - global_step / decay_steps) ^ (power) +end_learning_rate

    lr_schedule = [(i, max(0.0003, START_LR * (lr_scale ** i))) for i in range(cfg.epoch)]

    plr_schedule = [(i,
                     (START_LR - end_learning_rate) * ((1 - i / (cfg.epoch)) ** power) + end_learning_rate)
                    for i in range(cfg.epoch)]
    # plr_schedule.extend([(self.config.epoch-49,1e-2)])
    # plr_schedule.extend([(251,1e-1),(252,1e-1)])
    # plr_schedule.extend([(301,5e-1),(302,5e-1)])


    elr_schedule = [(i, START_LR * (0.96 ** i)) for i in range(cfg.epoch)]

    step_schedule = [(0, START_LR), (50, START_LR / 10),
                     (100, START_LR / 100), (150, START_LR / 1000),
                     (200, START_LR / 10000), (250, START_LR / 100000)
                     ]

    step_stride = 30
    step_schedule = [(step_side, START_LR / (10 ** pow))
                     for pow, step_side in
                     zip(range(0, cfg.epoch // step_stride, 1), range(0, cfg.epoch, step_stride))]
    # print(step_schedule)
    print(plr_schedule)
    return plr_schedule