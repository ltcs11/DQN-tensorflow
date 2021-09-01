# DQN-tensorflow with latest pkg

## init date
- 2021.08.30

## adapted for 
- latest gym(0.19.0) + atari-py(0.3.0)
- tensorflow 1.13.x
- python3.7+

## change
- Breakout-v0 -> BreakoutNoFrameskip-v0
- action_repeat = 1->4 in 'M1'
- auto adapt for GPU usage
- other api and syntax change
- add docker scripts for GPU training env (cuda9 + tf-gpu1.10)

**mostly referenced from origin repo issue**