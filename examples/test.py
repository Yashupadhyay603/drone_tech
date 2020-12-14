from amc import drone_init

drone=drone_init.start_drone()
for i in range(10000):
    drone.move(iterator=i,target_pos=[i*0.001,0,0.5],target_ori=[0,0,0])