from flownet import *

if __name__=='__main__':
    batch_size=3

    height = 160; width = 160;
    model = getModel(height=height,width=width,batch_size=batch_size)
    left_image = np.random.standard_normal((batch_size,height,width,3));
    right_image = np.random.standard_normal((batch_size,height,width,3));
    imu = np.random.standard_normal((batch_size,10,6))
    foo = model.predict_on_batch(x = [left_image,right_image,imu])
    positions = foo[-1]
    delta_poses = []
    for i in range(len(positions)-1)                            
        c = positions[i]                                       
        nextC = positions[i+1]
        delta = np.dot(nextC , np.linalg.inv(c))
        delta_poses.append(c)

