import numpy as np

N = 50
mass = 1.89 # kg
I = 0.00189 # kg*m^2
link = 0.31 # m


def inverseKin(pos_ee):
        theta = []
        for i in range(pos_ee.shape[0]):
            if pos_ee[i,0] == 0.0: 
                theta.append(np.pi/2)
            else:
                theta.append(np.arctan(pos_ee[i,1]/pos_ee[i,0]))
        return np.array([theta])
        
def inverseDyn(pos,vel,acc, I, mass, link):
        torques = I * acc + g * mass * link/2 *np.sin(pos)
        return torques



def minimumJerk(x_init, x_des, timespan):
    T_max = timespan[ len(timespan)-1 ]
    tmspn = timespan.reshape(timespan.size,1)

    a =   6*(x_des-x_init)/np.power(T_max,5)
    b = -15*(x_des-x_init)/np.power(T_max,4)
    c =  10*(x_des-x_init)/np.power(T_max,3)
    d =  np.zeros(x_init.shape)
    e =  np.zeros(x_init.shape)
    g =  x_init

    pol = np.array([a,b,c,d,e,g])
    pp  = a*np.power(tmspn,5) + b*np.power(tmspn,4) + c*np.power(tmspn,3) + g

    return pp, pol
    
# Double derivative of the trajectory
def minimumJerk_ddt(x_init, x_des, timespan):
    T_max = timespan[ len(timespan)-1 ]
    tmspn = timespan.reshape(timespan.size,1)

    a =  120*(x_des-x_init)/np.power(T_max,5)
    b = -180*(x_des-x_init)/np.power(T_max,4)
    c =  60*(x_des-x_init)/np.power(T_max,3)
    d =  np.zeros(x_init.shape)

    pol = np.array([a,b,c,d])
    pp  = a*np.power(tmspn,3) + b*np.power(tmspn,2) + c*np.power(tmspn,1) + d
    ('pp: ', len(pp))
    return pp, pol
    
# Time and value of the minimum jerk curve
def minJerk_ddt_minmax(x_init, x_des, timespan):

    T_max   = timespan[ len(timespan)-1 ]
    t1      = T_max/2 - T_max/720 * np.sqrt(43200)
    t2      = T_max/2 + T_max/720 * np.sqrt(43200)
    pp, pol = minimumJerk_ddt(x_init, x_des, timespan)

    ext    = np.empty(shape=(2,x_init.size))
    ext[:] = 0.0
    t      = np.empty(shape=(2,x_init.size))
    t[:]   = 0.0

    for i in range(x_init.size):
        if (x_init[i]!=x_des[i]):
            tmp      = np.polyval( pol[:,i],[t1,t2] )
            ext[:,i] = np.reshape( tmp,(1,2) )
            t[:,i]   = np.reshape( [t1,t2],(1,2) )
    
    return t, ext
    
# Compute the torques via inverse dynamics
def generateMotorCommands(init_pos, des_pos, time_vector):
    # Last simulation time
    T_max = time_vector[ len(time_vector)-1 ]
    # Time and value of the minimum jerk curve
    ext_t, ext_val = minJerk_ddt_minmax(init_pos, des_pos, time_vector)

    # Define njt based on the context of your data
    njt = ext_val.shape[1]  # Assuming ext_val is a 2D array and njt is the number of columns

    # Approximate with sin function
    tmp_ext = np.reshape( ext_val[0,:], (1,njt) ) # First extreme (positive)
    tmp_sin = np.sin( (2*np.pi*time_vector/T_max) )
    mp_sin = np.reshape( tmp_sin,(tmp_sin.size,1) )

    # Motor commands: Inverse dynamics given approximated acceleration
    dt   = (time_vector[1]-time_vector[0])/1e3
    pos,pol  = minimumJerk(init_pos, des_pos, time_vector)
    vel  = np.gradient(pos,dt,axis=0)
    acc  = tmp_ext*tmp_sin
    mcmd = dynSys.inverseDyn(pos, vel, acc)
    return mcmd[0]


