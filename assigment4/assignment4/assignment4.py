
''' ####################
    EN605.613 - Introduction to Robotics
    Assignment 4
    Kinematic Chains
    -------
    For this assignment you must implement the following functions
    1. axis_angle_rot_matrix
    2. hr_matrix
    3. inverse_hr_matrix
    4. KinematicChain.pose
    5. KinematicChain.jacobian
    ==================================
    Copyright 2020,
    The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).
    All Rights Reserved.
    #################### '''

import numpy as np

def axis_angle_rot_matrix(k,q):
    """
    Creates a 3x3 rotation matrix in 3D space from an axis and an angle.

    Input
    :param k: A 3 element array containing the unit axis to rotate around (x,y,z) 
    :param k: The angle (in radians) to rotate by

    Output
    :return: A 3x3 element matix containing the rotation matrix

    """

    return None

def hr_matrix(k,t,q):
    '''
    Create the Homogenous Representaiton matrix that transforms a point from Frame B to Frame A.
    Using the axis-angle representation
    Input
    :param k: A 3 element array containing the unit axis to rotate around (x,y,z) 
    :param t: The translation from the current frame to the next frame
    :param q: The rotation angle (i.e. joint angle)

    Output
    :return: A 4x4 Homogenous representation matrix
    '''
    
    return None


def inverse_hr_matrix(k_i,t_i,q_i):
    '''
    Create the Inverse Homogenous Representaiton matrix that transforms a point from Frame A to Frame B.
    Using using the axis-angle representation
    Input
    :param k: A 3 element array containing the unit axis to rotate around (x,y,z) 
    :param t: The translation from the current frame to the next frame
    :param q: The rotation angle (i.e. joint angle)

    Output
    :return: A 4x4 Inverse Homogenous representation matrix
    '''
    return None

class KinematicChain:
    def __init__(self,ks,ts):
        '''
        Creates a kinematic chain class for computing poses and velocities

        Input
        :param ks: A 2D array that lists the different axes of rotation (rows) for each joint
        :param tt: A 2D array that lists the translation from the previous joint to the current

        '''
        self.k = np.array(ks)
        self.t = np.array(ts)
        assert ks.shape == ts.shape, 'Warning! Improper definition of rotation axes and translations'
        self.N_joints = ks.shape[0]

    def pose(self,Q,index=-1,p_i=[0,0,0]):
        '''
        Compute the pose in the global frame of a point given in a joint frame
        (default values will compute the position of the last joint)
        Input
        :param Q: A N element array containing the joint angles in radians
        :param p_i: A 3 element vector containing a position in the frame of the index joint
        :param index: The index of the joint frame being converted from (first joint is 0, the last joint is N_joints)

        Output
        :return: A 3 element vector containing the new pose with respect to the global frame
        '''
        return None 

    def pseudo_inverse(self,theta_start,p_eff_N,x_end,max_steps=np.inf):
        '''
        Performs the inverse_kinematics using the pseudo-jacobian

        :param theta_start: A N element array containing the current joint angles in radians
        :param p_eff_N: A 3 element vector containing translation from the last joint to the end effector in the last joints frame of reference
        :param xend: A 3 element vector containing the desired end pose for the end effector in the base frame
        :param max_steps: (Optional) Maximum number of iterations to compute 
        (Note: If it takes more than 200 iterations it is something the computation went wrong)

        Output
        :return: An N element vector containing the joint angles that result in the end effector reaching xend
        '''
        v_step_size = 0.05
        theta_max_step = 0.2
        Q_j = theta_start
        p_end = np.array([x_end[0],x_end[1],x_end[2]])
        p_j = self.pose(Q_j,p_i=p_eff_N)
        delta_p = p_j - p_end
        j=0
        while np.linalg.norm(delta_p) > 0.01 and j<max_steps:
            print(f'j{j}: Q[{Q_j}] , P[{p_j}]')
            v_p = delta_p * v_step_size / np.linalg.norm(delta_p)

            J_j = self.jacobian(Q_j,p_eff_N)
            J_invj = np.linalg.pinv(J_j)

            v_Q = np.matmul(J_invj,v_p)

            Q_j = Q_j +np.clip(v_Q,-1*theta_max_step,theta_max_step)

            p_j = self.pose(Q_j,p_i=p_eff_N)
            j+=1
            delta_p = p_j - p_end

        return Q_j


    def jacobian(self,Q,p_eff_N=[0,0,0]):
        '''
        Computes the Jacobian (position portions only, not orientation)

        :param Q: A N element array containing the current joint angles in radians
        :param p_eff_N: A 3 element vector containing translation from the last joint to the end effector in the last joints frame of reference

        Output
        :return: An 3xN 2d matrix containing the jacobian matrix
        '''
        return None

def main():

    ks = np.array([[0,0,1],[0,0,1]])
    ts = np.array([[0,0,0.5],[1,0,0]])
    p_eff_2 = [1,0,0]
    kc = KinematicChain(ks,ts)

    q_0 = np.array([np.pi/2,np.pi/2])
    x_1 = np.array([1,0.5,0.5])

    for i in np.arange(0,kc.N_joints):
        print(f'joint {i} pose = {kc.pose(q_0,index=i)}')
    print(f'end_effector = {kc.pose(q_0,index=-1,p_i=p_eff_2)}')

    kc.pseudo_inverse(q_0, p_eff_N=p_eff_2, x_end=x_1, max_steps=100)

if __name__ == '__main__':
    main()

