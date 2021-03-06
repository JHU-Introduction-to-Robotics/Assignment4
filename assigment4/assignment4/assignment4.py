'''
####################
Sahil Sharma
10/10/21
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
####################
'''

import numpy as np

"""
Creates a 3x3 rotation matrix in 3D space from an axis and an angle.

Input
:param k: A 3 element array containing the unit axis to rotate around (x,y,z) 
:param q: The angle (in radians) to rotate by

Output
:return: A 3x3 element matix containing the rotation matrix
"""
def axis_angle_rot_matrix(k, q):

    kx, ky, kz = k[0], k[1], k[2]
    sin_theta = np.sin(q)
    cos_theta = np.cos(q)
    v_theta = 1 - cos_theta

    R = np.zeros((3, 3))

    # rotation matrix assembly with k and theta
    R[0, 0] = ((kx * kx) * v_theta) + cos_theta
    R[0, 1] = (kx * ky * v_theta) - (kz * sin_theta)
    R[0, 2] = (kx * kz * v_theta) + (ky * sin_theta)
    R[1, 0] = (kx * ky * v_theta) + (kz * sin_theta)
    R[1, 1] = (ky * ky * v_theta) + cos_theta
    R[1, 2] = (ky * kz * v_theta) - (kx * sin_theta)
    R[2, 0] = (kx * kz * v_theta) - (ky * sin_theta)
    R[2, 1] = (ky * kz * v_theta) + (kx * sin_theta)
    R[2, 2] = (kz * kz * v_theta) + cos_theta

    return R

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
def hr_matrix(k, t, q):

    R = axis_angle_rot_matrix(k, q)
    E = np.zeros((4, 4))

    # Adding in rotation matrix to the homogenous representation matrix
    for i in range(0, 3):
        for j in range(0, 3):
            E[i, j] = R[i, j]

    # Adding in translation matrix to the homogenous representation matrix
    for i in range(0, 3):
        E[i, 3] = t[i]
    E[3, 3] = 1

    return E

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
def inverse_hr_matrix(k_i, t_i, q_i):

    R = axis_angle_rot_matrix(k_i, q_i)
    R_tranpose = np.transpose(R)
    t_inv = np.matmul(np.multiply(-1, R_tranpose), t_i)

    E = np.zeros((4, 4))

    # Adding in rotation matrix transpose to the homogenous representation matrix
    for i in range(0, 3):
        for j in range(0, 3):
            E[i, j] = R_tranpose[i, j]

    # Adding in translation matrix to the homogenous representation matrix
    for i in range(0, 3):
        E[i, 3] = t_inv[i]
    E[3, 3] = 1

    return E

class KinematicChain:
    '''
    Creates a kinematic chain class for computing poses and velocities

    Input
    :param ks: A 2D array that lists the different axes of rotation (rows) for each joint
    :param ts: A 2D array that lists the translation from the previous joint to the current
    '''
    def __init__(self, ks, ts):

        self.k = np.array(ks)
        self.t = np.array(ts)
        assert ks.shape == ts.shape, 'Warning! Improper definition of rotation axes and translations'
        self.N_joints = ks.shape[0]

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
    def pose(self, Q, index=-1, p_i = [0, 0, 0]):

        # pad point with 1 to make it 4x1 vector
        p = np.zeros((4, 1))

        # copy values from p_i
        for i in range(0, 3):
            p[i, 0] = p_i[i]
        p[3, 0] = 1 # pad point with 1 to make it 4x1 vector

        E = np.identity(4) # forward declaring the homogenous matrix

        # check for valid index
        # calculate full forward kinematics by multiplying homogenous matrices together until index joint is reached
        if (index >= 0) and (index < self.N_joints):

            # calculate full kinematic chain by muliplying homogenous matrices together
            for i in range(0, (index + 1)):
                E_B_i = hr_matrix(self.k[i], self.t[i], Q[i])
                E = np.matmul(E, E_B_i)

        # calculate position of last joint in case where no index is provided
        else:

            # calculate full kinematic chain by muliplying homogenous matrices together
            for i in range(0, self.N_joints):
                E_B_i = hr_matrix(self.k[i], self.t[i], Q[i])
                E = np.matmul(E, E_B_i)

        p_res = np.matmul(E, p) # calculate new pose
        return [float(p_res[0]), float(p_res[1]), float(p_res[2])]

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
    def pseudo_inverse(self, theta_start, p_eff_N, x_end, max_steps = np.inf):

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

    '''
    Computes the Jacobian (position portions only, not orientation)

    :param Q: A N element array containing the current joint angles in radians
    :param p_eff_N: A 3 element vector containing translation from the last joint to the end effector in the last joints frame of reference

    Output
    :return: An 3xN 2d matrix containing the jacobian matrix
    '''
    def jacobian(self, Q, p_eff_N = [0, 0, 0]):

        Jv = np.zeros(((self.N_joints + 1), self.N_joints)) # forward declare Jacobian matrix which is 3 x N
        p_eff_0 = self.pose(Q, p_i = p_eff_N) # pose of end effector in global frame

        # calculate each column of the Jacobian
        for i in range(0, self.N_joints):
            p_i_0 = self.pose(Q, i) # pose of i_th joint in global frame
            p_eff_0_minus_p_i_0 = np.subtract(p_eff_0, p_i_0)
            k_i = self.k[i]
            Jv_i = np.cross(k_i, p_eff_0_minus_p_i_0)

            # copy jacobian column into Jacobian matrix
            for j in range(0, (self.N_joints + 1)):
                Jv[j, i] = Jv_i[j]

        return Jv

def main():

    # Some notes about my implementation:
    # Unfortunately no matter what I did/tried I could not get the simulation to converge
    # I am pretty confident I implemented the axis-angle rotation matrix correctly since that came from the lecture notes
    # I am also confident that I implemented the homogenous and inverse homogenous matrices correctly since those also
    # came from the lecture notes
    # Regarding the pose function I am also confident of my implementation since it follows the convention/logic laid out
    # in the lecture notes.
    # Regarding the jacobian I implemented it based on the examples that we went over in lecture but I suspect my calculation,
    # of the p^B_eff based on p^N_eff is wrong since my simulation did not converge. I was also confused by the fact that
    # function headers referred to p^N_eff as a translation and not a point in the last joints frame of reference.
    # Overall I am unsure quite what I did wrong here but I suspect there is a mistake in my Jacobian function.
    # If you could provide any feedback on what I missed or did wrong as part of the grading I would greatly appreciate it,
    # since I am honestly unsure what I missed and I feel like I have a solid grasp on the material for lecture 4.
    # Thanks so much!
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
