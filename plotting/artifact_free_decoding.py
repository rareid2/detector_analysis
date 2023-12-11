import numpy as np
from scipy import signal

def add_and_limit_to_5(number1, number2):
    result = number1 + number2
    if result > 5:
        result = 5
    return result

# start by generating simple A
A = np.array([[0,0,0,0,0],[0,1,0,0,0],[0,0,0,0,0],
                [0,1,0,1,0],[0,0,0,0,0]])

# now do the auto correlation to find C
C = signal.correlate2d(A, A, mode='same')

center_i = 2
center_j = 2

crs_ind = []

for j in range(5): # go by row (aka j-star)
    for i in range(5): # go by column (aka k-star)
        # find sigma and tau (equal to i and j)
        sigma = (i - center_i)
        tau = -1*(j - center_j)

        # assign values to it
        phi = np.arctan2(tau,sigma)
        if phi < 0:
            phi += 2*np.pi
        crs_ind.append((j,i,sigma**2+tau**2,phi,sigma,tau))
sorted_list = sorted(crs_ind, key=lambda x: (x[2], x[3]))

# now we have coordinates in terms of sigma and tau
# create B
B  = np.full((5, 5), np.nan)
B[center_j,center_i] = 0
# first constrain to 0 based on holes
for j in range(5): # go by row (aka j-star)
    for i in range(5): # go by column (aka k-star)
        if A[j,i] == 1:
            B[j,i] = 0

B[3,1]=-1

# now go rthough loops for sigma and tau
R = np.sqrt(5)
alpha = 2
beta = 2

for si,stl in enumerate(sorted_list[1:]):
    sigma_s = stl[-2]
    tau_t = stl[-1]

    istar = sigma_s + center_i
    jstar = (-1*tau_t) + center_j
    c_val = C[jstar,istar]
    
    #if c_val == 1:
    #    print(sigma_s,tau_t,si)

    c_sum = 0

    # find where to constrain
    for jj in range(-2,3):
        for ii in range(-2,3):
            # convert back to indices
            #sigma = (i - center_i)
            #tau = -1*(j - center_j)

            i = ii + center_i
            j = (-1*jj) + center_j

            if sigma_s**2 + tau_t**2 <= R**2:
                new_i = ((ii+sigma_s) + center_i)
                new_j = ((-1*(jj+tau_t)) + center_j)

                # if A is 1 here
                if A[j,i] == 1:
                    if new_j > 4 or new_i > 4 or new_i < 0 or new_j < 0:
                        c_sum+=0
                    elif np.isnan(B[new_j, new_i]) and c_val==0:
                        B[new_j, new_i] = 0
                        c_sum+=0
                    elif np.isnan(B[new_j, new_i]) and c_val==1:
                        B[new_j, new_i] = -1
                        c_sum+=-1
                    else:
                        # already assigned
                        c_sum+=B[new_j, new_i]
    #if c_val != -1*c_sum:
    #    print(si)
    #    print(c_val,c_sum)
          
# corner value
B[0,4]=-1
