# Plotting Laser Populations and Fields 
import numpy as np
import matplotlib.pyplot as plt

Population = np.array(np.fromfile("pop_dynamical.out", dtype = np.double)).reshape(10, 100000)

field = np.array(np.fromfile("field.out", dtype = np.double))

truncate = 2000
t = np.linspace(0., truncate/1000., truncate)
t_field = np.linspace (0., 1000, 1000)

plt.figure()
plt.subplot(221)
plt.plot(t, Population[0][:truncate], 'b', label='R1')
plt.plot(t, Population[1][:truncate], 'r', label='R2')
plt.plot(t, Population[2][:truncate], 'g', label='R3')
plt.plot(t, Population[3][:truncate], 'k', label='R4')
plt.plot(t, Population[4][:truncate], 'y', label='R5')
plt.xlabel('time (in ps)')
plt.ylabel('population of Pr states')
plt.legend(loc = 1)

plt.subplot(223)
plt.plot(t, Population[5][:truncate], 'b', label='FR1')
plt.plot(t, Population[6][:truncate], 'r', label='FR2')
plt.plot(t, Population[7][:truncate], 'g', label='FR3')
plt.plot(t, Population[8][:truncate], 'y', label='FR4')
plt.plot(t, Population[9][:truncate], 'k', label='FR5')
plt.xlabel('time (in ps)')
plt.ylabel('population of Pfr states')
plt.legend(loc = 1)

t1 = np.linspace(0., 100., 100000)

plt.subplot(222)
plt.plot(t1, Population[0], 'b', label='R1')
plt.plot(t1, Population[1], 'r', label='R2')
plt.plot(t1, Population[2], 'g', label='R3')
plt.plot(t1, Population[3], 'k', label='R4')
plt.plot(t1, Population[4], 'y', label='R5')
plt.xlabel('time (in ps)')
plt.ylabel('population of Pr states')
plt.legend(loc = 1)

plt.subplot(224)
plt.plot(t1, Population[5], 'b', label='FR1')
plt.plot(t1, Population[6], 'r', label='FR2')
plt.plot(t1, Population[7], 'g', label='FR3')
plt.plot(t1, Population[8], 'y', label='FR4')
plt.plot(t1, Population[9], 'k', label='FR5')
plt.xlabel('time (in ps)')
plt.ylabel('population of Pfr states')
plt.legend(loc=1)

# plt.figure()
# plt.plot(t_field, field, label = 'field')
# plt.xlabel('time (in ps)')
# plt.ylabel('field in $2.35*10^8$ V/m')
# plt.legend(loc = 1)

plt.show()
