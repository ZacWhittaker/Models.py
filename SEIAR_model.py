import numpy as np

from scipy.integrate import odeint
import matplotlib.pyplot as plt


def delta_S(beta, Lambda, mu, S, I, N, q, A):
    return Lambda - beta * S * I / N - q * S * A / N - mu * S


def delta_E(beta, eta, mu, S, I, E, N, q, A):
    return beta*S*I/N + q*S*A/N - eta*E - mu*E


def delta_I(alpha, eta, mu, I, E, p):
    return p*eta*E - alpha*I - mu*I


def delta_A(gamma, eta, mu, A, E, p):
    return (1-p)*eta*E - gamma*A - mu*A


def delta_R(I, R, alpha, mu, gamma, A):
    return alpha*I + gamma*A - mu*R

'''
p is the probability of becoming infectious
beta is transmission rate 
alpha is recovery rate
eta is per-capita rate of becoming infectious
Lambda is birth rate 
mu is death rate 
'''


def population(S, E, I, A, R, Lambda, mu):
    return Lambda - mu*(S+E+I+A+R)


def sir_model(y, t, beta, alpha, Lambda, eta, mu, p, q, gamma):
    S, E, I, A, R, N = y
    N = S+E+I+R

    result = [delta_S(beta, Lambda, mu, S, I, N, q, A),
              delta_E(beta, eta, mu, S, I, E, N, q, A),
              delta_I(alpha, eta, mu, I, E, p),
              delta_A(gamma, eta, mu, A, E, p),
              delta_R(I, R, alpha, mu, gamma, A),
              population(S, E, I, A, R, Lambda, mu)]

    return result

inital_conditions = [1000, 0, 1, 0, 0, 1001]
T = np.linspace(0, 150, num=3000)
sol = odeint(sir_model, inital_conditions, T, args=(0.8, 0.05, 1, 0.1, 0.0003, 0.5, 0.3, 0.25))



plt.plot(T, sol[:, 0], 'b', label='Susceptible')
plt.plot(T, sol[:, 1], 'g', label='Exposed')
plt.plot(T, sol[:, 2], 'r', label='Infected')
plt.plot(T, sol[:, 3], 'y', label='Asymptomatic')
plt.plot(T, sol[:, 4], 'k', label='Recovered')
#plt.plot(T, sol[:, 5], 'm', label='Population')
plt.legend(loc='best')
plt.xlabel('time (days)')
plt.ylabel('Number of people)')
plt.title('SEIAR')

plt.show()

beta_vals = [0.2, 0.5, 0.8]
alpha_vals = [0.1, 0.4, 0.7]
eta_vals = [0.3, 0.6, 0.9]
line_style = ['-', '--', '-.']
'''
this function is used for plotting specific plots to support my report 
'''
def plot_sections():
    line = 0
    for beta in beta_vals:
        label = 'beta: ' + str(beta)
        sol = odeint(sir_model, inital_conditions, T, args=(0.8, 0.05, 1, 0.1, 0.0003, 0.3, beta, 0.1))
        #plt.plot(T, sol[:, 0], 'b', label='Susceptible')
        #plt.plot(T, sol[:, 1], 'g', label='Exposed')
        plt.plot(T, sol[:, 2], 'r', label='Infected, q= ' +str(beta), linestyle=line_style[line])
        plt.plot(T, sol[:, 3], 'y', label='Asymptomatic, q= ' +str(beta), linestyle=line_style[line])
        #plt.plot(T, sol[:, 4], 'k', label='Recovered')
        #plt.plot(T, sol[:, 5], 'm', label='Population')
        plt.legend(loc='best')
        plt.xlabel('time (days)')
        plt.ylabel('Number of people')
        plt.title('Varying q')

        line += 1

    plt.show()

