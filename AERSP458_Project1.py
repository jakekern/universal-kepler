#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jacob
"""
import math
import numpy as np

r0v = np.array([-2.4258, 0.8420, -2.2982])
r0 = math.sqrt(r0v[0]**2+r0v[1]**2+r0v[2]**2)
v0v = np.array([-1.9857, -0.4478, 0.4309])
v0 = math.sqrt(v0v[0]**2+v0v[1]**2+v0v[2]**2)
t0 = 0
t1 = 2.1
mu = 4*math.pi**2
tol = 10**-6
sigma0 = np.dot(r0v,v0v)/math.sqrt(mu)
alpha = (v0**2-2*mu/r0)/(-mu)

chi = alpha*math.sqrt(mu)*(t1-t0)
b = np.array([1])

# Newton-Raphson to solve universal time equation for chi
chiOld = chi
chiNew = 0
while abs(chiNew-chi)>=tol:
    a0 = chiOld/2
    b0 = 1
    delta0 = 1
    deltaOld = delta0
    Sigma0 = a0/b0
    SigmaOld = Sigma0
    u0 = a0/b0
    uOld = u0
    n = 1
    while abs(uOld)>=tol:
        b = np.append(b,2*n+1)
        aN = alpha*(chiOld/2)**2
        deltaNew = 1/(1-(aN/(b[n-1]*b[n]))*deltaOld)
        uNew = uOld*(deltaNew-1)
        SigmaNew = SigmaOld+uNew
        deltaOld = deltaNew
        uOld = uNew
        SigmaOld = SigmaNew
        n = n+1
        u = SigmaNew
    U0 = (1-alpha*u**2)/(1+alpha*u**2)
    U1 = (2*u)/(1+alpha*u**2)
    U2 = (2*u**2)/(1+alpha*u**2)
    N = alpha*math.sqrt(mu)*(t1-t0)+(1-alpha*r0)*U1-alpha*sigma0*U2-chiOld
    D = (1-alpha*r0)*U0-alpha*sigma0*U1-1
    chi = chiOld
    chiNew = chiOld-N/D
    chiOld = chiNew
# Calculating r1 and v1 using Lagrange coefficients
F = 1-1/r0*U2
G = r0/math.sqrt(mu)*U1+sigma0/math.sqrt(mu)*U2
r1 = F*r0v+G*v0v
Ft = -math.sqrt(mu)/(math.sqrt(r1[0]**2+r1[1]**2+r1[2]**2)*r0)*U1
Gt = 1-U2/math.sqrt(r1[0]**2+r1[1]**2+r1[2]**2)
v1 = Ft*r0v+Gt*v0v
print("Position at time t1 is: " + str(r1))
print("Velocity at time t1 is: " + str(v1))
