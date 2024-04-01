import random
import pandas as pd
import numpy as np
import shutil
import sys
import os.path
import math
from pyomo.environ import *

def solve_ode(init_state, budget, para, length):
    p_ei  = para['pei']
    p_er  = para['per']
    p_ea  = 1-p_ei-p_er
    p_ai  = 0.10714
    p_ar  = 0.04545
    p_irr = 1/8
    alpha  = 0.6
    beta_2 = para['beta1']
    beta_1 = para['beta2']
    cost_tr = 0.0977
    p_id  = 1.1/1000
    p_idd = 0.1221/1000
    p_srr = 0.7/3
    S = init_state[0]
    E = init_state[1]
    A = init_state[2]
    I = init_state[3]
    R = init_state[4]
    D = init_state[5]
    N = np.sum(init_state)
    K = length
    w_max  = 1.0
    tr_max = 1.0
    v_max = 0.2/14
    M = budget
    
    m = ConcreteModel()
    m.k = RangeSet(0, K)
    m.t = RangeSet(0, K-1)
    m.N = N
    m.tri = Var(m.k, bounds=(0, 1)) 
    m.wi = Var(m.k, bounds=(0, 1))  
    m.v = Var(m.k, bounds=(0, 1))
    m.aq = Var(m.k, bounds=(0, 1))
    m.iq = Var(m.k, bounds=(0, 1))
    m.x_s = Var(m.k, domain = NonNegativeReals)
    m.x_e = Var(m.k, domain = NonNegativeReals)
    m.x_a = Var(m.k, domain = NonNegativeReals)
    m.x_i = Var(m.k, domain = NonNegativeReals)
    m.x_r = Var(m.k, domain = NonNegativeReals)
    m.x_d = Var(m.k, domain = NonNegativeReals)
    
    m.xs_c = Constraint(m.t, rule=lambda m, k: m.x_s[k+1] == m.x_s[k]
                    -(beta_1*(1-m.iq[k])*m.x_i[k]+beta_2*(m.x_e[k]+(1-m.aq[k])*m.x_a[k]))
                    *(1-m.v[k]*v_max+(1-p_srr)*m.v[k]*v_max)*m.x_s[k]/(m.x_s[k]+m.x_e[k]+m.x_a[k]+m.x_i[k]+m.x_r[k])
                    -p_srr*m.v[k]*v_max*m.x_s[k]
                    +m.x_r[k]/180)

    m.xe_c = Constraint(m.t, rule=lambda m, k: m.x_e[k+1] == m.x_e[k]
                        +alpha*(beta_1*(1-m.iq[k])*m.x_i[k]+beta_2*(m.x_e[k]+(1-m.aq[k])*m.x_a[k]))
                        *(1-m.v[k]*v_max+(1-p_srr)*m.v[k]*v_max)*m.x_s[k]/(m.x_s[k]+m.x_e[k]+m.x_a[k]+m.x_i[k]+m.x_r[k])
                        -p_ei*m.x_e[k]
                        -p_er*m.x_e[k]
                        -m.wi[k]*p_ea*m.x_e[k])
                     
    m.xa_c = Constraint(m.t, rule=lambda m, k: m.x_a[k+1] == m.x_a[k]
                        +m.wi[k]*p_ea*m.x_e[k]
                        -p_ai*m.x_a[k]
                        -p_ar*m.x_a[k])
                                        
    m.xi_c = Constraint(m.t, rule=lambda m, k: m.x_i[k+1] == m.x_i[k]
                        +(1-alpha)*(beta_1*(1-m.iq[k])*m.x_i[k]+beta_2*(m.x_e[k]+(1-m.aq[k])*m.x_a[k]))
                        *(1-m.v[k]*v_max+(1-p_srr)*m.v[k]*v_max)*m.x_s[k]/(m.x_s[k]+m.x_e[k]+m.x_a[k]+m.x_i[k]+m.x_r[k])
                        +p_ei*m.x_e[k]
                        +p_ai*m.x_a[k]
                        -m.tri[k]*p_irr*m.x_i[k]
                        -(1-m.tri[k])*m.x_i[k]*p_id
                        -m.tri[k]*m.x_i[k]*p_idd)
                                                                                       
    m.xr_c = Constraint(m.t, rule=lambda m, k: m.x_r[k+1] == m.x_r[k]
                        +m.tri[k]*p_irr*m.x_i[k]
                        +p_er*m.x_e[k]
                        +p_ar*m.x_a[k]
                        +p_srr*m.v[k]*v_max*m.x_s[k]
                        -m.x_r[k]/180)

    m.xd_c = Constraint(m.t, rule=lambda m, k: m.x_d[k+1] == m.x_d[k]
                        +(1-m.tri[k])*m.x_i[k]*p_id
                        +m.tri[k]*m.x_i[k]*p_idd)
    
    m.pc = ConstraintList()
    m.pc.add(m.x_s[0]==S)
    m.pc.add(m.x_e[0]==E)
    m.pc.add(m.x_a[0]==A)
    m.pc.add(m.x_i[0]==I)
    m.pc.add(m.x_r[0]==R)
    m.pc.add(m.x_d[0]==D)

    gamma = 1
    m.sumcost = sum(m.wi[k]*(0.000369*(m.wi[k]*(m.x_s[k]+m.x_a[k]+m.x_e[k]+m.x_r[k])/m.N)
                         *(m.wi[k]*(m.x_s[k]+m.x_e[k]+m.x_a[k]+m.x_r[k])/m.N) + 0.001057)
                         *(m.x_s[k]+m.x_e[k]+m.x_a[k]+m.x_r[k])
                    +0.0977*m.tri[k]*m.x_i[k]+m.x_s[k]
                    *m.v[k]*v_max*0.07 + m.x_i[k]*m.iq[k]*0.02 + m.x_a[k]*m.aq[k]*0.02 for k in m.t) 
    m.budget_c = Constraint(expr = m.sumcost <= M)

    m.suminfected = sum(2.6 * (beta_1*(1-m.iq[k])*m.x_i[k]+beta_2*(m.x_e[k]+(1-m.aq[k])*m.x_a[k]))
                        *(1-m.v[k]*v_max+(1-p_srr)*m.v[k]*v_max)*m.x_s[k]/(m.x_s[k]+m.x_e[k]+m.x_a[k]+m.x_i[k]+m.x_r[k])
                        +(m.x_d[k+1] - m.x_d[k]) * 60 for k in m.t)
    m.obj = Objective(expr = m.suminfected, sense = minimize)
    solver = SolverFactory('ipopt')
    solver.options['max_iter']= 100000
    solver.solve(m,tee=False)
    k = np.array([k for k in m.k])
    tri = np.array([m.tri[k]() for k in m.k])
    wi = np.array([m.wi[k]() for k in m.k])
    vi = np.array([m.v[k]() for k in m.k])
    aq  = np.array([m.aq[k]() for k in m.k])
    iq  = np.array([m.iq[k]() for k in m.k])
    return [vi[0], tri[0], aq[0], iq[0], wi[0]]