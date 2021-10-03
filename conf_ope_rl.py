import numpy as np
import pandas as pd
from random import seed
from random import random
import matplotlib.pyplot as plt
import gurobipy as gp
from joblib import Parallel, delayed
from copy import deepcopy
import cvxpy as cvx
from scipy.sparse import lil_matrix
import mosek
from datetime import datetime
import pickle



reshape_byxrow = lambda a,nU: a.reshape(-1,nU,a.shape[-1]).sum(1)


def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )


# Simulate from multinomial distribution
def simulate_multinomial(vmultinomial):
    r=np.random.uniform(0.0, 1.0)
    CS=np.cumsum(vmultinomial)
    CS=np.insert(CS,0,0)
    m=(np.where(CS<r))[0]
    nextState=m[len(m)-1]
    return nextState

def get_pib(nA,nS, a_s, stateHist): 
    '''
    # estimate p(a \mid s)
    '''
    p_a1_su = np.zeros([nA, nS])
    for a in range(nA): 
        p_a1_su[a,:] = [sum( (a_s==a) & (stateHist[:-1,s]==1) ) / sum(stateHist[:-1,s]) for s in range(nS)]
    return p_a1_su

def get_pib_counts(nA,nS, a_s, stateHist): 
    '''
    # estimate counts of p(a \mid s)
    '''
    p_a1_su_counts = np.zeros([nA, nS])
    for a in range(nA): 
        p_a1_su_counts[a,:] = [sum( (a_s==a) & (stateHist[:-1,s]==1) )  for s in range(nS)]
    return p_a1_su_counts

def get_cndl_s_a_sprime(s_a_sprime, distrib):
    assert np.isclose(distrib.sum(), 1)
    joint_s_a_sprime = s_a_sprime / np.sum(s_a_sprime)
    s_a_giv_sprime = joint_s_a_sprime / distrib
    for k in range(s_a_giv_sprime.shape[2]):
        if not np.isclose((s_a_giv_sprime[:,:,k].sum()), 1,atol = 0.01):
            print 'conditional not well behaved for s='+str(k)
    return [joint_s_a_sprime, s_a_giv_sprime]


# take in transition matrix, policy, initial state s0 
def simulate_rollouts( nS, nA, P, Pi, state_dist, n ):
    stateChangeHist = np.zeros([nS,nS])
    s_a_sprime = np.zeros([nS,nA,nS])
    currentState=0; 
    s0 = np.zeros([1,nS])
    print state_dist
    s0_ = np.random.choice(range(nS), p = state_dist)
    s0[0,s0_] = 1 
    stateHist=s0
    dfStateHist=pd.DataFrame(s0)
    distr_hist = [ [0] * nS]
    a_s = np.zeros(n)

    for x in range(n):
        a = np.random.choice(np.arange(0, nA), p=Pi[:,currentState])
        a_s[x] = a
        currentRow=np.ma.masked_values(( P[currentState, a, :] ) , 0.0)
        nextState=np.random.choice(np.arange(0, nS), p=currentRow)
        # Keep track of state changes
        stateChangeHist[currentState,nextState]+=1
        # Keep track of the state vector itself
        state=np.zeros([1,nS]) #np.array([[0,0,0,0]])
        state[0,nextState]=1.0
        # Keep track of state history
        stateHist=np.append(stateHist,state,axis=0)
        # get s,a,s' distribution 
        s_a_sprime[currentState, a, nextState] += 1
        currentState=nextState
        # calculate the actual distribution over the 3 states so far
        totals=np.sum(stateHist,axis=0)
        gt=np.sum(totals)
        distrib=totals*1.0/gt
        distrib=np.reshape(distrib,(1,nS))
        distr_hist=np.append(distr_hist,distrib,axis=0)

    return [ stateChangeHist, stateHist, a_s, s_a_sprime, distrib, distr_hist ]


def agg_state(nS,nSmarg,nU,nA,s_a_sprime):
    ''' Aggregate every nSmarg states 
    '''
    agger = np.zeros([nS, nSmarg])
    eye_nSmarg = np.eye(nSmarg)
    agger = np.repeat(eye_nSmarg, nU,axis = 0).reshape([nS, nSmarg])
    agg_s_a_sprime = np.zeros([nSmarg, nA, nSmarg])
    for a in range(nA): 
        agg_s_a_sprime[:,a,:] = agger.T.dot(s_a_sprime[:,a,:].dot(agger)) 
    return agg_s_a_sprime


def get_bnds(est_Q,LogGamma):
    ''' Odds ratio with respect to 1-a 
    '''
    n = len(est_Q)
    p_hi = np.multiply(np.exp(LogGamma), est_Q ) / (np.ones(n) - est_Q + np.multiply(np.exp(LogGamma), est_Q ))
    p_lo = np.multiply(np.exp(-LogGamma), est_Q ) / (np.ones(n) - est_Q + np.multiply(np.exp(-LogGamma), est_Q ))
    assert (p_lo < p_hi).all()
    a_bnd = 1/p_hi;
    b_bnd = 1/p_lo
    return [ a_bnd, b_bnd ]


def get_bnds_as( p_a1_s, LogGamma ):
    a_bnd = np.zeros(p_a1_s.shape); b_bnd = np.zeros(p_a1_s.shape)
    for a in range(p_a1_s.shape[0]):
        [a_bnd_, b_bnd_] = get_bnds(p_a1_s[a,:],LogGamma)
        a_bnd[a,:] = a_bnd_; b_bnd[a,:] = b_bnd_
    return [a_bnd, b_bnd]


def get_auxiliary_info_from_traj(stateChangeHist, stateHist, a_s, s_a_sprime, distrib, distr_hist, nA,nS): 
    '''
    get empirical joint distribution
    '''
    p_a1_su = get_pib(nA,nS, a_s, stateHist); 
    [joint_s_a_sprime, s_a_giv_sprime] = get_cndl_s_a_sprime(s_a_sprime, distrib)
    return [ p_a1_su, joint_s_a_sprime, s_a_giv_sprime ]


def get_agg_auxiliary_info_from_all_trajectories(res, nA,nS, nSmarg, nU): 
    # s_a_sprime_cum = np.zeros([nS,nA,nS])
    # assume all trajectories of same length
    # better to compute running averages rather than 
    [ stateChangeHist, stateHist, a_s, s_a_sprime, distrib, distr_hist ] = res[0] 
    N = len(res); 
    s_a_sprime_cum = s_a_sprime
    p_a1_su = get_pib_counts(nA,nS, a_s, stateHist); 

    aggStateHist = reshape_byxrow(stateHist.T, nU).T; 
    agg_s_a_sprime_cum = agg_state(nS,nSmarg,nU,nA,s_a_sprime)
    p_a1_s = get_pib_counts(nA,nSmarg, a_s, aggStateHist); 
    i = 0
    # totals=np.sum(stateHist,axis=0); gt=np.sum(totals); distrib=totals/gt; distrib=np.reshape(distrib,(1,nS))
    for traj in res[1:]: 
        if i%100==0: 
            print i
        i+=1; 
        [ stateChangeHist_, stateHist_, a_s_, s_a_sprime_, distrib_, distr_hist ] = traj 
        p_a1_su_ = get_pib_counts(nA,nS, a_s_, stateHist_) # takes too much memory to store history  # stateHist = np.vstack([stateHist, stateHist_])        # a_s = np.vstack([a_s, a_s_.reshape([len(a_s_),1]) ])
        # append 
        s_a_sprime_cum = s_a_sprime_cum + s_a_sprime_; p_a1_su += p_a1_su_; distrib += distrib_
        # aggregate history online 
        aggStateHist_ = reshape_byxrow(stateHist_.T, nU).T
        agg_s_a_sprime_ = agg_state(nS,nSmarg,nU,nA,s_a_sprime_)
        p_a1_s_ = get_pib_counts(nA,nSmarg, a_s, aggStateHist_)
        agg_s_a_sprime_cum = agg_s_a_sprime_cum + agg_s_a_sprime_; p_a1_s += p_a1_s_
    # take average over trajectories
    distrib = (distrib / distrib.sum()) ; # p_infty_b_su
    # print distrib
    p_a1_su = p_a1_su / N; p_a1_su = p_a1_su/ p_a1_su.sum(axis=0) # return probabilities
    p_a1_s = p_a1_s / N; p_a1_s = p_a1_s/ p_a1_s.sum(axis=0)
    # print ((s_a_sprime_cum/s_a_sprime_cum.sum())/distrib)[:,:,0]
    [joint_s_a_sprime, s_a_giv_sprime] = get_cndl_s_a_sprime(s_a_sprime_cum, distrib.flatten())

    p_infty_b_s = (reshape_byxrow(distrib.T,nU).T ).flatten()
    [joint_s_a_sprime_agg, s_a_giv_sprime_agg] = get_cndl_s_a_sprime(agg_s_a_sprime_cum, p_infty_b_s)
    # return [aggStateHist, p_a1_s, p_e_s, agg_s_a_sprime, joint_s_a_sprime_agg, s_a_giv_sprime_agg]

    return [ p_a1_su, joint_s_a_sprime, s_a_giv_sprime, s_a_sprime_cum, p_a1_s, joint_s_a_sprime_agg, s_a_giv_sprime_agg, agg_s_a_sprime_cum, distrib]

## deprecated version that keeps things in memory
# def get_auxiliary_info_from_all_trajectories(res, nA,nS): 

#     # s_a_sprime_cum = np.zeros([nS,nA,nS])
#     # assume all trajectories of same length
#     [ stateChangeHist, stateHist, a_s, s_a_sprime, distrib, distr_hist ] = res[0] 
#     N = len(res)
#     a_s = a_s.reshape([len(a_s),1])
#     s_a_sprime_cum = s_a_sprime
#     totals=np.sum(stateHist,axis=0); gt=np.sum(totals); distrib=totals/gt; distrib=np.reshape(distrib,(1,nS))
#     for traj in res[1:]: 
#         [ stateChangeHist_, stateHist_, a_s_, s_a_sprime_, distrib, distr_hist ] = traj 
#         stateHist = np.vstack([stateHist, stateHist_])
#         a_s = np.vstack([a_s, a_s_.reshape([len(a_s_),1]) ])
#         s_a_sprime_cum = s_a_sprime_cum + s_a_sprime_
#         totals=np.sum(stateHist_,axis=0); gt=np.sum(totals); distrib_=totals/gt; distrib_=np.reshape(distrib,(1,nS))
#         distrib += distrib_
#     print 'a_s shape', a_s.shape
#     print 'statehist shape', stateHist.shape
#     distrib = distrib / N # take average over trajectories
#     print 'stat dist p_b(s)', distrib
#     # totals=np.sum(stateHist,axis=0); gt=np.sum(totals); distrib=totals/gt; distrib=np.reshape(distrib,(1,nS))
#     p_a1_su = get_pib(nA,nS, a_s, stateHist); 
    
#     print ((s_a_sprime_cum/s_a_sprime_cum.sum())/distrib)[:,:,0]
#     [joint_s_a_sprime, s_a_giv_sprime] = get_cndl_s_a_sprime(s_a_sprime_cum, distrib.flatten())
#     return [ p_a1_su, joint_s_a_sprime, s_a_giv_sprime, s_a_sprime, stateHist, a_s, distrib]

def agg_history(stateHist, s_a_sprime, p_infty_b_s, a_s, p_e_su, nA, nS, nSmarg, nU): 
    '''
    # agg history and process
    '''
    # assert np.isclose(sum(p_infty_b_s), 1)
    aggStateHist = reshape_byxrow(stateHist.T, nU).T
    p_a1_s = get_pib(nA,nSmarg, a_s, aggStateHist); 
    p_e_s = reshape_byxrow(p_e_su.T,nU).T / nU #get_agg(p_e_su, nU)/2
    agg_s_a_sprime = agg_state(nS,nSmarg,nU,nA,s_a_sprime)
    [joint_s_a_sprime_agg, s_a_giv_sprime_agg] = get_cndl_s_a_sprime(agg_s_a_sprime, p_infty_b_s)
    return [aggStateHist, p_a1_s, p_e_s, agg_s_a_sprime, joint_s_a_sprime_agg, s_a_giv_sprime_agg]

def rollout_parallel(i, nS, nA, P, Pi, state, n ):
    res = simulate_rollouts( nS, nA, P, Pi, state, n )
    return res



def get_w_lp(gamma, s_a_giv_sprime, p_infty_b_su, pe_su, p_a1_su, nA, nS, tight= True, quiet = True):
    m = gp.Model()
    w = m.addVars(nS)
    if quiet: m.setParam("OutputFlag", 0)
    for k in range(nS): 
        assert np.isclose((s_a_giv_sprime[:,:,k].sum()), 1,atol = 0.01)
    assert np.isclose(sum(p_infty_b_su), 1)
    assert len(p_infty_b_su) == nS
    epsilon = 0.2
    p_infty_b_su=p_infty_b_su.flatten()
    for k in range(nS): 
        m.addConstr( 0 == (-1*w[k] + gp.quicksum(w[j]* gp.quicksum([s_a_giv_sprime[j,a,k]*(pe_su[a,j] / p_a1_su[a,j]) for a in range(nA)]) for j in range(nS) ) ) ) 
#         m.addConstr( 0 == (-1*w[k] + gp.quicksum( [w[j]*s_a_giv_sprime[j,a,k]*(pe_su[a,j] / p_a1_su[a,j]) for a in range(nA)]) for j in range(nS) )) 
#         m.addConstr( 0 == (1-gamma)*(1-gp.quicksum(w[k] * p_infty_b[k] for k in range(nS)))+   gamma*(-1*w[k] + gp.quicksum( sum([s_a_giv_sprime[j,a,k] * (pe_su[a,j] / p_a1_su[a,j]) for a in range(nA)]) for j in range(nS) )) )
    if tight:
        m.addConstr(gp.quicksum([ w[k]*p_infty_b_su[k] for k in range(nS)]) - 1 <= epsilon)
    else: 
        m.addConstr(gp.quicksum([ w[k]*p_infty_b_su[k] for k in range(nS)]) >= 0.1)
    m.update()
    m.optimize()
    w_ = np.asarray([w[i].X for i in range(nS)])
    return w_

def get_w_lp_testfunction(gamma, s_a_giv_sprime, p_infty_b_su, pe_su, p_a1_su, nA, nS, tight= True, quiet = True):
    m = gp.Model()
    w = m.addVars(nS)
    if quiet: m.setParam("OutputFlag", 0)
    for k in range(nS): 
        assert np.isclose((s_a_giv_sprime[:,:,k].sum()), 1,atol = 0.01)
    assert np.isclose(sum(p_infty_b_su), 1)
    assert len(p_infty_b_su) == nS
    epsilon = 0.2
    p_infty_b_su=p_infty_b_su.flatten()
    for k in range(nS): 
        m.addConstr( 0 == (-1*w[k] + gp.quicksum(w[j]* gp.quicksum([s_a_giv_sprime[j,a,k]* p_infty_b_su[k]*(pe_su[a,j] / p_a1_su[a,j]) for a in range(nA)]) for j in range(nS) ) ) ) 
#         m.addConstr( 0 == (-1*w[k] + gp.quicksum( [w[j]*s_a_giv_sprime[j,a,k]*(pe_su[a,j] / p_a1_su[a,j]) for a in range(nA)]) for j in range(nS) )) 
#         m.addConstr( 0 == (1-gamma)*(1-gp.quicksum(w[k] * p_infty_b[k] for k in range(nS)))+   gamma*(-1*w[k] + gp.quicksum( sum([s_a_giv_sprime[j,a,k] * (pe_su[a,j] / p_a1_su[a,j]) for a in range(nA)]) for j in range(nS) )) )
    if tight:
        m.addConstr(gp.quicksum([ w[k]*p_infty_b_su[k] for k in range(nS)]) - 1 <= epsilon)
    else: 
        m.addConstr(gp.quicksum([ w[k]*p_infty_b_su[k] for k in range(nS)]) >= 0.1)
    m.update()
    m.optimize()
    w_ = np.asarray([w[i].X for i in range(nS)])
    return w_

""" subgrad descent template algo
automatically augments data ! 
take in theta_0, # rounds
LOSS: loss function
GRAD_: fn to obtain parametric subgradient
POL_GRAD: gradient of parametrized policy wrt parameters
PI_1: return prob of pi(x) = 1
data: dictionary, e.g. of x, t01, y, a_, b_
Projected onto bounds 
"""


def proj_grad_descent(g, N_RNDS, data, eta_0=1, step_schedule=0.5, sense_min = True):
    risks = np.zeros(N_RNDS)
    THTS = [None] * N_RNDS; PARAMS = [None] * N_RNDS; losses = [None] * N_RNDS
    gs_proj = [None] * N_RNDS; 
    gs_init = [None] * N_RNDS; 
    # initialize randomly in [a_, b_]
    for k in range(N_RNDS):
        eta_t = eta_0 * 1.0 / np.power((k + 1) * 1.0, step_schedule)
        # [loss, param] = LOSS_(th, data)
        # subgrad = GRAD_(th, data)
        [obj, th, A] = obj_eval(g, *[data])
        data['A']=A; data['theta'] = th
        g_grad = grad_H_wrt_g(g, *[data])
        
        if sense_min == True:
            g = -1*g 
            g_grad = -1*g_grad
        g_step = g + eta_t * g_grad; gs_init[k] = g_step
        [g_proj, proj_val] = proj_g_(g_step, *[data])
        gs_proj[k] = g_proj
        g = g_proj
        THTS[k] = th; losses[k] = obj
    return [losses, gs_init, gs_proj, THTS]


def random_g(a_bnd,b_bnd): 
    [nA,nS] = a_bnd.shape
    g = np.zeros([nS,nA,nS])
    draw = np.random.uniform(size = a_bnd.shape) * (b_bnd - a_bnd)
    for k in range(nS): 
        g[k,:,:] = a_bnd + draw
    return g

def obj_eval(g, *args): 
    ''' Evaluate objective
    Assume g is [k,a,j]
    '''
    data = args[0]
    Phi = data['Phi'] # state
    p_infty_b_s = data['pbs'] 
    nSmarg = len(p_infty_b_s)
    s_a_giv_sprime = data['s_a_giv_sprime'] 
    p_e_s = data['p_e_s'];     nA = len(p_e_s); check_grad = data['check_grad']
    A = np.zeros([nSmarg,nSmarg])
    for k in range(nSmarg): 
        for j in range(nSmarg): 
            A[k,j] += sum( [ s_a_giv_sprime[j,a,k]*p_infty_b_s[k] * (p_e_s[a,j] *g[k,a,j] ) for a in range(nA)] )
        A[k,k] -= p_infty_b_s[k]
    tildeA = A
    # Make regular and add normalization 
    tildeA[-1,:] = p_infty_b_s
    v = np.zeros([nSmarg]); v[-1] = 1 
    theta = np.linalg.inv(tildeA).dot(v)
    if check_grad: 
        return Phi.dot(theta)
    return [Phi.dot(theta), theta, tildeA]



def grad_H_wrt_g(g, *args): 
    ''' Evaluate gradient
    Assume g is [k,a,j]
     -({I}[j /= |S|]  pi^e_a,j Pb_{j,a,k}) (\E[A']^{-\top } \Phi \theta^\top)_{i,j}

    Build A matrix explicitly 
    '''
    data = args[0]
    A = data['A']
    theta = data['theta']
    Phi = data['Phi']
    outer_ = np.outer(Phi, theta)
    M = np.linalg.inv(A).dot(outer_)
    g_grad = np.zeros(g.shape)
    s_a_giv_sprime = data['s_a_giv_sprime'] ;p_e_s = data['p_e_s'];    p_infty_b_s = data['pbs'] 
    for k in range(g_grad.shape[0]): 
        for a in range(g_grad.shape[1]): 
            for j in range(g_grad.shape[2])[:-1]: # omit last row of J  
                g_grad[k,a,j] = M[j,k]*s_a_giv_sprime[j,a,k]*p_infty_b_s[k]*p_e_s[a,j]
    return g_grad

def proj_g_(g_tilde, *args): 
    ''' Project a given g vector onto feasible vector 
    '''
    data = args[0] 
    a_bnd = data['a_bnd']; b_bnd = data['b_bnd']
    p_infty_b_s = data['pbs'] ; p_e_s = data['p_e_s']
    nS = len(p_infty_b_s)
    nA = len(p_e_s)
    s_a_giv_sprime = data['s_a_giv_sprime'] 
    tight = data['tight']

    m = gp.Model()
    g = m.addVars(nS,nA,nS) 
    quiet = True
    if quiet: m.setParam("OutputFlag", 0)

    for k in range(nS): 
        for a in range(nA): 
            for j in range(nS): 
                m.addConstr(g[k,a,j] <= b_bnd[a,j])
                m.addConstr(g[k,a,j] >= a_bnd[a,j])
    for a in range(nA): 
        if tight:
            m.addConstr(gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b_s[k] for k in range(nS) for j in range(nS) ] ) == 1)
        else: 
            m.addConstr(gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b_s[k] for k in range(nS) for j in range(nS)] ) - 1 <= epsilon)
            m.addConstr(1 - gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b_s[k] for k in range(nS) for j in range(nS)] )  <= epsilon)
    m.update()
    obj = gp.quicksum( (g[k,a,j]*g[k,a,j] - 2*g[k,a,j]*g_tilde[k,a,j] + g_tilde[k,a,j]*g_tilde[k,a,j] for k in range(nS)for a in range(nA) for j in range(nS)))
    m.setObjective(obj)

    m.optimize()
    if (m.status == gp.GRB.OPTIMAL): 
        g_ = np.asarray([g[k,a,j].x for k in range(nS) for a in range(nA) for j in range(nS) ]).reshape([nS,nA,nS])
        return [g_,m.objVal]
    else:
        return [None, None]





def optw_ls(th, *args):
    data = dict(args[0])
    X = data['x']
    t01 = data['t01']
    Y = data['y']
    a_ = data['a_']
    b_ = data['b_']
    x0 = data['x0']
    sign = data['sign']
    n = X.shape[0]
    W = np.diag(th.flatten())
    beta = np.linalg.inv(X.T * W * X) * X.T * W * Y
    loss = np.asscalar((np.dot(x0, beta)))*sign
    return loss


def primal_scalarized_L1_feasibility_for_saddle(gamma, w, a_bnd,b_bnd, s_a_giv_sprime, p_infty_b, pe_s, p_a1_s,
                       nS, nA, tight= True, quiet = True):
# Minimize L1 residuals over g
    for k in range(nS): 
        assert np.isclose((s_a_giv_sprime[:,:,k].sum()), 1,atol = 0.01)
    m = gp.Model()
    p_infty_b = p_infty_b.flatten()
#     w = m.addVars(nS)
    g = m.addVars(nS,nA,nS) #\beta_k(a\mid j)
    z = m.addVars(nS) 
    if quiet: m.setParam("OutputFlag", 0)
    epsilon = 0.5

    for k in range(nS): 
        m.addConstr( z[k] >= (-1*w[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
        m.addConstr( z[k] >= -1*(-1*w[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
#         m.addConstr( 0 == (1-gamma)*(1-gp.quicksum(w))+   gamma*(-1*w[k] + gp.quicksum( sum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
    for k in range(nS): 
        for a in range(nA): 
            for j in range(nS): 
                m.addConstr(g[k,a,j] <= b_bnd[a,j])
                m.addConstr(g[k,a,j] >= a_bnd[a,j])

    for a in range(nA): 
        if tight:
            m.addConstr(gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS) ] ) == 1)
        else: 
            m.addConstr(gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS)] ) - 1 <= epsilon)
            m.addConstr(1 - gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS)] )  <= epsilon)
    m.update()
    expr = gp.quicksum(z)
    m.setObjective(expr, gp.GRB.MINIMIZE)
    m.optimize()
    if (m.status == gp.GRB.OPTIMAL): 
        g_ = np.asarray([g[k,a,j].x for k in range(nS) for a in range(nA) for j in range(nS) ]).reshape([nS,nA,nS])
        return [m.objVal, g_]
    else: 
        g = None
        return [None, g]
#     g_ = m.getAttr('x', g)

def saddle_outer_min_w(gamma, g, Phi, eta, a_bnd,b_bnd, s_a_giv_sprime, p_infty_b, pe_s, p_a1_s,nS, nA, sense_min = True): 
    for k in range(nS): 
        assert np.isclose((s_a_giv_sprime[:,:,k].sum()), 1,atol = 0.01)
    assert np.isclose(sum(p_infty_b), 1)
    m = gp.Model()
    p_infty_b = p_infty_b.flatten()
    w = m.addVars(nS)
    z = m.addVars(nS) 
    quiet = True
    if quiet: m.setParam("OutputFlag", 0)

    for k in range(nS): 
        m.addConstr( z[k] >= (-1*w[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
        m.addConstr( z[k] >= -1*(-1*w[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
#         m.addConstr( 0 == (1-gamma)*(1-gp.quicksum(w))+   gamma*(-1*w[k] + gp.quicksum( sum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
    m.addConstr( gp.quicksum(w ) == 1)
    m.update()
    sense_min01 = 1 if sense_min else -1
    expr = eta * gp.quicksum(z) + sense_min01 * gp.quicksum( w[k] *Phi[k]*p_infty_b[k] for k in range(nS))
    m.setObjective(expr, gp.GRB.MINIMIZE)
    m.optimize()
    if (m.status == gp.GRB.OPTIMAL): 
        w_ = np.asarray([w[k].x for k in range(nS)  ])
        z_ = np.asarray([z[k].x for k in range(nS)  ])
        wphi_obj = np.dot(w_,Phi)
        print m.objVal
        return [wphi_obj, z_.sum() , w_]
    else:
        return None






def get_w_withAmatrix(s_a_giv_sprime,p_infty_b_s, p_e_s,p_a1_s, nSmarg): 
    '''
    with indic[s=k] test function
    '''
    nA = len(p_e_s)
    A = np.zeros([nSmarg,nSmarg]) # - np.eye(nSmarg)
    for k in range(nSmarg): 
        for j in range(nSmarg): 
            A[k,j] += sum( [ s_a_giv_sprime[j,a,k]*p_infty_b_s[k] * (p_e_s[a,j] / p_a1_s[a,j]) for a in range(nA)] )
        A[k,k] -= p_infty_b_s[k]
    tildeA = A
    tildeA[-1,:] = p_infty_b_s
    v = np.zeros(nSmarg); v[-1] = 1
    w = np.linalg.solve(tildeA, v)
    return w 

def get_w_withAmatrix_cond(s_a_giv_sprime,p_infty_b_s, p_e_s,p_a1_s, nSmarg): 
    '''
    conditional on s = k 
    '''
    nA = len(p_e_s)
    A = np.zeros([nSmarg,nSmarg]) # - np.eye(nSmarg)
    for k in range(nSmarg): 
        for j in range(nSmarg): 
            A[k,j] += sum( [ s_a_giv_sprime[j,a,k]* (p_e_s[a,j] / p_a1_s[a,j]) for a in range(nA)] )
        A[k,k] -= 1
    tildeA = A
    tildeA[-1,:] = np.ones(nSmarg)
    v = np.zeros(nSmarg); v[-1] = 1
    w = np.linalg.solve(tildeA, v)
    return w 

def get_w_withAmatrix_cond_from_g(g,s_a_giv_sprime,p_infty_b_s, p_e_s, nSmarg): 
    '''
    conditional on s = k 
    '''
    nA = len(p_e_s)
    A = np.zeros([nSmarg,nSmarg]) # - np.eye(nSmarg)
    for k in range(nSmarg): 
        for j in range(nSmarg): 
            A[k,j] += sum( [ s_a_giv_sprime[j,a,k] * (p_e_s[a,j] / g[k,a,j]) for a in range(nA)] )
        A[k,k] -= 1
    tildeA = A
    tildeA[-1,:] = np.ones(nSmarg)
    v = np.zeros(nSmarg); v[-1] = 1
    w = np.linalg.solve(tildeA, v)
    return w 


def primal_feasibility(gamma, w, a_bnd,b_bnd, s_a_giv_sprime, p_infty_b, pe_s, p_a1_s,
                       nS, nA, tight= True, quiet = True):
    for k in range(nS): 
        assert np.isclose((s_a_giv_sprime[:,:,k].sum()), 1,atol = 0.01)
    assert np.isclose(sum(p_infty_b), 1)
    m = gp.Model()
    p_infty_b = p_infty_b.flatten()
#     w = m.addVars(nS)
    g = m.addVars(nS,nA,nS) #\beta_k(a\mid j)
    if quiet: m.setParam("OutputFlag", 0)
    epsilon = 0.5

    for k in range(nS): 
        m.addConstr( 0 == (-1*w[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
#         m.addConstr( 0 == (1-gamma)*(1-gp.quicksum(w))+   gamma*(-1*w[k] + gp.quicksum( sum([w[j]*s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
    for k in range(nS): 
        for a in range(nA): 
            for j in range(nS): 
                m.addConstr(g[k,a,j] <= b_bnd[a,j])
                m.addConstr(g[k,a,j] >= a_bnd[a,j])
    for a in range(nA): 
        if tight:
            m.addConstr(gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS) ] ) == 1)
        else: 
            m.addConstr(gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS)] ) - 1 <= epsilon)
            m.addConstr(1 - gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS)] )  <= epsilon)
    m.update()
    m.optimize()
    if (m.status == gp.GRB.OPTIMAL): 
        feasibility = True
        g = np.asarray([g[k,a,j].x for k in range(nS) for a in range(nA) for j in range(nS) ]).reshape([nS,nA,nS])
    else: 
        feasibility = False
        g = None
#     g_ = m.getAttr('x', g)
    return [feasibility, g]



def primal_scalarized_L1_feasibility(gamma, w, a_bnd,b_bnd, s_a_giv_sprime, p_infty_b, pe_s, p_a1_s,
                       nS, nA, tight= True, quiet = True):
    for k in range(nS): 
        assert np.isclose((s_a_giv_sprime[:,:,k].sum()), 1,atol = 0.01)
    assert np.isclose(sum(p_infty_b), 1)
    m = gp.Model()
    p_infty_b = p_infty_b.flatten()
#     w = m.addVars(nS)
    g = m.addVars(nS,nA,nS) #\beta_k(a\mid j)
    z = m.addVars(nS) 
    if quiet: m.setParam("OutputFlag", 0)
    epsilon = 0.5

    for k in range(nS): 
        m.addConstr( z[k] >= (-1*w[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
        m.addConstr( z[k] >= -1*(-1*w[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
#         m.addConstr( 0 == (1-gamma)*(1-gp.quicksum(w))+   gamma*(-1*w[k] + gp.quicksum( sum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
    for k in range(nS): 
        for a in range(nA): 
            for j in range(nS): 
                m.addConstr(g[k,a,j] <= b_bnd[a,j])
                m.addConstr(g[k,a,j] >= a_bnd[a,j])

    for a in range(nA): 
        if tight:
            m.addConstr(gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS) ] ) == 1)
        else: 
            m.addConstr(gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS)] ) - 1 <= epsilon)
            m.addConstr(1 - gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS)] )  <= epsilon)
    m.update()
    expr = gp.quicksum(z)
    m.optimize()
    if (m.status == gp.GRB.OPTIMAL): 
        g = np.asarray([g[k,a,j].x for k in range(nS) for a in range(nA) for j in range(nS) ]).reshape([nS,nA,nS])

        return [m.objVal, g]
    else: 
        g = None
        return [None, g]
#     g_ = m.getAttr('x', g)

def dual_feasibility(gamma, w, a_bnd,b_bnd, s_a_giv_sprime, p_infty_b_s, pe_s, p_a1_s,
                       nS, nA, tight= True, quiet = True):
    for k in range(nS): 
        assert np.isclose((s_a_giv_sprime[:,:,k].sum()), 1,atol = 0.01)
    assert np.isclose(sum(p_infty_b_s), 1)
    m = gp.Model()
    p_infty_b_s = p_infty_b_s.flatten()
    c = m.addVars(nS,nA,nS, lb = 0) 
    d = m.addVars(nS,nA,nS, lb = 0) 
    lmbda01 = m.addVars(nS, vtype=gp.GRB.BINARY); 
    # lmbda = m.addVars(nS, lb = -1*gp.GRB.INFINITY)
    if tight: 
        mu = m.addVars(nA, lb = -1*gp.GRB.INFINITY)
    m.update()
    if quiet: m.setParam("OutputFlag", 0)
    for j in range(nS): 
        for a in range(nA): 
            for k in range(nS): 
                if tight: 
                    m.addConstr(c[j,a,k] - d[j,a,k] + 
                    - w[j]*pe_s[a,j]*(2*lmbda01[k]-1)*s_a_giv_sprime[j,a,k]#  gp.quicksum(lmbda[ind]*s_a_giv_sprime[j,a,ind]*w[j]*pe_s[a,j] for ind in range(nS)) 
                    + mu[a]*s_a_giv_sprime[j,a,k]*p_infty_b_s[k]  
                    <= 0 )
                else: 
                    m.addConstr(c[j,a,k] - d[j,a,k] 
                    - w[j]*pe_s[a,j]*(2*lmbda01[k]-1)*s_a_giv_sprime[j,a,k]  <= 0 )
        # m.addConstr(lmbda[j] <= 1)
        # m.addConstr(lmbda[j] >= -1)
    
    m.update()
    expr = gp.LinExpr()
    expr += gp.quicksum( c[j,a,k]*a_bnd[a,j] for k in range(nS) for a in range(nA) for j in range(nS) )
    expr += gp.quicksum( -1*d[j,a,k]*b_bnd[a,j] for k in range(nS) for a in range(nA) for j in range(nS) )
    expr += gp.quicksum( -1*(2*lmbda01[k]-1) * w[k] for k in range(nS) )
    if tight: 
        expr += gp.quicksum( mu )
    # need to change for discounted case
    m.setObjective(expr, gp.GRB.MAXIMIZE); m.optimize()

    if (m.status == gp.GRB.OPTIMAL): 
        print 'c, ', [c[j,a,k].x for j in range(nS) for a in range(nA) for k in range(nS)]
        print 'd, ', [d[j,a,k].x for j in range(nS) for a in range(nA) for k in range(nS)]
        lmbda_ = [(2*lmbda01[k].x-1) for k in range(nS)]
        return [m.objVal, lmbda_]
    else: 
        return [None, None]


def proj_w_(w_tilde, *args): 
    ''' Project a given w vector onto feasible vector 
    '''
    data = args[0] 
    a_bnd = data['a_bnd']; b_bnd = data['b_bnd']
    p_infty_b_s = data['pbs'] ; p_e_s = data['p_e_s']
    nS = len(p_infty_b_s)
    nA = len(p_e_s)
    s_a_giv_sprime = data['s_a_giv_sprime'] 

    m = gp.Model()
    w = m.addVars(nS) 
    quiet = True
    if quiet: m.setParam("OutputFlag", 0)
    m.addConstr( gp.quicksum(w[k]  for k in range(nS)) == 1 )

    m.update()
    obj = gp.quicksum( (w_tilde[k]*w_tilde[k] - 2*w_tilde[k]*w[k] + w[k]*w[k] for k in range(nS) ))
    m.setObjective(obj, gp.GRB.MINIMIZE)
    m.optimize()
    w_ = np.asarray([w[k].x for k in range(nS) ])
    return [w_,m.objVal]

def subgrad_H_wrt_w(w,*args): 
    '''
    Gradient of: 
    \max_{g \in \Wset}   \{  \sum_k 
    \abs{w_k \Pstat_b(k) - \sum_j w_j \Pstatcompactb_{j,a,k} \pi^e_{a,j} g_k(a\mid j)) }   $$
    and let $m_k =  {w_k \Pstat_b(k) - \sum_j w_j \Pstatcompactb_{j,a,k} \pi^e_{a,j} g_k(a\mid j)) } 
    with respect to w
    Requires optimal g for computation 
    '''
    data = args[0] 
    a_bnd = data['a_bnd']; b_bnd = data['b_bnd']
    p_infty_b_s = data['pbs'] ; p_e_s = data['p_e_s']
    s_a_giv_sprime = data['s_a_giv_sprime'] 
    g = data['g']
    nSmarg = len(p_infty_b_s)
    nA = len(p_e_s)

    A = np.zeros([nSmarg,nSmarg]) # - np.eye(nSmarg)
    for k in range(nSmarg): 
        for j in range(nSmarg): 
            A[k,j] += sum( [ s_a_giv_sprime[j,a,k]*p_infty_b_s[k] * (p_e_s[a,j] *g[k,a,j]) for a in range(nA)] )
        A[k,k] -= p_infty_b_s[k]

    m = np.dot(A, w)
    subgrad_wrt_w = np.zeros(nSmarg)
    # \partial_{w_j} H = \sum_k \op{sgn}(m_k) ( \Pstatcompactb_k \mathbb{I}[k=j] - \sum_a \Pstatcompactb_{j,a,k}   \pi^e_{a,j} g_{k,a,j} ) 
    for k in range(nSmarg):  
        subgrad_wrt_w[k] = sum([np.sign(m[kprime]) * (p_infty_b_s[kprime]*(kprime== k)
         - sum([s_a_giv_sprime[j,a,k]*p_infty_b_s[k]*p_e_s[a,j]*g[k,a,j] for a in range(nA)])  )  for kprime in range(nSmarg)])
    return subgrad_wrt_w



def max_G_primal_scalarized_L1_for_saddle(gamma, w, a_bnd,b_bnd, s_a_giv_sprime, p_infty_b, pe_s,nS, nA, tight= True, quiet = True):
    # maximize KKT residuals over g
    for k in range(nS): 
        assert np.isclose((s_a_giv_sprime[:,:,k].sum()), 1,atol = 0.01)
    m = gp.Model()
    p_infty_b = p_infty_b.flatten()
#     w = m.addVars(nS)
    g = m.addVars(nS,nA,nS) #\beta_k(a\mid j)
    z = m.addVars(nS) 
    if quiet: m.setParam("OutputFlag", 0)
    tight = True
    epsilon = 0.5

    for k in range(nS): 
        m.addConstr( z[k] >= (-1*w[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
        m.addConstr( z[k] >= -1*(-1*w[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
#         m.addConstr( 0 == (1-gamma)*(1-gp.quicksum(w))+   gamma*(-1*w[k] + gp.quicksum( sum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
    for k in range(nS): 
        for a in range(nA): 
            for j in range(nS): 
                m.addConstr(g[k,a,j] <= b_bnd[a,j])
                m.addConstr(g[k,a,j] >= a_bnd[a,j])

    for a in range(nA): 
        if tight:
            m.addConstr(gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS) ] ) == 1)
        else: 
            m.addConstr(gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS)] ) - 1 <= epsilon)
            m.addConstr(1 - gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS)] )  <= epsilon)
    m.update()
    expr = 0
    # gp.quicksum(z)
    m.setObjective(expr,gp.GRB.MAXIMIZE)
    m.optimize()
    if (m.status == gp.GRB.OPTIMAL): 
        g_ = np.asarray([g[k,a,j].x for k in range(nS) for a in range(nA) for j in range(nS) ]).reshape([nS,nA,nS])
        return [m.objVal, g_]
    else: 
        g = None
        return [None, g]




def alternating(Phi, g, w, N_RNDS, data, eta_0=1, eta_step_schedule=1.4, sigma_step_schedule = 0.6, gamma = 1):
    # min w on the inside 
    # max g on the outside 
    [a_bnd,b_bnd, s_a_giv_sprime, p_infty_b, p_e_s, p_a1_s] = data
    nS = len(p_infty_b); nA = len(p_e_s)
    risks = np.zeros(N_RNDS)
    THTS = [None] * N_RNDS; PARAMS = [None] * N_RNDS; losses = [None] * N_RNDS
    gs = [None] * N_RNDS; 
    quiet = True
    # initialize randomly in [a_, b_]
    for k in range(N_RNDS):
        eta_t = eta_0 *  np.power((k + 1) * 1.0, eta_step_schedule)
        print eta_t
        # Minimize residuals wrt g over w 
        [obj_phival, residuals, w] = saddle_outer_min_w(gamma, g, Phi, eta_t, a_bnd,b_bnd, s_a_giv_sprime, p_infty_b, p_e_s, p_a1_s,nS, nA)
        print 'obj phi', obj_phival, ', residuals: ', residuals
        # Maximize over g
        print w
        [objVal, g] = max_G_primal_scalarized_L1_for_saddle(gamma, w, a_bnd,b_bnd, s_a_giv_sprime, p_infty_b, p_e_s, nS, nA, quiet = quiet)
        gs[k] = g
        print g
        THTS[k] = w; losses[k] = obj_phival
    return [losses, gs, THTS]

def saddle_inner_min_w(gamma, g, eta, *args): 
    data = args[0]
    a_bnd = data['a_bnd']; b_bnd = data['b_bnd']
    Phi = data['Phi']
    p_infty_b = data['pbs'] ; p_e_s = data['p_e_s']
    nS = len(p_infty_b)
    nA = len(p_e_s)
    s_a_giv_sprime = data['s_a_giv_sprime'] 
    sense_min = data['sense_min']
    for k in range(nS): 
        assert np.isclose((s_a_giv_sprime[:,:,k].sum()), 1,atol = 0.01)
    assert np.isclose(sum(p_infty_b), 1)
    m = gp.Model()
    p_infty_b = p_infty_b.flatten()
    w = m.addVars(nS)
    z = m.addVars(nS) 
    quiet = True
    if quiet: m.setParam("OutputFlag", 0)

    for k in range(nS): 
        m.addConstr( z[k] >= (-1*w[k]*p_infty_b[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k]*p_infty_b[k] * (p_e_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
        m.addConstr( z[k] >= -1*(-1*w[k]*p_infty_b[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k]*p_infty_b[k] * (p_e_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
#         m.addConstr( 0 == (1-gamma)*(1-gp.quicksum(w))+   gamma*(-1*w[k] + gp.quicksum( sum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
    m.addConstr( gp.quicksum(w[k] *p_infty_b[k] for k in range(nS) ) == 1)
    m.update()
    sense_min01 = 1 if sense_min else -1
    expr_proj =  gp.quicksum(z) #+ sense_min01 * gp.quicksum( w[k] *Phi[k]*p_infty_b[k] for k in range(nS))
    m.setObjective(expr_proj, gp.GRB.MINIMIZE)

    # expr = eta * gp.quicksum(z) + sense_min01 * gp.quicksum( w[k] *Phi[k]*p_infty_b[k] for k in range(nS))
    # m.setObjective(expr, gp.GRB.MINIMIZE)
    m.optimize()
    if (m.status == gp.GRB.OPTIMAL): 
        w_ = np.asarray([w[k].x for k in range(nS)  ])
        z_ = np.asarray([z[k].x for k in range(nS)  ])
        wphi_obj = np.dot(w_,Phi)
        return [wphi_obj, z_.sum() , w_]
    else:
        return None




def proj_grad_descent_smoothed(g, N_RNDS, data, 
    eta_0=1, step_schedule=0.5, sigma_0 = 0.5, sigma_step_schedule = 0.5, sense_min = True, gamma = 1,
    quiet = True):
    # solve penalized inner minimization over w 
    risks = np.zeros(N_RNDS)
    THTS = [None] * N_RNDS; PARAMS = [None] * N_RNDS; losses = [None] * N_RNDS
    gs_proj = [None] * N_RNDS; 
    gs_init = [None] * N_RNDS; residuals_ = [None] * N_RNDS
    a_bnd = data['a_bnd']; b_bnd = data['b_bnd']
    p_infty_b_s = data['pbs'] ; p_e_s = data['p_e_s']
    nS = len(p_infty_b_s)
    nA = len(p_e_s)
    Phi = data['Phi']

    s_a_giv_sprime = data['s_a_giv_sprime'] 
    # initialize randomly in [a_, b_]
    for k in range(N_RNDS):
        eta_t = eta_0 * 1.0 / np.power((k + 1) * 1.0, step_schedule)
        sigma_t = sigma_0 * 1.0 / np.power((k + 1) * 1.0, step_schedule)
# Project before taking gradient steps
        [obj_phival, residuals, w] = saddle_inner_min_w(gamma, g, eta_t, *[data])
        [feas, g_] = primal_feasibility_testg(gamma, w,g , a_bnd, b_bnd, s_a_giv_sprime, p_infty_b_s, p_e_s,nS, nA)
        if g_ is not None: 
        # if feasibility oracle 
            [g_grad,theta] = grad_H_wrt_g_explicit(g_, *[data])
        else: 
            [g_grad,theta] = grad_H_wrt_g_explicit(g, *[data])

        if not quiet: 
            print 'eta', eta_t
            print 'w', w, 'w-norm, ', w/w.sum()
            print 'residuals, ', residuals
            print feas
            print g_
            print 'th,', theta / theta.sum()
        # default to max behavior
        if sense_min == True:  # maximize the negative of  
            g = -g
            g_grad = -1*g_grad
            losses[k] = -1*np.dot(theta,Phi)
        losses[k] = np.dot(theta,Phi)
        if g_ is not None: 
            g_step = g_ + sigma_t * g_grad; 
        else: 
            g_step = g + sigma_t * g_grad; 
        # check if this step introduces cycling
        gs_init[k] = g_step
        [g_proj, proj_val] = proj_g_(g_step, *[data])
        gs_proj[k] = g_proj; residuals_[k] = residuals
        g = g_proj
        THTS[k] = w; 
    return [losses, gs_init, gs_proj, THTS, residuals_]

def grad_H_wrt_g_explicit(g, *args): 
    ''' Evaluate gradient
    Assume g is [k,a,j]
     -({I}[j /= |S|]  pi^e_a,j Pb_{j,a,k}) (\E[A']^{-\top } \Phi \theta^\top)_{i,j}
    '''
    data = args[0]
    Phi = data['Phi'] # state
    p_infty_b_s = data['pbs'] 
    nSmarg = len(p_infty_b_s)
    s_a_giv_sprime = data['s_a_giv_sprime'] 
    p_e_s = data['p_e_s'];     nA = len(p_e_s); check_grad = data['check_grad']
    A = np.zeros([nSmarg,nSmarg])
    for k in range(nSmarg): 
        for j in range(nSmarg): 
            A[k,j] += sum( [ s_a_giv_sprime[j,a,k]*p_infty_b_s[k] * (p_e_s[a,j] *g[k,a,j] ) for a in range(nA)] )
        A[k,k] -= p_infty_b_s[k]
    tildeA = A
    # Make regular and add normalization 
    tildeA[-1,:] = p_infty_b_s
    v = np.zeros([nSmarg]); v[-1] = 1 
    theta = np.linalg.inv(tildeA).dot(v)
    # print 'theta from grad comp,', theta
    # gradient specific operations 
    outer_ = np.outer(Phi, theta)
    M = np.linalg.inv(tildeA).dot(outer_)
    g_grad = np.zeros(g.shape)
    s_a_giv_sprime = data['s_a_giv_sprime'] ;p_e_s = data['p_e_s'];    p_infty_b_s = data['pbs'] 
    for k in range(g_grad.shape[0]): 
        for a in range(g_grad.shape[1]): 
            for j in range(g_grad.shape[2])[:-1]: # omit last row of J  
                g_grad[k,a,j] = -1*M[j,k]*s_a_giv_sprime[j,a,k]*p_infty_b_s[k]*p_e_s[a,j]
                # j,k or k,j ? 
    return [g_grad,theta]


def primal_feasibility_testg(gamma, w, g_tilde, a_bnd,b_bnd, s_a_giv_sprime, p_infty_b_s, pe_s, 
                       nS, nA, tight= True, quiet = True):
    # use test function I[ s=k  ] (aka solve with tthe unconditional expectation )
    # project in L2 norm of g 
    for k in range(nS): 
        assert np.isclose((s_a_giv_sprime[:,:,k].sum()), 1,atol = 0.01)
    assert np.isclose(sum(p_infty_b_s), 1)
    m = gp.Model()
    p_infty_b_s = p_infty_b_s.flatten()
#     w = m.addVars(nS)
    g = m.addVars(nS,nA,nS) #\beta_k(a\mid j)
    if quiet: m.setParam("OutputFlag", 0)
    epsilon = 0.5

    for k in range(nS): 
        m.addConstr( 0 == (-1*w[k]*p_infty_b_s[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k]*p_infty_b_s[k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
#         m.addConstr( 0 == (1-gamma)*(1-gp.quicksum(w))+   gamma*(-1*w[k] + gp.quicksum( sum([w[j]*s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
    for k in range(nS): 
        for a in range(nA): 
            for j in range(nS): 
                m.addConstr(g[k,a,j] <= b_bnd[a,j])
                m.addConstr(g[k,a,j] >= a_bnd[a,j])
    for a in range(nA): 
        if tight:
            m.addConstr(gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b_s[k] for k in range(nS) for j in range(nS) ] ) == 1)
        else: 
            m.addConstr(gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b_s[k] for k in range(nS) for j in range(nS)] ) - 1 <= epsilon)
            m.addConstr(1 - gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b_s[k] for k in range(nS) for j in range(nS)] )  <= epsilon)
    m.update()
    obj = gp.quicksum( (g[k,a,j]*g[k,a,j] - 2*g[k,a,j]*g_tilde[k,a,j] + g_tilde[k,a,j]*g_tilde[k,a,j] for k in range(nS)for a in range(nA) for j in range(nS)))
    m.setObjective(obj, gp.GRB.MINIMIZE)

    m.optimize()
    if (m.status == gp.GRB.OPTIMAL): 
        feasibility = True
        g = np.asarray([g[k,a,j].x for k in range(nS) for a in range(nA) for j in range(nS) ]).reshape([nS,nA,nS])
    else: 
        feasibility = False
        g = None
#     g_ = m.getAttr('x', g)
    return [feasibility, g]



def opt_w_restarts(N_RST, N_RNDS, data_, g0,
    logging=False, step_schedule=0.5, sigma_step_schedule = 0.5):
    # default is maximization 
    ls = np.zeros(N_RST)
    gamma = data_['gamma']
    ths = [None] * N_RST;best_gs = [None] * N_RST
    iterator = log_progress(range(N_RST), every=1) if logging else range(N_RST)
    a_bnd = data_['a_bnd']; b_bnd = data_['b_bnd']
    p_infty_b_s = data_['pbs'] ; p_e_s = data_['p_e_s']
    nS = len(p_infty_b_s); nA = len(p_e_s); Phi = data_['Phi']
    s_a_giv_sprime = data_['s_a_giv_sprime'] 
    for j in iterator:
        # initialize feasible w by projecting onto g, w, g
        if j == 0 and g0 is not None: # if handed an initial iterate: include as one of the restarts
            [g_proj, resid] = proj_g_(g0, *[data_])
        else:
            g_init = random_g(a_bnd,b_bnd)
            [g_proj, resid] = proj_g_(g_init, *[data_])

        if g_proj is None: 
            return [None, None, None]
        [obj_phival, residuals, w] = saddle_inner_min_w(gamma, g_proj, 0, *[data_])
        [feas, g_proj] = primal_feasibility_testg(gamma, w,g_proj , a_bnd, b_bnd, s_a_giv_sprime, p_infty_b_s, p_e_s,nS, nA)
        # print g_proj
        if g_proj is None: 
            return [None, None, None]
        [losses, gs_init, gs_proj, THTS, residuals_] = proj_grad_descent_smoothed(g_proj, 
            N_RNDS, *[data_], eta_0 = 1000,step_schedule=step_schedule,sigma_step_schedule = sigma_step_schedule)
        # return the best so far, from this initialization not last
        best_so_far = np.argmax(losses)
        ls[j] = losses[best_so_far]
        ths[j] = THTS[best_so_far]; best_gs[j] = gs_proj[best_so_far]
        if logging:
            plt.plot(range(N_RNDS), losses)
            plt.pause(0.05)
        # return best of restarts
    return [ths[np.argmax(ls)], max(ls), best_gs[np.argmax(ls)]]  # return tht achieving min loss

def proj_grad_descent_smoothed_initialize(g, N_RNDS, j,data, 
    eta_0=1, step_schedule=0.5, sigma_0 = 0.5, sigma_step_schedule = 0.5, sense_min = True, gamma = 1,
    quiet = True):
    # solve penalized inner minimization over w 
    risks = np.zeros(N_RNDS) ;THTS = [None] * N_RNDS; PARAMS = [None] * N_RNDS; losses = [None] * N_RNDS; gs_proj = [None] * N_RNDS; gs_init = [None] * N_RNDS; residuals_ = [None] * N_RNDS
    a_bnd = data['a_bnd']; b_bnd = data['b_bnd']
    p_infty_b_s = data['pbs'] ; p_e_s = data['p_e_s']; nS = len(p_infty_b_s); nA = len(p_e_s); Phi = data['Phi']; s_a_giv_sprime = data['s_a_giv_sprime'] 
     # if handed an initial iterate: include as one of the restarts
    np.random.seed(j)
    seed(j)
    if j == 0 and g is not None:
        [g_proj, resid] = proj_g_(g, *[data])
    else:
        g_init = random_g(a_bnd,b_bnd)
        [g_proj, resid] = proj_g_(g_init, *[data])
    if g_proj is None: 
        return [None, None, None, None, None]
    [obj_phival, residuals, w] = saddle_inner_min_w(gamma, g_proj, 0, *[data])
    [feas, g] = primal_feasibility_testg(gamma, w, g_proj , a_bnd, b_bnd, s_a_giv_sprime, p_infty_b_s, p_e_s,nS, nA)
    if g is None: # print g_proj
        return [None, None, None, None, None]
    # initialize randomly in [a_, b_]
    for k in range(N_RNDS):
        eta_t = eta_0 * 1.0 / np.power((k + 1) * 1.0, step_schedule)
        sigma_t = sigma_0 * 1.0 / np.power((k + 1) * 1.0, step_schedule)
# Project before taking gradient steps
        [obj_phival, residuals, w] = saddle_inner_min_w(gamma, g, 0, *[data])
        [feas, g_] = primal_feasibility_testg(gamma, w, g , a_bnd, b_bnd, s_a_giv_sprime, p_infty_b_s, p_e_s,nS, nA)
        if g_ is not None: # if feasibility oracle 
            [g_grad,theta] = grad_H_wrt_g_explicit(g_, *[data])
        else: 
            [g_grad,theta] = grad_H_wrt_g_explicit(g, *[data])
        # default to max behavior
        losses[k] = np.dot(theta,Phi)
        if losses[k] > 5: 
            print theta
            print Phi
            print w
            print g_
            break

        if g_ is not None: 
            g_step = g_ + sigma_t * g_grad; 
        else: 
            g_step = g + sigma_t * g_grad; 
        # check if this step introduces cycling
        gs_init[k] = g_step
        [g_proj, proj_val] = proj_g_(g_step, *[data])
        gs_proj[k] = g_proj; residuals_[k] = residuals
        g = g_proj
        THTS[k] = w; 
    best_so_far = np.argmax(losses)
    return [losses[best_so_far], gs_init[best_so_far], gs_proj[best_so_far], THTS[best_so_far], residuals_[best_so_far] ]

    # return [losses, gs_init, gs_proj, THTS, residuals_]

def opt_w_restarts_parallel(N_RST, N_RNDS, data_, g0,
    logging=False, step_schedule=0.5, sigma_step_schedule = 0.5, vbs = 10):
    # default is maximization 
    ls = np.zeros(N_RST)
    gamma = data_['gamma']
    ths = [None] * N_RST;best_gs = [None] * N_RST
    iterator = log_progress(range(N_RST), every=1) if logging else range(N_RST)
    a_bnd = data_['a_bnd']; b_bnd = data_['b_bnd']
    p_infty_b_s = data_['pbs'] ; p_e_s = data_['p_e_s']
    nS = len(p_infty_b_s); nA = len(p_e_s); Phi = data_['Phi']
    s_a_giv_sprime = data_['s_a_giv_sprime'] 

    res_ = Parallel(n_jobs=12, verbose = vbs)(delayed(proj_grad_descent_smoothed_initialize)(g0, 
            N_RNDS, j,  *[data_], eta_0 = 1000,step_schedule=step_schedule,sigma_step_schedule = sigma_step_schedule) for j in range(N_RST))

    feasible_ = ([ True if res_[j][4] is not None else False for j in range(N_RST) ])
    if sum(feasible_) > 0: 
        # res__ = [res_[k] for k in np.where(feasible_)[0] ]
        # nfeas = sum(feasible_)
        # res object is losses, gs_init, gs_proj, THTS, residuals_
        losses = [res_[j][0] for j in np.where(feasible_)[0] ]
        print losses
        best_so_far_ = np.argmax(losses) # which initialization was best 
        best_so_far_orig = np.where(feasible_)[0][best_so_far_]
        ls = losses[best_so_far_]
        best_th = res_[best_so_far_orig][3]
        print best_th
        best_g = res_[best_so_far_orig][2]

        return [best_th, ls, best_g]  # return tht, loss, best-g achieving max loss
    else: 
        return [None, None, None]

        # [losses, gs_init, gs_proj, THTS, residuals_] = proj_grad_descent_smoothed(g_proj, 
        #     N_RNDS, *[data_], eta_0 = 1000,step_schedule=step_schedule,sigma_step_schedule = sigma_step_schedule)
        # # return the best so far, from this initialization not last
        # best_so_far = np.argmax(losses)
        # ls[j] = losses[best_so_far]
        # ths[j] = THTS[best_so_far]; best_gs[j] = gs_proj[best_so_far]

def get_bounds_pgd(logGams,N_RST,N_RNDS, p_a1_s, *args):
    # Get bounds for all gamma parameter values
    data_ = args[0]
    ngams = len(logGams)
    min_pgd_bnds = [None] * ngams; w_min_pgd_bnds = [None] * ngams
    max_pgd_bnds = [None] * ngams; w_max_pgd_bnds = [None] * ngams
    Phi = data_['Phi']
    for ind,logGam in enumerate(logGams): 
        print logGam
        data_['Phi'] = Phi
        [a_bnd, b_bnd] = get_bnds_as( p_a1_s, logGam ); data_['a_bnd']=a_bnd; data_['b_bnd']=b_bnd
        # initialize 
        g_init = random_g(a_bnd,b_bnd)
        [g_proj, resid] = proj_g_(g_init, *[data_])
        g0_max = g_proj; g0_min = g_proj
        [th, ls, g0_max] = opt_w_restarts(N_RST, N_RNDS, data_, g0_max, logging = True)
        print th, ls
        max_pgd_bnds[ind] = ls; w_max_pgd_bnds[ind] = th; 
        data_['Phi'] = -Phi
        [th, ls, g0_min] = opt_w_restarts(N_RST, N_RNDS, data_, g0_min, logging = True)
        print th, ls
        min_pgd_bnds[ind] = ls; w_min_pgd_bnds[ind] = th; 
    return [ w_min_pgd_bnds, w_max_pgd_bnds ]



def get_bounds_pgd_parallelize_gammas(logGam, N_RST,N_RNDS, p_a1_s, Phi, *args):
    # Get bounds for all gamma parameter values
    data_ = args[0]
    data__ = deepcopy(data_)
    data__['Phi'] = Phi
    [a_bnd, b_bnd] = get_bnds_as( p_a1_s, logGam ); data__['a_bnd']=a_bnd; data__['b_bnd']=b_bnd
    # initialize 
    g_init = random_g(a_bnd,b_bnd)
    [g_proj, resid] = proj_g_(g_init, *[data__])
    g0_max = g_proj; g0_min = g_proj
    [th, ls, g0_max] = opt_w_restarts(N_RST, N_RNDS, data__, g0_max, logging = False)
    print 'gamma ', logGam, th, ls #max_pgd_bnds[ind] = ls; 
    w_max_pgd_bnd = th; 
    data__['Phi'] = -1*Phi
    [th, ls, g0_min] = opt_w_restarts(N_RST, N_RNDS, data__, g0_min, logging = False)
    print 'gamma ', logGam, th, ls #min_pgd_bnds[ind] = ls; 
    w_min_pgd_bnd  = th; 
    return [ w_min_pgd_bnd, w_max_pgd_bnd ]

def parallelize_pgd_over_gamma_helper(logGams,N_RST,N_RNDS, p_a1_s, Phi, data_, sigma_step_schedule = 0.5): 

    res_ = Parallel(n_jobs=12, verbose = 20)(delayed(get_bounds_pgd_parallelize_gammas)(logGam, N_RST, 
            N_RNDS, p_a1_s, Phi, *[data_]) for logGam in logGams)
    return res_ 


def get_bounds_pgd_parallel(logGams,N_RST,N_RNDS, p_a1_s, *args):
    data_ = args[0]
    ngams = len(logGams)
    min_pgd_bnds = [None] * ngams; w_min_pgd_bnds = [None] * ngams
    max_pgd_bnds = [None] * ngams; w_max_pgd_bnds = [None] * ngams
    Phi = data_['Phi']
    for ind,logGam in enumerate(logGams): 
        print logGam
        data_['Phi'] = Phi
        [a_bnd, b_bnd] = get_bnds_as( p_a1_s, logGam ); data_['a_bnd']=a_bnd; data_['b_bnd']=b_bnd
        # initialize 
        if ind == 0:
            g_init = random_g(a_bnd,b_bnd)
            [g_proj, resid] = proj_g_(g_init, *[data_])
            g0_max = g_proj; g0_min = g_proj
        [th, ls, g0_max] = opt_w_restarts_parallel(N_RST, N_RNDS, data_, g0_max, logging = False)
        print th, ls
        max_pgd_bnds[ind] = ls; w_max_pgd_bnds[ind] = th; 
        data_['Phi'] = -Phi
        [th, ls, g0_min] = opt_w_restarts_parallel(N_RST, N_RNDS, data_, g0_min, logging = False)
        print th, ls
        min_pgd_bnds[ind] = ls; w_min_pgd_bnds[ind] = th; 
    return [ w_min_pgd_bnds, w_max_pgd_bnds ]

def primal_opt_outer_L1_(gamma, phi, a_bnd,b_bnd, s_a_giv_sprime, p_infty_b, pe_s, p_a1_s,
                       nS, nA, tight= True, sense_min = True, quiet = True):
    for k in range(nS): 
        assert np.isclose((s_a_giv_sprime[:,:,k].sum()), 1,atol = 0.01)
    assert np.isclose(sum(p_infty_b), 1)
    m = gp.Model()
    p_infty_b = p_infty_b.flatten()
#     w = m.addVars(nS)
    g = m.addVars(nS,nA,nS) #\beta_k(a\mid j)
    z = m.addVars(nS) 
    w = m.addVars(nS) 
    if quiet: m.setParam("OutputFlag", 0)
    epsilon = 0.1
    m.params.NonConvex = 2

    m.addConstr( gp.quicksum(z) == 0 )
    for k in range(nS): 
        m.addConstr( z[k] >= (-1*w[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
        m.addConstr( z[k] >= -1*(-1*w[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
#         m.addConstr( 0 == (1-gamma)*(1-gp.quicksum(w))+   gamma*(-1*w[k] + gp.quicksum( sum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
    for k in range(nS): 
        for a in range(nA): 
            for j in range(nS): 
                m.addConstr(g[k,a,j] <= b_bnd[a,j])
                m.addConstr(g[k,a,j] >= a_bnd[a,j])
    m.addConstr( gp.quicksum(w) >= 0.1) 
    m.addConstr( gp.quicksum(w) == 1) 
    for a in range(nA): 
        if tight:
            m.addConstr(gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS) ] ) == 1)
        else: 
            m.addConstr(gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS)] ) - 1 <= epsilon)
            m.addConstr(1 - gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS)] )  <= epsilon)
    m.update()
    expr = gp.LinExpr();
    if sense_min: 
        expr += gp.quicksum( [w[k] * phi[k] for k in range(nS)] )
    else: 
        expr += gp.quicksum( [-1*w[k] * phi[k] for k in range(nS)] )
    m.setObjective(expr, gp.GRB.MINIMIZE);
    m.optimize()
    if (m.status == gp.GRB.OPTIMAL): 
        w_ = [w[k].x for k in range(nS)]
        if sense_min: 
            return [m.objVal, w_, m]
        else: 
            return [-1*m.objVal, w_, m]
    else: 
        return [None, None, None]


def primal_opt_outer_L1_test_function(gamma, phi, a_bnd,b_bnd, s_a_giv_sprime, p_infty_b, pe_s, p_a1_s,
                       nS, nA, tight= True, sense_min = True, quiet = True):
    '''
    Optimize over primal with L1 feasibility oracle; with p_infty_b() test function
    '''

    for k in range(nS): 
        assert np.isclose((s_a_giv_sprime[:,:,k].sum()), 1,atol = 0.01)
    assert np.isclose(sum(p_infty_b), 1)
    m = gp.Model()
    p_infty_b = p_infty_b.flatten()
#     w = m.addVars(nS)
    g = m.addVars(nS,nA,nS, name='g') #\beta_k(a\mid j)
    z = m.addVars(nS,name='z') 
    w = m.addVars(nS,name='w') 
    if quiet: m.setParam("OutputFlag", 0)
    epsilon = 0.1
    m.params.NonConvex = 2

    m.addConstr( gp.quicksum(z) == 0 )
    for k in range(nS): 
        m.addConstr( z[k] >= (-1*w[k]*p_infty_b[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k]*p_infty_b[k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
        m.addConstr( z[k] >= -1*(-1*w[k]*p_infty_b[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k]*p_infty_b[k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
#         m.addConstr( 0 == (1-gamma)*(1-gp.quicksum(w))+   gamma*(-1*w[k] + gp.quicksum( sum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
    for k in range(nS): 
        for a in range(nA): 
            for j in range(nS): 
                m.addConstr(g[k,a,j] <= b_bnd[a,j])
                m.addConstr(g[k,a,j] >= a_bnd[a,j])
    m.addConstr( gp.quicksum(w) >= 0.1) 
    m.addConstr( gp.quicksum(w[k] * p_infty_b[k] for k in range(nS)) == 1) 
    for a in range(nA): 
        if tight:
            m.addConstr(gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS) ] ) == 1)
        else: 
            m.addConstr(gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS)] ) - 1 <= epsilon)
            m.addConstr(1 - gp.quicksum([g[k,a,j] *s_a_giv_sprime[j,a,k]*p_infty_b[k] for k in range(nS) for j in range(nS)] )  <= epsilon)
    m.update()
    expr = gp.LinExpr();
    if sense_min: 
        expr += gp.quicksum( [w[k] * phi[k]*p_infty_b[k] for k in range(nS)] )
    else: 
        expr += gp.quicksum( [-1*w[k] * phi[k]*p_infty_b[k] for k in range(nS)] )
    m.setObjective(expr, gp.GRB.MINIMIZE);
    m.optimize()
    if (m.status == gp.GRB.OPTIMAL): 
        w_ = [w[k].x for k in range(nS)]
        if sense_min: 
            return [m.objVal, w_, m]
        else: 
            return [-1*m.objVal, w_, m]
    else: 
        return [None, None, None]



def primal_opt_outer_L1_test_function_joint_distn(gamma, phi, a_bnd,b_bnd, joint_s_a_sprime, p_infty_b, pe_s, 
                       nS, nA, tight= True, sense_min = True, quiet = True):
    '''

    instrument function
    next-state conditional control variates
    '''
    assert np.isclose((joint_s_a_sprime.sum()), 1,atol = 0.01)
    print nS
    # assert np.isclose((p_infty_b.sum()), 1,atol = 0.01)
    m = gp.Model()
    p_infty_b = p_infty_b.flatten()
#     w = m.addVars(nS)
    g = m.addVars(nS,nA,nS) #\beta_k(a\mid j)
    z = m.addVars(nS) 
    w = m.addVars(nS) 
    if quiet: m.setParam("OutputFlag", 0)
    epsilon = 0.02
    m.params.NonConvex = 2

    epsilon_ka = m.addVars(nS,nA)
    epsilon_ka_sum = 0.05

    # get p^infty (k | a) 
    p_infty_a = joint_s_a_sprime.sum(axis=(0,2))
    joint_ak = joint_s_a_sprime.sum(axis=0)
    p_infty_b_k_given_a = np.zeros([nS,nA])
    for k in range(nS): 
        for a in range(nA):
            p_infty_b_k_given_a[k,a] = joint_ak[a,k] / p_infty_a[a]

    m.addConstr( gp.quicksum(z) == 0 )
    for k in range(nS): 
        m.addConstr( z[k] >= (-1*w[k]*p_infty_b[k] + gp.quicksum( w[j]*gp.quicksum([joint_s_a_sprime[j,a,k]* (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
        m.addConstr( z[k] >= -1*(-1*w[k]*p_infty_b[k] + gp.quicksum( w[j]*gp.quicksum([joint_s_a_sprime[j,a,k]* (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
    for k in range(nS): 
        for a in range(nA): 
            for j in range(nS): 
                m.addConstr(g[k,a,j] <= b_bnd[a,j])
                m.addConstr(g[k,a,j] >= a_bnd[a,j])
    m.addConstr( gp.quicksum(w) >= 0.1) 
    m.addConstr( gp.quicksum(w[k] * p_infty_b[k] for k in range(nS)) == 1) 
    for k in range(nS): 
        for a in range(nA): 
            if tight:
                m.addConstr(gp.quicksum([g[k,a,j] *joint_s_a_sprime[j,a,k] for j in range(nS) ] ) == p_infty_b_k_given_a[k,a])
            else: 
                m.addConstr(gp.quicksum([g[k,a,j] *joint_s_a_sprime[j,a,k] for j in range(nS) ] ) - p_infty_b_k_given_a[k,a] <= epsilon_ka[k,a])
                m.addConstr(p_infty_b_k_given_a[k,a] - gp.quicksum([g[k,a,j] *joint_s_a_sprime[j,a,k] for j in range(nS) ] )  <= epsilon_ka[k,a])
    m.addConstr( gp.quicksum(epsilon_ka) <= epsilon_ka_sum )

    epsilon_a = 0.03
    # for a in range(nA): 
    #     if tight:
    #         m.addConstr(gp.quicksum([g[k,a,j] *joint_s_a_sprime[j,a,k] for k in range(nS) for j in range(nS) ] ) == 1)
    #     else: 
    #         m.addConstr(gp.quicksum([g[k,a,j] *joint_s_a_sprime[j,a,k] for k in range(nS) for j in range(nS)] ) - 1 <= epsilon_a)
    #         m.addConstr(1 - gp.quicksum([g[k,a,j] *joint_s_a_sprime[j,a,k] for k in range(nS) for j in range(nS)] )  <= epsilon_a)
    m.update()
    expr = gp.LinExpr();
    if sense_min: 
        expr += gp.quicksum( [w[k] * phi[k]*p_infty_b[k] for k in range(nS)] )
    else: 
        expr += gp.quicksum( [-1*w[k] * phi[k]*p_infty_b[k] for k in range(nS)] )
    m.setObjective(expr, gp.GRB.MINIMIZE);
    m.optimize()
    if (m.status == gp.GRB.OPTIMAL): 
        w_ = [w[k].x for k in range(nS)]
        if sense_min: 
            return [m.objVal, w_, m]
        else: 
            return [-1*m.objVal, w_, m]
    else: 
        return [None, None, None]


def plot_bounds(w_min_pgd_bnds, w_max_pgd_bnds, Phi, ngams, nSmarg, logGams,rearrange=True, label = '', color = 'b'): 
# preprocess to remove none values
    feasible_max = np.asarray([True if w_max_pgd_bnds[k] is not None else False for k in range(ngams)]).astype(bool)
    feasible_min = np.asarray([True if w_min_pgd_bnds[k] is not None else False for k in range(ngams)]).astype(bool)
    w_max_pgd_bnds__ = [w_max_pgd_bnds[i] for i in np.where(feasible_max)[0]]
    w_min_pgd_bnds__ = [w_min_pgd_bnds[i] for i in np.where(feasible_min)[0]]
    max_ = np.asarray([np.dot(w_max_pgd_bnds__[k], Phi) for k in range(sum(feasible_max))]); 
    min_ = np.asarray([np.dot(w_min_pgd_bnds__[k], Phi) for k in range(sum(feasible_min))]); 
    # print max_
    # print min_
    if rearrange: 
        cum_max =np.zeros(len(max_)); cum_max[0] = max_[0]
        cum_max[1:] = np.asarray([max(max_[k], max(max_[0:k])) for k in range(len(max_))[1:]])
        cum_min =np.zeros(len(min_)); cum_min[0] = min_[0]
        cum_min[1:] = np.asarray([min(min_[k], min(min_[0:k])) for k in range(len(min_))[1:]])
        plt.plot(logGams[feasible_max],cum_max , label=label, color = color) 
        plt.plot(logGams[feasible_min], cum_min, label=label, color = color) 
        return [cum_min,cum_max]
    else: 
        # Need to have better masking procedure for infeasible gamma
        plt.plot(logGams[feasible_max],max_ , label=label, color = color) 
        plt.plot(logGams[feasible_min ], min_, label=label, color = color) 
        return [min_,max_]
    
####
# Helper functions to generate gridworld trajectories
def run(func):
    func()
    return func

def grid_world_example(grid_size=(3, 3),
                       black_cells=[(5,5)],
                       white_cell_reward=-0.02,
                       green_cell_locs=[(2,2)],
                       red_cell_locs=[(0,1), (0,2), (1,2), (2,0)],
                       green_cell_reward=1.0,
                       red_cell_reward=-1.0,
                       action_lrfb_prob=(.2, .2, .6, 0.), #probability of left right forward backward 
                       start_loc=(0, 0)
                      ):
    '''
    From https://stats.stackexchange.com/questions/339592/how-to-get-p-and-r-values-for-a-markov-decision-process-grid-world-problem  
    '''
    num_states = grid_size[0] * grid_size[1]
    num_actions = 4
    P = np.zeros((num_actions, num_states, num_states))
    R = np.zeros((num_states, num_actions))

    @run
    def fill_in_probs():
        # helpers
        to_2d = lambda x: np.unravel_index(x, grid_size)
        to_1d = lambda x: np.ravel_multi_index(x, grid_size)

        def hit_wall(cell):
            if cell in black_cells:
                return True
            try: # ...good enough...
                to_1d(cell)
            except ValueError as e:
                return True
            return False

        # make probs for each action
        a_up = [action_lrfb_prob[i] for i in (0, 1, 2, 3)]
        a_down = [action_lrfb_prob[i] for i in (1, 0, 3, 2)]
        a_left = [action_lrfb_prob[i] for i in (2, 3, 1, 0)]
        a_right = [action_lrfb_prob[i] for i in (3, 2, 0, 1)]
        actions = [a_up, a_down, a_left, a_right]
        for i, a in enumerate(actions):
            actions[i] = {'up':a[2], 'down':a[3], 'left':a[0], 'right':a[1]}

        # work in terms of the 2d grid representation

        def update_P_and_R(cell, new_cell, a_index, a_prob):
            if cell in green_cell_locs:
                P[a_index, to_1d(cell), to_1d(cell)] = 1.0
                R[to_1d(cell), a_index] = green_cell_reward

#             elif cell in red_cell_locs:
#                 P[a_index, to_1d(cell), to_1d(cell)] = 1.0
#                 R[to_1d(cell), a_index] = red_cell_reward

            elif hit_wall(new_cell):  # add prob to current cell
                P[a_index, to_1d(cell), to_1d(cell)] += a_prob
                R[to_1d(cell), a_index] = white_cell_reward

            else:
                P[a_index, to_1d(cell), to_1d(new_cell)] = a_prob
                R[to_1d(cell), a_index] = white_cell_reward

        for a_index, action in enumerate(actions):
            for cell in np.ndindex(grid_size):
                # up
                new_cell = (cell[0]-1, cell[1])
                update_P_and_R(cell, new_cell, a_index, action['up'])

                # down
                new_cell = (cell[0]+1, cell[1])
                update_P_and_R(cell, new_cell, a_index, action['down'])

                # left
                new_cell = (cell[0], cell[1]-1)
                update_P_and_R(cell, new_cell, a_index, action['left'])

                # right
                new_cell = (cell[0], cell[1]+1)
                update_P_and_R(cell, new_cell, a_index, action['right'])
        for green_cell in green_cell_locs: 
            print green_cell
            P[:, to_1d(green_cell), :] = 0
            P[:, to_1d(green_cell), (0,0)] = 1
    # custom postprocessing in our setting 
    # green cell goes to start state

    return P, R

def westward_wind_grid_world_example(grid_size=(3, 3),
                       black_cells=[(5,5)],
                       white_cell_reward=-0.02,
                       green_cell_locs=[(2,2)],
                       red_cell_locs=[(0,1), (0,2), (1,2), (2,0)],
                       green_cell_reward=1.0,
                       red_cell_reward=-1.0,
                       action_lrfb_prob=(.2, .2, .6, 0.), #probability of left right forward backward 
                       start_loc=(0, 0)
                      ):
    '''
    From https://stats.stackexchange.com/questions/339592/how-to-get-p-and-r-values-for-a-markov-decision-process-grid-world-problem  
    '''
    num_states = grid_size[0] * grid_size[1]
    num_actions = 4
    P = np.zeros((num_actions, num_states, num_states))
    R = np.zeros((num_states, num_actions))

    @run
    def fill_in_probs():
        # helpers
        to_2d = lambda x: np.unravel_index(x, grid_size)
        to_1d = lambda x: np.ravel_multi_index(x, grid_size)

        def hit_wall(cell):
            if cell in black_cells:
                return True
            try: # ...good enough...
                to_1d(cell)
            except ValueError as e:
                return True
            return False

        # make probs for each action
        a_up = [action_lrfb_prob[i] for i in (0, 1, 2, 3)]
        a_down = [action_lrfb_prob[i] for i in (1, 0, 3, 2)]
        a_left = [action_lrfb_prob[i] for i in (2, 3, 1, 0)]
        a_right = [action_lrfb_prob[i] for i in (3, 2, 0, 1)]
        actions = [a_up, a_down, a_left, a_right]
        action_order = { 0:'up', 1:'down', 2:'left', 3:'right' }
        for i, a in enumerate(actions):
            #transition matrix for going 
            actions[i] = {'up':a[2], 'down':a[3], 'left':a[0], 'right':a[1]}
            #westward wind 
#             actions[i][action_order[i]]

        # work in terms of the 2d grid representation

        def update_P_and_R(cell, new_cell, a_index, a_prob):
            if cell in green_cell_locs:
                P[a_index, to_1d(cell), to_1d(cell)] = 1.0
                R[to_1d(cell), a_index] = green_cell_reward
# pass through red cells 
#             elif cell in red_cell_locs:
#                 P[a_index, to_1d(cell), to_1d(cell)] = 1.0
#                 R[to_1d(cell), a_index] = red_cell_reward

            elif hit_wall(new_cell):  # add prob to current cell
                P[a_index, to_1d(cell), to_1d(cell)] += a_prob
                R[to_1d(cell), a_index] = white_cell_reward

            else:
                # unless you go left
                if a_index == 2: # if left
                    P[a_index, to_1d(cell), to_1d(cell)] = a_prob
                else:
                    P[a_index, to_1d(cell), to_1d(new_cell)] = a_prob
#                     P[a_index, to_1d(cell), to_1d(new_cell)] = a_prob
                R[to_1d(cell), a_index] = white_cell_reward

        for a_index, action in enumerate(actions):
            for cell in np.ndindex(grid_size):
                # update each transition 
                # up
                new_cell = (cell[0]-1, cell[1])
                update_P_and_R(cell, new_cell, a_index, 0)

                # down
                new_cell = (cell[0]+1, cell[1])
                update_P_and_R(cell, new_cell, a_index, 0)

                # left
                new_cell = (cell[0], cell[1]-1)
                update_P_and_R(cell, new_cell, a_index, 0)

                # right
                # wind takes you right
                new_cell = (cell[0], cell[1]+1)
                update_P_and_R(cell, new_cell, a_index, 1)
        for green_cell in green_cell_locs: 
            print green_cell
            P[:, to_1d(green_cell), :] = 0
            P[:, to_1d(green_cell), (0,0)] = 1
    # custom postprocessing in our setting 
    # green cell goes to start state

    return P, R

'''
Modifications for F(w), e.g. for linear function approximation 
'''
def primal_F_beta_L1_empirical_expectations(theta, states, a_bnd,b_bnd, pi_s_obsa,
                       feature_coefficients, N,T, tight= True, quiet = True):
    '''
    :param theta:
    :param states:
    :param a_bnd:
    :param b_bnd:
    :param pi_s_obsa:
    :param feature_coefficients:
    :param N:
    :param T:
    :param tight:
    :param sense_min:
    :param quiet:
    :return:
    '''
    m = gp.Model()
    dTheta = len(theta)
    g = m.addVars(N,T) #\beta_k(a\mid j)
    z = m.addVars(dTheta)
    if quiet: m.setParam("OutputFlag", 0)
    epsilon = 0.1
    m.params.NonConvex = 2

    [Psi_theta_NT_tt1, Psi_theta_NT_t1t1, Psi_NT_tt1,  bar_psi, bar_Psi_t1t1] = feature_coefficients

    for k in range(dTheta):
        m.addConstr( z[k] >= 1.0/ (N*(T-1)) * gp.quicksum(gp.quicksum( ([pi_s_obsa[i,t] *  Psi_theta_NT_tt1[i,t] * g[i,t] - Psi_theta_NT_t1t1[i,t] for t in range(T-1)]) for i in range(N) )) )
        m.addConstr( z[k] >= -1.0/ (N*(T-1)) * gp.quicksum(gp.quicksum( ([pi_s_obsa[i,t] *  Psi_theta_NT_tt1[i,t] * g[i,t] - Psi_theta_NT_t1t1[i,t] for t in range(T-1)]) for i in range(N) )) )
    for i in range(N):
        for t in range(T):
                m.addConstr(g[i,t] <= b_bnd[i,t])
                m.addConstr(g[i,t] >= a_bnd[i,t])
    for a in range(nA):
        if tight:
            m.addConstr(1.0/(N*T) * gp.quicksum([g[i,t] * (a_s[i,t] == a).astype(int) for i in range(N) for t in range(T) ] ) == 1)
        else:
            m.addConstr(1.0 / (N * T) * gp.quicksum(
                        [g[i, t] * (a_s[i, t] == a).astype(int) for i in range(N) for t in range(T)]) - 1 <= epsilon)
            m.addConstr(1 - 1.0 / (N * T) * gp.quicksum(
                        [g[i, t] * (a_s[i, t] == a).astype(int) for i in range(N) for t in range(T)])<= epsilon)
    m.update()
    expr = gp.quicksum( z )
    m.setObjective(expr, gp.GRB.MINIMIZE);
    m.optimize()
    if (m.status == gp.GRB.OPTIMAL):
        return [m.objVal, m]
    else:
        return [None, None]

def primal_outerF_beta_L1_empirical_expectations(reward_coefs, feature_coefficients, a_bnd,b_bnd, pi_s_obsa,a_s, N,T,nA,
                                        tight= True, sense_min = True, quiet = True):
    '''
    Preprocess coefficients
    :param psi_fn:
    :param reward_coefs:
    :param a_bnd:
    :param b_bnd:
    :param pi_s_obsa:
    :param feature_coefficients:
    :param N:
    :param T:
    :param tight:
    :param sense_min:
    :param quiet:
    :return:
    '''
    m = gp.Model()
    [Psi_theta_NT_tt1, Psi_theta_NT_t1t1, Psi_NT_tt1,  bar_psi, bar_Psi_t1t1] = feature_coefficients
    dtheta = len(bar_psi)
    g = m.addVars(N,T) #\beta_k(a\mid j)
    z = m.addVars(dtheta)
    theta = m.addVars(dtheta)
    if quiet: m.setParam("OutputFlag", 0)
    epsilon = 0.1
    m.params.NonConvex = 2

    m.addConstr( gp.quicksum(z) <= 5 , 'optimalityresidual')
    for k in range(dtheta):
        m.addConstr(z[k] >= -1*N*T* gp.quicksum(bar_Psi_t1t1[k,l] * theta[l] for l in range(dtheta))
            +  gp.quicksum(
            gp.quicksum(pi_s_obsa[i, t] *gp.quicksum([Psi_NT_tt1[i,t,k,l]*theta[l] for l in range(dtheta)])
              * g[i, t] for t in range(T - 1)) for i in range(N) )
        )
        m.addConstr(z[k] >=
            N*T*gp.quicksum(bar_Psi_t1t1[k,l] * theta[l] for l in range(dtheta))
        - gp.quicksum(
            gp.quicksum(pi_s_obsa[i, t] *gp.quicksum( [Psi_NT_tt1[i,t,k,l]*theta[l] for l in range(dtheta)])
              * g[i, t] for t in range(T - 1))
            for i in range(N)
            )
        )
    # scale NT
    # for k in range(dtheta):
    #     m.addConstr(z[k] >= -1* gp.quicksum(bar_Psi_t1t1[k,l] * theta[l] for l in range(dtheta))
    #         +  1.0 / (N * (T )) * gp.quicksum(
    #         gp.quicksum(pi_s_obsa[i, t] *gp.quicksum([Psi_NT_tt1[i,t,k,l]*theta[l] for l in range(dtheta)])
    #           * g[i, t] for t in range(T - 1)) for i in range(N) )
    #     )
    #     m.addConstr(z[k] >=
    #         gp.quicksum(bar_Psi_t1t1[k,l] * theta[l] for l in range(dtheta))
    #     - 1.0 / (N * (T )) * gp.quicksum(
    #         gp.quicksum(pi_s_obsa[i, t] *gp.quicksum( [Psi_NT_tt1[i,t,k,l]*theta[l] for l in range(dtheta)])
    #           * g[i, t] for t in range(T - 1))
    #         for i in range(N)
    #         )
    #     )
    for i in range(N):
        for t in range(T):
                m.addConstr(g[i,t] <= b_bnd[i,t])
                m.addConstr(g[i,t] >= a_bnd[i,t])
    # m.addConstr( gp.quicksum(theta[k] for k in range(dtheta)) >= 0.1)
    m.addConstr( gp.quicksum( theta[k]*bar_psi[k] for k in range(dtheta)) == 1) # \E[ w(s) ] = 1 where w = \phi(s)^\top \theta

    # for a in range(nA):
    #     if tight:
    #         m.addConstr(1.0 / (N * T) * gp.quicksum(
    #             [g[i, t] * (a_s[i, t] == a).astype(int) for i in range(N) for t in range(T)]) == 1)
    #     else:
    #         m.addConstr(1.0 / (N * T) * gp.quicksum(
    #             [g[i, t] * (a_s[i, t] == a).astype(int) for i in range(N) for t in range(T)]) - 1 <= epsilon)
    #         m.addConstr(1 - 1.0 / (N * T) * gp.quicksum(
    #             [g[i, t] * (a_s[i, t] == a).astype(int) for i in range(N) for t in range(T)]) <= epsilon)

    m.update()
    expr = gp.LinExpr();
    if sense_min:
        expr += gp.quicksum( [theta[k] * reward_coefs[k] for k in range(dtheta)] )
    else:
        expr += gp.quicksum( [-1*theta[k] * reward_coefs[k] for k in range(dtheta)] )
    m.setObjective(expr, gp.GRB.MINIMIZE);
    m.params.NonConvex = 2
    m.update()
    m.optimize()

    # Feasibility relaxation
    if m.status != gp.GRB.INFEASIBLE:
        print 'feasibility relaxation'
        m.feasRelax(0, False, None, None, None, [m.getConstrByName('optimalityresidual')], [1])
        m.params.NonConvex = 2
        m.update()
        m.optimize()

    if (m.status == gp.GRB.OPTIMAL):
        theta_ = [theta[k].x for k in range(dtheta)]
        if sense_min:
            return [m.objVal, theta_, m]
        else:
            return [-1*m.objVal, theta_, m]
    else:
        return [None, None, None]


def dual_opt_outer_L1_empirical_expectations(reward_coefs, feature_coefficients, a_bnd,b_bnd, pi_s_obsa,a_s, N,T,nA,
                                        tight= True, sense_min = True, quiet = True):
    m = gp.Model()
    c = m.addVars(N,T, lb=0)
    d = m.addVars(N,T, lb=0)
    [Psi_theta_NT_tt1, Psi_theta_NT_t1t1, Psi_NT_tt1,  bar_psi, bar_Psi_t1t1] = feature_coefficients
    dtheta = len(bar_psi)
    theta = m.addVars(dtheta)
    lmbda01 = m.addVars(dtheta, vtype=gp.GRB.BINARY);
    #lmbda = m.addVars(nS, lb = -1*gp.GRB.INFINITY)
    if tight:
        mu = m.addVars(nA, lb=-1 * gp.GRB.INFINITY)
    m.update()
    if quiet: m.setParam("OutputFlag", 0)
    m.params.NonConvex = 2

    for i in range(N):
        for t in range(T-1):
            m.addConstr(c[i,t] - d[i,t] +
                        - 1.0/(N*T) * pi_s_obsa[i,t] * gp.quicksum(
                (2 * lmbda01[k] - 1)*
                gp.quicksum( Psi_NT_tt1[i,t,k,l]*theta[k] for l in range(dtheta) )
                for k in range(dtheta) )
                        + 1.0/(N*T) *mu[a_s[i,t]]
                        == 0)
    # for k in range(dtheta):
    #     m.addConstr(lmbda[k] <= 1)
    #     m.addConstr(lmbda[k] >= -1)

    m.addConstr( gp.quicksum(c[i,t] * a_bnd[i,t] for i in range(N) for t in range(T) )
                 + gp.quicksum(-1 * d[i,t] * b_bnd[i,t] for i in range(N) for t in range(T))
                 + gp.quicksum(-1 * (2 * lmbda01[k] - 1)*
                gp.quicksum( bar_Psi_t1t1[k,l]*theta[k] for l in range(dtheta) )
                        for k in range(dtheta) )
                + gp.quicksum(mu)
                 == 0 )
    m.update()

    expr = gp.LinExpr()
    if sense_min:
        expr += gp.quicksum( [theta[k] * reward_coefs[k] for k in range(dtheta)] )
    else:
        expr += gp.quicksum( [-1*theta[k] * reward_coefs[k] for k in range(dtheta)] )

    # need to change for discounted case
    m.setObjective(expr, gp.GRB.MINIMIZE);
    m.optimize()

    if (m.status == gp.GRB.OPTIMAL):
        print 'c, ', [c[i,t].x for i in range(N) for t in range(T)]
        print 'd, ', [d[i,t].x for i in range(N) for t in range(T)]
        lmbda_ = [(2 * lmbda01[k].x - 1) for k in range(dtheta)]
        return [m.objVal, lmbda_]
    else:
        return [None, None]



def get_reward_coefs(states, Phi_fn, psi_fn):
    '''
    :param states: N x T x dS
    :param Phi_fn:
    :return:
    '''
    N= states.shape[0]; T = states.shape[1]
    return 1.0/(N*T)*np.sum([ Phi_fn(states[i,t,:]) * psi_fn(states[i,t,:]) for i in range(N) for t in range(T) ],axis=0)



def primal_scalarized_L1_min_resid(gamma, a_bnd,b_bnd, s_a_giv_sprime, p_infty_b, pe_s, p_a1_s,
                       nS, nA, tight= True, quiet = True):
    '''
    opt for closest w for the observed behavior policy
    '''
    for k in range(nS):
        assert np.isclose((s_a_giv_sprime[:,:,k].sum()), 1,atol = 0.01)
    assert np.isclose(sum(p_infty_b), 1)
    m = gp.Model()
    p_infty_b = p_infty_b.flatten()
    w = m.addVars(nS)
    z = m.addVars(nS)
    if quiet: m.setParam("OutputFlag", 0)
    epsilon = 0.5

    for k in range(nS):
        m.addConstr( z[k] >= (-1*w[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * p_a1_s[a,j]) for a in range(nA)]) for j in range(nS) )) )
        m.addConstr( z[k] >= -1*(-1*w[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * p_a1_s[a,j]) for a in range(nA)]) for j in range(nS) )) )
    m.addConstr(gp.quicksum(w[k] * p_infty_b[k] for k in range(nS)) == 1)
    m.update()
    expr = gp.quicksum(z)
    m.setObjective(expr, gp.GRB.MINIMIZE);
    m.optimize()
    if (m.status == gp.GRB.OPTIMAL):
        w_ = np.asarray([w[j].x for j in range(nS) ])
        return [m.objVal, w_]
    else:
        g = None
        return [None, g]

def primal_scalarized_L1_min_resid_given_g(gamma, g, a_bnd,b_bnd, s_a_giv_sprime, p_infty_b, pe_s, p_a1_s,
                       nS, nA, tight= True, quiet = True):
    '''
    opt for closest w for the observed behavior policy
    given g
    '''
    for k in range(nS):
        assert np.isclose((s_a_giv_sprime[:,:,k].sum()), 1,atol = 0.01)
    assert np.isclose(sum(p_infty_b), 1)
    m = gp.Model()
    p_infty_b = p_infty_b.flatten()
    w = m.addVars(nS)
    z = m.addVars(nS)
    if quiet: m.setParam("OutputFlag", 0)
    epsilon = 0.5

    for k in range(nS):
        m.addConstr( z[k] >= (-1*w[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
        m.addConstr( z[k] >= -1*(-1*w[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * g[k,a,j]) for a in range(nA)]) for j in range(nS) )) )
    m.addConstr(gp.quicksum(w[k] * p_infty_b[k] for k in range(nS)) == 1)
    m.update()
    expr = gp.quicksum(z)
    m.setObjective(expr, gp.GRB.MINIMIZE);
    m.optimize()
    if (m.status == gp.GRB.OPTIMAL):
        w_ = np.asarray([w[j].x for j in range(nS) ])
        return [m.objVal, w_]
    else:
        g = None
        return [None, g]

def get_esteqn(w, g, s_a_giv_sprime, p_infty_b_s, p_e_s, p_a1_s, nS):
    est_eqn = np.zeros(nS)
    for k in range(nS):
        est_eqn[k] = np.abs(-1*w[k] *p_infty_b_s[k] + sum( [w[j] * sum([s_a_giv_sprime[j, a, k] * (g[k, a, j] * p_e_s[a, j]) *p_infty_b_s[k] for a in range(nA)]) for j in range(nS)]))
    return est_eqn


def get_epsradius(w, gamma, a_bnd, b_bnd, s_a_giv_sprime, p_infty_b, pe_s, p_a1_s,
                                   nS, nA, tight=True, quiet=True, minimize=True):
    '''
    opt for closest w for the observed behavior policy
    '''
    for k in range(nS):
        assert np.isclose((s_a_giv_sprime[:, :, k].sum()), 1, atol=0.01)
    assert np.isclose(sum(p_infty_b), 1)
    m = gp.Model()
    p_infty_b = p_infty_b.flatten()
    g = m.addVars(nS,nA,nS) #\beta_k(a\mid j)
    z = m.addVars(nS)
    t = m.addVars(nS)
    if quiet: m.setParam("OutputFlag", 0)

    for k in range(nS):
        m.addConstr(t[k] == -1 * w[k] + gp.quicksum(w[j] * gp.quicksum([s_a_giv_sprime[j, a, k] * (pe_s[a, j] * g[k,a, j]) for a in range(nA)])
            for j in range(nS)) )
        m.addConstr(z[k] == gp.abs_( t[k] ) )

    for a in range(nA):
        if tight:
            m.addConstr(gp.quicksum(
                [g[k, a, j] * s_a_giv_sprime[j, a, k] * p_infty_b[k] for k in range(nS) for j in range(nS)]) == 1)
    for k in range(nS):
        for a in range(nA):
            for j in range(nS):
                m.addConstr(g[k,a,j] <= b_bnd[a,j])
                m.addConstr(g[k,a,j] >= a_bnd[a,j])
    m.update()
    expr = gp.quicksum( z )
    if minimize:
        m.setObjective(expr, gp.GRB.MINIMIZE);
    else:
        m.setObjective(expr, gp.GRB.MAXIMIZE);
    m.optimize()

    if (m.status == gp.GRB.OPTIMAL):
        feasibility = True
        g_ = np.asarray([g[k,a,j].x for k in range(nS) for a in range(nA) for j in range(nS) ]).reshape([nS,nA,nS])
        return [m.objVal, g_]
    else:
        g = None
    return [None, g]

def primal_scalarized_L1_min_epsradius(g, epsradius, phi, gamma, a_bnd, b_bnd, s_a_giv_sprime, p_infty_b, pe_s, p_a1_s,
                                   nS, nA, tight=True, quiet=True, sense_min = True):
    '''
    minimize over w within epsilonradius of feasible w,g
    '''
    for k in range(nS):
        assert np.isclose((s_a_giv_sprime[:, :, k].sum()), 1, atol=0.01)
    assert np.isclose(sum(p_infty_b), 1)
    m = gp.Model()
    p_infty_b = p_infty_b.flatten()
    w = m.addVars(nS, lb = 0)
    z = m.addVars(nS)
    t = m.addVars(nS)
    if quiet: m.setParam("OutputFlag", 0)

    for k in range(nS):
        m.addConstr(t[k] == -1 * w[k] *p_infty_b[k] + gp.quicksum(
            w[j] * gp.quicksum([s_a_giv_sprime[j, a, k] * (g[k, a, j] * pe_s[a, j]) *p_infty_b[k] for a in range(nA)]) for j in
            range(nS)))
        m.addConstr(z[k] == gp.abs_(t[k]))


    m.addConstr( gp.quicksum( z[k] for k in range(nS) ) <= epsradius )
    # constraints on w
    m.addConstr( gp.quicksum(w) >= 0.1 )
    m.addConstr( gp.quicksum(w[k] * p_infty_b[k] for k in range(nS)) == 1)
    m.update()
    expr = gp.LinExpr();
    if sense_min:
        expr += gp.quicksum( [w[k] * phi[k]*p_infty_b[k] for k in range(nS)] )

    else:
        expr += gp.quicksum( -1*[w[k] * phi[k]*p_infty_b[k] for k in range(nS)] )
    m.setObjective(expr, gp.GRB.MAXIMIZE);
    m.optimize()
    print [t[k].x for k in range(nS)]
    if (m.status == gp.GRB.OPTIMAL):
        w_ = np.asarray([w[j].x for j in range(nS)])
        return [m.objVal, w_]
    else:
        g = None
    return [None, g]

def primal_l1_check_realizability(w, gamma, a_bnd,b_bnd, s_a_giv_sprime, p_infty_b, pe_s, p_a1_s,
                       nS, nA, tight= True, quiet = True):
    '''
    is w realizable under any g?
    '''
    for k in range(nS):
        assert np.isclose((s_a_giv_sprime[:,:,k].sum()), 1,atol = 0.01)
    assert np.isclose(sum(p_infty_b), 1)
    m = gp.Model()
    p_infty_b = p_infty_b.flatten()
    g = m.addVars( nS, nA, nS, lb = 1,name='g')
    z = m.addVars(nS)
    if quiet: m.setParam("OutputFlag", 0)
    epsilon = 0.5

    for k in range(nS):
        m.addConstr( z[k] >= (-1*w[k]*p_infty_b[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * p_a1_s[a,j])*p_infty_b[k] for a in range(nA)]) for j in range(nS) )) )
        m.addConstr( z[k] >= -1*(-1*w[k]*p_infty_b[k] + gp.quicksum( w[j]*gp.quicksum([s_a_giv_sprime[j,a,k] * (pe_s[a,j] * p_a1_s[a,j])*p_infty_b[k] for a in range(nA)]) for j in range(nS) )) )
    m.addConstr(gp.quicksum(w[k] * p_infty_b[k] for k in range(nS)) == 1)

      # \beta_k(a\mid j)

    for a in range(nA):
        if tight:
            m.addConstr(gp.quicksum(
                [g[k, a, j] * s_a_giv_sprime[j, a, k] * p_infty_b[k] for k in range(nS) for j in range(nS)]) == 1)

    m.update()
    expr = gp.quicksum(z)
    m.setObjective(expr, gp.GRB.MINIMIZE);
    m.optimize()
    if (m.status == gp.GRB.OPTIMAL):
        feasibility = True
        g = np.asarray([g[k,a,j].x for k in range(nS) for a in range(nA) for j in range(nS) ]).reshape([nS,nA,nS])
        return [m.objVal, g]
    else:
        g = None
        return [None, g]


def cvx_opt_epsradius(g, epsradius, phi, s_a_giv_sprime, p_infty_b_s, p_e_s,
                       nS, nA, sense = 1, vbs = False):

    A = np.zeros([nS,nS])
    for k in range(nS):
        for j in range(nS):
            A[k,j] += sum( [ s_a_giv_sprime[j,a,k]*p_infty_b_s[k] * (p_e_s[a,j] *g[k,a,j] ) for a in range(nA)] )
        A[k,k] -= p_infty_b_s[k]

    w = cvx.Variable(nS)
    obj = cvx.Minimize(cvx.sum(w*np.multiply(p_infty_b_s, phi))) if sense == 1 else cvx.Maximize(
        cvx.sum(w*np.multiply(p_infty_b_s, phi)))   # probably need to change maximize code
    constraints = [ cvx.norm( A*w , 1 ) <= epsradius,
                    cvx.sum(p_infty_b_s*w) == 1,
                    cvx.sum(w) >= 0.1, w >= 0
                   ]

    prob = cvx.Problem(obj, constraints)
    prob.solve(verbose=vbs)
    return [prob, w]


def sdp_relax(phi, a_bnd, b_bnd, s_a_giv_sprime, joint_s_a_sprime,  p_infty_b_s, p_e_s,
                       nS, nA, sense = 1, vbs = False, tight = True):

    p_infty_a = joint_s_a_sprime.sum(axis=(0,2))
    joint_ak = joint_s_a_sprime.sum(axis=0)
    p_infty_b_k_given_a = np.zeros([nS,nA])
    for k in range(nS):
        for a in range(nA):
            p_infty_b_k_given_a[k,a] = joint_ak[a,k] / p_infty_a[a]


    w = cvx.Variable(nS)
    g = [None] * nS;
    for k in range(nS):
        g[k] = cvx.Variable((nA,nS))
    nX = nS*nA*nS + nS
    x = cvx.Variable(nX)
    X = cvx.Variable((nX, nX), symmetric=True)
    bX = cvx.Variable( (nX+1, nX+1), symmetric=True )

    P = [None] * nS # list of matrices
    obj = cvx.Minimize( np.multiply(p_infty_b_s, phi).T * w ) if sense == 1 else cvx.Maximize(
        np.multiply(p_infty_b_s, phi).T * w )   # probably need to change maximize code
    constraints = [ cvx.sum(p_infty_b_s*w) == 1,
                    cvx.sum(w) >= 0.1, w >= 0
                   ]
    # Build coefficient matrices
    for k in range(nS):
        P[k] = lil_matrix((nX, nX))
        for a in range(nA):
            for j in range(nS):
                ind = np.ravel_multi_index([k, a, j], (nS, nA, nS))
                # fill mixed terms
                P[k][ ind, nS*nA*nS+j ] = p_e_s[a,j] * s_a_giv_sprime[j,a,k]*p_infty_b_s[k]
                # P[k][ nS*nA*nS+j, ind ] = 0.5 * p_e_s[a,j] * s_a_giv_sprime[j,a,k]*p_infty_b_s[k]
    eps = 0.05
    # a control variate

    for a in range(nA):
        constraints += [ cvx.sum([cvx.sum( [g[k][a, j] * s_a_giv_sprime[j, a, k] * p_infty_b_s[k] for j in range(nS)] ) for k in range(nS)])  == 1]

    for k in range(nS):
        constraints += [x[nS*nA*nS + k] == w[k],  #,
                        cvx.trace(X* P[k]) - w[k] * p_infty_b_s[k] <= 0.1 ,cvx.trace(X* P[k])- w[k] * p_infty_b_s[k]  >= -0.1
                        ]
        for a in range(nA):
            # s,a control variate
            # if tight:
                # constraints += [cvx.sum( [g[k][a, j] *s_a_giv_sprime[j,a,k]*p_infty_b_s[k] for j in range(nS) ]) == p_infty_b_k_given_a[k, a]   ]
            # else:
            #     constraints += [cvx.sum(g[k][:, j] * s_a_giv_sprime[j, :, k] * p_infty_b_s[k]) - p_infty_b_k_given_a[k, a] <= eps,
            #                     p_infty_b_k_given_a[k, a] - cvx.sum(g[k][:, j] * s_a_giv_sprime[j, :, k] * p_infty_b_s[k])  <= eps]

            for j in range(nS):
                ind = np.ravel_multi_index([k,a,j], (nS,nA,nS))
                # bounds
                constraints += [ x[ ind ] == g[k][a,j],
                                 g[k][a,j] <= b_bnd[a,j],
                                 g[k][a, j] >= a_bnd[a, j],
                                 x[ ind ] <= b_bnd[a, j],
                                 x[ ind ] >= a_bnd[a, j]
                                 ]

    # # bmat constraint
    constraints += [bX[0:nX, 0:nX] == X, bX[0:nX,-1] == x, bX[-1,0:nX] == x.T, bX[-1,-1] == 0]
    # blockx = cvx.hstack( [ [ X, x.reshape([nX,1]) ], [x.reshape([1,nX]), 0]])
    constraints += [ bX >> 0  ]

    prob = cvx.Problem(obj, constraints)
    prob.solve(verbose=vbs, solver = 'MOSEK', mosek_params = {mosek.dparam.optimizer_max_time:  50, mosek.dparam.basis_tol_s:1e-1})#,abstol = 1e-2, reltol = 1e-3, feastol=1e-2)
    return [prob, w, g]



def generate_data_get_bounds(phi, nns, nS, nA, nU, nSmarg, P, Pi, state_dist, n, PI_E, uniform_pi, mu, logGams_full, tight = True):
    ngams = len(logGams_full)
    Nns = len(nns)
    min_bnds = [[None] * ngams for m_ in range(Nns)];
    max_bnds = [[None] * ngams for m_ in range(Nns)];

    for ind_n, nn in enumerate(nns):
        n = nn
        print 'n', n
        # generate data
        [stateChangeHist, stateHist, a_s, s_a_sprime, p_infty_b_su, distr_hist] = simulate_rollouts(
            nS, nA, P, Pi, state_dist, n)
        quiet = True
        p_infty_b_s = (reshape_byxrow(p_infty_b_su.T, nU).T).flatten()
        p_infty_b_su = p_infty_b_su.flatten()

        #laplace smoothing
        if (p_infty_b_s == 0).any():
            smoother = np.ones(p_infty_b_s.shape)*1.0 / len(p_infty_b_s); p_infty_b_s = smoother *0.01+ p_infty_b_s*0.99
        # agg history and process


        [ p_a1_su, joint_s_a_sprime, s_a_giv_sprime ] = get_auxiliary_info_from_traj(stateChangeHist,
                                                stateHist, a_s, s_a_sprime, p_infty_b_su, distr_hist, nA,nS)
        p_e_su = PI_E*mu + uniform_pi*(1-mu); p_e_s = reshape_byxrow(p_e_su.T,nU).T / nU
        [aggStateHist, p_a1_s, p_e_s, agg_s_a_sprime, joint_s_a_sprime_agg, s_a_giv_sprime_agg] = agg_history(
                    stateHist, s_a_sprime, p_infty_b_s, a_s, p_e_su, nA, nS, nSmarg, nU)
        # laplace smoothing
        if (p_a1_s == 0).any():
            smoother = np.ones(p_a1_s.shape)*1.0/len(p_a1_s.flatten()); p_a1_s = smoother *0.01+ p_a1_s*0.99

        for ind,logGam in enumerate(logGams_full):
            sense_min = False; [a_bnd, b_bnd] = get_bnds_as( p_a1_s, logGam )
            [objVal, w_, m] = primal_opt_outer_L1_test_function_joint_distn(None, phi, a_bnd,b_bnd, joint_s_a_sprime_agg, p_infty_b_s, p_e_s, nSmarg, nA, tight, sense_min, quiet)
            min_bnds[ind_n][ind] = objVal#; w_min_bnds[ind] = w_;
            sense_min = True
            [objVal, w_, m] = primal_opt_outer_L1_test_function_joint_distn(None, phi, a_bnd,b_bnd, joint_s_a_sprime_agg, p_infty_b_s, p_e_s, nSmarg, nA, tight, sense_min, quiet)
            max_bnds[ind_n][ind] = objVal#; w_max_bnds[ind] = w_;
    pickle.dump([min_bnds, max_bnds, nns, logGams_full], open('output-log'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.p','wb') )
    return [min_bnds, max_bnds]


### simulate from a single trajectory
# tight = False
# # def get_full_agg_w_densratio( gamma, nS, nA, nU, P, Pi, p_e_su, state, n ): 
# nSmarg = nS / nU
# [ stateChangeHist, stateHist, a_s, s_a_sprime, p_infty_b_su, distr_hist ] =  simulate_rollouts( 
#     nS, nA, P, Pi, state_dist, n )
# p_infty_b_s = reshape_byxrow(p_infty_b_su.T, nU).T
# print p_infty_b_su, p_infty_b_s
# print('stationary distn on s,u: ', distrib)

# # process history 
# [ p_a1_su, joint_s_a_sprime, s_a_giv_sprime ] = get_auxiliary_info_from_traj(stateChangeHist, 
#                                         stateHist, a_s, s_a_sprime, distrib, distr_hist, nA,nS)

# # agg history and process
# [aggStateHist, p_a1_s, p_e_s, agg_s_a_sprime, joint_s_a_sprime_agg, s_a_giv_sprime_agg] = agg_history(
#             stateHist, s_a_sprime, p_infty_b_s, a_s, p_e_su, nA, nS, nSmarg, nU)
# # print aggStateHist
# # Simulate from evaluation policy: 
# [ stateChangeHist_e, stateHist_e, a_s, s_a_sprime_e, p_infty_e, distr_hist_e ] =  simulate_rollouts( 
#     nS, nA, P, p_e_su, state_dist, n )


# # Solve on the full information space 
# w_su = get_w_lp(gamma, s_a_giv_sprime, p_infty_b_su, p_e_su, p_a1_su, nS, tight = tight)
# print 'LP solution norm', w_su / sum(w_su)
# print w_su
# print 'oracle solution norm', (p_infty_e / p_infty_b_su) / np.sum((p_infty_e / p_infty_b_su))
# print (p_infty_e / p_infty_b_su)

# # Solve linear system on the aggregated space 
# w_ = get_w_lp(gamma, s_a_giv_sprime_agg, p_infty_b_s, p_e_s, p_a1_s, nSmarg, tight = tight)
# print ' solving with the biased weights ', w_ / sum(w_)
# w_su_oracle_norm = (p_infty_e / p_infty_b_su)/ np.sum((p_infty_e / p_infty_b_su))

# w_s_oracle_norm = reshape_byxrow(w_su_oracle_norm.T,nU).T.flatten()
# print 'marginalized stationary distribution', w_s_oracle_norm


# class ConfoundingRobustMargOPE:
#     def __init__(self, params={}):
#         self.save = save
#         self.nS = nS
#         self.nA = nA
#         self.nU = nU
#         self.params = params

#     def simulate_rollouts( nS, nA, P, Pi, state, n , verbose = False)""
#     ''' generate a trajectory of length n 
#     '''
#         stateChangeHist = np.zeros([nS,nS])
#         s_a_sprime = np.zeros([nS,nA,nS])
#         currentState=0; 
#         stateHist=state
#         dfStateHist=pd.DataFrame(state)
#         distr_hist = [ [0] * nS]
#         a_s = np.zeros(n)

#         for x in range(n):
#             a = np.random.choice(np.arange(0, nA), p=Pi[:,currentState])
#             a_s[x] = a
#             currentRow=np.ma.masked_values(( P[currentState, a, :] ) , 0.0)
#             nextState=np.random.choice(np.arange(0, nS), p=currentRow)
#             # Keep track of state changes
#             stateChangeHist[currentState,nextState]+=1
#             # Keep track of the state vector itself
#             state=np.zeros([1,nS]) #np.array([[0,0,0,0]])
#             state[0,nextState]=1.0
#             # Keep track of state history
#             stateHist=np.append(stateHist,state,axis=0)
#             # get s,a,s' distribution 
#             s_a_sprime[currentState, a, nextState] += 1

#             currentState=nextState
#             # calculate the actual distribution over the 3 states so far
#             totals=np.sum(stateHist,axis=0)
#             gt=np.sum(totals)
#             distrib=totals/gt
#             distrib=np.reshape(distrib,(1,nS))
#             distr_hist=np.append(distr_hist,distrib,axis=0)
            
#         if verbose: 
#             P_hat=stateChangeHist/stateChangeHist.sum(axis=1)[:,None]
#             # Check estimated state transition probabilities based on history so far:
#             dfDistrHist = pd.DataFrame(distr_hist)
#             # Plot the distribution as the simulation progresses over time
#             dfDistrHist.plot(title="Simulation History")
#             plt.show()
    
#         return [ stateChangeHist, stateHist, a_s, s_a_sprime]
    
#     def fit(self, x, t, y, q0, GAMS, method_params, eval_conf={'eval':False} ):

#     def predict(self, x, gamma):
#         pol_fn = POLS_dict[str(gamma)]
#         return opt_params['POL_PROB_1'](pol_fn, x)
#             # self.POLS_dict[str(gamma)](x)





# # get_full_agg_w_densratio( gamma, nS, nA, nU, P, Pi, p_e_su, state, n )